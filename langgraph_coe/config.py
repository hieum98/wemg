from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class TierConfig(BaseModel):
    """LLM configuration for one tier."""

    model_name: str = "Qwen3-Next-80B-A3B-Thinking-FP8"
    api_base: str = "http://n0142:4000/v1"
    api_key: Optional[str] = None  # inherits from top-level llm.api_key
    temperature: float = 0.7
    max_tokens: int = 8192
    max_input_tokens: int = 65536
    top_p: float = 0.95
    # Extra sampling controls. None = omit from the request so SGLang's own
    # default applies (don't send a neutral value you don't intend). top_k,
    # min_p and repetition_penalty are non-OpenAI knobs forwarded to SGLang's
    # request body via model_kwargs; presence/frequency_penalty and seed are
    # OpenAI-standard. See docs/setup_generation_params.md.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    enable_thinking: bool = True
    # Cap on *reasoning* tokens (SGLang custom logit processor forces </think>
    # once spent). None = no cap. Only takes effect when enable_thinking is True
    # and the model's think-token ids are known (see langgraph_coe.thinking_budget).
    # Must be < max_tokens, which still bounds thinking + answer combined.
    thinking_budget: Optional[int] = None
    max_retries: int = 3
    timeout: int = 300


class LLMConfig(BaseModel):
    """Extended LLM config with tier-based model selection."""

    api_key: Optional[str] = None
    tiers: Dict[str, TierConfig] = {
        "heavy": TierConfig(
            model_name="Qwen3-Next-80B-A3B-Thinking-FP8",
            temperature=0.7,
            max_tokens=8192,
            enable_thinking=True,
        ),
        "medium": TierConfig(
            model_name="Qwen3-Next-80B-A3B-Thinking-FP8",
            temperature=0.4,
            max_tokens=4096,
            enable_thinking=False,
        ),
        "light": TierConfig(
            model_name="Qwen3-32B-FP8",
            temperature=0.2,
            max_tokens=2048,
            enable_thinking=False,
        ),
        # §3b: minimal tier for roles whose output is small and bounded
        # (e.g. a list of ≤16 indices, a short entity list). Thinking off, tight
        # token ceiling — these cannot need more, so the cap only trims worst-case
        # decode and lets the server schedule tighter. Never put a reasoning or
        # open-ended-list role here.
        "classify": TierConfig(
            model_name="Qwen3-32B-FP8",
            temperature=0.0,
            max_tokens=1024,
            enable_thinking=False,
        ),
    }
    role_tiers: Dict[str, str] = {
        # Heavy reasoning roles
        "answer_generator": "heavy",
        "self_corrector": "heavy",
        "final_answer_synthesizer": "heavy",
        "subquestion_generator": "heavy",
        "reasoning_synthesizer": "heavy",
        # Medium structured-output roles
        "memory_consolidation": "medium",
        "relation_extraction": "medium",
        "open_ie": "medium",
        # verifier stays on medium pending the accuracy eval (its rating is the
        # MCTS reward; do not shrink its budget until measured).
        "verifier": "medium",
        "evaluator": "medium",
        # Bounded-output classification roles (§3b): tiny, fixed-shape outputs.
        "triple_pruner": "classify",  # keep_indices: ≤16 ints
        "named_entity_recognition": "classify",  # short entity list
        "question_rephraser": "classify",  # single rephrased query
        # Light extraction / agent roles
        "web_researcher": "light",
        "extractor": "light",
        # KG subgraph: LangChain create_agent tool-calling agents
        "kg_ner_agent": "light",
        "kg_triple_search_agent": "medium",
    }


class WebSearchConfig(BaseModel):
    # Off by default for paper-parity (the reference framework keeps web search
    # disabled in its fair-comparison setup; KG + corpus are the active surfaces).
    # When False, ``CoTGraph`` skips the ``web_one`` ReAct fan-out entirely.
    enabled: bool = False
    api_key: Optional[str] = None
    top_k: int = 5
    crawl_full_text: bool = True
    max_crawl_requests_per_second: float = 2.0
    max_queries_per_agent: int = 2


class CacheRedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    llm_db: int = 0
    wikidata_db: int = 1


class CacheWikidataConfig(BaseModel):
    entity_ttl: int = 2_592_000
    search_ttl: int = 604_800
    triples_ttl: int = 604_800
    enrich_ttl: int = 2_592_000


class CacheWebConfig(BaseModel):
    ttl: int = 86_400


class CacheConfig(BaseModel):
    enabled: bool = False
    redis: CacheRedisConfig = Field(default_factory=CacheRedisConfig)
    wikidata: CacheWikidataConfig = Field(default_factory=CacheWikidataConfig)
    web: CacheWebConfig = Field(default_factory=CacheWebConfig)


class MCTSConfig(BaseModel):
    """Knobs for the MCTSGraph (Phase 3). Ports legacy ``coe/reasoning/mcts.py``."""

    num_iterations: int = 15
    exploration_weight: float = 2.0
    max_tree_depth: int = 10
    # Per-rollout CoTGraph depth. Lower than coe's 5 because each rollout step
    # now runs a full CoT iteration (retrieval + rerank + extractor + memory).
    max_simulation_depth: int = 3
    # Floor on iterations before any early-termination condition (high-confidence,
    # semantic-sufficiency, convergence-patience) may fire. ``num_iterations``
    # always wins as the hard cap. Honored by ``route_after_iteration``.
    min_iterations: int = 3
    # Minimum tree depth (root = 0) before expand emits a FINAL_ANSWER child from
    # ``_gen_final``. Matches coe ``should_explore = depth < 2`` — skip synthesis
    # while memory is still shallow (legacy skipped retrieval there; we skip the
    # whole final-answer expansion call).
    final_answer_min_depth: int = 2
    high_confidence_threshold: float = 0.9
    convergence_patience: int = 5
    semantic_sufficiency_count: int = 3
    # LangGraph superstep budget for one MCTS run. Each iteration is ~7 supersteps
    # (select→expand→simulate→evaluate→backprop→mem_update→route); size with
    # headroom over ``num_iterations × 7``.
    recursion_limit: int = 150


class CoTConfig(BaseModel):
    """Knobs for the standalone CoTGraph strategy (Phase 2)."""

    # Maximum decomposition depth (CoT loop iterations) before forced synthesis.
    # Drives ``route_after_subq``; without it the loop finalizes immediately.
    max_depth: int = 5
    # LangGraph superstep budget for one CoT run. Each iteration is ~8 supersteps
    # (gen_subq→fan-out→rerank→extract→subanswers→mem_update→increment); size with
    # headroom over ``max_depth × 8``.
    recursion_limit: int = 75


class SearchConfig(BaseModel):
    """Top-level strategy selection (``system.py`` reads ``strategy``)."""

    strategy: str = "mcts"  # "mcts" | "cot"
    cot: CoTConfig = Field(default_factory=CoTConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)


class MemoryConfig(BaseModel):
    """Working-memory limits used by ``MemoryUpdateGraph`` and the CoT extractor."""

    max_textual_memory_tokens: int = 16_384
    prune_batch_size: int = 16
    # CoT extractor: split joined reranked passages into char-budgeted batches
    # so the EXTRACTOR's input never exceeds the model's context window.
    # Default ≈ 24k characters ≈ 7k tokens — well under any tier's max_input_tokens
    # and leaves room for system prompt + thinking + structured output.
    extractor_max_input_chars: int = 24_000


class EmbedderConfig(BaseModel):
    model_name: str = "Qwen3-Embedding-4B"
    url: str = "http://n0385:4000/v1"
    api_key: Optional[str] = None
    # Qwen3-Embedding is asymmetric: passages are embedded raw (build_index.py) but
    # queries must carry this instruction prefix ({query} placeholder).
    query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: {query}"
    )


class CorpusConfig(BaseModel):
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    # LangChain bundle (<name>.faiss + <name>.pkl) or raw HF-datasets index.
    index_path: str = (
        "/home/hieum/uonlp/coe/retriever_corpora/Qwen3-4B-Emb-index.faiss"
    )
    search_k: int = 10  # FAISS fetch depth before optional rerank / caller top_k cap
    # HF hub id or local ``load_from_disk`` path (same field name as legacy ``coe``).
    corpus_path: Optional[str] = "Hieuman/wiki23-processed"
    # Alias used in raw-index FAISS layout; kept in sync with ``corpus_path`` when one is set.
    corpus_dataset: Optional[str] = None
    corpus_split: str = "train"
    text_column: str = "contents"

    @model_validator(mode="after")
    def _sync_corpus_source_fields(self) -> "CorpusConfig":
        """Allow either ``corpus_path`` or ``corpus_dataset`` in YAML (legacy vs langgraph name)."""
        path = (self.corpus_path or "").strip() or None
        dataset = (self.corpus_dataset or "").strip() or None
        resolved = path or dataset
        if resolved is None:
            return self
        return self.model_copy(
            update={"corpus_path": resolved, "corpus_dataset": resolved}
        )

    def resolved_corpus_source(self) -> Optional[str]:
        """HF dataset / disk path for raw-index docstore lookup."""
        return self.corpus_dataset or self.corpus_path


class RetrieverConfig(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen3-Reranker-4B"
    url: str = "http://n0999:30002/v1"
    api_key: Optional[str] = "EMPTY"
    top_k: int = 10
    # Qwen3-Reranker SGLang template (<Instruct>: …); sent as JSON ``instruct``.
    instruction: Optional[str] = (
        "Given a web search query, retrieve relevant passages that answer the query."
    )


class WikidataConfig(BaseModel):
    """Configuration for the Wikidata knowledge-graph tools."""

    # Local QEndpoint (e.g. ``http://127.0.0.1:30162/api/endpoint/sparql`` via SSH tunnel).
    # None → public ``https://query.wikidata.org/sparql``.
    sparql_endpoint: Optional[str] = None

    # SPARQL / Wikipedia rate limits
    max_sparql_rps: float = 2.0
    max_wikipedia_rps: float = 10.0
    triple_cache_max_entries: int = 5000

    # Loop prevention: maximum number of fetch_and_prune_subgraph calls per question
    max_hops: int = 3

    # Stage A pruning knobs (reranker-based)
    reranker_url: Optional[str] = "http://n0999:30002/v1"
    reranker_model: Optional[str] = "Qwen3-Reranker-4B"
    # Task instruction sent to Qwen3-Reranker as the ``instruct`` field. Tells the
    # model what "relevant" means for triple selection; without it the reranker
    # saturates (~0.998 flat scores) on short, near-identical triple strings.
    reranker_instruction: Optional[str] = (
        "Given a question, retrieve the knowledge-graph triples whose relation and "
        "object are needed to answer it."
    )
    pruning_top_k: int = 64  # max triples kept after Stage A
    pruning_delta: float = 0.05  # score tolerance below the top score


class LangGraphCoeConfig(BaseModel):
    """Root settings for ``langgraph_coe`` loaded from ``config.yaml``."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    wikidata: WikidataConfig = Field(default_factory=WikidataConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @staticmethod
    def default_yaml_path() -> Path:
        return Path(__file__).resolve().parent / "config.yaml"

    @classmethod
    def from_yaml(
        cls, path: Optional[Path | str] = None, *, merge_api_key_env: bool = True
    ) -> LangGraphCoeConfig:
        """Load from YAML; when *path* is omitted, use ``default_yaml_path()`` if it exists."""

        p = Path(path) if path is not None else cls.default_yaml_path()
        raw: Dict[str, Any] | None = None
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        merged = cls.model_validate(raw or {})
        if merge_api_key_env:
            key = (
                merged.llm.api_key
                or os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            merged.llm.api_key = key
            if not merged.retriever.corpus.embedder.api_key:
                merged.retriever.corpus.embedder.api_key = key
            if not merged.reranker.api_key:
                merged.reranker.api_key = key or "EMPTY"

            # Optional runtime overrides (YAML in config.yaml is the default source of truth).
            env_index = os.environ.get("LANGGRAPH_CORPUS_INDEX_PATH") or os.environ.get(
                "SAR_CORPUS_INDEX_PATH"
            )
            if env_index:
                merged.retriever.corpus.index_path = env_index

            env_corpus = (
                os.environ.get("LANGGRAPH_CORPUS_DATASET")
                or os.environ.get("LANGGRAPH_CORPUS_PATH")
                or os.environ.get("SAR_CORPUS_DATASET")
                or os.environ.get("SAR_CORPUS_PATH")
            )
            if env_corpus:
                merged.retriever.corpus.corpus_path = env_corpus
                merged.retriever.corpus.corpus_dataset = env_corpus

            env_embed_url = os.environ.get("LANGGRAPH_TEST_EMBED_URL")
            if env_embed_url:
                merged.retriever.corpus.embedder.url = env_embed_url

            env_embed_model = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL")
            if env_embed_model:
                merged.retriever.corpus.embedder.model_name = env_embed_model

            env_rerank_url = os.environ.get("LANGGRAPH_TEST_RERANK_URL")
            if env_rerank_url:
                merged.reranker.url = env_rerank_url

            env_rerank_model = os.environ.get("LANGGRAPH_TEST_RERANK_MODEL")
            if env_rerank_model:
                merged.reranker.model_name = env_rerank_model

        return merged
