from typing import Dict, Optional
from pydantic import BaseModel

class TierConfig(BaseModel):
    """LLM configuration for one tier."""
    model_name: str = "Qwen3-Next-80B-A3B-Thinking-FP8"
    api_base: str = "http://n0142:4000/v1"
    api_key: Optional[str] = None  # inherits from top-level llm.api_key
    temperature: float = 0.7
    max_tokens: int = 8192
    max_input_tokens: int = 65536
    top_p: float = 0.95
    enable_thinking: bool = True
    max_retries: int = 3
    timeout: int = 300


class LLMConfig(BaseModel):
    """Extended LLM config with tier-based model selection."""
    api_key: Optional[str] = None
    tiers: Dict[str, TierConfig] = {
        "heavy": TierConfig(model_name="Qwen3-Next-80B-A3B-Thinking-FP8", temperature=0.7, max_tokens=8192, enable_thinking=True),
        "medium": TierConfig(model_name="Qwen3-Next-80B-A3B-Thinking-FP8", temperature=0.4, max_tokens=4096, enable_thinking=False),
        "light": TierConfig(model_name="Qwen3-32B-FP8", temperature=0.2, max_tokens=2048, enable_thinking=False),
    }
    role_tiers: Dict[str, str] = {
        "answer_generator": "heavy",
        "self_corrector": "heavy",
        "reasoning_synthesizer": "heavy",
        "final_answer_synthesizer": "heavy",
        "subquestion_generator": "heavy",
        "memory_consolidation": "medium",
        "relation_extraction": "medium",
        "triple_pruner": "medium",
        "evaluator": "medium",
        "verifier": "medium",
        "consensus_evaluator": "medium",
        "majority_voter": "medium",
        "query_generator": "light",
        "extractor": "light",
        "named_entity_recognition": "light",
        "question_rephraser": "light",
        # KG subgraph: LangChain create_agent tool-calling agents
        "kg_ner_agent": "light",
        "kg_triple_search_agent": "medium",
    }


class WebSearchConfig(BaseModel):
    api_key: Optional[str] = None
    top_k: int = 5
    crawl_full_text: bool = True
    max_crawl_requests_per_second: float = 2.0


class EmbedderConfig(BaseModel):
    model_name: str = "Qwen3-Embedding-4B"
    url: str = "http://n0385:4000/v1"
    api_key: Optional[str] = None


class CorpusConfig(BaseModel):
    embedder: EmbedderConfig = EmbedderConfig()
    index_path: str = "/home/hieum/uonlp/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss"


class RetrieverConfig(BaseModel):
    web_search: WebSearchConfig = WebSearchConfig()
    corpus: CorpusConfig = CorpusConfig()


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen3-Reranker-4B"
    url: str = "http://n0999:30002/v1"
    api_key: Optional[str] = "EMPTY"
    top_k: int = 10


class WikidataConfig(BaseModel):
    """Configuration for the Wikidata knowledge-graph tools."""
    # SPARQL / Wikipedia rate limits
    max_sparql_rps: float = 2.0
    max_wikipedia_rps: float = 10.0
    triple_cache_max_entries: int = 5000

    # Loop prevention: maximum number of fetch_and_prune_subgraph calls per question
    max_hops: int = 3

    # Stage A pruning knobs (reranker-based)
    reranker_url: Optional[str] = "http://n0999:30002/v1"
    reranker_model: Optional[str] = "Qwen3-Reranker-4B"
    pruning_top_k: int = 64      # max triples kept after Stage A
    pruning_delta: float = 0.05  # score tolerance below the top score
