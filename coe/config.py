import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import logging

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DOTENV_LOADED_PATHS: set = set()


class LLMGenerationConfig(BaseModel):
    timeout: int = 300
    temperature: float = 0.8
    n: int = 1
    top_p: float = 0.95
    min_p: float = 0.0
    max_tokens: int = 32768
    max_input_tokens: int = 65536
    top_k: int = 20
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = True
    random_seed: Optional[int] = None


class LLMConfig(BaseModel):
    model_name: str = "Qwen3-Next-80B-A3B-Thinking-FP8"
    url: str = "http://n0142:4000/v1"
    api_key: Optional[str] = None
    client_type: Literal["openai", "azure", "anthropic"] = "openai"
    concurrency: int = 64
    max_retries: int = 3
    generation: LLMGenerationConfig = LLMGenerationConfig()


class CacheConfig(BaseModel):
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    prefix: str = "coe"
    ttl: int = 86400


class WebSearchConfig(BaseModel):
    api_key: Optional[str] = None
    top_k: int = 5
    crawl_full_text: bool = True
    max_crawl_requests_per_second: float = 2.0
    query_cache_ttl: Optional[int] = 3600  # seconds; None disables query caching, 0 = no expiry
    url_cache_ttl: Optional[int] = 3600    # seconds; None disables URL caching, 0 = no expiry


class EmbedderConfig(BaseModel):
    model_name: str = "Qwen3-Embedding-4B"
    url: str = "http://n0142:4000/v1"
    api_key: Optional[str] = None
    embedder_type: Literal["openai", "huggingface"] = "openai"


class CorpusConfig(BaseModel):
    embedder: EmbedderConfig = EmbedderConfig()
    corpus_path: str = "Hieuman/wiki23-processed"
    index_path: str = "/gpfs/projects/uonlp/hieum/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss"


class RetrieverConfig(BaseModel):
    type: Literal["corpus", "web_search"] = "corpus"
    web_search: WebSearchConfig = WebSearchConfig()
    corpus: CorpusConfig = CorpusConfig()


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen3-Next-80B-A3B-Thinking-FP8"
    url: str = "http://n0142:4000/v1"
    api_key: Optional[str] = None
    top_k: int = 10
    concurrency: int = 32
    instruction: Optional[str] = None


class MCTSEarlyTerminationConfig(BaseModel):
    enabled: bool = True
    min_iterations: int = 5
    high_confidence_threshold: float = 0.9
    convergence_patience: int = 5
    semantic_sufficiency_count: int = 3


class MCTSConfig(BaseModel):
    num_iterations: int = 15
    max_tree_depth: int = 10
    exploration_weight: float = 2.5
    max_simulation_depth: int = 5
    early_termination: MCTSEarlyTerminationConfig = MCTSEarlyTerminationConfig()
    use_golden_answer_for_reward: bool = False

class CoTConfig(BaseModel):
    max_depth: int = 10


class SearchConfig(BaseModel):
    strategy: Literal["mcts", "cot"] = "mcts"
    mcts: MCTSConfig = MCTSConfig()
    cot: CoTConfig = CoTConfig()


class NodeGenerationConfig(BaseModel):
    n: int = 1
    n_subquestions: int = 3
    top_k_websearch: int = 5
    top_k_entities: int = 3
    n_hops: int = 2
    entity_linking_method: Literal["llm", "azure"] = "llm"
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    rerank_kb_documents: bool = True
    triple_pruning_delta: float = 0.05
    triple_pruning_top_k: int = 64
    kb_source: Literal["wikidata", "freebase_subgraph", "freebase_live"] = "wikidata"
    freebase_subgraph_cache: Optional[str] = None
    freebase_sparql_url: str = "http://n0387:3001/sparql"
    freebase_qid_to_mid_map_path: Optional[str] = None
    freebase_qid_to_mid_candidates: int = 5


class WorkingMemoryConfig(BaseModel):
    max_textual_memory_tokens: int = 16384
    annotate_steps: bool = True


class InteractionMemoryConfig(BaseModel):
    enabled: bool = False
    scope: Literal["question", "dataset"] = "question"
    log_dir: str = "./logs"
    save_to_file: bool = False
    db_path: Optional[str] = None
    collection_name: str = "interaction_memory"
    token_budget: int = 8192
    batch_size: int = 32
    batch_size_async: int = 100
    is_local_embedding_api: bool = True
    embedding_model_name: str = "Qwen3-Embedding-4B"
    embedding_base_url: str = "http://n0142:4000/v1"
    embedding_api_key: Optional[str] = None
    enable_embedding_cache: bool = True
    cache_max_size: int = 10000


class MemoryConfig(BaseModel):
    working_memory: WorkingMemoryConfig = WorkingMemoryConfig()
    interaction_memory: InteractionMemoryConfig = InteractionMemoryConfig()


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class OutputConfig(BaseModel):
    include_reasoning: bool = True
    include_concise_answer: bool = True
    show_search_tree: bool = True
    verbose: bool = False


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_override_value(raw: str) -> Any:
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "null" or raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


class COEConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    cache: CacheConfig = CacheConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    search: SearchConfig = SearchConfig()
    node_generation: NodeGenerationConfig = NodeGenerationConfig()
    memory: MemoryConfig = MemoryConfig()
    logging: LoggingConfig = LoggingConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(
        cls,
        path: Optional[str | Path] = None,
        overrides: Optional[List[str]] = None,
    ) -> "COEConfig":
        _load_dotenv_once()
        if path is None:
            path = get_default_config_path()
        path = Path(path)

        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Merge env before CLI overrides so explicit values (e.g. llm.api_key=null) win.
        _resolve_env_vars(data)

        if overrides:
            for override in overrides:
                key, _, raw_value = override.partition("=")
                _set_nested(data, key.strip(), _parse_override_value(raw_value.strip()))

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "COEConfig":
        _load_dotenv_once()
        _resolve_env_vars(config_dict)
        return cls.model_validate(config_dict)

    def validate_config(self) -> List[str]:
        errors: List[str] = []

        if not self.llm.api_key:
            errors.append("llm.api_key is required (set API_KEY env var or in config)")

        if self.search.strategy not in ("mcts", "cot"):
            errors.append(f"Invalid search strategy: {self.search.strategy}")

        if self.retriever.type == "web_search" and not self.retriever.web_search.api_key:
            errors.append(
                "retriever.web_search.api_key is required when retriever type is 'web_search' "
                "(set SERPER_API_KEY env var or in config)"
            )

        if self.retriever.type == "corpus":
            if not self.retriever.corpus.corpus_path:
                errors.append("retriever.corpus.corpus_path is required when retriever type is 'corpus'")

        if self.node_generation.entity_linking_method == "azure":
            if not self.node_generation.azure_endpoint or not self.node_generation.azure_key:
                errors.append(
                    "azure_endpoint and azure_key are required when entity_linking_method is 'azure'"
                )

        return errors


def _resolve_env_vars(data: Dict[str, Any]) -> None:
    api_key = os.environ.get("API_KEY")
    if api_key:
        _set_nested(data, "llm.api_key", data.get("llm", {}).get("api_key") or api_key)
        _set_nested(
            data,
            "retriever.corpus.embedder.api_key",
            data.get("retriever", {}).get("corpus", {}).get("embedder", {}).get("api_key") or api_key,
        )
        _set_nested(
            data,
            "memory.interaction_memory.embedding_api_key",
            data.get("memory", {}).get("interaction_memory", {}).get("embedding_api_key") or api_key,
        )

    serper_key = os.environ.get("SERPER_API_KEY")
    if serper_key:
        _set_nested(
            data,
            "retriever.web_search.api_key",
            data.get("retriever", {}).get("web_search", {}).get("api_key") or serper_key,
        )

    redis_password = os.environ.get("REDIS_PASSWORD")
    if redis_password:
        _set_nested(
            data,
            "cache.password",
            data.get("cache", {}).get("password") or redis_password,
        )


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv is not installed; skipping .env loading fallback.")
        return

    candidate_paths = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    for dotenv_path in candidate_paths:
        dotenv_path = dotenv_path.resolve()
        if dotenv_path in _DOTENV_LOADED_PATHS or not dotenv_path.is_file():
            continue
        _DOTENV_LOADED_PATHS.add(dotenv_path)
        load_dotenv(dotenv_path, override=False)


def get_default_config_path() -> Path:
    return Path(__file__).parent / "config.yaml"
