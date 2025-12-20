"""Configuration module for WEMG using Hydra and OmegaConf.

This module provides configuration management for the WEMG system,
allowing flexible parameter configuration through YAML files, command-line
overrides, and environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf, MISSING


# =====================================================================
# Dataclass Configurations
# =====================================================================

@dataclass
class GenerationConfig:
    """LLM generation parameters."""
    timeout: int = 60
    temperature: float = 0.7
    n: int = 1
    top_p: float = 0.8
    max_tokens: int = 8192
    max_input_tokens: int = 32768
    top_k: int = 20
    enable_thinking: bool = True
    random_seed: Optional[int] = None


@dataclass
class LLMConfig:
    """LLM agent configuration."""
    model_name: str = "gpt-4o-mini"
    url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    client_type: str = "openai"
    concurrency: int = 64
    max_retries: int = 3
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class CacheConfig:
    """Redis cache configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    prefix: str = "wemg"
    ttl: int = 86400


@dataclass
class WebSearchConfig:
    """Web search retriever configuration."""
    api_key: Optional[str] = None
    top_k: int = 5
    crawl_full_text: bool = True


@dataclass
class EmbedderConfig:
    """Embedder configuration for corpus retriever."""
    model_name: str = "text-embedding-3-small"
    url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    embedder_type: str = "openai"


@dataclass
class CorpusConfig:
    """Corpus-based retriever configuration."""
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    corpus_path: Optional[str] = None
    index_path: Optional[str] = None


@dataclass
class RetrieverConfig:
    """Retriever configuration."""
    type: str = "web_search"  # "web_search", "corpus", or "hybrid"
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)


@dataclass
class MCTSConfig:
    """MCTS search strategy configuration."""
    num_iterations: int = 10
    max_tree_depth: int = 10
    max_simulation_depth: int = 5
    exploration_weight: float = 1.0
    is_cot_simulation: bool = True


@dataclass
class CoTConfig:
    """Chain-of-Thought search strategy configuration."""
    max_depth: int = 10


@dataclass
class SearchConfig:
    """Search strategy configuration."""
    strategy: str = "cot"  # "mcts" or "cot"
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    cot: CoTConfig = field(default_factory=CoTConfig)


@dataclass
class NodeGenerationConfig:
    """Node generation parameters."""
    n: int = 1
    top_k_websearch: int = 5
    top_k_entities: int = 1
    top_k_properties: int = 1
    n_hops: int = 1
    use_question_for_graph_retrieval: bool = True


@dataclass
class WorkingMemoryConfig:
    """Working memory configuration."""
    max_textual_memory_tokens: int = 8192


@dataclass
class InteractionMemoryConfig:
    """Interaction memory (logging) configuration."""
    enabled: bool = False
    log_dir: str = "./logs"
    save_to_file: bool = False


@dataclass
class MemoryConfig:
    """Memory configuration."""
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    interaction_memory: InteractionMemoryConfig = field(default_factory=InteractionMemoryConfig)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class OutputConfig:
    """Output configuration."""
    include_reasoning: bool = True
    include_concise_answer: bool = True
    verbose: bool = False


@dataclass
class WEMGConfig:
    """Main WEMG configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    node_generation: NodeGenerationConfig = field(default_factory=NodeGenerationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# =====================================================================
# Configuration Loading Functions
# =====================================================================

def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    return Path(__file__).parent / "config.yaml"


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None
) -> DictConfig:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to the configuration file. If None, uses default config.
        overrides: List of override strings in Hydra format (e.g., ["llm.model_name=gpt-4o"])
    
    Returns:
        OmegaConf DictConfig object with the loaded configuration.
    
    Example:
        >>> cfg = load_config(overrides=["llm.temperature=0.5", "search.strategy=mcts"])
        >>> print(cfg.llm.temperature)
        0.5
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base configuration
    cfg = OmegaConf.load(config_path)
    
    # Apply overrides
    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            OmegaConf.update(cfg, key, _parse_value(value))
    
    # Resolve environment variables
    cfg = _resolve_env_vars(cfg)
    
    return cfg


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate type."""
    # Handle null/None
    if value.lower() in ("null", "none", "~"):
        return None
    
    # Handle booleans
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    
    # Handle integers
    try:
        return int(value)
    except ValueError:
        pass
    
    # Handle floats
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def _resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Resolve environment variables in the configuration.
    
    Automatically fills in API keys from environment variables if not set.
    """
    # LLM API key
    if cfg.llm.api_key is None:
        cfg.llm.api_key = os.environ.get("OPENAI_API_KEY")
    
    # Web search API key
    if cfg.retriever.web_search.api_key is None:
        cfg.retriever.web_search.api_key = os.environ.get("SERPER_API_KEY")
    
    # Embedder API key
    if cfg.retriever.corpus.embedder.api_key is None:
        cfg.retriever.corpus.embedder.api_key = os.environ.get("OPENAI_API_KEY")
    
    # Cache password
    if cfg.cache.password is None:
        cfg.cache.password = os.environ.get("REDIS_PASSWORD")
    
    return cfg


def create_config_from_dict(config_dict: Dict[str, Any]) -> DictConfig:
    """Create configuration from a dictionary.
    
    Args:
        config_dict: Dictionary containing configuration parameters.
    
    Returns:
        OmegaConf DictConfig object.
    
    Example:
        >>> cfg = create_config_from_dict({
        ...     "llm": {"model_name": "gpt-4o", "temperature": 0.5},
        ...     "search": {"strategy": "mcts"}
        ... })
    """
    # Load default config first
    default_cfg = load_config()
    
    # Merge with provided dict
    user_cfg = OmegaConf.create(config_dict)
    merged_cfg = OmegaConf.merge(default_cfg, user_cfg)
    
    # Resolve environment variables
    merged_cfg = _resolve_env_vars(merged_cfg)
    
    return merged_cfg


def to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert configuration to a plain dictionary.
    
    Args:
        cfg: OmegaConf DictConfig object.
    
    Returns:
        Plain dictionary representation.
    """
    return OmegaConf.to_container(cfg, resolve=True)


def validate_config(cfg: DictConfig) -> bool:
    """Validate the configuration.
    
    Args:
        cfg: Configuration to validate.
    
    Returns:
        True if configuration is valid.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    # Check required fields
    if cfg.llm.api_key is None:
        raise ValueError(
            "LLM API key is required. Set it in config or via OPENAI_API_KEY environment variable."
        )
    
    # Validate search strategy
    if cfg.search.strategy not in ("mcts", "cot"):
        raise ValueError(f"Invalid search strategy: {cfg.search.strategy}. Must be 'mcts' or 'cot'.")
    
    # Validate retriever type
    if cfg.retriever.type not in ("web_search", "corpus", "hybrid"):
        raise ValueError(
            f"Invalid retriever type: {cfg.retriever.type}. Must be 'web_search', 'corpus', or 'hybrid'."
        )
    
    # Validate corpus retriever settings
    if cfg.retriever.type in ("corpus", "hybrid"):
        if cfg.retriever.corpus.corpus_path is None:
            raise ValueError("Corpus path is required for corpus/hybrid retriever.")
    
    return True


def print_config(cfg: DictConfig) -> None:
    """Print the configuration in a readable format.
    
    Args:
        cfg: Configuration to print.
    """
    print(OmegaConf.to_yaml(cfg))


# =====================================================================
# Helper Functions for Building Components
# =====================================================================

def get_cache_config(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Extract cache configuration for BaseLLMAgent.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Cache configuration dictionary or None if disabled.
    """
    if not cfg.cache.enabled:
        return None
    
    return {
        "enabled": True,
        "host": cfg.cache.host,
        "port": cfg.cache.port,
        "db": cfg.cache.db,
        "password": cfg.cache.password,
        "prefix": cfg.cache.prefix,
        "ttl": cfg.cache.ttl,
    }


def get_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract generation kwargs for LLM agent.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Generation kwargs dictionary.
    """
    return {
        "timeout": cfg.llm.generation.timeout,
        "temperature": cfg.llm.generation.temperature,
        "n": cfg.llm.generation.n,
        "top_p": cfg.llm.generation.top_p,
        "max_tokens": cfg.llm.generation.max_tokens,
        "max_input_tokens": cfg.llm.generation.max_input_tokens,
        "top_k": cfg.llm.generation.top_k,
        "enable_thinking": cfg.llm.generation.enable_thinking,
        "random_seed": cfg.llm.generation.random_seed,
    }


def get_node_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract node generation kwargs for search algorithms.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Node generation kwargs dictionary.
    """
    return {
        "n": cfg.node_generation.n,
        "top_k_websearch": cfg.node_generation.top_k_websearch,
        "top_k_entities": cfg.node_generation.top_k_entities,
        "top_k_properties": cfg.node_generation.top_k_properties,
        "n_hops": cfg.node_generation.n_hops,
        "use_question_for_graph_retrieval": cfg.node_generation.use_question_for_graph_retrieval,
    }
