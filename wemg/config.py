"""Configuration module for WEMG using OmegaConf.

This module provides configuration management for the WEMG system,
allowing flexible parameter configuration through YAML files, command-line
overrides, and environment variables. All configuration is handled through
OmegaConf DictConfig objects for maximum flexibility.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


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
        >>> cfg = load_config(overrides=["llm.generation.temperature=0.5", "search.strategy=mcts"])
        >>> print(OmegaConf.select(cfg, "llm.generation.temperature"))
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
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Resolve environment variables
    cfg = _resolve_env_vars(cfg)
    
    return cfg


def _resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Resolve environment variables in the configuration."""
    # LLM API key
    llm_api_key = OmegaConf.select(cfg, "llm.api_key")
    if llm_api_key is None:
        cfg.llm.api_key = os.environ.get("API_KEY")
    
    # Web search API key
    web_search_api_key = OmegaConf.select(cfg, "retriever.web_search.api_key")
    if web_search_api_key is None:
        cfg.retriever.web_search.api_key = os.environ.get("SERPER_API_KEY")
    
    # Embedder API key
    embedder_api_key = OmegaConf.select(cfg, "retriever.corpus.embedder.api_key")
    if embedder_api_key is None:
        cfg.retriever.corpus.embedder.api_key = os.environ.get("API_KEY")
    
    # Interaction memory embedding API key
    interaction_memory_embedding_api_key = OmegaConf.select(cfg, "memory.interaction_memory.embedding_api_key")
    if interaction_memory_embedding_api_key is None:
        cfg.memory.interaction_memory.embedding_api_key = os.environ.get("API_KEY")
    
    # Cache password
    cache_password = OmegaConf.select(cfg, "cache.password")
    if cache_password is None:
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
        ...     "llm": {"model_name": "gpt-4o", "generation": {"temperature": 0.5}},
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


def validate_config(cfg: DictConfig) -> bool:
    """Validate the configuration.
    
    Args:
        cfg: Configuration to validate.
    
    Returns:
        True if configuration is valid.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    # Check required fields using OmegaConf.select()
    llm_api_key = OmegaConf.select(cfg, "llm.api_key")
    if llm_api_key is None:
        raise ValueError(
            "LLM API key is required. Set it in config or via API_KEY environment variable."
        )
    
    # Validate search strategy
    search_strategy = OmegaConf.select(cfg, "search.strategy") or "cot"
    if search_strategy not in ("mcts", "cot"):
        raise ValueError(f"Invalid search strategy: {search_strategy}. Must be 'mcts' or 'cot'.")
    
    # Validate retriever type
    retriever_type = OmegaConf.select(cfg, "retriever.type") or "web_search"
    if retriever_type not in ("web_search", "corpus"):
        raise ValueError(
            f"Invalid retriever type: {retriever_type}. Must be 'web_search' or 'corpus'."
        )
    
    # Validate corpus retriever settings
    if retriever_type == "corpus":
        corpus_path = OmegaConf.select(cfg, "retriever.corpus.corpus_path")
        if corpus_path is None:
            raise ValueError("Corpus path is required for corpus retriever.")
        index_path = OmegaConf.select(cfg, "retriever.corpus.index_path")
        if index_path is None:
            raise ValueError("Index path is required for corpus retriever.")
    
    return True

# =====================================================================
# Helper Functions for Building Components
# =====================================================================

def get_cache_config(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Extract cache configuration for BaseLLMAgent.
    
    Uses OmegaConf.select() with defaults for flexible access. New cache
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Cache configuration dictionary or None if disabled.
    """
    enabled = OmegaConf.select(cfg, "cache.enabled")
    if enabled is None:
        enabled = True
    if not enabled:
        return None
    
    return {
        "enabled": True,
        "host": OmegaConf.select(cfg, "cache.host") or "localhost",
        "port": OmegaConf.select(cfg, "cache.port") or 6379,
        "db": OmegaConf.select(cfg, "cache.db") or 0,
        "password": OmegaConf.select(cfg, "cache.password"),
        "prefix": OmegaConf.select(cfg, "cache.prefix") or "wemg",
        "ttl": OmegaConf.select(cfg, "cache.ttl") or 86400,
    }


def get_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract generation kwargs for LLM agent.
    
    Uses OmegaConf.select() with defaults for flexible access. New generation
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Generation kwargs dictionary.
    """
    enable_thinking = OmegaConf.select(cfg, "llm.generation.enable_thinking")
    if enable_thinking is None:
        enable_thinking = True
    
    return {
        "timeout": OmegaConf.select(cfg, "llm.generation.timeout") or 300,
        "temperature": OmegaConf.select(cfg, "llm.generation.temperature") or 0.7,
        "n": OmegaConf.select(cfg, "llm.generation.n") or 1,
        "top_p": OmegaConf.select(cfg, "llm.generation.top_p") or 0.8,
        "max_tokens": OmegaConf.select(cfg, "llm.generation.max_tokens") or 65536,
        "max_input_tokens": OmegaConf.select(cfg, "llm.generation.max_input_tokens") or 65536,
        "top_k": OmegaConf.select(cfg, "llm.generation.top_k") or 20,
        "enable_thinking": enable_thinking,
        "random_seed": OmegaConf.select(cfg, "llm.generation.random_seed"),
    }


def get_node_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract node generation kwargs for search algorithms.
    
    Uses OmegaConf.select() with defaults for flexible access. New node generation
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Node generation kwargs dictionary.
    """
    use_question_for_graph_retrieval = OmegaConf.select(cfg, "node_generation.use_question_for_graph_retrieval")
    if use_question_for_graph_retrieval is None:
        use_question_for_graph_retrieval = True
    
    return {
        "n": OmegaConf.select(cfg, "node_generation.n") or 1,
        "top_k_websearch": OmegaConf.select(cfg, "node_generation.top_k_websearch") or 5,
        "top_k_entities": OmegaConf.select(cfg, "node_generation.top_k_entities") or 1,
        "top_k_properties": OmegaConf.select(cfg, "node_generation.top_k_properties") or 1,
        "n_hops": OmegaConf.select(cfg, "node_generation.n_hops") or 1,
        "use_question_for_graph_retrieval": use_question_for_graph_retrieval,
    }
