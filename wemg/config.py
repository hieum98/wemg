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
        >>> print(OmegaConf.get(cfg, "llm.generation.temperature"))
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
            OmegaConf.set(cfg, key, _parse_value(value))
    
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
    Uses OmegaConf.get() with defaults for flexible access.
    """
    # LLM API key
    llm_api_key = OmegaConf.get(cfg, "llm.api_key", None)
    if llm_api_key is None:
        OmegaConf.set(cfg, "llm.api_key", os.environ.get("OPENAI_API_KEY"))
    
    # Web search API key
    web_search_api_key = OmegaConf.get(cfg, "retriever.web_search.api_key", None)
    if web_search_api_key is None:
        OmegaConf.set(cfg, "retriever.web_search.api_key", os.environ.get("SERPER_API_KEY"))
    
    # Embedder API key
    embedder_api_key = OmegaConf.get(cfg, "retriever.corpus.embedder.api_key", None)
    if embedder_api_key is None:
        OmegaConf.set(cfg, "retriever.corpus.embedder.api_key", os.environ.get("OPENAI_API_KEY"))
    
    # Cache password
    cache_password = OmegaConf.get(cfg, "cache.password", None)
    if cache_password is None:
        OmegaConf.set(cfg, "cache.password", os.environ.get("REDIS_PASSWORD"))
    
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
    # Check required fields using OmegaConf.get() with defaults
    llm_api_key = OmegaConf.get(cfg, "llm.api_key", None)
    if llm_api_key is None:
        raise ValueError(
            "LLM API key is required. Set it in config or via OPENAI_API_KEY environment variable."
        )
    
    # Validate search strategy
    search_strategy = OmegaConf.get(cfg, "search.strategy", "cot")
    if search_strategy not in ("mcts", "cot"):
        raise ValueError(f"Invalid search strategy: {search_strategy}. Must be 'mcts' or 'cot'.")
    
    # Validate retriever type
    retriever_type = OmegaConf.get(cfg, "retriever.type", "web_search")
    if retriever_type not in ("web_search", "corpus", "hybrid"):
        raise ValueError(
            f"Invalid retriever type: {retriever_type}. Must be 'web_search', 'corpus', or 'hybrid'."
        )
    
    # Validate corpus retriever settings
    if retriever_type in ("corpus", "hybrid"):
        corpus_path = OmegaConf.get(cfg, "retriever.corpus.corpus_path", None)
        if corpus_path is None:
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
    
    Uses OmegaConf.get() with defaults for flexible access. New cache
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Cache configuration dictionary or None if disabled.
    """
    enabled = OmegaConf.get(cfg, "cache.enabled", True)
    if not enabled:
        return None
    
    return {
        "enabled": True,
        "host": OmegaConf.get(cfg, "cache.host", "localhost"),
        "port": OmegaConf.get(cfg, "cache.port", 6379),
        "db": OmegaConf.get(cfg, "cache.db", 0),
        "password": OmegaConf.get(cfg, "cache.password", None),
        "prefix": OmegaConf.get(cfg, "cache.prefix", "wemg"),
        "ttl": OmegaConf.get(cfg, "cache.ttl", 86400),
    }


def get_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract generation kwargs for LLM agent.
    
    Uses OmegaConf.get() with defaults for flexible access. New generation
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Generation kwargs dictionary.
    """
    return {
        "timeout": OmegaConf.get(cfg, "llm.generation.timeout", 60),
        "temperature": OmegaConf.get(cfg, "llm.generation.temperature", 0.7),
        "n": OmegaConf.get(cfg, "llm.generation.n", 1),
        "top_p": OmegaConf.get(cfg, "llm.generation.top_p", 0.8),
        "max_tokens": OmegaConf.get(cfg, "llm.generation.max_tokens", 8192),
        "max_input_tokens": OmegaConf.get(cfg, "llm.generation.max_input_tokens", 32768),
        "top_k": OmegaConf.get(cfg, "llm.generation.top_k", 20),
        "enable_thinking": OmegaConf.get(cfg, "llm.generation.enable_thinking", True),
        "random_seed": OmegaConf.get(cfg, "llm.generation.random_seed", None),
    }


def get_node_generation_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Extract node generation kwargs for search algorithms.
    
    Uses OmegaConf.get() with defaults for flexible access. New node generation
    parameters can be added to config.yaml without code changes.
    
    Args:
        cfg: Main configuration.
    
    Returns:
        Node generation kwargs dictionary.
    """
    return {
        "n": OmegaConf.get(cfg, "node_generation.n", 1),
        "top_k_websearch": OmegaConf.get(cfg, "node_generation.top_k_websearch", 5),
        "top_k_entities": OmegaConf.get(cfg, "node_generation.top_k_entities", 1),
        "top_k_properties": OmegaConf.get(cfg, "node_generation.top_k_properties", 1),
        "n_hops": OmegaConf.get(cfg, "node_generation.n_hops", 1),
        "use_question_for_graph_retrieval": OmegaConf.get(cfg, "node_generation.use_question_for_graph_retrieval", True),
    }
