"""WEMG - LLM agents with caching support and graph-enhanced retrieval."""

__version__ = "0.1.0"

from wemg.agents.base_llm_agent import BaseClient, OpenAIClient, BaseLLMAgent
from wemg.config import (
    WEMGConfig,
    load_config,
    create_config_from_dict,
    validate_config,
)
from wemg.main import (
    WEMGSystem,
    AnswerResult,
    answer_question,
    answer_questions_batch,
)

__all__ = [
    # Version
    "__version__",
    # Agents
    "BaseClient",
    "OpenAIClient",
    "BaseLLMAgent",
    # Configuration
    "WEMGConfig",
    "load_config",
    "create_config_from_dict",
    "validate_config",
    # Main interface
    "WEMGSystem",
    "AnswerResult",
    "answer_question",
    "answer_questions_batch",
]
