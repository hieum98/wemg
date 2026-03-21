"""WEMG - When Embedding Models Meet Graph RAG."""

__version__ = "0.2.0"

from wemg.config import WEMGConfig, get_default_config_path
from wemg.system import WEMGSystem, AnswerResult, answer_question, answer_questions_batch

__all__ = [
    "__version__",
    "WEMGConfig",
    "get_default_config_path",
    "WEMGSystem",
    "AnswerResult",
    "answer_question",
    "answer_questions_batch",
]
