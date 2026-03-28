"""WEMG - When Embedding Models Meet Graph RAG."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

from wemg.config import WEMGConfig, get_default_config_path

__all__ = [
    "__version__",
    "WEMGConfig",
    "get_default_config_path",
    "WEMGSystem",
    "AnswerResult",
    "answer_question",
    "answer_questions_batch",
]


def __getattr__(name: str) -> Any:
    """Lazy-load system/LLM stack so lightweight imports (e.g. evaluation artifacts) avoid litellm."""
    if name == "WEMGSystem":
        from wemg.system import WEMGSystem

        return WEMGSystem
    if name == "AnswerResult":
        from wemg.system import AnswerResult

        return AnswerResult
    if name == "answer_question":
        from wemg.system import answer_question

        return answer_question
    if name == "answer_questions_batch":
        from wemg.system import answer_questions_batch

        return answer_questions_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *globals()})
