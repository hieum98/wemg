"""COE - When Embedding Models Meet Graph RAG."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

from coe.config import COEConfig, get_default_config_path

__all__ = [
    "__version__",
    "COEConfig",
    "get_default_config_path",
    "COESystem",
    "AnswerResult",
    "answer_question",
    "answer_questions_batch",
]


def __getattr__(name: str) -> Any:
    """Lazy-load system/LLM stack so lightweight imports (e.g. evaluation artifacts) avoid litellm."""
    if name == "COESystem":
        from coe.system import COESystem

        return COESystem
    if name == "AnswerResult":
        from coe.system import AnswerResult

        return AnswerResult
    if name == "answer_question":
        from coe.system import answer_question

        return answer_question
    if name == "answer_questions_batch":
        from coe.system import answer_questions_batch

        return answer_questions_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *globals()})
