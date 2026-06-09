"""LangGraph Chain-of-Evidence (CoE) package (import as ``langgraph_coe``)."""

from .logging_config import configure_logging

# Quiet noisy third-party loggers (httpx/litellm) system-wide on first import.
# Tunable via LANGGRAPH_COE_LOG_LEVEL / LANGGRAPH_COE_THIRDPARTY_LOG env vars.
configure_logging()

__all__: list[str] = ["configure_logging"]
