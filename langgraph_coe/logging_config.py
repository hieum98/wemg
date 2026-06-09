"""Central logging configuration for the langgraph_coe system.

``configure_logging`` is invoked once on package import (see ``__init__``) so
every entry point — the CLI, the evaluation runner, the profiling script, and
any notebook/REPL that does ``import langgraph_coe`` — gets the same behavior:

  * Third-party libraries that log one INFO line per network round-trip
    (``httpx``: ``HTTP Request: ...``) or per LLM call (``litellm``:
    ``LiteLLM completion() model=...``) are pinned to WARNING so they stop
    flooding stdout/stderr.
  * Our own ``langgraph_coe.*`` loggers keep emitting at the configured level
    (INFO by default) so genuine progress/diagnostic output is preserved.

Override at runtime without code changes:

    LANGGRAPH_COE_LOG_LEVEL=DEBUG     # our package logger level
    LANGGRAPH_COE_THIRDPARTY_LOG=INFO # un-silence httpx/litellm/... again
"""

from __future__ import annotations

import logging
import os

# Libraries whose per-request INFO chatter we never want by default.
_NOISY_LOGGERS = ("httpx", "httpcore", "LiteLLM", "litellm", "openai", "urllib3")

_configured = False


def _coerce_level(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    named = logging.getLevelName(str(value).strip().upper())
    return named if isinstance(named, int) else default


def configure_logging(force: bool = False) -> None:
    """Apply system-wide logging levels. Idempotent unless ``force=True``.

    Respects ``LANGGRAPH_COE_LOG_LEVEL`` (our package, default INFO) and
    ``LANGGRAPH_COE_THIRDPARTY_LOG`` (noisy third parties, default WARNING).
    """
    global _configured
    if _configured and not force:
        return

    pkg_level = _coerce_level(os.environ.get("LANGGRAPH_COE_LOG_LEVEL"), logging.INFO)
    thirdparty_level = _coerce_level(
        os.environ.get("LANGGRAPH_COE_THIRDPARTY_LOG"), logging.WARNING
    )

    # Ensure there is a handler so our package output is actually emitted even
    # when no entry point called logging.basicConfig.
    logging.basicConfig(level=pkg_level)

    # LiteLLM has its own verbose switches independent of the stdlib logger, and
    # importing it reconfigures third-party loggers (e.g. resets httpx) — so do
    # this BEFORE pinning levels below, otherwise it clobbers our settings.
    os.environ.setdefault("LITELLM_LOG", logging.getLevelName(thirdparty_level))
    try:
        import litellm

        litellm.set_verbose = False
        litellm.suppress_debug_info = True
    except Exception:
        pass

    logging.getLogger("langgraph_coe").setLevel(pkg_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(thirdparty_level)

    _configured = True
