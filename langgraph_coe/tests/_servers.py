"""Shared helpers for langgraph_coe integration / real-server test modules.

The ``test_*_integration.py`` and ``test_*_real_servers.py`` suites all need to
answer the same question — *is this OpenAI-compatible endpoint reachable?* — and
each used to carry a private copy of the probe. Centralize it here so a new
integration suite can ``from .._servers import endpoint_alive`` instead of
re-pasting the same ``httpx.Client(...).get('/models')`` dance.

The default endpoint constants mirror the values the existing suites fall back
to when their ``LANGGRAPH_TEST_*`` env overrides are unset; import them to keep
new suites consistent with the current single-GPU test deployment.
"""

from __future__ import annotations

import httpx

# Defaults for the standard single-node test deployment. Suites override these
# per run via the matching ``LANGGRAPH_TEST_*`` environment variables.
DEFAULT_LLM_URL = "http://localhost:30172/v1"
DEFAULT_LLM_MODEL = "openai/Qwen/Qwen3-8B"
DEFAULT_EMBED_URL = "http://localhost:30164/v1"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"


def endpoint_alive(url: str, *, timeout: float = 10.0) -> bool:
    """Return True iff an OpenAI-compatible server answers ``GET {url}/models``.

    Any connection/timeout/HTTP error is treated as "down" so callers can use
    this directly in ``pytest.mark.skipif`` guards without their own try/except.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.get(f"{url.rstrip('/')}/models").status_code == 200
    except Exception:
        return False
