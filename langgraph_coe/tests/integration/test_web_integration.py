"""Integration smoke tests — live ``web_search`` (Serper or DuckDuckGo).

Exercises ``init_web_search`` + ``web_search`` against the configured provider.
Skipped when the provider is unreachable or credentials are missing.

Run manually (from repo root). Redis cache setup: ``docs/setup_redis_cache.md``::

    uv run pytest langgraph_coe/tests/phase0/test_web_integration.py -v -s

Defaults come from ``langgraph_coe/config.yaml`` (``web_search``).
Optional env overrides::

    SERPER_API_KEY Serper when set (else DuckDuckGo fallback)
    LANGGRAPH_WEB_SEARCH_TOP_K override top_k for the smoke query
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langgraph_coe.config import LangGraphCoeConfig, WebSearchConfig
from langgraph_coe.tools.web import (
    init_web_search,
    reset_web_research_session,
    web_search,
)

from .._fixtures import log_config_override

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

_QUERY = "What is the capital of France?"


def _integration_cfg() -> LangGraphCoeConfig:
    return LangGraphCoeConfig.from_yaml()


def _web_cfg() -> WebSearchConfig:
    """config.yaml ``web_search`` block; full-text crawl disabled for a fast smoke."""
    cfg = _integration_cfg().web_search.model_copy(deep=True)
    cfg.crawl_full_text = log_config_override(
        "web_search.crawl_full_text",
        cfg.crawl_full_text,
        False,
        reason="smoke test asserts empty full_text and skips slow page crawls",
    )
    env_top_k = os.environ.get("LANGGRAPH_WEB_SEARCH_TOP_K")
    if env_top_k:
        cfg.top_k = log_config_override(
            "web_search.top_k",
            cfg.top_k,
            int(env_top_k),
            reason="LANGGRAPH_WEB_SEARCH_TOP_K env override",
        )
    return cfg


def _provider_label() -> str:
    key = _web_cfg().api_key or os.environ.get("SERPER_API_KEY")
    return "serper" if key else "duckduckgo"


def _web_provider_up() -> bool:
    """Lightweight reachability check without starting pytest's event loop."""
    cfg = _web_cfg()
    api_key = cfg.api_key or os.environ.get("SERPER_API_KEY")
    if api_key:
        try:
            import httpx

            resp = httpx.post(
                "https://google.serper.dev/search",
                json={"q": "ping"},
                headers={"X-API-KEY": api_key},
                timeout=20.0,
            )
            if resp.status_code != 200:
                return False
            organic = resp.json().get("organic") or []
            return bool(organic)
        except Exception:
            return False
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text("ping", max_results=1))
        return bool(hits)
    except Exception:
        return False


requires_web = pytest.mark.skipif(
    not _web_provider_up(),
    reason=(
        f"Web search provider ({_provider_label()}) unreachable or returned no URLs for "
        f"{_QUERY!r}. Set SERPER_API_KEY in .env or ensure DuckDuckGo is reachable."
    ),
)

pytestmark = [pytest.mark.integration, requires_web]


@pytest.fixture(autouse=True)
def _wire_web_search():
    from langgraph_coe.tools import web as web_mod

    web_mod._web_search_instance = None
    web_mod._web_config = None
    web_mod._web_cache = None
    init_web_search(_web_cfg())
    reset_web_research_session()
    yield
    web_mod._web_search_instance = None
    web_mod._web_config = None
    web_mod._web_cache = None


@requires_web
async def test_web_search_returns_structured_hits():
    """Live provider returns dicts with title, url, and snippet."""
    results = await web_search.ainvoke({"query": _QUERY})
    assert results, f"expected hits from {_provider_label()}"
    assert len(results) <= _web_cfg().top_k
    first = results[0]
    for key in ("title", "url", "snippet", "full_text"):
        assert key in first, f"result must include {key!r}"
    assert str(first["url"]).startswith("http"), first
    assert str(first["title"]).strip(), first
    assert _web_cfg().crawl_full_text is False
    assert first.get("full_text") == ""


@requires_web
async def test_web_search_dedupes_urls_within_session():
    """ContextVar session drops URLs already returned earlier in the same question."""
    first = await web_search.ainvoke({"query": _QUERY})
    urls_first = {str(r["url"]) for r in first if r.get("url")}
    assert urls_first

    second = await web_search.ainvoke({"query": _QUERY})
    urls_second = {str(r["url"]) for r in second if r.get("url")}
    assert urls_first.isdisjoint(urls_second), (
        "URLs from the first call must not reappear in the second call within one session; "
        f"overlap={urls_first & urls_second!r}"
    )

    reset_web_research_session()
    third = await web_search.ainvoke({"query": _QUERY})
    assert third, "after reset_web_research_session, provider should return hits again"
