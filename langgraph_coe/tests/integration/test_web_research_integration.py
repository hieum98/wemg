"""Phase 0 integration smoke test — live ``WebResearchGraph`` (LLM + web search).

Exercises ``build_web_research_graph`` end-to-end against the configured light-tier
LLM (default ``config.yaml``: ``openai/Qwen/Qwen3.5-4B`` @ ``http://n0152:30000/v1``)
and the live ``web_search`` provider (Serper when ``SERPER_API_KEY`` is set, else
DuckDuckGo). Skipped when either endpoint is unreachable.

Run manually (from repo root)::

    uv run pytest langgraph_coe/tests/phase0/test_web_research_integration.py -v -s

Optional env overrides::

    API_KEY / OPENAI_API_KEY     LLM auth (loaded from repo-root ``.env``)
    SERPER_API_KEY               web search provider
    LANGGRAPH_TEST_LLM_URL       override light-tier ``api_base``
    LANGGRAPH_TEST_LLM_MODEL     override light-tier ``model_name``
    LANGGRAPH_WEB_SEARCH_TOP_K   override ``web_search.top_k`` for the smoke query
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, WebSearchConfig
from langgraph_coe.graphs.web_research import (
    build_web_research_graph,
    set_web_research_config,
)
from langgraph_coe.llm import RoleModelRegistry
from langgraph_coe.tools.web import init_web_search, reset_web_research_session

from .._fixtures import log_config_override, override_tier_endpoint
from .._servers import endpoint_alive as _endpoint_alive

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Must require external lookup: a thinking-enabled model answers trivia (e.g. a
# capital city) from parametric memory and never calls ``web_search``, which would
# fail the ``queries_issued >= 1`` assertion below.
_QUERY = (
    "Use the web_search tool to find the current CEO of the Mozilla Corporation "
    "and give one source URL."
)


def _integration_cfg() -> LangGraphCoeConfig:
    return LangGraphCoeConfig.from_yaml()


def _llm_url() -> str:
    env = os.environ.get("LANGGRAPH_TEST_LLM_URL")
    if env:
        return env
    return _integration_cfg().llm.tiers["light"].api_base


def _llm_model() -> str:
    env = os.environ.get("LANGGRAPH_TEST_LLM_MODEL")
    if env:
        return env
    return _integration_cfg().llm.tiers["light"].model_name


def _web_cfg() -> WebSearchConfig:
    """config.yaml ``web_search`` block; full-text crawl disabled for a fast smoke."""
    cfg = _integration_cfg().web_search.model_copy(deep=True)
    cfg.crawl_full_text = log_config_override(
        "web_search.crawl_full_text",
        cfg.crawl_full_text,
        False,
        reason="smoke test skips slow page crawls",
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
    cfg = _web_cfg()
    api_key = cfg.api_key or os.environ.get("SERPER_API_KEY")
    if api_key:
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(
                    "https://google.serper.dev/search",
                    json={"q": "ping"},
                    headers={"X-API-KEY": api_key},
                    timeout=20.0,
                )
            if resp.status_code != 200:
                return False
            return bool(resp.json().get("organic"))
        except Exception:
            return False
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return bool(list(ddgs.text("ping", max_results=1)))
    except Exception:
        return False


def _real_stack_available() -> bool:
    return _endpoint_alive(_llm_url()) and _web_provider_up()


_skip_reason = (
    f"WebResearchGraph integration requires a reachable LLM ({_llm_url()}) and "
    f"web search provider ({_provider_label()}). "
    "Set API_KEY and SERPER_API_KEY in .env, or override LANGGRAPH_TEST_LLM_URL."
)

requires_web_research_stack = pytest.mark.skipif(
    not _real_stack_available(),
    reason=_skip_reason,
)

pytestmark = [pytest.mark.integration, requires_web_research_stack]


def _build_registry() -> RoleModelRegistry:
    """light tier from config.yaml; only model/endpoint deviate (env-driven, logged)."""
    cfg = _integration_cfg()
    cfg.llm.tiers["light"] = override_tier_endpoint(
        cfg,
        "light",
        model_name=_llm_model(),
        api_base=_llm_url(),
        reason="web-research integration LLM endpoint (LANGGRAPH_TEST_LLM_URL/MODEL)",
    )
    return RoleModelRegistry(cfg.llm)


@pytest.fixture(autouse=True)
def _wire_web_search():
    from langgraph_coe.tools import web as web_mod

    web_mod._web_search_instance = None
    web_mod._web_config = None
    web_mod._web_cache = None
    web_cfg = _web_cfg()
    init_web_search(web_cfg)
    set_web_research_config(web_cfg)
    reset_web_research_session()
    yield
    web_mod._web_search_instance = None
    web_mod._web_config = None
    web_mod._web_cache = None


@requires_web_research_stack
async def test_web_research_graph_live_smoke():
    """Full graph: real LLM agent calls ``web_search``; finalize returns structured hits."""
    registry = _build_registry()
    graph = build_web_research_graph(registry)

    final = await graph.ainvoke(
        {
            "subquery": _QUERY,
            "original_query": _QUERY,
            "context": "",
        }
    )

    assert not final.get("errors"), f"unexpected errors: {final.get('errors')!r}"
    assert (final.get("queries_issued") or 0) >= 1, (
        "agent should issue at least one web_search call"
    )

    results = final.get("results") or []
    assert results, (
        f"expected structured results from live stack (LLM={_llm_model()} @ {_llm_url()}, "
        f"web={_provider_label()})"
    )
    assert len(results) <= _web_cfg().top_k

    first: dict[str, Any] = results[0]
    for key in ("title", "url", "snippet", "full_text"):
        assert key in first, f"result must include {key!r}"
    assert str(first["url"]).startswith("http"), first
    assert str(first["title"]).strip(), first

    urls = {str(r.get("url", "")).strip() for r in results if r.get("url")}
    assert len(urls) == len(results), f"finalize must dedupe by url; saw {results!r}"
