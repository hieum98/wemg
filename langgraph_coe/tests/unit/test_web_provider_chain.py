"""Web search provider chain: ordering, isolation, and spend accounting.

The chain exists because both original tiers died at once — ``ddgs`` became
unreachable from the eval host and the Tavily free tier hit its 1000-request cap —
and the failure was invisible, because a dead provider returns ``[]`` rather than
raising. These tests pin the properties that made it invisible, plus the ordering
that keeps the finite buckets from being spent on queries the unmetered providers
can answer.

Tests:
  * A provider that answers stops the chain, so later (metered/billed) providers are
    never called.
  * A provider that raises or returns [] is skipped, not fatal.
  * An unknown provider name is rejected at init instead of silently dropped.
  * ``wikipedia`` returns page text inline so the crawl is skipped.
  * ``searxng`` backfills from ``infoboxes``, which is where its wikipedia/wikidata
    engines answer.
  * Tavily's billed-call counter counts only requests that actually succeeded.
"""

from __future__ import annotations

import json
from typing import Dict, List

import httpx
import pytest

from langgraph_coe.config import WebSearchConfig
from langgraph_coe.tools import web as web_mod


def _init(monkeypatch, providers: List[str], **kw) -> None:
    """Initialise the module with a chain and no crawling (crawl hits the network)."""
    monkeypatch.setattr(web_mod, "_web_cache", None, raising=False)
    cfg = WebSearchConfig(
        enabled=True, top_k=3, crawl_full_text=False, providers=providers, **kw
    )
    web_mod.init_web_search(cfg)


def _stub(calls: List[str], name: str, results: List[Dict[str, str]] | None = None):
    async def _run(query: str, top_k: int):
        calls.append(name)
        return list(results or [])

    return _run


def _raiser(calls: List[str], name: str):
    async def _run(query: str, top_k: int):
        calls.append(name)
        raise RuntimeError(f"{name} is down")

    return _run


def _hit(url: str) -> Dict[str, str]:
    return {"title": "T", "snippet": "S", "url": url, "full_text": ""}


# ── ordering ──────────────────────────────────────────────────────────────────


async def test_first_answering_provider_stops_the_chain(monkeypatch):
    """The whole point of the ordering: SearXNG answering must cost nothing later."""
    calls: List[str] = []
    _init(monkeypatch, ["searxng", "builtin", "brave", "tavily"])
    monkeypatch.setitem(
        web_mod._PROVIDERS, "searxng", _stub(calls, "searxng", [_hit("https://a.test")])
    )
    for later in ("builtin", "brave", "tavily"):
        monkeypatch.setitem(web_mod._PROVIDERS, later, _stub(calls, later, [_hit("https://x.test")]))

    out = await web_mod.web_search.ainvoke({"query": "q"})

    assert len(out) == 1
    assert calls == ["searxng"], (
        f"a satisfied query must not reach the metered providers; chain ran {calls}"
    )


async def test_chain_walks_past_empty_and_raising_providers(monkeypatch):
    """An empty provider and a raising provider are both skipped, not fatal."""
    calls: List[str] = []
    _init(monkeypatch, ["searxng", "builtin", "brave", "wikipedia"])
    monkeypatch.setitem(web_mod._PROVIDERS, "searxng", _stub(calls, "searxng", []))
    monkeypatch.setitem(web_mod._PROVIDERS, "builtin", _raiser(calls, "builtin"))
    monkeypatch.setitem(web_mod._PROVIDERS, "brave", _stub(calls, "brave", []))
    monkeypatch.setitem(
        web_mod._PROVIDERS, "wikipedia", _stub(calls, "wikipedia", [_hit("https://w.test")])
    )

    out = await web_mod.web_search.ainvoke({"query": "q"})

    assert calls == ["searxng", "builtin", "brave", "wikipedia"]
    assert [r["url"] for r in out] == ["https://w.test"]


async def test_fully_dead_chain_returns_empty_and_warns(monkeypatch, caplog):
    """The exact regression that hid for a whole sweep: everything dead, no signal."""
    calls: List[str] = []
    _init(monkeypatch, ["searxng", "tavily"])
    monkeypatch.setitem(web_mod._PROVIDERS, "searxng", _stub(calls, "searxng", []))
    monkeypatch.setitem(web_mod._PROVIDERS, "tavily", _stub(calls, "tavily", []))

    with caplog.at_level("WARNING"):
        out = await web_mod.web_search.ainvoke({"query": "q"})

    assert out == []
    assert any("no results from any provider" in r.message for r in caplog.records), (
        "a fully dead chain must warn; silence is what made this cost a full sweep"
    )


async def test_unknown_provider_name_rejected_at_init(monkeypatch):
    with pytest.raises(ValueError, match="unknown provider"):
        _init(monkeypatch, ["searxng", "gogle"])


# ── provider response parsing ─────────────────────────────────────────────────


def _mock_http(monkeypatch, handler) -> None:
    """Point every ``httpx.AsyncClient()`` in the module at a mock transport."""
    # Bind the real class before patching; referring to ``httpx.AsyncClient`` inside
    # the factory would resolve to the factory itself and recurse.
    real = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", _factory)


async def test_wikipedia_returns_page_text_inline(monkeypatch):
    """``full_text`` populated means ``web_search`` skips the per-URL crawl."""
    payload = {
        "query": {
            "pages": {
                "42": {
                    "pageid": 42,
                    "index": 2,
                    "title": "Second Page",
                    "extract": "second body",
                    "fullurl": "https://en.wikipedia.org/wiki/Second_Page",
                },
                "7": {
                    "pageid": 7,
                    "index": 1,
                    "title": "First Page",
                    "extract": "first body",
                    "fullurl": "https://en.wikipedia.org/wiki/First_Page",
                },
            }
        }
    }
    _mock_http(monkeypatch, lambda req: httpx.Response(200, json=payload))
    _init(monkeypatch, ["wikipedia"])

    out = await web_mod._search_wikipedia("q", 3)

    # ``pages`` is keyed by pageid, so only ``index`` carries the search rank.
    assert [r["title"] for r in out] == ["First Page", "Second Page"], (
        f"results must follow search rank, not dict order; got {[r['title'] for r in out]}"
    )
    assert out[0]["full_text"] == "first body"
    assert out[0]["url"] == "https://en.wikipedia.org/wiki/First_Page"


async def test_wikipedia_url_falls_back_to_constructed_link(monkeypatch):
    """``inprop=url`` can be absent on a page; a result with no URL cannot be cited."""
    payload = {"query": {"pages": {"1": {"index": 1, "title": "Ada Lovelace", "extract": "x"}}}}
    _mock_http(monkeypatch, lambda req: httpx.Response(200, json=payload))
    _init(monkeypatch, ["wikipedia"])

    out = await web_mod._search_wikipedia("q", 3)

    assert out[0]["url"] == "https://en.wikipedia.org/wiki/Ada_Lovelace"


async def test_searxng_backfills_from_infoboxes(monkeypatch):
    """wikipedia/wikidata engines answer into ``infoboxes``, not ``results``."""
    payload = {
        "results": [],
        "infoboxes": [
            {
                "infobox": "Bolivia",
                "content": "Bolivia is a country in South America.",
                "id": "https://en.wikipedia.org/wiki/Bolivia",
                "urls": [{"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Bolivia"}],
            }
        ],
    }
    _mock_http(monkeypatch, lambda req: httpx.Response(200, json=payload))
    _init(monkeypatch, ["searxng"])

    out = await web_mod._search_searxng("capital of Bolivia", 3)

    assert len(out) == 1, "an infobox-only response must not look empty"
    assert out[0]["url"] == "https://en.wikipedia.org/wiki/Bolivia"
    assert "South America" in out[0]["snippet"]


async def test_searxng_unreachable_is_not_fatal(monkeypatch):
    """The container is optional; without it the chain must simply move on."""

    def _boom(req):
        raise httpx.ConnectError("connection refused", request=req)

    _mock_http(monkeypatch, _boom)
    _init(monkeypatch, ["searxng"])

    assert await web_mod._search_searxng("q", 3) == []


async def test_brave_strips_highlight_markup(monkeypatch):
    payload = {
        "web": {
            "results": [
                {
                    "title": "Henry Ford",
                    "description": "<strong>Henry Ford</strong> founded the company.",
                    "url": "https://en.wikipedia.org/wiki/Henry_Ford",
                }
            ]
        }
    }
    _mock_http(monkeypatch, lambda req: httpx.Response(200, json=payload))
    _init(monkeypatch, ["brave"], brave_api_key="k")

    out = await web_mod._search_brave("q", 3)

    assert out[0]["snippet"] == "Henry Ford founded the company."


async def test_brave_without_key_is_inert(monkeypatch):
    """A keyless provider must no-op rather than fire an unauthenticated request."""
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    fired: List[str] = []
    _mock_http(monkeypatch, lambda req: fired.append(str(req.url)) or httpx.Response(200, json={}))
    _init(monkeypatch, ["brave"])

    assert await web_mod._search_brave("q", 3) == []
    assert fired == []


# ── spend accounting ──────────────────────────────────────────────────────────


async def test_quota_rejected_tavily_call_is_not_counted_as_spend(monkeypatch):
    """A 432 is a refusal, not a billed search; counting it overstated spend 1000x."""
    _mock_http(
        monkeypatch,
        lambda req: httpx.Response(
            432, json={"detail": {"error": "This request exceeds your plan's set usage limit"}}
        ),
    )
    _init(monkeypatch, ["tavily"], tavily_api_key="k")

    out = await web_mod._search_tavily("q", 3)

    assert out == []
    assert web_mod.read_tavily_usage() == 0, (
        f"quota-rejected call must not count as spend; counter reads "
        f"{web_mod.read_tavily_usage()}"
    )


async def test_successful_tavily_call_is_counted(monkeypatch):
    payload = {
        "results": [
            {"title": "T", "content": "C", "url": "https://a.test", "raw_content": "BODY"}
        ]
    }
    _mock_http(monkeypatch, lambda req: httpx.Response(200, json=payload))
    _init(monkeypatch, ["tavily"], tavily_api_key="k")

    out = await web_mod._search_tavily("q", 3)

    assert out[0]["full_text"] == "BODY"
    assert web_mod.read_tavily_usage() == 1
    assert web_mod.read_provider_usage()["tavily"] == 1


async def test_init_resets_usage_between_runs(monkeypatch):
    """Per-process counters would otherwise leak across sweeps in one interpreter."""
    _init(monkeypatch, ["tavily"], tavily_api_key="k")
    web_mod._count("tavily")
    assert web_mod.read_tavily_usage() == 1

    _init(monkeypatch, ["tavily"], tavily_api_key="k")
    assert web_mod.read_provider_usage() == {}
