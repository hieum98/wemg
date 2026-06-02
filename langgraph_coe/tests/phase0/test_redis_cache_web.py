"""Phase 0 §3.4 (db=1, prefix ``web:``) — Web-search cache target specs.

Plan §3.4:
  Web search cache (also db=1, prefix `web:`): keyed by ``(query, top_k)``,
  TTL 24h. Cuts agent-driven re-queries within ``WebResearchGraph`` to near-free.

Tests:
  * Cache miss writes a key under the ``web:`` prefix with 24h TTL.
  * Cache hit skips the provider call.
  * Distinct ``(query, top_k)`` tuples produce distinct keys.
  * LLM (db=0), Wikidata (db=1 ``wd:*``), and web (db=1 ``web:*``) coexist.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

TTL_24H = 86_400


def _init_web_search_with_cache(monkeypatch, fake_redis, *, provider_results: List[Dict[str, str]]):
    """Wire ``langgraph_coe.tools.web`` against a stub provider + cache.

    Returns a ``call_count`` dict the test can inspect.
    """
    from langgraph_coe.config import WebSearchConfig
    from langgraph_coe.tools import web as web_mod

    call_count = {"n": 0}

    class _StubProvider:
        def results(self, query: str, *args, **kwargs):
            call_count["n"] += 1
            return list(provider_results)

    monkeypatch.setattr(web_mod, "_web_search_instance", _StubProvider(), raising=False)
    monkeypatch.setattr(
        web_mod, "_web_config",
        WebSearchConfig(top_k=5, crawl_full_text=False),
        raising=False,
    )

    # Phase 0 introduces a cache attribute on the web module — patch it in.
    if not hasattr(web_mod, "_web_cache"):
        pytest.skip(
            "Phase 0 §3.4 introduces a cache hook on langgraph_coe.tools.web; "
            "module attribute _web_cache not present yet."
        )

    # Build the cache wrapper used by Phase 0.
    try:
        from langgraph_coe.tools.cache import RedisDictCache  # type: ignore
    except ImportError:
        pytest.skip("RedisDictCache (Phase 0 §3.4) not yet implemented")
    cache = RedisDictCache(client=fake_redis, ttls={"web": TTL_24H})
    monkeypatch.setattr(web_mod, "_web_cache", cache, raising=False)

    return web_mod, call_count


async def test_web_search_miss_writes_24h_ttl(monkeypatch, fake_redis):
    web_mod, _ = _init_web_search_with_cache(
        monkeypatch, fake_redis,
        provider_results=[{"title": "T", "snippet": "S", "link": "https://example.com/a"}],
    )

    await web_mod.web_search.ainvoke({"query": "berlin"})

    keys = list(fake_redis.scan_iter(match="web:*"))
    assert keys, "first web_search must populate a `web:*` key (§3.4)"
    ttl = fake_redis.ttl(keys[0])
    assert TTL_24H - 2 <= ttl <= TTL_24H, f"expected 24h TTL, got {ttl}s"


async def test_web_search_hit_skips_provider(monkeypatch, fake_redis):
    web_mod, calls = _init_web_search_with_cache(
        monkeypatch, fake_redis,
        provider_results=[{"title": "T", "snippet": "S", "link": "https://example.com/a"}],
    )

    await web_mod.web_search.ainvoke({"query": "berlin"})
    n_after_warmup = calls["n"]
    await web_mod.web_search.ainvoke({"query": "berlin"})

    assert calls["n"] == n_after_warmup, (
        f"second web_search('berlin') must hit the cache; provider was called "
        f"{calls['n'] - n_after_warmup} extra time(s)"
    )


async def test_web_search_distinct_top_k_distinct_keys(monkeypatch, fake_redis):
    web_mod, _ = _init_web_search_with_cache(
        monkeypatch, fake_redis,
        provider_results=[{"title": "T", "snippet": "S", "link": "https://example.com/a"}],
    )

    # Vary top_k via _web_config to force key disambiguation.
    web_mod._web_config.top_k = 5
    await web_mod.web_search.ainvoke({"query": "berlin"})
    web_mod._web_config.top_k = 10
    await web_mod.web_search.ainvoke({"query": "berlin"})

    keys = list(fake_redis.scan_iter(match="web:*"))
    assert len(set(keys)) >= 2, (
        f"cache must key on (query, top_k); saw only {keys!r} after top_k variation"
    )


async def test_wd_and_web_keys_coexist_in_db1(monkeypatch, fake_redis):
    """``wd:*`` and ``web:*`` prefixes share db=1 without collision."""
    # Seed a wd key first.
    fake_redis.set("wd:entity:Q64", b"{}", ex=10)
    web_mod, _ = _init_web_search_with_cache(
        monkeypatch, fake_redis,
        provider_results=[{"title": "T", "snippet": "S", "link": "https://example.com/a"}],
    )
    await web_mod.web_search.ainvoke({"query": "berlin"})

    keys = {k.decode() if isinstance(k, bytes) else k for k in fake_redis.scan_iter()}
    assert any(k.startswith("wd:") for k in keys), "wd:* key disappeared after web write"
    assert any(k.startswith("web:") for k in keys), "web:* key must coexist with wd:* in db=1"
