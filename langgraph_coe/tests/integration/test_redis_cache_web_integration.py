"""Phase 0 integration — ``web_search`` cache against a live Redis server (db=1).

Mirrors ``test_redis_cache_web.py`` but uses a real ``redis.Redis`` client from
``config.yaml`` (``cache.redis``). Keys are namespaced and deleted in teardown.

Run (from repo root). Full setup: ``docs/setup_redis_cache.md``::

    conda activate wemg
    redis-server --daemonize yes   # if not already listening on cache.redis.host:port
    uv run pytest langgraph_coe/tests/phase0/test_redis_cache_web_integration.py -v -s

Optional env overrides::

    LANGGRAPH_TEST_REDIS_HOST / LANGGRAPH_TEST_REDIS_PORT
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import redis

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.tools.cache import RedisDictCache

from .._fixtures import log_config_override

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

_TEST_QUERY_PREFIX = "__pytest_web_redis__"


def _cfg() -> LangGraphCoeConfig:
    return LangGraphCoeConfig.from_yaml()


def _web_ttl() -> int:
    """Web cache TTL under test — sourced from config.yaml ``cache.web.ttl``."""
    return _cfg().cache.web.ttl


def _redis_host() -> str:
    return os.environ.get("LANGGRAPH_TEST_REDIS_HOST", _cfg().cache.redis.host)


def _redis_port() -> int:
    return int(os.environ.get("LANGGRAPH_TEST_REDIS_PORT", _cfg().cache.redis.port))


def _redis_db() -> int:
    return _cfg().cache.redis.wikidata_db


def _redis_alive() -> bool:
    try:
        client = redis.Redis(
            host=_redis_host(),
            port=_redis_port(),
            db=_redis_db(),
            decode_responses=False,
            socket_connect_timeout=3.0,
        )
        return bool(client.ping())
    except Exception:
        return False


def _test_query(suffix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{_TEST_QUERY_PREFIX}{suffix}_{token}"


requires_redis = pytest.mark.skipif(
    not _redis_alive(),
    reason=(
        f"Redis unreachable at {_redis_host()}:{_redis_port()} db={_redis_db()}. "
        "Start with: conda activate wemg && redis-server --daemonize yes"
    ),
)

pytestmark = [pytest.mark.integration, requires_redis]


@pytest.fixture
def real_redis():
    """Live Redis on the wikidata/web cache db; scrub integration keys after each test."""
    client = redis.Redis(
        host=_redis_host(),
        port=_redis_port(),
        db=_redis_db(),
        decode_responses=False,
    )
    assert client.ping()
    yield client
    for key in client.scan_iter(match=f"web:{_TEST_QUERY_PREFIX}*"):
        client.delete(key)
    client.delete("wd:entity:Q64")


def _init_web_search_with_real_cache(
    monkeypatch,
    real_redis: redis.Redis,
    *,
    provider_results: List[Dict[str, str]],
    top_k: Optional[int] = None,
):
    from langgraph_coe.tools import web as web_mod

    call_count = {"n": 0}

    class _StubProvider:
        def results(self, query: str, *args, **kwargs):
            call_count["n"] += 1
            return list(provider_results)

    # Web cache config + TTL both come from config.yaml; crawl is disabled
    # because the stub provider returns snippet-only rows.
    web_cfg = _cfg().web_search.model_copy(deep=True)
    web_cfg.crawl_full_text = log_config_override(
        "web_search.crawl_full_text",
        web_cfg.crawl_full_text,
        False,
        reason="stub provider returns snippet-only rows; no crawl",
    )
    if top_k is not None:
        web_cfg.top_k = log_config_override(
            "web_search.top_k",
            web_cfg.top_k,
            top_k,
            reason="distinct-top_k keying test sweeps explicit values",
        )

    cache = RedisDictCache(client=real_redis, ttls={"web": _web_ttl()})
    monkeypatch.setattr(web_mod, "_web_search_instance", _StubProvider(), raising=False)
    monkeypatch.setattr(web_mod, "_web_config", web_cfg, raising=False)
    monkeypatch.setattr(web_mod, "_web_cache", cache, raising=False)
    return web_mod, call_count


@requires_redis
async def test_web_search_miss_writes_24h_ttl_on_real_redis(monkeypatch, real_redis):
    web_mod, _ = _init_web_search_with_real_cache(
        monkeypatch,
        real_redis,
        provider_results=[
            {"title": "T", "snippet": "S", "link": "https://example.com/a"}
        ],
    )
    query = _test_query("miss")

    await web_mod.web_search.ainvoke({"query": query})

    keys = [
        k.decode() if isinstance(k, bytes) else k
        for k in real_redis.scan_iter(match=f"web:{query.lower()}:*")
    ]
    assert keys, f"expected web:* key for query {query!r}"
    ttl = real_redis.ttl(keys[0])
    expected_ttl = _web_ttl()
    assert expected_ttl - 5 <= ttl <= expected_ttl, (
        f"expected ~{expected_ttl}s (config.yaml cache.web.ttl), got {ttl}s"
    )


@requires_redis
async def test_web_search_hit_skips_provider_on_real_redis(monkeypatch, real_redis):
    web_mod, calls = _init_web_search_with_real_cache(
        monkeypatch,
        real_redis,
        provider_results=[
            {"title": "T", "snippet": "S", "link": "https://example.com/a"}
        ],
    )
    query = _test_query("hit")

    await web_mod.web_search.ainvoke({"query": query})
    n_after_warmup = calls["n"]
    await web_mod.web_search.ainvoke({"query": query})

    assert calls["n"] == n_after_warmup, (
        f"second call must hit Redis cache; provider called "
        f"{calls['n'] - n_after_warmup} extra time(s)"
    )


@requires_redis
async def test_web_search_distinct_top_k_distinct_keys_on_real_redis(
    monkeypatch, real_redis
):
    web_mod, _ = _init_web_search_with_real_cache(
        monkeypatch,
        real_redis,
        provider_results=[
            {"title": "T", "snippet": "S", "link": "https://example.com/a"}
        ],
    )
    query = _test_query("topk")

    # This test sweeps top_k itself to prove the cache key includes top_k; the two
    # values are the variable under test, not config hyper-parameters.
    web_mod._web_config.top_k = 5
    await web_mod.web_search.ainvoke({"query": query})
    web_mod._web_config.top_k = 10
    await web_mod.web_search.ainvoke({"query": query})

    keys = {
        (k.decode() if isinstance(k, bytes) else k)
        for k in real_redis.scan_iter(match=f"web:{query.lower()}:*")
    }
    assert len(keys) >= 2, f"expected distinct keys per top_k; saw {keys!r}"


@requires_redis
async def test_wd_and_web_keys_coexist_on_real_redis_db(monkeypatch, real_redis):
    real_redis.set("wd:entity:Q64", b"{}", ex=60)
    web_mod, _ = _init_web_search_with_real_cache(
        monkeypatch,
        real_redis,
        provider_results=[
            {"title": "T", "snippet": "S", "link": "https://example.com/a"}
        ],
    )
    query = _test_query("coexist")

    await web_mod.web_search.ainvoke({"query": query})

    keys = {k.decode() if isinstance(k, bytes) else k for k in real_redis.scan_iter()}
    assert any(k.startswith("wd:") for k in keys), (
        "wd:* key must survive web cache write"
    )
    assert any(k.startswith("web:") and query.lower() in k for k in keys), (
        "web:* key must coexist with wd:* on the same Redis db"
    )
