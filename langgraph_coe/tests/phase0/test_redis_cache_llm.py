"""Phase 0 §3.4 (db=0) — LangChain LLM cache wiring target specs.

Plan §3.4 wiring:

    LLM cache (db=0): LangChain ``RedisCache`` over ``ChatLiteLLM`` invocations.

    Both initialized in ``system.py`` after config load when ``config.cache.enabled``.

These tests assert:
  * ``system.answer()`` calls ``set_llm_cache`` with a ``RedisCache`` backed by
    a Redis client targeting db=0.
  * When ``config.cache.enabled = False`` the cache is left unset.
  * The LLM cache and the Wikidata cache use distinct Redis db indices.

Also a behavioral spec: a second identical LLM call after warm-up does not
re-invoke the underlying model (LangChain's RedisCache handles the actual
storage; we only verify wiring at this layer).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache


def _system_or_skip():
    sys_mod = pytest.importorskip(
        "langgraph_coe.system",
        reason="Phase 0 §3.4 wires the LLM cache into system.py",
    )
    if not hasattr(sys_mod, "answer"):
        pytest.skip("system.answer not implemented yet (Phase 0 §4 target)")
    return sys_mod


def _ensure_cache_config(cfg):
    """Phase 0 adds ``cfg.cache``. Skip if it's missing instead of failing.

    Once Phase 0 lands the attribute exists and the test runs.
    """
    if not hasattr(cfg, "cache"):
        pytest.skip("LangGraphCoeConfig.cache field is a Phase 0 §3.4 addition")
    return cfg


@pytest.fixture
def fresh_global_cache():
    """Ensure ``langchain.globals`` has no LLM cache set at test entry/exit."""
    # Already imported at module top.
    previous = get_llm_cache()
    set_llm_cache(None)
    yield
    set_llm_cache(previous)


def _stub_strategy(monkeypatch, system):
    stub = MagicMock()
    stub.ainvoke = AsyncMock(return_value={"final_answer": "x", "errors": []})
    monkeypatch.setattr(system, "build_mcts_graph", lambda *a, **kw: stub, raising=False)
    monkeypatch.setattr(system, "build_cot_graph", lambda *a, **kw: stub, raising=False)


async def test_set_llm_cache_called_with_redis_db_0(monkeypatch, fresh_global_cache):
    """After ``answer()`` init, the global LLM cache is a RedisCache on db=0."""
    system = _system_or_skip()
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = _ensure_cache_config(LangGraphCoeConfig.from_yaml())
    cfg.cache.enabled = True

    _stub_strategy(monkeypatch, system)
    # Don't actually try to connect to localhost:6379 in CI: patch Redis client
    # construction so any redis.Redis(...) call is captured.
    captured: dict = {}
    import redis as redis_mod
    orig_redis_cls = redis_mod.Redis

    def _capture_redis(*args, **kwargs):
        captured.update(kwargs)
        # Return a fakeredis-backed instance so RedisCache init doesn't fail.
        import fakeredis
        return fakeredis.FakeStrictRedis(decode_responses=False)

    monkeypatch.setattr(redis_mod, "Redis", _capture_redis, raising=True)

    await system.answer("Q?", cfg)

    cache = get_llm_cache()
    assert cache is not None, "system.answer must call set_llm_cache when cache enabled"
    # Either RedisCache from langchain_community or langchain_redis.
    cls_name = type(cache).__name__
    assert "RedisCache" in cls_name, f"Expected a RedisCache, got {cls_name!r}"
    assert captured.get("db") == 0, (
        f"LLM cache must target Redis db=0; saw db={captured.get('db')!r}"
    )


async def test_llm_cache_disabled_leaves_global_cache_unset(monkeypatch, fresh_global_cache):
    """``config.cache.enabled = False`` ⇒ no LLM cache wiring."""
    system = _system_or_skip()
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = _ensure_cache_config(LangGraphCoeConfig.from_yaml())
    cfg.cache.enabled = False

    _stub_strategy(monkeypatch, system)
    await system.answer("Q?", cfg)

    assert get_llm_cache() is None, (
        "system.answer must NOT call set_llm_cache when cache disabled"
    )


async def test_llm_and_wikidata_use_separate_db_indices(monkeypatch, fresh_global_cache):
    """Plan §3.4 table: db=0 → LLM, db=1 → Wikidata + web."""
    system = _system_or_skip()
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = _ensure_cache_config(LangGraphCoeConfig.from_yaml())
    cfg.cache.enabled = True

    _stub_strategy(monkeypatch, system)

    dbs_used: list[int] = []
    import redis as redis_mod

    def _capture_redis(*args, **kwargs):
        dbs_used.append(kwargs.get("db"))
        import fakeredis
        return fakeredis.FakeStrictRedis(decode_responses=False)

    monkeypatch.setattr(redis_mod, "Redis", _capture_redis, raising=True)

    await system.answer("Q?", cfg)

    assert 0 in dbs_used, f"LLM cache must use db=0; saw {dbs_used}"
    assert 1 in dbs_used, f"Wikidata/web cache must use db=1; saw {dbs_used}"


async def test_llm_cache_hit_skips_second_model_call(fresh_global_cache):
    """End-to-end behavior spec for the LLM cache.

    Build a tiny LangChain-cached pipeline: the second identical invocation
    against the cached path must NOT call the underlying chat model.
    """
    pytest.importorskip("langchain_community.cache")
    from langchain_community.cache import RedisCache
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    import fakeredis

    redis_client = fakeredis.FakeStrictRedis(decode_responses=False)
    set_llm_cache(RedisCache(redis_=redis_client))

    call_count = {"n": 0}

    class CountingFake(FakeListChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            call_count["n"] += 1
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = CountingFake(responses=["the answer"], cache=True)

    out1 = await model.ainvoke("ping")
    out2 = await model.ainvoke("ping")

    assert out1.content == out2.content
    assert call_count["n"] == 1, (
        f"Second identical LLM call must be served from RedisCache; underlying model was hit {call_count['n']}× "
        "(expected 1)"
    )
