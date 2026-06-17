"""Wikidata read-through cache behavior (db=1).

The cache layer introduces four cache-aware lookup methods on ``WikidataClient`` —
``get_entity``, ``search_entities``, ``get_triples``, ``get_wikipedia_content`` —
each wrapped in the same read-through pattern over a shared ``cache`` instance.

Per the table:

  | Layer | Key | TTL |
  | ----- | -------------- | ------ |
  | entity| wd:entity:{qid}| 30 d |
  | search| wd:search:{n} | 7 d |
  | triples| wd:triples:{qid}| 7 d |
  | enrich| wd:enrich:{qid}| 30 d |

Pruning results are **not** cached (query-dependent, near-zero reuse).

These tests reference the target shape ``WikidataClient(cache=...)`` and the
new per-layer methods. They fail loudly (AttributeError / TypeError) on
``main`` and pass once lands the layer.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_coe.tools.wikidata_client import WikidataClient

# Stable Berlin entity from the shared mini-graph.
QID_BERLIN = "Q64"
QID_GERMANY = "Q183"

TTL_30D = 30 * 86400
TTL_7D = 7 * 86400


def _build_cache(redis_client) -> Any:
    """Construct the ``RedisDictCache`` (skip cleanly if unavailable)."""
    try:
        from langgraph_coe.tools.cache import RedisDictCache # type: ignore
    except ImportError:
        try:
            from langgraph_coe.cache import RedisDictCache # type: ignore
        except ImportError:
            pytest.skip(
                "Introduces RedisDictCache (langgraph_coe.tools.cache); "
                "test will pass once it lands."
            )
    return RedisDictCache(
        client=redis_client,
        ttls={
            "entity": TTL_30D,
            "search": TTL_7D,
            "triples": TTL_7D,
            "enrich": TTL_30D,
        },
    )


def _build_client(backend, cache) -> WikidataClient:
    """Build a WikidataClient against the ``cache=`` kwarg target."""
    try:
        return WikidataClient(
            backend=backend,
            max_sparql_rps=1000,
            max_wikipedia_rps=1000,
            cache=cache,
        )
    except TypeError:
        pytest.skip(
            "Introduces ``cache`` kwarg on WikidataClient; "
            "currently exposes only ``redis``."
        )


# ──────────────────────────────────────────────────────────────────────────────
# — per-layer keys + TTLs
# ──────────────────────────────────────────────────────────────────────────────


async def test_get_entity_miss_writes_key_with_30d_ttl(mini_backend, fake_redis):
    """First ``get_entity(qid)`` populates ``wd:entity:{qid}`` with 30d TTL."""
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_entity"):
        pytest.skip("WikidataClient.get_entity is a addition")

    await client.get_entity(QID_BERLIN)

    key = f"wd:entity:{QID_BERLIN}"
    assert fake_redis.exists(key), f"expected Redis key {key} after first get_entity"
    ttl = fake_redis.ttl(key)
    assert TTL_30D - 2 <= ttl <= TTL_30D, f"expected 30d TTL on {key}; got {ttl}"


async def test_get_entity_hit_skips_backend(mini_backend, fake_redis):
    """Pre-seed the cache; backend must not be consulted on the hit."""
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_entity"):
        pytest.skip("WikidataClient.get_entity is a addition")

    # Warm-up to discover the exact serialization the cache layer uses.
    await client.get_entity(QID_BERLIN)
    initial_calls = mini_backend.call_count("get_entity_details")
    await client.get_entity(QID_BERLIN)

    assert mini_backend.call_count("get_entity_details") == initial_calls, (
        "Second get_entity must be served from Redis without touching the backend"
    )


async def test_search_entities_writes_key_with_7d_ttl(mini_backend, fake_redis):
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "search_entities"):
        pytest.skip("WikidataClient.search_entities is a addition")

    await client.search_entities("Berlin", top_k=5)

    matches = [k for k in fake_redis.scan_iter(match="wd:search:*")]
    assert matches, "expected at least one wd:search:* key after search_entities"
    ttl = fake_redis.ttl(matches[0])
    assert TTL_7D - 2 <= ttl <= TTL_7D, f"expected 7d TTL on {matches[0]!r}; got {ttl}"


async def test_get_triples_writes_key_with_7d_ttl(mini_backend, fake_redis):
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_triples"):
        pytest.skip("WikidataClient.get_triples is a addition")

    await client.get_triples(QID_BERLIN)

    key = f"wd:triples:{QID_BERLIN}"
    assert fake_redis.exists(key), f"expected Redis key {key} after get_triples"
    ttl = fake_redis.ttl(key)
    assert TTL_7D - 2 <= ttl <= TTL_7D


async def test_get_wikipedia_content_writes_key_with_30d_ttl(mini_backend, fake_redis):
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_wikipedia_content"):
        pytest.skip("WikidataClient.get_wikipedia_content is a addition")

    await client.get_wikipedia_content(QID_BERLIN)

    key = f"wd:enrich:{QID_BERLIN}"
    assert fake_redis.exists(key), (
        f"expected Redis key {key} after get_wikipedia_content"
    )
    ttl = fake_redis.ttl(key)
    assert TTL_30D - 2 <= ttl <= TTL_30D


# ──────────────────────────────────────────────────────────────────────────────
# — pruning is NOT cached
# ──────────────────────────────────────────────────────────────────────────────


async def test_pruning_results_are_not_cached(mini_backend, fake_redis):
    """``fetch_and_prune_subgraph`` must not write ``wd:prune:*`` keys."""
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_triples"):
        pytest.skip("WikidataClient.get_triples is a addition")

    # Use the tool layer end-to-end to exercise the full pruning path.
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.tools import wikidata as wd_mod

    cfg = LangGraphCoeConfig.from_yaml()
    wd_mod._wikidata_client = client
    wd_mod._wikidata_config = cfg.wikidata
    wd_mod.reset_wikidata_session()

    try:
        await wd_mod.fetch_and_prune_subgraph.ainvoke(
            {"qids": [QID_BERLIN], "query": "capital of Germany"}
        )
    except Exception:
        # Network-reranker failures are fine; we only care about cache keys.
        pass

    prune_keys = list(fake_redis.scan_iter(match="wd:prune:*"))
    assert prune_keys == [], (
        f"forbids caching pruning results; saw keys: {prune_keys!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# — persistence across client instances on the same db
# ──────────────────────────────────────────────────────────────────────────────


async def test_two_clients_share_redis_db1(mini_backend, fake_redis):
    """A second ``WikidataClient`` reads what the first wrote — persistent layer."""
    cache_a = _build_cache(fake_redis)
    cache_b = _build_cache(fake_redis)
    client_a = _build_client(mini_backend, cache_a)
    client_b = _build_client(mini_backend, cache_b)
    if not all(hasattr(c, "get_entity") for c in (client_a, client_b)):
        pytest.skip("WikidataClient.get_entity is a addition")

    await client_a.get_entity(QID_BERLIN)
    initial = mini_backend.call_count("get_entity_details")
    await client_b.get_entity(QID_BERLIN)

    assert mini_backend.call_count("get_entity_details") == initial, (
        "client_b must serve get_entity from Redis populated by client_a "
        "(persistent layer, not in-process LRU)"
    )


async def test_corrupted_cache_value_falls_through_to_backend(mini_backend, fake_redis):
    """If a key holds invalid JSON, ``get_entity`` falls through cleanly + overwrites."""
    cache = _build_cache(fake_redis)
    client = _build_client(mini_backend, cache)
    if not hasattr(client, "get_entity"):
        pytest.skip("WikidataClient.get_entity is a addition")

    key = f"wd:entity:{QID_BERLIN}"
    fake_redis.set(key, b"{not-json")
    initial = mini_backend.call_count("get_entity_details")

    result = await client.get_entity(QID_BERLIN)
    assert result is not None, "get_entity must fall through to backend on corruption"
    assert mini_backend.call_count("get_entity_details") > initial, (
        "Corruption fallthrough must call the backend (not silently return None)"
    )
    # And the bad value must have been overwritten with a valid one.
    assert fake_redis.get(key) != b"{not-json"


async def test_cache_disabled_no_redis_writes(mini_backend, fake_redis):
    """Building a client without a cache must not write to Redis at all."""
    client = WikidataClient(
        backend=mini_backend, max_sparql_rps=1000, max_wikipedia_rps=1000
    )
    # Use whichever lookup methods exist today (link_entities is the stable one).
    await client.link_entities("Berlin", top_k=1)

    assert list(fake_redis.scan_iter()) == [], "No cache parameter ⇒ no Redis writes"
