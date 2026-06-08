"""Contract tests for caching: L1 LRU + L2 Redis.

The client must hit the cache on the second call for the same input and avoid
re-calling the backend. With a Redis instance provided, writes propagate to
Redis with the configured TTL; reads on L1 miss check Redis before the backend.
"""

from __future__ import annotations

import pytest

from ._fixtures import (
    PID_CAPITAL,
    QID_BERLIN,
    QID_GERMANY,
    QID_PARIS,
)


# ---------------- L1 LRU ----------------


async def test_link_repeat_same_name_no_backend_call(client, mini_graph):
    await client.link_entities("Berlin", top_k=1)
    n_search = mini_graph.call_count("search_entities_text")
    n_detail = mini_graph.call_count("get_entity_details")
    await client.link_entities("Berlin", top_k=1)
    assert mini_graph.call_count("search_entities_text") == n_search
    assert mini_graph.call_count("get_entity_details") == n_detail


async def test_enrich_repeat_same_qid_no_backend_call(client, mini_graph):
    await client.enrich_entities([QID_BERLIN])
    n_after_first = mini_graph.call_count("get_entity_details")
    await client.enrich_entities([QID_BERLIN])
    assert mini_graph.call_count("get_entity_details") == n_after_first


async def test_khop_outgoing_cache_hit_avoids_fetch(client, mini_graph):
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    n_after = mini_graph.call_count("fetch_outgoing")
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert mini_graph.call_count("fetch_outgoing") == n_after


async def test_khop_bidirectional_populates_outgoing_subset(client, mini_graph):
    """Bidirectional fetch should populate the outgoing cache, so a subsequent
    forward-only call for the same seed is served from cache."""
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=True, enrich=False
    )
    n_after_bidir = mini_graph.call_count("fetch_outgoing")
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert mini_graph.call_count("fetch_outgoing") == n_after_bidir


async def test_property_search_pid_lookup_cached(client, mini_graph):
    await client.search_properties("P36", top_k=1)
    n_after = mini_graph.call_count("get_property_details")
    await client.search_properties("P36", top_k=1)
    assert mini_graph.call_count("get_property_details") == n_after


async def test_lru_evicts_oldest_at_capacity(mini_graph):
    """When capacity is exceeded, the oldest entry is evicted (re-fetched on next use)."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
        lru_capacity=2,
    )
    await c.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
    await c.get_k_hop_triples(QID_GERMANY, k=1, bidirectional=False, enrich=False)
    await c.get_k_hop_triples(QID_PARIS, k=1, bidirectional=False, enrich=False)  # evicts Berlin
    n_after_three = mini_graph.call_count("fetch_outgoing")
    # Berlin should be evicted → re-fetched
    await c.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
    assert mini_graph.call_count("fetch_outgoing") == n_after_three + 1


# ---------------- L2 Redis ----------------


async def test_redis_write_after_l1_miss_then_fetch(client_with_redis, mini_graph, fake_redis):
    assert len(list(fake_redis.scan_iter())) == 0
    await client_with_redis.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    keys = list(fake_redis.scan_iter())
    assert len(keys) > 0, "client should write fetched triples into Redis"


async def test_redis_read_on_l1_miss_no_backend_call(mini_graph, fake_redis):
    """A fresh client (empty L1) with Redis pre-populated must avoid the backend."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    # First client warms Redis
    warm = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000,
        redis=fake_redis,
    )
    await warm.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    n_after_warm = mini_graph.call_count("fetch_outgoing")
    # Fresh client with same Redis instance, empty L1
    cold = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000,
        redis=fake_redis,
    )
    await cold.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert mini_graph.call_count("fetch_outgoing") == n_after_warm, (
        "cold L1 must read from Redis, not refetch from backend"
    )


async def test_redis_ttl_set_to_configured_seconds(mini_graph, fake_redis):
    """Redis writes carry the configured TTL (default 24h = 86400s)."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000,
        redis=fake_redis, redis_ttl_seconds=86400,
    )
    await c.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    keys = list(fake_redis.scan_iter())
    assert keys, "expected at least one Redis key"
    ttls = [fake_redis.ttl(k) for k in keys]
    # TTLs should be set (positive) and ≤ configured value
    assert all(0 < t <= 86400 for t in ttls), f"unexpected TTLs: {ttls}"


async def test_redis_unavailable_falls_back_to_l1_only(mini_graph):
    """A broken Redis must not break the client — L1 keeps working."""
    from langgraph_coe.tools.wikidata_client import WikidataClient

    class BrokenRedis:
        def __getattr__(self, name):
            def raiser(*a, **kw):
                raise ConnectionError(f"Redis.{name} unavailable")
            return raiser

    c = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000,
        redis=BrokenRedis(),
    )
    result = await c.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert result, "client must survive Redis failure and serve from backend"
    # Second call must still benefit from L1 cache
    n_after_first = mini_graph.call_count("fetch_outgoing")
    await c.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert mini_graph.call_count("fetch_outgoing") == n_after_first


async def test_redis_disabled_when_not_provided(client, mini_graph):
    """Without ``redis=``, only L1 is active — no Redis interaction expected."""
    # Sanity: a normal call works; nothing to inspect for Redis side-effects.
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    # Repeated call serves from L1 only
    n_after = mini_graph.call_count("fetch_outgoing")
    await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert mini_graph.call_count("fetch_outgoing") == n_after
