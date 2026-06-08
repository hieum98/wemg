"""Contract tests for rare-but-critical async/concurrency bugs.

The new implementation will be heavily async (LangGraph orchestration runs
many agent tasks in parallel). These tests target failure modes that only
surface under load or unusual scheduling — the same class of bug as
``datasets.map`` blocking the event loop or deadlocking on shared locks.

Failure modes covered:
  - Cache stampede / thundering herd (single-flight)
  - Event-loop blocking by sync libraries (threading.Lock, requests, SPARQLWrapper)
  - asyncio.to_thread wrapping (the messy-wrapper pattern)
  - Non-reentrant Semaphore deadlocks (nested fan-out under concurrency_limit)
  - 429 retry deadlocks (retry while holding slot)
  - Cancellation corruption (orphan locks, half-written cache)
  - Resource leaks (orphan tasks, unbounded lock dict growth)
  - The ``dict O(1) lookup`` lesson (PROPERTY_LABELS short-circuit)
  - Memory and backpressure under load
"""

from __future__ import annotations

import asyncio
import gc
import time

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_GERMANY,
    QID_PARIS,
)
from .contracts import WikidataRateLimitError
from .test_khop_triples import _triple_signature


# ============================================================
# Cache stampede / single-flight
# ============================================================


async def test_concurrent_identical_calls_coalesce_to_single_backend_call(client, mini_graph):
    """N concurrent calls for the same QID → exactly 1 backend fetch.

    Without single-flight, this is the cache-stampede bug: 50 LangGraph
    questions all asking for ``Q30`` (United States) produce 50 SPARQL calls.
    """
    mini_graph.inject_delay("fetch_outgoing", 0.05, times=1)
    results = await asyncio.gather(*[
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
        for _ in range(50)
    ])
    # All coalesced callers receive the same result set
    sigs0 = {_triple_signature(t) for t in results[0]}
    for r in results[1:]:
        assert {_triple_signature(t) for t in r} == sigs0, (
            "result drift across coalesced callers"
        )
    assert mini_graph.call_count("fetch_outgoing") == 1, (
        f"single-flight failed: {mini_graph.call_count('fetch_outgoing')} "
        f"fetches for 50 concurrent callers"
    )


async def test_concurrent_enrich_same_qid_coalesces(client, mini_graph):
    """N concurrent enrich calls for the same QID → 1 backend fetch."""
    mini_graph.inject_delay("get_entity_details", 0.05, times=1)
    results = await asyncio.gather(*[
        client.enrich_entities([QID_BERLIN]) for _ in range(50)
    ])
    assert all(r and r[0].qid == QID_BERLIN for r in results)
    assert mini_graph.call_count("get_entity_details") == 1


async def test_concurrent_link_same_name_coalesces(client, mini_graph):
    """N concurrent link calls for the same name → 1 search + 1 detail call."""
    mini_graph.inject_delay("search_entities_text", 0.05, times=1)
    results = await asyncio.gather(*[
        client.link_entities("Berlin", top_k=1) for _ in range(50)
    ])
    assert all(r and r[0].qid == QID_BERLIN for r in results)
    assert mini_graph.call_count("search_entities_text") == 1


# ============================================================
# Event-loop responsiveness (no sync blocking)
# ============================================================


async def test_slow_fetch_does_not_block_event_loop(client, mini_graph):
    """A slow backend call must yield the loop so other tasks make progress.

    Catches the bug class where the impl uses sync libraries (requests,
    SPARQLWrapper, threading.Lock) that block the entire event loop.
    """
    mini_graph.inject_delay("fetch_outgoing", 0.2, times=1)

    async def heartbeat() -> int:
        beats = 0
        for _ in range(10):
            await asyncio.sleep(0.02)
            beats += 1
        return beats

    triples, beats = await asyncio.gather(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False),
        heartbeat(),
    )
    assert triples
    assert beats == 10, f"event loop blocked: only {beats}/10 heartbeats"


async def test_client_does_not_use_asyncio_to_thread(mini_graph, monkeypatch):
    """Forbid the ``asyncio.to_thread`` fallback pattern.

    The current "messy wrapper" wraps sync functions with ``asyncio.to_thread``.
    Under load this exhausts the default thread pool and ContextVars don't
    propagate cleanly. The native-async rewrite must talk to async backends
    directly.
    """
    from langgraph_coe.tools import wikidata_client as mod

    calls: list = []
    original = mod.asyncio.to_thread

    async def tracking(*args, **kwargs):
        calls.append((args, kwargs))
        return await original(*args, **kwargs)

    monkeypatch.setattr(mod.asyncio, "to_thread", tracking)

    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000
    )
    await c.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=True)
    await c.enrich_entities([QID_BERLIN], get_details=True)
    await c.link_entities("Berlin", top_k=1)
    await c.search_properties("capital", top_k=1)

    assert len(calls) == 0, (
        f"client used asyncio.to_thread {len(calls)}x — native async required"
    )


# ============================================================
# Deadlock prevention
# ============================================================


async def test_nested_concurrency_does_not_deadlock_with_limit_1(mini_graph):
    """Inner fan-out while holding the outer concurrency slot must not deadlock.

    asyncio.Semaphore is NOT reentrant. If the impl acquires a slot in
    ``get_k_hop_triples`` and then awaits an enrichment that also acquires
    the same semaphore, the second acquire blocks forever at limit=1.
    """
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
        concurrency_limit=1,
    )
    triples = await asyncio.wait_for(
        c.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=True, enrich=True),
        timeout=3.0,
    )
    assert triples


async def test_429_retry_does_not_deadlock_under_concurrency_pressure(mini_graph):
    """A 429 retry must release its concurrency slot during the wait, not hold it.

    With concurrency_limit=1, if task A holds the slot, sleeps for Retry-After,
    and then re-acquires, a parallel task B is starved for the full retry
    duration. Worse: if the retry re-enters the same code path while still
    holding the slot, it deadlocks.
    """
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
        concurrency_limit=1,
    )
    mini_graph.inject_error(
        "fetch_outgoing", WikidataRateLimitError(retry_after=0.05), times=1,
    )
    r1, r2 = await asyncio.wait_for(
        asyncio.gather(
            c.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False),
            c.get_k_hop_triples(QID_GERMANY, k=1, bidirectional=False, enrich=False),
        ),
        timeout=3.0,
    )
    assert r1 and r2


# ============================================================
# Cancellation safety
# ============================================================


async def test_cancellation_does_not_corrupt_cache_or_lock_state(client, mini_graph):
    """Cancel mid-fetch; subsequent call must still succeed (no orphan locks)."""
    mini_graph.inject_delay("fetch_outgoing", 0.5, times=1)
    task = asyncio.create_task(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
    )
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The next call must succeed quickly — no stuck single-flight lock
    result = await asyncio.wait_for(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False),
        timeout=2.0,
    )
    assert result


async def test_cancellation_during_coalesce_does_not_starve_other_waiters(client, mini_graph):
    """If the leader of a single-flight group is cancelled, waiters must still complete."""
    mini_graph.inject_delay("fetch_outgoing", 0.3, times=1)
    leader = asyncio.create_task(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
    )
    await asyncio.sleep(0.02)
    follower = asyncio.create_task(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
    )
    await asyncio.sleep(0.02)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    result = await asyncio.wait_for(follower, timeout=2.0)
    assert result


# ============================================================
# The ``dict O(1) lookup`` lesson — well-known PID short-circuit
# ============================================================


async def test_well_known_pid_served_from_bundled_dict_no_backend_call(mini_graph):
    """Well-known PIDs (in PROPERTY_LABELS) must NOT trigger backend property fetches.

    This is the same lesson as the ``datasets.map → dict O(1) lookup`` fix:
    pre-compute hot paths into a dict so the async hot loop is constant-time
    and never blocks on I/O.
    """
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph, max_sparql_rps=1000, max_wikipedia_rps=1000
    )
    n_before = mini_graph.call_count("get_property_details")
    result = await c.search_properties("P36", top_k=1)
    n_after = mini_graph.call_count("get_property_details")
    assert n_after == n_before, (
        "well-known PID (P36) must be served from PROPERTY_LABELS, not the backend"
    )
    assert result[0].pid == "P36"
    assert result[0].label == "capital"


# ============================================================
# Resource leaks
# ============================================================


async def test_no_task_leak_after_many_calls(client):
    """50 sequential calls must not leave orphan ``asyncio.Task``s behind."""
    initial = {id(t) for t in asyncio.all_tasks() if not t.done()}
    for _ in range(50):
        await client.get_k_hop_triples(
            QID_BERLIN, k=1, bidirectional=False, enrich=False
        )
    gc.collect()
    final = {id(t) for t in asyncio.all_tasks() if not t.done()}
    leaked = final - initial
    # Allow up to 1 (the currently-running test task)
    assert len(leaked) <= 1, f"task leak: {len(leaked)} new uncancelled tasks"


async def test_module_global_entity_cache_concurrent_writes_dont_corrupt(
    client, mini_graph, monkeypatch
):
    """The tool-layer ``entity_cache`` is intentionally shared across questions
    (the design choice in the current code). Concurrent writes must not
    produce torn / cross-contaminated entries."""
    from langgraph_coe.config import WikidataConfig
    from langgraph_coe.tools import wikidata as mod

    monkeypatch.setattr(mod, "_wikidata_client", client, raising=False)
    monkeypatch.setattr(mod, "_wikidata_config", WikidataConfig(), raising=False)
    mod.entity_cache.clear()
    mod.reset_wikidata_session()

    names = [f"NAME_{i}" for i in range(50)]
    for n in names:
        mini_graph.add_entity(f"QFRESH_{n}", label=n, search_terms=(n,))

    await asyncio.gather(*[
        mod.link_entities.ainvoke({"entity_names": [n]}) for n in names
    ])

    for n in names:
        assert mod.entity_cache.get(n) == f"QFRESH_{n}", (
            f"entity_cache[{n!r}] = {mod.entity_cache.get(n)!r}, "
            f"expected QFRESH_{n} — torn write under concurrent linkers"
        )


# ============================================================
# Memory / backpressure under load
# ============================================================


async def test_thousand_concurrent_calls_complete_in_bounded_time(client, mini_graph):
    """1000 concurrent calls for the same key complete in bounded time.

    With proper single-flight + non-blocking async, this should be O(1) backend
    calls and O(1) wall clock — independent of N callers.
    """
    mini_graph.inject_delay("fetch_outgoing", 0.01, times=1)
    t0 = time.monotonic()
    results = await asyncio.wait_for(
        asyncio.gather(*[
            client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False)
            for _ in range(1000)
        ]),
        timeout=10.0,
    )
    elapsed = time.monotonic() - t0
    assert len(results) == 1000
    assert elapsed < 5.0, (
        f"1000 coalesced calls took {elapsed:.2f}s — single-flight likely missing"
    )


async def test_high_concurrency_with_diverse_keys_completes(client, mini_graph):
    """500 distinct concurrent seeds complete without hang or memory blowup."""
    qids = [f"Q_DIVERSE_{i}" for i in range(500)]
    for q in qids:
        mini_graph.add_entity(q, label=q)
        mini_graph.add_triple(q, "P31", "QCLS")
    results = await asyncio.wait_for(
        asyncio.gather(*[
            client.get_k_hop_triples(q, k=1, bidirectional=False, enrich=False)
            for q in qids
        ]),
        timeout=10.0,
    )
    assert len(results) == 500
    assert all(r for r in results), "some distinct-key fetches returned empty"
