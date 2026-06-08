"""Contract tests for async concurrency.

Uses real (not virtual) ``asyncio.sleep`` here because we want backend calls
to actually overlap in event-loop time so the in-flight cap is observable.
"""

from __future__ import annotations

import asyncio

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_BRANDENBURG,
    QID_FRANCE,
    QID_GERMANY,
    QID_HAMBURG,
    QID_PARIS,
    build_mini_graph,
)
from .fake_backend import FakeWikidataBackend
from .test_khop_triples import _triple_signature


async def test_asyncio_gather_link_batch_no_cross_contamination(client, mini_graph):
    """Many concurrent link calls return correct per-input results."""
    inputs = ["Berlin", "Germany", "Paris", "France", "Hamburg", "Brandenburg"]
    expected = {
        "Berlin": QID_BERLIN, "Germany": QID_GERMANY, "Paris": QID_PARIS,
        "France": QID_FRANCE, "Hamburg": QID_HAMBURG, "Brandenburg": QID_BRANDENBURG,
    }
    results = await asyncio.gather(*[
        client.link_entities(name, top_k=1) for name in inputs
    ])
    for name, candidates in zip(inputs, results):
        assert candidates, f"no candidate for {name}"
        assert candidates[0].qid == expected[name], (
            f"{name}: expected {expected[name]}, got {candidates[0].qid}"
        )


async def test_in_flight_backend_calls_capped_by_concurrency_limit():
    """With concurrency_limit=3, no more than 3 backend calls are in-flight at once."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    backend = build_mini_graph(FakeWikidataBackend())
    c = WikidataClient(
        backend=backend,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
        concurrency_limit=3,
    )
    # Make each backend call take ~50ms so they overlap.
    backend.inject_delay("fetch_outgoing", 0.05, times=100)
    seeds = [
        QID_BERLIN, QID_GERMANY, QID_PARIS, QID_FRANCE,
        QID_HAMBURG, QID_BRANDENBURG, "QFAKE1", "QFAKE2",
        "QFAKE3", "QFAKE4",
    ]
    await asyncio.gather(*[
        c.get_k_hop_triples(s, k=1, bidirectional=False, enrich=False)
        for s in seeds
    ])
    peak = backend.max_in_flight.get("fetch_outgoing", 0)
    assert peak <= 3, f"observed max concurrency {peak}, expected ≤ 3"


async def test_gather_aggregate_matches_sequential(client):
    """Concurrent gather of k-hop calls produces the same per-seed results as sequential."""
    seeds = [QID_BERLIN, QID_GERMANY, QID_PARIS, QID_FRANCE]
    gathered = await asyncio.gather(*[
        client.get_k_hop_triples(s, k=1, bidirectional=False, enrich=False)
        for s in seeds
    ])
    sequential = [
        await client.get_k_hop_triples(s, k=1, bidirectional=False, enrich=False)
        for s in seeds
    ]
    g_sigs = [{_triple_signature(t) for t in lst} for lst in gathered]
    s_sigs = [{_triple_signature(t) for t in lst} for lst in sequential]
    assert g_sigs == s_sigs


async def test_concurrent_calls_do_not_leak_per_call_state(client):
    """Distinct concurrent k-hop calls do not produce mixed seed attribution."""
    out_berlin, out_paris = await asyncio.gather(
        client.get_k_hop_triples(QID_BERLIN, k=1, bidirectional=False, enrich=False),
        client.get_k_hop_triples(QID_PARIS, k=1, bidirectional=False, enrich=False),
    )
    berlin_subjects = {_triple_signature(t)[0] for t in out_berlin}
    paris_subjects = {_triple_signature(t)[0] for t in out_paris}
    assert berlin_subjects == {QID_BERLIN}
    assert paris_subjects == {QID_PARIS}
