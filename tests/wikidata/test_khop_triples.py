"""Contract tests for ``WikidataClient.get_k_hop_triples``.

Target signature::

    async def get_k_hop_triples(
        self,
        qids: str | list[str],
        *,
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> list[WikiTriple] | list[list[WikiTriple]]
"""

from __future__ import annotations

import pytest

from ._fixtures import (
    PID_CAPITAL,
    PID_CAPITAL_OF,
    PID_CONTINENT,
    PID_COUNTRY,
    PID_HEAD_OF_GOV,
    PID_INSTANCE_OF,
    QID_BERLIN,
    QID_BRANDENBURG,
    QID_EUROPE,
    QID_FRANCE,
    QID_GERMANY,
    QID_HAMBURG,
    QID_MERKEL,
    QID_PARIS,
)


def _triple_signature(t):
    """(subj_qid, rel_pid, obj_qid_or_str) for set comparisons."""
    subj = t.subject.qid if hasattr(t.subject, "qid") else str(t.subject)
    rel = t.relation.pid if hasattr(t.relation, "pid") else str(t.relation)
    obj = t.object.qid if hasattr(t.object, "qid") else str(t.object)
    return (subj, rel, obj)


# ---------------- correctness ----------------


async def test_k1_forward_returns_outgoing_only(client):
    """bidirectional=False → only outgoing edges from the seed."""
    triples = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    sigs = {_triple_signature(t) for t in triples}
    # All outgoing edges of Berlin must be present
    assert (QID_BERLIN, PID_CAPITAL_OF, QID_GERMANY) in sigs
    assert (QID_BERLIN, PID_COUNTRY, QID_GERMANY) in sigs
    # No incoming edge (Germany --capital--> Berlin) should be in forward-only
    assert all(s == QID_BERLIN for s, _, _ in sigs), \
        "forward-only must contain only triples where subject == seed"


async def test_k1_bidirectional_includes_both_directions(client):
    """bidirectional=True → outgoing AND incoming edges of the seed."""
    triples = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=True, enrich=False
    )
    sigs = {_triple_signature(t) for t in triples}
    # Outgoing
    assert (QID_BERLIN, PID_CAPITAL_OF, QID_GERMANY) in sigs
    # Incoming (Germany --capital--> Berlin)
    assert (QID_GERMANY, PID_CAPITAL, QID_BERLIN) in sigs


async def test_k1_enrich_true_attaches_full_entity_property_details(client):
    """enrich=True → subjects/objects are full WikidataEntity, relations have labels."""
    triples = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=True
    )
    assert triples
    t = next(
        x for x in triples
        if _triple_signature(x) == (QID_BERLIN, PID_CAPITAL_OF, QID_GERMANY)
    )
    assert t.subject.label == "Berlin"
    assert t.relation.label == "capital of"
    assert t.object.label == "Germany"


async def test_k1_enrich_false_returns_qid_only_stubs(client, mini_graph):
    """enrich=False → entity/property objects carry only qid/pid, no labels fetched."""
    before_details = mini_graph.call_count("get_entity_details")
    before_props = mini_graph.call_count("get_property_details")
    triples = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    after_details = mini_graph.call_count("get_entity_details")
    after_props = mini_graph.call_count("get_property_details")
    assert triples
    # No enrichment-related backend calls should be made
    assert after_details == before_details, "enrich=False must not call get_entity_details"
    assert after_props == before_props, "enrich=False must not call get_property_details"


async def test_k2_traverses_frontier_seed_to_a_to_b(client):
    """k=2 returns triples from seed→A AND A→B."""
    triples = await client.get_k_hop_triples(
        QID_BERLIN, k=2, bidirectional=False, enrich=False
    )
    sigs = {_triple_signature(t) for t in triples}
    # Hop 1: Berlin -> Germany
    assert (QID_BERLIN, PID_CAPITAL_OF, QID_GERMANY) in sigs
    # Hop 2: Germany -> Europe (via continent)
    assert (QID_GERMANY, PID_CONTINENT, QID_EUROPE) in sigs


async def test_k2_dedup_prevents_loops(client, mini_graph):
    """Cycle Q1↔Q2 must terminate with bounded triple count, not infinite."""
    QA, QB = "QLOOPA", "QLOOPB"
    mini_graph.add_entity(QA, label="A")
    mini_graph.add_entity(QB, label="B")
    mini_graph.add_triple(QA, "P31", QB)
    mini_graph.add_triple(QB, "P31", QA)
    triples = await client.get_k_hop_triples(
        QA, k=5, bidirectional=True, enrich=False
    )
    sigs = {_triple_signature(t) for t in triples}
    # The cycle has exactly 2 unique directed edges
    assert sigs <= {(QA, "P31", QB), (QB, "P31", QA)}


async def test_per_seed_partitioning_preserves_seed_attribution(client):
    """List input → per-seed results, each containing only its own-reachable triples."""
    result = await client.get_k_hop_triples(
        [QID_BERLIN, QID_PARIS], k=1, bidirectional=False, enrich=False
    )
    assert isinstance(result, list) and len(result) == 2
    berlin_sigs = {_triple_signature(t) for t in result[0]}
    paris_sigs = {_triple_signature(t) for t in result[1]}
    # Berlin's bucket has Berlin-rooted triples; Paris's has Paris-rooted
    assert all(s == QID_BERLIN for s, _, _ in berlin_sigs)
    assert all(s == QID_PARIS for s, _, _ in paris_sigs)
    # And the two buckets are disjoint
    assert berlin_sigs.isdisjoint(paris_sigs)


# ---------------- batching parity ----------------


async def test_multi_seed_k1_forward_single_fetch_outgoing_call(client, mini_graph):
    """Multi-seed k=1 forward → exactly one batched fetch_outgoing."""
    await client.get_k_hop_triples(
        [QID_BERLIN, QID_GERMANY, QID_PARIS],
        k=1, bidirectional=False, enrich=False,
    )
    out_calls = mini_graph.calls("fetch_outgoing")
    in_calls = mini_graph.calls("fetch_incoming")
    assert len(out_calls) == 1
    assert set(out_calls[0].args[0]) == {QID_BERLIN, QID_GERMANY, QID_PARIS}
    assert len(in_calls) == 0


async def test_multi_seed_k1_bidirectional_one_outgoing_one_incoming(client, mini_graph):
    """Multi-seed k=1 bidirectional → exactly one outgoing + one incoming batched call."""
    await client.get_k_hop_triples(
        [QID_BERLIN, QID_PARIS],
        k=1, bidirectional=True, enrich=False,
    )
    assert mini_graph.call_count("fetch_outgoing") == 1
    assert mini_graph.call_count("fetch_incoming") == 1


async def test_frontier_capped_at_max_entities_per_hop(client, mini_graph):
    """Large fanout at hop 1 → hop-2 frontier is bounded (not all 600 fanned out)."""
    Q_ROOT = "QROOT"
    fan_out = 600
    mini_graph.add_entity(Q_ROOT, label="Root")
    for i in range(fan_out):
        oid = f"Q{1_000_000 + i}"
        mini_graph.add_entity(oid, label=f"Obj{i}")
        mini_graph.add_triple(Q_ROOT, "P31", oid)
    await client.get_k_hop_triples(
        Q_ROOT, k=2, bidirectional=False, enrich=False
    )
    out_calls = mini_graph.calls("fetch_outgoing")
    assert len(out_calls) >= 2
    second_hop_qids = out_calls[1].args[0]
    assert len(second_hop_qids) <= 500, (
        f"hop-2 frontier should be capped at MAX_ENTITIES_PER_HOP=500, "
        f"got {len(second_hop_qids)}"
    )


# ---------------- error handling ----------------


async def test_invalid_qid_skipped_not_raised(client):
    """An unknown qid in a batch is silently skipped — no raise."""
    result = await client.get_k_hop_triples(
        [QID_BERLIN, "QNOSUCHENTITY"],
        k=1, bidirectional=False, enrich=False,
    )
    assert isinstance(result, list) and len(result) == 2
    # Berlin still produces triples; bogus qid produces an empty bucket
    assert len(result[0]) >= 1
    assert len(result[1]) == 0


async def test_empty_seeds_returns_empty_list(client):
    """Empty input → empty output, no backend call."""
    result = await client.get_k_hop_triples(
        [], k=1, bidirectional=False, enrich=False
    )
    assert result == []
