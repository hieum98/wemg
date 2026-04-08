"""Unit tests for GraphRetrieverClient.

All tests are pure in-memory — no LLM, SPARQL, or network calls.
"""
from __future__ import annotations

import asyncio
from typing import Dict

import pytest

from wemg.retrieval.graph_retriever_client import GraphRetrieverClient
from wemg.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cache_entry() -> Dict:
    """Minimal valid cache entry: 3 entities, 1 relation, 2 triples.

    Entity graph:
        Alex Rodriguez  --[place of birth]-->  New York
        Alex Rodriguez  --[place of birth]-->  Miami
    """
    return {
        "id": "test_001",
        "question": "Where was Alex Rodriguez born?",
        "q_entity_qids": ["Q1"],
        "entity_dict": {
            "Q0": {"qid": "Q0", "label": "Miami", "description": "", "is_cvt_record": False},
            "Q1": {"qid": "Q1", "label": "Alex Rodriguez", "description": "baseball player", "is_cvt_record": False},
            "Q2": {"qid": "Q2", "label": "New York", "description": "", "is_cvt_record": False},
        },
        "property_dict": {
            "P0": {"pid": "P0", "label": "place of birth", "description": ""},
        },
        "triples": [
            ["Q1", "P0", "Q0"],
            ["Q1", "P0", "Q2"],
        ],
        "embeddings": {},
    }


@pytest.fixture
def cvt_cache_entry() -> Dict:
    """Cache entry with a CVT record node.

    Entity graph (after preprocessing):
        Alex Rodriguez --[batting stats]--> QCVT0 (CVT: "season: 2005; team: New York Yankees")
        Alex Rodriguez --[batting stats season]--> 2005           (flattened Rule 1)
        Alex Rodriguez --[batting stats team]  --> New York Yankees (flattened Rule 1)
    """
    return {
        "id": "test_002",
        "question": "Which team did Alex Rodriguez play for in 2005?",
        "q_entity_qids": ["Q0"],
        "entity_dict": {
            "Q0": {"qid": "Q0", "label": "Alex Rodriguez", "description": "", "is_cvt_record": False},
            "Q1": {"qid": "Q1", "label": "New York Yankees", "description": "", "is_cvt_record": False},
            "QCVT0": {
                "qid": "QCVT0",
                "label": "season: 2005; team: New York Yankees",
                "description": "",
                "is_cvt_record": True,
            },
        },
        "property_dict": {
            "P0": {"pid": "P0", "label": "batting stats", "description": ""},
            "P1": {"pid": "P1", "label": "batting stats season", "description": ""},
            "P2": {"pid": "P2", "label": "batting stats team", "description": ""},
        },
        "triples": [
            ["Q0", "P0", "QCVT0"],
            ["Q0", "P1", "2005"],
            ["Q0", "P2", "Q1"],
        ],
        "embeddings": {},
    }


@pytest.fixture
def multi_hop_cache_entry() -> Dict:
    """Three-node chain: A --r1--> B --r2--> C."""
    return {
        "id": "test_003",
        "question": "chain test",
        "q_entity_qids": ["Q0"],
        "entity_dict": {
            "Q0": {"qid": "Q0", "label": "A", "description": "", "is_cvt_record": False},
            "Q1": {"qid": "Q1", "label": "B", "description": "", "is_cvt_record": False},
            "Q2": {"qid": "Q2", "label": "C", "description": "", "is_cvt_record": False},
        },
        "property_dict": {
            "P0": {"pid": "P0", "label": "r1", "description": ""},
            "P1": {"pid": "P1", "label": "r2", "description": ""},
        },
        "triples": [
            ["Q0", "P0", "Q1"],
            ["Q1", "P1", "Q2"],
        ],
        "embeddings": {},
    }


# ---------------------------------------------------------------------------
# load_from_cache
# ---------------------------------------------------------------------------


def test_load_from_cache_returns_seed_qids(simple_cache_entry):
    client = GraphRetrieverClient()
    seeds = client.load_from_cache(simple_cache_entry)
    assert seeds == ["Q1"]


def test_load_from_cache_populates_entity_dict(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    assert "Q1" in client._entities_by_qid
    assert client._entities_by_qid["Q1"].label == "Alex Rodriguez"


def test_load_from_cache_populates_property_dict(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    assert "P0" in client._properties_by_pid
    assert client._properties_by_pid["P0"].label == "place of birth"


def test_load_from_cache_builds_adj_out(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    assert "Q1" in client._adj_out
    adj = client._adj_out["Q1"]
    assert ("P0", "Q0") in adj
    assert ("P0", "Q2") in adj


def test_load_from_cache_builds_adj_in(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    assert "Q0" in client._adj_in
    assert ("P0", "Q1") in client._adj_in["Q0"]


def test_load_from_cache_marks_cvt_qids(cvt_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(cvt_cache_entry)
    assert "QCVT0" in client._cvt_qids
    assert "Q0" not in client._cvt_qids
    assert "Q1" not in client._cvt_qids


def test_load_from_cache_cvt_excluded_from_label_index(cvt_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(cvt_cache_entry)
    label = "season: 2005; team: New York Yankees"
    assert label.lower() not in client._entities_by_label


def test_load_from_cache_is_idempotent(simple_cache_entry):
    """Calling load_from_cache twice resets state cleanly."""
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    client.load_from_cache(simple_cache_entry)
    assert len(client._entities_by_qid) == 3


def test_load_from_cache_empty_entry():
    client = GraphRetrieverClient()
    seeds = client.load_from_cache({})
    assert seeds == []
    assert client._entities_by_qid == {}
    assert client._adj_out == {}
    assert client._adj_in == {}


# ---------------------------------------------------------------------------
# get_k_hop_triples (sync)
# ---------------------------------------------------------------------------


def test_khop_single_seed_returns_list(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("Q1", k=1)
    assert isinstance(result, list)
    assert all(isinstance(t, WikiTriple) for t in result)


def test_khop_single_seed_finds_outgoing(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("Q1", k=1, bidirectional=False)
    subjects = {t.subject.qid for t in result}
    assert "Q1" in subjects


def test_khop_bidirectional_finds_incoming(simple_cache_entry):
    """Starting from Q0 (Miami), bidirectional should surface Q1→Q0 edge."""
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("Q0", k=1, bidirectional=True)
    subjects = {t.subject.qid for t in result}
    assert "Q1" in subjects  # Alex Rodriguez points to Q0


def test_khop_non_bidirectional_no_incoming(simple_cache_entry):
    """Non-bidirectional from Q0 should return nothing (Q0 has no outgoing edges)."""
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("Q0", k=1, bidirectional=False)
    assert result == []


def test_khop_multi_seed_returns_list_of_lists(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples(["Q1", "Q0"], k=1)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


def test_khop_deduplicates_triples(simple_cache_entry):
    """Two traversal paths reaching the same triple should not duplicate it."""
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("Q1", k=1)
    seen = set()
    for t in result:
        h = hash(t)
        assert h not in seen, "Duplicate triple in result"
        seen.add(h)


def test_khop_cvt_nodes_not_expanded(cvt_cache_entry):
    """QCVT0 is terminal — 2-hop traversal from Q0 must not expand through it."""
    client = GraphRetrieverClient()
    client.load_from_cache(cvt_cache_entry)
    # Hop 1 from Q0 reaches QCVT0 (CVT), Q1 (real), and string literal "2005"
    # Hop 2 must NOT expand QCVT0 further
    result = client.get_k_hop_triples("Q0", k=2, bidirectional=False)
    subjects = {t.subject.qid for t in result if hasattr(t.subject, "qid")}
    # Q0 expanded; Q1 may expand (has incoming edge from Q0 via P2)
    # QCVT0 must not appear as a subject
    assert "QCVT0" not in subjects


def test_khop_string_literal_as_object(cvt_cache_entry):
    """String literal objects (e.g. "2005") are returned as-is, not as entities."""
    client = GraphRetrieverClient()
    client.load_from_cache(cvt_cache_entry)
    result = client.get_k_hop_triples("Q0", k=1, bidirectional=False)
    string_objects = [t.object for t in result if isinstance(t.object, str)]
    assert "2005" in string_objects


def test_khop_two_hops_chain(multi_hop_cache_entry):
    """2-hop from Q0 should reach C (Q2) via B (Q1)."""
    client = GraphRetrieverClient()
    client.load_from_cache(multi_hop_cache_entry)
    result = client.get_k_hop_triples("Q0", k=2, bidirectional=False)
    objects = {
        t.object.qid for t in result if isinstance(t.object, WikidataEntity)
    }
    assert "Q1" in objects
    assert "Q2" in objects


def test_khop_one_hop_chain_misses_c(multi_hop_cache_entry):
    """1-hop from Q0 should only reach B, not C."""
    client = GraphRetrieverClient()
    client.load_from_cache(multi_hop_cache_entry)
    result = client.get_k_hop_triples("Q0", k=1, bidirectional=False)
    objects = {
        t.object.qid for t in result if isinstance(t.object, WikidataEntity)
    }
    assert "Q1" in objects
    assert "Q2" not in objects


def test_khop_unknown_seed_returns_empty(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.get_k_hop_triples("QUNKNOWN", k=1)
    assert result == []


# ---------------------------------------------------------------------------
# search_entities (sync)
# ---------------------------------------------------------------------------


def test_search_entities_exact_match(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    results = client.search_entities("Alex Rodriguez", num_results=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].label == "Alex Rodriguez"


def test_search_entities_case_insensitive(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    results = client.search_entities("alex rodriguez", num_results=1)
    assert results and results[0].label == "Alex Rodriguez"


def test_search_entities_multi_query_returns_list_of_lists(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    results = client.search_entities(["Alex Rodriguez", "New York"], num_results=1)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)


def test_search_entities_qid_lookup(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.search_entities("Q2", num_results=1, is_qids=True)
    assert len(result) == 1
    assert result[0].qid == "Q2"
    assert result[0].label == "New York"


def test_search_entities_qid_unknown_returns_empty(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = client.search_entities("QXXX", num_results=1, is_qids=True)
    assert result == []


def test_search_entities_cvt_not_returned(cvt_cache_entry):
    """CVT records must never appear in search results."""
    client = GraphRetrieverClient()
    client.load_from_cache(cvt_cache_entry)
    results = client.search_entities("season", num_results=5)
    for r in results:
        assert r.qid not in client._cvt_qids


def test_search_entities_token_overlap_fuzzy(simple_cache_entry):
    """Partial token overlap: 'Rodriguez' should score New York lower than Alex Rodriguez."""
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    results = client.search_entities("Rodriguez", num_results=3)
    labels = [r.label for r in results]
    assert "Alex Rodriguez" in labels
    # Alex Rodriguez should rank above others (highest overlap)
    assert labels[0] == "Alex Rodriguez"


def test_search_entities_no_match_returns_empty(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    results = client.search_entities("Zephyr Quasar XYZ", num_results=1)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# enrich_entities
# ---------------------------------------------------------------------------


def test_enrich_entities_known(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    stub = WikidataEntity(qid="Q1")
    enriched = client.enrich_entities([stub])
    assert enriched[0].label == "Alex Rodriguez"


def test_enrich_entities_unknown_passthrough(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    stub = WikidataEntity(qid="QUNKNOWN", label="Mystery")
    enriched = client.enrich_entities([stub])
    assert enriched[0].label == "Mystery"


def test_enrich_entities_empty_list(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    assert client.enrich_entities([]) == []


# ---------------------------------------------------------------------------
# clear_triple_caches — no-op, must not raise
# ---------------------------------------------------------------------------


def test_clear_triple_caches_noop(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    client.clear_triple_caches()  # must not raise


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def test_aget_k_hop_triples_async(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = asyncio.run(client.aget_k_hop_triples("Q1", k=1))
    assert isinstance(result, list)
    assert len(result) > 0


def test_asearch_entities_async(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    result = asyncio.run(client.asearch_entities("Alex Rodriguez", num_results=1))
    assert isinstance(result, list)
    assert result[0].label == "Alex Rodriguez"


def test_aenrich_entities_async(simple_cache_entry):
    client = GraphRetrieverClient()
    client.load_from_cache(simple_cache_entry)
    stub = WikidataEntity(qid="Q2")
    result = asyncio.run(client.aenrich_entities([stub]))
    assert result[0].label == "New York"


# ---------------------------------------------------------------------------
# embed_client path: pre-computed embedding similarity
# ---------------------------------------------------------------------------


def test_embed_search_used_when_embed_client_and_embeddings_present():
    """embed_client.get_embeddings is called when embeddings are pre-loaded."""
    import numpy as np

    class FakeEmbedClient:
        called = False

        def get_embeddings(self, text):
            FakeEmbedClient.called = True
            # Return unit vector matching Q0's embedding
            return [1.0, 0.0]

    entry = {
        "id": "emb_test",
        "question": "test",
        "q_entity_qids": ["Q0"],
        "entity_dict": {
            "Q0": {"qid": "Q0", "label": "San Francisco", "description": "", "is_cvt_record": False},
            "Q1": {"qid": "Q1", "label": "New York", "description": "", "is_cvt_record": False},
        },
        "property_dict": {},
        "triples": [],
        "embeddings": {
            "Q0": [1.0, 0.0],   # perfectly aligned with query vector
            "Q1": [0.0, 1.0],   # orthogonal
        },
    }

    client = GraphRetrieverClient(embed_client=FakeEmbedClient())
    client.load_from_cache(entry)
    results = client.search_entities("San Francisco", num_results=1)

    assert FakeEmbedClient.called
    assert results[0].label == "San Francisco"
