"""Unit tests for batched k-hop Wikidata retrieval (no live endpoint required)."""

import pytest

pytest.importorskip("SPARQLWrapper")

from wemg.retrieval.wikidata import WikidataClient


def test_multi_seed_k1_uses_two_batch_sparql_queries_when_bidirectional(monkeypatch):
    client = WikidataClient()
    calls: list[str] = []

    def fake_sparql(sparql: str):
        calls.append(sparql)
        return []

    monkeypatch.setattr(client, "_sparql_query", fake_sparql)
    client.get_k_hop_triples(["Q64", "Q183"], k=1, bidirectional=True, enrich=False)
    assert len(calls) == 2
    assert all("VALUES ?seed" in c for c in calls)


def test_multi_seed_k1_uses_one_batch_sparql_query_when_unidirectional(monkeypatch):
    client = WikidataClient()
    calls: list[str] = []

    def fake_sparql(sparql: str):
        calls.append(sparql)
        return []

    monkeypatch.setattr(client, "_sparql_query", fake_sparql)
    client.get_k_hop_triples(["Q64", "Q183"], k=1, bidirectional=False, enrich=False)
    assert len(calls) == 1
    assert "VALUES ?seed" in calls[0]


def test_bidirectional_triples_cache_avoids_repeat_sparql(monkeypatch):
    client = WikidataClient()
    calls: list[str] = []

    def fake_sparql(sparql: str):
        calls.append(sparql)
        return []

    monkeypatch.setattr(client, "_sparql_query", fake_sparql)
    client._get_bidirectional_triples("Q64")
    n_after_first = len(calls)
    client._get_bidirectional_triples("Q64")
    assert len(calls) == n_after_first


def test_outgoing_cache_populated_by_bidirectional(monkeypatch):
    client = WikidataClient()
    calls: list[str] = []

    def fake_sparql(sparql: str):
        calls.append(sparql)
        return []

    monkeypatch.setattr(client, "_sparql_query", fake_sparql)
    client._get_bidirectional_triples("Q64")
    n_after_bidir = len(calls)
    client._get_outgoing_triples("Q64")
    assert len(calls) == n_after_bidir
