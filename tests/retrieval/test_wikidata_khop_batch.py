"""Unit tests for batched k-hop Wikidata retrieval (no live endpoint required)."""

import pytest
from wemg.retrieval import wikidata as wikidata_module

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


def test_sparql_query_429_uses_retry_after(monkeypatch):
    client = WikidataClient()

    calls = {"count": 0}
    sleeps: list[float] = []

    class Fake429(Exception):
        def __init__(self):
            super().__init__("HTTP Error 429: Too Many Requests")
            self.code = 429
            self.headers = {"Retry-After": "0.25"}

    class FakeResult:
        def convert(self):
            calls["count"] += 1
            if calls["count"] == 1:
                raise Fake429()
            return {"results": {"bindings": [{"x": {"value": "ok"}}]}}

    class FakeSPARQLWrapper:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def setTimeout(self, timeout: int):
            return None

        def setQuery(self, sparql: str):
            return None

        def setReturnFormat(self, fmt):
            return None

        def addCustomHttpHeader(self, name: str, value: str):
            return None

        def query(self):
            return FakeResult()

    monkeypatch.setattr(wikidata_module, "SPARQLWrapper", FakeSPARQLWrapper)
    monkeypatch.setattr(wikidata_module, "_sparql_rate_limit", lambda _rps: None)
    monkeypatch.setattr(wikidata_module.time, "sleep", lambda s: sleeps.append(s))

    out = client._sparql_query("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert len(out) == 1
    assert calls["count"] == 2
    assert sleeps
    assert sleeps[0] == pytest.approx(0.25, abs=1e-6)


def test_sparql_query_calls_rate_limiter(monkeypatch):
    client = WikidataClient(max_sparql_requests_per_second=1.5)
    seen: list[float] = []

    class FakeResult:
        def convert(self):
            return {"results": {"bindings": []}}

    class FakeSPARQLWrapper:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def setTimeout(self, timeout: int):
            return None

        def setQuery(self, sparql: str):
            return None

        def setReturnFormat(self, fmt):
            return None

        def addCustomHttpHeader(self, name: str, value: str):
            return None

        def query(self):
            return FakeResult()

    monkeypatch.setattr(wikidata_module, "SPARQLWrapper", FakeSPARQLWrapper)
    monkeypatch.setattr(
        wikidata_module,
        "_sparql_rate_limit",
        lambda rps: seen.append(rps),
    )

    client._sparql_query("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    assert seen == [1.5]
