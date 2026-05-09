"""Unit tests for wbgetentities / Action API entity batch path (no live WDQS for core cases)."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("SPARQLWrapper")

from wemg.retrieval.wikidata import WikidataClient, WikidataEntity


def test_entity_from_wb_payload_parses_labels_sitelinks():
    payload = {
        "type": "item",
        "id": "Q64",
        "labels": {"en": {"language": "en", "value": "Berlin"}},
        "descriptions": {"en": {"language": "en", "value": "Capital of Germany"}},
        "aliases": {"en": [{"language": "en", "value": "Berlin DE"}]},
        "sitelinks": {
            "enwiki": {
                "site": "enwiki",
                "title": "Berlin",
                "url": "https://en.wikipedia.org/wiki/Berlin",
            }
        },
    }
    ent = WikidataClient._entity_from_wb_payload(payload)
    assert ent is not None
    assert ent.qid == "Q64"
    assert ent.label == "Berlin"
    assert ent.description == "Capital of Germany"
    assert "Berlin DE" in ent.aliases
    assert ent.wikipedia_url == "https://en.wikipedia.org/wiki/Berlin"
    assert ent.url == "https://www.wikidata.org/wiki/Q64"


def test_entity_from_wb_payload_sitelink_title_without_url():
    """wbgetentities commonly returns enwiki with title but no url field."""
    payload = {
        "type": "item",
        "id": "Q64",
        "labels": {"en": {"language": "en", "value": "Berlin"}},
        "sitelinks": {
            "enwiki": {"site": "enwiki", "title": "Berlin"},
        },
    }
    ent = WikidataClient._entity_from_wb_payload(payload)
    assert ent is not None
    assert ent.wikipedia_url == "https://en.wikipedia.org/wiki/Berlin"


def test_resolve_wb_entity_payload_follows_redirect():
    entities = {
        "Q395": {"type": "item", "id": "Q395", "redirect": "Q394"},
        "Q394": {
            "type": "item",
            "id": "Q394",
            "labels": {"en": {"language": "en", "value": "Canonical"}},
        },
    }
    leaf = WikidataClient._resolve_wb_entity_payload("Q395", entities)
    assert leaf is not None
    assert leaf["id"] == "Q394"


def test_get_entities_batch_uses_action_api_when_full_coverage(monkeypatch):
    client = WikidataClient()

    def fake_action(qids):
        return {
            q: WikidataEntity(
                qid=q,
                label=f"L-{q}",
                url=f"https://www.wikidata.org/wiki/{q}",
            )
            for q in qids
        }

    def no_sparql(_qids):
        raise AssertionError("SPARQL fallback should not run when Action API covers all QIDs")

    monkeypatch.setattr(client, "_get_entities_batch_via_action_api", fake_action)
    monkeypatch.setattr(client, "_get_entities_batch_via_sparql", no_sparql)

    out = client._get_entities_batch(["Q1", "Q2"], get_details=False)
    assert set(out.keys()) == {"Q1", "Q2"}
    assert out["Q1"].label == "L-Q1"


def test_get_entities_batch_falls_back_to_sparql_for_missing(monkeypatch):
    client = WikidataClient()

    monkeypatch.setattr(
        client,
        "_get_entities_batch_via_action_api",
        lambda qids: {"Q1": WikidataEntity(qid="Q1", label="One", url="https://www.wikidata.org/wiki/Q1")},
    )

    def fake_sparql(qids):
        return {
            "Q2": WikidataEntity(qid="Q2", label="Two", url="https://www.wikidata.org/wiki/Q2"),
        }

    monkeypatch.setattr(client, "_get_entities_batch_via_sparql", fake_sparql)

    out = client._get_entities_batch(["Q1", "Q2"], get_details=False)
    assert out["Q1"].label == "One"
    assert out["Q2"].label == "Two"


def test_get_entities_batch_sparql_only_when_flag_off(monkeypatch):
    client = WikidataClient(use_action_api_for_entities=False)
    monkeypatch.setattr(
        client,
        "_get_entities_batch_via_action_api",
        lambda qids: (_ for _ in ()).throw(AssertionError("action API should not be called")),
    )
    monkeypatch.setattr(
        client,
        "_get_entities_batch_via_sparql",
        lambda qids: {
            "Q9": WikidataEntity(qid="Q9", label="Nine", url="https://www.wikidata.org/wiki/Q9"),
        },
    )
    out = client._get_entities_batch(["Q9"], get_details=False)
    assert out["Q9"].label == "Nine"


def test_wbsearchentities_parses_search_response(monkeypatch):
    client = WikidataClient()
    client.clear_action_api_caches()

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "search": [
                    {"id": "Q64", "label": "Berlin", "description": "federated state"},
                    {"id": "bad"},
                ],
                "success": 1,
            }

    sess = MagicMock()

    def fake_get(_url, params=None, timeout=None):
        assert params["action"] == "wbsearchentities"
        assert params["type"] == "item"
        assert params["search"] == "Berlin"
        return FakeResp()

    sess.get = fake_get
    monkeypatch.setattr(client, "_get_wikidata_action_session", lambda: sess)
    out = client._wbsearchentities_search_items("Berlin", 3)
    assert len(out) == 1
    assert out[0].qid == "Q64"
    assert out[0].label == "Berlin"
    assert out[0].description == "federated state"


def test_search_entity_by_text_skips_sparql_when_wbsearch_succeeds(monkeypatch):
    client = WikidataClient()
    monkeypatch.setattr(client, "_rest_v1_search_entities", lambda *a, **k: [])
    monkeypatch.setattr(
        client,
        "_wbsearchentities_search_items",
        lambda text, n: [WikidataEntity(qid="Q1", url="https://www.wikidata.org/wiki/Q1")],
    )
    sparql_calls: list = []
    monkeypatch.setattr(client, "_sparql_query", lambda sparql: sparql_calls.append(sparql) or [])
    out = client._search_entity_by_text("anything", num_results=1)
    assert len(out) == 1 and out[0].qid == "Q1"
    assert sparql_calls == []


def test_wbgetentities_memory_cache_second_batch_skips_http(monkeypatch):
    client = WikidataClient()
    client.clear_action_api_caches()
    calls = {"n": 0}

    def fake_req(qids):
        calls["n"] += 1
        return {
            q: WikidataEntity(qid=q, label=f"L-{q}", url=f"https://www.wikidata.org/wiki/{q}")
            for q in qids
        }

    monkeypatch.setattr(client, "_wbgetentities_request", fake_req)
    client._get_entities_batch_via_action_api(["Q1", "Q2"])
    client._get_entities_batch_via_action_api(["Q1", "Q2"])
    assert calls["n"] == 1


def test_wbgetentities_cache_disabled_fetches_each_time(monkeypatch):
    client = WikidataClient(cache_action_api_responses=False)
    calls = {"n": 0}

    def fake_req(qids):
        calls["n"] += 1
        return {q: WikidataEntity(qid=q, url=f"https://www.wikidata.org/wiki/{q}") for q in qids}

    monkeypatch.setattr(client, "_wbgetentities_request", fake_req)
    client._get_entities_batch_via_action_api(["Q9"])
    client._get_entities_batch_via_action_api(["Q9"])
    assert calls["n"] == 2


def test_wbsearchentities_memory_cache_second_call_skips_http(monkeypatch):
    client = WikidataClient()
    client.clear_action_api_caches()
    calls = {"n": 0}

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"search": [{"id": "Q1", "label": "One"}], "success": 1}

    sess = MagicMock()

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return FakeResp()

    sess.get = fake_get
    monkeypatch.setattr(client, "_get_wikidata_action_session", lambda: sess)
    assert client._wbsearchentities_search_items("unique-cache-test-xyz", 5)[0].qid == "Q1"
    assert client._wbsearchentities_search_items("unique-cache-test-xyz", 5)[0].qid == "Q1"
    assert calls["n"] == 1


def test_entity_metadata_redis_cache_shared_between_clients():
    class FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, k):
            return self.data.get(k)

        def setex(self, k, _ttl, v):
            self.data[k] = v

    r = FakeRedis()
    c1 = WikidataClient(redis_client=r, entity_metadata_cache_max_entries=64)
    c1.clear_action_api_caches()
    c1._entity_metadata_cache_set(
        "Q55",
        WikidataEntity(qid="Q55", label="Portugal", url="https://www.wikidata.org/wiki/Q55"),
    )
    c2 = WikidataClient(redis_client=r, entity_metadata_cache_max_entries=64)
    c2.clear_action_api_caches()
    got = c2._entity_metadata_cache_get("Q55")
    assert got is not None and got.label == "Portugal"


def test_get_entities_batch_no_sparql_when_action_hits(monkeypatch):
    """Phase 5: full batch path uses Action API only (incl. cache) — SPARQL unused."""
    client = WikidataClient()
    client.clear_action_api_caches()

    def fake_action(qids):
        return {
            q: WikidataEntity(qid=q, label="ok", url=f"https://www.wikidata.org/wiki/{q}")
            for q in qids
        }

    sparql_calls: list = []
    monkeypatch.setattr(client, "_get_entities_batch_via_action_api", fake_action)
    monkeypatch.setattr(client, "_get_entities_batch_via_sparql", lambda q: sparql_calls.append(q) or {})
    out = client._get_entities_batch(["Q10"], get_details=False)
    assert out["Q10"].label == "ok"
    assert sparql_calls == []
