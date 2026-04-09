"""Integration tests for FreebaseClient against the live Freebase server."""

from __future__ import annotations

from pathlib import Path

import pytest

from wemg.retrieval.freebase_client import FreebaseClient, SKIP_PREFIXES
from wemg.retrieval.wikidata import WikidataEntity, WikiTriple


DEFAULT_FB_SPARQL_URL = "http://n0387:3001/sparql"
KNOWN_MID = "m.0282x"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PATH = REPO_ROOT / "data" / "freebase" / "qid_to_mid.json"


pytest.importorskip("SPARQLWrapper")
pytestmark = [pytest.mark.requires_freebase, pytest.mark.integration]


def _entity_rows(entities: list[WikidataEntity]) -> list[tuple[str | None, str | None]]:
    return [(e.qid, e.label) for e in entities]


def _triple_strings(triples: list[WikiTriple], limit: int = 5) -> list[str]:
    return [str(t) for t in triples[:limit]]


def _is_mid(value: str | None) -> bool:
    return isinstance(value, str) and (value.startswith("m.") or value.startswith("g.")) and len(value) > 2


def _make_live_client(**kwargs) -> FreebaseClient:
    client = FreebaseClient(sparql_url=DEFAULT_FB_SPARQL_URL, **kwargs)
    probe = client._sparql_query("SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
    if not probe:
        pytest.skip(f"Freebase endpoint unavailable or empty: {DEFAULT_FB_SPARQL_URL}")
    return client


def test_default_sparql_url_matches_config_default():
    client = FreebaseClient()
    assert client._sparql_url == DEFAULT_FB_SPARQL_URL


def test_lookup_by_mid_normalizes_slash_and_returns_live_label():
    client = _make_live_client()
    out = client._lookup_by_mid("/m/0282x")
    print("lookup_by_mid:", _entity_rows(out))

    assert len(out) == 1
    assert out[0].qid == KNOWN_MID
    assert out[0].label is not None
    assert isinstance(out[0].label, str)
    assert out[0].label.strip() != ""
    assert out[0].label != out[0].qid


def test_search_entities_is_qids_path_hits_live_server():
    client = _make_live_client()
    out = client.search_entities("/m/0282x", num_results=1, is_qids=True)
    print("search_entities(is_qids=True):", _entity_rows(out))

    assert len(out) == 1
    assert out[0].qid == KNOWN_MID
    assert out[0].label is not None
    assert out[0].label.strip() != ""


def test_search_entities_text_without_mapping_raises():
    client = _make_live_client()
    with pytest.raises(RuntimeError, match="requires wikidata_client \+ qid_to_mid_map_path"):
        client.search_entities("Douglas Adams", num_results=3)


def test_search_entities_batch_text_without_mapping_raises():
    client = _make_live_client()
    queries = ["Douglas Adams", "Berlin"]
    with pytest.raises(RuntimeError, match="requires wikidata_client \+ qid_to_mid_map_path"):
        client.search_entities(queries, num_results=2)


def test_search_entities_is_qids_batch_normalizes_and_returns_live_labels():
    client = _make_live_client()
    mids = ["/m/0282x", "m.0156q"]
    out = client.search_entities(mids, num_results=1, is_qids=True)
    print("search_entities(is_qids batch):", [[(e.qid, e.label) for e in group] for group in out])

    assert isinstance(out, list)
    assert len(out) == len(mids)
    assert out[0][0].qid == KNOWN_MID
    assert out[1][0].qid == "m.0156q"
    assert out[0][0].label and out[0][0].label.strip()
    assert out[1][0].label and out[1][0].label.strip()


def test_get_k_hop_triples_returns_structured_real_triples():
    client = _make_live_client()
    triples = client.get_k_hop_triples(KNOWN_MID, k=1)
    print("get_k_hop_triples sample:\n" + "\n---\n".join(_triple_strings(triples)))

    assert isinstance(triples, list)
    assert len(triples) >= 1
    assert all(isinstance(t, WikiTriple) for t in triples)

    for triple in triples:
        assert triple.subject.qid == KNOWN_MID
        assert triple.relation.pid is not None
        assert isinstance(triple.relation.pid, str)
        assert triple.relation.pid.strip() != ""
        assert triple.relation.label is not None
        assert isinstance(triple.relation.label, str)
        assert triple.relation.label.strip() != ""


def test_get_k_hop_triples_batch_and_cache_lifecycle_real():
    client = _make_live_client()
    mids = [KNOWN_MID, KNOWN_MID]

    batch = client.get_k_hop_triples(mids, k=1)
    print("get_k_hop_triples(batch) sizes:", [len(t) for t in batch])
    assert isinstance(batch, list)
    assert len(batch) == len(mids)
    assert all(isinstance(group, list) for group in batch)
    assert all(len(group) >= 1 for group in batch)

    assert KNOWN_MID in client._triple_cache
    assert len(client._triple_cache) == 1

    client.clear_triple_caches()
    assert client._triple_cache == {}

    single = client.get_k_hop_triples(KNOWN_MID, k=1)
    assert len(single) >= 1
    assert KNOWN_MID in client._triple_cache


def test_get_k_hop_triples_filters_skipped_relation_prefixes():
    client = _make_live_client()
    triples = client.get_k_hop_triples(KNOWN_MID, k=1)
    print("filtered triples sample:\n" + "\n---\n".join(_triple_strings(triples)))

    assert len(triples) >= 1
    for triple in triples:
        for rel_part in triple.relation.pid.split("::"):
            assert not any(rel_part.startswith(prefix) for prefix in SKIP_PREFIXES)


def test_get_k_hop_triples_excludes_non_freebase_predicates():
    client = _make_live_client()
    triples = client.get_k_hop_triples(KNOWN_MID, k=1)
    rels = [t.relation.pid for t in triples]
    print("relation pid sample:", rels[:20])

    assert len(rels) >= 1
    assert all(not rel.startswith("http://") for rel in rels)
    assert all(not rel.startswith("https://") for rel in rels)


def test_enrich_entities_populates_missing_label_from_live_server():
    client = _make_live_client()
    bare = [WikidataEntity(qid=KNOWN_MID, label=None)]
    enriched = client.enrich_entities(bare, get_details=False)
    print("enrich_entities:", _entity_rows(enriched))

    assert len(enriched) == 1
    assert enriched[0].qid == KNOWN_MID
    assert enriched[0].label is not None
    assert enriched[0].label.strip() != ""


@pytest.mark.asyncio
async def test_async_wrappers_match_sync_shape():
    client = _make_live_client()

    sync_entities = client.search_entities("/m/0282x", num_results=1, is_qids=True)
    async_entities = await client.asearch_entities("/m/0282x", num_results=1, is_qids=True)
    print("asearch_entities sync:", _entity_rows(sync_entities))
    print("asearch_entities async:", _entity_rows(async_entities))

    assert isinstance(async_entities, list)
    assert len(async_entities) >= 1
    assert all(isinstance(e, WikidataEntity) for e in async_entities)
    assert len(sync_entities) >= 1
    assert all(_is_mid(e.qid) for e in async_entities)

    sync_triples = client.get_k_hop_triples(KNOWN_MID, k=1)
    async_triples = await client.aget_k_hop_triples(KNOWN_MID, k=1)
    print("aget_k_hop_triples sync sample:\n" + "\n---\n".join(_triple_strings(sync_triples)))
    print("aget_k_hop_triples async sample:\n" + "\n---\n".join(_triple_strings(async_triples)))

    assert len(sync_triples) >= 1
    assert len(async_triples) >= 1
    assert all(isinstance(t, WikiTriple) for t in async_triples)


@pytest.mark.requires_wikidata
def test_wikidata_mapping_path_returns_mids_when_map_present(live_wikidata_client):
    if not DEFAULT_MAP_PATH.is_file():
        pytest.skip(f"Missing QID->MID map file: {DEFAULT_MAP_PATH}")

    client = _make_live_client(
        wikidata_client=live_wikidata_client,
        qid_to_mid_map_path=str(DEFAULT_MAP_PATH),
        qid_to_mid_candidates=8,
    )

    mapped = client._wikidata_search_then_map("Berlin", num_results=5)
    if not mapped:
        pytest.skip("No mapped MID returned from Wikidata search for 'Berlin'")
    print("wikidata_search_then_map('Berlin'):", _entity_rows(mapped))

    assert len(mapped) >= 1
    assert all(isinstance(e, WikidataEntity) for e in mapped)
    assert all(_is_mid(e.qid) for e in mapped)
    assert all(isinstance(e.label, str) and e.label.strip() for e in mapped)


@pytest.mark.requires_wikidata
def test_search_entities_uses_real_wikidata_mapping_via_public_api(live_wikidata_client):
    if not DEFAULT_MAP_PATH.is_file():
        pytest.skip(f"Missing QID->MID map file: {DEFAULT_MAP_PATH}")

    client = _make_live_client(
        wikidata_client=live_wikidata_client,
        qid_to_mid_map_path=str(DEFAULT_MAP_PATH),
        qid_to_mid_candidates=8,
    )

    # Q42 is Douglas Adams; with mapping enabled we should resolve to known MID m.0282x.
    out = client.search_entities("Douglas Adams", num_results=3)
    print("search_entities('Douglas Adams') with real Wikidata mapping:", _entity_rows(out))

    assert isinstance(out, list)
    assert len(out) >= 1
    assert all(isinstance(e, WikidataEntity) for e in out)
    assert all(_is_mid(e.qid) for e in out)
    assert any(e.qid == KNOWN_MID for e in out)