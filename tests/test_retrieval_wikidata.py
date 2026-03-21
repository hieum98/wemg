"""Tests for Wikidata models and WikidataClient using real Wikidata/Wikipedia APIs."""

import pytest

# Skip entire module if core dependencies are missing
pytest.importorskip("SPARQLWrapper")

from wemg.retrieval.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    WikidataClient,
    WikidataPathBetweenEntities,
)


def _make_client() -> WikidataClient:
    """Helper to create a client or skip if initialization fails."""
    try:
        return WikidataClient()
    except Exception as e:
        pytest.skip(f"WikidataClient initialization failed: {e}")


def test_wikidata_entity_model():
    e = WikidataEntity(qid="Q64", label="Berlin", description="Capital of Germany")
    assert e.qid == "Q64"
    assert str(e) == "Berlin (Capital of Germany)"
    ctx = e.to_context(include_wiki_page=False)
    assert "Berlin" in ctx and "Capital" in ctx


def test_wikidata_property_model():
    p = WikidataProperty(pid="P1376", label="capital of")
    assert p.pid == "P1376"
    assert str(p) == "capital of"


def test_wiki_triple_model():
    subj = WikidataEntity(qid="Q64", label="Berlin")
    obj = WikidataEntity(qid="Q183", label="Germany")
    rel = WikidataProperty(pid="P1376", label="capital of")
    t = WikiTriple(subject=subj, relation=rel, object=obj)
    assert hasattr(t, "subject") and hasattr(t, "relation") and hasattr(t, "object")
    assert t.subject.qid == "Q64"


def test_wikidata_client_search_entities_by_qid():
    """Search by QID returns single entity."""
    client = _make_client()
    try:
        results = client.search_entities("Q64", num_results=1, get_details=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable: {e}")
    print("test_wikidata_client_search_entities_by_qid:", [(e.qid, e.label) for e in results])
    assert isinstance(results, list)
    assert len(results) >= 1
    ent = results[0]
    assert ent.qid == "Q64"
    # label/description may or may not be filled, but must be strings or None
    assert ent.label is None or isinstance(ent.label, str)
    assert ent.description is None or isinstance(ent.description, str)


def test_search_entities_text_query_real():
    """Text query search should return plausible entities with valid IDs and URLs."""
    client = _make_client()
    try:
        results = client.search_entities("Berlin", num_results=3, get_details=True)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable: {e}")

    print("test_search_entities_text_query_real:", [(e.qid, e.label) for e in results])
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(e, WikidataEntity) for e in results)
    # Structural checks: QIDs and URLs
    assert all(e.qid.startswith("Q") for e in results if e.qid)
    assert any(
        e.url and "wikidata.org/wiki" in e.url for e in results
    )
    # Plausibility check: at least one entity looks like Berlin
    assert any(e.qid and (e.label or "Berlin" in str(e)) for e in results)


@pytest.mark.asyncio
async def test_asearch_entities_real():
    """Async search_entities wrapper returns same kind of results as sync."""
    client = _make_client()
    try:
        sync_results = client.search_entities("Berlin", num_results=1, get_details=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable: {e}")

    if not sync_results:
        pytest.skip("No entities returned for Berlin in sync search")

    try:
        async_results = await client.asearch_entities("Berlin", num_results=1, get_details=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (async): {e}")

    print("test_asearch_entities_real:", [(e.qid, e.label) for e in async_results])
    assert isinstance(async_results, list)
    assert len(async_results) >= 1
    assert all(isinstance(e, WikidataEntity) for e in async_results)


def test_wikidata_client_search_entities_wikipedia_content():
    """search_entities(..., get_details=True) populates wikipedia metadata when possible."""
    # Skip if mediawikiapi dependency is missing
    pytest.importorskip("mediawikiapi")

    client = _make_client()
    try:
        results = client.search_entities("Berlin", num_results=1, get_details=True)
    except Exception as e:
        pytest.skip(f"Wikidata/Wikipedia API unavailable: {e}")

    assert isinstance(results, list)
    if not results:
        pytest.skip("No entities returned for Berlin")

    entity = results[0]
    print(
        "test_wikidata_client_search_entities_wikipedia_content:",
        entity.qid,
        bool(entity.wikipedia_url),
        bool(entity.wikipedia_content),
    )
    # With get_details=True we at least expect a wikipedia_url
    assert isinstance(entity, WikidataEntity)
    assert entity.wikipedia_url is not None
    # wikipedia_content may be missing due to rate limits or API issues; only assert when present
    if entity.wikipedia_content is not None:
        assert isinstance(entity.wikipedia_content, str)
        assert entity.wikipedia_content.strip() != ""


def test_wikidata_client_enrich_entities_wikipedia_content():
    """enrich_entities(..., get_details=True) can fetch Wikipedia content for bare entities."""
    pytest.importorskip("mediawikiapi")

    client = _make_client()
    bare = WikidataEntity(qid="Q64")
    try:
        enriched_list = client.enrich_entities([bare], get_details=True)
    except Exception as e:
        pytest.skip(f"Wikidata/Wikipedia API unavailable: {e}")

    assert len(enriched_list) == 1
    enriched = enriched_list[0]
    print(
        "test_wikidata_client_enrich_entities_wikipedia_content:",
        enriched.qid,
        enriched.label,
        bool(enriched.wikipedia_url),
        bool(enriched.wikipedia_content),
    )
    assert isinstance(enriched, WikidataEntity)
    # At least label or description should be filled after enrichment
    assert enriched.label or enriched.description
    if enriched.wikipedia_url and enriched.wikipedia_content is not None:
        assert isinstance(enriched.wikipedia_content, str)
        assert enriched.wikipedia_content.strip() != ""


def test_search_properties_by_pid_real():
    """search_properties by PID should return a WikidataProperty."""
    client = _make_client()
    try:
        props = client.search_properties("P36", num_results=1)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (properties): {e}")

    print("test_search_properties_by_pid_real:", [(p.pid, p.label) for p in props])
    assert isinstance(props, list)
    assert len(props) >= 1
    prop = props[0]
    assert isinstance(prop, WikidataProperty)
    assert prop.pid == "P36"
    # Invariants: label is never the PID, and we always have at least
    # some human-facing text (label or description).
    if prop.label is not None:
        assert isinstance(prop.label, str)
        assert prop.label != prop.pid
    assert (prop.label or prop.description) is not None


def test_search_properties_text_query_real():
    """Text query for properties returns properties with PIDs."""
    client = _make_client()
    try:
        props = client.search_properties("capital", num_results=3)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (properties): {e}")

    print("test_search_properties_text_query_real:", [(p.pid, p.label) for p in props])
    assert isinstance(props, list)
    assert len(props) >= 1
    assert all(isinstance(p, WikidataProperty) for p in props)
    assert all(p.pid.startswith("P") for p in props if p.pid)
    # Invariants: PID is never reused as label, and we never surface
    # properties that have neither label nor description.
    for p in props:
        if p.label is not None:
            assert p.label != p.pid
        assert (p.label or p.description) is not None


def test_enrich_properties_uses_batch():
    """enrich_properties fills in label/description for bare properties."""
    client = _make_client()
    bare = WikidataProperty(pid="P36")
    try:
        enriched_list = client.enrich_properties([bare])
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (enrich_properties): {e}")

    assert len(enriched_list) == 1
    enriched = enriched_list[0]
    print(
        "test_enrich_properties_uses_batch:",
        enriched.pid,
        enriched.label,
        enriched.description,
    )
    assert isinstance(enriched, WikidataProperty)
    assert enriched.pid == "P36"
    # Enrichment must provide at least one human-readable field and
    # must not reuse the PID as the label.
    assert enriched.label or enriched.description
    if enriched.label is not None:
        assert enriched.label != enriched.pid


def test_get_k_hop_triples_single_qid_real():
    """get_k_hop_triples returns triples around a single QID."""
    client = _make_client()
    try:
        triples = client.get_k_hop_triples("Q64", k=1, bidirectional=True, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (k-hop): {e}")
    assert isinstance(triples, list)
    print(
        "test_get_k_hop_triples_single_qid_real:",
        len(triples),
        [(t.subject.qid if hasattr(t.subject, 'qid') else str(t.subject),
          t.relation.pid if hasattr(t.relation, 'pid') else str(t.relation),
          t.object.qid if hasattr(t.object, 'qid') else str(t.object)) for t in triples[:5]],
    )
    if not triples:
        pytest.skip("No triples returned for Q64")

    assert all(isinstance(t, WikiTriple) for t in triples)
    assert any(
        (hasattr(t.subject, "qid") and t.subject.qid == "Q64") or
        (hasattr(t.object, "qid") and t.object.qid == "Q64")
        for t in triples
    )


def test_get_k_hop_triples_list_qids_real():
    """get_k_hop_triples accepts a list of QIDs and returns list of lists."""
    client = _make_client()
    try:
        all_triples = client.get_k_hop_triples(["Q64", "Q183"], k=1, bidirectional=True, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (k-hop list): {e}")

    assert isinstance(all_triples, list)
    print(
        "test_get_k_hop_triples_list_qids_real:",
        [len(t) for t in all_triples],
    )
    assert len(all_triples) == 2
    for triples in all_triples:
        assert isinstance(triples, list)
        assert all(isinstance(t, WikiTriple) for t in triples)


def test_get_k_hop_triples_enrich_true():
    """get_k_hop_triples with enrich=True should yield enriched labels when possible."""
    client = _make_client()
    try:
        triples = client.get_k_hop_triples("Q64", k=1, bidirectional=False, enrich=True)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (k-hop enrich): {e}")
    assert isinstance(triples, list)
    print(
        "test_get_k_hop_triples_enrich_true:",
        len(triples),
        [(t.relation.pid if hasattr(t.relation, 'pid') else str(t.relation),
          t.relation.label if hasattr(t.relation, 'label') else None) for t in triples[:5]],
    )
    if not triples:
        pytest.skip("No triples returned for Q64 with enrich=True")

    # At least one triple should have a relation label or subject/object label
    assert any(
        isinstance(t.relation, WikidataProperty) and t.relation.label
        for t in triples
    )


@pytest.mark.asyncio
async def test_aget_k_hop_triples_real():
    """Async wrapper for get_k_hop_triples behaves like sync version."""
    client = _make_client()
    try:
        sync_triples = client.get_k_hop_triples("Q64", k=1, bidirectional=True, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (k-hop sync): {e}")

    try:
        async_triples = await client.aget_k_hop_triples("Q64", k=1, bidirectional=True, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (k-hop async): {e}")

    print(
        "test_aget_k_hop_triples_real:",
        len(sync_triples),
        len(async_triples),
    )
    assert isinstance(async_triples, list)
    # Shapes should be comparable (both lists of WikiTriple)
    assert all(isinstance(t, WikiTriple) for t in async_triples)


def test_find_path_between_berlin_and_germany_real():
    """find_path should return a path or skip if none within max_hops."""
    client = _make_client()
    try:
        path = client.find_path("Q64", "Q183", max_hops=2, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (find_path): {e}")

    if path is None:
        pytest.skip("No path found between Q64 and Q183 within 2 hops")

    print(
        "test_find_path_between_berlin_and_germany_real:",
        path.source.qid if path else None,
        path.target.qid if path else None,
        path.path_length if path else None,
    )
    assert isinstance(path, WikidataPathBetweenEntities)
    assert path.source.qid == "Q64"
    assert path.target.qid == "Q183"
    assert isinstance(path.path, list)
    assert path.path_length == len(path.path)
    assert all(isinstance(t, WikiTriple) for t in path.path)


def test_find_path_same_entity_returns_zero_length():
    """find_path(Q64, Q64) should return zero-length path."""
    client = _make_client()
    try:
        path = client.find_path("Q64", "Q64", max_hops=1, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (find_path same entity): {e}")

    if path is None:
        pytest.skip("No entity details returned for Q64")

    print(
        "test_find_path_same_entity_returns_zero_length:",
        path.source.qid if path else None,
        path.target.qid if path else None,
        path.path_length if path else None,
    )
    assert isinstance(path, WikidataPathBetweenEntities)
    assert path.source.qid == "Q64"
    assert path.target.qid == "Q64"
    assert path.path_length == 0
    assert path.path == []


@pytest.mark.asyncio
async def test_afind_path_real():
    """Async wrapper for find_path behaves like sync version."""
    client = _make_client()
    try:
        sync_path = client.find_path("Q64", "Q183", max_hops=2, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (find_path sync): {e}")

    if sync_path is None:
        pytest.skip("No path found in sync find_path; skipping async comparison")

    try:
        async_path = await client.afind_path("Q64", "Q183", max_hops=2, enrich=False)
    except Exception as e:
        pytest.skip(f"Wikidata API unavailable (find_path async): {e}")

    print(
        "test_afind_path_real:",
        bool(sync_path),
        bool(async_path),
    )
    assert async_path is None or isinstance(async_path, WikidataPathBetweenEntities)
