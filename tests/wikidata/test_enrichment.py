"""Contract tests for ``WikidataClient.enrich_entities``.

Target signature::

    async def enrich_entities(
        self, qids: str | list[str], *, get_details: bool = False
    ) -> list[WikidataEntity]
"""

from __future__ import annotations

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_GERMANY,
    QID_PARIS,
)


# ---------------- correctness ----------------


async def test_enrich_fills_label_description_aliases(client):
    """Bare qids come back with label, description, and aliases populated."""
    result = await client.enrich_entities([QID_BERLIN, QID_GERMANY])
    by_qid = {e.qid: e for e in result}
    assert by_qid[QID_BERLIN].label == "Berlin"
    assert by_qid[QID_BERLIN].description and "capital" in by_qid[QID_BERLIN].description.lower()
    assert by_qid[QID_GERMANY].label == "Germany"
    assert "Deutschland" in (by_qid[QID_GERMANY].aliases or [])


async def test_enrich_get_details_false_skips_wikipedia(client, mini_graph):
    """With get_details=False, Wikipedia must not be fetched."""
    await client.enrich_entities([QID_BERLIN], get_details=False)
    assert mini_graph.call_count("get_wikipedia_contents") == 0


async def test_enrich_get_details_true_populates_wikipedia_content(client):
    """With get_details=True, wikipedia_content is filled when available."""
    result = await client.enrich_entities([QID_BERLIN], get_details=True)
    assert len(result) == 1
    e = result[0]
    assert e.wikipedia_url is not None
    assert e.wikipedia_content is not None
    assert "Berlin" in e.wikipedia_content


async def test_enrich_does_not_refetch_after_first_call(client, mini_graph):
    """Repeated enrichment of the same qid must not re-hit the backend (cache)."""
    await client.enrich_entities([QID_BERLIN])
    n_after_first = mini_graph.call_count("get_entity_details")
    await client.enrich_entities([QID_BERLIN])
    assert mini_graph.call_count("get_entity_details") == n_after_first, (
        "second enrich for the same qid must hit the cache"
    )


async def test_enrich_preserves_input_order(client):
    """Output is ordered to match the input qid order."""
    qids = [QID_PARIS, QID_BERLIN, QID_GERMANY]
    result = await client.enrich_entities(qids)
    assert [e.qid for e in result] == qids


# ---------------- batching parity ----------------


async def test_enrich_single_get_entity_details_call_for_n_qids(client, mini_graph):
    """N qids must be enriched in a single batched backend call."""
    qids = [QID_BERLIN, QID_GERMANY, QID_PARIS]
    await client.enrich_entities(qids)
    detail_calls = mini_graph.calls("get_entity_details")
    assert len(detail_calls) == 1
    assert set(detail_calls[0].args[0]) == set(qids)


async def test_enrich_single_get_wikipedia_contents_call_when_details_true(client, mini_graph):
    """Wikipedia content must be fetched in a single batched call."""
    qids = [QID_BERLIN, QID_GERMANY, QID_PARIS]
    await client.enrich_entities(qids, get_details=True)
    wiki_calls = mini_graph.calls("get_wikipedia_contents")
    assert len(wiki_calls) == 1, f"expected 1 batched call, got {len(wiki_calls)}"
    titles = set(wiki_calls[0].args[0])
    assert {"Berlin", "Germany", "Paris"} <= titles


# ---------------- error handling ----------------


async def test_enrich_missing_wikipedia_title_skips_wiki_fetch(client, mini_graph):
    """Entities without a Wikipedia title must not be requested from Wikipedia."""
    mini_graph.add_entity("QNOWIKI", label="NoWiki", description="no wikipedia title")
    await client.enrich_entities(["QNOWIKI"], get_details=True)
    wiki_calls = mini_graph.calls("get_wikipedia_contents")
    if wiki_calls:
        titles = wiki_calls[0].args[0]
        assert "NoWiki" not in titles
        assert all(t for t in titles), "no empty titles requested"


async def test_enrich_wikipedia_returns_none_does_not_raise(client, mini_graph):
    """If Wikipedia returns None for a title, enrichment still succeeds."""
    mini_graph.add_entity(
        "QGHOST", label="Ghost", wikipedia_title="GhostMissingArticle"
    )
    # No add_wikipedia call → backend returns None for "GhostMissingArticle"
    result = await client.enrich_entities(["QGHOST"], get_details=True)
    assert len(result) == 1
    assert result[0].label == "Ghost"
    assert result[0].wikipedia_content is None


async def test_enrich_partial_backend_response_yields_partial_results(client, mini_graph):
    """Qids missing from the backend response come back as bare stubs, no raise."""
    result = await client.enrich_entities([QID_BERLIN, "QDOESNOTEXIST"])
    assert len(result) == 2
    by_qid = {e.qid: e for e in result}
    assert by_qid[QID_BERLIN].label == "Berlin"
    ghost = by_qid["QDOESNOTEXIST"]
    assert ghost.qid == "QDOESNOTEXIST"
    assert ghost.label is None
