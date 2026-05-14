"""Contract tests for ``WikidataClient.link_entities``.

Target signature::

    async def link_entities(
        self, names: str | list[str], *, top_k: int = 1
    ) -> list[WikidataEntity] | list[list[WikidataEntity]]
"""

from __future__ import annotations

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_FRANCE,
    QID_GERMANY,
    QID_PARIS,
)


# ---------------- correctness ----------------


async def test_link_single_returns_top_k_qids_ordered(client, mini_graph):
    """Single name input returns up to top_k candidates, ordered by index."""
    mini_graph.add_entity(
        "Q821", label="Berlin",
        description="another Berlin",
        search_terms=("Berlin",),
    )
    mini_graph.add_entity(
        "Q822", label="Berlin",
        description="yet another Berlin",
        search_terms=("Berlin",),
    )
    result = await client.link_entities("Berlin", top_k=3)
    assert isinstance(result, list)
    assert all(hasattr(e, "qid") for e in result)
    qids = [e.qid for e in result]
    assert qids[0] == QID_BERLIN, "primary match (added first) must come first"
    assert "Q821" in qids
    assert len(result) <= 3


async def test_link_batch_returns_per_input_lists(client):
    """List input returns one candidate-list per input name, same order."""
    result = await client.link_entities(["Berlin", "Paris"], top_k=1)
    assert isinstance(result, list) and len(result) == 2
    assert isinstance(result[0], list) and isinstance(result[1], list)
    assert result[0][0].qid == QID_BERLIN
    assert result[1][0].qid == QID_PARIS


async def test_link_batch_equals_sequential(client):
    """Batched results match per-name sequential results."""
    names = ["Berlin", "Germany", "Paris"]
    batch_result = await client.link_entities(names, top_k=1)
    seq_results = [await client.link_entities(n, top_k=1) for n in names]
    batch_qids = [lst[0].qid for lst in batch_result]
    seq_qids = [lst[0].qid for lst in seq_results]
    assert batch_qids == seq_qids


async def test_link_unknown_name_yields_empty_list(client):
    """No match → empty result for that name (no exception)."""
    result = await client.link_entities("AtlantisUnknownPlace123", top_k=1)
    assert isinstance(result, list)
    assert len(result) == 0


async def test_link_qid_input_passthrough(client, mini_graph):
    """Input matching ``Q\\d+`` is a direct lookup, not a text search."""
    before = mini_graph.call_count("search_entities_text")
    result = await client.link_entities("Q64", top_k=1)
    after = mini_graph.call_count("search_entities_text")
    assert after == before, "QID input must not invoke entity text-search"
    assert len(result) >= 1
    assert result[0].qid == QID_BERLIN


async def test_link_top_k_clamped_to_available_matches(client):
    """top_k > available returns all available, no error."""
    result = await client.link_entities("Berlin", top_k=50)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0].qid == QID_BERLIN


# ---------------- batching parity ----------------


async def test_link_batch_one_search_call_per_query(client, mini_graph):
    """N names → exactly N text-search calls (one per query)."""
    names = ["Berlin", "Germany", "Paris"]
    await client.link_entities(names, top_k=1)
    assert mini_graph.call_count("search_entities_text") == len(names)


async def test_link_batch_uses_single_get_entity_details(client, mini_graph):
    """All qids resolved across a batch must be enriched in one backend call."""
    names = ["Berlin", "Germany", "Paris", "France"]
    await client.link_entities(names, top_k=1)
    detail_calls = mini_graph.calls("get_entity_details")
    assert len(detail_calls) <= 1, (
        f"expected ≤1 batched get_entity_details call, got {len(detail_calls)}"
    )
    if detail_calls:
        called_qids = set(detail_calls[0].args[0])
        assert called_qids >= {QID_BERLIN, QID_GERMANY, QID_PARIS, QID_FRANCE}


async def test_link_top_k_does_not_inflate_detail_calls(client, mini_graph):
    """top_k=5 must not result in 5x detail calls."""
    await client.link_entities("Berlin", top_k=5)
    detail_calls = mini_graph.calls("get_entity_details")
    assert len(detail_calls) <= 1
