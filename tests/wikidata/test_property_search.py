"""Contract tests for ``WikidataClient.search_properties``.

Target signature::

    async def search_properties(
        self, query: str | list[str], *, top_k: int = 1
    ) -> list[WikidataProperty] | list[list[WikidataProperty]]
"""

from __future__ import annotations

import pytest

from ._fixtures import PID_CAPITAL, PID_CONTINENT, PID_COUNTRY


# ---------------- correctness ----------------


async def test_search_property_pid_input_direct_lookup(client, mini_graph):
    """``Pxxx`` input is a direct lookup, no text-search call."""
    before = mini_graph.call_count("search_properties_text")
    result = await client.search_properties("P36", top_k=1)
    after = mini_graph.call_count("search_properties_text")
    assert after == before
    assert len(result) >= 1
    assert result[0].pid == PID_CAPITAL


async def test_search_property_text_returns_top_k_ordered(client, mini_graph):
    """Text query returns up to top_k properties, ordered by index."""
    mini_graph.add_property(
        "P9999", label="capital",
        description="alternative capital property",
        search_terms=("capital",),
    )
    result = await client.search_properties("capital", top_k=3)
    assert isinstance(result, list)
    pids = [p.pid for p in result]
    assert pids[0] == PID_CAPITAL
    assert "P9999" in pids
    assert len(result) <= 3


async def test_search_property_batch_returns_per_query_lists(client):
    """List input returns list-of-list, one per query."""
    result = await client.search_properties(["capital", "country"], top_k=1)
    assert isinstance(result, list) and len(result) == 2
    assert isinstance(result[0], list) and isinstance(result[1], list)
    assert result[0][0].pid == PID_CAPITAL
    assert result[1][0].pid == PID_COUNTRY


async def test_search_property_top_k_respected(client, mini_graph):
    """top_k caps the returned length per query."""
    for i in range(5):
        mini_graph.add_property(
            f"P777{i}", label="continent",
            description=f"alternate continent {i}",
            search_terms=("continent",),
        )
    result = await client.search_properties("continent", top_k=2)
    assert len(result) == 2
    pids = [p.pid for p in result]
    assert pids[0] == PID_CONTINENT


# ---------------- batching parity ----------------


async def test_property_batch_single_get_property_details_call(client, mini_graph):
    """All PIDs across a batch must be enriched in one backend call."""
    await client.search_properties(["capital", "country", "continent"], top_k=1)
    detail_calls = mini_graph.calls("get_property_details")
    assert len(detail_calls) <= 1
    if detail_calls:
        called_pids = set(detail_calls[0].args[0])
        assert called_pids >= {PID_CAPITAL, PID_COUNTRY, PID_CONTINENT}


# ---------------- error handling ----------------


async def test_property_search_no_match_returns_empty(client):
    """No-match query returns empty list, not error."""
    result = await client.search_properties("zzzz_nonexistent_relation", top_k=3)
    assert isinstance(result, list)
    assert len(result) == 0


async def test_property_uninformative_label_filtered(client, mini_graph):
    """Properties whose label equals their own PID are uninformative and filtered out."""
    mini_graph.add_property(
        "P88888",
        label="P88888",  # uninformative: label same as PID
        description="P88888",
        search_terms=("garbage",),
    )
    result = await client.search_properties("garbage", top_k=5)
    assert all(p.pid != "P88888" for p in result), (
        "uninformative property (label == pid) must be filtered"
    )
