"""Contract tests for the LangChain @tool wrappers.

Targets ``langgraph_coe.tools.wikidata``:
  - ``link_entities``
  - ``enrich_entities``
  - ``fetch_and_prune_subgraph``
  - ``create_fetch_and_prune_tool``

Tools are exercised via ``BaseTool.ainvoke({...})`` so they go through the
LangChain plumbing the agent uses.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_GERMANY,
    QID_PARIS,
)


@pytest.fixture
async def tool_module(client, monkeypatch):
    """Inject the test client + config into the @tool module, reset per-test.

    Async fixture so ``reset_wikidata_session()`` runs in the test's asyncio
    Task context; subsequent ``ainvoke`` child tasks inherit the same
    ``_SessionState`` object and mutations accumulate across calls.
    """
    from langgraph_coe.config import WikidataConfig
    from langgraph_coe.tools import wikidata as mod

    cfg = WikidataConfig()
    monkeypatch.setattr(mod, "_wikidata_client", client, raising=False)
    monkeypatch.setattr(mod, "_wikidata_config", cfg, raising=False)
    mod.entity_cache.clear()
    mod.reset_wikidata_session()
    return mod


# ---------------- link_entities tool ----------------


async def test_link_entities_tool_returns_name_qid_description_dicts(tool_module):
    result = await tool_module.link_entities.ainvoke(
        {"entity_names": ["Berlin", "Paris"]}
    )
    assert isinstance(result, list)
    by_name = {r["name"]: r for r in result}
    assert by_name["Berlin"]["qid"] == QID_BERLIN
    assert by_name["Paris"]["qid"] == QID_PARIS
    assert "description" in by_name["Berlin"]


async def test_link_entities_tool_global_entity_cache_across_calls(tool_module, mini_graph):
    await tool_module.link_entities.ainvoke({"entity_names": ["Berlin"]})
    assert "Berlin" in tool_module.entity_cache
    n_after_first = mini_graph.call_count("search_entities_text")
    await tool_module.link_entities.ainvoke({"entity_names": ["Berlin"]})
    assert mini_graph.call_count("search_entities_text") == n_after_first, (
        "second call must not re-search a globally-cached name"
    )


async def test_link_entities_tool_skips_cached_names_no_extra_backend_call(
    tool_module, mini_graph
):
    await tool_module.link_entities.ainvoke({"entity_names": ["Berlin"]})
    n_search_after = mini_graph.call_count("search_entities_text")
    # Mixed batch: one cached + one new
    await tool_module.link_entities.ainvoke(
        {"entity_names": ["Berlin", "Paris"]}
    )
    # Only Paris should produce a new text-search call
    assert mini_graph.call_count("search_entities_text") == n_search_after + 1


# ---------------- fetch_and_prune_subgraph tool ----------------


async def test_fetch_and_prune_subgraph_no_qids_marker(tool_module):
    result = await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": [], "query": "anything"}
    )
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], str)
    assert "No valid QIDs" in result[0]


async def test_fetch_and_prune_subgraph_already_visited_marker(tool_module):
    # First call marks QID_BERLIN as visited
    await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": [QID_BERLIN], "query": "Where is Berlin?"}
    )
    # Re-calling with the same qid → "Already explored" marker
    result = await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": [QID_BERLIN], "query": "Where is Berlin?"}
    )
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], str)
    assert "Already explored" in result[0]


async def test_fetch_and_prune_subgraph_hop_budget_exhausted_marker(tool_module):
    """After max_hops calls the budget marker is returned, no further fetches."""
    cfg = tool_module._wikidata_config
    max_hops = cfg.max_hops
    # Walk distinct seeds to exhaust hop budget without triggering the
    # "already visited" path.
    seeds = [QID_BERLIN, QID_GERMANY, QID_PARIS, "QSEED4", "QSEED5"]
    for i in range(max_hops):
        await tool_module.fetch_and_prune_subgraph.ainvoke(
            {"qids": [seeds[i]], "query": "Q"}
        )
    result = await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": ["QFRESHSEED"], "query": "Q"}
    )
    assert isinstance(result, list) and len(result) >= 1
    assert isinstance(result[0], str)
    assert "budget" in result[0].lower() or "hop" in result[0].lower()


async def test_fetch_and_prune_subgraph_stage_a_reranker_failure_returns_unpruned(
    tool_module, monkeypatch
):
    """If Stage A reranker fails (network/HTTP), the tool returns the unpruned set."""
    import httpx
    from langgraph_coe.tools import wikidata as mod

    class FailingAsyncClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, *a, **kw):
            raise httpx.ConnectError("reranker down")

    monkeypatch.setattr(mod.httpx, "AsyncClient", FailingAsyncClient)
    result = await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": [QID_BERLIN], "query": "Where is Berlin?"}
    )
    assert isinstance(result, list)
    # Returns triples (or markers) — must not raise


async def test_fetch_and_prune_subgraph_stage_b_registry_called(tool_module, monkeypatch):
    """When a registry is supplied via the factory, Stage B LLM prune runs."""
    from langgraph_coe.tools import wikidata as mod

    called = {"count": 0}

    async def fake_execute_role_lc(registry, role, inp):
        called["count"] += 1
        # Pretend the LLM kept every triple.
        from types import SimpleNamespace
        keep = list(range(len(inp.triples)))
        return SimpleNamespace(keep_indices=keep), None

    monkeypatch.setattr("langgraph_coe.llm.execute_role_lc", fake_execute_role_lc)

    factory_tool = mod.create_fetch_and_prune_tool(registry=object())
    await factory_tool.ainvoke({"qids": [QID_BERLIN], "query": "Q"})
    assert called["count"] >= 1


# ---------------- enrich_entities tool ----------------


async def test_enrich_entities_tool_empty_qids_returns_empty(tool_module):
    result = await tool_module.enrich_entities.ainvoke({"qids": []})
    assert result == []


async def test_enrich_entities_tool_returns_entity_objects(tool_module):
    result = await tool_module.enrich_entities.ainvoke(
        {"qids": [QID_BERLIN, QID_GERMANY]}
    )
    assert len(result) == 2
    by_qid = {e.qid: e for e in result}
    assert by_qid[QID_BERLIN].label == "Berlin"
    assert by_qid[QID_GERMANY].label == "Germany"


# ---------------- init / session reset ----------------


async def test_init_required_before_tools_callable(monkeypatch):
    """Tools must raise if init_wikidata was never called."""
    from langgraph_coe.tools import wikidata as mod
    monkeypatch.setattr(mod, "_wikidata_client", None, raising=False)
    monkeypatch.setattr(mod, "_wikidata_config", None, raising=False)
    with pytest.raises(RuntimeError, match="not initialised|not initialized|init_wikidata"):
        await mod.link_entities.ainvoke({"entity_names": ["Berlin"]})


async def test_reset_wikidata_session_clears_per_question_state(tool_module):
    """reset_wikidata_session must rebind to a fresh visited-QID + hop counter."""
    await tool_module.fetch_and_prune_subgraph.ainvoke(
        {"qids": [QID_BERLIN], "query": "Q"}
    )
    visited_before = set(tool_module._get_session().visited)
    assert QID_BERLIN in visited_before
    tool_module.reset_wikidata_session()
    visited_after = set(tool_module._get_session().visited)
    assert QID_BERLIN not in visited_after


async def test_per_task_session_isolation(tool_module):
    """Concurrent asyncio.Tasks each call reset → fully isolated session state."""
    async def task_for(qid):
        # Rebind ContextVar to a fresh SessionState inside THIS task's context;
        # the rebinding stays local to this task and isolates it from siblings.
        tool_module.reset_wikidata_session()
        await tool_module.fetch_and_prune_subgraph.ainvoke(
            {"qids": [qid], "query": "Q"}
        )
        return set(tool_module._get_session().visited)

    visited_a, visited_b = await asyncio.gather(
        task_for(QID_BERLIN), task_for(QID_PARIS)
    )
    assert visited_a == {QID_BERLIN}
    assert visited_b == {QID_PARIS}
