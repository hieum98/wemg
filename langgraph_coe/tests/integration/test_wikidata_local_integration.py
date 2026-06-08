"""Full integration tests for Wikidata client + @tools against local QEndpoint SPARQL.

SPARQL k-hop uses ``wikidata.sparql_endpoint`` (n0162:1234). Entity search,
property metadata, and Wikipedia still use the public Wikidata / Wikipedia APIs.

Setup: ``docs/setup_wikidata_tools.md`` and ``docs/deploy_local_wikidata-v2.md``.

From WSL:

    ssh -fN -L 30162:n0162:1234 t2
    export LANGGRAPH_TEST_SPARQL_URL=http://127.0.0.1:30162/api/endpoint/sparql
    uv run pytest langgraph_coe/tests/phase0/test_wikidata_local_integration.py -v
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, WikidataConfig
from langgraph_coe.tools.wikidata import (
    enrich_entities,
    fetch_and_prune_subgraph,
    init_wikidata,
    link_entities,
    reset_wikidata_session,
)
from langgraph_coe.tools.wikidata_client import WikidataClient, WikiTriple

from .._fixtures import log_config_override


SPARQL_URL = os.environ.get(
    "LANGGRAPH_TEST_SPARQL_URL",
    "http://127.0.0.1:30162/api/endpoint/sparql",
)


def _sparql_alive(url: str) -> bool:
    probe = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                url,
                params={"query": probe},
                headers={"Accept": "application/sparql-results+json"},
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            return bool(data.get("results", {}).get("bindings"))
    except Exception:
        return False


_LOCAL_SPARQL_UP = _sparql_alive(SPARQL_URL)

pytestmark = [
    pytest.mark.requires_wikidata,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _LOCAL_SPARQL_UP,
        reason=(
            f"Local SPARQL endpoint unreachable at {SPARQL_URL}. "
            "Open: ssh -fN -L 30162:n0162:1234 t2"
        ),
    ),
]


def _local_cfg(*, max_hops: Optional[int] = None) -> WikidataConfig:
    """config.yaml ``wikidata`` block; only endpoint/hop deviations are logged."""
    cfg = LangGraphCoeConfig.from_yaml().wikidata.model_copy(deep=True)
    cfg.sparql_endpoint = log_config_override(
        "wikidata.sparql_endpoint",
        cfg.sparql_endpoint,
        SPARQL_URL,
        reason="tunnelled local QEndpoint / LANGGRAPH_TEST_SPARQL_URL",
    )
    cfg.reranker_url = log_config_override(
        "wikidata.reranker_url",
        cfg.reranker_url,
        None,
        reason="Wikidata-only path; Stage-A rerank skipped",
    )
    cfg.reranker_model = None
    if max_hops is not None:
        cfg.max_hops = log_config_override(
            "wikidata.max_hops",
            cfg.max_hops,
            max_hops,
            reason="test pins the hop budget for this scenario",
        )
    return cfg


@pytest.fixture
def local_client() -> WikidataClient:
    cfg = LangGraphCoeConfig.from_yaml().wikidata
    return WikidataClient(
        sparql_endpoint=SPARQL_URL,
        max_sparql_rps=cfg.max_sparql_rps,
        max_wikipedia_rps=cfg.max_wikipedia_rps,
    )


@pytest.fixture
def local_tools():
    init_wikidata(_local_cfg())
    reset_wikidata_session()
    yield
    reset_wikidata_session()


# ── WikidataClient (HTTP backend + local SPARQL) ─────────────────────────────


@pytest.mark.asyncio
async def test_client_link_berlin_q64(local_client: WikidataClient) -> None:
    result = await local_client.link_entities("Berlin", top_k=1)
    assert result and result[0].qid == "Q64"


@pytest.mark.asyncio
async def test_client_link_batch_paris_berlin(local_client: WikidataClient) -> None:
    batch = await local_client.link_entities(["Paris", "Berlin"], top_k=1)
    assert len(batch) == 2
    qids = {row[0].qid for row in batch if row}
    assert "Q90" in qids and "Q64" in qids


@pytest.mark.asyncio
async def test_client_enrich_q64(local_client: WikidataClient) -> None:
    result = await local_client.enrich_entities(["Q64"], get_details=True)
    assert len(result) == 1
    assert result[0].qid == "Q64"
    assert result[0].label and "Berlin" in result[0].label


@pytest.mark.asyncio
async def test_client_search_properties_capital(local_client: WikidataClient) -> None:
    result = await local_client.search_properties("capital", top_k=1)
    assert result and result[0].pid
    if result[0].label:
        assert "capital" in result[0].label.lower()


@pytest.mark.asyncio
async def test_client_khop_q64_forward(local_client: WikidataClient) -> None:
    triples = await local_client.get_k_hop_triples(
        "Q64", k=1, bidirectional=False, enrich=False
    )
    assert isinstance(triples, list) and len(triples) > 0


@pytest.mark.asyncio
async def test_client_khop_q64_bidirectional(local_client: WikidataClient) -> None:
    triples = await local_client.get_k_hop_triples(
        "Q64", k=1, bidirectional=True, enrich=False
    )
    assert isinstance(triples, list) and len(triples) > 0


@pytest.mark.asyncio
async def test_client_fetch_outgoing_uoregon(local_client: WikidataClient) -> None:
    edges = await local_client._fetch_outgoing(["Q766145"])  # noqa: SLF001
    assert "Q766145" in edges and len(edges["Q766145"]) > 0


# ── @tool wrappers ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_link_entities(local_tools) -> None:
    linked = await link_entities.ainvoke(
        {"entity_names": ["University of Oregon", "Berlin"]}
    )
    by_name = {r["name"]: r for r in linked}
    assert by_name["University of Oregon"]["qid"] == "Q766145"
    assert by_name["Berlin"]["qid"] == "Q64"


@pytest.mark.asyncio
async def test_tool_fetch_subgraph_uoregon(local_tools) -> None:
    reset_wikidata_session()
    out = await fetch_and_prune_subgraph.ainvoke(
        {
            "qids": ["Q766145"],
            "query": "When was the University of Oregon founded?",
        }
    )
    assert out and not str(out[0]).startswith("[")
    triples = [t for t in out if isinstance(t, WikiTriple)]
    assert triples
    assert any("P571" in str(t) or "founded" in str(t).lower() for t in triples)


@pytest.mark.asyncio
async def test_tool_enrich_entities(local_tools) -> None:
    enriched = await enrich_entities.ainvoke({"qids": ["Q766145", "Q64"]})
    assert len(enriched) == 2
    labels = {e.qid: e.label for e in enriched}
    assert labels["Q766145"] and "Oregon" in labels["Q766145"]
    assert labels["Q64"] and "Berlin" in labels["Q64"]


@pytest.mark.asyncio
async def test_tool_fetch_empty_qids(local_tools) -> None:
    out = await fetch_and_prune_subgraph.ainvoke({"qids": [], "query": "x"})
    assert out and "No valid QIDs" in out[0]


@pytest.mark.asyncio
async def test_tool_fetch_already_visited(local_tools) -> None:
    reset_wikidata_session()
    await fetch_and_prune_subgraph.ainvoke(
        {"qids": ["Q64"], "query": "capital of Germany"}
    )
    out = await fetch_and_prune_subgraph.ainvoke(
        {"qids": ["Q64"], "query": "capital of Germany"}
    )
    assert out and "Already explored" in out[0]


@pytest.mark.asyncio
async def test_tool_hop_budget_exhausted(local_tools) -> None:
    """One real SPARQL hop, then hop-budget marker (no second SPARQL)."""
    init_wikidata(_local_cfg(max_hops=1))
    reset_wikidata_session()
    first = await fetch_and_prune_subgraph.ainvoke(
        {"qids": ["Q64"], "query": "Berlin facts"}
    )
    assert first and not str(first[0]).startswith("[")
    second = await fetch_and_prune_subgraph.ainvoke(
        {"qids": ["Q90"], "query": "Paris facts"}
    )
    assert second and ("budget" in second[0].lower() or "hop" in second[0].lower())


@pytest.mark.asyncio
async def test_tool_reset_session_clears_visited(local_tools) -> None:
    from langgraph_coe.tools import wikidata as mod

    reset_wikidata_session()
    await fetch_and_prune_subgraph.ainvoke({"qids": ["Q64"], "query": "q"})
    assert "Q64" in mod._get_session().visited
    reset_wikidata_session()
    assert "Q64" not in mod._get_session().visited


@pytest.mark.asyncio
async def test_tool_pipeline_link_fetch_enrich(local_tools) -> None:
    reset_wikidata_session()
    linked = await link_entities.ainvoke({"entity_names": ["Berlin"]})
    qid = linked[0]["qid"]
    triples = await fetch_and_prune_subgraph.ainvoke(
        {"qids": [qid], "query": "What country is Berlin in?"}
    )
    assert any(isinstance(t, WikiTriple) for t in triples)
    enriched = await enrich_entities.ainvoke({"qids": [qid]})
    assert enriched[0].label and "Berlin" in enriched[0].label
