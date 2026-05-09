"""Exercise every LangChain tool exported by ``langgraph_coe.tools``."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from langgraph_coe.tools.retrieval import corpus_search, init_retrieval_pipeline
from langgraph_coe.tools.web import web_search

# ── Wikidata: optional dependency -------------------------------------------


def _wikidata_tools():
    if importlib.util.find_spec("SPARQLWrapper") is None:
        pytest.skip("SPARQLWrapper not installed")
    return importlib.import_module("langgraph_coe.tools.wikidata")


@pytest.fixture
def wd_tools(langgraph_coe_wikidata_initialized):  # noqa: ARG002
    return _wikidata_tools()


# ── Guards (no mocks of tool behaviour — only globals reset) ──────────────


@pytest.mark.asyncio
async def test_corpus_search_requires_init(monkeypatch):
    import langgraph_coe.tools.retrieval as rmod

    monkeypatch.setattr(rmod, "_retriever_instance", None, raising=False)
    monkeypatch.setattr(rmod, "_reranker_config", None, raising=False)
    with pytest.raises(RuntimeError, match="init_retrieval_pipeline"):
        await corpus_search.ainvoke({"query": "anything"})


@pytest.mark.asyncio
async def test_web_search_requires_init(monkeypatch):
    import langgraph_coe.tools.web as wmod

    monkeypatch.setattr(wmod, "_web_search_instance", None, raising=False)
    monkeypatch.setattr(wmod, "_web_config", None, raising=False)
    with pytest.raises(RuntimeError, match="init_web_search"):
        await web_search.ainvoke({"query": "anything"})


# ── Wikidata (live APIs) ────────────────────────────────────────────────────


@pytest.mark.requires_wikidata
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_link_entities_resolves_well_known_city(wd_tools):
    link_entities = wd_tools.link_entities
    out = await link_entities.ainvoke({"entity_names": ["Berlin"]})
    assert isinstance(out, list)
    assert len(out) >= 1
    row = out[0]
    assert row.get("name") == "Berlin"
    assert row.get("qid", "").startswith("Q")
    assert isinstance(row.get("description", ""), str)


@pytest.mark.requires_wikidata
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_enrich_entities_returns_details_for_known_qids(wd_tools):
    enrich_entities = wd_tools.enrich_entities
    out = await enrich_entities.ainvoke({"qids": ["Q64"]})
    assert isinstance(out, list)
    assert len(out) >= 1
    ent = out[0]
    assert getattr(ent, "qid", None) == "Q64"
    assert getattr(ent, "label", None) or getattr(ent, "description", None) is not None


@pytest.mark.requires_wikidata
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_fetch_and_prune_subgraph_no_registry(wd_tools):
    fetch_and_prune_subgraph = wd_tools.fetch_and_prune_subgraph
    out = await fetch_and_prune_subgraph.ainvoke(
        {"qids": ["Q64"], "query": "What is the capital of Germany?"}
    )
    assert isinstance(out, list)
    assert len(out) >= 1
    assert any(
        hasattr(item, "subject") or (isinstance(item, str) and item.strip())
        for item in out
    )


@pytest.mark.requires_wikidata
@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_create_fetch_and_prune_via_registry(langgraph_coe_config, wd_tools):
    from tests.langgraph_coe.conftest import requires_coe_llm

    requires_coe_llm(langgraph_coe_config)

    create_fetch_and_prune_tool = wd_tools.create_fetch_and_prune_tool
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(langgraph_coe_config.llm)
    wrapped = create_fetch_and_prune_tool(registry)
    out = await wrapped.ainvoke({"qids": ["Q64"], "query": "Is Berlin in Germany?"})
    assert isinstance(out, list)
    assert len(out) >= 1


# ── Web search (live) ─────────────────────────────────────────────────────────


@pytest.mark.requires_web_search
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_web_search_returns_document_strings(langgraph_coe_web_ready):
    out = await web_search.ainvoke({"query": "Python programming language"})
    assert isinstance(out, list)
    assert len(out) >= 1
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in out)


# ── Corpus retrieval (live embedder + FAISS when index present) ───────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_corpus_search_roundtrip(langgraph_coe_config):
#     """Initialises globals only inside this test; skips when index/embedder unreachable."""

#     from tests.langgraph_coe.conftest import requires_coe_corpus_index

#     import langgraph_coe.tools.retrieval as rmod

#     requires_coe_corpus_index(langgraph_coe_config)
#     init_retrieval_pipeline(langgraph_coe_config.retriever, langgraph_coe_config.reranker)
#     try:
#         out = await corpus_search.ainvoke({"query": "machine learning retrieval"})
#         assert isinstance(out, list)
#         assert all(isinstance(chunk, str) for chunk in out)
#     except Exception as exc:
#         pytest.skip(f"Live corpus/embedder unavailable: {exc}")
#     finally:
#         rmod._retriever_instance = None
#         rmod._reranker_config = None
