"""Phase 2 integration smoke test against real Qwen endpoints.

End-to-end exercises ``CoTGraph`` through the live ``Qwen/Qwen3-8B`` LLM
(SGLang via the SSH-tunneled ``localhost:30172`` → ``n0172:30000``). Retrieval
subgraphs (KG, Web, corpus) are replaced with deterministic stubs so the test
isolates the Phase 2 control flow + role-driven reasoning, without depending on
Wikidata / Serper / a FAISS index.

What this test verifies:
- The CoT loop completes end-to-end against a real model, including the parallel
  fan-out → barrier → rerank → subanswer → memory-update → increment cycle.
- ``iteration_history`` accumulates one entry per non-terminal iteration.
- ``text_memory`` is mutated by ``MemoryUpdateGraph`` (Phase 1 already covered).
- A final answer is produced and mentions the seeded retrieval facts.

Skipped when the LLM/embedder endpoints are unreachable, mirroring the Phase 1
integration test exactly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import httpx
import networkx as nx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, TierConfig
from langgraph_coe.llm import RoleModelRegistry


LLM_URL = os.environ.get("LANGGRAPH_TEST_LLM_URL", "http://localhost:30172/v1")
LLM_MODEL = os.environ.get("LANGGRAPH_TEST_LLM_MODEL", "openai/Qwen/Qwen3-8B")
EMBED_URL = os.environ.get("LANGGRAPH_TEST_EMBED_URL", "http://localhost:30164/v1")
EMBED_MODEL = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


def _endpoint_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{url.rstrip('/')}/models")
            return resp.status_code == 200
    except Exception:
        return False


def _real_endpoints_available() -> bool:
    return _endpoint_alive(LLM_URL) and _endpoint_alive(EMBED_URL)


pytestmark = pytest.mark.skipif(
    not _real_endpoints_available(),
    reason=(
        "Real model endpoints unreachable. Open SSH tunnels:\n"
        "  ssh -fN -L 30172:n0172:30000 -L 30164:n0164:30000 t2"
    ),
)


def _build_real_config() -> LangGraphCoeConfig:
    cfg = LangGraphCoeConfig.from_yaml()
    # Reranker stays off; the integration uses the default identity reranker
    # in ``cot.rerank_context``.
    cfg.reranker.enabled = False
    cfg.reranker.top_k = 4
    return cfg


def _build_real_registry(cfg: LangGraphCoeConfig) -> RoleModelRegistry:
    tier = TierConfig(
        model_name=LLM_MODEL,
        api_base=LLM_URL,
        api_key="EMPTY",
        temperature=0.2,
        # Thinking-mode Qwen3 spends thousands of tokens before emitting JSON;
        # bumping to 16k matches Phase 1 integration.
        max_tokens=16384,
        max_input_tokens=32768,
        top_p=0.95,
        enable_thinking=True,
        max_retries=2,
        timeout=300,
    )
    # §3b: bounded-output roles (triple_pruner, NER) route to a lean, thinking-off
    # tier. Keep generous token headroom on the 8B so the path under test is
    # purely "no-think structured output", not truncation.
    classify_tier = tier.model_copy(update={"temperature": 0.0, "enable_thinking": False, "max_tokens": 4096})
    cfg.llm.tiers = {"heavy": tier, "medium": tier, "light": tier, "classify": classify_tier}
    cfg.llm.api_key = "EMPTY"
    return RoleModelRegistry(cfg.llm)


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic stubs for the three retrieval subgraphs / corpus tool
# ──────────────────────────────────────────────────────────────────────────────


class _StubKGGraph:
    """Stand-in for ``build_kg_search_graph`` output.

    Returns a fixed pair of (article, triple) per subquery, so the CoT loop sees
    real-looking KG output without touching live Wikidata.
    """

    def __init__(self, articles: Sequence[str], triples: Sequence[str]) -> None:
        self._articles = list(articles)
        self._triples = list(triples)
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, state: Dict[str, Any], *_a: Any, **_kw: Any) -> Dict[str, Any]:
        self.calls.append(state)
        return {"kg_articles": list(self._articles), "triples": list(self._triples)}


class _StubWebGraph:
    """Stand-in for ``build_web_research_graph`` output."""

    def __init__(self, results: Sequence[Dict[str, str]]) -> None:
        self._results = [dict(r) for r in results]
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, state: Dict[str, Any], *_a: Any, **_kw: Any) -> Dict[str, Any]:
        self.calls.append(state)
        return {"results": [dict(r) for r in self._results]}


class _StubCorpusSearch:
    """Stand-in for the ``corpus_search`` LangChain tool.

    Mimics ``BaseTool.ainvoke`` so the CoT corpus_join node sees the same shape
    a real FAISS-backed retriever would produce.
    """

    name = "corpus_search"

    def __init__(self, results: Sequence[str]) -> None:
        self._results = list(results)
        self.invocations: List[Any] = []

    async def ainvoke(self, payload: Any, *_a: Any, **_kw: Any) -> List[str]:
        self.invocations.append(payload)
        return list(self._results)


class _StubLinkEntities:
    """Stand-in for the Wikidata ``link_entities`` tool used by MemoryUpdateGraph.

    The CoT loop invokes ``MemoryUpdateGraph`` which would otherwise reach the
    real Wikidata SPARQL endpoint via ``link_entities``. Mirroring the Phase 1
    integration test, we mock it with a deterministic surface-form → QID map.
    """

    name = "link_entities"

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping
        self.invocations: List[Any] = []

    async def ainvoke(self, inp: Any, *args: Any, **kwargs: Any) -> List[Dict[str, str]]:
        self.invocations.append(inp)
        names = inp.get("entity_names", inp) if isinstance(inp, dict) else inp
        out: List[Dict[str, str]] = []
        for name in names or []:
            qid = self._mapping.get(str(name))
            if qid:
                out.append({"name": str(name), "qid": qid, "description": ""})
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_cot_graph_runs_against_real_qwen(monkeypatch):
    """Full CoT loop completes end-to-end with the real Qwen3-8B model."""
    from langgraph_coe.graphs import cot as cot_mod

    cfg = _build_real_config()
    registry = _build_real_registry(cfg)

    kg_stub = _StubKGGraph(
        articles=[
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        ],
        triples=[
            "France | capital | Paris",
            "Eiffel Tower | located_in | Paris",
        ],
    )
    web_stub = _StubWebGraph(
        results=[
            {
                "title": "Paris — Wikipedia",
                "url": "https://example.org/paris",
                "snippet": "Paris, capital of France, is on the Seine.",
                "full_text": "",
            }
        ]
    )
    corpus_stub = _StubCorpusSearch(
        results=[
            "Paris has been the capital of France since 987 AD.",
        ]
    )

    monkeypatch.setattr(
        cot_mod, "build_kg_search_graph", lambda *_a, **_kw: kg_stub, raising=False
    )
    monkeypatch.setattr(
        cot_mod, "build_web_research_graph", lambda *_a, **_kw: web_stub, raising=False
    )
    monkeypatch.setattr(cot_mod, "corpus_search", corpus_stub, raising=False)

    # Stub the Wikidata link_entities tool reached via MemoryUpdateGraph so the
    # integration stays self-contained (no live SPARQL endpoint required).
    from langgraph_coe.graphs import memory_update as mem_mod

    link_stub = _StubLinkEntities(
        {"France": "Q142", "Paris": "Q90", "Eiffel Tower": "Q243"}
    )
    monkeypatch.setattr(mem_mod, "link_entities", link_stub, raising=False)

    graph = cot_mod.build_cot_graph(registry, cfg)

    initial_state: Dict[str, Any] = {
        "question": "What is the capital of France and one famous landmark there?",
        "max_depth": 2,
        "depth": 0,
        "is_answerable": False,
        "subquestions": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": ["France is a country located in Western Europe."],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "final_answer": "",
    }

    final = await graph.ainvoke(initial_state, config={"recursion_limit": 75})
    print(final)

    assert final.get("final_answer"), "CoT must produce a non-empty final answer"
    answer = str(final["final_answer"]).lower()
    assert "paris" in answer, (
        f"final answer must reference Paris (the retrieval seed); got: {final['final_answer']!r}"
    )

    history = final.get("iteration_history") or []
    assert history, "expected at least one CoT iteration before terminating"
    for entry in history:
        assert "subquestions" in entry and "subanswers" in entry
        assert isinstance(entry["subquestions"], list)
        assert isinstance(entry["subanswers"], list)

    if history:
        # Per-iteration scratch must reset by the end of the run.
        assert final.get("retrieved_raw_context") == []
        assert final.get("retrieved_raw_triples") == []


async def test_cot_graph_respects_depth_limit_against_real_qwen(monkeypatch):
    """``max_depth=0`` routes through gen_subq → gen_final without retrieval."""
    from langgraph_coe.graphs import cot as cot_mod

    cfg = _build_real_config()
    registry = _build_real_registry(cfg)

    kg_stub = _StubKGGraph(articles=["unused"], triples=["unused"])
    web_stub = _StubWebGraph(results=[{"title": "u", "url": "u", "snippet": "u", "full_text": ""}])
    corpus_stub = _StubCorpusSearch(results=["unused"])

    monkeypatch.setattr(
        cot_mod, "build_kg_search_graph", lambda *_a, **_kw: kg_stub, raising=False
    )
    monkeypatch.setattr(
        cot_mod, "build_web_research_graph", lambda *_a, **_kw: web_stub, raising=False
    )
    monkeypatch.setattr(cot_mod, "corpus_search", corpus_stub, raising=False)

    # Even though depth_limit short-circuits before MemoryUpdateGraph, stub
    # link_entities anyway so the test cannot accidentally regress into a real
    # SPARQL hit.
    from langgraph_coe.graphs import memory_update as mem_mod

    monkeypatch.setattr(mem_mod, "link_entities", _StubLinkEntities({}), raising=False)

    graph = cot_mod.build_cot_graph(registry, cfg)

    initial_state: Dict[str, Any] = {
        "question": "What is 2 + 2?",
        "max_depth": 0,
        "depth": 0,
        "is_answerable": False,
        "subquestions": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": ["Basic arithmetic: 2 + 2 = 4."],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "final_answer": "",
    }

    final = await graph.ainvoke(initial_state, config={"recursion_limit": 25})
    print(final)

    assert final.get("final_answer"), "depth-limited CoT must still produce a final answer"
    # No retrieval branch should have been invoked.
    assert not kg_stub.calls, "depth-limited path must skip KG retrieval"
    assert not web_stub.calls, "depth-limited path must skip web retrieval"
    assert not corpus_stub.invocations, "depth-limited path must skip corpus retrieval"
    # No CoT iterations completed (no increment ever ran).
    assert final.get("iteration_history") == []
