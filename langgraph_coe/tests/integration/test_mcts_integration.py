"""Integration smoke test against real Qwen endpoints.

End-to-end exercises ``MCTSGraph`` through the live ``Qwen/Qwen3-8B`` LLM
(SGLang via the SSH-tunneled ``localhost:30172`` → ``n0172:30000``). The full
search loop runs against the real model: ``select → expand → simulate →
evaluate → backprop → mem_update → synthesize``.

The rollout subgraph (``CoTGraph``) and ``MemoryUpdateGraph`` are the **real**
compiled graphs driven by the live LLM — only their external retrieval surfaces
(KG / Web / corpus) and the Wikidata ``link_entities`` tool are replaced with
deterministic stubs, so the test isolates control flow + role-driven
reasoning without depending on Wikidata / Serper / a FAISS index.

What this verifies against a real model:
- The MCTS iteration completes end-to-end (root seeding, pUCT select, parallel
  expansion, CoT rollout, 3-view verifier reward, backprop, memory sync).
- The search tree gains the planned node types (``user_question`` root,
  ``sub_qa``, ``final_answer``) with the documented priors.
- ``reward`` is normalized into ``[-1, 1]`` and backprop visits/values land on
  the evaluated ``current_path``.
- A final answer is synthesized and references the seeded retrieval facts.
- ``FINAL_ANSWER`` leaves are terminal: a pre-seeded final-answer node is
  evaluated without re-running expansion generators.

Skipped when the LLM/embedder endpoints are unreachable, mirroring the 
and integration tests exactly. Open the tunnels with::

    ssh -fN -L 30172:n0172:30000 -L 30164:n0164:30000 t2
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import networkx as nx
import pytest

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.llm import RoleModelRegistry

from .._fixtures import log_config_override, override_tier_endpoint
from .._servers import endpoint_alive as _endpoint_alive

LLM_URL = os.environ.get("LANGGRAPH_TEST_LLM_URL", "http://localhost:30172/v1")
LLM_MODEL = os.environ.get("LANGGRAPH_TEST_LLM_MODEL", "openai/Qwen/Qwen3-8B")
EMBED_URL = os.environ.get("LANGGRAPH_TEST_EMBED_URL", "http://localhost:30164/v1")
EMBED_MODEL = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


def _real_endpoints_available() -> bool:
    return _endpoint_alive(LLM_URL) and _endpoint_alive(EMBED_URL)


pytestmark = pytest.mark.skipif(
    not _real_endpoints_available(),
    reason=(
        "Real model endpoints unreachable. Open SSH tunnels:\n"
        " ssh -fN -L 30172:n0172:30000 -L 30164:n0164:30000 t2"
    ),
)


def _build_real_config() -> LangGraphCoeConfig:
    cfg = LangGraphCoeConfig.from_yaml()
    # Reranker stays off; the rollout uses the identity reranker in
    # ``cot.rerank_context`` (no reranker server here). top_k + other reranker /
    # search params come from config.yaml.
    cfg.reranker.enabled = log_config_override(
        "reranker.enabled",
        cfg.reranker.enabled,
        False,
        reason="stub integration has no reranker server; identity rerank path",
    )
    # Keep the search cheap and bounded for a smoke test: a single-step CoT rollout.
    cfg.search.mcts.max_simulation_depth = log_config_override(
        "search.mcts.max_simulation_depth",
        cfg.search.mcts.max_simulation_depth,
        1,
        reason="bounded integration smoke: one-step rollout",
    )
    return cfg


def _build_real_registry(cfg: LangGraphCoeConfig) -> RoleModelRegistry:
    # Keep every config.yaml generation knob (temperature, top_p, top_k, min_p,
    # penalties, seed, enable_thinking, …); override only the endpoint + token
    # budgets the single Qwen3-8B test deployment (32k context) requires.
    for name in list(cfg.llm.tiers):
        cfg.llm.tiers[name] = override_tier_endpoint(
            cfg,
            name,
            model_name=LLM_MODEL,
            api_base=LLM_URL,
            api_key="EMPTY",
            max_tokens=16384,
            max_input_tokens=32768,
            reason="integration endpoint: Qwen3-8B @ LLM_URL with a 32k context",
        )
    cfg.llm.api_key = "EMPTY"
    return RoleModelRegistry(cfg.llm)


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic stubs for the rollout's retrieval surfaces (reused from the
# integration shape so the rollout sees real-looking evidence).
# ──────────────────────────────────────────────────────────────────────────────


class _StubKGGraph:
    def __init__(self, articles: Sequence[str], triples: Sequence[str]) -> None:
        self._articles = list(articles)
        self._triples = list(triples)
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(
        self, state: Dict[str, Any], *_a: Any, **_kw: Any
    ) -> Dict[str, Any]:
        self.calls.append(state)
        return {"kg_articles": list(self._articles), "triples": list(self._triples)}


class _StubWebGraph:
    def __init__(self, results: Sequence[Dict[str, str]]) -> None:
        self._results = [dict(r) for r in results]
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(
        self, state: Dict[str, Any], *_a: Any, **_kw: Any
    ) -> Dict[str, Any]:
        self.calls.append(state)
        return {"results": [dict(r) for r in self._results]}


class _StubCorpusSearch:
    name = "corpus_search"

    def __init__(self, results: Sequence[str]) -> None:
        self._results = list(results)
        self.invocations: List[Any] = []

    async def ainvoke(self, payload: Any, *_a: Any, **_kw: Any) -> List[str]:
        self.invocations.append(payload)
        return list(self._results)


class _StubLinkEntities:
    name = "link_entities"

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping
        self.invocations: List[Any] = []

    async def ainvoke(
        self, inp: Any, *args: Any, **kwargs: Any
    ) -> List[Dict[str, str]]:
        self.invocations.append(inp)
        names = inp.get("entity_names", inp) if isinstance(inp, dict) else inp
        out: List[Dict[str, str]] = []
        for name in names or []:
            qid = self._mapping.get(str(name))
            if qid:
                out.append({"name": str(name), "qid": qid, "description": ""})
        return out


def _stub_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the rollout's external retrieval surfaces with deterministic stubs.

    ``MCTSGraph`` builds the **real** ``CoTGraph`` / ``MemoryUpdateGraph`` at
    construction time, so patching their module-level retrieval dependencies
    here keeps the whole rollout self-contained (no live Wikidata / Serper /
    FAISS), while every LLM role still runs against the real Qwen model.
    """
    from langgraph_coe.graphs import cot as cot_mod
    from langgraph_coe.graphs import memory_update as mem_mod

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
        results=["Paris has been the capital of France since 987 AD."]
    )
    link_stub = _StubLinkEntities(
        {"France": "Q142", "Paris": "Q90", "Eiffel Tower": "Q243"}
    )

    monkeypatch.setattr(
        cot_mod, "build_kg_search_graph", lambda *_a, **_kw: kg_stub, raising=False
    )
    monkeypatch.setattr(
        cot_mod, "build_web_research_graph", lambda *_a, **_kw: web_stub, raising=False
    )
    monkeypatch.setattr(cot_mod, "corpus_search", corpus_stub, raising=False)
    monkeypatch.setattr(mem_mod, "link_entities", link_stub, raising=False)


def _initial_state(question: str, **overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "question": question,
        "max_iterations": 1,
        "iteration": 0,
        "tree": {},
        "root_id": "root",
        "current_path": [],
        "expanded_node_ids": [],
        "simulation_result": {},
        "reward": 0.0,
        "new_raw_triples": [],
        "text_memory": ["France is a country located in Western Europe."],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "semantic_sufficiency_signals": 0,
        "iterations_without_improvement": 0,
        "best_value": 0.0,
        "final_answer": "",
    }
    state.update(overrides)
    return state


def _ntype(node: Dict[str, Any]) -> str:
    value = node.get("node_type")
    return getattr(value, "value", value)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_mcts_graph_runs_against_real_qwen(monkeypatch):
    """One full MCTS iteration completes end-to-end with the real Qwen3-8B model."""
    from langgraph_coe.graphs import mcts as mcts_mod

    cfg = _build_real_config()
    registry = _build_real_registry(cfg)
    _stub_retrieval(monkeypatch)

    graph = mcts_mod.build_mcts_graph(registry, cfg)

    initial = _initial_state(
        "What is the capital of France and one famous landmark there?"
    )
    final = await graph.ainvoke(initial, config={"recursion_limit": 75})
    print(final)

    # ── Final answer ─────────────────────────────────────────────────────────
    assert final.get("final_answer"), "MCTS must synthesize a non-empty final answer"
    answer = str(final["final_answer"]).lower()
    assert "paris" in answer, (
        f"final answer must reference Paris (the retrieval seed); got: {final['final_answer']!r}"
    )

    # ── Tree shape ─────────────────────────────────────────────────────────────
    tree = final.get("tree") or {}
    assert tree, "search tree must be populated"
    root = tree[final["root_id"]]
    assert _ntype(root) == "user_question"

    node_types = {nid: _ntype(n) for nid, n in tree.items()}
    assert "sub_qa" in node_types.values(), (
        "expansion must emit at least one SUB_QA node"
    )
    assert "final_answer" in node_types.values(), (
        "expansion must emit a FINAL_ANSWER node"
    )

    # Priors port from NODE_TYPE_PRIOR.
    for node in tree.values():
        nt = _ntype(node)
        if nt == "sub_qa":
            assert (
                node["prior"] == mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SUB_QA]
            )
        elif nt == "final_answer":
            assert (
                node["prior"]
                == mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.FINAL_ANSWER]
            )

    # ── Reward + backprop ─────────────────────────────────────────────────────
    assert -1.0 <= float(final["reward"]) <= 1.0, "reward must be normalized to [-1, 1]"
    path = final.get("current_path") or []
    assert path, "select/simulate must leave the evaluated path in state"
    for node_id in path:
        assert tree[node_id]["visits"] >= 1, (
            "backprop must increment visits along the path"
        )

    # One iteration ran and the loop terminated on the iteration cap.
    assert final["iteration"] == 1

    # ── Memory sync ────────────────────────────────────────────────────────────
    assert final.get("text_memory"), (
        "mem_update must leave non-empty consolidated memory"
    )
    # Per-iteration retrieval scratch is cleared after consumption.
    assert final.get("new_raw_triples") == []


async def test_mcts_terminal_final_answer_is_evaluated_without_expansion(monkeypatch):
    """A pre-seeded FINAL_ANSWER leaf is scored by the verifier, never re-expanded.

    This is the cheap path: no expansion generators or CoT rollout run, so the
    iteration is just (select → terminal) → evaluate (3 verifier views) →
    backprop → mem_update → synthesize.
    """
    from langgraph_coe.graphs import mcts as mcts_mod

    cfg = _build_real_config()
    registry = _build_real_registry(cfg)
    _stub_retrieval(monkeypatch)

    root_id = "root"
    final_id = "final-seed"
    seeded_tree = {
        root_id: {
            "node_id": root_id,
            "parent_id": None,
            "children_ids": [final_id],
            "node_type": mcts_mod.MCTSNodeType.USER_QUESTION,
            "content": {"question": "What is the capital of France?"},
            "visits": 5,
            "value": 2.0,
            "prior": 1.0,
        },
        final_id: {
            "node_id": final_id,
            "parent_id": root_id,
            "children_ids": [],
            "node_type": mcts_mod.MCTSNodeType.FINAL_ANSWER,
            "content": {
                "final_answer": "Paris is the capital of France.",
                "concise_answer": "Paris",
                "reasoning": "Paris has been the French capital for centuries.",
            },
            "visits": 0,
            "value": 0.0,
            "prior": mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.FINAL_ANSWER],
        },
    }

    graph = mcts_mod.build_mcts_graph(registry, cfg)
    final = await graph.ainvoke(
        _initial_state(
            "What is the capital of France?",
            tree=seeded_tree,
            root_id=root_id,
            text_memory=[
                "France is a country in Western Europe; its capital is Paris."
            ],
        ),
        config={"recursion_limit": 50},
    )
    print(final)

    # The terminal leaf was selected and never expanded.
    assert final["expanded_node_ids"] == []
    assert final["current_path"][-1] == final_id

    # It was still scored (verifier ran) and backpropagated onto the seeded path.
    assert -1.0 <= float(final["reward"]) <= 1.0
    tree = final["tree"]
    assert tree[final_id]["visits"] >= 1
    assert tree[root_id]["visits"] >= 6 # 5 seeded + 1 from this iteration

    # A synthesized answer is still produced.
    assert final.get("final_answer")
    assert "paris" in str(final["final_answer"]).lower()
