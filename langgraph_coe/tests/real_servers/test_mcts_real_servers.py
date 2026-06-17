"""Full-stack MCTSGraph integration — every subsystem is **real**.

The sibling ``test_mcts_integration.py`` keeps the rollout's external retrieval
surfaces (KG / Web / corpus / ``link_entities``) as deterministic stubs so it can
run against a lone Qwen deployment. This module is the MCTS analog of
``phase2/test_cot_real_servers.py``: it wires the **entire** search loop to the
production endpoints declared in ``langgraph_coe/config.yaml`` and runs with
**no stubs at all**:

  - LLM SGLang ``Qwen`` @ ``n0152:30000`` (all role tiers)
  - Embedder SGLang ``Qwen3-Embedding`` @ ``n0152:30001`` (corpus queries)
  - Reranker SGLang ``Qwen3-Reranker`` @ ``n0997:30000``
  - Wikidata QEndpoint SPARQL @ ``n0162:1234``
  - Corpus local 99 GB FAISS index (``retriever.corpus.index_path``)

It is the only test that drives the real ``select → expand → simulate (CoT
rollout) → evaluate (3 verifier views) → backprop → mem_update → route →
synthesize`` cycle against live models, so it is the one that catches:

  * multi-turn SGLang 400s when reasoning blocks are not stripped between turns
    inside the CoT rollout (``_reasoning_middleware.strip_reasoning_middleware``);
  * structured-output / tool-call schema drift on the real model for the
    subquestion / answer / self-correction / verifier / synthesis roles;
  * KG ↔ MemoryUpdate ↔ corpus ↔ tree state-shape mismatches the stubs hide;
  * reward-normalization / backprop / dict-merge regressions under real reward.

It exercises every distinct MCTS node path:

  * ``test_mcts_full_stack_real_servers`` — multi-iteration loop with a real CoT
    rollout (root expansion + ``_gen_final`` + rollout + 3-view verifier).
  * ``test_mcts_self_correction_real_servers`` — a seeded ``SUB_QA`` leaf so
    ``expand`` dispatches the ``_gen_self_correct`` branch against the live model.
  * ``test_mcts_terminal_final_answer_real_servers`` — a seeded ``FINAL_ANSWER``
    leaf scored by the verifier with **no** expansion / rollout (the cheap path).

Because the corpus index is ~99 GB it must be loaded on a compute node with
enough RAM that can also reach every endpoint (the fixture is module-scoped so
the index loads **once** for all three tests). Run e.g.::

    # files are shared across nodes; n0162 hosts QEndpoint and has the RAM
    ssh n0162 'cd /gpfs/projects/uonlp/hieum/coe && \
        uv run pytest langgraph_coe/tests/phase3/test_mcts_real_servers.py -v -s'

The whole module skips cleanly when any endpoint or the corpus is unreachable.

Optional env overrides (default to config.yaml; point at SSH tunnels for CI)::

    LANGGRAPH_TEST_LLM_URL default config.yaml heavy-tier api_base
    LANGGRAPH_TEST_EMBED_URL default config.yaml retriever embedder url
    LANGGRAPH_TEST_RERANKER_URL default config.yaml reranker url
    LANGGRAPH_TEST_RERANKER_MODEL default config.yaml reranker model_name
    LANGGRAPH_TEST_SPARQL_URL default config.yaml wikidata sparql_endpoint
    LANGGRAPH_TEST_MCTS_ITERS hard cap on MCTS iterations (default 2)
    LANGGRAPH_TEST_MCTS_SIM_DEPTH per-rollout CoT depth (default 1)
    API_KEY / OPENAI_API_KEY LLM/embedder/reranker auth (repo-root .env)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import httpx
import networkx as nx
import pytest
import pytest_asyncio

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.llm import RoleModelRegistry

from .._fixtures import log_config_override

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints (config.yaml is the source of truth; env vars repoint to tunnels)
# ──────────────────────────────────────────────────────────────────────────────

_CFG_DEFAULTS = LangGraphCoeConfig.from_yaml()

LLM_URL = os.environ.get(
    "LANGGRAPH_TEST_LLM_URL", _CFG_DEFAULTS.llm.tiers["heavy"].api_base
)
EMBED_URL = os.environ.get(
    "LANGGRAPH_TEST_EMBED_URL", _CFG_DEFAULTS.retriever.corpus.embedder.url
)
RERANKER_URL = os.environ.get("LANGGRAPH_TEST_RERANKER_URL", _CFG_DEFAULTS.reranker.url)
RERANKER_MODEL = os.environ.get(
    "LANGGRAPH_TEST_RERANKER_MODEL", _CFG_DEFAULTS.reranker.model_name
)
SPARQL_URL = os.environ.get(
    "LANGGRAPH_TEST_SPARQL_URL", _CFG_DEFAULTS.wikidata.sparql_endpoint
)
MCTS_ITERS = int(os.environ.get("LANGGRAPH_TEST_MCTS_ITERS", "2"))
MCTS_SIM_DEPTH = int(os.environ.get("LANGGRAPH_TEST_MCTS_SIM_DEPTH", "1"))

# Verifiable, KG-anchored question (proven entity in test_kg_search_integration:
# University of Oregon → Q766145, founded 1876, located in Oregon).
_QUERY = (
    "When was the University of Oregon founded and in which US state is it located?"
)


# ──────────────────────────────────────────────────────────────────────────────
# Liveness probes — module skips unless the entire stack is up
# ──────────────────────────────────────────────────────────────────────────────


def _models_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=8.0) as client:
            return client.get(f"{url.rstrip('/')}/models").status_code == 200
    except Exception:
        return False


def _reranker_alive(url: str, model: str) -> bool:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url.rstrip('/')}/rerank",
                json={"model": model, "query": "ping", "documents": ["a", "b"]},
                headers={"Authorization": "Bearer EMPTY"},
            )
            return resp.status_code == 200
    except Exception:
        return False


def _sparql_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                url,
                params={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
                headers={"Accept": "application/sparql-results+json"},
            )
            return resp.status_code == 200 and bool(
                resp.json().get("results", {}).get("bindings")
            )
    except Exception:
        return False


def _corpus_present() -> bool:
    return os.path.isfile(_CFG_DEFAULTS.retriever.corpus.index_path)


_LLM_UP = _models_alive(LLM_URL)
_EMBED_UP = _models_alive(EMBED_URL)
_RERANKER_UP = _reranker_alive(RERANKER_URL, RERANKER_MODEL)
_SPARQL_UP = _sparql_alive(SPARQL_URL)
_CORPUS_UP = _corpus_present()
_STACK_UP = _LLM_UP and _EMBED_UP and _RERANKER_UP and _SPARQL_UP and _CORPUS_UP

_skip_reason = (
    "Full MCTS stack unavailable — "
    f"LLM({LLM_URL})={_LLM_UP}, embedder({EMBED_URL})={_EMBED_UP}, "
    f"reranker({RERANKER_URL})={_RERANKER_UP}, SPARQL({SPARQL_URL})={_SPARQL_UP}, "
    f"corpus({_CFG_DEFAULTS.retriever.corpus.index_path})={_CORPUS_UP}."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow_integration,
    pytest.mark.requires_wikidata,
    pytest.mark.skipif(not _STACK_UP, reason=_skip_reason),
    # Pin all three tests to ONE module-scoped event loop so the module-scoped
    # ``_wire_runtime`` fixture (which loads the 99 GB index once) and the
    # Wikidata client's ``httpx.AsyncClient`` live on the same loop for the whole
    # module. Without this each async test gets its own loop, the client is
    # reused across closed loops, and its connection pool raises
    # "Event loop is closed" when finally garbage-collected at process exit.
    pytest.mark.asyncio(loop_scope="module"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Config / runtime wiring
# ──────────────────────────────────────────────────────────────────────────────


def _build_config() -> LangGraphCoeConfig:
    """config.yaml, with only endpoint/auth/search-bound deviations (all logged)."""
    cfg = LangGraphCoeConfig.from_yaml()

    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or cfg.llm.api_key
        or "EMPTY"
    )
    cfg.llm.api_key = api_key

    # Point every LLM tier at LLM_URL (no-op when it already equals config.yaml).
    for name, tier in cfg.llm.tiers.items():
        tier.api_base = log_config_override(
            f"llm.tiers.{name}.api_base",
            tier.api_base,
            LLM_URL,
            reason="LANGGRAPH_TEST_LLM_URL endpoint for this run",
        )
        tier.api_key = api_key

    # Retrieval / reranker / wikidata endpoints (logged only if they deviate).
    cfg.retriever.corpus.embedder.url = log_config_override(
        "retriever.corpus.embedder.url",
        cfg.retriever.corpus.embedder.url,
        EMBED_URL,
        reason="LANGGRAPH_TEST_EMBED_URL endpoint for this run",
    )
    cfg.retriever.corpus.embedder.api_key = (
        cfg.retriever.corpus.embedder.api_key or api_key
    )
    cfg.reranker.url = log_config_override(
        "reranker.url",
        cfg.reranker.url,
        RERANKER_URL,
        reason="LANGGRAPH_TEST_RERANKER_URL endpoint for this run",
    )
    cfg.reranker.api_key = cfg.reranker.api_key or api_key
    cfg.wikidata.sparql_endpoint = log_config_override(
        "wikidata.sparql_endpoint",
        cfg.wikidata.sparql_endpoint,
        SPARQL_URL,
        reason="LANGGRAPH_TEST_SPARQL_URL endpoint for this run",
    )

    # Bound the live search so a thinking-enabled run finishes in reasonable wall
    # time while still exercising every node path: at least one full iteration
    # (expand → simulate rollout → evaluate → backprop → mem_update) plus the
    # terminal route into synthesize.
    cfg.search.mcts.num_iterations = log_config_override(
        "search.mcts.num_iterations",
        cfg.search.mcts.num_iterations,
        MCTS_ITERS,
        reason="bound live MCTS wall time (still >=1 full iteration)",
    )
    # Floor must not exceed the hard cap, or the loop can never reach the cap.
    cfg.search.mcts.min_iterations = log_config_override(
        "search.mcts.min_iterations",
        cfg.search.mcts.min_iterations,
        0,
        reason="let the hard iteration cap terminate the bounded run",
    )
    cfg.search.mcts.max_simulation_depth = log_config_override(
        "search.mcts.max_simulation_depth",
        cfg.search.mcts.max_simulation_depth,
        MCTS_SIM_DEPTH,
        reason="bound per-rollout CoT depth for a live smoke run",
    )
    # Let the root expansion already emit a FINAL_ANSWER child so the live run
    # exercises ``_gen_final`` (not just ``_gen_subqa``) on iteration 1.
    cfg.search.mcts.final_answer_min_depth = log_config_override(
        "search.mcts.final_answer_min_depth",
        cfg.search.mcts.final_answer_min_depth,
        0,
        reason="exercise _gen_final from the root within the bounded run",
    )
    # Disable the LLM/Wikidata Redis cache so the test exercises live calls, not
    # a warm cache that would mask endpoint regressions.
    cfg.cache.enabled = log_config_override(
        "cache.enabled",
        cfg.cache.enabled,
        False,
        reason="exercise live endpoints, not a warm cache",
    )
    return cfg


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _wire_runtime():
    """Initialize the real Wikidata client + FAISS retriever once for the module.

    Mirrors ``system._init_runtime`` without the web tool (web_search stays
    disabled in config.yaml) so the only retrieval subsystems are KG + corpus.
    Module-scoped: the ~99 GB FAISS index is loaded a single time and shared by
    every test in this file.

    Async + ``loop_scope="module"`` so teardown can ``await`` the Wikidata
    client's ``aclose`` on the **same** loop the tests issued requests on,
    tearing the ``httpx`` connection pool down cleanly instead of letting it be
    garbage-collected after the loop closes ("Event loop is closed").
    """
    from langgraph_coe.tools import retrieval as rmod
    from langgraph_coe.tools import wikidata as wd_mod
    from langgraph_coe.tools.retrieval import init_retrieval_pipeline
    from langgraph_coe.tools.wikidata import init_wikidata, reset_wikidata_session

    cfg = _build_config()

    wd_mod._wikidata_client = None
    wd_mod._wikidata_config = None
    rmod._retriever_instance = None
    rmod._reranker_config = None

    init_wikidata(cfg.wikidata)
    reset_wikidata_session()
    print(
        f"\n[setup] loading FAISS corpus {cfg.retriever.corpus.index_path!r} "
        "(this can take minutes for the 99 GB index)…",
        flush=True,
    )
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)
    print("[setup] corpus loaded.", flush=True)

    yield cfg

    client = wd_mod._wikidata_client
    if client is not None:
        try:
            await client.aclose()
        except Exception: # noqa: BLE001 — teardown is best-effort
            pass
    wd_mod._wikidata_client = None
    wd_mod._wikidata_config = None
    rmod._retriever_instance = None
    rmod._reranker_config = None


# ──────────────────────────────────────────────────────────────────────────────
# Verbose tracing
# ──────────────────────────────────────────────────────────────────────────────


def _truncate(value: Any, *, max_len: int = 1000) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return f"{value[:max_len]}… [{len(value)} chars]"
    if isinstance(value, list):
        return [_truncate(v, max_len=max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate(v, max_len=max_len) for k, v in value.items()}
    return value


def _print_block(title: str, payload: Any) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    if isinstance(payload, (dict, list)):
        text = json.dumps(_truncate(payload), indent=2, ensure_ascii=False, default=str)
    else:
        text = str(payload)
    print(text, flush=True)


def _ntype(node: Mapping[str, Any]) -> str:
    value = node.get("node_type")
    return getattr(value, "value", value)


def _tree_summary(tree: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Compact view of the search tree: per-type counts + (visits, value) per node."""
    by_type: Dict[str, int] = {}
    nodes: Dict[str, Any] = {}
    for nid, node in (tree or {}).items():
        nt = _ntype(node)
        by_type[nt] = by_type.get(nt, 0) + 1
        nodes[nid] = {
            "type": nt,
            "parent": node.get("parent_id"),
            "children": node.get("children_ids"),
            "visits": node.get("visits"),
            "value": round(float(node.get("value", 0.0) or 0.0), 3),
            "prior": node.get("prior"),
            "content": _truncate(node.get("content"), max_len=240),
        }
    return {"node_count": len(tree or {}), "by_type": by_type, "nodes": nodes}


def _state_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact, printable view of an MCTS state snapshot (tree + memory summarized)."""
    graph_mem = state.get("graph_memory")
    return {
        "iteration": state.get("iteration"),
        "max_iterations": state.get("max_iterations"),
        "current_path": state.get("current_path"),
        "expanded_node_ids": state.get("expanded_node_ids"),
        "reward": state.get("reward"),
        "best_value": state.get("best_value"),
        "iterations_without_improvement": state.get("iterations_without_improvement"),
        "new_raw_triples": state.get("new_raw_triples"),
        "simulation_result": _truncate(state.get("simulation_result")),
        "text_memory": state.get("text_memory"),
        "graph_memory": (
            f"<DiGraph nodes={graph_mem.number_of_nodes()} "
            f"edges={graph_mem.number_of_edges()}>"
            if isinstance(graph_mem, nx.DiGraph)
            else graph_mem
        ),
        "entity_dict_keys": list((state.get("entity_dict") or {}).keys()),
        "tree": _tree_summary(state.get("tree") or {}),
        "final_answer": state.get("final_answer"),
    }


async def _run_mcts_verbose(
    graph: Any, initial: Mapping[str, Any], *, recursion_limit: int
) -> Dict[str, Any]:
    """Stream the MCTS graph in values-mode, printing each superstep snapshot.

    ``stream_mode="values"`` yields the full accumulated state after every
    superstep (the ``dict_merge`` / ``append_or_clear`` reducers already
    applied), so the last chunk is the authoritative final state.
    """
    final: Dict[str, Any] = dict(initial)
    step = 0
    async for state in graph.astream(
        initial, stream_mode="values", config={"recursion_limit": recursion_limit}
    ):
        _print_block(f"SUPERSTEP {step}", _state_summary(state))
        final = state
        step += 1
    return final


def _initial_mcts_state(
    question: str, cfg: LangGraphCoeConfig, **overrides: Any
) -> Dict[str, Any]:
    """Fresh ``MCTSState`` (``select`` seeds the root when the tree is empty)."""
    state: Dict[str, Any] = {
        "question": question,
        "max_iterations": int(cfg.search.mcts.num_iterations),
        "iteration": 0,
        "tree": {},
        "root_id": "root",
        "current_path": [],
        "expanded_node_ids": [],
        "simulation_result": {},
        "reward": 0.0,
        "new_raw_triples": [],
        "text_memory": [],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "semantic_sufficiency_signals": 0,
        "iterations_without_improvement": 0,
        "best_value": 0.0,
        "final_answer": "",
    }
    state.update(overrides)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_mcts_full_stack_real_servers(_wire_runtime):
    """End-to-end MCTS loop against live KG + corpus + reranker + LLM."""
    from langgraph_coe.graphs import mcts as mcts_mod

    cfg = _wire_runtime
    registry = RoleModelRegistry(cfg.llm)

    _print_block(
        "integration setup",
        {
            "llm_url": LLM_URL,
            "embed_url": EMBED_URL,
            "reranker_url": RERANKER_URL,
            "sparql_url": SPARQL_URL,
            "corpus_index": cfg.retriever.corpus.index_path,
            "num_iterations": cfg.search.mcts.num_iterations,
            "max_simulation_depth": cfg.search.mcts.max_simulation_depth,
            "final_answer_min_depth": cfg.search.mcts.final_answer_min_depth,
            "reranker_enabled": cfg.reranker.enabled,
            "web_enabled": cfg.web_search.enabled,
            "query": _QUERY,
        },
    )

    graph = mcts_mod.build_mcts_graph(registry, cfg)
    initial = _initial_mcts_state(_QUERY, cfg)

    final = await _run_mcts_verbose(
        graph, initial, recursion_limit=int(cfg.search.mcts.recursion_limit)
    )

    _print_block(
        "FINAL STATE (summary)",
        {
            "final_answer": final.get("final_answer"),
            "iteration": final.get("iteration"),
            "reward": final.get("reward"),
            "best_value": final.get("best_value"),
            "text_memory": final.get("text_memory"),
            "entity_dict_keys": list((final.get("entity_dict") or {}).keys()),
            "tree": _tree_summary(final.get("tree") or {}),
        },
    )

    # 1. A non-empty final answer is synthesized.
    answer = str(final.get("final_answer") or "")
    assert answer.strip(), "MCTS must synthesize a non-empty final answer"

    # 2. It grounds on the retrieved facts (founding year or the state).
    low = answer.lower()
    assert ("1876" in low) or ("oregon" in low), (
        f"final answer should reference the founding year (1876) or state (Oregon); "
        f"got: {answer!r}"
    )

    # 3. The tree was built with the planned node types + ported priors.
    tree = final.get("tree") or {}
    assert tree, "search tree must be populated"
    root = tree[final["root_id"]]
    assert _ntype(root) == "user_question"
    node_types = {_ntype(n) for n in tree.values()}
    assert "sub_qa" in node_types, (
        "expansion/rollout must emit at least one SUB_QA node"
    )
    assert "final_answer" in node_types, "a FINAL_ANSWER node must exist"
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

    # 4. Reward is normalized and backprop landed on the evaluated path.
    assert -1.0 <= float(final["reward"]) <= 1.0, "reward must be normalized to [-1, 1]"
    path = final.get("current_path") or []
    assert path, "select/simulate must leave the evaluated path in state"
    for node_id in path:
        assert tree[node_id]["visits"] >= 1, (
            "backprop must increment visits along the path"
        )

    # 5. At least one full iteration ran and the loop terminated on the cap.
    assert int(final.get("iteration", 0)) >= 1
    assert int(final.get("iteration", 0)) <= int(cfg.search.mcts.num_iterations)

    # 6. Cross-iteration memory was consolidated by MemoryUpdateGraph (real KG link).
    assert final.get("text_memory"), (
        "mem_update must leave non-empty consolidated textual memory"
    )

    # 7. Per-iteration retrieval scratch was cleared after consumption.
    assert final.get("new_raw_triples") == []


async def test_mcts_self_correction_real_servers(_wire_runtime):
    """A seeded ``SUB_QA`` leaf drives the live ``_gen_self_correct`` expansion path.

    ``select`` descends root → the lone SUB_QA child (its only non-terminal
    leaf), so ``expand`` dispatches the SUB_QA branch — ``_gen_subqa`` **and**
    ``_gen_self_correct`` — exercising the self-corrector role against the real
    model, a path the full-run test does not deterministically hit on iteration 1.
    """
    from langgraph_coe.graphs import mcts as mcts_mod

    cfg = _wire_runtime
    registry = RoleModelRegistry(cfg.llm)

    root_id = "root"
    subqa_id = "subqa-seed"
    seeded_tree = {
        root_id: {
            "node_id": root_id,
            "parent_id": None,
            "children_ids": [subqa_id],
            "node_type": mcts_mod.MCTSNodeType.USER_QUESTION,
            "content": {"question": _QUERY},
            "visits": 1,
            "value": 0.0,
            "prior": 1.0,
        },
        subqa_id: {
            "node_id": subqa_id,
            "parent_id": root_id,
            "children_ids": [],
            "node_type": mcts_mod.MCTSNodeType.SUB_QA,
            "content": {
                "sub_question": "In what year was the University of Oregon founded?",
                "sub_answer": "The University of Oregon was founded in 1876.",
            },
            "visits": 0,
            "value": 0.0,
            "prior": mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SUB_QA],
        },
    }

    graph = mcts_mod.build_mcts_graph(registry, cfg)
    initial = _initial_mcts_state(
        _QUERY,
        cfg,
        max_iterations=1,
        tree=seeded_tree,
        root_id=root_id,
        text_memory=[
            "The University of Oregon is a public research university in Eugene, Oregon."
        ],
    )

    final = await _run_mcts_verbose(
        graph, initial, recursion_limit=int(cfg.search.mcts.recursion_limit)
    )

    tree = final.get("tree") or {}
    _print_block("self-correction FINAL tree", _tree_summary(tree))

    # The seeded SUB_QA leaf was selected and expanded.
    assert subqa_id in (final.get("current_path") or []), (
        "select must descend to the seeded SUB_QA leaf"
    )
    assert final.get("expanded_node_ids"), "the SUB_QA leaf must have been expanded"

    # Expansion produced a SELF_CORRECTED child (the live self-corrector ran).
    sc_nodes = [
        n
        for n in tree.values()
        if _ntype(n) == "self_corrected" and n.get("parent_id") == subqa_id
    ]
    assert sc_nodes, "expand of a SUB_QA leaf must emit a SELF_CORRECTED child"
    assert (
        sc_nodes[0]["prior"]
        == mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SELF_CORRECTED]
    )
    assert str(sc_nodes[0]["content"].get("sub_answer", "")).strip(), (
        "self-corrector must return a non-empty refined answer"
    )

    # The iteration completed: reward normalized, backprop applied, answer synthesized.
    assert -1.0 <= float(final["reward"]) <= 1.0
    assert int(final.get("iteration", 0)) == 1
    assert str(final.get("final_answer") or "").strip()


async def test_mcts_terminal_final_answer_real_servers(_wire_runtime):
    """A pre-seeded ``FINAL_ANSWER`` leaf is scored by the verifier, never expanded.

    This is the cheap path with no expansion generators or CoT rollout: the
    iteration is just (select → terminal) → evaluate (3 verifier views) →
    backprop → mem_update → synthesize. It validates the real verifier and
    synthesizer roles and the terminal-leaf short-circuit in ``expand``.
    """
    from langgraph_coe.graphs import mcts as mcts_mod

    cfg = _wire_runtime
    registry = RoleModelRegistry(cfg.llm)

    root_id = "root"
    final_id = "final-seed"
    seeded_tree = {
        root_id: {
            "node_id": root_id,
            "parent_id": None,
            "children_ids": [final_id],
            "node_type": mcts_mod.MCTSNodeType.USER_QUESTION,
            "content": {"question": _QUERY},
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
                "final_answer": "The University of Oregon was founded in 1876 and is located in the state of Oregon.",
                "concise_answer": "Founded 1876, in Oregon.",
                "reasoning": "Wikidata records inception 1876 (P571) and located in Oregon (P131).",
            },
            "visits": 0,
            "value": 0.0,
            "prior": mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.FINAL_ANSWER],
        },
    }

    graph = mcts_mod.build_mcts_graph(registry, cfg)
    initial = _initial_mcts_state(
        _QUERY,
        cfg,
        max_iterations=1,
        tree=seeded_tree,
        root_id=root_id,
        text_memory=["The University of Oregon was founded in 1876 in Eugene, Oregon."],
    )

    final = await _run_mcts_verbose(
        graph, initial, recursion_limit=int(cfg.search.mcts.recursion_limit)
    )

    # The terminal leaf was selected and never expanded.
    assert final.get("expanded_node_ids") == [], (
        "FINAL_ANSWER leaves must not be expanded"
    )
    assert (final.get("current_path") or [])[-1] == final_id

    # It was still scored (verifier ran) and backpropagated onto the seeded path.
    assert -1.0 <= float(final["reward"]) <= 1.0
    tree = final.get("tree") or {}
    assert tree[final_id]["visits"] >= 1
    assert tree[root_id]["visits"] >= 6 # 5 seeded + 1 from this iteration

    # A synthesized answer is still produced and grounds on the seeded facts.
    answer = str(final.get("final_answer") or "")
    assert answer.strip(), "synthesize must produce a non-empty final answer"
    low = answer.lower()
    assert ("1876" in low) or ("oregon" in low), (
        f"synthesized answer should reference 1876 or Oregon; got: {answer!r}"
    )
