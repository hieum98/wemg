"""Full-stack CoTGraph integration — every retrieval subsystem is **real**.

Unlike ``test_cot_integration.py`` (which stubs KG/web/corpus to isolate the
Phase 2 control flow against a lone Qwen3-8B deployment), this test wires the
CoT loop to the *production* endpoints declared in ``langgraph_coe/config.yaml``
and runs with **no stubs**:

  - LLM            SGLang ``Qwen/Qwen3.5-4B`` @ ``n0152:30000``  (all tiers)
  - Embedder       SGLang ``Qwen3-Embedding-4B`` @ ``n0152:30001`` (corpus queries)
  - Reranker       SGLang ``Qwen3-Reranker-4B`` @ ``n0997:30000``
  - Wikidata       QEndpoint SPARQL @ ``n0162:1234``
  - Corpus         local 99 GB FAISS index (``retriever.corpus.index_path``)

It is the only test that exercises the real ``gen_subq → kg_one + corpus_join →
rerank → extract_relevant → gen_subanswers → mem_update → increment → gen_subq``
cycle against live models, so it is the one that catches:

  * multi-turn SGLang 400s when reasoning blocks are not stripped between turns
    (see ``_reasoning_middleware.strip_reasoning_middleware``);
  * structured-output / tool-call schema drift on the real model;
  * KG ↔ MemoryUpdate ↔ corpus state-shape mismatches the stubs paper over.

Because the corpus index is ~99 GB it must be loaded on a compute node with
enough RAM — run from a node that can reach every endpoint, e.g.::

    # files are shared across nodes; n0162 hosts QEndpoint and has the RAM
    ssh n0162 'cd /gpfs/projects/uonlp/hieum/wemg && \
        uv run pytest langgraph_coe/tests/phase2/test_cot_real_servers.py -v -s'

The whole module skips cleanly when any endpoint or the corpus is unreachable.

Optional env overrides (default to config.yaml; point at SSH tunnels for CI)::

    LANGGRAPH_TEST_LLM_URL        default ``http://n0152:30000/v1``
    LANGGRAPH_TEST_EMBED_URL      default config.yaml retriever embedder url
    LANGGRAPH_TEST_RERANKER_URL   default config.yaml reranker url
    LANGGRAPH_TEST_SPARQL_URL     default config.yaml wikidata sparql_endpoint
    LANGGRAPH_TEST_COT_DEPTH      max CoT depth for the run (default 2)
    API_KEY / OPENAI_API_KEY      LLM/embedder/reranker auth (repo-root .env)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import httpx
import networkx as nx
import pytest

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
COT_DEPTH = int(os.environ.get("LANGGRAPH_TEST_COT_DEPTH", "2"))

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
    "Full CoT stack unavailable — "
    f"LLM({LLM_URL})={_LLM_UP}, embedder({EMBED_URL})={_EMBED_UP}, "
    f"reranker({RERANKER_URL})={_RERANKER_UP}, SPARQL({SPARQL_URL})={_SPARQL_UP}, "
    f"corpus({_CFG_DEFAULTS.retriever.corpus.index_path})={_CORPUS_UP}."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow_integration,
    pytest.mark.requires_wikidata,
    pytest.mark.skipif(not _STACK_UP, reason=_skip_reason),
]


# ──────────────────────────────────────────────────────────────────────────────
# Config / runtime wiring
# ──────────────────────────────────────────────────────────────────────────────


def _build_config() -> LangGraphCoeConfig:
    """config.yaml, with only endpoint/auth/depth deviations (all logged)."""
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

    # Cap the loop so a live, thinking-enabled run finishes in reasonable wall
    # time while still crossing the iteration boundary (depth>=1 exercises the
    # turn-2 reasoning-strip path that single-shot tests miss).
    cfg.search.cot.max_depth = log_config_override(
        "search.cot.max_depth",
        cfg.search.cot.max_depth,
        COT_DEPTH,
        reason="bound live multi-iteration runtime (still >=2 to cross turn-2)",
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


@pytest.fixture
def _wire_runtime():
    """Initialize the real Wikidata client + FAISS retriever; tear down after.

    Mirrors ``system._init_runtime`` but without the web tool (web_search stays
    disabled in config.yaml) so the only retrieval subsystems are KG + corpus.
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


def _state_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact, printable view of a CoT state snapshot (graphs summarized)."""
    graph_mem = state.get("graph_memory")
    return {
        "depth": state.get("depth"),
        "is_answerable": state.get("is_answerable"),
        "subquestions": state.get("subquestions"),
        "subquestion_needs_kg": state.get("subquestion_needs_kg"),
        "retrieved_raw_context": state.get("retrieved_raw_context"),
        "retrieved_raw_triples": state.get("retrieved_raw_triples"),
        "reranked_context": state.get("reranked_context"),
        "extracted_facts": state.get("extracted_facts"),
        "current_subanswers": state.get("current_subanswers"),
        "text_memory": state.get("text_memory"),
        "graph_memory": (
            f"<DiGraph nodes={graph_mem.number_of_nodes()} "
            f"edges={graph_mem.number_of_edges()}>"
            if isinstance(graph_mem, nx.DiGraph)
            else graph_mem
        ),
        "entity_dict_keys": list((state.get("entity_dict") or {}).keys()),
        "final_answer": state.get("final_answer"),
    }


async def _run_cot_verbose(
    graph: Any, initial: Mapping[str, Any], *, recursion_limit: int
) -> Dict[str, Any]:
    """Stream the CoT graph in values-mode, printing each superstep snapshot.

    ``stream_mode="values"`` yields the full accumulated state after every
    superstep (reducers already applied), so the last chunk is the authoritative
    final state — unlike ``"updates"``, which would force us to re-implement the
    append/Clear reducers by hand.
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


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


async def test_cot_full_stack_real_servers(_wire_runtime):
    """End-to-end CoT loop against live KG + corpus + reranker + LLM."""
    from langgraph_coe.graphs import cot as cot_mod

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
            "max_depth": cfg.search.cot.max_depth,
            "reranker_enabled": cfg.reranker.enabled,
            "web_enabled": cfg.web_search.enabled,
            "query": _QUERY,
        },
    )

    graph = cot_mod.build_cot_graph(registry, cfg)

    initial_state: Dict[str, Any] = {
        "question": _QUERY,
        "max_depth": int(cfg.search.cot.max_depth),
        "depth": 0,
        "is_answerable": False,
        "subquestions": [],
        "subquestion_needs_kg": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "extracted_facts": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": [],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "final_answer": "",
    }

    final = await _run_cot_verbose(
        graph,
        initial_state,
        recursion_limit=int(cfg.search.cot.recursion_limit),
    )

    _print_block(
        "FINAL STATE (summary)",
        {
            "final_answer": final.get("final_answer"),
            "num_iterations": len(final.get("iteration_history") or []),
            "text_memory": final.get("text_memory"),
            "entity_dict_keys": list((final.get("entity_dict") or {}).keys()),
            "graph_edges": (
                final.get("graph_memory").number_of_edges()
                if isinstance(final.get("graph_memory"), nx.DiGraph)
                else None
            ),
            "iteration_history": final.get("iteration_history"),
        },
    )

    # 1. A non-empty final answer is produced.
    answer = str(final.get("final_answer") or "")
    assert answer.strip(), "CoT must produce a non-empty final answer"

    # 2. It grounds on the retrieved facts (founding year or the state).
    low = answer.lower()
    assert ("1876" in low) or ("oregon" in low), (
        f"final answer should reference the founding year (1876) or state (Oregon); "
        f"got: {answer!r}"
    )

    # 3. The loop actually iterated (decomposition happened before synthesis).
    history = final.get("iteration_history") or []
    assert history, "expected at least one CoT iteration before terminating"
    for entry in history:
        assert isinstance(entry.get("subquestions"), list)
        assert isinstance(entry.get("subanswers"), list)

    # 4. Cross-iteration memory was populated by MemoryUpdateGraph (real KG link).
    assert final.get("text_memory"), (
        "MemoryUpdateGraph should have written textual memory from subanswers"
    )

    # 5. Per-iteration scratch was reset by the Clear reducer at loop end.
    assert final.get("retrieved_raw_context") == []
    assert final.get("retrieved_raw_triples") == []
