"""Full-stack **whole-system** integration — ``system.answer()`` end-to-end.

Where ``phase2/test_cot_real_servers.py`` and ``phase3/test_mcts_real_servers.py``
drive the compiled CoT / MCTS graphs directly, this module exercises the public
entrypoint :func:`langgraph_coe.system.answer` — the true top of the stack:

    answer(question, cfg)
      → _init_runtime         (cache / wikidata / web / retrieval init)
      → reset_*_session       (per-question ContextVar resets)
      → build_<strategy>_graph + _initial_<strategy>_state
      → graph.ainvoke(...)
      → AnswerResult.from_state(...)   (public envelope)

It runs with **no stubs** against the production endpoints in
``langgraph_coe/config.yaml`` and verifies the system works correctly through
**both** strategies (``cot`` and ``mcts``), proving the orchestrator's strategy
dispatch, runtime wiring, and result adaptation all hold against live models:

  - LLM            SGLang ``Qwen`` @ ``n0152:30000`` (all role tiers)
  - Embedder       SGLang ``Qwen3-Embedding`` @ ``n0152:30001`` (corpus queries)
  - Reranker       SGLang ``Qwen3-Reranker`` @ ``n0997:30000``
  - Wikidata       QEndpoint SPARQL @ ``n0162:1234``
  - Corpus         local 99 GB FAISS index (``retriever.corpus.index_path``)

The ~99 GB FAISS index is loaded **once** by a module-scoped fixture; the
fixture then neutralises the in-``answer()`` retrieval/wikidata re-init so each
strategy's ``answer()`` call reuses the already-loaded retriever + client rather
than reloading the index. All tests share one module-scoped event loop so the
Wikidata client's ``httpx.AsyncClient`` lives on a single loop and is closed
cleanly at teardown.

Run on a node with the RAM + endpoint reach (files are shared across nodes)::

    ssh n0162 'cd /gpfs/projects/uonlp/hieum/wemg && \
        uv run pytest langgraph_coe/tests/phase4/test_system_real_servers.py -v -s'

The whole module skips cleanly when any endpoint or the corpus is unreachable.

Optional env overrides (default to config.yaml; point at SSH tunnels for CI)::

    LANGGRAPH_TEST_LLM_URL        default config.yaml heavy-tier api_base
    LANGGRAPH_TEST_EMBED_URL      default config.yaml retriever embedder url
    LANGGRAPH_TEST_RERANKER_URL   default config.yaml reranker url
    LANGGRAPH_TEST_RERANKER_MODEL default config.yaml reranker model_name
    LANGGRAPH_TEST_SPARQL_URL     default config.yaml wikidata sparql_endpoint
    LANGGRAPH_TEST_COT_DEPTH      max CoT depth for the cot run (default 2)
    LANGGRAPH_TEST_MCTS_ITERS     hard cap on MCTS iterations (default 2)
    LANGGRAPH_TEST_MCTS_SIM_DEPTH per-rollout CoT depth for mcts (default 1)
    API_KEY / OPENAI_API_KEY      LLM/embedder/reranker auth (repo-root .env)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from langgraph_coe.config import LangGraphCoeConfig

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
MCTS_ITERS = int(os.environ.get("LANGGRAPH_TEST_MCTS_ITERS", "2"))
MCTS_SIM_DEPTH = int(os.environ.get("LANGGRAPH_TEST_MCTS_SIM_DEPTH", "1"))

# Verifiable, KG-anchored question (University of Oregon → Q766145, founded
# 1876, located in Oregon).
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
    "Full system stack unavailable — "
    f"LLM({LLM_URL})={_LLM_UP}, embedder({EMBED_URL})={_EMBED_UP}, "
    f"reranker({RERANKER_URL})={_RERANKER_UP}, SPARQL({SPARQL_URL})={_SPARQL_UP}, "
    f"corpus({_CFG_DEFAULTS.retriever.corpus.index_path})={_CORPUS_UP}."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow_integration,
    pytest.mark.requires_wikidata,
    pytest.mark.skipif(not _STACK_UP, reason=_skip_reason),
    # One module-scoped event loop for the whole file — see the MCTS real-server
    # test for the full rationale (keeps the Wikidata httpx pool on one loop).
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

    for name, tier in cfg.llm.tiers.items():
        tier.api_base = log_config_override(
            f"llm.tiers.{name}.api_base",
            tier.api_base,
            LLM_URL,
            reason="LANGGRAPH_TEST_LLM_URL endpoint for this run",
        )
        tier.api_key = api_key

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

    # Bound both strategies so a live, thinking-enabled run finishes in reasonable
    # wall time while still crossing iteration boundaries.
    cfg.search.cot.max_depth = log_config_override(
        "search.cot.max_depth",
        cfg.search.cot.max_depth,
        COT_DEPTH,
        reason="bound live multi-iteration CoT runtime (still >=2 to cross turn-2)",
    )
    cfg.search.mcts.num_iterations = log_config_override(
        "search.mcts.num_iterations",
        cfg.search.mcts.num_iterations,
        MCTS_ITERS,
        reason="bound live MCTS wall time (still >=1 full iteration)",
    )
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
    cfg.search.mcts.final_answer_min_depth = log_config_override(
        "search.mcts.final_answer_min_depth",
        cfg.search.mcts.final_answer_min_depth,
        0,
        reason="exercise _gen_final from the root within the bounded run",
    )
    # Live calls, not a warm cache.
    cfg.cache.enabled = log_config_override(
        "cache.enabled",
        cfg.cache.enabled,
        False,
        reason="exercise live endpoints, not a warm cache",
    )
    return cfg


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _system_runtime():
    """Load the real Wikidata client + FAISS retriever once, reused by every test.

    ``system.answer()`` would normally re-init the retriever (reloading the 99 GB
    index) and the Wikidata client on every call via ``_init_runtime``. We pre-
    load both once here, then patch ``system.init_retrieval_pipeline`` and
    ``system.init_wikidata`` to no-ops for the module so each strategy's
    ``answer()`` reuses the already-loaded singletons. Everything else in
    ``_init_runtime`` (web init, ``set_web_research_config``, session resets) still
    runs for real, so the orchestration path is exercised faithfully.
    """
    from langgraph_coe import system as system_mod
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

    # Neutralise the heavy in-answer() re-init so both strategy calls reuse the
    # singletons loaded above (the index must NOT be read a second time).
    orig_init_retrieval = system_mod.init_retrieval_pipeline
    orig_init_wikidata = system_mod.init_wikidata
    system_mod.init_retrieval_pipeline = lambda *a, **k: None  # type: ignore[assignment]
    system_mod.init_wikidata = lambda *a, **k: None  # type: ignore[assignment]

    try:
        yield cfg
    finally:
        system_mod.init_retrieval_pipeline = orig_init_retrieval  # type: ignore[assignment]
        system_mod.init_wikidata = orig_init_wikidata  # type: ignore[assignment]
        client = wd_mod._wikidata_client
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
        wd_mod._wikidata_client = None
        wd_mod._wikidata_config = None
        rmod._retriever_instance = None
        rmod._reranker_config = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _print_block(title: str, payload: Any) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    if isinstance(payload, (dict, list)):
        print(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True
        )
    else:
        print(str(payload), flush=True)


def _assert_grounded_answer(result: Any, *, strategy: str) -> None:
    """Shared contract every strategy's ``AnswerResult`` must satisfy."""
    assert result.question == _QUERY
    answer = str(result.answer or "")
    assert answer.strip(), f"{strategy}: system must return a non-empty answer"
    low = answer.lower()
    assert ("1876" in low) or ("oregon" in low), (
        f"{strategy}: answer should reference the founding year (1876) or state "
        f"(Oregon); got: {answer!r}"
    )
    assert result.metadata.get("strategy") == strategy, (
        f"AnswerResult metadata must record the strategy used; "
        f"got {result.metadata.get('strategy')!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_system_answer_cot_real_servers(_system_runtime):
    """``answer()`` with ``strategy=cot`` works end-to-end against the live stack."""
    from langgraph_coe import system as system_mod

    cfg = _system_runtime
    cfg.search.strategy = "cot"

    _print_block(
        "system.answer(strategy=cot) setup",
        {
            "llm_url": LLM_URL,
            "embed_url": EMBED_URL,
            "reranker_url": RERANKER_URL,
            "sparql_url": SPARQL_URL,
            "corpus_index": cfg.retriever.corpus.index_path,
            "cot_max_depth": cfg.search.cot.max_depth,
            "query": _QUERY,
        },
    )

    result = await system_mod.answer(_QUERY, config=cfg)

    _print_block(
        "AnswerResult (cot)",
        {
            "answer": result.answer,
            "concise_answer": result.concise_answer,
            "reasoning": result.reasoning,
            "metadata": result.metadata,
        },
    )

    _assert_grounded_answer(result, strategy="cot")
    # CoT bookkeeping: at least one decomposition iteration was recorded.
    assert int(result.metadata.get("num_iterations", 0)) >= 1, (
        "CoT run should record at least one iteration in metadata"
    )


async def test_system_answer_mcts_real_servers(_system_runtime):
    """``answer()`` with ``strategy=mcts`` works end-to-end against the live stack."""
    from langgraph_coe import system as system_mod

    cfg = _system_runtime
    cfg.search.strategy = "mcts"

    _print_block(
        "system.answer(strategy=mcts) setup",
        {
            "llm_url": LLM_URL,
            "embed_url": EMBED_URL,
            "reranker_url": RERANKER_URL,
            "sparql_url": SPARQL_URL,
            "corpus_index": cfg.retriever.corpus.index_path,
            "mcts_num_iterations": cfg.search.mcts.num_iterations,
            "mcts_max_simulation_depth": cfg.search.mcts.max_simulation_depth,
            "query": _QUERY,
        },
    )

    result = await system_mod.answer(_QUERY, config=cfg)

    _print_block(
        "AnswerResult (mcts)",
        {
            "answer": result.answer,
            "concise_answer": result.concise_answer,
            "reasoning": result.reasoning,
            "metadata": result.metadata,
        },
    )

    _assert_grounded_answer(result, strategy="mcts")
    # MCTS bookkeeping: at least one search iteration completed.
    assert int(result.metadata.get("iteration", 0)) >= 1, (
        "MCTS run should record at least one completed iteration in metadata"
    )


async def test_system_rejects_unknown_strategy(_system_runtime):
    """``answer()`` fails loud on an unknown strategy — never silently defaults."""
    from langgraph_coe import system as system_mod

    cfg = _system_runtime
    cfg.search.strategy = "not-a-strategy"
    try:
        with pytest.raises(ValueError, match="Unknown search strategy"):
            await system_mod.answer(_QUERY, config=cfg)
    finally:
        # Leave the shared config in a valid state for any later test/use.
        cfg.search.strategy = "cot"
