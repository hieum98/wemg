"""Phase 4 §9 — top-level ``system.py`` orchestration target specs.

Phase 4 wires the already-compiled strategy graphs into a public answer API:

  1. initialize tools/cache/runtime once per question,
  2. reset per-question ContextVars,
  3. select MCTS vs CoT via ``config.search.strategy``,
  4. build a complete strategy initial state,
  5. return ``AnswerResult.from_state(result)``,
  6. preserve a batch entrypoint over independent ``answer`` calls.

These tests use stub graphs and patched runtime initialization so they do not
touch real Redis, FAISS, Wikidata, web search, or LLM endpoints.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import networkx as nx
import pytest


def _system_or_skip():
    system = pytest.importorskip(
        "langgraph_coe.system",
        reason="Phase 4 wires the top-level system orchestrator",
    )
    if not hasattr(system, "answer"):
        pytest.skip("system.answer not implemented yet (§9 target)")
    return system


def _cfg(strategy: str = "cot"):
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.cache.enabled = False
    cfg.search.strategy = strategy
    return cfg


class StrategyGraphSpy:
    """Compiled-graph-like stub recording ``ainvoke`` calls."""

    def __init__(self, output: Dict[str, Any] | None = None) -> None:
        self.output = output or {
            "question": "What is the capital of France?",
            "final_answer": "Paris.",
            "concise_answer": "Paris",
            "reasoning": "The state says Paris.",
        }
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(state)
        return dict(self.output)


def _patch_noop_runtime(monkeypatch: pytest.MonkeyPatch, system: Any) -> None:
    monkeypatch.setattr(system, "_init_runtime", lambda _cfg: None, raising=False)
    monkeypatch.setattr(system, "reset_wikidata_session", lambda: None, raising=False)
    monkeypatch.setattr(system, "reset_web_research_session", lambda: None, raising=False)


def _patch_strategy_builders(
    monkeypatch: pytest.MonkeyPatch,
    system: Any,
    *,
    cot_graph: StrategyGraphSpy | None = None,
    mcts_graph: StrategyGraphSpy | None = None,
) -> Dict[str, List[Any]]:
    calls: Dict[str, List[Any]] = {"cot": [], "mcts": []}
    cot_graph = cot_graph or StrategyGraphSpy()
    mcts_graph = mcts_graph or StrategyGraphSpy()

    def _cot(*args: Any, **kwargs: Any):
        calls["cot"].append((args, kwargs))
        return cot_graph

    def _mcts(*args: Any, **kwargs: Any):
        calls["mcts"].append((args, kwargs))
        return mcts_graph

    monkeypatch.setattr(system, "build_cot_graph", _cot, raising=False)
    monkeypatch.setattr(system, "build_mcts_graph", _mcts, raising=False)
    return calls


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AwaitableResult:
    """Result object that also works if an implementation awaits ``answer``."""

    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer

    def __await__(self):
        async def _coro():
            return self

        return _coro().__await__()


def test_answer_result_from_state_contract():
    """``AnswerResult.from_state`` is the public adapter from graph state (§9)."""
    system = _system_or_skip()
    assert hasattr(system, "AnswerResult"), "Phase 4 must define system.AnswerResult"
    assert hasattr(system.AnswerResult, "from_state"), (
        "AnswerResult must expose from_state(result_state)"
    )

    result = system.AnswerResult.from_state(
        {
            "question": "What is the capital of France?",
            "final_answer": "Paris.",
            "concise_answer": "Paris",
            "reasoning": "France's capital is Paris.",
            "strategy": "cot",
        }
    )

    assert result.question == "What is the capital of France?"
    assert result.answer == "Paris."
    assert result.concise_answer == "Paris"
    assert result.reasoning == "France's capital is Paris."
    assert result.metadata["strategy"] == "cot"


async def test_answer_initializes_runtime_before_resets_and_graph_invocation(monkeypatch):
    """Tool/cache init happens before per-question reset and strategy invocation (§9)."""
    system = _system_or_skip()
    calls: List[str] = []
    cache_sentinel = object()

    monkeypatch.setattr(
        system,
        "_init_cache_layer",
        lambda _cfg: calls.append("cache") or cache_sentinel,
        raising=False,
    )
    monkeypatch.setattr(
        system,
        "init_wikidata",
        lambda _cfg, cache=None: calls.append(f"wikidata:{cache is cache_sentinel}"),
        raising=False,
    )
    monkeypatch.setattr(system, "init_web_search", lambda _cfg: calls.append("web"), raising=False)
    monkeypatch.setattr(
        system,
        "init_retrieval_pipeline",
        lambda _retriever, _reranker: calls.append("retrieval"),
        raising=False,
    )
    monkeypatch.setattr(
        system,
        "set_web_research_config",
        lambda _cfg: calls.append("web_graph_config"),
        raising=False,
    )
    monkeypatch.setattr(system, "reset_wikidata_session", lambda: calls.append("reset_wd"), raising=False)
    monkeypatch.setattr(system, "reset_web_research_session", lambda: calls.append("reset_web"), raising=False)

    graph = StrategyGraphSpy()

    def _build(*_args: Any, **_kwargs: Any):
        calls.append("build_cot")
        return graph

    monkeypatch.setattr(system, "build_cot_graph", _build, raising=False)
    monkeypatch.setattr(system, "build_mcts_graph", lambda *_a, **_kw: graph, raising=False)

    cfg = _cfg("cot")
    await system.answer("What is the capital of France?", cfg)

    assert calls[:7] == [
        "cache",
        "wikidata:True",
        "web",
        "retrieval",
        "web_graph_config",
        "reset_wd",
        "reset_web",
    ]
    assert calls[7] == "build_cot"
    assert graph.calls, "strategy graph must be invoked after initialization and resets"


async def test_strategy_switch_selects_only_configured_graph(monkeypatch):
    """``config.search.strategy`` chooses MCTS vs CoT, with no fallback build."""
    system = _system_or_skip()
    _patch_noop_runtime(monkeypatch, system)

    mcts_graph = StrategyGraphSpy()
    cot_graph = StrategyGraphSpy()
    calls = _patch_strategy_builders(
        monkeypatch,
        system,
        cot_graph=cot_graph,
        mcts_graph=mcts_graph,
    )

    await system.answer("Q mcts?", _cfg("mcts"))
    assert len(calls["mcts"]) == 1
    assert len(calls["cot"]) == 0
    assert mcts_graph.calls and not cot_graph.calls

    calls["mcts"].clear()
    calls["cot"].clear()
    mcts_graph.calls.clear()
    cot_graph.calls.clear()

    await system.answer("Q cot?", _cfg("cot"))
    assert len(calls["cot"]) == 1
    assert len(calls["mcts"]) == 0
    assert cot_graph.calls and not mcts_graph.calls


async def test_strategy_builders_receive_role_registry(monkeypatch):
    """The selected graph builder receives a ``RoleModelRegistry`` built from config.llm."""
    system = _system_or_skip()
    _patch_noop_runtime(monkeypatch, system)
    graph = StrategyGraphSpy()
    captured: Dict[str, Any] = {}

    def _build(*args: Any, **kwargs: Any):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return graph

    monkeypatch.setattr(system, "build_cot_graph", _build, raising=False)
    monkeypatch.setattr(system, "build_mcts_graph", lambda *_a, **_kw: graph, raising=False)

    await system.answer("Q?", _cfg("cot"))

    from langgraph_coe.llm import RoleModelRegistry

    assert captured["args"], "strategy builder must receive a registry argument"
    assert isinstance(captured["args"][0], RoleModelRegistry)


async def test_cot_strategy_receives_complete_initial_state(monkeypatch):
    """``answer`` must invoke CoTGraph with a full ``CoTState`` shape (§7/§9)."""
    system = _system_or_skip()
    _patch_noop_runtime(monkeypatch, system)
    cot_graph = StrategyGraphSpy()
    _patch_strategy_builders(monkeypatch, system, cot_graph=cot_graph)

    await system.answer("What is the capital of France?", _cfg("cot"))

    state = cot_graph.calls[0]
    assert state["question"] == "What is the capital of France?"
    for key in (
        "max_depth",
        "depth",
        "is_answerable",
        "subquestions",
        "retrieved_raw_context",
        "retrieved_raw_triples",
        "reranked_context",
        "current_subanswers",
        "iteration_history",
        "text_memory",
        "graph_memory",
        "entity_dict",
        "final_answer",
    ):
        assert key in state, f"initial CoT state missing {key!r}"
    assert state["depth"] == 0
    assert isinstance(state["graph_memory"], nx.DiGraph)
    assert state["entity_dict"] == {}


async def test_mcts_strategy_receives_complete_initial_state(monkeypatch):
    """``answer`` must invoke MCTSGraph with a full ``MCTSState`` shape (§8/§9)."""
    system = _system_or_skip()
    _patch_noop_runtime(monkeypatch, system)
    mcts_graph = StrategyGraphSpy()
    _patch_strategy_builders(monkeypatch, system, mcts_graph=mcts_graph)

    await system.answer("What is the capital of France?", _cfg("mcts"))

    state = mcts_graph.calls[0]
    assert state["question"] == "What is the capital of France?"
    for key in (
        "max_iterations",
        "iteration",
        "tree",
        "root_id",
        "current_path",
        "expanded_node_ids",
        "simulation_result",
        "reward",
        "new_raw_triples",
        "text_memory",
        "graph_memory",
        "entity_dict",
        "semantic_sufficiency_signals",
        "iterations_without_improvement",
        "best_value",
        "final_answer",
    ):
        assert key in state, f"initial MCTS state missing {key!r}"
    assert state["max_iterations"] == _cfg("mcts").search.mcts.num_iterations
    assert state["iteration"] == 0
    assert isinstance(state["graph_memory"], nx.DiGraph)
    root = state["tree"][state["root_id"]]
    assert root["node_type"] == "user_question"
    assert root["content"]["question"] == "What is the capital of France?"


async def test_answer_returns_answer_result_not_raw_state(monkeypatch):
    """Public ``answer`` returns ``AnswerResult.from_state(result)``, not graph dict."""
    system = _system_or_skip()
    assert hasattr(system, "AnswerResult"), "Phase 4 must define system.AnswerResult"
    _patch_noop_runtime(monkeypatch, system)
    graph = StrategyGraphSpy(
        {
            "question": "What is the capital of France?",
            "final_answer": "Paris.",
            "concise_answer": "Paris",
            "reasoning": "Synthesized by graph.",
        }
    )
    _patch_strategy_builders(monkeypatch, system, cot_graph=graph)

    result = await system.answer("What is the capital of France?", _cfg("cot"))

    assert isinstance(result, system.AnswerResult)
    assert result.answer == "Paris."
    assert result.concise_answer == "Paris"
    assert result.metadata["strategy"] == "cot"


async def test_invalid_strategy_raises_value_error_without_building_graph(monkeypatch):
    """Invalid ``config.search.strategy`` must not silently fall back to CoT."""
    system = _system_or_skip()
    _patch_noop_runtime(monkeypatch, system)
    calls = _patch_strategy_builders(monkeypatch, system)
    cfg = _cfg("cot")
    cfg.search.strategy = "bogus"

    with pytest.raises(ValueError, match="strategy"):
        await system.answer("Q?", cfg)

    assert calls["cot"] == []
    assert calls["mcts"] == []


async def test_answer_batch_exists_preserves_order_and_calls_answer_per_question(monkeypatch):
    """``answer_batch`` preserves input order over independent ``answer`` calls (§9)."""
    system = _system_or_skip()
    assert hasattr(system, "answer_batch"), "Phase 4 must expose system.answer_batch"

    calls: List[str] = []

    def _fake_answer(question: str, config: Any | None = None):
        calls.append(question)
        return AwaitableResult(question=question, answer=f"answer:{question}")

    monkeypatch.setattr(system, "answer", _fake_answer, raising=True)

    out = system.answer_batch(["Q1", "Q2", "Q3"], config=_cfg("cot"), max_workers=2)
    results = await _maybe_await(out)

    assert calls == ["Q1", "Q2", "Q3"]
    assert [r.question for r in results] == ["Q1", "Q2", "Q3"]
    assert [r.answer for r in results] == ["answer:Q1", "answer:Q2", "answer:Q3"]
