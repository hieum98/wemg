"""Phase 3 §8 — MCTSGraph target specs.

``MCTSGraph`` stores search state as a dict-backed tree, expands only the node
types used by legacy MCTS, rolls out through ``CoTGraph``, evaluates terminal
answers with three verifier views, and synchronizes memory after each search
iteration.

These tests are target specs: they skip cleanly only while the Phase 3 graph
module itself is absent, then fail on contract drift once implemented.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import networkx as nx
import pytest

from langgraph_coe import roles as roles_mod


def _import_module():
    """Phase 3 introduces ``langgraph_coe.graphs.mcts``."""
    try:
        from langgraph_coe.graphs import mcts as mcts_mod  # type: ignore
    except ImportError:
        pytest.skip("Phase 3 §8 introduces langgraph_coe.graphs.mcts")
    if not hasattr(mcts_mod, "build_mcts_graph"):
        pytest.skip("build_mcts_graph not implemented yet (§8 target)")
    return mcts_mod


def _registry():
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    return registry


def _mcts_config(
    *,
    max_simulation_depth: int = 3,
    final_answer_min_depth: int = 2,
    high_confidence_threshold: float = 0.9,
    convergence_patience: int = 5,
    semantic_sufficiency_count: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        search=SimpleNamespace(
            mcts=SimpleNamespace(
                max_simulation_depth=max_simulation_depth,
                final_answer_min_depth=final_answer_min_depth,
                high_confidence_threshold=high_confidence_threshold,
                convergence_patience=convergence_patience,
                semantic_sufficiency_count=semantic_sufficiency_count,
            )
        )
    )


def _subq_out(*subquestions: str, answerable: bool = False) -> roles_mod.SubquestionGenerationOutput:
    return roles_mod.SubquestionGenerationOutput(
        is_answerable=answerable,
        subquestions=list(subquestions),
    )


def _answer_out(text: str) -> roles_mod.AnswerGenerationOutput:
    return roles_mod.AnswerGenerationOutput(
        answer=text,
        concise_answer=text,
        reasoning=f"Reasoning for {text}",
        confidence_level="high",
    )


def _self_correct_out(text: str) -> roles_mod.SelfCorrectionOutput:
    return roles_mod.SelfCorrectionOutput(
        status="correct",
        refined_answer=text,
        confidence_level="high",
    )


def _final_out(text: str) -> roles_mod.FinalAnswerSynthesisOutput:
    return roles_mod.FinalAnswerSynthesisOutput(
        final_answer=text,
        concise_answer=text,
        reasoning=f"Reasoning for {text}",
        confidence_level="high",
    )


def _verify_out(rating: float, reasoning: str) -> roles_mod.AnswerVerificationOutput:
    return roles_mod.AnswerVerificationOutput(rating=rating, reasoning=reasoning)


class RoleExecutorSpy:
    """Async stand-in for ``execute_role_lc`` with per-role canned outputs."""

    def __init__(
        self,
        *,
        subquestions: Sequence[str] = ("What is France's capital?",),
        answers: Sequence[str] = ("Paris is the capital.",),
        self_corrections: Sequence[str] = ("Paris is still the capital.",),
        final_answers: Sequence[str] = ("Paris.",),
        verifier_ratings: Sequence[float] = (9.0, 8.0, 7.0),
        verifier_reasons: Sequence[str] = ("no-context ok", "text-memory ok", "graph-memory ok"),
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._subquestions = list(subquestions)
        self._answers = list(answers)
        self._self_corrections = list(self_corrections)
        self._final_answers = list(final_answers)
        self._verifier_ratings = list(verifier_ratings)
        self._verifier_reasons = list(verifier_reasons)

    async def __call__(
        self,
        registry: Any,
        role: roles_mod.Role,
        input_data: Any,
        n: int = 1,
        tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        self.calls.append(
            {
                "role": role.name,
                "input": input_data,
                "n": n,
                "tier_override": tier_override,
            }
        )

        if role.name == "subquestion_generator":
            return _subq_out(*self._subquestions), {}

        if role.name == "answer_generator":
            inputs = input_data if isinstance(input_data, list) else [input_data]
            outputs = []
            for i, _item in enumerate(inputs):
                text = self._answers.pop(0) if self._answers else f"Subanswer {i + 1}"
                outputs.append(_answer_out(text))
            return (outputs if isinstance(input_data, list) else outputs[0]), {}

        if role.name == "self_corrector":
            text = self._self_corrections.pop(0) if self._self_corrections else "Corrected."
            return _self_correct_out(text), {}

        if role.name == "final_answer_synthesizer":
            text = self._final_answers.pop(0) if self._final_answers else "Final."
            return _final_out(text), {}

        if role.name == "verifier":
            inputs = input_data if isinstance(input_data, list) else [input_data]
            outputs = []
            for i, _item in enumerate(inputs):
                rating = self._verifier_ratings.pop(0) if self._verifier_ratings else 5.0
                reason = self._verifier_reasons.pop(0) if self._verifier_reasons else f"view {i}"
                outputs.append(_verify_out(rating, reason))
            return (outputs if isinstance(input_data, list) else outputs[0]), {}

        if role.name == "extractor":
            raise AssertionError("MCTSGraph must not run an extractor pass (§8.3)")

        raise AssertionError(f"Unexpected role call: {role.name}")

    def role_inputs(self, role_name: str) -> List[Any]:
        return [c["input"] for c in self.calls if c["role"] == role_name]


class CompiledGraphSpy:
    """Small compiled-graph-like object recording ``ainvoke`` payloads."""

    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(state)
        return dict(self.output)


def _memory_graph(output: Dict[str, Any] | None = None) -> CompiledGraphSpy:
    return CompiledGraphSpy(
        output
        or {
            "updated_text_memory": ["Updated MCTS memory"],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {"Q90": "Paris"},
        }
    )


def _cot_graph(output: Dict[str, Any] | None = None) -> CompiledGraphSpy:
    return CompiledGraphSpy(
        output
        or {
            "iteration_history": [
                {
                    "depth": 0,
                    "subquestions": ["What is France's capital?", "Where is Paris?"],
                    "subanswers": ["Paris is France's capital.", "Paris is in France."],
                }
            ],
            "final_answer": "Paris.",
            "text_memory": ["Rollout memory"],
            "graph_memory": nx.DiGraph(),
            "entity_dict": {"Q90": "Paris"},
            "retrieved_raw_triples": ["France | capital | Paris"],
        }
    )


def _install_graph_spies(
    monkeypatch: pytest.MonkeyPatch,
    mcts_mod: Any,
    *,
    executor: RoleExecutorSpy,
    cot_graph: CompiledGraphSpy | None = None,
    memory_graph: CompiledGraphSpy | None = None,
) -> tuple[CompiledGraphSpy, CompiledGraphSpy]:
    cot_graph = cot_graph or _cot_graph()
    memory_graph = memory_graph or _memory_graph()

    monkeypatch.setattr(mcts_mod, "execute_role_lc", executor, raising=False)
    monkeypatch.setattr(mcts_mod, "build_cot_graph", lambda *_a, **_kw: cot_graph, raising=False)
    monkeypatch.setattr(
        mcts_mod,
        "build_memory_update_graph",
        lambda *_a, **_kw: memory_graph,
        raising=False,
    )
    return cot_graph, memory_graph


def _build_graph(mcts_mod: Any, cfg: Any | None = None):
    cfg = cfg or _mcts_config()
    try:
        return mcts_mod.build_mcts_graph(_registry(), cfg)
    except TypeError:
        return mcts_mod.build_mcts_graph(_registry())


def _state(
    *,
    question: str = "What is the capital of France?",
    max_iterations: int = 1,
    iteration: int = 0,
    tree: Dict[str, Dict[str, Any]] | None = None,
    root_id: str = "root",
    text_memory: Sequence[str] = ("France is relevant.",),
    graph_memory: nx.DiGraph | None = None,
    entity_dict: Dict[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    state = {
        "question": question,
        "max_iterations": max_iterations,
        "iteration": iteration,
        "tree": tree or {},
        "root_id": root_id,
        "current_path": [],
        "expanded_node_ids": [],
        "simulation_result": {},
        "reward": 0.0,
        "new_raw_triples": [],
        "text_memory": list(text_memory),
        "graph_memory": graph_memory or nx.DiGraph(),
        "entity_dict": dict(entity_dict or {}),
        "semantic_sufficiency_signals": 0,
        "iterations_without_improvement": 0,
        "best_value": 0.0,
        "final_answer": "",
    }
    state.update(extra)
    return state


def _input_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(_input_text(v) for v in value)
    if hasattr(value, "context"):
        return str(value.context)
    if hasattr(value, "candidate_answer"):
        return str(value.candidate_answer)
    if hasattr(value, "candidate_answers"):
        return "\n".join(map(str, value.candidate_answers))
    if hasattr(value, "new_text_items"):
        return "\n".join(map(str, value.new_text_items))
    return str(value)


def _node_type_value(node: Dict[str, Any]) -> str:
    value = node.get("node_type")
    return getattr(value, "value", str(value))


def _nodes_of_type(tree: Dict[str, Dict[str, Any]], node_type: str) -> List[Dict[str, Any]]:
    return [node for node in tree.values() if _node_type_value(node) == node_type]


def test_node_types_priors_and_dict_merge_contracts():
    """§8.1/§8.2 expose only the planned node types, priors, and rightward merge."""
    mcts_mod = _import_module()

    assert hasattr(mcts_mod, "MCTSNodeType")
    assert hasattr(mcts_mod, "NODE_TYPE_PRIOR")
    assert hasattr(mcts_mod, "dict_merge")

    values = {item.value for item in mcts_mod.MCTSNodeType}
    assert values == {"user_question", "sub_qa", "self_corrected", "final_answer"}
    assert "sub_qa_batch" not in values
    assert "rephrased_question" not in values
    assert "synthesis" not in values

    assert mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SUB_QA] == 0.60
    assert mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SELF_CORRECTED] == 0.50
    assert mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.FINAL_ANSWER] == 0.30

    assert mcts_mod.dict_merge({"a": {"value": 1}}, {"a": {"value": 2}, "b": {}}) == {
        "a": {"value": 2},
        "b": {},
    }


async def test_root_expansion_skips_gen_final_at_shallow_depth(monkeypatch):
    """Root expand (depth 0) must not call ``_gen_final`` when ``final_answer_min_depth=2``."""
    mcts_mod = _import_module()
    executor = RoleExecutorSpy(
        subquestions=["What is France's capital?", "Which country is Paris in?"],
        answers=["Paris is the capital.", "Paris is in France."],
        final_answers=["Expand final.", "Synth final."],
    )
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor)

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(_state(max_iterations=1))

    tree = final["tree"]
    root = tree["root"]
    root_children = [tree[cid] for cid in root.get("children_ids") or []]
    subqa_nodes = _nodes_of_type(tree, "sub_qa")
    assert len(subqa_nodes) >= 2
    assert all(mcts_mod._ntype(node) == "sub_qa" for node in root_children), (
        "root expand at depth 0 must emit SUB_QA children only, not FINAL_ANSWER"
    )
    assert all(node["prior"] == mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.SUB_QA] for node in subqa_nodes)
    assert executor.role_inputs("subquestion_generator")
    assert executor.role_inputs("answer_generator")
    expand_final_calls = [
        c for c in executor.calls
        if c["role"] == "final_answer_synthesizer"
        and getattr(c["input"], "candidate_answers", None) == ["France is relevant."]
    ]
    assert not expand_final_calls, (
        "expand must not invoke final_answer_synthesizer at shallow depth; "
        f"saw {len(expand_final_calls)} expand-style call(s)"
    )
    assert len(executor.role_inputs("final_answer_synthesizer")) == 1, (
        "only the terminal synthesize node should call final_answer_synthesizer"
    )


async def test_root_expansion_calls_gen_final_when_min_depth_met(monkeypatch):
    """When ``final_answer_min_depth=0``, root expand emits a FINAL_ANSWER child."""
    mcts_mod = _import_module()
    executor = RoleExecutorSpy(
        subquestions=["What is France's capital?"],
        answers=["Paris is the capital."],
        final_answers=["Expand final.", "Synth final."],
    )
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor)

    graph = _build_graph(mcts_mod, _mcts_config(final_answer_min_depth=0))
    final = await graph.ainvoke(_state(max_iterations=1))

    final_nodes = _nodes_of_type(final["tree"], "final_answer")
    expand_final_nodes = [
        node
        for node in final_nodes
        if "Expand final." in str(node.get("content", {}))
    ]
    assert expand_final_nodes, "expand must emit FINAL_ANSWER when min depth is met"
    assert all(
        node["prior"] == mcts_mod.NODE_TYPE_PRIOR[mcts_mod.MCTSNodeType.FINAL_ANSWER]
        for node in expand_final_nodes
    )
    assert len(executor.role_inputs("final_answer_synthesizer")) == 2, (
        "expand + synthesize should each call final_answer_synthesizer"
    )


async def test_simulate_invokes_cot_graph_with_shared_memory_and_configured_depth(monkeypatch):
    """Rollout calls ``CoTGraph`` with shared memory refs and configured max depth (§8.3)."""
    mcts_mod = _import_module()
    text_memory = ["Shared text memory"]
    graph_memory = nx.DiGraph()
    entity_dict = {"Q142": "France"}
    cot = _cot_graph()
    executor = RoleExecutorSpy()
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor, cot_graph=cot)

    graph = _build_graph(mcts_mod, _mcts_config(max_simulation_depth=2))
    initial = _state(
        max_iterations=1,
        text_memory=text_memory,
        graph_memory=graph_memory,
        entity_dict=entity_dict,
    )
    await graph.ainvoke(initial)

    assert cot.calls, "simulate must invoke the compiled CoTGraph"
    payload = cot.calls[0]
    assert payload["text_memory"] is initial["text_memory"]
    assert payload["graph_memory"] is initial["graph_memory"]
    assert payload["entity_dict"] is initial["entity_dict"]
    assert payload["max_depth"] == 2


async def test_cot_iteration_history_is_adapted_to_linear_subqa_chain(monkeypatch):
    """CoT batch iterations import as concatenated ``SUB_QA`` nodes, then final answer."""
    mcts_mod = _import_module()
    cot = _cot_graph(
        {
            "iteration_history": [
                {
                    "depth": 0,
                    "subquestions": ["Q one?", "Q two?"],
                    "subanswers": ["A one.", "A two."],
                },
                {
                    "depth": 1,
                    "subquestions": ["Q three?"],
                    "subanswers": ["A three."],
                },
            ],
            "final_answer": "Rollout final.",
            "text_memory": ["Rollout memory"],
            "graph_memory": nx.DiGraph(),
            "entity_dict": {},
        }
    )
    executor = RoleExecutorSpy(final_answers=["Expand final.", "Synth final."])
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor, cot_graph=cot)

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(_state(max_iterations=1))

    tree = final["tree"]
    rollout_nodes = [
        node
        for node in _nodes_of_type(tree, "sub_qa")
        if "Sub Q1: Q one?" in str(node.get("content", {}))
    ]
    assert rollout_nodes, "simulate must adapt CoT iteration_history into SUB_QA nodes"
    content = rollout_nodes[0]["content"]
    assert "Sub Q1: Q one?" in content["sub_question"]
    assert "Sub Q2: Q two?" in content["sub_question"]
    assert "Sub A1: A one." in content["sub_answer"]
    assert "Sub A2: A two." in content["sub_answer"]

    rollout_final = [
        node
        for node in _nodes_of_type(tree, "final_answer")
        if "Rollout final." in str(node.get("content", {}))
    ]
    assert rollout_final, "simulate must append a FINAL_ANSWER node for final_state['final_answer']"
    assert rollout_final[0]["parent_id"] in tree


async def test_evaluate_uses_three_verifier_views_and_normalizes_reward(monkeypatch):
    """Evaluation calls verifier for no-context/text-memory/graph-memory views (§8.3)."""
    mcts_mod = _import_module()
    graph_memory = nx.DiGraph()
    graph_memory.add_edge("France", "Paris", relation="capital")
    executor = RoleExecutorSpy(
        verifier_ratings=[10.0, 5.0, 0.0],
        verifier_reasons=["known answer", "text partial", "graph contradicts"],
    )
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor)

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(
        _state(
            max_iterations=1,
            text_memory=["France has capital Paris."],
            graph_memory=graph_memory,
        )
    )

    verifier_inputs = executor.role_inputs("verifier")
    assert len(verifier_inputs) == 3
    contexts = [_input_text(v) for v in verifier_inputs]
    assert any("Not provided" in c or not c.strip() for c in contexts)
    assert any("France has capital Paris." in c for c in contexts)
    assert any("France" in c and "Paris" in c and "capital" in c for c in contexts)
    assert final["reward"] == pytest.approx(0.0), (
        "reward must be normalized as (mean_rating - 5.0) / 5.0"
    )


async def test_backprop_updates_visits_and_values_for_current_path(monkeypatch):
    """Backprop increments visits and adds reward along ``current_path`` (§8.3)."""
    mcts_mod = _import_module()
    executor = RoleExecutorSpy(verifier_ratings=[10.0, 10.0, 10.0])
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor)

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(_state(max_iterations=1))

    tree = final["tree"]
    path = final["current_path"]
    assert path, "select/simulate/evaluate must leave the evaluated path in state"
    for node_id in path:
        assert tree[node_id]["visits"] >= 1
        assert tree[node_id]["value"] == pytest.approx(1.0)


async def test_memory_update_receives_expansion_rollout_and_verifier_text(monkeypatch):
    """MCTS memory sync merges expanded text, rollout text/triples, and verifier critiques."""
    mcts_mod = _import_module()
    memory = _memory_graph()
    cot = _cot_graph(
        {
            "iteration_history": [
                {
                    "depth": 0,
                    "subquestions": ["What is France's capital?"],
                    "subanswers": ["Rollout says Paris."],
                }
            ],
            "final_answer": "Rollout final Paris.",
            "text_memory": ["Rollout memory"],
            "graph_memory": nx.DiGraph(),
            "entity_dict": {},
            "retrieved_raw_triples": ["France | capital | Paris"],
        }
    )
    executor = RoleExecutorSpy(
        answers=["Expanded subanswer Paris."],
        verifier_ratings=[9.0, 9.0, 9.0],
        verifier_reasons=["critique no context", "critique text", "critique graph"],
    )
    _install_graph_spies(
        monkeypatch,
        mcts_mod,
        executor=executor,
        cot_graph=cot,
        memory_graph=memory,
    )

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(_state(max_iterations=1))

    assert memory.calls, "mem_update must invoke MemoryUpdateGraph after backprop"
    payload = memory.calls[0]
    joined_text = "\n".join(map(str, payload["new_text_items"]))
    assert "Expanded subanswer Paris." in joined_text
    assert "Rollout says Paris." in joined_text
    assert "critique no context" in joined_text
    assert "critique text" in joined_text
    assert "critique graph" in joined_text
    assert "France | capital | Paris" in payload["new_raw_triples"]
    assert final["text_memory"] == ["Updated MCTS memory"]
    assert final["entity_dict"] == {"Q90": "Paris"}


async def test_final_answer_nodes_are_terminal_not_expanded(monkeypatch):
    """``FINAL_ANSWER`` nodes are terminal and do not run subqa/self-correction expansion."""
    mcts_mod = _import_module()
    root_id = "root"
    final_id = "final-1"
    tree = {
        root_id: {
            "node_id": root_id,
            "parent_id": None,
            "children_ids": [final_id],
            "node_type": "user_question",
            "content": {"question": "What is the capital of France?"},
            "visits": 10,
            "value": 5.0,
            "prior": 1.0,
        },
        final_id: {
            "node_id": final_id,
            "parent_id": root_id,
            "children_ids": [],
            "node_type": "final_answer",
            "content": {"final_answer": "Paris."},
            "visits": 0,
            "value": 0.0,
            "prior": 0.30,
        },
    }
    executor = RoleExecutorSpy()
    _install_graph_spies(monkeypatch, mcts_mod, executor=executor)

    graph = _build_graph(mcts_mod)
    final = await graph.ainvoke(_state(max_iterations=1, tree=tree, root_id=root_id))

    assert final["expanded_node_ids"] == []
    assert not executor.role_inputs("subquestion_generator")
    assert not executor.role_inputs("self_corrector")
    assert final["current_path"][-1] == final_id


def test_search_mcts_config_exposes_termination_knobs():
    """§8.3 route conditions require ``config.search.mcts`` knobs."""
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    assert hasattr(cfg, "search"), "Phase 3 must add LangGraphCoeConfig.search"
    assert hasattr(cfg.search, "mcts"), "Phase 3 must add search.mcts"
    assert cfg.search.mcts.high_confidence_threshold == 0.9
    assert cfg.search.mcts.semantic_sufficiency_count >= 1
    assert cfg.search.mcts.convergence_patience >= 1
    assert cfg.search.mcts.max_simulation_depth == 3
    assert cfg.search.mcts.final_answer_min_depth == 2
