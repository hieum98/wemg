"""Graph-level regressions for the bug fixes, plus branch-local MCTS memory.

Separate from ``test_bugfixes`` because these drive the compiled graphs rather
than the pure helpers — the defects are only observable in the wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import networkx as nx
import pytest

from langgraph_coe import llm as llm_mod
from langgraph_coe import roles as roles_mod
from langgraph_coe.graphs import cot as cot_mod
from langgraph_coe.graphs import mcts as mcts_mod

from .test_cot_graph import (
    CompiledGraphSpy,
    RoleExecutorSpy,
    _install_graph_spies,
    _subq_out,
)


def _config():
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.reranker.enabled = False
    cfg.reranker.top_k = 3
    cfg.web_search.enabled = False
    return cfg


def _registry(cfg: Any):
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(cfg.llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    return registry


def _state(*, max_depth: int = 1) -> Dict[str, Any]:
    return {
        "question": "What is the capital of France?",
        "max_depth": max_depth,
        "depth": 0,
        "is_answerable": False,
        "subquestions": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": [],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "final_answer": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 0.1 — an outage must not be reported as convergence
# ──────────────────────────────────────────────────────────────────────────────


class _AllParseFailuresExecutor(RoleExecutorSpy):
    """Every SUBQUESTION_GENERATOR call exhausts its retries."""

    async def __call__(
        self, registry: Any, role: roles_mod.Role, input_data: Any,
        n: int = 1, tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        if role.name == "subquestion_generator":
            self.calls.append({"role": role.name, "input": input_data, "n": n,
                               "tier_override": tier_override})
            outs = [
                llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR)
                for _ in range(n)
            ]
            return (outs if n > 1 else outs[0]), {}
        return await super().__call__(registry, role, input_data, n, tier_override)


@pytest.mark.asyncio
async def test_total_parse_failure_retries_instead_of_claiming_answerable(monkeypatch):
    """Before the fix, ``is_answerable = should_direct or not subqs`` turned three
    parse failures into "answerable" and routed straight to synthesis.

    Now the loop burns iterations re-asking, bounded by ``max_depth``, and only
    finalizes once the budget is spent — which is honest about what happened.
    """
    cfg = _config()
    executor = _AllParseFailuresExecutor(subq_outputs=[])
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_state(max_depth=3))

    assert final["subq_parse_failed"] is True
    assert final["is_answerable"] is False, (
        "a retry-exhausted parse failure must never read as 'answerable'"
    )
    # It re-asked rather than synthesizing on the first failure.
    assert len([c for c in executor.calls if c["role"] == "subquestion_generator"]) > 1
    # No retrieval happened, so no sub-answers were fabricated.
    assert not [c for c in executor.calls if c["role"] == "answer_generator"]
    # It still terminates with an answer rather than hanging.
    assert final["final_answer"]


# ──────────────────────────────────────────────────────────────────────────────
# 0.2 — index alignment between subquestions and sub-answers
# ──────────────────────────────────────────────────────────────────────────────


class _BlankMiddleAnswerExecutor(RoleExecutorSpy):
    """Returns a blank answer for the middle sub-question."""

    async def __call__(
        self, registry: Any, role: roles_mod.Role, input_data: Any,
        n: int = 1, tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        if role.name == "answer_generator":
            self.calls.append({"role": role.name, "input": input_data, "n": n,
                               "tier_override": tier_override})
            inputs = input_data if isinstance(input_data, list) else [input_data]
            texts = ["First answer.", "", "Third answer."]
            outputs = [
                roles_mod.AnswerGenerationOutput(
                    answer=texts[i] if i < len(texts) else "",
                    concise_answer=texts[i] if i < len(texts) else "",
                    reasoning="r",
                    confidence_level="high",
                )
                for i in range(len(inputs))
            ]
            return (outputs if isinstance(input_data, list) else outputs[0]), {}
        return await super().__call__(registry, role, input_data, n, tier_override)


@pytest.mark.asyncio
async def test_blank_answer_keeps_the_remaining_answers_on_their_own_subquestions(
    monkeypatch,
):
    """Before the fix a blank answer was skipped, so "Third answer." moved up into
    slot 1 and was recorded against the *second* sub-question — and the MCTS
    rollout chain zips these lists by index, so the misattribution propagated into
    the tree."""
    cfg = _config()
    subqs = ["Q one?", "Q two?", "Q three?"]
    executor = _BlankMiddleAnswerExecutor(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=subqs, needs_kg=[True] * 3),
            _subq_out(answerable=True),
        ]
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_state(max_depth=2))

    entry = final["iteration_history"][0]
    assert entry["subquestions"] == subqs
    assert len(entry["subanswers"]) == len(subqs), "lists must stay index-aligned"
    assert entry["subanswers"] == ["First answer.", "", "Third answer."]
    assert entry["subanswers"][2] == "Third answer.", (
        "the third answer must remain on the third sub-question"
    )


@pytest.mark.asyncio
async def test_blank_alignment_placeholders_do_not_enter_memory(monkeypatch):
    """Alignment is for the trajectory; memory should only see real text."""
    cfg = _config()
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["Updated."],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {},
            "retractions": [],
        }
    )
    executor = _BlankMiddleAnswerExecutor(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["A?", "B?", "C?"],
                      needs_kg=[True] * 3),
            _subq_out(answerable=True),
        ]
    )
    _install_graph_spies(
        monkeypatch, cot_mod, executor=executor, memory_graph=memory_graph
    )

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    await graph.ainvoke(_state(max_depth=2))

    assert memory_graph.calls
    assert "" not in memory_graph.calls[0]["new_text_items"]


# ──────────────────────────────────────────────────────────────────────────────
# 0.3 — verifier parse failure excluded from the reward
# ──────────────────────────────────────────────────────────────────────────────


def _mcts_config(*, branch_local: bool = False, plan_enabled: bool = False):
    cfg = _config()
    cfg.search.mcts.branch_local_memory = branch_local
    cfg.search.plan.enabled = plan_enabled
    cfg.search.mcts.num_iterations = 1
    cfg.search.mcts.min_iterations = 0
    return cfg


@pytest.mark.asyncio
async def test_unparsed_verifier_view_is_excluded_rather_than_scored_zero():
    """Two views rating 9.0 with one parse failure must give the reward for 9.0,
    not the reward for a 6.0 mean dragged down by a phantom zero."""
    cfg = _mcts_config()
    calls: List[str] = []

    async def fake_execute(registry, role, input_data, n=1, tier_override=None):
        calls.append(role.name)
        if role.name == "verifier":
            # First view fails to parse; the other two rate 9.0.
            idx = sum(1 for c in calls if c == "verifier")
            if idx == 1:
                return llm_mod.build_safe_default_output(roles_mod.VERIFIER), {}
            return roles_mod.AnswerVerificationOutput(rating=9.0, reasoning="ok"), {}
        raise AssertionError(role.name)

    # Drive ``evaluate`` in isolation via a minimal state.
    import langgraph_coe.graphs.mcts as m

    orig = m.execute_role_lc
    m.execute_role_lc = fake_execute
    try:
        graph_builder_state = {
            "question": "Q",
            "tree": {
                "n1": {
                    "node_id": "n1",
                    "parent_id": None,
                    "children_ids": [],
                    "node_type": m.MCTSNodeType.FINAL_ANSWER,
                    "content": {"final_answer": "A", "reasoning": "R"},
                    "visits": 0,
                    "value": 0.0,
                    "prior": 0.3,
                }
            },
            "current_path": ["n1"],
            "simulation_result": {},
            "text_memory": ["mem"],
            "graph_memory": nx.DiGraph(),
        }
        compiled = m.build_mcts_graph(_registry(cfg), cfg)
        evaluate = compiled.nodes["evaluate"].bound  # type: ignore[attr-defined]
        out = await evaluate.ainvoke(graph_builder_state)
    finally:
        m.execute_role_lc = orig

    assert out["reward"] == pytest.approx((9.0 - 5.0) / 5.0)
    assert out["simulation_result"]["verifier_ratings"] == [9.0, 9.0]
    # The failed view contributes no critique either.
    assert len(out["simulation_result"]["verifier_critiques"]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 0.5 — assessments excluded from re-verification
# ──────────────────────────────────────────────────────────────────────────────


def test_reverification_skips_assessments_and_retrieval_but_keeps_predictions():
    """The filter that stops a verifier critique becoming a search query."""
    from langgraph_coe.graphs.memory_update import (
        _is_assessment,
        _is_retrieval_grounded,
    )

    memory = [
        "[Assessment]: Verifier (no context): the answer is plausible but thin.",
        "[Retrieval]: Paris is the capital of France.",
        "[System Prediction]: The tower is 300m tall.",
    ]
    eligible = [
        item
        for item in memory
        if not (_is_retrieval_grounded(item) or _is_assessment(item))
    ]
    assert eligible == ["[System Prediction]: The tower is 300m tall."]


# ──────────────────────────────────────────────────────────────────────────────
# Branch-local MCTS memory
# ──────────────────────────────────────────────────────────────────────────────


def test_snapshot_resolution_falls_back_to_global_when_disabled():
    state = {
        "text_memory": ["global"],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "plan": "global plan",
        "plan_version": 1,
        "plan_ledger": [],
        "snapshots": {"n1": {"text_memory": ["branch"], "plan": "branch plan"}},
    }
    view = mcts_mod.resolve_snapshot(state, ["root", "n1"], enabled=False)
    assert view["text_memory"] == ["global"]
    assert view["plan"] == "global plan"


def test_snapshot_resolution_prefers_the_path_tip():
    state = {
        "text_memory": ["global"],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "plan": "global plan",
        "plan_version": 1,
        "plan_ledger": [],
        "snapshots": {
            "root": {"text_memory": ["root mem"]},
            "n1": {"text_memory": ["branch mem"]},
        },
    }
    view = mcts_mod.resolve_snapshot(state, ["root", "n1"], enabled=True)
    assert view["text_memory"] == ["branch mem"]


def test_fresh_child_inherits_the_nearest_ancestor_snapshot():
    """A newly expanded child has no snapshot of its own, so it must inherit its
    parent's view rather than seeing whatever another branch last wrote."""
    state = {
        "text_memory": ["global — another branch wrote this"],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "plan": "",
        "plan_version": 0,
        "plan_ledger": [],
        "snapshots": {"root": {"text_memory": ["ancestor mem"]}},
    }
    view = mcts_mod.resolve_snapshot(state, ["root", "brand_new_child"], enabled=True)
    assert view["text_memory"] == ["ancestor mem"]


def test_snapshot_deep_copies_the_graph_so_siblings_cannot_alias():
    g = nx.DiGraph()
    g.add_edge("a", "b", relation={"r1"})
    snap = mcts_mod._snapshot_from(
        {
            "text_memory": ["m"],
            "graph_memory": g,
            "entity_dict": {},
            "plan": "p",
            "plan_version": 1,
            "plan_ledger": [],
        }
    )
    snap["graph_memory"].edges["a", "b"]["relation"].add("leaked")
    assert g.edges["a", "b"]["relation"] == {"r1"}


def test_branch_local_memory_defaults_off_to_preserve_documented_parity():
    """The module docstring documents shared-by-reference memory as coe parity, so
    the flag must default off — it is what makes the two regimes comparable."""
    cfg = _config()
    assert cfg.search.mcts.branch_local_memory is False


# ──────────────────────────────────────────────────────────────────────────────
# ReAct-agent prompt budget
# ──────────────────────────────────────────────────────────────────────────────


def _msgs():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    return [
        SystemMessage(content="S" * 100),
        HumanMessage(content="H" * 100),
        AIMessage(content="A1" * 50),
        ToolMessage(content="T1" * 500, tool_call_id="1"),
        ToolMessage(content="T2" * 500, tool_call_id="2"),
        AIMessage(content="A2" * 50),
        ToolMessage(content="T3" * 50, tool_call_id="3"),
    ]


def test_agent_prompt_under_budget_is_untouched():
    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    msgs = _msgs()
    assert trim_messages_to_char_budget(msgs, 10_000) == msgs


def test_agent_prompt_over_budget_drops_oldest_droppable_messages():
    """``execute_role_lc`` guards single-shot calls, but a ReAct loop appends a
    tool result per iteration and re-calls the model — nothing else bounds it."""
    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    msgs = _msgs()
    trimmed = trim_messages_to_char_budget(msgs, 600)
    assert len(trimmed) < len(msgs)
    # The system prompt and the original request always survive: an agent that
    # forgets its own task fails worse than one that forgets an observation.
    assert trimmed[0].type == "system"
    assert any(m.type == "human" for m in trimmed)
    # The newest turns survive — they are what the next tool call depends on.
    assert trimmed[-1].content == msgs[-1].content


def test_budget_trim_never_silently_mangles_a_message():
    """Dropping is preferred over truncating, and any truncation is marked.

    A tool result chopped mid-JSON is worse than one that is absent, so whole
    messages go first. Truncation only happens as a last resort on an undroppable
    giant, and when it does the message carries the truncation notice — so a
    downstream reader can always tell a partial message from a complete one.
    """
    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    msgs = _msgs()
    trimmed = trim_messages_to_char_budget(msgs, 600)
    originals = {m.content for m in msgs}
    for m in trimmed:
        assert m.content in originals or "input truncated" in m.content


def test_budget_trim_gives_up_rather_than_dropping_the_request():
    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = [SystemMessage(content="S" * 5000), HumanMessage(content="H" * 5000)]
    # Nothing droppable, so it returns the input rather than mangling it.
    assert trim_messages_to_char_budget(msgs, 100) == msgs


def test_budget_middleware_is_registered_on_both_react_agents():
    """These two agents are the only unbounded prompt paths in the system."""
    import inspect

    from langgraph_coe.graphs import kg_search, web_research

    for module in (kg_search, web_research):
        src = inspect.getsource(module)
        assert "make_budget_middleware" in src, (
            f"{module.__name__} must bound its ReAct agent's prompt"
        )


def test_budget_trim_truncates_an_undroppable_giant_as_a_last_resort():
    """The observed live failure: after dropping every eligible message the guard
    was still 128k chars over, because the agent's *request* embeds the whole
    accumulated memory and must not be dropped. Shrinking it beats letting the
    provider reject the call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    msgs = [SystemMessage(content="S" * 100), HumanMessage(content="H" * 50_000)]
    trimmed = trim_messages_to_char_budget(msgs, 5_000)
    assert sum(len(m.content) for m in trimmed) <= 5_000
    # The system prompt is untouched: a truncated instruction produces
    # confidently wrong tool calls rather than a visible failure.
    assert trimmed[0].content == "S" * 100
    assert "input truncated" in trimmed[1].content
    # Head and tail both survive, so a leading instruction and a trailing
    # constraint are each still present.
    assert trimmed[1].content.startswith("H")
    assert trimmed[1].content.endswith("H")


def test_budget_trim_handles_several_undroppable_giants():
    """Truncating only the largest left the prompt over budget in the live run
    ("still 122054 chars over ... after dropping and truncating"), so the pass
    iterates until it fits or stops making progress."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from langgraph_coe.graphs._budget_middleware import trim_messages_to_char_budget

    msgs = [
        SystemMessage(content="S" * 200),
        HumanMessage(content="H" * 80_000),
        HumanMessage(content="J" * 80_000),
    ]
    trimmed = trim_messages_to_char_budget(msgs, 48_000)
    assert sum(len(m.content) for m in trimmed) <= 48_000
    assert trimmed[0].content == "S" * 200


def test_every_direct_guard_call_passes_the_configured_chars_per_token():
    """``chars_per_token`` is per-tier, so a direct guard call that omits it
    silently reverts to the 3.0 default — which over-estimates how much fits and
    let a small-window model overflow even with the guard in place."""
    import inspect
    import re

    from langgraph_coe.graphs import kg_search

    for module in (kg_search,):
        src = inspect.getsource(module)
        for call in re.findall(
            r"_truncate_messages_to_budget\((.*?)\n\s*\)", src, re.DOTALL
        ):
            assert "get_chars_per_token" in call, (
                f"{module.__name__} calls the guard without the tier's "
                f"chars_per_token: {call.strip()[:120]}"
            )


def test_plan_gate_refreshes_the_node_snapshot_with_its_new_ledger():
    """``mem_update`` commits the snapshot before ``plan_gate`` computes the
    ledger, so without this refresh the snapshot keeps the previous iteration's
    bookkeeping and the next selection through the subtree re-opens closed
    intents."""
    import inspect

    src = inspect.getsource(mcts_mod.build_mcts_graph)
    gate = src[src.index("async def plan_gate") :]
    gate = gate[: gate.index("async def synthesize")]
    assert '"snapshots"' in gate, (
        "plan_gate must re-emit the node snapshot with the fresh plan_ledger"
    )
