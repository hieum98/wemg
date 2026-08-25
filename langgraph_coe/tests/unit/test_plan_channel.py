"""Plan channel: UPDATE, contested/falsified discharge, and the REPLAN gate.

A *plan* is prose stating what must be found out. It conditions
``SUBQUESTION_GENERATOR`` / ``SELF_CORRECTOR`` through a typed prompt field and
never enters ``text_memory``. Two operations act on it:

* **UPDATE** — deterministic, no LLM: an intent closes and its resolved referent
  surfaces via the ``intermediate_answer`` slot.
* **REPLAN** — one PLANNER call, fired on *contested discharge* (two or more
  distinct-QID referents survive for one intent) or *falsified discharge* (a
  ``[Retrieval]`` fact the plan cited was evicted as ``contradicted``).

The pure helpers are tested directly because they are the whole trigger: QID
identity is the discriminator, so there is no threshold to tune and the
classification is fully determined by the ledger.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import networkx as nx
import pytest

from langgraph_coe import roles as roles_mod
from langgraph_coe.graphs import cot as cot_mod

from .test_cot_graph import (
    CompiledGraphSpy,
    RoleExecutorSpy,
    _install_graph_spies,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────


def _plan_config(*, enabled: bool = True, replan_max: int = 0):
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.reranker.enabled = False
    cfg.reranker.top_k = 3
    cfg.web_search.enabled = False
    cfg.search.plan.enabled = enabled
    cfg.search.plan.replan_max = replan_max
    cfg.search.plan.replan_min_depth_headroom = 0
    return cfg


def _registry(cfg: Any):
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(cfg.llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    return registry


def _entity(qid: str, label: str) -> roles_mod.WikidataEntity:
    return roles_mod.WikidataEntity(qid=qid, label=label, description="")


def _ledger(*intents: str, premises: Sequence[str] = ()) -> List[Dict[str, Any]]:
    return cot_mod.build_plan_ledger(list(intents), list(premises))


class PlanRoleExecutorSpy(RoleExecutorSpy):
    """``RoleExecutorSpy`` plus a canned PLANNER queue."""

    def __init__(self, *, plan_outputs: Sequence[Any] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._plan_outputs = list(plan_outputs)

    async def __call__(
        self,
        registry: Any,
        role: roles_mod.Role,
        input_data: Any,
        n: int = 1,
        tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        if role.name == "planner":
            self.calls.append(
                {
                    "role": role.name,
                    "input": input_data,
                    "n": n,
                    "tier_override": tier_override,
                }
            )
            if not self._plan_outputs:
                return (
                    roles_mod.PlanOutput(plan="Fallback plan.", intents=["Find X."]),
                    {},
                )
            return self._plan_outputs.pop(0), {}
        return await super().__call__(registry, role, input_data, n, tier_override)


def _plan_out(
    plan: str, intents: Sequence[str], premises: Sequence[str] = ()
) -> roles_mod.PlanOutput:
    return roles_mod.PlanOutput(
        plan=plan, intents=list(intents), premises=list(premises) or None
    )


def _subq_out_with_intent(
    *,
    answerable: bool,
    subquestions: Sequence[str] = (),
    needs_kg: Sequence[bool] | None = None,
    serves_intent: Sequence[int] | None = None,
) -> roles_mod.SubquestionGenerationOutput:
    return roles_mod.SubquestionGenerationOutput(
        is_answerable=answerable,
        subquestions=list(subquestions),
        needs_kg=list(needs_kg) if needs_kg is not None else None,
        serves_intent=list(serves_intent) if serves_intent is not None else None,
    )


def _plan_state(*, max_depth: int = 1, entity_dict: Dict[str, Any] | None = None):
    return {
        "question": "Who is the father of the father of computer science?",
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
        "entity_dict": dict(entity_dict or {}),
        "plan": "",
        "plan_version": 0,
        "plan_ledger": [],
        "plan_action": "none",
        "plan_action_log": [],
        "plan_frozen": False,
        "final_answer": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Binding resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_resolve_binding_qids_prefers_longest_label():
    """A longer label must consume a shorter one nested inside it.

    Without longest-first matching, "Alan Turing" would also match a linked
    "Turing", so one referent would resolve to two QIDs and manufacture a
    contested discharge out of nothing.
    """
    label_to_qid = {"alan turing": "Q7251", "turing": "Q99999"}
    assert cot_mod.resolve_binding_qids("Alan Turing was a mathematician.", label_to_qid) == [
        "Q7251"
    ]
    # A standalone shorter label still resolves on its own.
    assert cot_mod.resolve_binding_qids("Turing machine", label_to_qid) == ["Q99999"]


def test_one_answer_proposes_exactly_one_referent():
    """Regression: the trigger must not fire on answer *verbosity*.

    A prose answer naming supporting entities alongside its subject mentions
    several linked entities. Taking all of them made a single unambiguous answer
    look like several competing referents — observed live, where one Turing answer
    produced five "competing" bindings and drove two spurious replans.
    """
    label_to_qid = {
        "alan mathison turing": "Q7251",
        "turing machine": "Q163310",
        "theoretical computer science": "Q2878974",
    }
    verbose = (
        "Alan Mathison Turing is most widely referred to as the father of computer "
        "science. His contributions include the Turing machine and the "
        "formalization of theoretical computer science."
    )
    # All three are present...
    assert len(cot_mod.resolve_binding_qids(verbose, label_to_qid)) == 3
    # ...but only the earliest-mentioned one is the referent.
    assert cot_mod.resolve_primary_qid(verbose, label_to_qid) == "Q7251"

    ledger = _ledger("Who is called the father of computer science?")
    ledger = cot_mod.apply_bindings(ledger, [(0, verbose)], label_to_qid, hop=0)
    assert len(ledger[0]["bindings"]) == 1
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED, (
        "one verbose answer must close the intent, not contest it"
    )
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_UPDATE


def test_resolve_primary_qid_is_none_without_a_linked_entity():
    assert cot_mod.resolve_primary_qid("no entities here", {"paris": "Q90"}) is None


def test_resolve_binding_qids_ignores_unlinked_text():
    assert cot_mod.resolve_binding_qids("Some unlinked prose.", {"paris": "Q90"}) == []
    assert cot_mod.resolve_binding_qids("", {"paris": "Q90"}) == []


def test_entity_label_to_qid_reverses_the_qid_keyed_store():
    """``entity_dict`` is keyed by QID, so a surface form needs a reverse index."""
    entity_dict = {"Q7251": _entity("Q7251", "Alan Turing")}
    assert cot_mod._entity_label_to_qid(entity_dict) == {"alan turing": "Q7251"}


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE: one distinct QID closes an intent
# ──────────────────────────────────────────────────────────────────────────────


def test_single_qid_closes_the_intent_and_signals_update():
    ledger = _ledger("Who is called the father of computer science?")
    label_to_qid = {"alan turing": "Q7251"}
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "Alan Turing is known as the father of computer science.")],
        label_to_qid, hop=0,
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED
    assert ledger[0]["closed_at"] == 0
    action, idx, competing = cot_mod.classify_discharge(ledger)
    assert (action, idx, competing) == (cot_mod.PLAN_ACTION_UPDATE, 0, [])


def test_paraphrases_of_one_referent_do_not_compete():
    """QID identity, not string identity, is the discriminator.

    Two differently-worded answers naming the same entity share a QID, so the
    intent still closes. This is what keeps paraphrase noise out of the trigger.
    """
    ledger = _ledger("Who founded geometry?")
    label_to_qid = {"euclid": "Q8747"}
    ledger = cot_mod.apply_bindings(
        ledger,
        [(0, "Euclid founded geometry."), (0, "The founder was Euclid of Alexandria.")],
        label_to_qid,
        hop=0,
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_UPDATE


def test_unattributed_binding_closes_nothing():
    """An answer with no ``serves_intent`` must not be guessed onto an intent."""
    ledger = _ledger("Who is called the father of computer science?")
    ledger = cot_mod.apply_bindings(
        ledger, [(None, "Alan Turing.")], {"alan turing": "Q7251"}, hop=0
    )
    assert ledger[0]["status"] == cot_mod.INTENT_OPEN
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_NONE


def test_binding_with_no_linked_entity_is_ignored():
    """Non-null-QID gating: unresolvable prose contributes no binding."""
    ledger = _ledger("Who is called the father of computer science?")
    ledger = cot_mod.apply_bindings(ledger, [(0, "It is unclear.")], {}, hop=0)
    assert ledger[0]["status"] == cot_mod.INTENT_OPEN
    assert ledger[0]["bindings"] == []


def test_latest_intermediate_answer_surfaces_the_newest_closure():
    ledger = _ledger("Intent A", "Intent B")
    label_to_qid = {"alan turing": "Q7251", "euclid": "Q8747"}
    ledger = cot_mod.apply_bindings(ledger, [(0, "Alan Turing")], label_to_qid, hop=0)
    ledger = cot_mod.apply_bindings(ledger, [(1, "Euclid")], label_to_qid, hop=1)
    surfaced = cot_mod.latest_intermediate_answer(ledger)
    assert "Euclid" in surfaced and "Intent B" in surfaced


def test_latest_intermediate_answer_is_none_before_any_closure():
    assert cot_mod.latest_intermediate_answer(_ledger("Intent A")) is None


# ──────────────────────────────────────────────────────────────────────────────
# Contested discharge
# ──────────────────────────────────────────────────────────────────────────────


def test_two_distinct_qids_contest_the_intent_and_signal_replan():
    """The core trigger: "father of computer science" → Turing vs Babbage.

    Both bind, neither wins, so the intent's closure is under-determined. That is
    a fact about the plan's bookkeeping, not a claim about the world.
    """
    ledger = _ledger("Who is called the father of computer science?")
    label_to_qid = {"alan turing": "Q7251", "charles babbage": "Q46633"}
    ledger = cot_mod.apply_bindings(
        ledger,
        [(0, "Alan Turing is the father of computer science."),
         (0, "Charles Babbage is the father of computer science.")],
        label_to_qid,
        hop=0,
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CONTESTED
    assert ledger[0]["closed_at"] is None
    action, idx, competing = cot_mod.classify_discharge(ledger)
    assert action == cot_mod.PLAN_ACTION_REPLAN
    assert idx == 0
    assert len(competing) == 2


def test_contested_beats_update_when_both_present():
    """A contested intent must win over another intent's routine closure."""
    ledger = _ledger("Intent A", "Intent B")
    label_to_qid = {"euclid": "Q8747", "alan turing": "Q7251", "charles babbage": "Q46633"}
    ledger = cot_mod.apply_bindings(ledger, [(0, "Euclid")], label_to_qid, hop=0)
    ledger = cot_mod.apply_bindings(
        ledger, [(1, "Alan Turing"), (1, "Charles Babbage")], label_to_qid, hop=0
    )
    action, idx, _ = cot_mod.classify_discharge(ledger)
    assert action == cot_mod.PLAN_ACTION_REPLAN
    assert idx == 1


def test_closed_intent_is_not_reopened_by_a_later_binding():
    """Closure is a commitment; a later hop must not silently contest it."""
    ledger = _ledger("Intent A")
    ledger = cot_mod.apply_bindings(ledger, [(0, "Euclid")], {"euclid": "Q8747"}, hop=0)
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "Charles Babbage")], {"charles babbage": "Q46633"}, hop=1
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED


# ──────────────────────────────────────────────────────────────────────────────
# Falsified discharge
# ──────────────────────────────────────────────────────────────────────────────


def test_contradicted_premise_falsifies_the_intent():
    premise = "Alan Turing is known as the father of computer science."
    ledger = _ledger("Find that person's father.", premises=[premise])
    ledger = cot_mod.apply_retractions(
        ledger, [{"content": premise, "reason": "contradicted"}]
    )
    assert ledger[0]["falsified"] == premise
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_REPLAN


@pytest.mark.parametrize(
    "reason", ["irrelevant", "duplicate", "hop_filtered", "superseded"]
)
def test_housekeeping_evictions_do_not_falsify(reason: str):
    """Only ``contradicted`` bears on the plan.

    The other reasons mean the consolidator tidied memory, not that a claim the
    plan leaned on turned out to be false.
    """
    premise = "Alan Turing is known as the father of computer science."
    ledger = _ledger("Find that person's father.", premises=[premise])
    ledger = cot_mod.apply_retractions(ledger, [{"content": premise, "reason": reason}])
    assert "falsified" not in ledger[0]
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_NONE


def test_retraction_of_an_uncited_fact_does_not_falsify():
    ledger = _ledger("Find that person's father.", premises=["A cited premise."])
    ledger = cot_mod.apply_retractions(
        ledger, [{"content": "Some unrelated fact.", "reason": "contradicted"}]
    )
    assert "falsified" not in ledger[0]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt rendering / channel isolation
# ──────────────────────────────────────────────────────────────────────────────


def test_rendered_plan_marks_intent_status_without_revealing_bindings():
    """Status is bookkeeping; a bound *value* must not enter the plan channel.

    A referent written into the plan prose would be a world-claim with no
    provenance tag, no hop tag and no eviction path.
    """
    ledger = _ledger("Who is called the father of computer science?", "Find their father.")
    ledger = cot_mod.apply_bindings(ledger, [(0, "Alan Turing")], {"alan turing": "Q7251"}, hop=0)
    rendered = cot_mod.render_plan_for_prompt("Establish X, then Y.", ledger)
    assert "[resolved]" in rendered
    assert "[open]" in rendered
    assert "Alan Turing" not in rendered


def test_rendered_plan_is_empty_when_there_is_no_plan():
    assert cot_mod.render_plan_for_prompt("", []) == ""


def test_plan_renders_last_in_the_subquestion_prompt():
    """Position matters: the input guard trims the middle of an oversized payload.

    Rendering the plan last means bulky accumulated memory is what gets dropped,
    not the plan.
    """
    text = str(
        roles_mod.SubquestionGenerationInput(
            question="Q", context="C" * 50, plan="THE-PLAN"
        )
    )
    assert text.rindex("THE-PLAN") > text.rindex("C" * 50)


def test_absent_plan_is_omitted_from_prompts_entirely():
    """An unset plan must not render as ``plan:\\nNone``."""
    assert "plan" not in str(
        roles_mod.SubquestionGenerationInput(question="Q", context="C")
    )
    assert "plan" not in str(
        roles_mod.SelfCorrectionInput(question="Q", proposed_answer="A", context="C")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Graph wiring
# ──────────────────────────────────────────────────────────────────────────────


def test_plan_disabled_graph_has_no_plan_nodes():
    """A0 must be structurally identical to the pre-plan graph.

    Not merely inert: an extra node is an extra superstep, and the ablation is
    only meaningful if the baseline arm is unchanged.
    """
    cfg = _plan_config(enabled=False)
    nodes = set(cot_mod.build_cot_graph(_registry(cfg), cfg).get_graph().nodes)
    assert not {"gen_plan", "plan_gate", "replan"} & nodes


def test_plan_enabled_graph_adds_the_plan_nodes():
    cfg = _plan_config(enabled=True)
    nodes = set(cot_mod.build_cot_graph(_registry(cfg), cfg).get_graph().nodes)
    assert {"gen_plan", "plan_gate", "replan"} <= nodes


@pytest.mark.asyncio
async def test_plan_is_generated_once_and_injected_into_subq(monkeypatch):
    cfg = _plan_config(enabled=True)
    executor = PlanRoleExecutorSpy(
        plan_outputs=[
            _plan_out(
                "First establish who is called the father of computer science.",
                ["Who is called the father of computer science?"],
            )
        ],
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who is called the father of computer science?"],
                needs_kg=[True],
                serves_intent=[0],
            ),
            _subq_out_with_intent(answerable=True),
        ],
        answers=["Alan Turing."],
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)
    monkeypatch.setattr("langgraph_coe.graphs.mcts.execute_role_lc", executor, raising=False)

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_plan_state(max_depth=2))

    planner_calls = [c for c in executor.calls if c["role"] == "planner"]
    assert len(planner_calls) == 1, "one PLANNER call per question"
    assert final["plan_version"] == 1
    assert len(final["plan_ledger"]) == 1

    subq_inputs = executor.role_inputs("subquestion_generator")
    assert subq_inputs[0].plan is not None
    assert "father of computer science" in subq_inputs[0].plan


@pytest.mark.asyncio
async def test_plan_text_never_reaches_memory(monkeypatch):
    """The channel constraint, asserted mechanically.

    An interrogative in ``text_memory`` is picked up by ``_reverify_memory`` as a
    retrieval query, then reaches the verifier as grounding and the synthesizer as
    a candidate answer.
    """
    cfg = _plan_config(enabled=True)
    plan_text = "UNIQUE-PLAN-SENTINEL: establish who founded geometry."
    executor = PlanRoleExecutorSpy(
        plan_outputs=[_plan_out(plan_text, ["Who founded geometry?"])],
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who founded geometry?"],
                needs_kg=[True],
                serves_intent=[0],
            ),
            _subq_out_with_intent(answerable=True),
        ],
        answers=["Euclid."],
    )
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["[Retrieval]: Euclid founded geometry."],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {},
            "retractions": [],
        }
    )
    _install_graph_spies(
        monkeypatch, cot_mod, executor=executor, memory_graph=memory_graph
    )

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_plan_state(max_depth=2))

    for payload in memory_graph.calls:
        for key in ("new_text_items", "new_retrieval_items", "new_critique_items"):
            joined = "\n".join(map(str, payload.get(key) or []))
            assert "UNIQUE-PLAN-SENTINEL" not in joined, f"plan leaked into {key}"
    assert "UNIQUE-PLAN-SENTINEL" not in "\n".join(final.get("text_memory") or [])

    synth_inputs = executor.role_inputs("final_answer_synthesizer")
    for inp in synth_inputs:
        assert "UNIQUE-PLAN-SENTINEL" not in "\n".join(map(str, inp.candidate_answers))
        # The plan is deliberately absent from synthesis: there the only question
        # is which candidate is true, not what remains to be found out.
        assert not hasattr(inp, "plan") or getattr(inp, "plan", None) is None


@pytest.mark.asyncio
async def test_log_only_mode_records_replan_without_taking_the_edge(monkeypatch):
    """``replan_max=0`` measures the trigger's fire rate before arming it."""
    cfg = _plan_config(enabled=True, replan_max=0)
    executor = PlanRoleExecutorSpy(
        plan_outputs=[
            _plan_out("Establish the father of computer science.",
                      ["Who is called the father of computer science?"])
        ],
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who is called the father of computer science?",
                              "Who else is called that?"],
                needs_kg=[True, True],
                serves_intent=[0, 0],
            ),
            _subq_out_with_intent(answerable=True),
        ],
        answers=["Alan Turing.", "Charles Babbage."],
    )
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["[Retrieval]: contested."],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {
                "Q7251": _entity("Q7251", "Alan Turing"),
                "Q46633": _entity("Q46633", "Charles Babbage"),
            },
            "retractions": [],
        }
    )
    _install_graph_spies(
        monkeypatch, cot_mod, executor=executor, memory_graph=memory_graph
    )

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_plan_state(max_depth=3))

    log = final.get("plan_action_log") or []
    assert any(e["action"] == cot_mod.PLAN_ACTION_REPLAN for e in log), (
        "contested discharge must be recorded even when the router is inert"
    )
    assert all(e["armed"] is False for e in log)
    # Exactly one PLANNER call: the initial plan. No replan was applied.
    assert len([c for c in executor.calls if c["role"] == "planner"]) == 1
    assert final["plan_version"] == 1


@pytest.mark.asyncio
async def test_armed_mode_applies_a_replan_and_bumps_the_version(monkeypatch):
    cfg = _plan_config(enabled=True, replan_max=2)
    executor = PlanRoleExecutorSpy(
        plan_outputs=[
            _plan_out("Establish the father of computer science.",
                      ["Who is called the father of computer science?"]),
            _plan_out(
                "Distinguish the two candidates by field of contribution.",
                ["Which of the two is credited for the theoretical foundation?"],
            ),
        ],
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who is called the father of computer science?",
                              "Who else is called that?"],
                needs_kg=[True, True],
                serves_intent=[0, 0],
            ),
            _subq_out_with_intent(answerable=True),
        ],
        answers=["Alan Turing.", "Charles Babbage."],
    )
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["[Retrieval]: contested."],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {
                "Q7251": _entity("Q7251", "Alan Turing"),
                "Q46633": _entity("Q46633", "Charles Babbage"),
            },
            "retractions": [],
        }
    )
    _install_graph_spies(
        monkeypatch, cot_mod, executor=executor, memory_graph=memory_graph
    )

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    final = await graph.ainvoke(_plan_state(max_depth=3))

    planner_calls = [c for c in executor.calls if c["role"] == "planner"]
    assert len(planner_calls) == 2, "initial plan + one replan"
    assert final["plan_version"] == 2

    replan_input = planner_calls[1]["input"]
    assert replan_input.current_plan, "the replanner must see the plan it revises"
    assert replan_input.failure, "the failure must be stated mechanically"
    assert replan_input.competing_bindings, "competing referents must be supplied"
    assert replan_input.attempts, "the attempt ledger prevents rewriting the same plan"
    # Surface forms only — never asserted as the answer.
    rendered = str(replan_input)
    assert "competing_bindings" in rendered


@pytest.mark.asyncio
async def test_frozen_plan_blocks_replan_inside_a_rollout(monkeypatch):
    """A rollout may observe a replan signal but must not act on it.

    The plan belongs to the MCTS node that spawned the rollout; revising it here
    would silently fork the parent's plan.
    """
    cfg = _plan_config(enabled=True, replan_max=2)
    executor = PlanRoleExecutorSpy(
        plan_outputs=[],  # a frozen rollout must not call the PLANNER at all
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who is called the father of computer science?",
                              "Who else is called that?"],
                needs_kg=[True, True],
                serves_intent=[0, 0],
            ),
            _subq_out_with_intent(answerable=True),
        ],
        answers=["Alan Turing.", "Charles Babbage."],
    )
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["[Retrieval]: contested."],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {
                "Q7251": _entity("Q7251", "Alan Turing"),
                "Q46633": _entity("Q46633", "Charles Babbage"),
            },
            "retractions": [],
        }
    )
    _install_graph_spies(
        monkeypatch, cot_mod, executor=executor, memory_graph=memory_graph
    )

    graph = cot_mod.build_cot_graph(_registry(cfg), cfg)
    state = _plan_state(max_depth=3)
    state.update(
        {
            "plan": "Inherited plan from the tree node.",
            "plan_version": 1,
            "plan_ledger": _ledger("Who is called the father of computer science?"),
            "plan_frozen": True,
        }
    )
    final = await graph.ainvoke(state)

    assert not [c for c in executor.calls if c["role"] == "planner"], (
        "a frozen rollout must neither regenerate nor revise the plan"
    )
    assert final["plan"] == "Inherited plan from the tree node."
    assert final["plan_version"] == 1
    log = final.get("plan_action_log") or []
    assert any(e["action"] == cot_mod.PLAN_ACTION_REPLAN for e in log), (
        "the signal is still recorded — only the action is suppressed"
    )
