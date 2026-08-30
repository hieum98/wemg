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

import inspect
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


def test_rendered_plan_cites_a_retrieval_grounded_binding():
    """The plan may *cite* a verified fact, not *originate* one.

    Showing the value is safe because the render is derived from the ledger on every
    call: the binding carries a QID, a hop, and an eviction path (a retracted premise
    falsifies the intent), and a retraction drops it from the next render. Writing it
    into the stored plan prose instead would strand a world-claim with none of those.
    """
    ledger = _ledger("Who is called the father of computer science?", "Find their father.")
    ledger = cot_mod.apply_bindings(
        ledger,
        [(0, "Alan Turing")],
        {"alan turing": "Q7251"},
        hop=0,
        retrieval_memory=[
            "[hop=1] [Retrieval]: Alan Turing is called the father of computer science."
        ],
    )
    assert ledger[0]["bindings"][0]["grounded"] is True
    rendered = cot_mod.render_plan_for_prompt("Establish X, then Y.", ledger)
    assert "[resolved: Alan Turing]" in rendered
    assert "[open]" in rendered
    # The stored prose is untouched — only the derived view carries the value.
    assert "Alan Turing" not in "Establish X, then Y."


def test_an_unverified_binding_is_not_presented_as_established():
    """An intent closed on the model's own inference renders without its value.

    Otherwise the plan would assert a guess, which is exactly the door-with-no-
    verifier case the planning/reasoning split exists to close.
    """
    ledger = _ledger("Who is called the father of computer science?")
    ledger = cot_mod.apply_bindings(
        ledger,
        [(0, "Alan Turing")],
        {"alan turing": "Q7251"},
        hop=0,
        # Only a [System Prediction] line mentions it — no retrieval corroboration.
        retrieval_memory=["[System Prediction]: Alan Turing invented the computer."],
    )
    assert ledger[0]["bindings"][0]["grounded"] is False
    rendered = cot_mod.render_plan_for_prompt("Establish X.", ledger)
    assert "[resolved, unverified]" in rendered
    assert "Alan Turing" not in rendered


def test_all_resolved_intents_keep_their_values_in_the_render():
    """``intermediate_answer`` carries only the *latest* closure, so on a 3-hop plan
    the earlier bindings would vanish from the prompt entirely. The render is what
    keeps them all present."""
    mem = [
        "[Retrieval]: In Utero is the album.",
        "[Retrieval]: It was released on September 13, 1993.",
    ]
    ledger = _ledger("Identify the album.", "Find its release date.", "Find its producer.")
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "In Utero")], {"in utero": "Q222001"}, hop=0, retrieval_memory=mem
    )
    ledger = cot_mod.apply_bindings(
        ledger, [(1, "September 13, 1993")], {}, hop=1, retrieval_memory=mem
    )
    rendered = cot_mod.render_plan_for_prompt("Identify, date, then attribute.", ledger)
    assert "[resolved: In Utero]" in rendered
    assert "[resolved: September 13, 1993]" in rendered
    assert "[open] Find its producer." in rendered
    # Whereas the single-slot anchor only carries the newest one.
    assert "In Utero" not in (cot_mod.latest_intermediate_answer(ledger) or "")


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


# ──────────────────────────────────────────────────────────────────────────────
# Widened trigger: literal bindings, stalled intents, low confidence
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("September 13, 1993", "september 13 1993"),
        ("13 September 1993", "13 september 1993"),
        ("1993-09-13", "1993-09-13"),
        ("2012", "2012"),
        ("320 km/h", "320 km/h"),
        ("310 square kilometers", "310 square kilometers"),
        ("1045 BC", "1045 bc"),
        # Not anchored at the start → not the referent this answer proposes.
        ("In Utero came out in 1993", None),
        ("Alan Turing", None),
        ("", None),
    ],
)
def test_literal_referents_are_recognized(text, expected):
    """The QID discriminator is blind to ordinal and temporal hops.

    "In what year was the tallest lattice tower completed?" binds a *year*, which
    links to no entity — so two competing years both resolve to no QID and the
    intent is never recorded as contested. 56% of the committed eval set is
    ordinal-then-chained, so that is a large blind spot, not an edge case.
    """
    assert cot_mod.resolve_primary_literal(text) == expected


def test_two_competing_years_contest_an_intent():
    ledger = _ledger("In what year was the tallest lattice tower completed?")
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "2012"), (0, "1935")], {}, hop=0
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CONTESTED
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_REPLAN


def test_the_same_year_twice_closes_rather_than_contests():
    ledger = _ledger("In what year was it completed?")
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "2012"), (0, "2012, per the tower's records")], {}, hop=0
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED


def test_a_qid_beats_a_literal_for_the_same_answer():
    """An entity match is stronger evidence than a surface-form literal."""
    key = cot_mod.resolve_binding_key("Alan Turing, born 1912", {"alan turing": "Q7251"})
    assert key == "Q7251"
    assert cot_mod.resolve_binding_key("1912", {"alan turing": "Q7251"}) == "lit:1912"


def test_stalled_intent_triggers_a_replan_after_enough_attempts():
    """The only branch that fires when nothing surprising happens.

    Contested and falsified discharge both need an *event*; an intent that quietly
    returns nothing produces none, so without this the plan is revised never rather
    than rarely.
    """
    ledger = _ledger("Retrieve the complete ranked list of lattice towers.")
    ledger[0]["attempts"] = [
        {"query": "tallest lattice towers", "n_facts": 0, "hop": 0},
        {"query": "lattice tower rankings", "n_facts": 0, "hop": 1},
    ]
    ledger = cot_mod.mark_stalled_intents(ledger, max_attempts=2)
    assert ledger[0]["stalled"] is True
    assert "no evidence returned" in ledger[0]["stall_reason"]
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_REPLAN


def test_stall_reason_distinguishes_no_evidence_from_no_referent():
    """The two need different repairs: a different route vs a different question."""
    ledger = _ledger("Who is the only cruise line flying the American flag?")
    ledger[0]["attempts"] = [
        {"query": "american flag cruise line", "n_facts": 5, "hop": 0},
        {"query": "us flagged cruise ships", "n_facts": 3, "hop": 1},
    ]
    ledger = cot_mod.mark_stalled_intents(ledger, max_attempts=2)
    assert "no referent resolved" in ledger[0]["stall_reason"]
    assert "presuppose" in ledger[0]["stall_reason"]


def test_a_closed_intent_is_never_marked_stalled():
    ledger = _ledger("Who founded geometry?")
    ledger = cot_mod.apply_bindings(ledger, [(0, "Euclid")], {"euclid": "Q8747"}, hop=0)
    ledger[0]["attempts"] = [{"query": "q", "n_facts": 0, "hop": 0}] * 5
    ledger = cot_mod.mark_stalled_intents(ledger, max_attempts=2)
    assert "stalled" not in ledger[0]


def test_precedence_is_contested_then_falsified_then_stalled():
    """Most specific failure first: discriminating between referents subsumes
    re-establishing a premise, which subsumes finding another route."""
    premise = "A cited premise."
    ledger = cot_mod.build_plan_ledger(["A", "B", "C"], [premise])
    # C stalls, B is falsified, A is contested.
    ledger[2]["attempts"] = [{"query": "q", "n_facts": 0, "hop": 0}] * 3
    ledger = cot_mod.mark_stalled_intents(ledger, max_attempts=2)
    ledger = cot_mod.apply_retractions(
        ledger, [{"content": premise, "reason": "contradicted"}]
    )
    ledger[0] = {
        **ledger[0],
        "status": cot_mod.INTENT_CONTESTED,
        "bindings": [{"surface": "X", "qid": "Q1"}, {"surface": "Y", "qid": "Q2"}],
    }
    action, idx, _ = cot_mod.classify_discharge(ledger)
    assert (action, idx) == (cot_mod.PLAN_ACTION_REPLAN, 0)


def test_failure_text_prescribes_the_matching_repair():
    """Conflating the three failure kinds is how a replan loops."""
    contested = cot_mod._describe_failure(
        {"intent": "I", "status": cot_mod.INTENT_CONTESTED}, ["a", "b"]
    )
    assert "discriminates" in contested and "do not re-ask" in contested.lower()

    falsified = cot_mod._describe_failure({"intent": "I", "falsified": "F"}, [])
    assert "contradicted" in falsified and "Re-establish" in falsified

    stalled = cot_mod._describe_failure(
        {"intent": "I", "stalled": True, "stall_reason": "R"}, []
    )
    # Contraction, not revision: re-asking in different words will keep failing.
    assert "presupposition" in stalled
    assert "what IS the case" in stalled
    assert "rather than re-asking" in stalled


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE has to actually reach a prompt
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "subq,dropped",
    [
        ("When was the album [album name from previous answer] released?", True),
        ("Who was [the person identified above]?", True),
        ("Find [insert name]'s birthplace", True),
        ("When did In Utero come out?", False),
        ("Who was the father of Alan Turing?", False),
    ],
)
def test_uninstantiated_subquestions_are_dropped(subq, dropped):
    """Observed live on a real dataset row: with no ``intermediate_answer`` to bind,
    the generator emitted the literal string "[album name from previous answer]" and
    that went to retrieval as a query.

    Retrieving on it is worse than not asking: the placeholder becomes part of the
    embedding query, returns noise, and the intent closes on that noise — so the
    dependency is never resolved. Dropping it keeps the intent open for the next hop.
    """
    assert cot_mod.is_uninstantiated(subq) is dropped

    out = roles_mod.SubquestionGenerationOutput(
        is_answerable=False, subquestions=[subq], needs_kg=[True], serves_intent=[0]
    )
    pooled = cot_mod.pool_subquestions([out])
    if dropped:
        assert pooled.subquestions == []
        assert pooled.n_uninstantiated == 1
    else:
        assert pooled.subquestions == [subq]
        assert pooled.n_uninstantiated == 0


def test_dropping_a_placeholder_keeps_the_parallel_arrays_aligned():
    """The dropped entry must not shift ``needs_kg`` / ``serves_intent``."""
    out = roles_mod.SubquestionGenerationOutput(
        is_answerable=False,
        subquestions=[
            "What is the band's second most selling album?",
            "When was [that album] released?",
            "Who produced it?",
        ],
        needs_kg=[True, False, True],
        serves_intent=[0, 1, 2],
    )
    pooled = cot_mod.pool_subquestions([out])
    assert pooled.subquestions == [
        "What is the band's second most selling album?",
        "Who produced it?",
    ]
    assert pooled.needs_kg == [True, True]
    assert pooled.serves_intent == [0, 2]
    assert pooled.n_uninstantiated == 1


def test_reworded_twins_are_dropped_but_different_properties_survive():
    """The measured distinction the threshold has to respect.

    On the d1 run, subquestions differing only in tense or word order scored
    0.92-0.97 and returned *identical* fact counts, while subquestions asking for
    different properties of one entity scored 0.84. Each survivor costs a full
    retrieval fan-out, so the twins are pure waste — but dropping a real question
    costs an entire extra hop, which is worse. Hence a high threshold.
    """
    twins = roles_mod.SubquestionGenerationOutput(
        is_answerable=False,
        subquestions=[
            "What is the original or historical name of Cologne?",
            "What was the original or historical name of Cologne?",  # tense only
            "In what year was Best Buy added to the S&P 500?",
            "What year was Best Buy added to the S&P 500?",  # word order only
        ],
        needs_kg=[True, True, True, True],
        serves_intent=[0, 0, 1, 1],
    )
    pooled = cot_mod.pool_subquestions([twins])
    assert pooled.n_near_duplicate == 2
    assert pooled.subquestions == [
        "What is the original or historical name of Cologne?",
        "In what year was Best Buy added to the S&P 500?",
    ]
    assert pooled.needs_kg == [True, True] and pooled.serves_intent == [0, 1]

    # 0.84 similar, genuinely different retrieval targets — both must survive.
    distinct = roles_mod.SubquestionGenerationOutput(
        is_answerable=False,
        subquestions=[
            "What is the name and location of the tallest lattice tower in the world?",
            "What is the completion year of the tallest lattice tower in the world?",
        ],
        needs_kg=[True, True],
        serves_intent=[0, 1],
    )
    kept = cot_mod.pool_subquestions([distinct])
    assert kept.n_near_duplicate == 0 and len(kept.subquestions) == 2


def test_one_intent_cannot_monopolise_a_hop():
    """The cap that a prompt rule cannot enforce.

    The ``n`` generator completions never see each other, so no instruction stops
    them proposing the same intent repeatedly — measured, three completions once
    produced 10 retrievals for one intent. Two are allowed because an ordinal intent
    has two genuinely different targets (the specific item, and the ranked list).
    """
    # Deliberately *not* twins — max pairwise similarity 0.55, well under the
    # near-duplicate threshold. Otherwise that filter would fire first and this
    # test would pass without the cap ever engaging.
    phrasings = [
        "What is the third fastest bird by maximum airspeed?",
        "Which species holds rank three in avian flight speed records?",
        "List all birds ordered by their top recorded flight velocity.",
        "Among all birds, which comes third when sorted by peak speed?",
        "What bird sits at number 3 on the fastest fliers list?",
        "Which avian species is the third quickest in level flight?",
    ]
    outs = [
        roles_mod.SubquestionGenerationOutput(
            is_answerable=False,
            subquestions=phrasings[i : i + 2],
            needs_kg=[True, True],
            serves_intent=[0, 0],
        )
        for i in (0, 2, 4)
    ]
    pooled = cot_mod.pool_subquestions(outs)
    assert pooled.n_near_duplicate == 0, "must exercise the cap, not the twin filter"
    assert len(pooled.subquestions) == cot_mod._MAX_PER_INTENT
    assert pooled.n_intent_capped == 4
    assert pooled.serves_intent == [0, 0]


def test_unattributed_subquestions_do_not_share_one_budget():
    """A subquestion with no intent carries no claim about *which* gap it fills, so
    a shared cap would make two unrelated gaps compete for one slot."""
    out = roles_mod.SubquestionGenerationOutput(
        is_answerable=False,
        subquestions=[
            "What is the population of Lyon?",
            "When was the Eiffel Tower completed?",
            "Who founded the Bauhaus?",
        ],
        needs_kg=[True, True, True],
        serves_intent=[-1, -1, -1],
    )
    pooled = cot_mod.pool_subquestions([out])
    assert len(pooled.subquestions) == 3 and pooled.n_intent_capped == 0
    assert pooled.serves_intent == [None, None, None]


def test_a_closed_intent_surfaces_its_binding_for_the_next_hop():
    """The mechanism UPDATE exists for, asserted directly.

    Across five live runs ``intermediate_answer`` was never once populated — every
    intent either closed in hop 0 (leaving no later hop to receive it) or never
    closed at all. The unit-level contract still has to hold.
    """
    ledger = _ledger("Identify the album.", "Find its release date.")
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "In Utero")], {"in utero": "Q222001"}, hop=0
    )
    surfaced = cot_mod.latest_intermediate_answer(ledger)
    assert surfaced is not None
    assert "In Utero" in surfaced
    assert "Identify the album." in surfaced
    # An ungrounded binding is still not presented as established.
    rendered = cot_mod.render_plan_for_prompt("Identify the album, then date it.", ledger)
    assert "[resolved, unverified]" in rendered
    assert "In Utero" not in rendered


# ──────────────────────────────────────────────────────────────────────────────
# What must survive a replan
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_attempt_record_survives_a_replan(monkeypatch):
    """``replan`` rebuilds the ledger, so per-intent ``attempts`` reset to empty.

    Sourcing ``already_attempted`` from the ledger therefore left the *second* and
    later replans with no negative record at all — free to re-propose a framing that
    had already yielded nothing, which is exactly the loop the record prevents. It
    now comes from the run-level ``plan_attempts_log``, which no replan clears.
    """
    cfg = _plan_config(enabled=True, replan_max=2)
    executor = PlanRoleExecutorSpy(
        plan_outputs=[
            _plan_out("Establish the referent.", ["Who is called that?"]),
            _plan_out("Discriminate the candidates.", ["Which one, by field?"]),
        ],
        subq_outputs=[
            _subq_out_with_intent(
                answerable=False,
                subquestions=["Who is called that?", "Who else is called that?"],
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

    assert final["plan_version"] == 2, "a replan must have been applied"
    # The run-level log kept both queries even though the ledger was rebuilt.
    logged = [a["query"] for a in final.get("plan_attempts_log") or []]
    assert "Who is called that?" in logged
    assert "Who else is called that?" in logged
    # And the replanner was handed them.
    planner_calls = [c for c in executor.calls if c["role"] == "planner"]
    assert planner_calls[1]["input"].attempts, "replan must receive already_attempted"


def test_a_reworded_intent_does_not_lose_its_resolved_binding():
    """Carry-forward matched on exact intent text, so a planner that reworded a
    *closed* intent dropped its binding — the render lost the value and
    ``intermediate_answer`` fell back to None, discarding a referent retrieval had
    already established. Unmatched closures are now carried as extra entries."""
    mem = ["[Retrieval]: Euclid founded geometry."]
    old = _ledger("Who founded geometry?", "Where did they live?")
    old = cot_mod.apply_bindings(
        old, [(0, "Euclid")], {"euclid": "Q8747"}, hop=0, retrieval_memory=mem
    )
    assert old[0]["status"] == cot_mod.INTENT_CLOSED

    # Simulate the carry-forward against a fully reworded plan.
    new = cot_mod.build_plan_ledger(["Identify geometry's originator.", "Find their city."])
    closed = {
        (e.get("intent") or "").strip().lower(): e
        for e in old
        if e["status"] == cot_mod.INTENT_CLOSED
    }
    matched = set()
    for fresh in new:
        key = (fresh.get("intent") or "").strip().lower()
        if key in closed:
            matched.add(key)
    for key, prior in closed.items():
        if key not in matched:
            new.append(dict(prior))

    assert any(e["status"] == cot_mod.INTENT_CLOSED for e in new), (
        "the resolved binding must survive a full rewording"
    )
    rendered = cot_mod.render_plan_for_prompt("Identify, then locate.", new)
    assert "[resolved: Euclid]" in rendered
    assert cot_mod.latest_intermediate_answer(new) is not None


def test_a_retracted_premise_stops_the_render_asserting_its_value():
    """``apply_retractions`` flags the intent but leaves ``status`` and ``bindings``
    alone (the classifier still needs the binding history). If the render trusted
    ``status`` first, a closed intent whose premise retrieval had just contradicted
    would keep restating its value — the plan asserting a fact the evidence
    overturned, with nothing to un-assert it."""
    mem = ["[Retrieval]: Euclid founded geometry."]
    premise = "Euclid founded geometry."
    ledger = _ledger("Who founded geometry?", premises=[premise])
    ledger = cot_mod.apply_bindings(
        ledger, [(0, "Euclid")], {"euclid": "Q8747"}, hop=0, retrieval_memory=mem
    )
    assert "[resolved: Euclid]" in cot_mod.render_plan_for_prompt("P", ledger)

    ledger = cot_mod.apply_retractions(
        ledger, [{"content": premise, "reason": "contradicted"}]
    )
    rendered = cot_mod.render_plan_for_prompt("P", ledger)
    assert "Euclid" not in rendered
    assert "premise contradicted" in rendered
    # The binding history is still there for the classifier.
    assert ledger[0]["bindings"]
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_REPLAN


def test_a_numeral_label_cannot_manufacture_a_referent():
    """The defect that fabricated contests out of arithmetic.

    ``entity_dict`` routinely links bare years as entities, so ``{"1": "Q199"}`` is a real
    dictionary state. Resolution matched labels with a plain ``str.find``, so the digits
    inside any number became referents: ``"150 km/h (93 mph)"`` resolved to TWO QIDs. Since
    the contested test is "two or more distinct referents on one intent", one speed reading
    was enough to block that intent from ever closing.
    """
    l2q = {"1": "Q199", "3": "Q201", "150 km/h": "Q999"}
    assert cot_mod.resolve_binding_qids("150 km/h (93 mph)", l2q) == ["Q999"]
    # Sub-3-character labels go the same way: "US" inside "USSR".
    assert cot_mod.resolve_binding_qids("The USSR collapsed", {"us": "Q30"}) == []
    # A real multi-word label still matches, including punctuation-adjacent.
    assert cot_mod.resolve_binding_qids("Alan Turing.", {"alan turing": "Q7251"}) == ["Q7251"]
    # And a label must not match inside a longer word.
    assert cot_mod.resolve_binding_qids("reborn in fire", {"born": "Q1"}) == []


@pytest.mark.parametrize(
    "surfaces,expect,note",
    [
        # Observed in results/e3_plan_logonly — one referent described twice.
        ((("The Danyang-Kunshan Grand Bridge was opened in 2011.", "Q1"), ("2011", "Q2")),
         1, "a sentence and the bare value it states"),
        ((("2012", "Q1"), ("29 February 2012", "Q2")), 1, "one date at two granularities"),
        ((("Voyager 2", "Q1"), ("NASA's Voyager 2 probe", "Q2")), 1, "one probe"),
        # Genuine contests, which must survive.
        ((("2009", "Q1"), ("1989", "Q2")), 2, "two different years"),
        # Enumerations must NEVER absorb a member: the question may be asking to rank them.
        ((("Walther Bothe", "Q1"), ("Max Born and Walther Bothe", "Q2")), 2, "X and Y"),
        ((("In Utero", "Q1"), ("Nevermind, In Utero, Bleach.", "Q2")), 2, "comma list"),
        ((("Tokyo Skytree", "Q1"), ("1. Burj Khalifa, 2. Tokyo Skytree", "Q2")), 2, "ranked list"),
        ((("Yangtze", "Q1"), ("Congo, Yangtze, Danube and Hudson", "Q2")), 2, "ranked set"),
        ((("Gyrfalcon", "Q1"), ("Peregrine Falcon (320 km/h); Golden Eagle", "Q2")), 2, "semicolons"),
    ],
)
def test_two_descriptions_of_one_answer_are_not_a_contest(surfaces, expect, note):
    """Referent identity before the contested test.

    Measured on the shipped config: **17 of 38 contested fires** had a numeral or
    sub-3-character rival, and the samples were two descriptions of a single answer. Those
    manufactured contests are the expensive error — with ``replan_max: 0`` a contest is
    absorbing, so the intent never closes and keeps drawing retrieval for the whole run.

    The merge is join-never-split: it can only collapse rivals, never invent one, so the
    worst case is a missed contest. The enumeration guard is what keeps that safe — a list
    containing a candidate is evidence about a *set*, and merging it would silently pick
    one member of the very set the question asks to rank.
    """
    bindings = [{"surface": s, "qid": q} for s, q in surfaces]
    assert len(cot_mod.count_rival_referents(bindings)) == expect, note


def test_a_contest_gets_an_exit_and_it_is_not_dead():
    """``contested`` was an absorbing state in the shipped config.

    ``apply_bindings`` closes an intent only at exactly one distinct referent and the rival
    set only grows, so the sole exit was a replan — and the shipped config is
    ``replan_max: 0``. Measured: ``results/e3_plan_logonly`` ends with **13 intents still
    contested** against 1 in the armed runs. Those 13 kept drawing retrieval to no purpose.

    The retirement must NOT be ``INTENT_DEAD``: ``abstention_signal`` filters DEAD out
    entirely, so that would delete the hedge on precisely the questions the system is least
    certain about. ``undecided`` stays live and unmet, so the answer still hedges.
    """
    entry = {
        "intent": "Which is the third fastest bird?",
        "status": cot_mod.INTENT_CONTESTED,
        "bindings": [
            {"surface": "Gyrfalcon", "qid": "Q1", "grounded": True},
            {"surface": "Golden Eagle", "qid": "Q2", "grounded": True},
        ],
        "attempts": [{"query": "a", "n_facts": 3, "hop": 0}, {"query": "b", "n_facts": 3, "hop": 0}],
    }
    # Same hop -> still contested: no new evidence could have separated the rivals yet,
    # and retiring here would report 'stalled' and lose the competing surfaces.
    same_hop = cot_mod.mark_stalled_intents([entry], max_attempts=2)
    assert same_hop[0]["status"] == cot_mod.INTENT_CONTESTED

    # A further hop of evidence, still contested -> retire.
    entry2 = dict(entry, attempts=entry["attempts"] + [{"query": "c", "n_facts": 3, "hop": 1}])
    retired = cot_mod.mark_stalled_intents([entry2], max_attempts=2)
    assert retired[0]["status"] == cot_mod.INTENT_UNDECIDED
    assert retired[0]["undecided_rivals"] == ["Gyrfalcon", "Golden Eagle"]

    # An intent that stalled while merely OPEN and became contested later must still
    # retire. The retirement was originally behind the `stalled` early-return, and that
    # ordering silently disabled it: measured on results/e4_plan_fixed, 8 of the 13
    # intents left contested had attempts across 3-4 distinct hops and never retired
    # because `stalled` was already set. Offline replay of the corrected order retires
    # exactly those 8.
    stalled_first = dict(entry2, stalled=True, stall_reason="set while still open")
    assert cot_mod.mark_stalled_intents([stalled_first], max_attempts=2)[0]["status"] == (
        cot_mod.INTENT_UNDECIDED
    ), "the stalled flag must not short-circuit the contested retirement"

    # Still counted as unmet, so the answer hedges; and named for what it is.
    sig = cot_mod.abstention_signal(retired)
    assert "referent_ambiguous" in sig["reasons"]
    assert sig["unmet"] == ["Which is the third fastest bird?"]
    assert sig["level"] != "none"
    # A DEAD intent, by contrast, is filtered out of the hedge entirely — which is why
    # retirement must not use it.
    dead = cot_mod.abstention_signal([dict(entry2, status=cot_mod.INTENT_DEAD)])
    assert dead["unmet"] == [] and dead["level"] == "none"
    assert "UNRESOLVABLE" in cot_mod.render_plan_for_prompt("P", retired)


def test_an_unverified_binding_is_broadcast_marked_not_silently():
    """The asymmetry between what the plan showed and what it broadcast.

    ``render_plan_for_prompt`` shows an ungrounded closure as ``[resolved, unverified]`` and
    withholds its value, but ``latest_intermediate_answer`` had no ``grounded`` check at
    all — so the same binding the plan refused to state was handed to the next hop as its
    anchor: 15/39 hops in ``e1_cot_armed``, 18/44 in ``e2``, 4/43 in ``e3_plan_logonly``.

    Withholding it would be worse: this slot is the chaining anchor that the "do NOT re-ask
    what was already resolved" prompt rule depends on. So it goes out labelled.
    """
    def entry(grounded):
        return {
            "intent": "Identify the album",
            "status": cot_mod.INTENT_CLOSED,
            "closed_at": 1,
            "bindings": [{"surface": "In Utero", "qid": "Q1", "grounded": grounded}],
        }
    grounded = cot_mod.latest_intermediate_answer([entry(True)])
    assert "In Utero" in grounded and "unverified" not in grounded
    inferred = cot_mod.latest_intermediate_answer([entry(False)])
    assert "In Utero" in inferred, "the anchor must survive, or the next hop has no referent"
    assert "(unverified)" in inferred, "but it must not pass as established"
    # Consistent with what the plan itself shows for the same binding.
    assert "[resolved, unverified]" in cot_mod.render_plan_for_prompt("P", [entry(False)])


def test_a_contested_intent_shows_both_candidates_symmetrically_never_one():
    """A contest must be shown as a contest — both rivals, or neither.

    The principle is that the plan must not present an unsettled choice as settled, i.e.
    must not pick a side the reasoning has not. Showing *one* referent violates that.
    Showing *both*, labelled as rivals, is the opposite: it is what makes a
    discriminating question writable.

    This started as "show neither", which turned out to be the over-strict reading and to
    cost accuracy: told only that an intent was "ambiguous", the generator had nothing to
    discriminate between and simply re-issued the same query — **13 of 17 measured
    cross-hop repeated subquestions were contested intents**. Naming both rivals is safe
    here because the plan is prompt-only and never enters ``text_memory``, so a surface
    shown here cannot become a ``[Retrieval]`` fact.
    """
    mem = ["[Retrieval]: Alan Turing.", "[Retrieval]: Charles Babbage."]
    ledger = cot_mod.apply_bindings(
        _ledger("Who is called that?"),
        [(0, "Alan Turing"), (0, "Charles Babbage")],
        {"alan turing": "Q7251", "charles babbage": "Q46633"},
        hop=0,
        retrieval_memory=mem,
    )
    rendered = cot_mod.render_plan_for_prompt("P", ledger)
    assert "ambiguous" in rendered
    # Both, or the render has picked a side.
    assert ("Alan Turing" in rendered) == ("Babbage" in rendered)
    assert "Alan Turing" in rendered and "Babbage" in rendered
    # And neither is presented as the resolved value.
    assert "[resolved:" not in rendered
    assert "tells them apart" in rendered, "the generator needs the instruction, not just the pair"


# ──────────────────────────────────────────────────────────────────────────────
# A — self-contradicting evidence is its own failure
# ──────────────────────────────────────────────────────────────────────────────


def test_conflicting_retrieval_sources_trigger_a_replan():
    """Consolidator rule 6 keeps two contradicting `[Retrieval]` items rather than
    adjudicating. Nothing was overturned, so re-asking returns the same two sources
    — the repair has to *discriminate between the sources*, which only a replan adds.
    The field was produced and dropped on the floor before this.
    """
    ledger = _ledger("Which city is the most populous in Punjab?")
    ledger = cot_mod.mark_conflicted_intents(
        ledger, [["Ludhiana is the most populous", "Lahore is the most populous"]]
    )
    assert ledger[0]["conflicted"]
    assert cot_mod.classify_discharge(ledger)[0] == cot_mod.PLAN_ACTION_REPLAN
    assert cot_mod._discharge_reason(ledger, 0, cot_mod.PLAN_ACTION_REPLAN) == "conflicted"
    failure = cot_mod._describe_failure(ledger[0], [])
    assert "DISCRIMINATES BETWEEN THE SOURCES" in failure
    assert "disagrees with itself" in failure


def test_a_conflict_attaches_to_the_intent_that_cites_it():
    ledger = cot_mod.build_plan_ledger(["A", "B"], ["Ludhiana is the most populous"])
    ledger = cot_mod.apply_bindings(
        ledger, [(1, "Lahore")], {"lahore": "Q8751"}, hop=0,
        retrieval_memory=["[Retrieval]: Lahore is the most populous"],
    )
    ledger = cot_mod.mark_conflicted_intents(
        ledger, [["lahore is the most populous", "ludhiana is the most populous"]]
    )
    # Intent 0 cites one side as a premise, so it is the one that must adjudicate.
    assert ledger[0].get("conflicted")


def test_a_single_sided_conflict_group_is_ignored():
    """A group of one is not a conflict — guards against a malformed report."""
    ledger = cot_mod.mark_conflicted_intents(_ledger("A"), [["only one side"]])
    assert "conflicted" not in ledger[0]


# ──────────────────────────────────────────────────────────────────────────────
# C — an exhausted plan is insufficient, not wrong
# ──────────────────────────────────────────────────────────────────────────────


def test_a_settled_plan_that_still_cannot_answer_triggers_a_replan():
    """Every intent closed but the question unanswered used to classify as ``update``
    and simply synthesize — the plan was never revised for being *insufficient*."""
    mem = ["[Retrieval]: Euclid", "[Retrieval]: Alexandria"]
    ledger = cot_mod.apply_bindings(
        _ledger("A", "B"),
        [(0, "Euclid"), (1, "Alexandria")],
        {"euclid": "Q8747", "alexandria": "Q87"},
        hop=0,
        retrieval_memory=mem,
    )
    assert all(e["status"] == cot_mod.INTENT_CLOSED for e in ledger)
    # Answerable → nothing to do.
    assert cot_mod.classify_discharge(ledger, exhausted=False)[0] == cot_mod.PLAN_ACTION_UPDATE
    # Not answerable → the plan was insufficient.
    action, idx, _ = cot_mod.classify_discharge(ledger, exhausted=True)
    assert action == cot_mod.PLAN_ACTION_REPLAN
    assert cot_mod._discharge_reason(ledger, idx, action) == "exhausted"
    failure = cot_mod._describe_failure(ledger[idx], [])
    assert "INSUFFICIENT" in failure and "ADD" in failure


def test_exhaustion_has_the_lowest_precedence():
    """Any specific failure is a better description of what to fix."""
    ledger = cot_mod.apply_bindings(
        _ledger("A", "B"), [(0, "Euclid")], {"euclid": "Q8747"}, hop=0,
        retrieval_memory=["[Retrieval]: Euclid"],
    )
    ledger[1] = {**ledger[1], "status": cot_mod.INTENT_CONTESTED,
                 "bindings": [{"surface": "X", "qid": "Q1"}, {"surface": "Y", "qid": "Q2"}]}
    action, idx, _ = cot_mod.classify_discharge(ledger, exhausted=True)
    assert (action, idx) == (cot_mod.PLAN_ACTION_REPLAN, 1)  # contested, not exhausted


# ──────────────────────────────────────────────────────────────────────────────
# D — abandoned intents stay visible as ruled out
# ──────────────────────────────────────────────────────────────────────────────


def test_an_abandoned_intent_is_marked_dead_not_deleted():
    """A deleted intent is invisible to the generator, which is then free to
    re-propose the exact framing the replan discarded."""
    ledger = _ledger("A stalled framing.")
    ledger[0]["attempts"] = [{"query": "q", "n_facts": 0, "hop": 0}] * 2
    ledger = cot_mod.mark_stalled_intents(ledger, max_attempts=2)
    dead = {**ledger[0], "status": cot_mod.INTENT_DEAD,
            "dead_reason": cot_mod._dead_reason(ledger[0])}
    rendered = cot_mod.render_plan_for_prompt("P", [dead])
    assert "RULED OUT" in rendered
    assert "do not re-propose" in rendered
    assert "A stalled framing." in rendered


@pytest.mark.parametrize(
    "entry,expect",
    [
        ({"status": cot_mod.INTENT_CONTESTED}, "two different referents"),
        ({"conflicted": ["a", "b"]}, "contradicted each other"),
        ({"falsified": "P"}, "premise it rested on"),
        ({"stalled": True, "stall_reason": "no evidence returned"}, "no evidence"),
        ({}, "superseded"),
    ],
)
def test_dead_reason_explains_why_it_was_dropped(entry, expect):
    assert expect in cot_mod._dead_reason(entry)


def test_a_dead_intent_is_excluded_from_the_abstention_count():
    """An intent the plan deliberately abandoned is not an unmet information need."""
    ledger = _ledger("A", "B")
    ledger[1] = {**ledger[1], "status": cot_mod.INTENT_DEAD}
    signal = cot_mod.abstention_signal(ledger)
    assert signal["total"] == 1
    assert "B" not in signal["unmet"]


# ──────────────────────────────────────────────────────────────────────────────
# E — abstention is confidence, never adjudication
# ──────────────────────────────────────────────────────────────────────────────


def test_abstention_levels_track_what_was_established():
    resolved = cot_mod.apply_bindings(
        _ledger("A"), [(0, "Euclid")], {"euclid": "Q8747"}, hop=0,
        retrieval_memory=["[Retrieval]: Euclid"],
    )
    assert cot_mod.abstention_signal(resolved)["level"] == "none"

    nothing = _ledger("A", "B")
    assert cot_mod.abstention_signal(nothing)["level"] == "high"

    contradicted = cot_mod.apply_retractions(
        cot_mod.apply_bindings(
            cot_mod.build_plan_ledger(["A", "B"], ["P."]),
            [(0, "Euclid")], {"euclid": "Q8747"}, hop=0,
            retrieval_memory=["[Retrieval]: Euclid"],
        ),
        [{"content": "P.", "reason": "contradicted"}],
    )
    # A contradicted premise is high abstention even though something resolved.
    assert cot_mod.abstention_signal(contradicted)["level"] == "high"


def test_abstention_never_reaches_the_synthesizer_prompt():
    """Agreed explicitly: at synthesis the only question is which candidate is true.
    An unmet intent beside candidate answers invites calling a correct answer
    deficient."""
    assert "plan" not in roles_mod.FinalAnswerSynthesisInput.model_fields
    assert "abstention" not in roles_mod.FinalAnswerSynthesisInput.model_fields
    assert "unmet" not in roles_mod.FinalAnswerSynthesisInput.model_fields


# ──────────────────────────────────────────────────────────────────────────────
# B — the verifier 3-view spread, and F/G
# ──────────────────────────────────────────────────────────────────────────────


def test_verifier_spread_is_emitted_by_name_and_direction():
    """Only the mean was used; the *direction* of disagreement is the signal.

    An answer rated high closed-book and low against retrieved memory means the
    evidence gathered does not support the line the search is on — a plan-level
    problem, distinct from a wrong answer.
    """
    import inspect

    from langgraph_coe.graphs import mcts as mcts_mod

    src = inspect.getsource(mcts_mod.build_mcts_graph)
    ev = src[src.index("async def evaluate") : src.index("async def backprop")]
    assert '"verifier_by_view"' in ev, "views must be keyed, not just averaged"
    assert '"memory_disagreement"' in ev
    gate = src[src.index("async def plan_gate") : src.index("async def synthesize")]
    assert "memory_disagreement_threshold" in gate, "the spread must have a consumer"
    assert "memory_disagreement" in gate


def test_memory_disagreement_threshold_defaults_high():
    """The verifier's own noise floor is unmeasured (experiment E5), so a low bar
    here would fire on sampling noise rather than on real disagreement."""
    from langgraph_coe.config import LangGraphCoeConfig

    assert LangGraphCoeConfig().search.plan.memory_disagreement_threshold >= 3.0


def test_plan_scoring_prefers_the_referent_clean_plan():
    """``select_plan`` ranks by referent discipline, so raising ``N_PLANS`` above 1
    stays a one-constant change. At the shipped ``N_PLANS=1`` it reduces to "take
    the one that parsed"; the ranking is what makes a larger sample meaningful."""
    q = "Who was the father of the father of computer science?"
    mem = ["[Retrieval]: Alan Turing is called the father of computer science."]
    clean = _plan_out("Identify who is called that, then find their father.",
                      ["Identify who is called that.", "Find their father."])
    unverified = _plan_out("Find the father of Charles Babbage.",
                           ["Find the father of Charles Babbage."])
    placeholder = _plan_out("Find [the person] and their father.",
                            ["Find [the person].", "Find their father."])

    chosen, runners = cot_mod.select_plan([unverified, placeholder, clean], q, mem)
    assert chosen is clean, "the referent-clean, fully-instantiated plan must win"
    assert len(runners) == 2, "the rest are returned as runners-up, not discarded"
    # An unverified name is penalised; a [Retrieval]-verified one is not.
    verified = _plan_out("Find the father of Alan Turing.", ["Find the father of Alan Turing."])
    assert cot_mod.score_plan(verified, q, mem)[0] == 0
    assert cot_mod.score_plan(unverified, q, mem)[0] < 0


def test_plan_selection_survives_all_samples_failing_to_parse():
    from langgraph_coe import llm as llm_mod

    defaults = [llm_mod.build_safe_default_output(roles_mod.PLANNER) for _ in range(3)]
    chosen, runners = cot_mod.select_plan(defaults, "Q", [])
    assert chosen is None and runners == []


@pytest.mark.parametrize(
    "resolved,restated",
    [
        # Every pair here was observed in the armed 62-row run, where the resolved
        # intent was carried as closed AND re-listed as fresh-and-open, so retrieval
        # ran on it a second time.
        (
            "Determine the opening date of the longest bridge in the world.",
            "Determine the date on which the identified bridge was opened.",
        ),
        (
            "Identify the structure that is recognized as the tallest fixed steel structure.",
            "Confirm which structure is recognized as the tallest fixed steel structure.",
        ),
        (
            "Who is the director of the highest grossing film?",
            "Identify the director of the film determined to be the highest-grossing.",
        ),
        (
            "Find the release date of the identified second most selling studio album.",
            "Identify the release date of the album confirmed as the second most selling.",
        ),
    ],
)
def test_a_replan_does_not_reopen_a_resolved_intent_by_rewording_it(resolved, restated):
    """The defect that made replanning cost twice as much for nothing.

    The merge matched carried closures on **exact** intent text. The planner rewords,
    so the match failed and the settled intent was kept as closed *and* re-listed as
    fresh and open — then asked again. Measured on the armed run: 16 such pairs across
    10 of the 20 replanned questions, intents per question doubling per replan
    (2.17 -> 4.21 -> 8.17), closure rate down to 32%, and retrieval on identical
    questions rising 4.35 -> 8.95 subquestions. That is the plan breaking its own
    central promise — never re-ask what is resolved — inside its own replan path.
    """
    prior = {
        "intent": resolved,
        "status": cot_mod.INTENT_CLOSED,
        "bindings": [{"surface": "Danyang-Kunshan", "qid": "Q331642", "grounded": True}],
        "closed_at": 1,
        "attempts": [{"query": "q", "n_facts": 3, "hop": 1}],
    }
    closed_by_intent = {resolved.strip().lower(): prior}
    hit = cot_mod._fuzzy_closed_match(restated, closed_by_intent, set())
    assert hit is prior, "a reworded resolved intent must carry its binding, not re-open"

    # The prompt half of the fix: the settled intent is named, with its referent.
    lines = cot_mod._resolved_intent_lines([prior])
    assert lines == [f"{resolved} -> Danyang-Kunshan"]
    rendered = str(
        roles_mod.PlanInput(question="Q", current_plan="p", failure="stalled", resolved=lines)
    )
    assert "already_resolved_do_not_relist" in rendered


def test_fuzzy_carry_forward_requires_a_grounded_binding():
    """What keeps a 0.60 threshold safe.

    At that threshold a false positive is possible, so eligibility is restricted to
    closures whose referent retrieval already corroborated: wrongly carrying one costs
    nothing beyond a value we trust, while wrongly re-opening costs a whole hop. An
    intent closed on the model's own unverified inference still needs exact text.
    """
    ungrounded = {
        "intent": "Determine the opening date of the longest bridge in the world.",
        "status": cot_mod.INTENT_CLOSED,
        "bindings": [{"surface": "guessed", "qid": None, "grounded": False}],
    }
    idx = {ungrounded["intent"].strip().lower(): ungrounded}
    assert cot_mod._fuzzy_closed_match(
        "Determine the date on which the identified bridge was opened.", idx, set()
    ) is None
    # And an already-claimed closure is not matched twice.
    grounded = dict(ungrounded, bindings=[{"surface": "X", "qid": "Q1", "grounded": True}])
    idx2 = {grounded["intent"].strip().lower(): grounded}
    assert cot_mod._fuzzy_closed_match("Determine the date the bridge opened.", idx2, set())
    assert cot_mod._fuzzy_closed_match(
        "Determine the date the bridge opened.", idx2, {grounded["intent"].strip().lower()}
    ) is None


def test_replan_is_conditioned_on_the_failure_not_on_iid_samples():
    """A replan revises; it does not redraw.

    Its whole informational advantage is that it knows the current plan, the memory
    accumulated since, and *why* the plan failed. An independently sampled plan
    predates all three — it was drawn at hop 0 against empty memory — so seeding a
    revision from one throws away the only thing the revision knows, and can
    discard bindings that closed intents already earned. So the inputs must be the
    conditioning signals and nothing else.
    """
    import inspect

    src = inspect.getsource(cot_mod.build_cot_graph)
    rp = src[src.index("async def replan") : src.index("async def increment")]
    for conditioning in ("current_plan=", "context=", "failure=", "attempts="):
        assert conditioning in rp, f"replan must be conditioned on {conditioning}"
    assert "alternative_plans" not in rp
    assert "plan_alternatives" not in rp


def test_planner_is_sampled_once():
    """N_PLANS>1 bought diversity whose only consumer was ``replan``'s alternatives.
    With that removed, extra completions are planner tokens nothing reads."""
    assert cot_mod.N_PLANS == 1


def test_planner_has_its_own_hot_tier():
    """Plans are cheap to explore and impossible to verify, so breadth — not
    precision — is what is worth buying here. Reasoning roles stay cold."""
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    assert cfg.llm.role_tiers.get("planner") == "plan"
    plan_tier = cfg.llm.tiers["plan"]
    assert plan_tier.temperature >= cfg.llm.tiers["classify"].temperature
    assert plan_tier.top_p >= cfg.llm.tiers["heavy"].top_p


def test_exhaustion_does_not_fire_on_the_hop_that_made_progress():
    """``is_answerable`` is judged at the TOP of a hop, before that hop's retrieval.

    Reading it at the gate alone fired ``exhausted`` on rows where the plan had just
    closed every intent with fresh evidence — the next ``gen_subq`` had not yet seen
    the new memory. Observed live on two of five smoke rows. Exhaustion must also
    require that nothing closed this hop.
    """
    mem = ["[Retrieval]: Euclid"]
    ledger = cot_mod.apply_bindings(
        _ledger("A"), [(0, "Euclid")], {"euclid": "Q8747"}, hop=3, retrieval_memory=mem
    )
    assert ledger[0]["closed_at"] == 3
    progress = any(
        e.get("status") == cot_mod.INTENT_CLOSED and e.get("closed_at") == 3
        for e in ledger
    )
    assert progress, "the fixture must represent a hop that made progress"
    # Progress this hop -> not exhausted, whatever the stale answerability flag says.
    assert cot_mod.classify_discharge(ledger, exhausted=False)[0] == cot_mod.PLAN_ACTION_UPDATE
    # A later hop with nothing new and still unanswerable -> genuinely exhausted.
    assert cot_mod.classify_discharge(ledger, exhausted=True)[0] == cot_mod.PLAN_ACTION_REPLAN


def test_the_plan_prompt_buys_breadth_across_intents_not_within_one():
    """Both halves of a measured regression, and they pull in opposite directions.

    First the prompt said "work on the earliest plan intent", and 49% of hops
    produced one subquestion — serialising work that retrieves in parallel. Fixing
    that with "give a hard intent more than one phrasing" over-corrected: distinct
    (intent, hop) work slots per question stayed flat (2.84 → 3.09) while attempts
    per slot went 1.47 → 2.38, tail out to **10 retrievals on one intent in one
    hop**. Hops fell 2.08 → 1.61, so the latency win was real, but retrievals per
    question rose 4.18 → 7.35 to buy it.

    Breadth has to mean *more intents per hop*, never more rewordings of one.
    """
    p = roles_mod.GENERATE_SUBQUESTION_PROMPT
    assert "earliest plan intent" not in p, "the serializing instruction must be gone"
    assert "EVERY open intent" in p, "must ask for all askable intents in one round"
    assert "more than one phrasing" not in p, (
        "the over-correction: three completions each emitting 2-3 phrasings of one "
        "intent union to as many as 10 retrievals for it"
    )
    assert "One subquestion per intent per round" in p
    assert "different retrieval target" in p, (
        "the ordinal exception has to be stated as a different *target*, or it reads "
        "as licence to reword"
    )
    assert "COMPLETE RANKED LIST" in p, (
        "ordinal intents must ask for the ranking, not just the item — the off-by-one "
        "failures (Congo as 3rd river, Obama Sr. as the grandfather) came from "
        "retrieving the first-ranked item instead"
    )
    # The placeholder rule survives, but must not read as licence to stop early.
    assert "placeholder is not a question" in p
    assert "not a reason to stop early" in p


# ──────────────────────────────────────────────────────────────────────────────
# Dependency ordering (``depends_on`` / ``is_executable``)
# ──────────────────────────────────────────────────────────────────────────────


def test_a_dependent_intent_is_not_executable_until_its_prerequisite_closes():
    """The failure this exists to stop, taken from a real 4-hop MuSiQue question.

    The ledger held intent 0 (*Elizabeth Berg's birthplace*) open while intent 2
    (*the river by the city bordering it*) was asked anyway. It retrieved 9 facts of
    topically-nearby noise and **closed on them**. Both rendered as ``[open]``, so
    nothing told the generator the second was unanswerable. At two hops that wastes a
    hop; at four it destroys the chain.
    """
    ledger = cot_mod.build_plan_ledger(
        [
            "Identify Elizabeth Berg's birthplace.",
            "Determine the city bordering it.",
            "Identify the river by that city.",
        ],
        None,
        [-1, 0, 1],
    )
    assert [e["depends_on"] for e in ledger] == [None, 0, 1]
    assert cot_mod.is_executable(ledger, 0) is True
    assert cot_mod.is_executable(ledger, 1) is False
    assert cot_mod.is_executable(ledger, 2) is False

    # Closing the head unblocks exactly one step, not the whole chain.
    ledger[0]["status"] = cot_mod.INTENT_CLOSED
    assert cot_mod.is_executable(ledger, 1) is True
    assert cot_mod.is_executable(ledger, 2) is False, (
        "intent 2 must stay blocked: walking only the immediate parent would unblock "
        "it while its own prerequisite is still open"
    )
    ledger[1]["status"] = cot_mod.INTENT_CLOSED
    assert cot_mod.is_executable(ledger, 2) is True


def test_blocked_intents_render_as_blocked_not_open():
    ledger = cot_mod.build_plan_ledger(
        ["Identify the city.", "Find the river by that city."], None, [-1, 0]
    )
    rendered = cot_mod.render_plan_for_prompt("First the city, then its river.", ledger)
    assert "[open] Identify the city." in rendered
    assert "[BLOCKED on #1 - do not ask yet] Find the river by that city." in rendered
    # 1-based in the render because the prose numbers steps from one for a reader.
    assert "[open] Find the river" not in rendered


def test_dependency_sanitizer_rejects_what_would_deadlock():
    """Self- and forward-references are dropped rather than trusted.

    Either would block an intent permanently: a self-reference can never close
    first, and a forward edge is a mistake or a cycle. Since the prose orders the
    intents, a real dependency always points backwards.
    """
    ledger = cot_mod.build_plan_ledger(
        ["A", "B", "C", "D"],
        None,
        [0, 5, "1", 2],  # self-ref, out of range, wrong type, valid
    )
    assert [e["depends_on"] for e in ledger] == [None, None, None, 2]
    assert all(cot_mod.is_executable(ledger, i) for i in (0, 1, 2))
    assert cot_mod.is_executable(ledger, 3) is False


def test_ledger_without_dependencies_behaves_exactly_as_before():
    """A0 regression lock: a PLANNER that omits ``depends_on`` must lose nothing."""
    ledger = cot_mod.build_plan_ledger(["A", "B"], ["[Retrieval]: x"])
    assert [e["depends_on"] for e in ledger] == [None, None]
    assert all(cot_mod.is_executable(ledger, i) for i in range(2))
    assert "[open] A" in cot_mod.render_plan_for_prompt("p", ledger)


def test_dependency_indices_follow_kept_intents_when_a_blank_is_dropped():
    """``build_plan_ledger`` skips blank intents, so indices must be remapped.

    The model's indices refer to its own emitted list; the ledger's refer to what
    survived. Reading the raw index against the compacted ledger would point a
    dependency at the wrong intent.
    """
    ledger = cot_mod.build_plan_ledger(["A", "   ", "C"], None, [-1, -1, 0])
    assert [e["intent"] for e in ledger] == ["A", "C"]
    assert ledger[1]["depends_on"] == 0


def test_a_contested_prerequisite_does_not_block_forever():
    """Blocking is for *unresolved*, not for *unresolvable*.

    A contested intent will not improve by waiting — two referents survived and no
    further hop settles it — so holding its dependents hostage would strand the rest
    of the plan. Ask with the ambiguity rather than never ask.
    """
    ledger = cot_mod.build_plan_ledger(["A", "B"], None, [-1, 0])
    ledger[0]["status"] = cot_mod.INTENT_CONTESTED
    assert cot_mod.is_executable(ledger, 1) is True


# ──────────────────────────────────────────────────────────────────────────────
# Grounded-phrase closure (the third binding tier)
# ──────────────────────────────────────────────────────────────────────────────


def test_a_grounded_phrase_answer_closes_an_intent():
    """The 70% leak, fixed.

    Of 69 intents left open across the 120-row MuSiQue depth run, 48 had retrieved
    facts but recorded **no binding at all**: the answer named no linked entity and
    was no date or quantity. "Treaty of Paris" is an answer; it was simply
    unrepresentable, so the intent stayed open and the hop was spent for nothing.
    """
    mem = ["[Retrieval][hop=0]: The Treaty of Paris ceded the territory to the US."]
    ledger = cot_mod.apply_bindings(
        _ledger("Identify the treaty that ceded the territory."),
        [(0, "Treaty of Paris")],
        {},  # entity linking produced nothing
        hop=0,
        retrieval_memory=mem,
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED
    assert ledger[0]["closed_at"] == 0
    b = ledger[0]["bindings"][0]
    assert b["qid"] == "phr:treaty of paris"
    assert b["grounded"] is True, "a phrase tier binding is grounded by construction"


def test_an_ungrounded_phrase_does_not_close_anything():
    """The gate that keeps this from becoming a confidently-wrong machine.

    41% of QID closures are already ungrounded — closing on the model's own
    inference. The phrase tier carries no evidence of referenthood at all, so
    without corroboration it must not close an intent.
    """
    ledger = cot_mod.apply_bindings(
        _ledger("Identify the treaty."),
        [(0, "Treaty of Ghent")],
        {},
        hop=0,
        retrieval_memory=["[Retrieval][hop=0]: Unrelated fact about canals."],
    )
    assert ledger[0]["status"] == cot_mod.INTENT_OPEN
    assert ledger[0]["bindings"] == []


@pytest.mark.parametrize(
    "answer",
    ["unknown", "Not specified", "n/a", "  ", "no information", "It"],
)
def test_contentless_answers_never_close_an_intent(answer):
    """Recording "unknown" as a resolved referent would let the chain build on it."""
    ledger = cot_mod.apply_bindings(
        _ledger("Identify the treaty."),
        [(0, answer)],
        {},
        hop=0,
        retrieval_memory=[f"[Retrieval][hop=0]: {answer}"],
    )
    assert ledger[0]["status"] == cot_mod.INTENT_OPEN


def test_a_sentence_is_not_a_phrase_key():
    """A key that varies with phrasing manufactures rivals, and a manufactured rival
    blocks closure permanently at ``replan_max=0``. So the tier only accepts concise
    text."""
    long_answer = (
        "The treaty in question, after reviewing the historical record carefully, "
        "turns out to have been the Treaty of Paris signed in 1783."
    )
    assert cot_mod.resolve_primary_phrase(long_answer) is None
    assert cot_mod.resolve_primary_phrase("Treaty of Paris") == "treaty of paris"
    # Determiner- and punctuation-insensitive, so restatements share one key.
    assert cot_mod.resolve_primary_phrase("the Treaty of Paris") == "treaty of paris"
    assert (
        cot_mod.resolve_primary_phrase("Best Buy Co., Inc.")
        == cot_mod.resolve_primary_phrase("Best Buy Co Inc")
    )


def test_a_phrase_never_contests_a_real_referent():
    """One answer reaching the ledger through two tiers is one referent.

    Letting the weak tier compete would manufacture a contest, and with
    ``replan_max=0`` a contested intent never closes — trading the stall this
    change removes for a different permanent stall.
    """
    bindings = [
        {"surface": "Alan Turing", "qid": "Q7251", "hop": 0, "grounded": True},
        {"surface": "Alan Turing", "qid": "phr:alan turing", "hop": 1, "grounded": True},
    ]
    assert cot_mod.count_rival_referents(bindings) == {"Q7251"}

    # Two phrases with no strong referent present still compete — that is a real
    # disagreement about the answer, not an artefact of tiering.
    two_phrases = [
        {"surface": "Treaty of Paris", "qid": "phr:treaty of paris", "hop": 0, "grounded": True},
        {"surface": "Treaty of Ghent", "qid": "phr:treaty of ghent", "hop": 0, "grounded": True},
    ]
    assert len(cot_mod.count_rival_referents(two_phrases)) == 2


def test_phrase_closure_feeds_the_next_hop():
    """Why this matters for a 4-hop chain: closure is what anchors the next step.

    ``intermediate_answer`` is populated from closed bindings, so an intent that
    cannot close cannot hand its referent forward — which is how a deep chain
    silently loses its anchor and the later hops guess.
    """
    mem = ["[Retrieval][hop=0]: Elizabeth Berg was born in Saint Paul."]
    ledger = cot_mod.apply_bindings(
        _ledger("Identify Elizabeth Berg's birthplace.", "Find the river by that city."),
        [(0, "Saint Paul")],
        {},
        hop=0,
        retrieval_memory=mem,
    )
    assert ledger[0]["status"] == cot_mod.INTENT_CLOSED
    assert cot_mod.latest_intermediate_answer(ledger)


# ──────────────────────────────────────────────────────────────────────────────
# Plan-derived depth budget and synthesis findings
# ──────────────────────────────────────────────────────────────────────────────


def test_chain_depth_drives_the_hop_budget():
    """A linear chain of N needs N hops; ``max_depth=4`` gave a 4-chain zero slack.

    44% of the depth-run failures were "partially resolved, ran out of hops" — a
    4-hop chain admits one executable intent per hop, so a single hop that fails to
    close makes the question unanswerable regardless of plan quality.
    """
    chain = cot_mod.build_plan_ledger(["A", "B", "C", "D"], None, [-1, 0, 1, 2])
    assert cot_mod.plan_chain_depth(chain) == 4
    assert cot_mod.effective_max_depth({"max_depth": 4, "plan_ledger": chain}) == 5

    # A flat plan asks for nothing extra — this is not a blanket budget increase.
    flat = cot_mod.build_plan_ledger(["A", "B", "C", "D"], None, [-1, -1, -1, -1])
    assert cot_mod.plan_chain_depth(flat) == 1
    assert cot_mod.effective_max_depth({"max_depth": 4, "plan_ledger": flat}) == 4


def test_the_budget_never_shrinks_and_never_applies_without_a_plan():
    # No ledger (the no-plan arm) must be untouched, or the control is not a control.
    assert cot_mod.effective_max_depth({"max_depth": 4, "plan_ledger": []}) == 4
    # A short chain under a generous configured budget must not lower it.
    short = cot_mod.build_plan_ledger(["A", "B"], None, [-1, 0])
    assert cot_mod.effective_max_depth({"max_depth": 10, "plan_ledger": short}) == 10


def test_chain_depth_terminates_on_a_cycle():
    """A malformed ledger must not take the run down with unbounded recursion."""
    led = cot_mod.build_plan_ledger(["A", "B"], None, [-1, 0])
    led[0]["depends_on"] = 1  # cycle injected past the sanitizer
    assert cot_mod.plan_chain_depth(led) <= len(led) + 1


def test_resolved_findings_carries_only_grounded_surviving_bindings():
    """The 10 recoverable rows: the answer was in evidence and synthesis lost it.

    Only grounded bindings, so synthesis never sees an unverified inference; and
    never a falsified intent, whose premise retrieval has already contradicted.
    """
    led = cot_mod.build_plan_ledger(["Who founded it?", "When?", "Where?"])
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Ada Lovelace", "qid": "Q7259", "hop": 0, "grounded": True}],
    )
    led[1].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "1843", "qid": "lit:1843", "hop": 1, "grounded": False}],
    )
    led[2].update(
        status=cot_mod.INTENT_CLOSED,
        falsified=True,
        bindings=[{"surface": "London", "qid": "Q84", "hop": 1, "grounded": True}],
    )
    findings = cot_mod.resolved_findings(led)
    assert findings == ["Who founded it -> Ada Lovelace"]
    assert not any("1843" in f for f in findings), "ungrounded inference must not reach synthesis"
    assert not any("London" in f for f in findings), "a falsified premise must not be restated"


def test_synthesis_input_renders_findings_last():
    """The input guard drops the middle of an oversized payload, so the shortest and
    highest-value block has to be at the tail."""
    inp = roles_mod.FinalAnswerSynthesisInput(
        question="Q?",
        candidate_answers=["a", "b"],
        context="some evidence",
        resolved_findings=["Who founded it -> Ada Lovelace"],
    )
    s = str(inp)
    assert s.index("resolved_findings") > s.index("supporting_evidence")
    assert "- Who founded it -> Ada Lovelace" in s
    # Absent findings must not add an empty section.
    assert "resolved_findings" not in str(
        roles_mod.FinalAnswerSynthesisInput(question="Q?", candidate_answers=["a"])
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evidence slicing: query-aware fallback + plan-directed escalation
# ──────────────────────────────────────────────────────────────────────────────


def test_the_no_reranker_slice_is_query_aware():
    """The identity slice ignored the query, which broke per-query reranking.

    ``rerank_per_query`` exists so each subquestion keeps its own top-k. With an
    identity slice every subquestion received the *same* first-k passages, so the
    union across a fan-out was k passages in total rather than k per query — only
    the first 10 passages in arrival order ever reached the extractor, however wide
    the retrieval.
    """
    items = [
        "filler about canals and barges",
        "more unrelated filler text here",
        "The Treaty of Paris ceded the territory west to the Mississippi",
    ]
    got = cot_mod._lexical_top_k("Which treaty ceded the territory?", items, 1)
    assert got == [items[2]], "the relevant passage must survive a slice of 1"
    # No content terms in the query -> fall back to arrival order, unchanged behaviour.
    assert cot_mod._lexical_top_k("the of and", items, 2) == items[:2]
    # A slice wider than the candidate set is a no-op.
    assert cot_mod._lexical_top_k("treaty", items, 99) == items


@pytest.mark.asyncio
async def test_per_query_budgets_give_each_subquestion_its_own_slice():
    items = [f"passage about alpha {i}" for i in range(5)] + [
        f"passage about beta {i}" for i in range(5)
    ]
    merged = await cot_mod.rerank_per_query(
        ["alpha", "beta"], items, 1, None, per_query_top_k={"beta": 3}
    )
    assert sum(1 for m in merged if "alpha" in m) == 1
    assert sum(1 for m in merged if "beta" in m) == 3, "beta bought a wider slice"


def test_only_an_already_failing_intent_gets_a_wider_slice():
    """Escalation must not fire on speculation — the cheap slice gets first refusal.

    This is the one lever the no-plan arm structurally cannot use: identifying a
    starving question needs the per-intent attempt history, which only the ledger
    has.
    """
    ledger = cot_mod.build_plan_ledger(["fresh intent", "failing intent"])
    ledger[1]["attempts"] = [{"query": "q1", "n_facts": 0, "hop": 0},
                             {"query": "q2", "n_facts": 0, "hop": 1}]
    state = {
        "plan_ledger": ledger,
        "subquestion_serves_intent": [0, 1],
    }
    subqs = ["ask about the fresh one", "ask about the failing one"]
    budgets = cot_mod._starving_query_budgets(state, subqs, 10, stall_after=2)
    # Only the failing intent's subquestion is selected. The *size* of its budget is
    # ``_STARVING_TOP_K_MULTIPLIER``, currently 1 — escalation is off because at 3x it
    # cost +34% input tokens for zero accuracy (dep_plan_v3 vs dep_plan_v2, both
    # 32/120). Assert the targeting, which is the part that must stay correct.
    assert set(budgets) == {"ask about the failing one"}
    assert budgets["ask about the failing one"] == 10 * cot_mod._STARVING_TOP_K_MULTIPLIER

    # A closed intent is not starving, whatever its attempt count.
    ledger[1]["status"] = cot_mod.INTENT_CLOSED
    assert cot_mod._starving_query_budgets(state, subqs, 10, stall_after=2) == {}


def test_no_plan_means_no_escalation():
    """A0 regression lock: without a ledger the slicing is untouched."""
    assert cot_mod._starving_query_budgets({"plan_ledger": []}, ["q"], 10, 2) == {}
    assert cot_mod._starving_query_budgets({}, ["q"], 10, 2) == {}


# ──────────────────────────────────────────────────────────────────────────────
# The negative record: failed queries reach the generator
# ──────────────────────────────────────────────────────────────────────────────


def test_an_unresolved_intent_shows_what_already_failed():
    """The measured gap: 43 of 65 attempt pairs on open intents were near-duplicate
    re-issues (66%), while retrieval returned zero facts only 3 times in 1010
    attempts. The angle was the problem and nothing in the prompt carried it."""
    led = cot_mod.build_plan_ledger(["Identify the birthplace."])
    led[0]["attempts"] = [
        {"query": "Where was Elizabeth Berg born?", "n_facts": 9, "hop": 0},
        {"query": "What is Elizabeth Berg's birthplace?", "n_facts": 4, "hop": 1},
    ]
    rendered = cot_mod.render_plan_for_prompt("First the birthplace.", led)
    assert "[open] Identify the birthplace." in rendered
    assert "already asked, did not resolve it: Where was Elizabeth Berg born?" in rendered
    assert "already asked, did not resolve it: What is Elizabeth Berg's birthplace?" in rendered
    # Most recent first — the freshest failure is the most informative.
    assert rendered.index("birthplace?") < rendered.index("born?")


def test_the_negative_record_is_bounded_and_deduplicated():
    led = cot_mod.build_plan_ledger(["Find it."])
    led[0]["attempts"] = (
        [{"query": f"query number {i}", "n_facts": 1, "hop": i} for i in range(6)]
        + [{"query": "query number 5", "n_facts": 1, "hop": 6}]  # exact repeat
    )
    rendered = cot_mod.render_plan_for_prompt("p", led)
    shown = [l for l in rendered.splitlines() if "already asked" in l]
    assert len(shown) == cot_mod._MAX_RENDERED_ATTEMPTS
    assert len(set(shown)) == len(shown), "a repeated query must be listed once"


def test_resolved_and_blocked_intents_do_not_list_attempts():
    """A closed intent's history is noise; a blocked one must not be asked at all."""
    led = cot_mod.build_plan_ledger(["A", "B"], None, [-1, 0])
    for e in led:
        e["attempts"] = [{"query": "some query", "n_facts": 3, "hop": 0}]
    # B is blocked on A while A is open.
    rendered = cot_mod.render_plan_for_prompt("p", led)
    assert rendered.count("already asked") == 1, "only the open intent lists attempts"

    led[0].update(status=cot_mod.INTENT_CLOSED, closed_at=0,
                  bindings=[{"surface": "X", "qid": "Q1", "hop": 0, "grounded": True}])
    assert "already asked" not in cot_mod.render_plan_for_prompt("p", [led[0]])


def test_the_prompt_forbids_rewording_a_ruled_out_angle():
    p = roles_mod.GENERATE_SUBQUESTION_PROMPT
    assert "already asked, did not resolve it:" in p, "the rule must name the rendered marker"
    assert "do not reword them" in p
    # The worked example is what stops "ask differently" collapsing into a synonym.
    assert "same query, reworded" in p


# ──────────────────────────────────────────────────────────────────────────────
# Oversized-passage compression (the evidence ceiling)
# ──────────────────────────────────────────────────────────────────────────────


def test_a_buried_fact_survives_compression_where_truncation_loses_it():
    """The dominant failure mode, reproduced and fixed at unit level.

    The EXTRACTOR was called with a mean prompt of 89,277 tokens against a
    20,000-token ceiling, and the guard keeps head+tail and drops the middle. A fact
    in the body of a long crawled page was therefore discarded however relevant —
    which is the most likely cause of "every intent resolved, gold never retrieved"
    (41% of failures).
    """
    filler = "unrelated boilerplate about shipping schedules. " * 400
    needle = "The Treaty of Paris ceded the territory west to the Mississippi River."
    page = filler + needle + filler
    budget = 4000
    assert len(page) > budget * 4, "the fixture must be genuinely oversized"

    # Head+tail truncation, i.e. the old behaviour, loses it.
    head_tail = page[: budget // 2] + page[-(budget // 2):]
    assert needle not in head_tail

    kept = cot_mod._relevant_windows(page, "Which treaty ceded the territory?", budget)
    assert needle in kept, "query-relevant window selection must keep the answer"
    assert len(kept) <= budget + len(cot_mod._ELISION) * 4


def test_compression_preserves_document_order_and_marks_gaps():
    a = "Alpha section mentions the treaty explicitly. " * 20
    mid = "irrelevant middle. " * 300
    b = "Beta section also discusses the treaty terms. " * 20
    kept = cot_mod._relevant_windows(a + mid + b, "treaty", 3000)
    assert kept.index("Alpha") < kept.index("Beta"), "windows must return to document order"
    assert cot_mod._ELISION in kept, "a gap must be marked so the extractor knows"


def test_compression_is_a_no_op_for_normal_passages():
    """Only oversized items are touched, so ordinary retrieval is unchanged."""
    small = "A short passage about the treaty."
    assert cot_mod._relevant_windows(small, "treaty", 10_000) == small


def test_compression_falls_back_to_the_head_without_a_usable_query():
    page = "content " * 5000
    got = cot_mod._relevant_windows(page, "the of and", 1000)
    assert got == page.strip()[:1000], "no query terms -> bounded head, not an error"


@pytest.mark.asyncio
async def test_extract_facts_compresses_before_batching(monkeypatch):
    """A single oversized page must not become one enormous prompt."""
    seen: List[str] = []

    async def fake_exec(registry, role, input_data, *a, **k):
        seen.append(input_data.raw_data)
        return roles_mod.ExtractionOutput(relevant_information=["fact"]), {}

    monkeypatch.setattr(cot_mod, "execute_role_lc", fake_exec)
    huge = ("filler. " * 2000) + "The answer is Paris." + ("filler. " * 2000)
    await cot_mod.extract_facts(None, "Where is it?", ["Where is the answer?"], [huge], 4000)
    assert seen, "the extractor must still be called"
    assert all(len(blob) <= 4000 + 200 for blob in seen), (
        f"no batch may blow the budget; got {[len(b) for b in seen]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# The four fixes for the measured plan failures. Each test names the number that
# motivated it, because each of these was a *harm* the plan was causing.
# ──────────────────────────────────────────────────────────────────────────────


def test_scaffolding_referents_never_reach_synthesis():
    """The measured harm: the plan doubled the rate of answering with an intermediate.

    Paired over 3 seeds x 120 questions, answering with a non-terminal referent ran at
    10.6% with the plan against 5.3% without, discordant 43 (31/12), sign test
    p = 0.0054, and 87% of those answers were wrong. Cause: ``resolved_findings`` fed
    *every* closed intent to synthesis, and the synthesis prompt ranks it above all
    other context — so a hop-1 referent arrived as the highest-authority statement.
    """
    led = cot_mod.build_plan_ledger(
        ["Who was Hagar's spouse?", "Who did he marry after Sarah died?"],
        depends_on=[None, 0],
    )
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Abraham", "qid": "Q17997608", "hop": 0, "grounded": True}],
    )
    led[1].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Keturah", "qid": "Q243871", "hop": 2, "grounded": True}],
    )
    findings = cot_mod.resolved_findings(led)
    assert findings == ["Who did he marry after Sarah died -> Keturah"]
    assert not any("Abraham" in f for f in findings), (
        "intent 0 is scaffolding for intent 1; promoting its referent to synthesis is "
        "what produced the answer 'Abraham' when the gold was 'Keturah'"
    )


def test_a_flat_plan_keeps_every_finding():
    """No dependencies means no scaffolding, so the terminal filter must be inert."""
    led = cot_mod.build_plan_ledger(["Who directed it?", "Who scored it?"])
    for i, (s, q) in enumerate([("Kubrick", "Q2201"), ("Ligeti", "Q76326")]):
        led[i].update(
            status=cot_mod.INTENT_CLOSED,
            bindings=[{"surface": s, "qid": q, "hop": i, "grounded": True}],
        )
    assert len(cot_mod.resolved_findings(led)) == 2


def test_the_plan_can_end_a_question_when_its_target_resolves():
    """96 of 360 questions ran a mean 1.65 hops after the plan had nothing left."""
    led = cot_mod.build_plan_ledger(["Who is the spouse?", "Where born?"], depends_on=[None, 0])
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Abraham", "qid": "Q1", "hop": 0, "grounded": True}],
    )
    # Scaffolding closed but the target still open — must NOT stop.
    assert not cot_mod.plan_target_resolved(led)
    led[1].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Ur", "qid": "Q2", "hop": 1, "grounded": True}],
    )
    assert cot_mod.plan_target_resolved(led)


def test_an_ungrounded_or_empty_plan_never_stops_the_loop():
    """A stop condition that fires on no evidence would truncate every question."""
    assert not cot_mod.plan_target_resolved([])
    led = cot_mod.build_plan_ledger(["Where born?"])
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "somewhere", "qid": None, "hop": 0, "grounded": False}],
    )
    assert not cot_mod.plan_target_resolved(led), "an uncorroborated phrase must not end the question"
    led[0].update(falsified=True, bindings=[{"surface": "Ur", "qid": "Q2", "hop": 0, "grounded": True}])
    assert not cot_mod.plan_target_resolved(led), "a falsified target must not end the question"


def test_a_circumlocuting_query_gets_its_referent_back():
    """23% of 3,328 issued queries omitted a referent the plan had already bound.

    The retrieval query is the subquestion, so a description retrieves documents about
    the description while the needed document is indexed under the name.
    """
    led = cot_mod.build_plan_ledger(
        ["Who performs 'Hits'?", "When were they born?"], depends_on=[None, 0]
    )
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Dolly Parton", "qid": "Q483994", "hop": 0, "grounded": True}],
    )
    q = "What is the date of birth for the performer associated with 'Hits'?"
    assert cot_mod.ground_retrieval_query(q, led, 1) == q + " Dolly Parton"
    # Already named → unchanged, no duplicate term.
    named = "What is Dolly Parton's date of birth?"
    assert cot_mod.ground_retrieval_query(named, led, 1) == named
    # No plan, or an unattributed subquestion → untouched.
    assert cot_mod.ground_retrieval_query(q, [], 1) == q
    assert cot_mod.ground_retrieval_query(q, led, None) == q
    # The scaffolding intent itself has no prerequisite → untouched.
    assert cot_mod.ground_retrieval_query(q, led, 0) == q


def test_grounding_refuses_a_sentence_and_a_falsified_referent():
    """The ledger's ``surface`` may be a fragment; appending one swamps the query."""
    led = cot_mod.build_plan_ledger(["What changed?", "When?"], depends_on=[None, 0])
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{
            "surface": "The name 'Burma' did not change to 'Thailand'.",
            "qid": None, "hop": 0, "grounded": True,
        }],
    )
    q = "When was the name changed?"
    assert cot_mod.ground_retrieval_query(q, led, 1) == q, "a sentence is not a name"
    led[0].update(
        falsified=True,
        bindings=[{"surface": "Siam", "qid": "Q3", "hop": 0, "grounded": True}],
    )
    assert cot_mod.ground_retrieval_query(q, led, 1) == q, "a retracted referent must not be re-asserted"


def test_the_pruning_budget_tracks_the_number_of_open_intents():
    """``triple_pruner`` is 45.3% of all LLM calls at a fixed ceil(64/16)=4 per fetch."""
    from langgraph_coe.tools import wikidata as wd

    try:
        wd.set_plan_focus(["a"], 1)
        assert wd._planned_top_k(64) == 16, "one open intent needs one batch, not four"
        wd.set_plan_focus(["a", "b"], 2)
        assert wd._planned_top_k(64) == 32
        wd.set_plan_focus([f"i{i}" for i in range(9)], 9)
        assert wd._planned_top_k(64) == 64, "must never exceed the configured budget"
        assert wd._focused_query("q") == "q i0 i1 i2 i3 i4 i5 i6 i7 i8"
        wd.clear_plan_focus()
        assert wd._planned_top_k(64) == 64, "no plan must be byte-identical to today"
        assert wd._focused_query("q") == "q"
        wd.set_plan_focus([], 0)
        assert wd._planned_top_k(64) == 64, "an empty intent list is not a focus"
    finally:
        wd.clear_plan_focus()


def test_scaffolding_is_labelled_not_merely_hidden():
    """Omitting scaffolding did not work: the leak ratio was 1.8x before AND after.

    The referents survive in ``candidate_answers`` via ``text_memory``, which both arms
    share, so synthesis has to be told which ones are inputs.
    """
    led = cot_mod.build_plan_ledger(
        ["Who was Hagar's spouse?", "Who did he marry after Sarah died?"],
        depends_on=[None, 0],
    )
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Abraham", "qid": "Q1", "hop": 0, "grounded": True}],
    )
    led[1].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Keturah", "qid": "Q2", "hop": 2, "grounded": True}],
    )
    scaff = cot_mod.scaffolding_findings(led)
    assert len(scaff) == 1 and scaff[0].startswith("Abraham")
    assert "input to" in scaff[0]
    # The two sets must partition the closed intents — never overlap.
    resolved = cot_mod.resolved_findings(led)
    assert not any("Abraham" in r for r in resolved)
    assert not any("Keturah" in s for s in scaff)


def test_a_flat_plan_has_no_scaffolding():
    led = cot_mod.build_plan_ledger(["Who directed it?", "Who scored it?"])
    for i, s in enumerate(["Kubrick", "Ligeti"]):
        led[i].update(
            status=cot_mod.INTENT_CLOSED,
            bindings=[{"surface": s, "qid": f"Q{i}", "hop": i, "grounded": True}],
        )
    assert cot_mod.scaffolding_findings(led) == []
    assert len(cot_mod.resolved_findings(led)) == 2


def test_an_ungrounded_or_falsified_bridge_is_not_listed_as_excluded():
    """Excluding a referent the evidence never supported would forbid a valid answer."""
    led = cot_mod.build_plan_ledger(["Who is X?", "Where born?"], depends_on=[None, 0])
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Guess", "qid": None, "hop": 0, "grounded": False}],
    )
    assert cot_mod.scaffolding_findings(led) == []
    led[0].update(
        falsified=True,
        bindings=[{"surface": "Abraham", "qid": "Q1", "hop": 0, "grounded": True}],
    )
    assert cot_mod.scaffolding_findings(led) == []


def test_the_synthesis_payload_names_the_exclusion_last():
    """Rendered after resolved_findings: the last thing read is what not to answer."""
    from langgraph_coe.roles import FinalAnswerSynthesisInput

    s = str(
        FinalAnswerSynthesisInput(
            question="Who did the spouse of Hagar marry after Sarah died?",
            candidate_answers=["Abraham", "Keturah"],
            resolved_findings=["Who did he marry -> Keturah"],
            scaffolding_findings=["Abraham (resolved only as the input to: Who was Hagar's spouse)"],
        )
    )
    assert s.index("resolved_findings:") < s.index("intermediate_steps_NOT_the_answer:")
    assert "None of them can be the final answer." in s
    # Absent field renders nothing at all.
    bare = str(
        FinalAnswerSynthesisInput(question="q", candidate_answers=["a"])
    )
    assert "intermediate_steps_NOT_the_answer" not in bare


def test_the_synthesis_prompt_forbids_returning_a_bridge_entity():
    from langgraph_coe.roles import SYNTHESIZE_FINAL_ANSWER_PROMPT as P

    assert "intermediate_steps_NOT_the_answer" in P
    assert "hard exclusion" in P
    assert "Never output one as the final answer" in P
    # The measured reason must stay in the prompt: these are the *crispest* candidates.
    assert "crispest" in P


def test_gather_evidence_grounds_all_three_fanouts():
    """MCTS's retrieval twin silently bypassed every plan mechanism.

    ``gather_evidence`` is used only by ``mcts.py`` and sent the raw subquestion to
    corpus, KG *and* web. So under ``search.strategy=mcts`` the plan's measured
    retrieval effect (41.4% recall vs 37.5% without, 461 paired questions) could not
    reproduce — the CoT arm and the MCTS arm were not running the same retrieval.
    """
    import asyncio

    led = cot_mod.build_plan_ledger(
        ["Who performs 'Hits'?", "When were they born?"], depends_on=[None, 0]
    )
    led[0].update(
        status=cot_mod.INTENT_CLOSED,
        bindings=[{"surface": "Dolly Parton", "qid": "Q483994", "hop": 0, "grounded": True}],
    )
    subq = "What is the date of birth for the performer associated with 'Hits'?"

    corpus_seen, kg_seen, web_seen = [], [], []

    class _Corpus:
        async def ainvoke(self, payload):
            corpus_seen.append(payload["query"])
            return []

    class _Graph:
        def __init__(self, sink, key):
            self.sink, self.key = sink, key

        async def ainvoke(self, payload):
            self.sink.append(payload["subquery"])
            return {self.key: [], "triples": [], "results": []}

    async def _fake_rerank(subqs, pooled, top_k, cfg):
        return list(pooled)

    async def _fake_extract(registry, question, subqs, ctx, max_chars):
        return []

    orig = (cot_mod.corpus_search, cot_mod.rerank_per_query, cot_mod.extract_facts)
    cot_mod.corpus_search = _Corpus()
    cot_mod.rerank_per_query = _fake_rerank
    cot_mod.extract_facts = _fake_extract
    try:
        asyncio.run(
            cot_mod.gather_evidence(
                None,
                "Who performs 'Hits' and when were they born?",
                [subq],
                needs_kg=[True],
                kg_graph=_Graph(kg_seen, "kg_articles"),
                web_graph=_Graph(web_seen, "results"),
                web_enabled=True,
                corpus_enabled=True,
                plan_ledger=led,
                serves_intent=[1],
            )
        )
    finally:
        cot_mod.corpus_search, cot_mod.rerank_per_query, cot_mod.extract_facts = orig

    expected = subq + " Dolly Parton"
    assert corpus_seen == [expected], f"corpus fan-out not grounded: {corpus_seen}"
    assert kg_seen == [expected], f"KG fan-out not grounded: {kg_seen}"
    assert web_seen == [expected], f"web fan-out not grounded: {web_seen}"


def test_gather_evidence_is_unchanged_without_a_plan():
    """``_reverify_memory`` passes facts, not subquestions — it must not be rewritten."""
    import asyncio

    seen = []

    class _Corpus:
        async def ainvoke(self, payload):
            seen.append(payload["query"])
            return []

    async def _fake_rerank(subqs, pooled, top_k, cfg):
        return list(pooled)

    async def _fake_extract(registry, question, subqs, ctx, max_chars):
        return []

    fact = "Abraham married Keturah after the death of Sarah."
    orig = (cot_mod.corpus_search, cot_mod.rerank_per_query, cot_mod.extract_facts)
    cot_mod.corpus_search = _Corpus()
    cot_mod.rerank_per_query = _fake_rerank
    cot_mod.extract_facts = _fake_extract
    try:
        asyncio.run(
            cot_mod.gather_evidence(
                None, "q", [fact], needs_kg=[False], corpus_enabled=True
            )
        )
    finally:
        cot_mod.corpus_search, cot_mod.rerank_per_query, cot_mod.extract_facts = orig
    assert seen == [fact], "a fact with no attribution must be re-verified verbatim"


def test_gather_evidence_clears_the_pruning_focus():
    """A leaked focus would scope a later retrieval to a stale plan state."""
    import asyncio

    from langgraph_coe.tools import wikidata as wd

    led = cot_mod.build_plan_ledger(["Who performs 'Hits'?"])

    async def _fake_rerank(subqs, pooled, top_k, cfg):
        return list(pooled)

    async def _fake_extract(registry, question, subqs, ctx, max_chars):
        return []

    orig = (cot_mod.rerank_per_query, cot_mod.extract_facts)
    cot_mod.rerank_per_query = _fake_rerank
    cot_mod.extract_facts = _fake_extract
    try:
        asyncio.run(
            cot_mod.gather_evidence(
                None, "q", ["some subquestion here"], corpus_enabled=False,
                plan_ledger=led,
            )
        )
    finally:
        cot_mod.rerank_per_query, cot_mod.extract_facts = orig
        wd.clear_plan_focus()
    assert wd.read_plan_focus() is None, "focus must not outlive the fan-out"


# ──────────────────────────────────────────────────────────────────────────────
# MCTS plan scope. Tree-scope sharing was measured to collapse the search
# (distinct subquestions 9.9 -> 6.3, sibling overlap 10.2% -> 23.1%). Rollout
# scope keeps the plan's within-chain benefits without coupling siblings.
# ──────────────────────────────────────────────────────────────────────────────


def test_rollout_scope_is_the_default():
    from langgraph_coe.config import PlanConfig

    assert PlanConfig().mcts_plan_scope == "rollout", (
        "tree scope cut distinct subquestions per question by 36% and more than "
        "doubled sibling-subtree overlap at identical accuracy"
    )


def test_tree_scope_adds_the_root_plan_nodes_and_rollout_scope_does_not():
    """Under rollout scope the tree must hold no plan at all."""
    import langgraph_coe.graphs.mcts as mcts_mod

    src = inspect.getsource(mcts_mod.build_mcts_graph)
    assert 'tree_plan = plan_enabled and plan_scope == "tree"' in inspect.getsource(
        mcts_mod.build_mcts_graph
    ) or 'plan_scope == "tree"' in src
    # The root-level gen_plan/plan_gate must be gated on tree scope, not on
    # plan_enabled — otherwise rollout scope would still mint a shared tree plan.
    assert 'if tree_plan:\n        builder.add_node("gen_plan"' in src


def test_the_rollout_plans_for_itself_under_rollout_scope():
    """``gen_plan`` returns early when ``plan`` is non-empty, so it must be unset.

    Seeding the parent's plan is exactly what made sibling rollouts converge.
    """
    import langgraph_coe.graphs.mcts as mcts_mod

    src = inspect.getsource(mcts_mod.build_mcts_graph)
    i = src.index("elif rollout_plan:")
    block = src[i : i + 700]
    assert '"plan_frozen"] = False' in block or '"plan_frozen": False' in block
    # Must NOT seed plan / plan_ledger in the rollout payload under this branch.
    upto_next = block.split("cot_out = await")[0]
    assert '"plan":' not in upto_next, "seeding the parent plan defeats rollout scope"
    assert '"plan_ledger":' not in upto_next


def test_gen_plan_regenerates_only_when_no_plan_was_handed_down():
    """The inherit-vs-generate switch that rollout scope depends on."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    i = src.index("async def gen_plan")
    head = src[i : i + 400]
    assert 'if state.get("plan"):' in head and "return {}" in head


def test_the_rollout_ledger_reaches_synthesis_but_never_a_sibling():
    """The confirmed mechanism (scaffolding exclusion) must survive rollout scope.

    It drove the scaffolding-answer rate from 1.8x the no-plan arm to exact parity in
    CoT, so losing it under MCTS would discard the one effect with a confirmed cause.
    """
    import langgraph_coe.graphs.mcts as mcts_mod

    src = inspect.getsource(mcts_mod.build_mcts_graph)
    # Written to its own channel, so nothing inherits it.
    assert 'out["rollout_plan_ledger"]' in src
    # And consumed by synthesis, with the tree ledger taking precedence.
    assert 'state.get("plan_ledger")\n            or state.get("rollout_plan_ledger")' in src
    assert "scaffolding_findings(synthesis_ledger)" in src
    assert "resolved_findings(synthesis_ledger)" in src


def test_initial_mcts_state_seeds_the_rollout_ledger():
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.system import _initial_mcts_state

    st = _initial_mcts_state("q", LangGraphCoeConfig())
    assert "rollout_plan_ledger" in st and st["rollout_plan_ledger"] == []


# ──────────────────────────────────────────────────────────────────────────────
# Bypasses found by the audit. Each was a place a plan mechanism was computed
# and then silently discarded.
# ──────────────────────────────────────────────────────────────────────────────


def test_the_kg_gate_tests_the_query_it_will_actually_send():
    """The gate read the ungrounded subquestion while sending the grounded one.

    A circumlocuting subquestion names no entity, so ``_subq_hits_known_entity`` was
    False and the KG was skipped — for precisely the queries that grounding had just
    made KG-answerable. Reachable in the shipped CoT config.
    """
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    i = src.index("tagged_kg = needs_kg[i]")
    block = src[i : i + 300]
    assert "_subq_hits_known_entity(rq, known_labels)" in block
    assert "_subq_hits_known_entity(sq, known_labels)" not in block
    # And the same gate inside gather_evidence.
    gsrc = inspect.getsource(c.gather_evidence)
    assert "_subq_hits_known_entity(queries[i], known_labels)" in gsrc

    # Behavioural: the grounded form hits, the bare form does not.
    labels = ["dolly parton"]
    bare = "What is the date of birth for the performer associated with 'Hits'?"
    assert not c._subq_hits_known_entity(bare, labels)
    assert c._subq_hits_known_entity(bare + " Dolly Parton", labels)


def test_a_plan_chain_cannot_deepen_an_mcts_rollout():
    """``effective_max_depth`` raises max_depth to fit a chain — right in CoT, wrong here.

    ``max_simulation_depth`` sizes the tree, so a rollout running deeper than its budget
    charges the difference to every iteration.
    """
    led = cot_mod.build_plan_ledger(
        ["a", "b", "c", "d"], depends_on=[None, 0, 1, 2]
    )
    assert cot_mod.plan_chain_depth(led) == 4
    soft = {"max_depth": 2, "plan_ledger": led}
    assert cot_mod.effective_max_depth(soft) == 5, "standalone CoT still gets the chain"
    hard = {"max_depth": 2, "plan_ledger": led, "max_depth_is_hard": True}
    assert cot_mod.effective_max_depth(hard) == 2, "a rollout budget is a hard cap"
    # And the rollout payload sets it.
    import langgraph_coe.graphs.mcts as m

    assert '"max_depth_is_hard": True' in inspect.getsource(m.build_mcts_graph)


def test_a_rollout_that_resolved_its_plan_counts_as_a_sufficiency_vote():
    """``route_after_subq`` reaches gen_final without setting ``is_answerable``.

    Reading only that flag discarded the plan's own stop condition, so MCTS never
    learned that a rollout finished because the plan was complete.
    """
    import langgraph_coe.graphs.mcts as m

    src = inspect.getsource(m.build_mcts_graph)
    assert 'plan_target_resolved(cot_out.get("plan_ledger") or [])' in src
    # Anchor on the real assignment, not the early-return ``: False`` above it.
    i = src.index('"rollout_semantic_signal": bool(')
    assert "or plan_target_resolved" in src[i : i + 260]


def test_self_correction_reretrieval_keeps_its_intent_attribution():
    """``_gen_subqa`` had the intent index and dropped it when writing the node.

    So ``_gen_self_correct`` re-retrieved a circumlocuting sub-question verbatim —
    the exact query class ground_retrieval_query exists to repair.
    """
    import langgraph_coe.graphs.mcts as m

    src = inspect.getsource(m.build_mcts_graph)
    assert '"serves_intent": intent_idx' in src, "attribution must be stored on the node"
    assert 'content.get("serves_intent")' in src, "and read back for re-retrieval"
    assert "serves_intent=[target_intent]" in src, "and forwarded to gather_evidence"


def test_memory_reverification_keeps_the_pruning_focus():
    """Omitting the ledger does not leave the focus alone — it disables it.

    ``gather_evidence`` calls ``set_plan_focus([], 0)``, which sets the ContextVar to
    None, so re-verification paid the unfocused 64-candidate pruner cost while every
    sibling retrieval in the same iteration paid 16.
    """
    import langgraph_coe.graphs.mcts as m

    src = inspect.getsource(m.build_mcts_graph)
    i = src.index("needs_kg=[False] * len(facts_q)")
    block = src[i : i + 700]
    assert 'plan_ledger=view.get("plan_ledger")' in block
    # No ``serves_intent=`` KWARG (the comment names it, which is fine): passing one
    # would let ground_retrieval_query rewrite the fact strings being re-verified.
    assert "serves_intent=" not in block.split("memory_context")[0], (
        "facts must not be rewritten — only the focus is wanted here"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terminal-referent verification. The dominant conversion failure: on 70
# questions whose memory held the gold and whose answer was still wrong, the
# terminal intent had closed on the WRONG referent 67% of the time, and the
# consolidator's conflict detection caught only 3% — the rest were silent.
# ──────────────────────────────────────────────────────────────────────────────


def test_terminal_intents_are_the_ones_nothing_depends_on():
    led = cot_mod.build_plan_ledger(
        ["who is A", "where born", "what year"], depends_on=[None, 0, 1]
    )
    assert cot_mod.terminal_intents(led) == [2]
    flat = cot_mod.build_plan_ledger(["who directed", "who scored"])
    assert cot_mod.terminal_intents(flat) == [0, 1], "a flat plan is all terminal"
    assert cot_mod.terminal_intents([]) == []


def test_verification_is_disabled_by_default_and_configurable():
    """Measured out: paired conversion 14/18 vs 14/18 at +5.8 calls/question."""
    from langgraph_coe.config import PlanConfig

    assert PlanConfig().verify_terminal_referents is False
    assert PlanConfig(verify_terminal_referents=True).verify_terminal_referents is True


def test_the_verifier_replaces_a_referent_the_evidence_contradicts():
    """The Canyon case: two different Canyons, the answerer picked the wrong county."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    i = src.index("if verify_terminal and candidates:")
    block = src[i : i + 4200]
    # Only terminal intents are checked — a scaffolding error surfaces when its
    # dependent fails, and checking every intent would cost a call per subquestion.
    assert "terminals = set(terminal_intents(ledger" in block
    assert "if idx in terminals" in block
    # A replacement must itself be corroborated by the evidence.
    assert "_is_corroborated(refined, evidence)" in block
    # correct/partial close unchanged; incorrect/unsupported are acted on.
    assert 'status in ("correct", "partial")' in block
    # An uncorroborated refinement withholds rather than binding either value.
    assert 'candidates[pos] = (idx, "")' in block
    # And a failure of the verification call itself must not block the hop.
    assert "closing on the unverified answers" in block


def test_the_verifier_reuses_self_corrector_rather_than_a_new_role():
    """It already returns correct/partial/incorrect/unsupported plus a refinement."""
    import langgraph_coe.graphs.cot as c
    from langgraph_coe.roles import SELF_CORRECTOR, SelfCorrectionOutput

    assert SELF_CORRECTOR.output_model is SelfCorrectionOutput
    src = inspect.getsource(c.build_cot_graph)
    assert "execute_role_lc(registry, SELF_CORRECTOR, inputs)" in src
    # The plan is passed so the check knows which intent is being verified.
    i = src.index("if verify_terminal and candidates:")
    assert "render_plan_for_prompt(" in src[i : i + 4200]


def test_a_safe_default_verification_does_not_disturb_the_binding():
    """A retry-exhausted parse failure must not be read as "incorrect"."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    i = src.index("if verify_terminal and candidates:")
    block = src[i : i + 4200]
    assert "is_safe_default(out)" in block and "continue" in block


# ──────────────────────────────────────────────────────────────────────────────
# Guards (polarity intents): a truth value is not a referent
#
# The PLANNER prompt asks for presuppositions to be hedged into conditionals, so the
# ledger is full of intents whose answer is yes or no. Everything downstream assumed a
# referent. Measured over 1,920 questions / 6,250 intents: 399 guards (6.4%), 188 of them
# terminal, 131 of those closed, and 139 of 284 closed guards bound a full sentence.
# ──────────────────────────────────────────────────────────────────────────────


def test_a_guard_is_recognised_and_an_ordinary_intent_is_not():
    pol = cot_mod.is_polarity_intent
    assert pol("Determine whether she had a spouse.")
    assert pol("Verify whether the identified country maintains border troops.")
    assert pol("Is the headquarters location of Yaxing Coach a capitol city?")
    assert pol("If the USS Kajeruna is real, determine whether it was commissioned.")
    assert pol("Establish whether or not the ranking is settled.")
    # Not guards: these name a thing to find.
    assert not pol("Determine the city that shares a border with Saint Paul.")
    assert not pol("Identify the birthplace of Elizabeth Berg.")
    assert not pol("Find the three-letter abbreviation for the identified country.")
    # "if" inside an ordinary intent must not trigger it.
    assert not pol("Find the river by that city, if it is named in the evidence.")
    assert not cot_mod.is_polarity_intent("")


def test_an_affirmation_is_stripped_so_the_referent_behind_it_resolves():
    """'Yes, Meg Ryan.' bound Dennis Quaid — the intent's own input — or nothing."""
    assert cot_mod.strip_affirmation("Yes, Meg Ryan.") == ("Meg Ryan.", True)
    assert cot_mod.strip_affirmation("No, Yangzhou is not a capital.")[1] is False
    assert cot_mod.strip_affirmation("Incorrect - Berlin.") == ("Berlin.", False)
    # No affirmation: unchanged, and polarity unknown rather than assumed.
    assert cot_mod.strip_affirmation("Meg Ryan") == ("Meg Ryan", None)
    assert cot_mod.strip_affirmation("") == ("", None)
    # "Nokia" must not be read as a leading "no", and neither must "no information" —
    # without the delimiter requirement that left "information", which the phrase tier
    # closed an intent on.
    assert cot_mod.strip_affirmation("Nokia") == ("Nokia", None)
    assert cot_mod.strip_affirmation("no information") == ("no information", None)


def test_a_guard_binds_a_truth_value_not_the_intent_s_own_input():
    led = cot_mod.build_plan_ledger(
        [
            "Identify the actor who plays the father.",
            "Determine whether the identified actor has a wife who is an actress.",
        ],
        depends_on=[None, 0],
    )
    labels = {"dennis quaid": "Q1", "meg ryan": "Q2"}
    mem = [
        "[Retrieval]: Dennis Quaid is married to Meg Ryan.",
        "[Retrieval]: Yes, Dennis Quaid is married to an actress.",
    ]
    out = cot_mod.apply_bindings(
        led,
        [
            (1, "Yes, Dennis Quaid is married to an actress."),
            (1, "Yes, Meg Ryan."),
        ],
        labels,
        hop=1,
        retrieval_memory=mem,
    )
    entry = out[1]
    # One polarity key, so the input can no longer masquerade as a rival answer. Before
    # this, the two surfaces resolved to Q1 and Q2 and the intent went CONTESTED, which
    # at replan_max=0 blocks closure permanently.
    assert entry["status"] == cot_mod.INTENT_CLOSED
    assert [b["qid"] for b in entry["bindings"]] == ["pol:true"]
    assert entry["bindings"][0]["polarity"] is True


def test_a_guard_answered_both_ways_records_the_conflict_and_still_closes():
    led = cot_mod.build_plan_ledger(["Determine whether X held office."])
    mem = ["[Retrieval]: Yes, X held office.", "[Retrieval]: No, X never held office."]
    out = cot_mod.apply_bindings(
        led,
        [(0, "Yes, X held office."), (0, "No, X never held office.")],
        {},
        hop=0,
        retrieval_memory=mem,
    )
    entry = out[0]
    assert entry["polarity_conflict"] is True
    # Closed anyway: a contested guard can never close, and the guard is not what the
    # question asks, so blocking termination on it only buys hops that cannot help.
    assert entry["status"] == cot_mod.INTENT_CLOSED
    assert len(entry["bindings"]) == 1


def test_an_affirmed_referent_reaches_synthesis_without_the_affirmation():
    led = cot_mod.build_plan_ledger(["Identify the wife of the actor."])
    out = cot_mod.apply_bindings(
        led,
        [(0, "Yes, Meg Ryan.")],
        {"meg ryan": "Q2"},
        hop=0,
        retrieval_memory=["[Retrieval]: Dennis Quaid is married to Meg Ryan."],
    )
    assert out[0]["bindings"][0]["qid"] == "Q2"
    assert out[0]["bindings"][0]["surface"] == "Meg Ryan."
    assert cot_mod.resolved_findings(out) == [
        "Identify the wife of the actor -> Meg Ryan."
    ]


def test_a_truth_value_is_never_offered_to_synthesis_as_a_candidate_answer():
    """This block is ranked above every other context source in the prompt."""
    led = cot_mod.build_plan_ledger(
        ["Confirm whether the author wrote a short story featuring Herman Wouk."]
    )
    out = cot_mod.apply_bindings(
        led,
        [(0, "No, Stephen King did not write a short story featuring Herman Wouk.")],
        {"stephen king": "Q3"},
        hop=0,
        retrieval_memory=[
            "[Retrieval]: No, Stephen King did not write a short story "
            "featuring Herman Wouk."
        ],
    )
    assert out[0]["status"] == cot_mod.INTENT_CLOSED
    assert cot_mod.resolved_findings(out) == []
    assert cot_mod.scaffolding_findings(out) == []


def test_a_sentence_is_never_offered_to_synthesis_as_a_candidate_answer():
    led = cot_mod.build_plan_ledger(["Identify the treaty that ceded the territory."])
    long_surface = (
        "The land that became St. Louis was acquired by the United States in 1804 "
        "as part of the Louisiana Purchase."
    )
    out = cot_mod.apply_bindings(
        led, [(0, long_surface)], {}, hop=0, retrieval_memory=[f"[Retrieval]: {long_surface}"]
    )
    # It may well close — the phrase tier is corroborated — but a 20-word sentence
    # presented at top authority is a paragraph, not a referent to return.
    assert cot_mod.resolved_findings(out) == []


def test_a_guard_is_not_a_termination_target_when_a_real_target_exists():
    led = cot_mod.build_plan_ledger(
        [
            "Identify the birth city.",
            "Determine whether the identified birth city hosts NASCAR races.",
            "Identify the track that hosts them.",
        ],
        depends_on=[None, 0, 1],
    )
    # Only intent 2 is a real target; intent 1 is depended on anyway.
    assert cot_mod.terminal_intents(led) == [2]
    flat = cot_mod.build_plan_ledger(
        [
            "Identify the track that hosts the races.",
            "Determine whether the city hosts NASCAR races.",
        ]
    )
    assert cot_mod.terminal_intents(flat) == [0], "the guard is not a target"
    # All-terminal-guard plans are malformed (2.8% of questions); return them rather
    # than an empty list so the caller decides.
    only = cot_mod.build_plan_ledger(["Determine whether the city hosts races."])
    assert cot_mod.terminal_intents(only) == [0]


def test_a_closed_guard_cannot_end_the_loop():
    """A guard now always closes, so this is what stops it standing in for the answer."""
    led = cot_mod.build_plan_ledger(["Determine whether the city hosts NASCAR races."])
    out = cot_mod.apply_bindings(
        led,
        [(0, "No, Tucson does not host NASCAR races.")],
        {},
        hop=0,
        retrieval_memory=["[Retrieval]: Tucson does not host NASCAR races."],
    )
    assert out[0]["status"] == cot_mod.INTENT_CLOSED
    assert cot_mod.plan_target_resolved(out) is False
    # A real target still ends it.
    real = cot_mod.apply_bindings(
        cot_mod.build_plan_ledger(["Identify the track."]),
        [(0, "Tucson Raceway Park")],
        {},
        hop=0,
        retrieval_memory=["[Retrieval]: Tucson Raceway Park is in Tucson."],
    )
    assert cot_mod.plan_target_resolved(real) is True


def test_a_truth_value_never_enters_a_retrieval_query():
    led = cot_mod.build_plan_ledger(
        [
            "Determine whether the country maintained border troops.",
            "Find the three-letter abbreviation for the identified country.",
        ],
        depends_on=[None, 0],
    )
    out = cot_mod.apply_bindings(
        led,
        [(0, "No, Germany does not maintain border troops.")],
        {"germany": "Q183"},
        hop=0,
        retrieval_memory=["[Retrieval]: Germany does not maintain border troops."],
    )
    subq = "What is the three-letter abbreviation for the identified country?"
    assert cot_mod.ground_retrieval_query(subq, out, 1) == subq


def test_the_planner_prompt_forbids_a_guard_as_the_final_step():
    prompt = roles_mod.PLANNER.system_prompt
    assert "never the plan's last step" in prompt
    # The worked example is the measured failure: the plan closed on "no" for a
    # question that asked *where* the races are held.
    assert "Determine whether the identified birthplace hosts NASCAR races" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# An intent must not bind the referent its own prerequisites already bound
#
# "Earliest mention wins" is right for a concise answer and backwards for a sentence one:
# in "Dennis Quaid is married to Meg Ryan" the earliest linked entity is the subject the
# intent was asked *about*. Measured on 5,593 intents with bindings: 897 (16%) resolved to
# a prerequisite's referent, and 640 closed on nothing else.
# ──────────────────────────────────────────────────────────────────────────────


def test_prerequisite_keys_walk_the_chain_transitively():
    led = cot_mod.build_plan_ledger(
        ["who is A", "where born", "what river"], depends_on=[None, 0, 1]
    )
    led[0]["bindings"] = [{"surface": "Elizabeth Berg", "qid": "Q1"}]
    led[1]["bindings"] = [{"surface": "Saint Paul", "qid": "Q2"}]
    assert cot_mod._prerequisite_keys(led, 2) == {"Q1", "Q2"}, "two hops back, not one"
    assert cot_mod._prerequisite_keys(led, 1) == {"Q1"}
    assert cot_mod._prerequisite_keys(led, 0) == set()
    # A guard's truth value is not an input referent, so it is not excluded.
    led[1]["bindings"] = [{"surface": "Yes", "qid": "pol:true", "polarity": True}]
    assert cot_mod._prerequisite_keys(led, 2) == {"Q1"}


def test_a_cycle_cannot_hang_the_prerequisite_walk():
    """``build_plan_ledger`` breaks cycles, so this is built by hand on purpose."""
    led = [
        {"intent": "a", "depends_on": 1, "bindings": [{"surface": "A", "qid": "Q1"}]},
        {"intent": "b", "depends_on": 0, "bindings": [{"surface": "B", "qid": "Q2"}]},
    ]
    assert cot_mod._prerequisite_keys(led, 0) == {"Q1", "Q2"}
    # An out-of-range index is ignored rather than raising.
    assert cot_mod._prerequisite_keys([{"intent": "a", "depends_on": 9}], 0) == set()


def test_the_answer_in_the_predicate_wins_over_the_subject_in_the_sentence():
    led = cot_mod.build_plan_ledger(
        ["Identify the actor.", "Identify the actor's wife."], depends_on=[None, 0]
    )
    led[0]["bindings"] = [{"surface": "Dennis Quaid", "qid": "Q1", "grounded": True}]
    labels = {"dennis quaid": "Q1", "meg ryan": "Q2"}
    mem = ["[Retrieval]: Dennis Quaid is married to Meg Ryan."]
    answer = "Dennis Quaid is married to Meg Ryan."
    # Off: the subject wins, and the intent closes on its own input.
    off = cot_mod.apply_bindings(
        led, [(1, answer)], labels, hop=1, retrieval_memory=mem
    )
    assert off[1]["bindings"][0]["qid"] == "Q1"
    # On: the input is skipped and the predicate's referent is bound.
    on = cot_mod.apply_bindings(
        led,
        [(1, answer)],
        labels,
        hop=1,
        retrieval_memory=mem,
        skip_input_referent=True,
    )
    assert on[1]["bindings"][0]["qid"] == "Q2"
    assert on[1]["status"] == cot_mod.INTENT_CLOSED


def test_an_answer_naming_only_the_input_still_binds_it():
    """Some intents legitimately re-name their subject; binding nothing is worse."""
    led = cot_mod.build_plan_ledger(
        ["Identify the city.", "Which of the two cities is meant?"], depends_on=[None, 0]
    )
    led[0]["bindings"] = [{"surface": "Canyon", "qid": "Q1", "grounded": True}]
    on = cot_mod.apply_bindings(
        led,
        [(1, "Canyon")],
        {"canyon": "Q1"},
        hop=1,
        retrieval_memory=["[Retrieval]: Canyon is the county seat of Randall County."],
        skip_input_referent=True,
    )
    assert on[1]["bindings"][0]["qid"] == "Q1"


def test_excluding_the_input_is_off_by_default_and_configurable():
    from langgraph_coe.config import PlanConfig

    assert PlanConfig().skip_input_referent_in_binding is False
    assert PlanConfig(skip_input_referent_in_binding=True).skip_input_referent_in_binding
    assert PlanConfig().guard_intents_are_not_referents is True


def test_resolve_primary_qid_excludes_without_disturbing_the_default():
    labels = {"dennis quaid": "Q1", "meg ryan": "Q2"}
    text = "Dennis Quaid is married to Meg Ryan."
    assert cot_mod.resolve_primary_qid(text, labels) == "Q1"
    assert cot_mod.resolve_primary_qid(text, labels, exclude={"Q1"}) == "Q2"
    # Every mention excluded -> fall back rather than resolving to nothing.
    assert cot_mod.resolve_primary_qid(text, labels, exclude={"Q1", "Q2"}) == "Q1"
    assert cot_mod.resolve_binding_key(text, labels, exclude={"Q1"}) == "Q2"


def test_both_binding_flags_are_threaded_from_config_to_apply_bindings():
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    assert 'getattr(plan_cfg, "guard_intents_are_not_referents", True)' in src
    assert 'getattr(plan_cfg, "skip_input_referent_in_binding", False)' in src
    assert "guard_intents=guard_intents" in src
    assert "skip_input_referent=skip_input_referent" in src


def test_a_corroborated_low_confidence_answer_can_be_rescued():
    """16.7% of sub-answers are dropped on the confidence label before binding."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    assert 'getattr(plan_cfg, "bind_corroborated_low_confidence", False)' in src
    i = src.index("if i < len(confidences) and confidences[i] in _LOW_CONFIDENCE:")
    block = src[i : i + 900]
    # Corroboration is the arbiter, so an uncorroborated guess still cannot bind.
    assert "rescue_low_confidence and _is_corroborated(answer, gate_lines)" in block
    assert "low_confidence += 1" in block and "continue" in block
    # And the rate is recorded per hop rather than only logged.
    assert '"answers_low_confidence_rescued": low_confidence_rescued,' in src


def test_the_low_confidence_rescue_is_off_by_default_and_configurable():
    from langgraph_coe.config import PlanConfig

    assert PlanConfig().bind_corroborated_low_confidence is False
    assert PlanConfig(
        bind_corroborated_low_confidence=True
    ).bind_corroborated_low_confidence


def test_the_answer_generator_is_told_to_answer_on_incomplete_context():
    """Why a low label is not evidence about the referent — it is about the context."""
    prompt = roles_mod.ANSWER_GENERATOR.system_prompt
    assert "confidence_level" in prompt
    assert "even if the context is incomplete" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# candidate_answers ordering: the 94% channel into synthesis
#
# resolved_findings is 0.49 lines per question; candidate_answers is the whole of
# text_memory. Measured over 149 conversion failures with distinct gold/answered lines, the
# gold sits LATER in memory 101 times against 48 (p < 0.0001) — synthesis returned the
# earlier line 68% of the time.
# ──────────────────────────────────────────────────────────────────────────────


def test_the_latest_evidence_is_presented_first():
    lines = ["[hop=0] a", "[hop=1] b", "[Retrieval]: c"]
    assert cot_mod.order_candidates_recent_first(lines) == [
        "[Retrieval]: c",
        "[hop=1] b",
        "[hop=0] a",
    ]
    assert cot_mod.order_candidates_recent_first([]) == []
    assert cot_mod.order_candidates_recent_first(None) == []


def test_an_untagged_line_is_not_demoted():
    """A measured case had the GOLD on a line with no [hop=N] tag.

    That is why this is a plain reverse and not a sort on the hop tag: sorting untagged
    lines to the end would demote exactly what the reordering exists to promote.
    """
    lines = ["[hop=0] early", "[Retrieval]: the gold, untagged", "[hop=3] late"]
    out = cot_mod.order_candidates_recent_first(lines)
    assert out.index("[Retrieval]: the gold, untagged") < out.index("[hop=0] early")


def test_the_ordering_is_off_by_default_and_configurable():
    from langgraph_coe.config import CoTConfig

    assert CoTConfig().recent_evidence_first is False
    assert CoTConfig(recent_evidence_first=True).recent_evidence_first is True


def test_synthesis_applies_the_ordering_only_when_enabled():
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    assert '"recent_evidence_first", False)' in src
    i = src.index("async def gen_final(")
    block = src[i : i + 500]
    assert "if recent_evidence_first:" in block
    assert "order_candidates_recent_first(candidate_answers)" in block


def test_relevance_ranking_is_documented_as_measured_harmful():
    """It favours the wrong line, so a future reader must not 'improve' this to a reranker."""
    doc = cot_mod.order_candidates_recent_first.__doc__ or ""
    assert "favour the WRONG line" in doc
    assert "28 / rival 71" in doc


def test_the_ordering_evidence_records_its_own_weakness():
    """The unmatched p < 0.0001 was an overstatement; the docstring must say so."""
    doc = cot_mod.order_candidates_recent_first.__doc__ or ""
    assert "67 times against 41" in doc and "p = 0.0157" in doc
    assert "0.425" in doc and "unconditioned" in doc


# ──────────────────────────────────────────────────────────────────────────────
# Consolidation loss: evidence that never reached synthesis at all
#
# Measured on 60 instrumented questions: the gold is in the retrieved facts on 22, in
# consolidated memory on 19 — 6 questions (10%) lose it to consolidation, and all 6 were
# answered wrong. MEMORY_CONSOLIDATOR keeps a mean 0.56 of retrieved facts.
# ──────────────────────────────────────────────────────────────────────────────


def test_only_the_dropped_facts_are_resurfaced():
    kept = "Elizabeth Berg's birthplace is Saint Paul"
    log = [kept, "Saint Paul borders Minneapolis", "Unrelated trivia about rowing"]
    out = cot_mod.dropped_evidence(log, [f"[hop=1] [Retrieval]: {kept}."])
    assert kept not in out, "already in candidate_answers; resending it pays tokens twice"
    assert "Saint Paul borders Minneapolis" in out
    assert "Unrelated trivia about rowing" in out


def test_a_paraphrased_survivor_counts_as_kept():
    """Exact containment would call almost everything dropped — the consolidator rewrites."""
    fact = "Dennis Quaid is married to the actress Meg Ryan"
    memory = ["[Retrieval]: Dennis Quaid is married to actress Meg Ryan."]
    assert cot_mod.dropped_evidence([fact], memory) == []


def test_dropped_evidence_is_most_recent_first_and_capped():
    log = [f"fact number {i} about a distinct subject" for i in range(40)]
    out = cot_mod.dropped_evidence(log, [], limit=5)
    assert len(out) == 5
    assert out[0] == "fact number 39 about a distinct subject", "latest hop first"


def test_dropped_evidence_tolerates_empty_and_junk():
    assert cot_mod.dropped_evidence([], []) == []
    assert cot_mod.dropped_evidence(None, None) == []
    assert cot_mod.dropped_evidence(["", "   ", "a"], []) == [], "no content tokens"


def test_the_dropped_block_is_appended_last_and_labelled_lower_reliability():
    """Anything at top authority gets returned as the answer — the scaffolding result."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c.build_cot_graph)
    i = src.index("async def gen_final(")
    block = src[i : i + 1800]
    assert "if synthesis_sees_dropped:" in block
    assert "ctx = ctx + (" in block, "appended, not prepended"
    assert "Retrieved but not retained" in block
    assert "lower reliability" in block


def test_the_dropped_evidence_flag_is_off_by_default_and_configurable():
    from langgraph_coe.config import CoTConfig

    assert CoTConfig().synthesis_sees_dropped_evidence is False
    assert CoTConfig(
        synthesis_sees_dropped_evidence=True
    ).synthesis_sees_dropped_evidence


def test_the_retrieval_log_accumulates_across_hops():
    """``extracted_facts`` is cleared by ``increment``, so a plain key would lose it."""
    import langgraph_coe.graphs.cot as c

    src = inspect.getsource(c)
    assert "retrieval_log: Annotated[List[str], operator.add]" in src
    # Written wherever facts are produced, so no hop's evidence is missed.
    assert '"retrieval_log": list(facts_out)' in src


def test_the_runner_persists_the_retrieval_log_separately():
    import inspect as _i

    from langgraph_coe.evaluation import runner as r

    src = _i.getsource(r._save_question_artifacts)
    assert 'q_dir / "retrieval_log.json"' in src
    assert '"retrieval_log_path"' in src
