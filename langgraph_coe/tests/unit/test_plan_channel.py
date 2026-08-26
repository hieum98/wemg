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
