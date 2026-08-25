"""Regression tests for five verified defects.

Each of these turned infrastructure flakiness into something shaped like a real
reasoning signal, which is why they are grouped: three of them would manufacture
false plan-failure triggers, and the fourth makes branch-local memory a fiction.

0.1 A retry-exhausted SUBQUESTION_GENERATOR parse failure was rewritten into
    "the question is answerable, synthesize now".
0.2 A blank sub-answer was dropped rather than blanked, shifting every later
    answer onto the wrong sub-question.
0.3 A VERIFIER parse failure produced ``rating=0.0``, which *passes* the
    ``ge=0.0`` bound and so read as a confident "worst possible answer".
0.4 ``nx.DiGraph.copy()`` shares edge attribute *sets*, so "work on a copy"
    wrote through to the caller's graph.
0.5 Verifier critiques were tagged ``[System Prediction]`` and therefore re-issued
    verbatim as corpus/KG retrieval queries.
"""

from __future__ import annotations

import networkx as nx

from langgraph_coe import llm as llm_mod
from langgraph_coe import roles as roles_mod
from langgraph_coe.graphs import cot as cot_mod
from langgraph_coe.graphs import memory_update as mem_mod


# ──────────────────────────────────────────────────────────────────────────────
# 0.1 — parse failure must not read as "answerable"
# ──────────────────────────────────────────────────────────────────────────────


def test_safe_default_is_distinguishable_from_a_real_completion():
    """The marker is the only reliable way to detect "the model never answered".

    A neutral default's field values are byte-identical to a genuine response
    whose zero value happens to be meaningful, so callers cannot tell them apart
    by inspecting fields.
    """
    default = llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR)
    assert llm_mod.is_safe_default(default)

    real = roles_mod.SubquestionGenerationOutput(
        is_answerable=False, subquestions=[], needs_kg=[]
    )
    assert not llm_mod.is_safe_default(real)
    # Identical fields, opposite meanings.
    assert real.is_answerable == default.is_answerable
    assert (real.subquestions or []) == (default.subquestions or [])


def test_safe_default_marker_does_not_leak_into_model_dump():
    default = llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR)
    assert "__coe_safe_default__" not in default.model_dump()


def test_pooling_reports_zero_survivors_when_nothing_parsed():
    """``n_survivors`` is what lets the caller branch before reading ``should_direct``."""
    defaults = [
        llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR)
        for _ in range(3)
    ]
    pooled = cot_mod.pool_subquestions(defaults)
    assert pooled.n_survivors == 0
    assert pooled.subquestions == []
    assert pooled.should_direct is False


def test_pooling_counts_only_parsed_completions_in_the_majority_vote():
    """One genuine "answerable" plus two parse failures is a unanimous vote of one.

    Averaging over the failures instead would make a real signal look like a
    minority opinion.
    """
    outs = [
        roles_mod.SubquestionGenerationOutput(is_answerable=True, subquestions=[]),
        llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR),
        llm_mod.build_safe_default_output(roles_mod.SUBQUESTION_GENERATOR),
    ]
    pooled = cot_mod.pool_subquestions(outs)
    assert pooled.n_survivors == 1
    assert pooled.should_direct is True


def test_pooling_normalizes_negative_serves_intent_to_none():
    """``-1`` is the "advances no plan intent" encoding, not a ledger index."""
    out = roles_mod.SubquestionGenerationOutput(
        is_answerable=False,
        subquestions=["A", "B"],
        needs_kg=[True, True],
        serves_intent=[-1, 0],
    )
    pooled = cot_mod.pool_subquestions([out])
    assert pooled.serves_intent == [None, 0]


# ──────────────────────────────────────────────────────────────────────────────
# 0.3 — a VERIFIER parse failure is not a rating of zero
# ──────────────────────────────────────────────────────────────────────────────


def test_verifier_zero_rating_passes_validation_and_so_needs_the_marker():
    """Documents *why* the marker is needed rather than a schema change.

    ``rating: float = Field(..., ge=0.0, le=10.0)`` means the neutral default of
    0.0 is a *valid* rating, so validation cannot catch this.
    """
    default = llm_mod.build_safe_default_output(roles_mod.VERIFIER)
    assert default.rating == 0.0
    assert llm_mod.is_safe_default(default)
    # 0.0 maps to the most negative reward on the (mean - 5) / 5 scale.
    assert (0.0 - 5.0) / 5.0 == -1.0


# ──────────────────────────────────────────────────────────────────────────────
# 0.4 — graph copy must not share mutable attribute containers
# ──────────────────────────────────────────────────────────────────────────────


def test_networkx_copy_shares_edge_attribute_sets():
    """The upstream behaviour this fix exists for. If this ever changes, the
    ``_deep_copy_graph`` indirection can be dropped."""
    g = nx.DiGraph()
    g.add_edge("a", "b", relation={"r1"})
    shallow = g.copy()
    shallow.edges["a", "b"]["relation"].add("leaked")
    assert "leaked" in g.edges["a", "b"]["relation"], (
        "nx.copy() is expected to alias attribute containers"
    )


def test_deep_copy_graph_isolates_edge_relation_sets():
    g = nx.DiGraph()
    g.add_edge("a", "b", relation={"r1"})
    isolated = mem_mod._deep_copy_graph(g)
    isolated.edges["a", "b"]["relation"].add("BAD_FROM_REJECTED_BRANCH")
    assert g.edges["a", "b"]["relation"] == {"r1"}
    assert (
        g.edges["a", "b"]["relation"] is not isolated.edges["a", "b"]["relation"]
    )


def test_deep_copy_graph_isolates_node_attributes_and_preserves_content():
    g = nx.DiGraph()
    g.add_node("Q1", name="France", aliases=["FR"])
    g.add_edge("Q1", "Q2", relation={"capital"})
    isolated = mem_mod._deep_copy_graph(g)
    isolated.nodes["Q1"]["aliases"].append("leaked")
    assert g.nodes["Q1"]["aliases"] == ["FR"]
    assert isolated.nodes["Q1"]["name"] == "France"
    assert isolated.edges["Q1", "Q2"]["relation"] == {"capital"}


def test_add_triple_on_an_existing_edge_is_the_leak_path():
    """A *new predicate between already-connected entities* is the case that leaked.

    ``_relation_already_in_graph`` filters only same-label duplicates, so this
    path is reached in normal operation, not just in principle.
    """
    g = nx.DiGraph()
    g.add_edge("Alice", "Bob", relation={"knows"})
    isolated = mem_mod._deep_copy_graph(g)
    rel = roles_mod.Relation(
        subject="Alice", relation="employs", object="Bob", context=None
    )
    mem_mod._add_triple_to_graph(isolated, rel, {})
    assert isolated.edges["Alice", "Bob"]["relation"] == {"knows", "employs"}
    assert g.edges["Alice", "Bob"]["relation"] == {"knows"}


# ──────────────────────────────────────────────────────────────────────────────
# 0.5 — assessments are not facts and must not become retrieval queries
# ──────────────────────────────────────────────────────────────────────────────


def test_assessment_provenance_round_trips():
    tagged = mem_mod._format_memory_item(
        "Verifier (no context): the answer is plausible but unsupported.",
        roles_mod.SourceType.ASSESSMENT,
    )
    assert tagged.startswith("[Assessment]")
    assert mem_mod._is_assessment(tagged)
    assert not mem_mod._is_retrieval_grounded(tagged)
    assert (
        mem_mod._strip_provenance_tag(tagged)
        == "Verifier (no context): the answer is plausible but unsupported."
    )


def test_assessment_detection_survives_a_hop_prefix():
    tagged = "[hop=2] [Assessment]: Verifier says the answer is thin."
    assert mem_mod._is_assessment(tagged)
    assert not mem_mod._is_retrieval_grounded(tagged)


def test_consolidated_output_preserves_assessment_provenance():
    """Without this mapping an Assessment would silently demote to a prediction,
    putting it straight back on the re-verification path."""
    out = roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[
            roles_mod.MemoryItem(
                content="Verifier: the date is unverified.",
                provenance="Assessment",
                hop_depth=None,
            ),
            roles_mod.MemoryItem(
                content="Paris is the capital of France.",
                provenance="Retrieval",
                hop_depth=1,
            ),
        ]
    )
    rendered = mem_mod._consolidated_to_text(out)
    assert mem_mod._is_assessment(rendered[0])
    assert mem_mod._is_retrieval_grounded(rendered[1])


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — retraction reporting
# ──────────────────────────────────────────────────────────────────────────────


def test_memory_lines_are_numbered_and_decodable():
    items = ["[Retrieval]: A.", "   ", "[System Prediction]: B."]
    blob = mem_mod._format_lines(items)
    assert blob.splitlines() == ["1. [Retrieval]: A.", "2. [System Prediction]: B."]
    # The decoder must apply identical filtering, or a reported line number would
    # resolve to the wrong item.
    assert mem_mod.enumerate_memory_lines(items) == {
        1: "[Retrieval]: A.",
        2: "[System Prediction]: B.",
    }


def test_retractions_resolve_line_numbers_back_to_content():
    out = roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[],
        evicted=[
            roles_mod.EvictedMemoryItem(line=2, reason="contradicted"),
        ],
    )
    line_map = {1: "[Retrieval]: A.", 2: "[System Prediction]: B."}
    resolved = mem_mod._resolve_retractions(out, line_map)
    assert resolved["retractions"] == [
        {"content": "B.", "tagged": "[System Prediction]: B.", "reason": "contradicted"}
    ]


def test_out_of_range_eviction_line_is_dropped_not_raised():
    """A hallucinated index must not break consolidation — ``evicted`` is a
    trigger channel, not a correctness dependency."""
    out = roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[],
        evicted=[roles_mod.EvictedMemoryItem(line=99, reason="contradicted")],
    )
    resolved = mem_mod._resolve_retractions(out, {1: "[Retrieval]: A."})
    assert resolved["retractions"] == []


def test_unresolved_conflicts_need_at_least_two_resolvable_lines():
    out = roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[], unresolved_conflicts=[[1, 2], [1, 99], [3]]
    )
    line_map = {1: "[Retrieval]: A.", 2: "[Retrieval]: not A."}
    resolved = mem_mod._resolve_retractions(out, line_map)
    assert resolved["unresolved_conflicts"] == [["A.", "not A."]]


def test_evicted_is_optional_so_an_omission_cannot_wipe_memory():
    """The failure mode this guards: a *required* field the model omits makes
    ``parse_fallback`` raise, which drives ``build_safe_default_output`` to
    ``consolidated_memory=[]`` → ``updated_text_memory=[]`` → memory is wiped."""
    out = roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[
            roles_mod.MemoryItem(content="A.", provenance="Retrieval", hop_depth=1)
        ]
    )
    assert out.evicted is None
    assert out.unresolved_conflicts is None
    resolved = mem_mod._resolve_retractions(out, {1: "[Retrieval]: A."})
    assert resolved == {"retractions": [], "unresolved_conflicts": []}
    # And the kept item still round-trips.
    assert mem_mod._consolidated_to_text(out) == ["[hop=1] [Retrieval]: A."]
