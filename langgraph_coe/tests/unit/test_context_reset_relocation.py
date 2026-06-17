"""ContextVar reset relocation behavior.

Why this matters:

  Under MCTS, KGSearchGraph runs many times per question (every CoT iteration
  × every rollout). Per-invocation reset would wipe the ``_cv_visited_qids``
  set that the three-layer loop prevention depends on.

  Target: ``reset_wikidata_session()`` (and the new ``reset_web_research_session()``)
  is called exactly once per question, from ``system.py``, *before* the
  strategy graph is invoked. Subsequent graph runs inside the same question
  do not reset.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph_coe.tools import wikidata as wd_mod

# ──────────────────────────────────────────────────────────────────────────────
# A — reset semantics
# ──────────────────────────────────────────────────────────────────────────────


def test_reset_clears_visited_and_hop_count():
    """Reset rebinds the ContextVar to a fresh ``_SessionState``."""
    s = wd_mod._get_session()
    s.visited.update(["Q1", "Q2"])
    s.hop_count = 5

    wd_mod.reset_wikidata_session()

    s2 = wd_mod._get_session()
    assert s2 is not s, "reset must rebind ContextVar to a NEW _SessionState"
    assert s2.visited == set()
    assert s2.hop_count == 0


def test_reset_does_not_clear_global_entity_cache():
    """``entity_cache`` is name→QID memoization across questions and must survive reset."""
    wd_mod.entity_cache["TestEntity"] = "Q999"
    wd_mod.reset_wikidata_session()
    assert wd_mod.entity_cache.get("TestEntity") == "Q999", (
        "reset_wikidata_session must NOT clear the global entity_cache (cross-question reuse)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# B — system.answer() orchestrates reset exactly once per question
# ──────────────────────────────────────────────────────────────────────────────


def _build_stub_strategy_graph():
    """Return a compiled-graph-like object whose ``ainvoke`` returns a benign state dict."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"final_answer": "stub", "errors": []})
    return graph


async def test_system_answer_calls_resets_once_per_question(monkeypatch):
    """``answer()`` resets both ContextVar sessions exactly once before invoking the graph."""
    system = pytest.importorskip(
        "langgraph_coe.system",
        reason="Wires resets into system.answer; module must exist post-refactor.",
    )
    if not hasattr(system, "answer"):
        pytest.skip("system.answer unavailable")

    reset_wd_calls = {"n": 0}
    reset_web_calls = {"n": 0}

    def _wd():
        reset_wd_calls["n"] += 1

    def _web():
        reset_web_calls["n"] += 1

    monkeypatch.setattr(system, "reset_wikidata_session", _wd, raising=False)
    monkeypatch.setattr(system, "reset_web_research_session", _web, raising=False)

    # Stub strategy-graph builders so we don't depend on Phases 1–3.
    stub = _build_stub_strategy_graph()
    monkeypatch.setattr(
        system, "build_mcts_graph", lambda *a, **kw: stub, raising=False
    )
    monkeypatch.setattr(system, "build_cot_graph", lambda *a, **kw: stub, raising=False)

    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()

    await system.answer("What is the capital of France?", cfg)

    assert reset_wd_calls["n"] == 1, (
        f"reset_wikidata_session must be called exactly once per question; saw {reset_wd_calls['n']}"
    )
    assert reset_web_calls["n"] == 1, (
        f"reset_web_research_session must be called exactly once per question; saw {reset_web_calls['n']}"
    )


async def test_two_sequential_answers_reset_twice(monkeypatch):
    """Each question gets its own pair of resets — exactly N resets for N questions."""
    system = pytest.importorskip("langgraph_coe.system")
    if not hasattr(system, "answer"):
        pytest.skip("system.answer unavailable")

    reset_wd_calls = {"n": 0}
    monkeypatch.setattr(
        system,
        "reset_wikidata_session",
        lambda: reset_wd_calls.__setitem__("n", reset_wd_calls["n"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        system, "reset_web_research_session", lambda: None, raising=False
    )

    stub = _build_stub_strategy_graph()
    monkeypatch.setattr(
        system, "build_mcts_graph", lambda *a, **kw: stub, raising=False
    )
    monkeypatch.setattr(system, "build_cot_graph", lambda *a, **kw: stub, raising=False)

    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()

    await system.answer("Q1?", cfg)
    await system.answer("Q2?", cfg)

    assert reset_wd_calls["n"] == 2, (
        f"Two sequential answer() calls must yield exactly two resets; saw {reset_wd_calls['n']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C — KGSearchGraph runs do NOT reset
# ──────────────────────────────────────────────────────────────────────────────


async def test_kg_search_runs_dont_clobber_visited_qids(
    monkeypatch, init_wikidata_tools
):
    """3× back-to-back KGSearchGraph invocations preserve visited QIDs.

    Simulates the MCTS rollout scenario: many CoT iterations × many rollouts
    inside one question, all sharing the same ``_cv_visited_qids`` set.
    """
    from langgraph_coe import roles as roles_mod
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.graphs import kg_search as kg_mod
    from langgraph_coe.llm import RoleModelRegistry

    from .._fixtures import StructuredOutputSpy, make_fake_react_agent

    # Pre-seed visited QIDs for the question.
    wd_mod.reset_wikidata_session()
    seeded = {"Q64", "Q183"} # Berlin, Germany
    wd_mod._get_session().visited.update(seeded)

    # Stub NER to return an entity name; stub create_agent to no-op so triple_search exits.
    NEROutput = roles_mod.NEROutput
    fields = NEROutput.model_fields
    if "entities" in fields:
        try:
            ner_value = NEROutput(entities=["Berlin"])
        except Exception:
            inner = next(
                iter(getattr(fields["entities"].annotation, "__args__", []) or [object])
            )
            ner_value = NEROutput(entities=[inner(name="Berlin")])
    else:
        ner_value = NEROutput()

    spy = StructuredOutputSpy(return_value=ner_value)
    registry = RoleModelRegistry(LangGraphCoeConfig.from_yaml().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)
    monkeypatch.setattr(
        kg_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent([]),
        raising=True,
    )
    # Critical: ensure kg_search module does NOT call reset.
    monkeypatch.setattr(
        kg_mod,
        "reset_wikidata_session",
        lambda: (_ for _ in ()).throw(
            AssertionError("KG node must not call reset_wikidata_session")
        ),
        raising=True,
    )

    graph = kg_mod.build_kg_search_graph(registry)
    for _ in range(3):
        await graph.ainvoke(
            {
                "subquery": "Berlin?",
                "original_query": "Berlin?",
                "context": "",
                "errors": [],
            }
        )

    visited = wd_mod._get_session().visited
    assert seeded.issubset(visited), (
        f"Pre-seeded visited QIDs were clobbered by KGSearchGraph runs: "
        f"missing {seeded - visited}"
    )


async def test_concurrent_questions_have_isolated_sessions():
    """Two concurrent ``asyncio.Task``s see independent ``_SessionState`` objects.

    This is the [project_langchain_ainvoke_context_isolation] invariant: each
    task that calls ``reset_wikidata_session`` rebinds its OWN ContextVar copy
    to a fresh object; sibling tasks keep their own.
    """

    async def _per_question(qid_to_seed: str) -> set:
        wd_mod.reset_wikidata_session()
        s = wd_mod._get_session()
        s.visited.add(qid_to_seed)
        await asyncio.sleep(0)
        return set(wd_mod._get_session().visited)

    visited_a, visited_b = await asyncio.gather(
        _per_question("Q-A"), _per_question("Q-B")
    )
    assert visited_a == {"Q-A"}
    assert visited_b == {"Q-B"}
