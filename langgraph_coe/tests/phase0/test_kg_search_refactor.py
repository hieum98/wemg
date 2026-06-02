"""Phase 0 §3.2 — KGSearchGraph refactor target specs.

Goal of the refactor (implementation_plan.md §3.2):
  1. Replace the NER ``create_agent`` loop with a one-shot
     ``model.with_structured_output(NEROutput)`` call.
  2. After NER, call the ``link_entities`` tool directly (no LLM mediation).
  3. ``triple_search_agent`` stays as a ReAct-style agent.
  4. ``enrich`` node unchanged.
  5. ``reset_wikidata_session()`` is *removed* from the NER node and lives in
     ``system.py`` (covered separately in test_context_reset_relocation.py).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from langgraph_coe import roles as roles_mod
from langgraph_coe.graphs import kg_search as kg_mod
from langgraph_coe.tools import wikidata as wd_mod

from .._fixtures import (
    StructuredOutputSpy,
    ToolSpy,
    make_fake_react_agent,
    tool_message,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ner_output(*names: str):
    """Build a NEROutput-shape value the spy will return from with_structured_output."""
    NEROutput = roles_mod.NEROutput  # exists today
    # Accommodate either ``entities: List[str]`` or richer item shapes.
    fields = NEROutput.model_fields
    if "entities" in fields:
        # Detect inner element type
        from typing import get_args, get_origin
        ann = fields["entities"].annotation
        if get_origin(ann) in (list, type([])):
            inner = get_args(ann)[0] if get_args(ann) else str
            if inner is str:
                return NEROutput(entities=list(names))
            # Inner is a Pydantic model — populate by 'name' if available.
            inner_fields = getattr(inner, "model_fields", {})
            key = "name" if "name" in inner_fields else next(iter(inner_fields))
            return NEROutput(entities=[inner(**{key: n}) for n in names])
    # Fallback — construct via first field.
    first = next(iter(fields))
    return NEROutput(**{first: list(names)})


# ──────────────────────────────────────────────────────────────────────────────
# §3.2.1 — NER becomes one-shot structured output
# ──────────────────────────────────────────────────────────────────────────────


async def test_ner_node_uses_with_structured_output_exactly_once(
    monkeypatch, init_wikidata_tools
):
    """The refactored NER node must call ``with_structured_output(NEROutput)`` once.

    No ``create_agent`` invocation may participate in the NER path — that
    recursion was the legacy behavior being removed (§3.2).
    """
    NEROutput = roles_mod.NEROutput
    spy = StructuredOutputSpy(return_value=_ner_output("Berlin", "Germany"))

    # Build a registry whose get_model returns the spy for the NER role only.
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)

    def _get_model(role_name: str):
        if role_name in ("named_entity_recognition", "kg_ner_agent"):
            return spy
        # Triple search agent gets a benign mock; we stub create_agent below.
        return MagicMock()

    monkeypatch.setattr(registry, "get_model", _get_model, raising=False)

    # Spy on create_agent to confirm the NER path does NOT go through it.
    create_agent_calls: List[Dict[str, Any]] = []

    def _fake_create_agent(model, *, tools=None, system_prompt=None, name=None, **kw):
        create_agent_calls.append({
            "model": model, "tools": tools, "system_prompt": system_prompt, "name": name,
        })
        # Return a no-op agent for the triple_search node so the graph still runs.
        return make_fake_react_agent([AIMessage(content="ok")])

    monkeypatch.setattr(kg_mod, "create_agent", _fake_create_agent, raising=True)

    graph = kg_mod.build_kg_search_graph(registry)
    await graph.ainvoke(
        {
            "subquery": "What is the capital of Germany?",
            "original_query": "What is the capital of Germany?",
            "context": "",
            "errors": [],
        }
    )

    # Exactly one with_structured_output call, on NEROutput.
    assert spy.calls == [NEROutput], (
        f"NER node must call with_structured_output(NEROutput) exactly once; saw {spy.calls}"
    )
    # No create_agent call should have been bound to the NER tool.
    for c in create_agent_calls:
        tool_names = [getattr(t, "name", str(t)) for t in (c["tools"] or [])]
        assert "link_entities" not in tool_names, (
            "create_agent must not be wired to the link_entities tool (NER is one-shot per §3.2)"
        )


async def test_ner_node_calls_link_entities_tool_directly(
    monkeypatch, init_wikidata_tools
):
    """After structured output, ``link_entities`` is invoked directly (not via an agent)."""
    spy = StructuredOutputSpy(return_value=_ner_output("Berlin"))

    link_spy = ToolSpy(wd_mod.link_entities)
    monkeypatch.setattr(kg_mod, "link_entities", link_spy, raising=True)

    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)
    monkeypatch.setattr(
        kg_mod, "create_agent",
        lambda *a, **kw: make_fake_react_agent([AIMessage(content="")]),
        raising=True,
    )

    graph = kg_mod.build_kg_search_graph(registry)
    await graph.ainvoke(
        {"subquery": "Berlin?", "original_query": "Berlin?", "context": "", "errors": []}
    )

    assert link_spy.invocations, (
        "Refactored NER node must invoke link_entities directly (§3.2); no invocations observed"
    )


async def test_ner_node_does_not_call_reset_session(monkeypatch, init_wikidata_tools):
    """§3.3 cross-check: the NER node must not call ``reset_wikidata_session`` anymore.

    Under MCTS, KGSearchGraph runs many times per question; a per-invocation
    reset would wipe the visited-QID set the loop-prevention depends on.
    """
    reset_count = {"n": 0}

    def _spy_reset():
        reset_count["n"] += 1

    # Patch at the module where kg_search imports it from.
    monkeypatch.setattr(kg_mod, "reset_wikidata_session", _spy_reset, raising=True)

    spy = StructuredOutputSpy(return_value=_ner_output("Berlin"))
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)
    monkeypatch.setattr(
        kg_mod, "create_agent",
        lambda *a, **kw: make_fake_react_agent([AIMessage(content="")]),
        raising=True,
    )

    graph = kg_mod.build_kg_search_graph(registry)
    await graph.ainvoke(
        {"subquery": "Berlin?", "original_query": "Berlin?", "context": "", "errors": []}
    )

    assert reset_count["n"] == 0, (
        "Refactored NER node must NOT call reset_wikidata_session (§3.3); "
        f"reset was called {reset_count['n']} time(s)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# §3.2.2 — triple_search_agent unchanged (still a tool-calling agent)
# ──────────────────────────────────────────────────────────────────────────────


async def test_triple_search_agent_still_uses_react_agent_with_fetch_tool(
    monkeypatch, init_wikidata_tools
):
    """Triple search remains an agent loop bound to fetch_and_prune (Stage A+B)."""
    spy = StructuredOutputSpy(return_value=_ner_output("Berlin"))
    create_agent_calls: List[Dict[str, Any]] = []

    def _fake_create_agent(model, *, tools=None, system_prompt=None, name=None, **kw):
        create_agent_calls.append({"tools": tools, "name": name})
        return make_fake_react_agent([
            tool_message("fetch_and_prune_subgraph", ["Berlin -- capital -- Germany"]),
        ])

    monkeypatch.setattr(kg_mod, "create_agent", _fake_create_agent, raising=True)

    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)

    graph = kg_mod.build_kg_search_graph(registry)
    final = await graph.ainvoke(
        {"subquery": "Berlin?", "original_query": "Berlin?", "context": "", "errors": []}
    )

    # Exactly one create_agent call, for triple_search, with a tool whose name
    # is fetch_and_prune_subgraph (Stage B wrapper from create_fetch_and_prune_tool).
    assert len(create_agent_calls) == 1, (
        f"create_agent must be used only for triple_search; saw {len(create_agent_calls)} call(s)"
    )
    tool_names = [getattr(t, "name", str(t)) for t in (create_agent_calls[0]["tools"] or [])]
    assert "fetch_and_prune_subgraph" in tool_names

    # The tool trace must propagate into state['triples'].
    assert "Berlin -- capital -- Germany" in (final.get("triples") or [])


# ──────────────────────────────────────────────────────────────────────────────
# §3.2.3 — enrich node unchanged, state shape preserved
# ──────────────────────────────────────────────────────────────────────────────


async def test_state_shape_includes_phase0_keys(monkeypatch, init_wikidata_tools):
    """Compiled KGSearchGraph still populates the keys listed in plan §5 (post-refactor)."""
    spy = StructuredOutputSpy(return_value=_ner_output("Berlin"))
    monkeypatch.setattr(
        kg_mod, "create_agent",
        lambda *a, **kw: make_fake_react_agent([
            tool_message("fetch_and_prune_subgraph", ["Berlin -- capital -- Germany"]),
        ]),
        raising=True,
    )

    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)

    graph = kg_mod.build_kg_search_graph(registry)
    final = await graph.ainvoke(
        {"subquery": "Berlin?", "original_query": "Berlin?", "context": "", "errors": []}
    )

    required = {
        "linked_entities",
        "qids_for_triples",
        "triples",
        "kg_articles",
        "enriched_entity_labels",
        "errors",
    }
    missing = required - set(final.keys())
    assert not missing, f"KGSearchState missing post-refactor keys: {missing}"


async def test_ner_empty_extraction_skips_link_entities(monkeypatch, init_wikidata_tools):
    """If structured output returns no entities, ``link_entities`` must not be called."""
    spy = StructuredOutputSpy(return_value=_ner_output())  # zero entities

    link_spy = ToolSpy(wd_mod.link_entities)
    monkeypatch.setattr(kg_mod, "link_entities", link_spy, raising=True)
    monkeypatch.setattr(
        kg_mod, "create_agent",
        lambda *a, **kw: make_fake_react_agent([AIMessage(content="")]),
        raising=True,
    )

    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    monkeypatch.setattr(registry, "get_model", lambda r: spy, raising=False)

    graph = kg_mod.build_kg_search_graph(registry)
    final = await graph.ainvoke(
        {"subquery": "nothing here", "original_query": "nothing here", "context": "", "errors": []}
    )

    assert link_spy.invocations == [], (
        "link_entities must be skipped when NER yields zero entities"
    )
    assert final.get("qids_for_triples") in ([], None)
