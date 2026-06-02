"""Phase 0 §3.1 — ``web_researcher`` role target specs.

The role does not exist on `main`. These tests describe the shape Phase 0 must
ship: a new ``Role`` with light-tier I/O models, registered in
``LLMConfig.role_tiers``, and resolvable through ``RoleModelRegistry``.
"""

from __future__ import annotations

from typing import get_args, get_origin

import pytest

# Phase 0 introduces these symbols; tests will fail to collect until they exist.
# Import at module top-level so collection failure pinpoints the gap.
from langgraph_coe import roles as roles_mod
from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.llm import RoleModelRegistry
from langgraph_coe.roles import Role


def test_web_researcher_role_exported():
    """A ``WEB_RESEARCHER`` instance of ``Role`` is exported from ``roles``."""
    web_researcher = getattr(roles_mod, "WEB_RESEARCHER", None)
    assert web_researcher is not None, "roles.WEB_RESEARCHER must be defined"
    assert isinstance(web_researcher, Role)
    assert web_researcher.name == "web_researcher"


def test_web_researcher_input_model_shape():
    """``WebResearcherInput`` exposes ``subquery: str`` + ``research_budget: int``.

    Per implementation_plan.md §3.1: "Input: subquery + research_budget".
    """
    inp_cls = getattr(roles_mod, "WebResearcherInput", None)
    assert inp_cls is not None, "WebResearcherInput must be defined"
    fields = inp_cls.model_fields
    assert "subquery" in fields
    assert "research_budget" in fields
    # __str__ used by execute_role_lc — required for the role to slot into
    # the LLM execution pipeline like every other role.
    instance = inp_cls(subquery="capital of France", research_budget=3)
    assert str(instance), "WebResearcherInput.__str__ must return non-empty"


def test_web_researcher_output_model_shape():
    """Output is a list of ``{title, url, snippet, full_text}`` items (§3.1)."""
    out_cls = getattr(roles_mod, "WebResearcherOutput", None)
    assert out_cls is not None, "WebResearcherOutput must be defined"
    fields = out_cls.model_fields
    # The carrier field must be a list-typed field of structured items.
    list_field_names = [
        n for n, f in fields.items() if get_origin(f.annotation) in (list, type([]))
    ]
    assert list_field_names, (
        "WebResearcherOutput must expose a list-typed field for results"
    )
    # Resolve the inner item type and check the four required keys.
    list_field = list_field_names[0]
    inner = get_args(fields[list_field].annotation)[0]
    inner_fields = getattr(inner, "model_fields", None)
    assert inner_fields is not None, (
        f"WebResearcherOutput.{list_field} items must be a Pydantic model"
    )
    for key in ("title", "url", "snippet", "full_text"):
        assert key in inner_fields, (
            f"WebResearcherOutput item missing required key '{key}' (plan §3.1)"
        )


def test_web_researcher_system_prompt_present_and_substantive():
    """System prompt covers the §3.1 design points (research goal, output shape, stopping criterion)."""
    web_researcher = roles_mod.WEB_RESEARCHER
    prompt = web_researcher.system_prompt
    assert isinstance(prompt, str) and len(prompt) > 100, (
        "web_researcher system_prompt must be substantive (>100 chars)"
    )
    # Soft contract: the prompt mentions the four output keys.
    lowered = prompt.lower()
    for key in ("title", "url", "snippet"):
        assert key in lowered, (
            f"web_researcher system_prompt should reference the output key '{key}'"
        )


def test_web_researcher_registered_in_light_tier():
    """§3.1: web_researcher is a *light tier* role."""
    cfg = LangGraphCoeConfig()
    assert cfg.llm.role_tiers.get("web_researcher") == "light", (
        "LLMConfig.role_tiers['web_researcher'] must be 'light' per plan §3.1"
    )


def test_registry_resolves_web_researcher_to_light_model():
    """``RoleModelRegistry.get_model('web_researcher')`` returns the *light* tier instance."""
    cfg = LangGraphCoeConfig()
    registry = RoleModelRegistry(cfg.llm)
    light = registry.get_model_by_tier("light")
    resolved = registry.get_model("web_researcher")
    assert resolved is light, (
        "web_researcher must resolve to the same ChatLiteLLM instance as the light tier"
    )
