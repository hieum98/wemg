"""Stubs and spies shared across the langgraph_coe test suite (mainly ``unit/``).

Provides the hermetic test doubles — fake chat models, fake ReAct agents,
``ToolSpy``, structured-output spies, and the config-override logger — that let
the unit tier run with no live LLM/SPARQL/Redis. Some helpers import optional
symbols lazily inside the fixture body so module collection still succeeds even
if a dependency is absent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type
from unittest.mock import MagicMock

import pydantic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda

if TYPE_CHECKING:
    from langgraph_coe.llm import RoleModelRegistry


# Stable test URLs / strings reused across tests.
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_C = "https://example.com/c"


_LOG = logging.getLogger("langgraph_coe.tests")


def log_config_override(param: str, default: Any, override: Any, *, reason: str) -> Any:
    """Record (and return) a deliberate deviation from ``config.yaml``.

    Project rule: tests run with the params declared in
    ``langgraph_coe/config.yaml`` — they must not silently redefine hyper-
    parameters. When a test genuinely needs a different value (an env-specific
    endpoint, a smaller model's token budget, a tiny ``top_k`` to exercise a
    cap, …) route it through this helper so the deviation is never silent: it
    logs ``config.yaml=<default> -> test=<override>`` plus the reason at WARNING
    level and to stdout (visible under ``pytest -s``).

    When ``override`` equals ``default`` nothing is logged and the value is
    returned unchanged, so wrapping a no-op is harmless.
    """
    if default == override:
        return override
    msg = (
        f"[config-override] {param}: config.yaml={default!r} -> test={override!r} "
        f"(reason: {reason})"
    )
    _LOG.warning(msg)
    print(msg, flush=True)
    return override


def override_tier_endpoint(
    cfg: Any,
    tier_name: str,
    *,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    reason: str,
) -> Any:
    """Return a ``TierConfig`` copy of ``cfg.llm.tiers[tier_name]`` that keeps
    every config.yaml generation knob (temperature, top_p, top_k, min_p,
    penalties, seed, enable_thinking, …) and overrides *only* the endpoint /
    token-budget fields a particular test deployment requires.

    Every changed field is logged via :func:`log_config_override`. The
    ``thinking_budget`` cap from config.yaml is clamped to stay strictly below an
    overridden (smaller) ``max_tokens`` so the SGLang constraint still holds.
    """
    base = cfg.llm.tiers[tier_name]
    updates: Dict[str, Any] = {}
    if api_key is not None:
        updates["api_key"] = api_key
    if model_name is not None:
        updates["model_name"] = log_config_override(
            f"llm.tiers.{tier_name}.model_name",
            base.model_name,
            model_name,
            reason=reason,
        )
    if api_base is not None:
        updates["api_base"] = log_config_override(
            f"llm.tiers.{tier_name}.api_base", base.api_base, api_base, reason=reason
        )
    if max_tokens is not None:
        updates["max_tokens"] = log_config_override(
            f"llm.tiers.{tier_name}.max_tokens",
            base.max_tokens,
            max_tokens,
            reason=reason,
        )
    if max_input_tokens is not None:
        updates["max_input_tokens"] = log_config_override(
            f"llm.tiers.{tier_name}.max_input_tokens",
            base.max_input_tokens,
            max_input_tokens,
            reason=reason,
        )
    eff_max = updates.get("max_tokens", base.max_tokens)
    if base.thinking_budget is not None and base.thinking_budget >= eff_max:
        updates["thinking_budget"] = log_config_override(
            f"llm.tiers.{tier_name}.thinking_budget",
            base.thinking_budget,
            max(eff_max - 1024, 1),
            reason=f"config cap must stay below overridden max_tokens={eff_max}",
        )
    return base.model_copy(update=updates)


def make_structured_runnable(value: pydantic.BaseModel) -> Runnable:
    """Return a ``Runnable`` whose ``ainvoke`` always yields *value*.

    Used by ``RoleModelRegistry.get_structured`` stubs in tests that exercise
    the refactored NER node (which must call
    ``model.with_structured_output(NEROutput)`` exactly once).
    """

    async def _coro(_inp: Any) -> pydantic.BaseModel:
        return value

    return RunnableLambda(_coro)


class StructuredOutputSpy:
    """Tracks ``with_structured_output`` calls on a fake chat model.

    Behaves enough like ``ChatLiteLLM`` for ``model.with_structured_output(cls)``
    to be exercised. Records each call's output-model class in ``.calls`` and
    returns a runnable that emits a pre-configured pydantic value.
    """

    def __init__(self, return_value: pydantic.BaseModel) -> None:
        self.return_value = return_value
        self.calls: List[Type[pydantic.BaseModel]] = []

    def with_structured_output(
        self,
        output_cls: Type[pydantic.BaseModel],
        *,
        include_raw: bool = False,
    ) -> Runnable:
        self.calls.append(output_cls)
        if include_raw:

            async def _wrapped(_inp: Any) -> Dict[str, Any]:
                return {"parsed": self.return_value, "raw": AIMessage(content="")}

            return RunnableLambda(_wrapped)
        return make_structured_runnable(self.return_value)


def make_fake_react_agent(
    messages: Sequence[BaseMessage],
) -> Runnable:
    """Return an agent-like Runnable whose ``ainvoke`` produces *messages*.

    Mirrors the shape returned by ``langchain.agents.create_agent`` /
    ``langgraph.prebuilt.create_react_agent``: a final state dict with a
    ``messages`` key. The graph nodes filter that trace for ``ToolMessage``
    entries (see ``kg_search._parse_triple_tool_payloads`` for the
    pattern tests verify).

    Exposes both ``ainvoke`` and ``astream(stream_mode="values")``. The graph
    nodes stream the agent so they can salvage partial messages when the
    ``recursion_limit`` is hit; the fake emits the full state as a single
    values-mode chunk (the shape production reads from ``last_state``).
    """

    async def _coro(
        _inp: Any, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {"messages": list(messages)}

    async def _astream(
        _inp: Any,
        config: Optional[Dict[str, Any]] = None,
        *,
        stream_mode: Optional[str] = None,
        **_kwargs: Any,
    ):
        yield {"messages": list(messages)}

    runnable = RunnableLambda(_coro)
    # Expose .ainvoke / .astream directly so spies can assert on the
    # (input, config=...) call shape and graphs can stream values-mode chunks.
    runnable.ainvoke = _coro # type: ignore[assignment]
    runnable.astream = _astream # type: ignore[assignment]
    return runnable


def tool_message(name: str, content: Any, tool_call_id: str = "call-0") -> ToolMessage:
    """Construct a ``ToolMessage`` matching the format the graph parsers expect."""
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)


class ToolSpy:
    """Wrap a LangChain ``StructuredTool`` to record ``.ainvoke`` invocations.

    Needed because ``StructuredTool`` is a Pydantic v2 model and rejects
    attribute reassignment, so ``monkeypatch.setattr(tool, "ainvoke", ...)``
    cannot work directly. Replace the *module-level binding* of the tool with
    a ``ToolSpy`` instance instead.
    """

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.invocations: List[Any] = []

    @property
    def name(self) -> str:
        return getattr(self._wrapped, "name", "tool")

    @property
    def description(self) -> str:
        return getattr(self._wrapped, "description", "")

    @property
    def args_schema(self):
        return getattr(self._wrapped, "args_schema", None)

    async def ainvoke(self, inp: Any, *args: Any, **kwargs: Any) -> Any:
        self.invocations.append(inp)
        return await self._wrapped.ainvoke(inp, *args, **kwargs)

    def invoke(self, inp: Any, *args: Any, **kwargs: Any) -> Any:
        self.invocations.append(inp)
        return self._wrapped.invoke(inp, *args, **kwargs)


def build_registry_with_models(
    role_to_model: Dict[str, Any],
) -> "RoleModelRegistry":
    """Build a ``RoleModelRegistry`` and monkey-replace its model accessors.

    Each value in *role_to_model* must be a chat-model-like object exposing
    ``with_structured_output`` and/or behaving as an agent backend. Roles not
    listed fall back to a ``MagicMock``.
    """
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    cfg = LangGraphCoeConfig.from_yaml().llm
    registry = RoleModelRegistry(cfg)
    default = MagicMock()

    def _get_model(role_name: str) -> Any:
        return role_to_model.get(role_name, default)

    def _get_structured(role: Any) -> Runnable:
        model = _get_model(role.name)
        if hasattr(model, "with_structured_output"):
            return model.with_structured_output(role.output_model)
        return make_structured_runnable(MagicMock())

    registry.get_model = _get_model # type: ignore[assignment]
    registry.get_structured = _get_structured # type: ignore[assignment]
    return registry
