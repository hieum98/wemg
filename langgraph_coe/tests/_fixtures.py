"""Helpers shared by langgraph_coe Phase 0+ test modules.

These tests are **target specs** ([feedback_tests_as_target_spec]): they describe the
goal the Phase 0 implementation must reach, not its current state. Some helpers
reference symbols (e.g. ``WEB_RESEARCHER``, ``RedisDictCache``,
``reset_web_research_session``) that intentionally do not exist yet — the
helpers import them lazily inside fixtures/tests so collection still succeeds
on a tree where Phase 0 is unimplemented.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Type
from unittest.mock import AsyncMock, MagicMock

import pydantic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda


# Stable test URLs / strings reused across §3.5 tests.
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_C = "https://example.com/c"


def make_structured_runnable(value: pydantic.BaseModel) -> Runnable:
    """Return a ``Runnable`` whose ``ainvoke`` always yields *value*.

    Used by ``RoleModelRegistry.get_structured`` stubs in tests that exercise
    the §3.2 refactored NER node (which must call
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
    entries (see ``kg_search._parse_link_entities_tool_payloads`` for the
    pattern Phase 0 tests verify).
    """

    async def _coro(_inp: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"messages": list(messages)}

    runnable = RunnableLambda(_coro)
    # Expose .ainvoke directly so spies can assert on (input, config=...) call shape.
    runnable.ainvoke = _coro  # type: ignore[assignment]
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

    registry.get_model = _get_model  # type: ignore[assignment]
    registry.get_structured = _get_structured  # type: ignore[assignment]
    return registry
