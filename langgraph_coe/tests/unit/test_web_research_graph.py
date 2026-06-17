"""WebResearchGraph behavior.

The subgraph lives at ``langgraph_coe.graphs.web_research``. It is a ReAct
agent wired to ``web_search`` with three-layer loop prevention, parity with
``kg_search``:

  1. LangGraph ``recursion_limit``.
  2. In-state ``queries_issued`` counter against ``config.web_search.max_queries_per_agent``.
  3. ContextVar ``_cv_visited_urls`` populated by the ``web_search`` tool.

State:

    class WebResearchState(TypedDict):
        subquery: str
        original_query: str
        context: str
        messages: Annotated[List[BaseMessage], add_messages]
        queries_issued: int
        results: List[Dict[str, Any]] # {title, url, snippet, full_text}
        errors: List[str]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from .._fixtures import URL_A, URL_B, URL_C, make_fake_react_agent, tool_message


def _import_module():
    """Introduces ``langgraph_coe.graphs.web_research``. Skip cleanly if absent."""
    try:
        from langgraph_coe.graphs import web_research as wr_mod # type: ignore
    except ImportError:
        pytest.skip("Introduces langgraph_coe.graphs.web_research")
    if not hasattr(wr_mod, "build_web_research_graph"):
        pytest.skip("build_web_research_graph unavailable")
    return wr_mod


def _registry_with_light_model(model: Any):
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig.from_yaml().llm)
    registry.get_model = lambda role_name: model # type: ignore[assignment]
    return registry


def _result_payload(
    url: str, *, title: str = "T", snippet: str = "S", full_text: str = ""
):
    return {"title": title, "url": url, "snippet": snippet, "full_text": full_text}


# ──────────────────────────────────────────────────────────────────────────────
# — graph wiring + state shape
# ──────────────────────────────────────────────────────────────────────────────


async def test_graph_compiles_with_expected_state_keys(monkeypatch):
    """``build_web_research_graph(registry)`` produces a compiled graph whose
    output state surfaces the keys."""
    wr_mod = _import_module()
    model = MagicMock()
    registry = _registry_with_light_model(model)

    # Stub the agent factory to avoid actually invoking an LLM.
    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                tool_message("web_search", [_result_payload(URL_A)]),
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "What is the capital of France?",
            "original_query": "What is the capital of France?",
            "context": "",
        }
    )

    for key in ("results", "queries_issued", "errors"):
        assert key in final, f"WebResearchState must surface '{key}'"
    assert isinstance(final["results"], list)


async def test_agent_uses_light_tier_and_web_search_tool(monkeypatch):
    """The ReAct agent is wired to ``model=registry.get_model('web_researcher')``,
    ``tools=[web_search]``, and the ``WEB_RESEARCHER.system_prompt``."""
    wr_mod = _import_module()
    captured: Dict[str, Any] = {}

    def _spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return make_fake_react_agent([AIMessage(content="ok")])

    monkeypatch.setattr(wr_mod, "create_agent", _spy, raising=False)

    # Wire registry so get_model("web_researcher") returns a sentinel.
    light_sentinel = MagicMock(name="LIGHT_MODEL")
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig.from_yaml().llm)
    registry.get_model = lambda role_name: light_sentinel # type: ignore[assignment]

    graph = wr_mod.build_web_research_graph(registry)
    await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    # Bound model is the light-tier sentinel.
    bound_model = captured["kwargs"].get("model") or (
        captured["args"][0] if captured["args"] else None
    )
    assert bound_model is light_sentinel, (
        "WebResearchGraph must bind the model from registry.get_model('web_researcher')"
    )

    # Tools include web_search.
    tools = captured["kwargs"].get("tools") or []
    tool_names = [getattr(t, "name", str(t)) for t in tools]
    assert "web_search" in tool_names, (
        f"WebResearchGraph must bind the web_search tool; saw {tool_names}"
    )

    # Prompt is the WEB_RESEARCHER role's system_prompt.
    from langgraph_coe import roles as roles_mod

    web_researcher = getattr(roles_mod, "WEB_RESEARCHER", None)
    if web_researcher is not None:
        prompt = captured["kwargs"].get("prompt") or captured["kwargs"].get(
            "system_prompt"
        )
        assert prompt == web_researcher.system_prompt, (
            "ReAct agent must be prompted with WEB_RESEARCHER.system_prompt"
        )


# ──────────────────────────────────────────────────────────────────────────────
# — finalize: dedup by URL, top_k cap
# ──────────────────────────────────────────────────────────────────────────────


async def test_finalize_dedupes_by_url(monkeypatch):
    """Two ``web_search`` ToolMessages sharing a URL collapse to one result."""
    wr_mod = _import_module()
    registry = _registry_with_light_model(MagicMock())

    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                tool_message(
                    "web_search", [_result_payload(URL_A), _result_payload(URL_B)]
                ),
                tool_message("web_search", [_result_payload(URL_A, title="dup")]),
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    urls = [r.get("url") for r in (final.get("results") or [])]
    assert urls.count(URL_A) == 1, (
        f"finalize must dedupe by url; saw URL_A {urls.count(URL_A)}× in {urls!r}"
    )
    assert URL_B in urls


async def test_finalize_parses_json_string_tool_messages(monkeypatch):
    """``create_agent`` may serialize ``web_search`` payloads as JSON strings."""
    wr_mod = _import_module()
    registry = _registry_with_light_model(MagicMock())

    payload = json.dumps([_result_payload(URL_A), _result_payload(URL_B)])
    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                ToolMessage(content=payload, name="web_search", tool_call_id="call-0"),
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    urls = [r.get("url") for r in (final.get("results") or [])]
    assert URL_A in urls and URL_B in urls, (
        f"expected parsed JSON tool payload; got {urls!r}"
    )


async def test_finalize_returns_top_n_from_config(monkeypatch, config):
    """When the agent yields more results than ``config.web_search.top_k``,
    ``finalize`` trims to that ceiling."""
    wr_mod = _import_module()
    registry = _registry_with_light_model(MagicMock())

    config.web_search.top_k = 5
    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                tool_message(
                    "web_search",
                    [_result_payload(f"https://example.com/{i}") for i in range(10)],
                ),
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    if hasattr(wr_mod, "set_web_research_config"):
        wr_mod.set_web_research_config(config.web_search) # type: ignore[attr-defined]

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    assert len(final.get("results") or []) == 5, (
        f"expected top-5 results, got {len(final.get('results') or [])}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# — three-layer loop prevention
# ──────────────────────────────────────────────────────────────────────────────


async def test_agent_invoked_with_recursion_limit(monkeypatch):
    """Layer 1: ``agent.astream(config={'recursion_limit': N}, stream_mode='values')``.

    The graph streams (rather than ``ainvoke``-ing) so it can salvage partial
    messages on ``GraphRecursionError``; the recursion_limit must still be
    passed through to the agent.
    """
    wr_mod = _import_module()
    registry = _registry_with_light_model(MagicMock())

    captured_config: Dict[str, Any] = {}

    class _Agent:
        async def astream(self, _inp, config=None, *, stream_mode=None, **_kw):
            captured_config["config"] = config or {}
            captured_config["stream_mode"] = stream_mode
            yield {"messages": []}

    monkeypatch.setattr(
        wr_mod, "create_agent", lambda *a, **kw: _Agent(), raising=False
    )

    graph = wr_mod.build_web_research_graph(registry)
    await graph.ainvoke({"subquery": "x", "original_query": "x", "context": ""})

    assert "recursion_limit" in captured_config["config"], (
        "WebResearchGraph must pass recursion_limit to agent.astream (layer 1)"
    )
    assert captured_config["config"]["recursion_limit"] > 0


async def test_queries_issued_counter_caps_agent(monkeypatch, config):
    """Layer 2: when ``queries_issued >= max_queries_per_agent`` further
    ``web_search`` calls are short-circuited and an error is recorded."""
    wr_mod = _import_module()
    if not hasattr(config.web_search, "max_queries_per_agent"):
        pytest.skip("max_queries_per_agent is a config addition")

    config.web_search.max_queries_per_agent = 2
    registry = _registry_with_light_model(MagicMock())

    # Simulate 3 search calls in a single agent run.
    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                tool_message("web_search", [_result_payload(URL_A)]),
                tool_message("web_search", [_result_payload(URL_B)]),
                tool_message("web_search", [_result_payload(URL_C)]), # over budget
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    # Final state shows the cap was enforced.
    assert (final.get("queries_issued") or 0) <= 2, (
        f"queries_issued must be capped at max_queries_per_agent; "
        f"saw {final.get('queries_issued')}"
    )
    errors = " ".join(final.get("errors") or [])
    assert "budget" in errors.lower() or "limit" in errors.lower(), (
        f"over-budget call must record a budget/limit error; saw errors={final.get('errors')!r}"
    )


async def test_visited_urls_contextvar_dedupes_across_tool_calls(monkeypatch):
    """Layer 3: ContextVar ``_cv_visited_urls`` filters duplicates inside one run."""
    wr_mod = _import_module()
    if not hasattr(wr_mod, "_cv_visited_urls") and not hasattr(
        wr_mod, "reset_web_research_session"
    ):
        pytest.skip("ContextVar visited-URL session is a addition")

    registry = _registry_with_light_model(MagicMock())
    # Two search calls returning the same URL.
    monkeypatch.setattr(
        wr_mod,
        "create_agent",
        lambda *a, **kw: make_fake_react_agent(
            [
                tool_message("web_search", [_result_payload(URL_A)]),
                tool_message("web_search", [_result_payload(URL_A, title="dup")]),
                AIMessage(content="done"),
            ]
        ),
        raising=False,
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    urls = [r.get("url") for r in (final.get("results") or [])]
    assert urls.count(URL_A) == 1, (
        f"Layer-3 ContextVar dedup must collapse repeat URL across tool calls; saw {urls!r}"
    )


async def test_reset_web_research_session_clears_visited_urls():
    """``reset_web_research_session`` rebinds the ContextVar to an empty set."""
    from langgraph_coe.tools import web as web_mod

    if not hasattr(web_mod, "reset_web_research_session"):
        pytest.skip("reset_web_research_session is a addition")

    sess = getattr(web_mod, "_get_web_session", None) or getattr(
        web_mod, "_get_session", None
    )
    if sess is None:
        pytest.skip("web module does not expose a session accessor yet")

    s = sess()
    s.visited.add("https://example.com/seed")
    web_mod.reset_web_research_session()
    s2 = sess()
    assert s2.visited == set(), "reset must produce an empty visited-URL set"
    assert s2 is not s, "reset must REBIND to a new session object"


async def test_concurrent_graph_runs_have_isolated_visited_urls(monkeypatch):
    """Two ``graph.ainvoke`` tasks executing in parallel keep distinct visited URLs.

    Mirrors [project_langchain_ainvoke_context_isolation]: the ContextVar must
    hold a *mutable session object* set ONCE at graph entry, not assigned per
    tool invocation.
    """
    wr_mod = _import_module()
    if not hasattr(wr_mod, "reset_web_research_session") and not hasattr(
        wr_mod.__class__, "reset_web_research_session"
    ):
        # The reset lives in web tools module; that's fine.
        from langgraph_coe.tools import web as web_mod

        if not hasattr(web_mod, "reset_web_research_session"):
            pytest.skip("session reset not yet present")

    registry = _registry_with_light_model(MagicMock())

    def _agent_factory(*a, **kw):
        # Each agent run yields a unique URL.
        return make_fake_react_agent(
            [
                tool_message("web_search", [_result_payload(URL_A)]),
                AIMessage(content="done"),
            ]
        )

    monkeypatch.setattr(wr_mod, "create_agent", _agent_factory, raising=False)
    graph = wr_mod.build_web_research_graph(registry)

    async def _run(label: str):
        return await graph.ainvoke(
            {
                "subquery": label,
                "original_query": label,
                "context": "",
            }
        )

    final_a, final_b = await asyncio.gather(_run("Q-A"), _run("Q-B"))
    urls_a = {r.get("url") for r in (final_a.get("results") or [])}
    urls_b = {r.get("url") for r in (final_b.get("results") or [])}
    # Both runs see their own URL; neither was poisoned by the other.
    assert URL_A in urls_a and URL_A in urls_b, (
        "isolated visited-URL state must NOT mask URL_A from a sibling task"
    )


# ──────────────────────────────────────────────────────────────────────────────
# — error containment
# ──────────────────────────────────────────────────────────────────────────────


async def test_provider_failure_recorded_in_errors_not_raised(monkeypatch):
    """A ``RuntimeError`` from the underlying provider lands in ``state['errors']``
    rather than escaping the graph."""
    wr_mod = _import_module()
    registry = _registry_with_light_model(MagicMock())

    class _BrokenAgent:
        async def astream(self, _inp, config=None, *, stream_mode=None, **_kw):
            raise RuntimeError("network down")
            yield # pragma: no cover - makes this an async generator

    monkeypatch.setattr(
        wr_mod, "create_agent", lambda *a, **kw: _BrokenAgent(), raising=False
    )

    graph = wr_mod.build_web_research_graph(registry)
    final = await graph.ainvoke(
        {
            "subquery": "x",
            "original_query": "x",
            "context": "",
        }
    )

    assert (final.get("results") or []) == []
    errors = final.get("errors") or []
    assert any("network down" in str(e) or "RuntimeError" in str(e) for e in errors), (
        f"agent failure must be captured in state['errors']; saw {errors!r}"
    )
