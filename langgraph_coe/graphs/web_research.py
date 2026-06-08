from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from typing_extensions import Annotated, TypedDict

from ._reasoning_middleware import strip_reasoning_middleware
from ..config import WebSearchConfig
from ..llm import RoleModelRegistry
from ..roles import WEB_RESEARCHER
from ..tools.web import web_search

logger = logging.getLogger(__name__)

_cv_visited_urls: ContextVar[set[str] | None] = ContextVar(
    "web_graph_visited_urls", default=None
)
_web_research_config: WebSearchConfig = WebSearchConfig()


class WebResearchState(TypedDict, total=False):
    subquery: str
    original_query: str
    context: str
    messages: Annotated[List[BaseMessage], add_messages]
    queries_issued: int
    results: List[Dict[str, Any]]
    errors: List[str]


def set_web_research_config(cfg: WebSearchConfig) -> None:
    global _web_research_config
    _web_research_config = cfg


def _parse_web_tool_payloads(messages: List[BaseMessage]) -> List[List[Dict[str, Any]]]:
    calls: List[List[Dict[str, Any]]] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tname = getattr(m, "name", None)
        if tname is not None and tname != "web_search":
            continue
        content = m.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = None
        rows: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    rows.append(item)
        elif isinstance(content, dict):
            rows.append(content)
        calls.append(rows)
    return calls


def build_web_research_graph(registry: RoleModelRegistry):
    async def web_research_agent_node(state: WebResearchState) -> Dict[str, Any]:
        model = registry.get_model("web_researcher")
        agent = create_agent(
            model,
            tools=[web_search],
            system_prompt=WEB_RESEARCHER.system_prompt,
            middleware=[strip_reasoning_middleware],
            name="web_research_agent",
        )
        errs = list(state.get("errors") or [])
        recursion_limit = 35
        # Stream with ``stream_mode="values"`` so we retain the last full state
        # emitted before the agent loop terminates. Unlike ``ainvoke`` (which is
        # all-or-nothing and discards everything on error), this lets us salvage
        # the web-search results gathered so far if the recursion limit is reached.
        last_state: Dict[str, Any] = {}
        try:
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=state.get("subquery", ""))]},
                config={"recursion_limit": recursion_limit},
                stream_mode="values",
            ):
                last_state = chunk
        except GraphRecursionError:
            # Graceful degradation: the agent didn't converge within the step
            # budget. Keep the partial messages and feed them downstream rather
            # than crashing. Warn (not exception) since this is recoverable.
            partial = list(last_state.get("messages", []))
            logger.warning(
                "Web research agent hit recursion_limit=%d; using %d partial "
                "messages gathered before the limit",
                recursion_limit,
                len(partial),
            )
            errs.append(
                f"web_research_agent: recursion_limit={recursion_limit} reached "
                "(partial results used)"
            )
        except Exception as exc:
            logger.exception("Web research agent failed")
            errs.append(f"web_research_agent: {exc}")
            return {"messages": [], "errors": errs}

        messages = list(last_state.get("messages", []))
        return {"messages": messages, "errors": errs}

    async def finalize_node(state: WebResearchState) -> Dict[str, Any]:
        tool_calls = _parse_web_tool_payloads(state.get("messages") or [])
        errs = list(state.get("errors") or [])
        max_queries = max(
            0, int(getattr(_web_research_config, "max_queries_per_agent", 5))
        )

        queries_issued = 0
        kept_rows: List[Dict[str, Any]] = []
        for rows in tool_calls:
            if queries_issued >= max_queries:
                errs.append("web_search query budget limit reached")
            queries_issued += 1
            kept_rows.extend(rows)

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in kept_rows:
            url = str(row.get("url", "")).strip()
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            deduped.append(
                {
                    "title": str(row.get("title", "")),
                    "url": url,
                    "snippet": str(row.get("snippet", "")),
                    "full_text": str(row.get("full_text", "")),
                }
            )

        top_k = max(1, int(getattr(_web_research_config, "top_k", 5)))
        return {
            "queries_issued": min(queries_issued, max_queries),
            "results": deduped[:top_k],
            "errors": errs,
        }

    builder = StateGraph(WebResearchState)
    builder.add_node("web_research_agent", web_research_agent_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "web_research_agent")
    builder.add_edge("web_research_agent", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
