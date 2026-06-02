from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated, TypedDict

from ..config import WebSearchConfig
from ..llm import RoleModelRegistry
from ..roles import WEB_RESEARCHER
from ..tools.web import web_search

logger = logging.getLogger(__name__)

_cv_visited_urls: ContextVar[set[str] | None] = ContextVar("web_graph_visited_urls", default=None)
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
        agent = create_react_agent(
            model=model,
            tools=[web_search],
            prompt=WEB_RESEARCHER.system_prompt,
            name="web_research_agent",
        )
        errs = list(state.get("errors") or [])
        try:
            out = await agent.ainvoke(
                {"messages": [HumanMessage(content=state.get("subquery", ""))]},
                config={"recursion_limit": 35},
            )
            messages = list(out.get("messages", []))
            return {"messages": messages, "errors": errs}
        except Exception as exc:
            logger.exception("Web research agent failed")
            errs.append(f"web_research_agent: {exc}")
            return {"messages": [], "errors": errs}

    async def finalize_node(state: WebResearchState) -> Dict[str, Any]:
        tool_calls = _parse_web_tool_payloads(state.get("messages") or [])
        errs = list(state.get("errors") or [])
        max_queries = max(0, int(getattr(_web_research_config, "max_queries_per_agent", 5)))

        queries_issued = 0
        kept_rows: List[Dict[str, Any]] = []
        for rows in tool_calls:
            if queries_issued >= max_queries:
                errs.append("web_search query budget limit reached")
                break
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

