"""KG_search subgraph: NER + entity linking → triple-search agent → enrichment.

The NER step is a one-shot ``model.with_structured_output(NEROutput)`` call
followed by a direct ``link_entities`` tool call (no agent loop). Only the
triple-search step is a tool-calling agent (``langchain.agents.create_agent``,
built on LangGraph); see LangChain Agents documentation.

Flow: ``START`` → ``ner_agent`` → ``triple_search_agent`` → ``enrich`` → ``END``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, cast

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..llm import RoleModelRegistry
from ..roles import KG_TRIPLE_AGENT_SYSTEM, NEROutput
from ..tools.wikidata import (
    create_fetch_and_prune_tool,
    enrich_entities,
    link_entities,
    # Imported only so tests can patch it on this module and assert the NER
    # node never calls it (session reset lives in system.py, not here).
    reset_wikidata_session,  # noqa: F401
)
from ._reasoning_middleware import strip_reasoning_middleware

logger = logging.getLogger(__name__)


class KGSearchState(TypedDict, total=False):
    """State for the KG search pipeline (subgraph input/output)."""

    subquery: str
    original_query: str
    context: str
    known_entities_summary: str
    linked_entities: List[Dict[str, str]]
    qids_for_triples: List[str]
    triples: List[str]
    kg_articles: List[str]
    enriched_entity_labels: List[str]
    errors: List[str]


def _build_ner_user_message(state: KGSearchState) -> str:
    parts = [
        f"subquery:\n{state.get('subquery', '')}",
    ]
    if state.get("context"):
        parts.append(f"extra_context:\n{state['context']}")
    if state.get("known_entities_summary"):
        parts.append(f"known_entities_from_memory:\n{state['known_entities_summary']}")
    return "\n\n".join(parts)


def _extract_entity_names(parsed: NEROutput) -> List[str]:
    names: List[str] = []
    for item in parsed.entities:
        if isinstance(item, str):
            name = item.strip()
        else:
            name = str(getattr(item, "name", "")).strip()
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def _build_triple_user_message(state: KGSearchState) -> str:
    qids = state.get("qids_for_triples") or []
    return "\n\n".join(
        [
            f"subquery:\n{state.get('subquery', '')}",
            f"context:\n{state.get('context', '')}",
            f"seed_qids:\n{', '.join(qids)}",
        ]
    )


def _parse_triple_tool_payloads(messages: Sequence[BaseMessage]) -> List[str]:
    """Stringify triple results from fetch_and_prune_subgraph tool messages."""
    lines: List[str] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tname = getattr(m, "name", None)
        if tname is not None and tname != "fetch_and_prune_subgraph":
            continue
        c = m.content
        if isinstance(c, list):
            for item in c:
                lines.append(str(item))
        elif isinstance(c, str):
            lines.append(c)
    # Dedupe while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        uniq.append(line)
    return uniq


def build_kg_search_graph(registry: RoleModelRegistry):
    """Compile the KG_search subgraph. *registry* supplies models and TRIPLE_PRUNER Stage B."""

    fetch_tool = create_fetch_and_prune_tool(registry)

    async def ner_agent_node(state: KGSearchState) -> Dict[str, Any]:
        user = _build_ner_user_message(state)
        errs = list(state.get("errors") or [])
        try:
            model = registry.get_model("kg_ner_agent")
            chain = model.with_structured_output(NEROutput)
            parsed = await chain.ainvoke(
                [HumanMessage(content=user)],
            )
        except Exception as exc:
            logger.exception("KG NER extraction failed")
            errs.append(f"ner_extraction: {exc}")
            return {"linked_entities": [], "qids_for_triples": [], "errors": errs}

        names = _extract_entity_names(parsed)
        if not names:
            return {"linked_entities": [], "qids_for_triples": [], "errors": errs}

        try:
            linked = await link_entities.ainvoke({"entity_names": names})
        except Exception as exc:
            logger.exception("Direct link_entities tool call failed")
            errs.append(f"link_entities: {exc}")
            return {"linked_entities": [], "qids_for_triples": [], "errors": errs}

        linked = [r for r in linked if isinstance(r, dict) and r.get("qid")]
        qids = list(dict.fromkeys(r["qid"] for r in linked if r.get("qid")))
        return {"linked_entities": linked, "qids_for_triples": qids, "errors": errs}

    async def triple_search_node(state: KGSearchState) -> Dict[str, Any]:
        model = registry.get_model("kg_triple_search_agent")
        agent = create_agent(
            model,
            tools=[fetch_tool],
            system_prompt=KG_TRIPLE_AGENT_SYSTEM,
            middleware=[strip_reasoning_middleware],
            name="kg_triple",
        )
        user = _build_triple_user_message(state)
        errs = list(state.get("errors") or [])
        qids = state.get("qids_for_triples") or []
        if not qids:
            return {"triples": [], "errors": errs}

        query_blob = " ".join(
            s
            for s in (
                state.get("original_query", ""),
                state.get("subquery", ""),
                state.get("context", ""),
            )
            if s
        )

        user_full = f"{user}\n\n**combined_query_for_tool** (use as the `query` argument):\n{query_blob.strip()}"

        recursion_limit = 35
        # Stream with ``stream_mode="values"`` so we retain the last full state
        # emitted before the agent loop terminates. Unlike ``ainvoke`` (which is
        # all-or-nothing and discards everything on error), this lets us salvage
        # the tool results gathered so far if the recursion limit is reached.
        last_state: Dict[str, Any] = {}
        try:
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=user_full)]},
                config={"recursion_limit": recursion_limit},
                stream_mode="values",
            ):
                last_state = chunk
        except GraphRecursionError:
            # Graceful degradation: the agent didn't converge within the step
            # budget. Keep the partial messages and feed them downstream rather
            # than crashing. Warn (not exception) since this is recoverable.
            partial = cast(List[BaseMessage], last_state.get("messages", []))
            logger.warning(
                "KG triple-search agent hit recursion_limit=%d; using %d "
                "partial messages gathered before the limit",
                recursion_limit,
                len(partial),
            )
            errs.append(
                f"triple_search_agent: recursion_limit={recursion_limit} reached "
                "(partial triples used)"
            )
        except Exception as exc:
            logger.exception("KG triple-search agent failed")
            errs.append(f"triple_search_agent: {exc}")
            return {"triples": [], "errors": errs}

        msgs = cast(List[BaseMessage], last_state.get("messages", []))
        triples = _parse_triple_tool_payloads(msgs)
        return {"triples": triples, "errors": errs}

    async def enrich_node(state: KGSearchState) -> Dict[str, Any]:
        qids = list(dict.fromkeys(state.get("qids_for_triples") or []))
        errs = list(state.get("errors") or [])
        if not qids:
            return {
                "kg_articles": [],
                "enriched_entity_labels": [],
                "errors": errs,
            }

        try:
            enriched = await enrich_entities.ainvoke({"qids": qids})
        except Exception as exc:
            logger.exception("enrich_entities failed")
            errs.append(f"enrich: {exc}")
            return {"kg_articles": [], "enriched_entity_labels": [], "errors": errs}

        articles: List[str] = []
        labels: List[str] = []
        for e in enriched:
            wc = getattr(e, "wikipedia_content", None)
            if wc:
                articles.append(str(wc))
            lab = getattr(e, "label", None)
            if lab:
                labels.append(str(lab))

        articles = list(dict.fromkeys(articles))
        return {
            "kg_articles": articles,
            "enriched_entity_labels": labels,
            "errors": errs,
        }

    builder = StateGraph(KGSearchState)
    builder.add_node("ner_agent", ner_agent_node)
    builder.add_node("triple_search_agent", triple_search_node)
    builder.add_node("enrich", enrich_node)
    builder.add_edge(START, "ner_agent")
    builder.add_edge("ner_agent", "triple_search_agent")
    builder.add_edge("triple_search_agent", "enrich")
    builder.add_edge("enrich", END)
    return builder.compile()


def run_kg_search_sync(
    registry: RoleModelRegistry,
    *,
    subquery: str,
    original_query: str,
    context: str = "",
    known_entities_summary: str = "",
) -> KGSearchState:
    """Convenience: build graph, invoke once, return final state dict."""
    graph = build_kg_search_graph(registry)
    result = graph.invoke(
        {
            "subquery": subquery,
            "original_query": original_query,
            "context": context,
            "known_entities_summary": known_entities_summary,
            "errors": [],
        }
    )
    return cast(KGSearchState, result)


async def run_kg_search_async(
    registry: RoleModelRegistry,
    *,
    subquery: str,
    original_query: str,
    context: str = "",
    known_entities_summary: str = "",
) -> KGSearchState:
    graph = build_kg_search_graph(registry)
    result = await graph.ainvoke(
        {
            "subquery": subquery,
            "original_query": original_query,
            "context": context,
            "known_entities_summary": known_entities_summary,
            "errors": [],
        }
    )
    return cast(KGSearchState, result)
