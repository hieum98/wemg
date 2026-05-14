"""Wikidata tools for the LangGraph CoE agent.

Three @tool functions are exposed:
  - link_entities:          text -> resolved (QID, label) pairs
  - fetch_and_prune_subgraph: QIDs + query -> pruned WikiTriple strings
  - enrich_entities:        QIDs -> full entity context (desc + Wikipedia)

Loop-prevention is layered:
  1. Tool-level visited-QID tracking  (ContextVar, isolated per async Task)
  2. State-level hop counter           (ContextVar, isolated per async Task)
  3. Hard LangGraph recursion_limit    (set on agent.invoke config)

Thread/async safety:
  - _wikidata_client and _wikidata_config are read-only after init_wikidata().
  - Per-question mutable state (_visited_qids, _entity_link_cache,
    _total_subgraph_hops) uses ContextVar so each asyncio.Task (i.e. each
    question in a .batch() run) gets its own isolated copy. No cross-question
    contamination possible, even with concurrent async execution.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Set

import httpx
from langchain_core.tools import tool

from .wikidata_client import (
    WikidataClient,
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    DEFAULT_PROPERTIES,
    PROPERTY_LABELS,
)
from ..config import WikidataConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Read-only singletons (set once at startup via init_wikidata)
# ──────────────────────────────────────────────────────────────────────────────

_wikidata_client: Optional[WikidataClient] = None
_wikidata_config: Optional[WikidataConfig] = None

# ──────────────────────────────────────────────────────────────────────────────
# Per-question mutable state.
#
# LangChain's BaseTool.ainvoke runs the tool coroutine in a child Task with its
# own copy of the parent's context (via ``asyncio.create_task(coro, context=...)``).
# Plain ContextVar assignments inside the tool do NOT propagate back to the
# parent, which breaks state that has to accumulate across sequential tool
# calls (e.g. the hop-budget counter).
#
# The fix is to bind a single mutable ``_SessionState`` object via ContextVar.
# Child tasks INHERIT the same object reference, so mutations to its fields
# propagate via shared identity. Concurrent agent runs (e.g.
# ``asyncio.gather`` of distinct questions) each call ``reset_wikidata_session()``
# inside their own task, which rebinds the ContextVar to a fresh object — that
# rebinding stays local to the rebinding task, so isolation is preserved.
# ──────────────────────────────────────────────────────────────────────────────


class _SessionState:
    """Per-question mutable state shared via ContextVar object identity."""

    __slots__ = ("visited", "hop_count")

    def __init__(self) -> None:
        self.visited: Set[str] = set()
        self.hop_count: int = 0


_cv_session: ContextVar[Optional[_SessionState]] = ContextVar(
    "wikidata_session", default=None
)

# Name -> QID cache persistent during the whole run (across questions).
entity_cache: Dict[str, str] = {}

_PRUNE_BATCH_SIZE = 16


def _get_session() -> _SessionState:
    s = _cv_session.get(None)
    if s is None:
        s = _SessionState()
        _cv_session.set(s)
    return s


def _get_visited() -> Set[str]:
    return _get_session().visited


def init_wikidata(config: WikidataConfig) -> None:
    """Initialise the Wikidata client once at startup."""
    global _wikidata_client, _wikidata_config
    _wikidata_config = config
    _wikidata_client = WikidataClient(
        max_sparql_rps=config.max_sparql_rps,
        max_wikipedia_rps=config.max_wikipedia_rps,
        lru_capacity=config.triple_cache_max_entries,
    )
    logger.info("WikidataClient initialised (max_hops=%d)", config.max_hops)


def reset_wikidata_session() -> None:
    """Reset per-question state for the current async Task.

    Call at the start of every new question (e.g. at the top of ``kb_node``)
    and inside any newly-spawned ``asyncio.gather`` child whose state must be
    isolated from siblings. Rebinds the ContextVar to a fresh ``_SessionState``
    object; sibling tasks keep their own object.
    """
    _cv_session.set(_SessionState())


# ──────────────────────────────────────────────────────────────────────────────
# Stage A pruning: fast reranker-based score filter
# ──────────────────────────────────────────────────────────────────────────────

async def _stage_a_prune(
    question: str,
    triples: List[WikiTriple],
    reranker_url: Optional[str],
    reranker_model: Optional[str],
    top_k: int = 64,
    delta: float = 0.05,
) -> List[WikiTriple]:
    """Score each triple against the query via the reranker API and keep the top tier."""
    if not triples or not reranker_url:
        return triples

    texts = [str(t) for t in triples]
    payload = {
        "model": reranker_model or "reranker",
        "query": question,
        "texts": texts,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{reranker_url.rstrip('/')}/rerank",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        if not results:
            return triples

        scored = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        top_score = scored[0].get("score", 0)
        kept_indices = [
            r["index"]
            for r in scored
            if r.get("score", 0) >= (top_score - delta)
        ][:top_k]
        return [triples[i] for i in kept_indices]

    except Exception as exc:
        logger.warning("Stage A reranker failed (%s); returning all triples.", exc)
        return triples


# ──────────────────────────────────────────────────────────────────────────────
# Stage B pruning: LLM TRIPLE_PRUNER role
# ──────────────────────────────────────────────────────────────────────────────

async def _stage_b_prune(
    question: str,
    triples: List[WikiTriple],
    registry: Any,
) -> List[WikiTriple]:
    """Use the TRIPLE_PRUNER role to LLM-select only causally-relevant triples."""
    if not triples or registry is None:
        return triples

    from ..roles import TriplePruneInput, TRIPLE_PRUNER
    from ..llm import execute_role_lc

    chunks = [
        triples[i : i + _PRUNE_BATCH_SIZE]
        for i in range(0, len(triples), _PRUNE_BATCH_SIZE)
    ]
    kept: List[WikiTriple] = []
    for chunk in chunks:
        inp = TriplePruneInput(question=question, triples=[str(t) for t in chunk])
        try:
            out, _ = await execute_role_lc(registry, TRIPLE_PRUNER, inp)
            if out and hasattr(out, "keep_indices"):
                keep_set = set(out.keep_indices)
                kept.extend(chunk[i] for i in range(len(chunk)) if i in keep_set)
            else:
                kept.extend(chunk)
        except Exception as exc:
            logger.warning("Stage B pruning failed for chunk (%s); keeping all.", exc)
            kept.extend(chunk)

    return list(set(kept))  # deduplicate


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1 – link_entities
# ──────────────────────────────────────────────────────────────────────────────

@tool
async def link_entities(entity_names: List[str]) -> List[Dict[str, str]]:
    """Link entities to Wikidata QIDs.

    Args:
        entity_names: List of entity names to link.

    Returns:
        List of dicts with keys 'name', 'qid', 'description'.
    """
    if _wikidata_client is None:
        raise RuntimeError("Wikidata not initialised. Call init_wikidata first.")
    global entity_cache

    to_resolve = [n for n in entity_names if n not in entity_cache]
    output: List[Dict[str, str]] = []

    if to_resolve:
        try:
            results = await _wikidata_client.link_entities(to_resolve, top_k=1)
            # link_entities for list input returns list[list[WikidataEntity]]
            if results and isinstance(results, list) and isinstance(results[0], list):
                for name, candidates in zip(to_resolve, results):
                    if candidates:
                        entity_cache[name] = candidates[0].qid
        except Exception as exc:
            logger.warning("Entity linking failed: %s", exc)

    for name in entity_names:
        qid = entity_cache.get(name)
        if qid:
            output.append({"name": name, "qid": qid, "description": ""})

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2 – fetch_and_prune_subgraph
# ──────────────────────────────────────────────────────────────────────────────


async def _fetch_and_prune_subgraph_core(
    qids: List[str],
    query: str,
    *,
    registry: Any = None,
) -> List[Any]:
    """Core implementation. *registry* enables Stage-B LLM pruning via TRIPLE_PRUNER."""
    if _wikidata_client is None or _wikidata_config is None:
        raise RuntimeError("Wikidata not initialised. Call init_wikidata first.")

    session = _get_session()
    visited = session.visited
    hop_count = session.hop_count

    if hop_count >= _wikidata_config.max_hops:
        return [
            f"[Wikidata hop budget exhausted after {hop_count} hops. "
            "Use existing context to formulate an answer.]"
        ]

    new_qids = [q for q in qids if q not in visited]
    already_done = [q for q in qids if q in visited]

    if already_done and not new_qids:
        return [
            f"[Already explored: {', '.join(already_done)}. "
            "No new QIDs to fetch. Please try different entities.]"
        ]

    if not new_qids:
        return ["[No valid QIDs provided.]"]

    visited.update(new_qids)
    session.hop_count = hop_count + 1

    try:
        raw_results = await _wikidata_client.get_k_hop_triples(
            new_qids, k=1, bidirectional=False, enrich=True
        )
    except Exception as exc:
        logger.error("SPARQL fetch failed for %s: %s", new_qids, exc)
        return [f"[SPARQL fetch failed: {exc}]"]

    triples: List[WikiTriple] = []
    if raw_results and isinstance(raw_results[0], list):
        for per_seed in raw_results:
            triples.extend(per_seed)
    else:
        triples = list(raw_results) if raw_results else []

    seen: Set[int] = set()
    unique_triples: List[WikiTriple] = []
    for t in triples:
        h = hash(t)
        if h not in seen:
            seen.add(h)
            unique_triples.append(t)
    triples = unique_triples

    if not triples:
        return [f"[No triples found for QIDs: {new_qids}]"]

    reranker_url = getattr(_wikidata_config, "reranker_url", None)
    reranker_model = getattr(_wikidata_config, "reranker_model", None)
    triples = await _stage_a_prune(
        query, triples, reranker_url, reranker_model,
        top_k=_wikidata_config.pruning_top_k,
        delta=_wikidata_config.pruning_delta,
    )

    if registry is not None:
        triples = await _stage_b_prune(query, triples, registry)

    return triples


@tool
async def fetch_and_prune_subgraph(
    qids: List[str],
    query: str,
) -> List[Any]:
    """Fetch the 1-hop Wikidata subgraph for *qids*, prune it (Stage A + optional Stage B if configured upstream).

    For Stage B LLM pruning inside agents, use ``create_fetch_and_prune_tool(registry)`` which closes over a
    ``RoleModelRegistry``.
    """
    return await _fetch_and_prune_subgraph_core(qids, query, registry=None)


def create_fetch_and_prune_tool(registry: Any):
    """Return a LangChain tool that runs fetch/prune with Stage-B pruning via *registry* (TRIPLE_PRUNER role)."""

    @tool("fetch_and_prune_subgraph")
    async def fetch_and_prune_subgraph(qids: List[str], query: str) -> List[Any]:
        """Fetch the 1-hop Wikidata subgraph for QIDs, rerank, then LLM-prune triples relevant to the query."""
        return await _fetch_and_prune_subgraph_core(qids, query, registry=registry)

    return fetch_and_prune_subgraph


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3 – enrich_entities
# ──────────────────────────────────────────────────────────────────────────────

@tool
async def enrich_entities(qids: List[str]) -> List[WikidataEntity]:
    """Fetch full entity details (label, description, aliases, Wikipedia) for *qids*.

    Returns one entity object per QID.
    """
    if _wikidata_client is None:
        raise RuntimeError("Wikidata not initialised. Call init_wikidata first.")

    valid_qids = [q for q in qids if q]
    if not valid_qids:
        return []

    try:
        enriched = await _wikidata_client.enrich_entities(
            valid_qids, get_details=True
        )
    except Exception as exc:
        logger.error("Entity enrichment failed for %s: %s", qids, exc)
        return []

    return enriched or []
