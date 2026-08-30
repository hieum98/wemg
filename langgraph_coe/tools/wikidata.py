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

import logging
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Set

import httpx
from langchain_core.tools import tool

from ..config import WikidataConfig
from .wikidata_client import (
    WikidataClient,
    WikidataEntity,
    WikiTriple,
)

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


def init_wikidata(config: WikidataConfig, *, cache: Any = None) -> None:
    """Initialise the Wikidata client once at startup."""
    global _wikidata_client, _wikidata_config
    _wikidata_config = config
    _wikidata_client = WikidataClient(
        sparql_endpoint=config.sparql_endpoint,
        max_sparql_rps=config.max_sparql_rps,
        max_wikipedia_rps=config.max_wikipedia_rps,
        lru_capacity=config.triple_cache_max_entries,
        cache=cache,
    )
    endpoint = config.sparql_endpoint or "https://query.wikidata.org/sparql"
    logger.info(
        "WikidataClient initialised (max_hops=%d, sparql=%s)",
        config.max_hops,
        endpoint,
    )


def reset_wikidata_session() -> None:
    """Reset per-question state for the current async Task.

    Call at the start of every new question (e.g. at the top of ``kb_node``)
    and inside any newly-spawned ``asyncio.gather`` child whose state must be
    isolated from siblings. Rebinds the ContextVar to a fresh ``_SessionState``
    object; sibling tasks keep their own object.
    """
    _cv_session.set(_SessionState())


# ──────────────────────────────────────────────────────────────────────────────
# Plan focus: the open intents this retrieval is serving
# ──────────────────────────────────────────────────────────────────────────────
#
# Stage B (``triple_pruner``) is 45.3% of every LLM call the system makes and its
# call count is ``ceil(pruning_top_k / 16)`` — a fixed 4, because ``pruning_top_k``
# is a constant 64 guess made without reference to what is being looked for. The
# measured input is 1,780,510 triples reduced to 141,248 (7.9%) by Stage A, so the
# 64 is what Stage B pays for, not the raw fetch.
#
# The plan is the only thing in the system that knows *how many distinct things are
# still being asked for*. A question with one open intent does not need the same
# candidate budget as one with four. Threading that through as a ContextVar rather
# than a parameter because ``query`` reaches the tool from an LLM tool call, so
# there is no call path from the graph to widen — and this module already binds
# per-question state this way for exactly that reason.
_cv_plan_focus: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "wikidata_plan_focus", default=None
)


def set_plan_focus(intents: Optional[List[str]], n_open: int) -> None:
    """Declare the open intents the next retrievals serve. ``None`` disables."""
    if not intents or n_open <= 0:
        _cv_plan_focus.set(None)
        return
    _cv_plan_focus.set({"intents": list(intents), "n_open": int(n_open)})


def clear_plan_focus() -> None:
    _cv_plan_focus.set(None)


def read_plan_focus() -> Optional[Dict[str, Any]]:
    return _cv_plan_focus.get(None)


def _planned_top_k(configured_top_k: int) -> int:
    """Scale the Stage-B candidate budget to the number of intents still open.

    ``_PRUNE_BATCH_SIZE`` per open intent, floored at one batch and capped at the
    configured value, so this can only ever *lower* cost and a question with four or
    more open intents behaves exactly as today. No plan → configured value.
    """
    focus = read_plan_focus()
    if not focus:
        return configured_top_k
    scaled = _PRUNE_BATCH_SIZE * max(1, int(focus.get("n_open") or 1))
    return max(_PRUNE_BATCH_SIZE, min(configured_top_k, scaled))


def _focused_query(query: str) -> str:
    """Append open-intent content words to the Stage-A ranking query.

    Lowering ``top_k`` only preserves recall if the ranking that fills it is
    sharper, and the ranking query is whatever the KG agent happened to pass. The
    open intents name the properties actually wanted, so they belong in the ranking
    signal. Appended, never substituted — the agent's query carries the entity that
    Stage A also has to match.
    """
    focus = read_plan_focus()
    if not focus:
        return query
    extra = " ".join(str(i) for i in (focus.get("intents") or []))
    return f"{query} {extra}".strip() if extra else query


# ──────────────────────────────────────────────────────────────────────────────
# Stage A pruning: fast reranker-based score filter
# ──────────────────────────────────────────────────────────────────────────────


_STOPWORDS = frozenset(
    "a an and are as at be by for from had has have he her his in into is it its of on or "
    "she that the their they this to was were what when where which who whom whose why "
    "with".split()
)


def _content_tokens(text: str) -> Set[str]:
    return {
        t
        for t in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    }


def _lexical_prefilter(
    question: str, triples: List[WikiTriple], top_k: int
) -> List[WikiTriple]:
    """Keep the ``top_k`` triples that share the most content words with the question.

    A deterministic stand-in for the reranker, and the point is only to stop Stage B being
    handed everything: the LLM pruner is what actually judges causal relevance, so this
    just has to avoid discarding the triples it would have kept.

    Ordering matters more than the cap. Arbitrary truncation would drop the answer at
    random; ranking by shared content words is the cheap approximation of what a reranker
    scores, and ties are broken by *fewer* tokens, preferring the more specific triple.
    Triples that share nothing with the question are kept only to fill the quota, never in
    preference to one that overlaps.

    Never silently truncates — a dropped-count is logged, because a cap that reads as
    "we looked at everything" is how a recall loss hides.
    """
    if top_k <= 0 or len(triples) <= top_k:
        return triples
    q = _content_tokens(question)
    if not q:
        logger.info(
            "[stage_a] no content words in the query; keeping the first %d of %d triples",
            top_k,
            len(triples),
        )
        return triples[:top_k]
    scored = sorted(
        ((len(q & _content_tokens(str(t))), -len(_content_tokens(str(t))), i) for i, t in enumerate(triples)),
        reverse=True,
    )
    kept = [triples[i] for _, _, i in scored[:top_k]]
    overlapping = sum(1 for s, _, _ in scored[:top_k] if s > 0)
    logger.info(
        "[stage_a] no reranker: kept %d of %d triples by lexical overlap (%d share a "
        "content word with the query); dropped %d before LLM pruning",
        len(kept),
        len(triples),
        overlapping,
        len(triples) - len(kept),
    )
    return kept


async def _stage_a_prune(
    question: str,
    triples: List[WikiTriple],
    reranker_url: Optional[str],
    reranker_model: Optional[str],
    top_k: int = 64,
    delta: float = 0.05,
    instruction: Optional[str] = None,
) -> List[WikiTriple]:
    """Score each triple against the query via the reranker API and keep the top tier.

    With no reranker configured this falls back to :func:`_lexical_prefilter` rather than
    passing everything through. Returning all triples made ``top_k`` dead in exactly the
    configuration the project runs in (``reranker_url: null``), and Stage B charges one
    LLM call per 16 triples: **measured 97.7 ``triple_pruner`` calls per question, 82% of
    every LLM completion the system makes**, against a configured ``pruning_top_k`` of 64
    that would have allowed 4.
    """
    if not triples:
        return triples
    if not reranker_url:
        return _lexical_prefilter(question, triples, top_k)

    texts = [str(t) for t in triples]
    payload = {
        "model": reranker_model or "reranker",
        "query": question,
        "documents": texts,
    }
    # Task instruction tells Qwen3-Reranker what "relevant" means; symmetric with
    # the context reranks in tools/retrieval.py. Omit when unset so the server
    # falls back to its default behavior.
    if instruction:
        payload["instruct"] = instruction
    try:
        # Generous read budget for large/queued batches (up to ~128 triples on a
        # busy reranker), but keep connect short so a dead endpoint fails fast.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0)
        ) as client:
            resp = await client.post(
                f"{reranker_url.rstrip('/')}/rerank",
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, list):
                results = body
            elif isinstance(body, dict):
                results = body.get("results", [])
            else:
                results = []

        if not results:
            return triples

        scored = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
        top_score = scored[0].get("score", 0)
        kept_indices = [
            r["index"] for r in scored if r.get("score", 0) >= (top_score - delta)
        ][:top_k]
        return [triples[i] for i in kept_indices]

    except Exception as exc:
        logger.warning(
            "Stage A reranker failed (%s: %r); falling back to lexical prefilter.",
            type(exc).__name__,
            exc,
        )
        # Not "return triples": a transient reranker outage would otherwise hand Stage B
        # the unpruned set and multiply that question's LLM cost by an order of magnitude.
        return _lexical_prefilter(question, triples, top_k)


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

    from ..llm import execute_role_lc
    from ..roles import TRIPLE_PRUNER, TriplePruneInput

    chunks = [
        triples[i : i + _PRUNE_BATCH_SIZE]
        for i in range(0, len(triples), _PRUNE_BATCH_SIZE)
    ]
    # One ``execute_role_lc`` over all chunks, not a chunk-at-a-time loop. The loop made
    # the chunks strictly serial, so pruning one question's triples was N sequential model
    # round-trips; ``execute_role_lc`` gathers a list of inputs concurrently, which is the
    # pattern ``memory_update`` already uses for this same role. Same token count, a
    # fraction of the wall-clock.
    inputs = [
        TriplePruneInput(question=question, triples=[str(t) for t in chunk])
        for chunk in chunks
    ]
    kept: List[WikiTriple] = []
    try:
        outs, _ = await execute_role_lc(registry, TRIPLE_PRUNER, inputs)
        if not isinstance(outs, list):
            outs = [outs]
    except Exception as exc:
        logger.warning("Stage B pruning failed (%s); keeping all triples.", exc)
        return list(set(triples))
    for chunk, out in zip(chunks, outs):
        # ``execute_role_lc`` returns one entry per input; for n=1 that entry may itself
        # be a single-element list depending on the caller shape, so unwrap defensively.
        if isinstance(out, list):
            out = out[0] if out else None
        if out is not None and hasattr(out, "keep_indices"):
            keep_set = {i for i in out.keep_indices if 0 <= i < len(chunk)}
            kept.extend(chunk[i] for i in range(len(chunk)) if i in keep_set)
        else:
            kept.extend(chunk)
    # A chunk with no corresponding output must not vanish silently.
    if len(outs) < len(chunks):
        logger.warning(
            "Stage B returned %d outputs for %d chunks; keeping the unpruned remainder",
            len(outs),
            len(chunks),
        )
        for chunk in chunks[len(outs):]:
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
            # str(exc) is empty for many failure types (timeouts, connection
            # errors); log the type + repr + names so the cause is visible.
            logger.warning(
                "Entity linking failed for %s: %s: %r",
                to_resolve[:5],
                type(exc).__name__,
                exc,
            )
            logger.debug("Entity linking traceback", exc_info=True)

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
    reranker_instruction = getattr(_wikidata_config, "reranker_instruction", None)
    # Plan-conditioned budget: a sharper ranking query, and only as many candidates
    # as there are open intents to serve. Both are no-ops without a plan focus.
    planned_top_k = _planned_top_k(_wikidata_config.pruning_top_k)
    if planned_top_k != _wikidata_config.pruning_top_k:
        logger.info(
            "[stage_a] plan focus: %d open intent(s) -> top_k %d (configured %d)",
            (read_plan_focus() or {}).get("n_open"),
            planned_top_k,
            _wikidata_config.pruning_top_k,
        )
    triples = await _stage_a_prune(
        _focused_query(query),
        triples,
        reranker_url,
        reranker_model,
        top_k=planned_top_k,
        delta=_wikidata_config.pruning_delta,
        instruction=reranker_instruction,
    )

    if registry is not None:
        triples = await _stage_b_prune(query, triples, registry)

    # Return readable triple strings (labels), each carrying the object QID in
    # ``[Q…]`` so the tool-calling agent can pick which object to extend into on
    # the next hop. Raw QIDs alone are unreadable; bare labels can't be re-seeded.
    # Format ``subject [Qs] -- relation -- object [Qo]`` is parsed back into a
    # graph edge downstream (memory_update._coerce_raw_triple_to_relation).
    return [_format_triple_for_agent(t) for t in triples]


def _format_triple_for_agent(triple: WikiTriple) -> str:
    """Render one triple as ``subject [Qs] -- relation -- object [Qo]``.

    Labels make it human/LLM readable; the bracketed object QID lets the agent
    re-seed ``fetch_and_prune_subgraph`` to traverse the next hop. Literal
    objects (dates, numbers) have no QID and render as plain text.
    """

    def _part(node: Any) -> str:
        if isinstance(node, WikidataEntity):
            # A bare QID is meaningless for the agent's reasoning, so fall back
            # through every semantic field we have before resorting to the id:
            # label → first alias → description → QID. The ``[QID]`` suffix is
            # always kept as the hop handle (so the agent can re-seed) and as the
            # id the downstream parser recovers.
            name = (node.label or "").strip()
            if not name:
                name = next(
                    (a.strip() for a in (node.aliases or []) if a and a.strip()), ""
                )
            if not name:
                name = (node.description or "").strip()
            if not name:
                name = (node.qid or "").strip()
                if node.qid:
                    logger.debug(
                        "KG triple entity %s has no label/alias/description; "
                        "rendering bare QID (no semantics for the agent)",
                        node.qid,
                    )
            return f"{name} [{node.qid}]" if node.qid else name
        return str(node).strip()

    relation = str(triple.relation).strip()
    return f"{_part(triple.subject)} -- {relation} -- {_part(triple.object)}"


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
        enriched = await _wikidata_client.enrich_entities(valid_qids, get_details=True)
    except Exception as exc:
        logger.error("Entity enrichment failed for %s: %s", qids, exc)
        return []

    return enriched or []
