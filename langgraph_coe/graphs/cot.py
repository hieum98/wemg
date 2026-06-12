"""CoTGraph (Phase 2).

Replaces the legacy ``coe/reasoning/cot.py`` while-loop with explicit LangGraph
nodes and native fan-out. See `implementation_plan.md` §7 for the full design.

Flow::

    START
      → gen_subq
      → route:
            answerable or depth >= max_depth → gen_final → END
            else → Sends to kg_one (per subq) + web_one (per subq) + corpus_join (per subq)
      → rerank
      → extract_relevant       (EXTRACTOR; batched on char budget)
      → gen_subanswers
      → mem_update
      → increment
      → gen_subq    (loops)

Per-iteration scratch (``retrieved_raw_context``, ``retrieved_raw_triples``) is
cleared via the :class:`Clear` sentinel reducer; cross-iteration memory
(``text_memory`` / ``graph_memory`` / ``entity_dict``) is updated by
``MemoryUpdateGraph`` and survives across iterations. The trajectory record
``iteration_history`` is append-only (``operator.add`` reducer).
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import networkx as nx
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

from ..config import LangGraphCoeConfig, RerankerConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import (
    ANSWER_GENERATOR,
    EXTRACTOR,
    FINAL_ANSWER_SYNTHESIZER,
    SUBQUESTION_GENERATOR,
    AnswerGenerationInput,
    ExtractionInput,
    FinalAnswerSynthesisInput,
    SubquestionGenerationInput,
)
from ..tools.retrieval import call_sglang_reranker, corpus_search
from ..tools.wikidata import reset_wikidata_session
from ._memory_text import textualize_graph as _textualize_graph
from .kg_search import build_kg_search_graph
from .memory_update import build_memory_update_graph
from .web_research import build_web_research_graph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# §7.1 Clear marker + append_or_clear reducer
# ──────────────────────────────────────────────────────────────────────────────


T = TypeVar("T")


@dataclass
class Clear:
    """Sentinel signaling :func:`append_or_clear` to reset the accumulator.

    A typed sentinel (rather than a string like ``"CLEAR"``) keeps the reducer
    signature unambiguous: a list of strings cannot collide with the marker.
    """


def append_or_clear(
    left: Union[List[T], None],
    right: Union[List[T], "Clear"],
) -> List[T]:
    """Reducer: append unless the right operand is a :class:`Clear` sentinel."""
    if isinstance(right, Clear):
        return []
    return (left or []) + list(right or [])


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────


class CoTState(TypedDict, total=False):
    # Inputs & config
    question: str
    max_depth: int

    # Loop control
    depth: int
    is_answerable: bool

    # Per-iteration scratch
    subquestions: List[str]
    # Parallel to ``subquestions``: per-subquestion KG-routing flags from the
    # SUBQUESTION_GENERATOR (``needs_kg``). Drives the adaptive retrieval gate in
    # ``route_after_subq``. Absent/short entries default to KG-on (recall-safe).
    subquestion_needs_kg: List[bool]
    retrieved_raw_context: Annotated[List[str], append_or_clear]
    retrieved_raw_triples: Annotated[List[Any], append_or_clear]
    # ``rerank`` node output: the top-k *full passages* from the merged pool.
    reranked_context: List[str]
    # ``extract_relevant`` node output: the EXTRACTOR's atomic, self-contained
    # *facts* distilled from ``reranked_context``. Kept in a distinct key (rather
    # than overwriting ``reranked_context``) so each node's output stays legible
    # in traces; ``gen_subanswers`` grounds on this, falling back to passages.
    extracted_facts: List[str]
    current_subanswers: List[str]

    # Append-only trajectory: each entry is one CoT iteration's decomposition.
    iteration_history: Annotated[List[Dict[str, Any]], operator.add]

    # Cross-iteration memory (updated by MemoryUpdateGraph)
    text_memory: List[str]
    graph_memory: nx.DiGraph
    entity_dict: Dict[str, Any]

    # Output
    final_answer: str
    concise_answer: str
    reasoning: str

    # Per-Send scratch fields (set when fan-out injects into a worker)
    subquery: str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _join_memory_context(state: CoTState) -> str:
    text_mem = list(state.get("text_memory") or [])
    graph_text = _textualize_graph(state.get("graph_memory"))
    sections: List[str] = []
    if text_mem:
        sections.append("Text memory:\n" + "\n".join(f"- {item}" for item in text_mem))
    if graph_text:
        sections.append("Graph memory:\n" + graph_text)
    return "\n\n".join(sections) if sections else "Not provided"


def _known_entity_labels(entity_dict: Optional[Dict[str, Any]]) -> List[str]:
    """Lowercased surface labels of already-linked entities in ``entity_dict``.

    Values are ``WikidataEntity`` (``.label``) in production, but tolerate dicts
    and raw strings so the override degrades gracefully under stubs/tests.
    """
    labels: List[str] = []
    for ent in (entity_dict or {}).values():
        label = getattr(ent, "label", None)
        if label is None and isinstance(ent, dict):
            label = ent.get("label")
        if label is None and isinstance(ent, str):
            label = ent
        if isinstance(label, str) and label.strip():
            labels.append(label.strip().lower())
    return labels


def _subq_hits_known_entity(subquery: str, labels: Sequence[str]) -> bool:
    """True if ``subquery`` mentions an entity we already hold a QID for.

    A known-entity hit forces the KG branch on regardless of the LLM's
    ``needs_kg`` tag: holding a resolved QID is the highest-yield KG case
    (cheap multi-hop from an existing graph node).
    """
    s = (subquery or "").lower()
    return any(label in s for label in labels if label)


def _format_web_result(row: Dict[str, Any]) -> str:
    """Render a single web_research result row as a single context string."""
    title = str(row.get("title", "")).strip()
    snippet = str(row.get("snippet", "")).strip()
    full_text = str(row.get("full_text", "")).strip()
    parts = [p for p in (title, snippet, full_text) if p]
    return "\n".join(parts)


async def rerank_context(
    query: str,
    contexts: Sequence[str],
    top_k: Optional[int] = None,
    cfg: Optional[RerankerConfig] = None,
    **_kwargs: Any,
) -> List[str]:
    """Score ``contexts`` against ``query`` and return the top-``top_k``.

    When ``cfg`` is ``None`` or ``cfg.enabled`` is False this falls back to an
    identity slice (filter empty + truncate). When the reranker is enabled the
    function POSTs to the SGLang Qwen3-Reranker endpoint via
    :func:`call_sglang_reranker` — errors propagate, no silent fallback.

    The test suite monkeypatches this symbol directly to inject deterministic
    rerank orderings; the ``cfg`` kwarg is harmless to ignore in those stubs.
    """
    items = [c for c in contexts if isinstance(c, str) and c.strip()]
    if not items:
        return []

    if cfg is None or not getattr(cfg, "enabled", False):
        if top_k is None or top_k <= 0:
            return items
        return items[:top_k]

    ranked = await call_sglang_reranker(query, items, cfg)
    if top_k is None or top_k <= 0:
        return [items[idx] for idx, _ in ranked]
    return [items[idx] for idx, _ in ranked[:top_k]]


async def rerank_per_query(
    queries: Sequence[str],
    contexts: Sequence[str],
    top_k: int,
    cfg: Optional[RerankerConfig],
) -> List[str]:
    """Rerank ``contexts`` against each query independently; union the top-k.

    Reranking against a single concatenated multi-subquestion query lets
    evidence for one subquestion crowd out the others inside one shared top-k
    budget. Per-query reranking guarantees each subquestion keeps its own
    ``top_k`` slots; the union is deduplicated preserving first-seen order.
    """
    qs = [q for q in queries if isinstance(q, str) and q.strip()]
    if not qs:
        return []
    ranked_lists = await asyncio.gather(
        *[rerank_context(q, contexts, top_k=top_k, cfg=cfg) for q in qs]
    )
    merged: List[str] = []
    seen: set[str] = set()
    for ranked in ranked_lists:
        for ctx in ranked or []:
            key = ctx.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(ctx)
    return merged


_EXTRACTOR_BATCH_SEP = "\n\n---\n\n"


def _split_into_char_batches(
    items: Sequence[str], max_chars: int, sep: str = _EXTRACTOR_BATCH_SEP
) -> List[str]:
    """Pack ``items`` into ``sep``-joined blobs each ≤ ``max_chars``.

    An item that exceeds ``max_chars`` on its own goes into its own batch
    untouched — truncation here would silently drop evidence; we trust the
    EXTRACTOR's model to handle a single oversized passage and rely on the
    caller's tier ``max_input_tokens`` as the real ceiling.
    """
    batches: List[str] = []
    current: List[str] = []
    current_len = 0
    sep_len = len(sep)
    for item in items:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        item_len = len(stripped)
        projected = current_len + (sep_len if current else 0) + item_len
        if current and projected > max_chars:
            batches.append(sep.join(current))
            current = [stripped]
            current_len = item_len
        else:
            current.append(stripped)
            current_len = projected
    if current:
        batches.append(sep.join(current))
    return batches


async def extract_facts(
    registry: RoleModelRegistry,
    question: str,
    subquestions: Sequence[str],
    contexts: Sequence[str],
    max_chars: int,
) -> List[str]:
    """EXTRACTOR pass: distill passages into atomic, self-contained facts.

    Shared core of the CoT ``extract_relevant`` node and MCTS-expansion
    evidence gathering. Batches by char budget, runs batches in parallel,
    dedupes case-insensitively. Returns ``[]`` when there is nothing to
    extract; callers decide their own fallback (typically the raw passages).
    """
    items = [c for c in contexts if isinstance(c, str) and c.strip()]
    if not items:
        return []

    subqs = [s for s in subquestions if s and s.strip()]
    question_blob = question or ""
    if subqs:
        subq_lines = "\n".join(f"- {sq}" for sq in subqs)
        question_blob = (
            f"{question_blob}\n\nCurrent subquestions:\n{subq_lines}"
            if question_blob
            else f"Current subquestions:\n{subq_lines}"
        )

    batches = _split_into_char_batches(items, max_chars)
    if not batches:
        return []

    async def _extract_one(blob: str) -> List[str]:
        out, _ = await execute_role_lc(
            registry,
            EXTRACTOR,
            ExtractionInput(question=question_blob, raw_data=blob),
        )
        return list(getattr(out, "relevant_information", None) or [])

    results = await asyncio.gather(*[_extract_one(b) for b in batches])

    facts: List[str] = []
    seen: set[str] = set()
    for batch in results:
        for fact in batch:
            if not isinstance(fact, str):
                continue
            key = fact.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            facts.append(fact.strip())
    return facts


async def gather_evidence(
    registry: RoleModelRegistry,
    question: str,
    subquestions: Sequence[str],
    *,
    needs_kg: Optional[Sequence[bool]] = None,
    memory_context: str = "Not provided",
    entity_dict: Optional[Dict[str, Any]] = None,
    kg_graph: Any = None,
    web_graph: Any = None,
    web_enabled: bool = False,
    reranker_cfg: Optional[RerankerConfig] = None,
    rerank_top_k: int = 10,
    extractor_max_chars: int = 24_000,
) -> Dict[str, Any]:
    """One full retrieval pass for a set of subquestions, outside the CoT loop.

    Mirrors the CoT iteration's evidence path (corpus + gated KG [+ web]
    fan-out → per-subquestion rerank → EXTRACTOR distillation) as a plain
    callable, so MCTS expansion can ground subanswers in retrieved evidence
    instead of memory/parametric knowledge alone.

    Returns ``{"extracted_facts": [...], "raw_triples": [...]}``. When the
    extractor yields nothing, ``extracted_facts`` falls back to the reranked
    passages — no evidence is silently lost.
    """
    subqs = [s.strip() for s in subquestions if isinstance(s, str) and s.strip()]
    if not subqs:
        return {"extracted_facts": [], "raw_triples": []}

    flags = list(needs_kg or [])
    known_labels = _known_entity_labels(entity_dict)

    async def _kg_search_isolated(payload: Dict[str, Any]) -> Any:
        # Fresh hop budget + visited set per KG search. The session ContextVar
        # rebinding stays local to this gather child task (see
        # tools/wikidata.py). Without it the whole question shares a single
        # ``max_hops`` budget: the first subquery's fetch exhausts it and every
        # later expansion's KG fan-out degrades to "[hop budget exhausted]",
        # so facts needed at hop 2+ (e.g. attributes of a bridge entity
        # resolved mid-search) never surface. Legacy coe scoped the budget per
        # retrieval call (`_retrieve_from_kb`), not per question.
        reset_wikidata_session()
        return await kg_graph.ainvoke(payload)

    tasks = [corpus_search.ainvoke({"query": sq}) for sq in subqs]
    n_corpus = len(tasks)
    kg_index: List[int] = []
    for i, sq in enumerate(subqs):
        tagged_kg = flags[i] if i < len(flags) else True
        if kg_graph is not None and (
            tagged_kg or _subq_hits_known_entity(sq, known_labels)
        ):
            kg_index.append(i)
            tasks.append(
                _kg_search_isolated(
                    {
                        "subquery": sq,
                        "original_query": question,
                        "context": memory_context,
                    }
                )
            )
    n_kg = len(kg_index)
    if web_enabled and web_graph is not None:
        for sq in subqs:
            tasks.append(
                web_graph.ainvoke(
                    {
                        "subquery": sq,
                        "original_query": question,
                        "context": memory_context,
                    }
                )
            )

    results = await asyncio.gather(*tasks)
    corpus_results = results[:n_corpus]
    kg_results = results[n_corpus : n_corpus + n_kg]
    web_results = results[n_corpus + n_kg :]

    pooled: List[str] = []
    seen: set[str] = set()

    def _add(item: Any) -> None:
        if not isinstance(item, str):
            return
        stripped = item.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            pooled.append(item)

    for batch in corpus_results:
        for item in batch or []:
            _add(item)
    raw_triples: List[Any] = []
    for kg_out in kg_results:
        for art in (kg_out or {}).get("kg_articles") or []:
            _add(art)
        raw_triples.extend(t for t in ((kg_out or {}).get("triples") or []) if t)
    for web_out in web_results:
        for row in (web_out or {}).get("results") or []:
            if isinstance(row, dict):
                _add(_format_web_result(row))

    reranked = await rerank_per_query(subqs, pooled, rerank_top_k, reranker_cfg)
    facts = await extract_facts(
        registry, question, subqs, reranked, extractor_max_chars
    )
    return {"extracted_facts": facts or reranked, "raw_triples": raw_triples}


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────


def build_cot_graph(
    registry: RoleModelRegistry,
    config: Optional[LangGraphCoeConfig] = None,
):
    """Compile the CoTGraph. ``config`` supplies reranker top_k + retrieval config."""

    cfg = config or LangGraphCoeConfig()
    rerank_top_k = max(1, int(getattr(cfg.reranker, "top_k", 5) or 5))
    # Adaptive retrieval (paper parity): web fan-out is gated off by default;
    # KG fan-out is gated per-subquestion by the generator's ``needs_kg`` tag.
    web_enabled = bool(getattr(getattr(cfg, "web_search", None), "enabled", False))
    extractor_max_chars = int(
        getattr(cfg.memory, "extractor_max_input_chars", 24_000) or 24_000
    )

    # Compiled subgraphs. Lazy build at compile time keeps the per-question
    # cost down — same compiled instance is reused across all iterations.
    kg_graph = build_kg_search_graph(registry)
    web_graph = build_web_research_graph(registry)
    memory_graph = build_memory_update_graph(registry, memory_cfg=cfg.memory)

    async def gen_subq(state: CoTState) -> Dict[str, Any]:
        question = state.get("question", "")
        ctx = _join_memory_context(state)
        inp = SubquestionGenerationInput(question=question, context=ctx)
        out, _ = await execute_role_lc(registry, SUBQUESTION_GENERATOR, inp)
        return {
            "is_answerable": bool(getattr(out, "is_answerable", False)),
            "subquestions": list(getattr(out, "subquestions", None) or []),
            "subquestion_needs_kg": list(getattr(out, "needs_kg", None) or []),
        }

    async def kg_one(state: CoTState) -> Dict[str, Any]:
        subquery = state.get("subquery") or ""
        if not subquery:
            return {}
        # Per-search hop budget — see ``_kg_search_isolated`` in
        # :func:`gather_evidence` for why the session must not span the question.
        reset_wikidata_session()
        result = await kg_graph.ainvoke(
            {
                "subquery": subquery,
                "original_query": state.get("question", ""),
                "context": _join_memory_context(state),
            }
        )
        new_ctx: List[str] = []
        for art in result.get("kg_articles") or []:
            if isinstance(art, str) and art.strip():
                new_ctx.append(art)
        triples = [t for t in (result.get("triples") or []) if t]
        return {
            "retrieved_raw_context": new_ctx,
            "retrieved_raw_triples": triples,
        }

    async def web_one(state: CoTState) -> Dict[str, Any]:
        subquery = state.get("subquery") or ""
        if not subquery:
            return {}
        result = await web_graph.ainvoke(
            {
                "subquery": subquery,
                "original_query": state.get("question", ""),
                "context": _join_memory_context(state),
            }
        )
        new_ctx: List[str] = []
        for row in result.get("results") or []:
            if isinstance(row, dict):
                rendered = _format_web_result(row)
                if rendered:
                    new_ctx.append(rendered)
        return {"retrieved_raw_context": new_ctx}

    async def corpus_join(state: CoTState) -> Dict[str, Any]:
        # One ``corpus_search`` per subquestion, in parallel — symmetric with
        # KG / web fan-out. ``corpus_search`` is a module-level reference so
        # tests can monkeypatch it.
        subqs = [
            s.strip() for s in (state.get("subquestions") or []) if s and s.strip()
        ]
        if not subqs:
            return {}

        results = await asyncio.gather(
            *[corpus_search.ainvoke({"query": sq}) for sq in subqs]
        )

        new_ctx: List[str] = []
        seen: set[str] = set()
        for batch in results:
            for item in batch or []:
                if not isinstance(item, str):
                    continue
                stripped = item.strip()
                if not stripped or stripped in seen:
                    continue
                seen.add(stripped)
                new_ctx.append(item)
        return {"retrieved_raw_context": new_ctx}

    async def rerank(state: CoTState) -> Dict[str, Any]:
        contexts = list(state.get("retrieved_raw_context") or [])
        subqs = [s for s in (state.get("subquestions") or []) if s and s.strip()]
        # Per-subquestion rerank (union of per-query top-k) so one subquestion's
        # evidence cannot crowd the others out of a single shared top-k budget.
        # ``cfg.reranker`` is captured from the builder closure and forwarded so
        # ``rerank_context`` can decide between SGLang call vs identity slice.
        queries = subqs or [state.get("question", "")]
        reranked = await rerank_per_query(
            queries, contexts, rerank_top_k, cfg.reranker
        )
        return {"reranked_context": list(reranked or [])}

    async def extract_relevant(state: CoTState) -> Dict[str, Any]:
        """Distill reranked passages into atomic, self-contained facts.

        Wraps the ``EXTRACTOR`` role over the joined top-k. The reranker
        already handled relevance filtering; the extractor's job here is
        rewriting for self-containment (anaphora resolution, removing
        document-internal references) and atomicity (one claim per item).

        Batches the joined input by ``cfg.memory.extractor_max_input_chars``
        so the EXTRACTOR's prompt never overflows the model's context window,
        with parallel calls across batches via ``asyncio.gather``. Falls back
        to the raw reranked passages if the extractor returns nothing — no
        evidence is silently lost.
        """
        contexts = list(state.get("reranked_context") or [])
        if not contexts:
            return {}

        # Anchor extraction on the original question + current subquestions so
        # the EXTRACTOR's relevance lens covers both the global intent and the
        # specific gaps the loop is currently filling.
        subqs = [s for s in (state.get("subquestions") or []) if s and s.strip()]
        facts = await extract_facts(
            registry,
            state.get("question", ""),
            subqs,
            contexts,
            extractor_max_chars,
        )

        # Write to ``extracted_facts`` (not ``reranked_context``) so the rerank
        # passages stay visible in the trace. If the extractor produced nothing,
        # fall back to the reranked passages so ``gen_subanswers`` still has
        # evidence to ground on — no evidence is silently lost.
        return {"extracted_facts": facts or contexts}

    async def gen_subanswers(state: CoTState) -> Dict[str, Any]:
        subqs = list(state.get("subquestions") or [])
        if not subqs:
            return {"current_subanswers": []}
        # Ground on the EXTRACTOR's atomic facts; fall back to the reranked
        # passages if extraction yielded nothing.
        evidence = state.get("extracted_facts") or state.get("reranked_context") or []
        ctx_joined = "\n".join(evidence)
        inputs = [
            AnswerGenerationInput(question=sq, context=ctx_joined or "Not provided")
            for sq in subqs
        ]
        results, _ = await execute_role_lc(registry, ANSWER_GENERATOR, inputs)
        if not isinstance(results, list):
            results = [results]
        answers: List[str] = []
        for r in results:
            text = (
                getattr(r, "answer", None) or getattr(r, "concise_answer", None) or ""
            )
            if text:
                answers.append(text)
        return {"current_subanswers": answers}

    async def mem_update(state: CoTState) -> Dict[str, Any]:
        payload = {
            "question": state.get("question", ""),
            "new_text_items": list(state.get("current_subanswers") or []),
            # Retrieval-grounded facts persist with [Retrieval] provenance so
            # consolidation can prefer them over generated subanswers.
            "new_retrieval_items": list(state.get("extracted_facts") or []),
            "new_raw_triples": list(state.get("retrieved_raw_triples") or []),
            "current_text_memory": list(state.get("text_memory") or []),
            "current_graph": state.get("graph_memory") or nx.DiGraph(),
            "entity_dict": dict(state.get("entity_dict") or {}),
        }
        result = await memory_graph.ainvoke(payload)
        return {
            "text_memory": list(result.get("updated_text_memory") or []),
            "graph_memory": result.get("updated_graph") or nx.DiGraph(),
            "entity_dict": dict(result.get("updated_entity_dict") or {}),
        }

    async def increment(state: CoTState) -> Dict[str, Any]:
        return {
            "iteration_history": [
                {
                    "depth": int(state.get("depth", 0) or 0),
                    "subquestions": list(state.get("subquestions") or []),
                    "subanswers": list(state.get("current_subanswers") or []),
                }
            ],
            "depth": int(state.get("depth", 0) or 0) + 1,
            "subquestions": [],
            "reranked_context": [],
            "extracted_facts": [],
            "current_subanswers": [],
            "retrieved_raw_context": Clear(),
            "retrieved_raw_triples": Clear(),
        }

    async def gen_final(state: CoTState) -> Dict[str, Any]:
        candidate_answers = list(state.get("text_memory") or [])
        if not candidate_answers:
            history = state.get("iteration_history") or []
            for entry in history:
                candidate_answers.extend(entry.get("subanswers") or [])
        if not candidate_answers:
            candidate_answers = ["No prior reasoning available."]

        ctx = _join_memory_context(state)
        inp = FinalAnswerSynthesisInput(
            question=state.get("question", ""),
            candidate_answers=candidate_answers,
            context=ctx,
        )
        out, _ = await execute_role_lc(registry, FINAL_ANSWER_SYNTHESIZER, inp)
        final_text = (
            getattr(out, "final_answer", None)
            or getattr(out, "concise_answer", None)
            or ""
        )
        concise = getattr(out, "concise_answer", None) or final_text
        reasoning = getattr(out, "reasoning", None) or ""
        return {
            "final_answer": str(final_text),
            "concise_answer": str(concise),
            "reasoning": str(reasoning),
        }

    # ── Routing ──────────────────────────────────────────────────────────────

    def route_after_subq(state: CoTState):
        if state.get("is_answerable"):
            return "gen_final"
        if int(state.get("depth", 0) or 0) >= int(state.get("max_depth", 0) or 0):
            return "gen_final"
        subqs = list(state.get("subquestions") or [])
        if not subqs:
            # No new gaps and not flagged answerable — degenerate; finalize.
            return "gen_final"

        # Adaptive retrieval gate (§1a). Corpus always fans out (embedding-only,
        # ~free) as the recall floor. KG fires per-subquestion when the generator
        # tagged it entity-centric (``needs_kg``) OR the subquestion mentions an
        # already-linked entity. Web fires only when explicitly enabled.
        needs_kg = list(state.get("subquestion_needs_kg") or [])
        known_labels = _known_entity_labels(state.get("entity_dict"))

        sends: List[Send] = []
        for i, sq in enumerate(subqs):
            # Missing/short tag → default KG-on (recall-safe).
            tagged_kg = needs_kg[i] if i < len(needs_kg) else True
            if tagged_kg or _subq_hits_known_entity(sq, known_labels):
                sends.append(Send("kg_one", {**state, "subquery": sq}))
            if web_enabled:
                sends.append(Send("web_one", {**state, "subquery": sq}))
        sends.append(Send("corpus_join", dict(state)))
        return sends

    builder = StateGraph(CoTState)
    builder.add_node("gen_subq", gen_subq)
    builder.add_node("kg_one", kg_one)
    builder.add_node("web_one", web_one)
    builder.add_node("corpus_join", corpus_join)
    builder.add_node("rerank", rerank)
    builder.add_node("extract_relevant", extract_relevant)
    builder.add_node("gen_subanswers", gen_subanswers)
    builder.add_node("mem_update", mem_update)
    builder.add_node("increment", increment)
    builder.add_node("gen_final", gen_final)

    builder.add_edge(START, "gen_subq")
    builder.add_conditional_edges(
        "gen_subq",
        route_after_subq,
        ["gen_final", "kg_one", "web_one", "corpus_join"],
    )
    builder.add_edge("kg_one", "rerank")
    builder.add_edge("web_one", "rerank")
    builder.add_edge("corpus_join", "rerank")
    builder.add_edge("rerank", "extract_relevant")
    builder.add_edge("extract_relevant", "gen_subanswers")
    builder.add_edge("gen_subanswers", "mem_update")
    builder.add_edge("mem_update", "increment")
    builder.add_edge("increment", "gen_subq")
    builder.add_edge("gen_final", END)

    return builder.compile()


__all__ = [
    "Clear",
    "append_or_clear",
    "CoTState",
    "build_cot_graph",
    "rerank_context",
]
