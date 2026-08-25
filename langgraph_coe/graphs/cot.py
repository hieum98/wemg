"""CoT strategy graph.

Chain-of-thought reasoning expressed as explicit LangGraph nodes with native
fan-out (one superstep per decomposition round).

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
import hashlib
import json
import logging
import operator
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import networkx as nx
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

from ..config import LangGraphCoeConfig, RerankerConfig
from ..llm import RoleModelRegistry, execute_role_lc, is_safe_default
from ..roles import (
    ANSWER_GENERATOR,
    EXTRACTOR,
    FINAL_ANSWER_SYNTHESIZER,
    PLANNER,
    SUBQUESTION_GENERATOR,
    AnswerGenerationInput,
    ExtractionInput,
    FinalAnswerSynthesisInput,
    PlanInput,
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
# Clear marker + append_or_clear reducer
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
    # Parallel to ``subquestions``: 0-based index of the plan intent each
    # subquestion advances (``None`` = unattributed). Populated only when a plan
    # is active. ``plan_gate`` uses it to decide which intent a resolved binding
    # closes; an unattributed binding closes nothing rather than guessing.
    subquestion_serves_intent: List[Optional[int]]
    # True when every SUBQUESTION_GENERATOR completion exhausted its shake
    # retries. Distinct from ``is_answerable`` because an unparseable response and
    # "no gaps remain" produce identical output fields.
    subq_parse_failed: bool
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
    # Index-parallel to ``current_subanswers``, holding each answer's
    # ``concise_answer``. ``plan_gate`` binds referents from this rather than the
    # full prose answer, which also names supporting entities.
    current_subanswers_concise: List[str]
    # This hop's consolidation evictions, as surfaced by ``MemoryUpdateGraph``.
    # ``reason == "contradicted"`` is the retraction signal ``plan_gate`` reads.
    last_retractions: List[Dict[str, Any]]

    # Append-only trajectory: each entry is one CoT iteration's decomposition.
    iteration_history: Annotated[List[Dict[str, Any]], operator.add]

    # Cross-iteration memory (updated by MemoryUpdateGraph)
    text_memory: List[str]
    graph_memory: nx.DiGraph
    entity_dict: Dict[str, Any]

    # ── Plan channel ────────────────────────────────────────────────────────
    # Prose statement of what must be found out. Injected into prompts via the
    # typed ``plan`` field on SubquestionGenerationInput / SelfCorrectionInput,
    # and NEVER written into ``text_memory`` / ``new_text_items`` /
    # ``candidate_answers``: an interrogative in memory is picked up by
    # ``_reverify_memory`` as a retrieval query, then reaches the verifier as
    # grounding and the synthesizer as a candidate answer.
    plan: str
    plan_version: int
    # One entry per plan intent. ``bindings`` accumulates the referents that
    # retrieval resolved for that intent; ``premises`` are the [Retrieval] facts
    # the plan cited when it was written (the set a retraction is matched
    # against); ``attempts`` is the negative record — what was queried and what it
    # yielded — which memory structurally cannot hold, since "nothing was found"
    # is not a fact about the world.
    plan_ledger: List[Dict[str, Any]]
    # "none" | "update" | "replan". Its own channel rather than a derived read of
    # the ledger tail, so a no-op gate cannot silently repeat the previous hop's
    # decision. Reset by ``increment``.
    plan_action: str
    # Append-only audit of every ``plan_gate`` decision, for the fire-rate
    # experiment. Written even when ``replan_max`` keeps the router inert.
    plan_action_log: Annotated[List[Dict[str, Any]], operator.add]
    # Set by MCTS when it passes its own plan into a rollout: the rollout may read
    # and log but must not regenerate or revise the parent's plan.
    plan_frozen: bool

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


# ──────────────────────────────────────────────────────────────────────────────
# Plan channel helpers
# ──────────────────────────────────────────────────────────────────────────────

PLAN_ACTION_NONE = "none"
PLAN_ACTION_UPDATE = "update"
PLAN_ACTION_REPLAN = "replan"

# Ledger intent statuses.
INTENT_OPEN = "open"
INTENT_CLOSED = "closed"
INTENT_CONTESTED = "contested"


def build_plan_ledger(
    intents: Sequence[str], premises: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """One ledger entry per plan intent, all open.

    ``premises`` (the ``[Retrieval]`` facts the plan cited) is recorded on every
    entry rather than attributed per-intent: the PLANNER quotes them for the plan
    as a whole, and the set is small, so matching a retraction against the union
    is both cheap and the conservative choice.
    """
    shared_premises = [p for p in (premises or []) if isinstance(p, str) and p.strip()]
    ledger: List[Dict[str, Any]] = []
    for intent in intents:
        if not (isinstance(intent, str) and intent.strip()):
            continue
        ledger.append(
            {
                "intent": intent.strip(),
                "status": INTENT_OPEN,
                "bindings": [],
                "premises": list(shared_premises),
                "attempts": [],
                "closed_at": None,
            }
        )
    return ledger


def _entity_label_to_qid(entity_dict: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Lowercased entity label → QID, from the linked-entity store.

    ``entity_dict`` is keyed by QID with ``WikidataEntity`` values, so resolving a
    *surface form* needs this reverse index. Mirrors the label handling in
    :func:`_known_entity_labels` (tolerating dicts/strings under test stubs) so
    both paths agree on what counts as a known entity.
    """
    out: Dict[str, str] = {}
    for qid, ent in (entity_dict or {}).items():
        label = getattr(ent, "label", None)
        if label is None and isinstance(ent, dict):
            label = ent.get("label")
        if label is None and isinstance(ent, str):
            label = ent
        resolved_qid = getattr(ent, "qid", None) or (
            ent.get("qid") if isinstance(ent, dict) else None
        )
        resolved_qid = resolved_qid or (qid if isinstance(qid, str) else None)
        if isinstance(label, str) and label.strip() and resolved_qid:
            out[label.strip().lower()] = str(resolved_qid)
    return out


def resolve_binding_qids(text: str, label_to_qid: Dict[str, str]) -> List[str]:
    """QIDs of known entities mentioned in ``text``, in order of first mention.

    Longest-first matching so "Alan Turing" is preferred over a bare "Turing"
    when both are linked, and a label wholly inside a longer match is not counted
    twice. Returns distinct QIDs ordered by where they appear in ``text``.

    Callers deciding *referents* want :func:`resolve_primary_qid` instead — see
    the warning there.
    """
    haystack = (text or "").lower()
    if not haystack:
        return []
    hits: List[tuple[int, int, str]] = []  # (start, -len, qid)
    consumed: List[tuple[int, int]] = []
    for label in sorted(label_to_qid, key=len, reverse=True):
        if not label:
            continue
        start = haystack.find(label)
        if start < 0:
            continue
        end = start + len(label)
        # Skip a label wholly inside a span already matched by a longer label.
        if any(s <= start and end <= e for s, e in consumed):
            continue
        consumed.append((start, end))
        hits.append((start, -len(label), label_to_qid[label]))
    ordered: List[str] = []
    for _, _, qid in sorted(hits):
        if qid not in ordered:
            ordered.append(qid)
    return ordered


def resolve_primary_qid(text: str, label_to_qid: Dict[str, str]) -> Optional[str]:
    """The single referent ``text`` proposes, or None.

    One answer proposes **one** referent, so binding must take at most one QID
    from it. Taking every linked entity mentioned instead makes the contested test
    fire on answer *verbosity*: an answer reading "Alan Mathison Turing … the
    Turing machine … theoretical computer science" mentions three linked entities
    and would look like three competing referents for one intent.

    Distinct referents therefore come from distinct *answers*, never from multiple
    entities inside one. The earliest mention wins (answers lead with their
    referent), with the longest label breaking a tie at the same position.
    """
    ordered = resolve_binding_qids(text, label_to_qid)
    return ordered[0] if ordered else None


def classify_discharge(
    ledger: Sequence[Dict[str, Any]],
) -> tuple[str, Optional[int], List[str]]:
    """Decide ``plan_action`` from the ledger's current bindings.

    Returns ``(action, intent_index, competing_surfaces)``.

    * **contested** — an intent bound two or more *distinct QIDs*. Its closure is
      under-determined: this is the selection logic cannot supply, it is a fact
      about the plan's bookkeeping rather than about the world, and it needs no
      judge because QID identity is the whole discriminator.
    * **falsified** — an intent's ``premises`` intersect this hop's retractions
      (marked by :func:`apply_retractions`). Also a replan.
    * **update** — exactly one distinct QID survives: the intent closes.
    * **none** — nothing linked, so there is nothing to record.

    Contested wins over falsified when both apply: the ambiguity is the more
    specific failure and its repair (discriminate between referents) subsumes
    re-establishing the premise.
    """
    falsified: Optional[int] = None
    for idx, entry in enumerate(ledger):
        if entry.get("status") == INTENT_CONTESTED:
            surfaces = [
                b.get("surface", "")
                for b in (entry.get("bindings") or [])
                if b.get("qid")
            ]
            return PLAN_ACTION_REPLAN, idx, surfaces
        if entry.get("falsified") and falsified is None:
            falsified = idx
    if falsified is not None:
        return PLAN_ACTION_REPLAN, falsified, []
    for idx, entry in enumerate(ledger):
        if entry.get("status") == INTENT_CLOSED and entry.get("closed_at") is not None:
            return PLAN_ACTION_UPDATE, idx, []
    return PLAN_ACTION_NONE, None, []


def apply_bindings(
    ledger: Sequence[Dict[str, Any]],
    candidates: Sequence[tuple[Optional[int], str]],
    label_to_qid: Dict[str, str],
    hop: int,
) -> List[Dict[str, Any]]:
    """Fold this hop's ``(intent_index, answer_text)`` pairs into the ledger.

    Each candidate contributes **at most one** referent
    (:func:`resolve_primary_qid`) — one answer proposes one referent, so an
    answer that happens to mention several linked entities must not read as
    several competing ones.

    Only candidates that resolve to a known QID are recorded. Non-null-QID gating
    is what keeps paraphrase noise out of the contested test: two differently
    worded answers naming the same entity share a QID and so do not compete,
    while two genuinely different referents do.
    """
    out = [dict(entry) for entry in ledger]
    for entry in out:
        entry["bindings"] = list(entry.get("bindings") or [])
    # Intents closed by an *earlier* hop are settled and must not be reopened.
    # Closure reached within THIS call is not yet settled: the second candidate of
    # a contested pair arrives after the first has already closed the intent, so
    # skipping on the live status would hide every contest behind its own first
    # binding.
    already_closed = {
        i for i, e in enumerate(out) if e.get("status") == INTENT_CLOSED
    }
    for intent_idx, text in candidates:
        if intent_idx is None or not (0 <= intent_idx < len(out)):
            continue
        if intent_idx in already_closed:
            continue
        entry = out[intent_idx]
        qid = resolve_primary_qid(text, label_to_qid)
        if qid and not any(b.get("qid") == qid for b in entry["bindings"]):
            entry["bindings"].append({"surface": text, "qid": qid, "hop": hop})
        distinct = {b.get("qid") for b in entry["bindings"] if b.get("qid")}
        if len(distinct) >= 2:
            entry["status"] = INTENT_CONTESTED
            entry["closed_at"] = None
        elif len(distinct) == 1:
            entry["status"] = INTENT_CLOSED
            entry["closed_at"] = hop
    return out


def apply_retractions(
    ledger: Sequence[Dict[str, Any]], retractions: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Mark intents whose cited premises were contradicted by retrieval.

    Only ``reason == "contradicted"`` counts. The other eviction reasons
    (irrelevant, duplicate, hop_filtered, superseded) are housekeeping — they mean
    the consolidator tidied memory, not that a claim the plan leaned on turned out
    to be false.
    """
    contradicted = [
        (r.get("content") or "").strip().lower()
        for r in retractions
        if (r.get("reason") or "") == "contradicted" and (r.get("content") or "").strip()
    ]
    out = [dict(entry) for entry in ledger]
    if not contradicted:
        return out
    for entry in out:
        for premise in entry.get("premises") or []:
            key = (premise or "").strip().lower()
            if key and any(key in c or c in key for c in contradicted):
                entry["falsified"] = premise
                break
    return out


def latest_intermediate_answer(ledger: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Most recently closed binding, rendered for the ``intermediate_answer`` slot.

    This is the whole of UPDATE: a deterministic write that surfaces a resolved
    referent through a typed field whose prompt rule ("do NOT re-ask what was
    already resolved") already exists. No bound value is written into the plan
    prose — a referent there would be a world-claim with no provenance tag, no hop
    tag and no eviction path.
    """
    best: Optional[tuple[int, str, str]] = None
    for entry in ledger:
        if entry.get("status") != INTENT_CLOSED:
            continue
        closed_at = entry.get("closed_at")
        if closed_at is None:
            continue
        surface = next(
            (b.get("surface") for b in (entry.get("bindings") or []) if b.get("qid")),
            None,
        )
        if not surface:
            continue
        if best is None or int(closed_at) >= best[0]:
            best = (int(closed_at), str(entry.get("intent", "")), str(surface))
    if best is None:
        return None
    _, intent, surface = best
    return f"{intent}\n→ {surface}" if intent else surface


def render_plan_for_prompt(plan: str, ledger: Sequence[Dict[str, Any]]) -> str:
    """Annotate the prose plan with per-intent status for prompt injection.

    Resolved intents are marked so the generator does not re-open them, and
    contested ones are marked so it knows the ambiguity is live. Statuses are
    bookkeeping, not claims — no bound value is rendered here.
    """
    text = (plan or "").strip()
    if not ledger:
        return text
    lines = []
    for entry in ledger:
        status = entry.get("status", INTENT_OPEN)
        marker = {
            INTENT_CLOSED: "[resolved]",
            INTENT_CONTESTED: "[ambiguous - two candidate referents]",
        }.get(status, "[open]")
        lines.append(f"- {marker} {entry.get('intent', '')}")
    return f"{text}\n\nIntent status:\n" + "\n".join(lines) if text else ""


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

    # When the candidate set already fits within top_k, reranking only reorders
    # (it filters nothing). Downstream extraction reads the whole set, so order
    # is immaterial — skip the network call. Avoids the bulk of small-batch
    # reranker traffic (e.g. the 10-candidate calls in the eval logs).
    if top_k is not None and top_k > 0 and len(items) <= top_k:
        return items

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


def _maybe_dump_extraction(
    question: str,
    subqs: Sequence[str],
    items: Sequence[str],
    batches: Sequence[str],
    facts: Sequence[str],
) -> None:
    """Diagnostic: dump EXTRACTOR input passages vs output facts to confirm or
    kill the batching-dilution hypothesis (does the gold detail survive when
    many reranked passages are packed into one extractor call?).

    Off by default. Enable with ``COE_EXTRACT_DUMP_DIR=<dir>``; optionally narrow
    to specific questions with ``COE_EXTRACT_DUMP_FILTER=<substr>`` (case-insensitive,
    matched against the question) to avoid dumping the whole dataset. One JSON
    file per extraction call. Never raises into the hot path.
    """
    dump_dir = os.environ.get("COE_EXTRACT_DUMP_DIR")
    if not dump_dir:
        return
    try:
        filt = os.environ.get("COE_EXTRACT_DUMP_FILTER")
        if filt and filt.lower() not in (question or "").lower():
            return
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        qhash = hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:10]
        record = {
            "question": question,
            "subquestions": list(subqs),
            "num_passages": len(items),
            "passage_chars": [len(i) for i in items],
            "num_batches": len(batches),
            "batch_chars": [len(b) for b in batches],
            "num_facts": len(facts),
            "passages": list(items),  # raw reranked passages fed to the EXTRACTOR
            "facts": list(facts),     # atomic facts that survived extraction
        }
        path = out_dir / f"{qhash}_{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    except Exception:  # diagnostics must never break a run
        logger.warning("extraction dump failed", exc_info=True)


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
    _maybe_dump_extraction(question, subqs, items, batches, facts)
    return facts


# SUBQUESTION_GENERATOR is sampled with n completions and their decompositions
# are pooled (see ``pool_subquestions``).
N_SUBQUESTIONS = 3


@dataclass
class PooledSubquestions:
    """Result of pooling ``n`` SUBQUESTION_GENERATOR completions.

    ``n_survivors`` counts completions that actually parsed. It exists because
    ``execute_role_lc`` never raises on a retry-exhausted parse failure — it
    returns a neutral default whose ``is_answerable=False, subquestions=[]``
    is byte-identical to a genuine "no gaps left" answer. Callers must branch on
    ``n_survivors == 0`` *before* interpreting ``should_direct``, or an LLM outage
    is silently rewritten into "the question is answerable, synthesize now".
    """

    subquestions: List[str]
    needs_kg: List[bool]
    serves_intent: List[Optional[int]]
    should_direct: bool
    n_survivors: int


def pool_subquestions(outputs: Any) -> PooledSubquestions:
    """Pool ``n`` SUBQUESTION_GENERATOR completions into one decomposition.

    Only completions that judged the question *not* answerable contribute their
    subquestions; ``should_direct`` is the majority ``is_answerable`` vote over
    the completions that parsed. Dedups by subquestion text, preserving
    first-seen order, and carries the parallel ``needs_kg`` / ``serves_intent``
    arrays alongside.
    """
    if not isinstance(outputs, list):
        outputs = [outputs]
    # Drop retry-exhausted neutral defaults: they carry no decomposition and
    # must not be counted as votes in the ``should_direct`` majority.
    survivors = [o for o in outputs if o is not None and not is_safe_default(o)]
    subqs: List[str] = []
    flags: List[bool] = []
    intents: List[Optional[int]] = []
    seen: set[str] = set()
    answerable = 0
    for out in survivors:
        if bool(getattr(out, "is_answerable", False)):
            answerable += 1
            continue  # coe: answerable completions propose no subquestions
        sq_list = getattr(out, "subquestions", None) or []
        kg_list = getattr(out, "needs_kg", None) or []
        intent_list = getattr(out, "serves_intent", None) or []
        for i, sq in enumerate(sq_list):
            if not (isinstance(sq, str) and sq.strip()):
                continue
            key = sq.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            subqs.append(sq.strip())
            flags.append(bool(kg_list[i]) if i < len(kg_list) else True)
            raw_intent = intent_list[i] if i < len(intent_list) else None
            # -1 (or any negative) is the generator's "advances no plan intent"
            # encoding; normalize it to None so attribution stays absent rather
            # than pointing at a bogus ledger slot.
            intents.append(
                int(raw_intent)
                if isinstance(raw_intent, int) and not isinstance(raw_intent, bool)
                and raw_intent >= 0
                else None
            )
    should_direct = (answerable / len(survivors) > 0.5) if survivors else False
    return PooledSubquestions(
        subquestions=subqs,
        needs_kg=flags,
        serves_intent=intents,
        should_direct=should_direct,
        n_survivors=len(survivors),
    )


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
    corpus_enabled: bool = True,
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

    # Corpus is the recall floor when available; skipped entirely when there is no
    # local index (``corpus_search`` raises rather than returning empty).
    tasks = (
        [corpus_search.ainvoke({"query": sq}) for sq in subqs]
        if corpus_enabled
        else []
    )
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
    corpus_enabled = bool(getattr(getattr(cfg, "retriever", None), "enabled", True))
    extractor_max_chars = int(
        getattr(cfg.memory, "extractor_max_input_chars", 24_000) or 24_000
    )

    plan_cfg = getattr(getattr(cfg, "search", None), "plan", None)
    plan_enabled = bool(getattr(plan_cfg, "enabled", False))
    replan_max = int(getattr(plan_cfg, "replan_max", 0) or 0)
    replan_headroom = int(getattr(plan_cfg, "replan_min_depth_headroom", 2) or 0)

    # Compiled subgraphs. Lazy build at compile time keeps the per-question
    # cost down — same compiled instance is reused across all iterations.
    kg_graph = build_kg_search_graph(registry)
    web_graph = build_web_research_graph(registry)
    memory_graph = build_memory_update_graph(registry, memory_cfg=cfg.memory)

    async def gen_plan(state: CoTState) -> Dict[str, Any]:
        # Inherit rather than regenerate. An MCTS rollout is handed its parent's
        # plan; regenerating here would overwrite it and make each rollout search
        # a different decomposition, which is exactly the sibling-incoherence the
        # plan exists to remove.
        if state.get("plan"):
            return {}
        out, _ = await execute_role_lc(
            registry,
            PLANNER,
            PlanInput(
                question=state.get("question", ""),
                context=_join_memory_context(state),
            ),
        )
        if is_safe_default(out):
            logger.error(
                "[gen_plan] PLANNER returned no parseable output; continuing "
                "without a plan (the loop degrades to plain decomposition)"
            )
            return {"plan": "", "plan_version": 0, "plan_ledger": []}
        plan_text = str(getattr(out, "plan", "") or "").strip()
        intents = list(getattr(out, "intents", None) or [])
        premises = list(getattr(out, "premises", None) or [])
        return {
            "plan": plan_text,
            "plan_version": 1,
            "plan_ledger": build_plan_ledger(intents, premises),
            "plan_action": PLAN_ACTION_NONE,
        }

    async def gen_subq(state: CoTState) -> Dict[str, Any]:
        question = state.get("question", "")
        ctx = _join_memory_context(state)
        inp = SubquestionGenerationInput(
            question=question,
            context=ctx,
            plan=render_plan_for_prompt(
                state.get("plan", ""), state.get("plan_ledger") or []
            )
            or None,
            # UPDATE surfaces here and nowhere else: a resolved referent reaches
            # the prompt through this typed slot, whose "do NOT re-ask what was
            # already resolved" rule (roles.py) was previously inert because
            # nothing ever populated the field.
            intermediate_answer=latest_intermediate_answer(
                state.get("plan_ledger") or []
            ),
        )
        # coe parity: sample N completions, pool their decompositions.
        outs, _ = await execute_role_lc(
            registry, SUBQUESTION_GENERATOR, inp, n=N_SUBQUESTIONS
        )
        pooled = pool_subquestions(outs)
        if pooled.n_survivors == 0:
            # Every completion exhausted its shake retries. ``is_answerable`` must
            # stay False here: claiming answerability would route an LLM outage
            # straight to ``gen_final`` and present a synthesis over whatever
            # memory happened to exist as if the loop had converged.
            logger.error(
                "[gen_subq] no SUBQUESTION_GENERATOR completion parsed (depth=%s); "
                "treating as a retrieval-less iteration rather than 'answerable'",
                state.get("depth"),
            )
            return {
                "is_answerable": False,
                "subquestions": [],
                "subquestion_needs_kg": [],
                "subquestion_serves_intent": [],
                "subq_parse_failed": True,
            }
        return {
            # "answerable" routes the loop to synthesis, same as the majority vote.
            "is_answerable": pooled.should_direct or not pooled.subquestions,
            "subquestions": pooled.subquestions,
            "subquestion_needs_kg": pooled.needs_kg,
            "subquestion_serves_intent": pooled.serves_intent,
            "subq_parse_failed": False,
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
        if not corpus_enabled:
            return {}
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
        # Keep ``current_subanswers`` index-aligned with ``subquestions``: emit ""
        # for a blank/unparsed answer rather than skipping it. Dropping the entry
        # shifts every later answer up one slot, and both ``iteration_history``
        # and the MCTS rollout chain zip the two lists by index — so one blank
        # answer silently reattributes all subsequent answers to the wrong
        # sub-question. Consumers that need non-empty text filter downstream.
        answers: List[str] = []
        concise: List[str] = []
        for i in range(len(subqs)):
            r = results[i] if i < len(results) else None
            text = (
                getattr(r, "answer", None) or getattr(r, "concise_answer", None) or ""
            )
            answers.append(str(text))
            # ``concise_answer`` is the referent-bearing form; the full ``answer``
            # is prose that also names supporting entities ("… the Turing machine
            # … theoretical computer science"), any of which would otherwise be
            # mistaken for a competing referent by ``plan_gate``.
            concise.append(
                str(
                    getattr(r, "concise_answer", None)
                    or getattr(r, "answer", None)
                    or ""
                )
            )
        return {"current_subanswers": answers, "current_subanswers_concise": concise}

    async def mem_update(state: CoTState) -> Dict[str, Any]:
        # This iteration's sub-answers and retrieval facts were produced one hop
        # below the current reasoning depth, so they get ``[hop=depth+1]`` —
        # matching legacy CoT (``update_working_memory(result,
        # hop_depth=current.depth + 1)``). ``depth`` here is still the current
        # iteration's value; the ``increment`` node bumps it afterwards.
        hop_depth = int(state.get("depth", 0) or 0) + 1
        payload = {
            "question": state.get("question", ""),
            # ``current_subanswers`` is index-aligned with ``subquestions`` and so
            # may hold "" placeholders for unanswered slots — filter them here,
            # where only the text matters.
            "new_text_items": [
                a
                for a in (state.get("current_subanswers") or [])
                if isinstance(a, str) and a.strip()
            ],
            # Retrieval-grounded facts persist with [Retrieval] provenance so
            # consolidation can prefer them over generated subanswers.
            "new_retrieval_items": list(state.get("extracted_facts") or []),
            "new_raw_triples": list(state.get("retrieved_raw_triples") or []),
            "current_text_memory": list(state.get("text_memory") or []),
            "current_graph": state.get("graph_memory") or nx.DiGraph(),
            "entity_dict": dict(state.get("entity_dict") or {}),
            "hop_depth": hop_depth,
        }
        result = await memory_graph.ainvoke(payload)
        return {
            "text_memory": list(result.get("updated_text_memory") or []),
            "graph_memory": result.get("updated_graph") or nx.DiGraph(),
            "entity_dict": dict(result.get("updated_entity_dict") or {}),
            # Surfaced for ``plan_gate``'s falsified-discharge test. A retraction
            # with reason "contradicted" means a claim we held was overturned by
            # retrieved evidence — the only eviction reason that bears on the plan.
            "last_retractions": list(result.get("retractions") or []),
        }

    async def plan_gate(state: CoTState) -> Dict[str, Any]:
        """Classify this hop's plan bookkeeping. Deterministic — no LLM call.

        Runs after ``mem_update`` so it sees the consolidated memory (and its
        retractions) rather than the raw sub-answers. Always records its verdict in
        ``plan_action_log``, including when ``replan_max`` leaves the router inert:
        the trigger's fire rate is the thing being measured.
        """
        ledger = list(state.get("plan_ledger") or [])
        if not ledger:
            return {"plan_action": PLAN_ACTION_NONE}

        hop = int(state.get("depth", 0) or 0)
        label_to_qid = _entity_label_to_qid(state.get("entity_dict"))

        # Attribute each sub-answer to the plan intent the generator said it
        # serves. ``current_subanswers`` is index-aligned with ``subquestions``
        # (see ``gen_subanswers``), which is what makes this zip sound.
        intents = list(state.get("subquestion_serves_intent") or [])
        # Bind from the concise answers; fall back to the full ones when a caller
        # (e.g. an older-shaped state) did not supply them.
        answers = list(
            state.get("current_subanswers_concise")
            or state.get("current_subanswers")
            or []
        )
        candidates: List[tuple[Optional[int], str]] = []
        unattributed = 0
        for i, answer in enumerate(answers):
            if not (isinstance(answer, str) and answer.strip()):
                continue
            intent_idx = intents[i] if i < len(intents) else None
            if intent_idx is None:
                unattributed += 1
            candidates.append((intent_idx, answer))
        if unattributed:
            # Attribution is the weakest link in the design: an answer with no
            # intent index closes nothing rather than being guessed onto an
            # intent. Logged so the rate is visible in the fire-rate experiment.
            logger.info(
                "[plan_gate] %d/%d sub-answers had no serves_intent attribution",
                unattributed,
                len(candidates),
            )

        ledger = apply_bindings(ledger, candidates, label_to_qid, hop)
        ledger = apply_retractions(ledger, state.get("last_retractions") or [])
        # Negative record: what was asked and what it yielded. Memory cannot hold
        # this — "nothing was found for X" is not a fact about the world — and
        # without it a replanner given only (plan, memory) rewrites the same plan.
        n_facts = len(state.get("extracted_facts") or [])
        for i, subq in enumerate(state.get("subquestions") or []):
            intent_idx = intents[i] if i < len(intents) else None
            if intent_idx is None or not (0 <= intent_idx < len(ledger)):
                continue
            ledger[intent_idx]["attempts"] = list(
                ledger[intent_idx].get("attempts") or []
            ) + [{"query": subq, "n_facts": n_facts, "hop": hop}]

        action, intent_idx, competing = classify_discharge(ledger)
        entry = {
            "hop": hop,
            "action": action,
            "intent_index": intent_idx,
            "intent": ledger[intent_idx].get("intent") if intent_idx is not None else None,
            "competing_bindings": competing,
            "plan_version": int(state.get("plan_version", 0) or 0),
            # ``armed`` records whether the router was allowed to act, so a
            # log-only run stays distinguishable from an armed one after the fact.
            "armed": replan_max > 0,
        }
        if action == PLAN_ACTION_REPLAN:
            logger.info(
                "[plan_gate] replan signalled at hop=%s intent=%r competing=%s "
                "(armed=%s)",
                hop,
                entry["intent"],
                competing,
                replan_max > 0,
            )
        return {
            "plan_ledger": ledger,
            "plan_action": action,
            "plan_action_log": [entry],
        }

    async def replan(state: CoTState) -> Dict[str, Any]:
        """One PLANNER call that revises the plan's failed part.

        The failure is stated *mechanically* — no diagnostic LLM in between. The
        planner is a model and can infer the cause from the raw signals; an extra
        call would only add cost and a place to confabulate a narrative.
        """
        ledger = list(state.get("plan_ledger") or [])
        action, intent_idx, competing = classify_discharge(ledger)
        if intent_idx is None:
            return {"plan_action": PLAN_ACTION_NONE}
        entry = ledger[intent_idx]
        if entry.get("status") == INTENT_CONTESTED:
            failure = (
                f"The intent {entry.get('intent')!r} bound "
                f"{len(competing)} different referents, so its closure is "
                "under-determined. Add a step that discriminates between them."
            )
        else:
            failure = (
                f"A fact this plan relied on was contradicted by retrieved "
                f"evidence: {entry.get('falsified')!r}. Anything the plan built on "
                "it must be re-established."
            )
        attempts = [
            f"{a.get('query')} → {a.get('n_facts')} facts"
            for e in ledger
            for a in (e.get("attempts") or [])
        ]
        out, _ = await execute_role_lc(
            registry,
            PLANNER,
            PlanInput(
                question=state.get("question", ""),
                context=_join_memory_context(state),
                current_plan=state.get("plan", ""),
                failure=failure,
                # Surface forms only — never presented as an answer. A contested
                # referent is evidence of ambiguity, not a fact to build on.
                competing_bindings=competing or None,
                attempts=attempts or None,
            ),
        )
        if is_safe_default(out):
            logger.warning(
                "[replan] PLANNER returned no parseable output; keeping the "
                "current plan and clearing the trigger"
            )
            return {"plan_action": PLAN_ACTION_NONE}
        new_intents = list(getattr(out, "intents", None) or [])
        new_premises = list(getattr(out, "premises", None) or [])
        new_ledger = build_plan_ledger(new_intents, new_premises)
        # Carry forward closures whose intent text survived the rewrite, so a
        # replan does not re-open work that is already done.
        closed_by_intent = {
            (e.get("intent") or "").strip().lower(): e
            for e in ledger
            if e.get("status") == INTENT_CLOSED
        }
        for fresh in new_ledger:
            prior = closed_by_intent.get((fresh.get("intent") or "").strip().lower())
            if prior is not None:
                fresh["status"] = INTENT_CLOSED
                fresh["bindings"] = list(prior.get("bindings") or [])
                fresh["closed_at"] = prior.get("closed_at")
        return {
            "plan": str(getattr(out, "plan", "") or "").strip(),
            "plan_version": int(state.get("plan_version", 0) or 0) + 1,
            "plan_ledger": new_ledger,
            "plan_action": PLAN_ACTION_NONE,
        }

    async def increment(state: CoTState) -> Dict[str, Any]:
        # Only annotate the trajectory with plan bookkeeping when a plan is
        # active, so the A0 (plan-disabled) record stays byte-identical to the
        # pre-plan one and the ablation compares like with like.
        history_entry: Dict[str, Any] = {
            "depth": int(state.get("depth", 0) or 0),
            "subquestions": list(state.get("subquestions") or []),
            "subanswers": list(state.get("current_subanswers") or []),
        }
        if plan_enabled:
            history_entry["plan_version"] = int(state.get("plan_version", 0) or 0)
            history_entry["plan_action"] = state.get("plan_action", PLAN_ACTION_NONE)
        return {
            "iteration_history": [history_entry],
            "depth": int(state.get("depth", 0) or 0) + 1,
            "subquestions": [],
            "subquestion_needs_kg": [],
            "subquestion_serves_intent": [],
            "reranked_context": [],
            "extracted_facts": [],
            "current_subanswers": [],
            "current_subanswers_concise": [],
            "retrieved_raw_context": Clear(),
            "retrieved_raw_triples": Clear(),
            "last_retractions": [],
            "plan_action": PLAN_ACTION_NONE,
        }

    async def gen_final(state: CoTState) -> Dict[str, Any]:
        candidate_answers = list(state.get("text_memory") or [])
        if not candidate_answers:
            history = state.get("iteration_history") or []
            for entry in history:
                # Skip "" alignment placeholders (see ``gen_subanswers``).
                candidate_answers.extend(
                    a
                    for a in (entry.get("subanswers") or [])
                    if isinstance(a, str) and a.strip()
                )
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
        if state.get("subq_parse_failed"):
            # Retry-exhausted parse failure, not convergence. Burn one iteration
            # and re-ask rather than synthesizing over unchanged memory; the
            # ``max_depth`` check above bounds how often this can repeat.
            return "increment"
        subqs = list(state.get("subquestions") or [])
        if not subqs:
            # No new gaps and not flagged answerable — degenerate; finalize.
            return "gen_final"

        # Adaptive retrieval gate. Corpus always fans out (embedding-only,
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

    def route_after_plan_gate(state: CoTState) -> str:
        """Take the replan edge only when armed, signalled, and with headroom.

        ``replan_max == 0`` keeps the gate in log-only mode: ``plan_action`` is
        still computed and recorded, so a single run measures how often the
        trigger fires before any budget is spent acting on it.
        """
        if state.get("plan_action") != PLAN_ACTION_REPLAN:
            return "increment"
        if state.get("plan_frozen"):
            # An MCTS rollout may observe and log a replan signal but must not act
            # on it: the plan belongs to the tree node that spawned the rollout,
            # and revising it here would silently fork the parent's plan.
            return "increment"
        if replan_max <= 0:
            return "increment"
        if int(state.get("plan_version", 0) or 0) > replan_max:
            return "increment"
        depth = int(state.get("depth", 0) or 0)
        max_depth = int(state.get("max_depth", 0) or 0)
        if depth >= max_depth - replan_headroom:
            # A plan rewritten with no hops left to execute it is pure cost.
            return "increment"
        return "replan"

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

    # The plan nodes exist only when the feature is on, so the A0 baseline graph
    # is structurally identical to the pre-plan one (no extra supersteps, no
    # chance of an inert node perturbing behaviour).
    if plan_enabled:
        builder.add_node("gen_plan", gen_plan)
        builder.add_node("plan_gate", plan_gate)
        builder.add_node("replan", replan)
        builder.add_edge(START, "gen_plan")
        builder.add_edge("gen_plan", "gen_subq")
        builder.add_edge("mem_update", "plan_gate")
        builder.add_conditional_edges(
            "plan_gate", route_after_plan_gate, ["replan", "increment"]
        )
        builder.add_edge("replan", "increment")
    else:
        builder.add_edge(START, "gen_subq")
        builder.add_edge("mem_update", "increment")

    builder.add_conditional_edges(
        "gen_subq",
        route_after_subq,
        ["gen_final", "kg_one", "web_one", "corpus_join", "increment"],
    )
    builder.add_edge("kg_one", "rerank")
    builder.add_edge("web_one", "rerank")
    builder.add_edge("corpus_join", "rerank")
    builder.add_edge("rerank", "extract_relevant")
    builder.add_edge("extract_relevant", "gen_subanswers")
    builder.add_edge("gen_subanswers", "mem_update")
    builder.add_edge("increment", "gen_subq")
    builder.add_edge("gen_final", END)

    return builder.compile()


__all__ = [
    "Clear",
    "append_or_clear",
    "CoTState",
    "PooledSubquestions",
    "build_cot_graph",
    "rerank_context",
    # Plan channel
    "PLAN_ACTION_NONE",
    "PLAN_ACTION_UPDATE",
    "PLAN_ACTION_REPLAN",
    "INTENT_OPEN",
    "INTENT_CLOSED",
    "INTENT_CONTESTED",
    "build_plan_ledger",
    "resolve_binding_qids",
    "resolve_primary_qid",
    "apply_bindings",
    "apply_retractions",
    "classify_discharge",
    "latest_intermediate_answer",
    "render_plan_for_prompt",
]
