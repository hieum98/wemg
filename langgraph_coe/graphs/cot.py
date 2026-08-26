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
import difflib
import hashlib
import json
import logging
import operator
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, TypeVar, Union

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
    # Index-parallel too: ``ANSWER_GENERATOR.confidence_level``, which had zero
    # readers repo-wide. ``plan_gate`` treats a low-confidence answer as no
    # binding — closing an intent on the model's own admitted guess would record a
    # referent the reasoning does not stand behind.
    current_subanswer_confidence: List[str]
    # What UPDATE surfaced into this hop's prompt via ``intermediate_answer``.
    # Recorded so the artifacts show whether a closed intent's binding actually
    # reached a generator, not merely that the intent closed.
    last_intermediate_answer: str
    # Subquestions dropped this hop for still referencing an unresolved earlier result.
    last_n_uninstantiated: int
    # This hop's consolidation evictions, as surfaced by ``MemoryUpdateGraph``.
    # ``reason == "contradicted"`` is the retraction signal ``plan_gate`` reads.
    last_retractions: List[Dict[str, Any]]
    # Groups of mutually-contradicting ``[Retrieval]`` lines the consolidator kept
    # instead of adjudicating. Distinct from a retraction: nothing was overturned,
    # the evidence simply disagrees with itself.
    last_unresolved_conflicts: List[List[str]]

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
    # Append-only across the whole run, including across replans. The per-intent
    # ``attempts`` in the ledger are rebuilt by ``replan`` and so cannot serve as the
    # negative record: after one replan the replanner would have no
    # ``already_attempted`` and could re-propose a framing that already yielded
    # nothing — the loop this record exists to prevent.
    plan_attempts_log: Annotated[List[Dict[str, Any]], operator.add]
    # Set by MCTS when it passes its own plan into a rollout: the rollout may read
    # and log but must not regenerate or revise the parent's plan.
    plan_frozen: bool
    # Confidence signal derived from unmet plan intents at synthesis time. Surfaced
    # in metadata and artifacts; deliberately never injected into a prompt.
    abstention: Dict[str, Any]

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

# Self-reported confidence labels that disqualify an answer from binding a
# referent. ``ANSWER_GENERATOR`` emits a free-text label, so match generously.
_LOW_CONFIDENCE = frozenset({"low", "very low", "uncertain", "unsupported", "none"})

PLAN_ACTION_NONE = "none"
PLAN_ACTION_UPDATE = "update"
PLAN_ACTION_REPLAN = "replan"

# Ledger intent statuses.
INTENT_OPEN = "open"
INTENT_CLOSED = "closed"
INTENT_CONTESTED = "contested"
# A contest that stopped moving: the rival referents have not changed for
# ``stall_after_attempts`` hops, so nothing further is going to separate them.
#
# ``contested`` is otherwise **absorbing**. ``apply_bindings`` only closes an intent at
# exactly one distinct referent and the rival set only ever grows, so the sole exit was a
# replan — and the shipped config sets ``replan_max: 0``. Measured on
# ``results/e3_plan_logonly``: 13 intents end still contested, against 1 in the armed runs
# where a replan cleared them. Those 13 kept drawing retrieval to no purpose.
#
# Deliberately NOT ``INTENT_DEAD``: ``abstention_signal`` opens with
# ``[e for e in ledger if e.get("status") != INTENT_DEAD]``, so marking a contest dead
# would delete the hedge on exactly the questions the system is least sure of. Undecided
# stays live and unmet, so it still hedges — it just stops being retried.
INTENT_UNDECIDED = "undecided"
# An intent a replan abandoned. Kept in the ledger rather than deleted so the
# rendered plan can show it as ruled out — a dropped intent is invisible to the
# generator, which is free to re-propose the very framing the replan discarded.
INTENT_DEAD = "dead"


# A subquestion the generator could not instantiate: it wanted a value from an
# earlier hop, had none, and emitted a bracketed placeholder in its place — e.g.
# "When was the album [album name from previous answer] released?". Observed live on
# a real dataset row, where that string went to retrieval verbatim as a query.
# Any bracketed span containing letters. Deliberately broader than a keyword list:
# the observed live case was "[album name from previous answer]", but a generator
# equally emits "[that album]", "[X]" or "[insert name]", none of which share a
# keyword. A genuine subquestion almost never contains square brackets, so the
# false-positive cost is low and the failure it prevents — retrieving on the
# placeholder text — is silent and expensive.
_UNRESOLVED_PLACEHOLDER = re.compile(r"\[[^\]]*[A-Za-z][^\]]*\]")


def is_uninstantiated(subquestion: str) -> bool:
    """True when a subquestion still references an unresolved earlier result.

    Retrieving on such a query is worse than not asking it: the placeholder text
    becomes part of the embedding/SPARQL query, so it returns noise, the intent
    closes on that noise, and the dependency is never actually resolved. Dropping it
    keeps the intent open so the next hop can ask it with the binding in hand.
    """
    return bool(_UNRESOLVED_PLACEHOLDER.search(subquestion or ""))


def _retrieval_lines(memory: Optional[Sequence[str]]) -> List[str]:
    """Lowercased text of the ``[Retrieval]``-tagged memory lines.

    Provenance is parsed here rather than imported from ``memory_update`` to keep
    the ledger helpers free of a graph dependency; the tag format is a stable
    contract (see ``_format_memory_item``).
    """
    out: List[str] = []
    for item in memory or []:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped.startswith("[hop="):
            end = stripped.find("]")
            if end != -1:
                stripped = stripped[end + 1 :].lstrip()
        if stripped.startswith("[Retrieval]"):
            out.append(stripped[len("[Retrieval]") :].lstrip(": ").strip().lower())
    return out


def _is_corroborated(surface: str, grounded_lines: Sequence[str]) -> bool:
    """True when ``surface`` appears in a ``[Retrieval]``-tagged memory line.

    Substring containment in either direction: the sub-answer is usually a short
    concise form ("In Utero") of a longer retrieved sentence, but can also be the
    longer prose. Deliberately crude — the question is only "did evidence mention
    this", and a stricter matcher would silently demote real corroboration.
    """
    key = (surface or "").strip().lower()
    if not key:
        return False
    return any(key in line or line in key for line in grounded_lines)


# PLANNER completions to sample. One, deliberately.
#
# This was 3, on the reasoning that a plan is cheap to explore and impossible to
# verify in advance, so breadth hedges against a bad decomposition. The samples do
# differ (mean pairwise similarity 0.57 over 124 measured pairs), but the argument
# does not survive contact with what the extra samples were *for*: the runners-up
# were only ever read by ``replan``, and a replan is conditioned on the current
# plan, current memory, and a stated failure. An alternative drawn at hop 0 —
# before any evidence existed — cannot know why the plan failed, so seeding a
# revision from it discards the only information the revision has, and can throw
# away bindings that closed intents already earned. Measured: the alternatives
# were consumed 0/62 times, at 3x planner tokens.
#
# If plan quality needs work, the cheaper and better-aimed tool is a *conditional*
# retry — reject and re-ask when the plan names a referent the question does not —
# not three blind draws.
N_PLANS = 1


def score_plan(
    plan: Any, question: str, retrieval_memory: Optional[Sequence[str]] = None
) -> tuple[int, int, int]:
    """Sort key for choosing among sampled plans. Higher is better.

    Deterministic and LLM-free — a judge call would cost as much as the sampling it
    is adjudicating. Ranks on three properties the design already requires:

    1. **fewest referent-discipline violations** — an entity the plan names that is
       in neither the question nor a ``[Retrieval]`` memory line is an originated
       claim, which is the one thing a plan must never contain.
    2. **fewest uninstantiated placeholders** — an intent written as
       ``"[the album]"`` cannot be asked.
    3. **most intents** — among equally clean plans, prefer the one that decomposes
       further; a one-step "answer the question" plan conditions nothing.
    """
    intents = [str(i) for i in (getattr(plan, "intents", None) or [])]
    text = " ".join([str(getattr(plan, "plan", "") or "")] + intents)
    allowed = (question or "").lower() + " " + " ".join(_retrieval_lines(retrieval_memory))
    # **Multiword** capitalised spans only. A single capitalised word is almost
    # always sentence-initial ("Identify…", "Find…") and swamped the real signal:
    # every plan scored the same until this was narrowed. "Charles Babbage" survives;
    # "Identify" does not.
    # No "." in the character class: it let a span run across a sentence boundary,
    # so "…Alan Turing. Find…" matched as the phantom entity "Turing. Find" and a
    # plan citing a properly verified name was penalised.
    named = re.findall(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+\b", text)
    violations = sum(1 for n in named if n.lower() not in allowed)
    placeholders = sum(1 for i in intents if is_uninstantiated(i))
    return (-violations, -placeholders, len(intents))


def select_plan(
    outputs: Any, question: str, retrieval_memory: Optional[Sequence[str]] = None
) -> tuple[Optional[Any], List[Any]]:
    """Pick one sampled plan; return ``(chosen, runners_up)``.

    The runners-up are kept as **replan seeds**: when the chosen plan fails, an
    alternative the model already produced is a cheaper and more diverse starting
    point than asking it to revise the one that just failed.
    """
    if not isinstance(outputs, list):
        outputs = [outputs]
    cands = [
        o
        for o in outputs
        if o is not None
        and not is_safe_default(o)
        and str(getattr(o, "plan", "") or "").strip()
    ]
    if not cands:
        return None, []
    ranked = sorted(
        cands, key=lambda o: score_plan(o, question, retrieval_memory), reverse=True
    )
    return ranked[0], ranked[1:]


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


def _is_matchable_label(label: str) -> bool:
    """Whether an entity label is specific enough to identify a referent by mention.

    A bare numeral is not. ``entity_dict`` routinely links the *year* ``"1"``, ``"3"``
    and so on as entities, and a plain substring scan then finds them inside any number:
    ``resolve_binding_qids("150 km/h (93 mph)", {"1": "Q199", "3": "Q201"})`` returned
    ``["Q199", "Q201"]`` — two referents from the digits of one speed. Since the contested
    test is "two or more distinct QIDs on one intent", that alone manufactured contests
    out of arithmetic. Sub-3-character labels go the same way ("US" inside "USSR").
    """
    stripped = label.strip()
    if len(stripped) < 3:
        return False
    return not stripped.isdigit()


def _find_whole(haystack: str, needle: str) -> int:
    """``haystack.find(needle)`` restricted to whole-token matches, else -1.

    ``.find`` alone matched "1" inside "150" and "born" inside "reborn". Boundaries are
    non-alphanumeric so multi-word labels and punctuation-adjacent mentions still match.
    """
    for m in re.finditer(re.escape(needle), haystack):
        before = haystack[m.start() - 1] if m.start() else " "
        after = haystack[m.end()] if m.end() < len(haystack) else " "
        if not before.isalnum() and not after.isalnum():
            return m.start()
    return -1


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
        if not label or not _is_matchable_label(label):
            continue
        start = _find_whole(haystack, label)
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


# A leading date, year, or quantity — the referent form an ordinal/temporal hop
# produces. Anchored at the start because a concise answer leads with its answer;
# matching anywhere would pick up dates mentioned in passing.
_LITERAL_PATTERNS = (
    # 13 September 1993 / September 13, 1993
    re.compile(
        r"^\W*((?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august"
        r"|september|october|november|december)\s+\d{1,2}?,?\s*\d{3,4})",
        re.I,
    ),
    # 1993-09-13
    re.compile(r"^\W*(\d{4}-\d{2}-\d{2})"),
    # 320 km/h, 310 square kilometers, 1045 BC. The unit must run to the end of the
    # clause: without that anchor the unit alternation swallows ordinary prose, so
    # "2012, per the tower's records" resolved to "2012 per the" and two paraphrases
    # of the same year would read as competing referents.
    re.compile(
        r"^\W*(\d[\d,.]*\s*(?:[a-z/²³]+(?:\s+[a-z]+)?|BCE?|CE|AD))\s*(?:[.,;:)\]]|$)",
        re.I,
    ),
    # a bare year
    re.compile(r"^\W*(\d{3,4})(?:\W|$)"),
)


def resolve_primary_literal(text: str) -> Optional[str]:
    """Normalized leading date/quantity in ``text``, or None.

    The QID discriminator cannot see an ordinal or temporal hop: "In what year was
    the tallest lattice tower completed?" binds a *year*, which links to no entity,
    so two competing years would both resolve to no QID and the intent would never
    be recorded as contested. Since the committed eval set is 56%
    ordinal-then-chained, that is a large blind spot rather than an edge case.

    Deliberately narrow: anchored at the start, and normalized only by case and
    punctuation. It exists to tell *two different literals apart*, not to parse
    dates — so "1993" and "September 13, 1993" are treated as different bindings,
    which is the conservative direction (it can over-report a contest, never
    silently merge two genuinely different answers).
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    for pattern in _LITERAL_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return re.sub(r"[\s,]+", " ", m.group(1)).strip().lower()
    return None


def resolve_binding_key(text: str, label_to_qid: Dict[str, str]) -> Optional[str]:
    """Identity of the referent ``text`` proposes: a QID, else a literal.

    QID first — an entity match is stronger evidence than a surface-form literal.
    Literals are prefixed so they can never collide with a QID.
    """
    qid = resolve_primary_qid(text, label_to_qid)
    if qid:
        return qid
    literal = resolve_primary_literal(text)
    return f"lit:{literal}" if literal else None


def classify_discharge(
    ledger: Sequence[Dict[str, Any]],
    *,
    exhausted: bool = False,
) -> tuple[str, Optional[int], List[str]]:
    """Decide ``plan_action`` from the ledger's current bindings.

    Returns ``(action, intent_index, competing_surfaces)``.

    * **contested** — an intent bound two or more *distinct QIDs*. Its closure is
      under-determined: this is the selection logic cannot supply, it is a fact
      about the plan's bookkeeping rather than about the world, and it needs no
      judge because QID identity is the whole discriminator.
    * **falsified** — an intent's ``premises`` intersect this hop's retractions
      (marked by :func:`apply_retractions`). Also a replan.
    * **stalled** — an intent queried ``max_attempts`` times without closing
      (marked by :func:`mark_stalled_intents`). An *efficacy* failure rather than a
      truth failure, and the only one of the three that fires when nothing
      surprising happens at all — without it a plan whose intents quietly return
      nothing is never revised.
    * **update** — exactly one distinct referent survives: the intent closes.
    * **none** — nothing resolved, so there is nothing to record.

    Precedence is contested > falsified > stalled, from most to least specific:
    discriminating between two referents subsumes re-establishing a premise, which
    subsumes finding a different route.
    """
    conflicted: Optional[int] = None
    falsified: Optional[int] = None
    stalled: Optional[int] = None
    for idx, entry in enumerate(ledger):
        if entry.get("status") == INTENT_CONTESTED:
            surfaces = [
                b.get("surface", "")
                for b in (entry.get("bindings") or [])
                if b.get("qid")
            ]
            return PLAN_ACTION_REPLAN, idx, surfaces
        if entry.get("conflicted") and conflicted is None:
            conflicted = idx
        if entry.get("falsified") and falsified is None:
            falsified = idx
        if entry.get("stalled") and stalled is None:
            stalled = idx
    if conflicted is not None:
        return (
            PLAN_ACTION_REPLAN,
            conflicted,
            list(ledger[conflicted].get("conflicted") or []),
        )
    if falsified is not None:
        return PLAN_ACTION_REPLAN, falsified, []
    if stalled is not None:
        return PLAN_ACTION_REPLAN, stalled, []
    # Exhaustion: every intent is settled yet the question is still not answerable,
    # so the plan was *insufficient* rather than wrong. Lowest precedence — any
    # specific failure above is a better description of what to fix.
    if exhausted and ledger and all(
        e.get("status") in (INTENT_CLOSED, INTENT_DEAD) for e in ledger
    ):
        return PLAN_ACTION_REPLAN, 0, []
    for idx, entry in enumerate(ledger):
        if entry.get("status") == INTENT_CLOSED and entry.get("closed_at") is not None:
            return PLAN_ACTION_UPDATE, idx, []
    return PLAN_ACTION_NONE, None, []


def mark_conflicted_intents(
    ledger: Sequence[Dict[str, Any]], conflicts: Sequence[Sequence[str]]
) -> List[Dict[str, Any]]:
    """Flag intents whose own evidence disagrees with itself.

    Consolidator rule 6 keeps two mutually-contradicting ``[Retrieval]`` items
    rather than adjudicating. That is a different failure from a retraction: nothing
    was overturned, so re-asking the intent in any wording returns the same two
    sources. The repair is a step that *discriminates between the sources*, which
    only a replan can add.

    A conflict group is attributed to an intent when the intent cites one side as a
    premise or bound one side as its referent. An unattributable conflict is recorded
    on the first still-open intent, because the plan as a whole is what has to
    adjudicate it.
    """
    out = [dict(entry) for entry in ledger]
    groups = [
        [str(c).strip().lower() for c in group if str(c or "").strip()]
        for group in (conflicts or [])
    ]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return out

    def _touches(entry: Dict[str, Any], group: Sequence[str]) -> bool:
        texts = [str(p or "").lower() for p in (entry.get("premises") or [])]
        texts += [
            str(b.get("surface") or "").lower() for b in (entry.get("bindings") or [])
        ]
        return any(
            side and t and (side in t or t in side) for side in group for t in texts
        )

    for group in groups:
        target = next(
            (i for i, e in enumerate(out) if _touches(e, group)),
            None,
        )
        if target is None:
            target = next(
                (i for i, e in enumerate(out) if e.get("status") == INTENT_OPEN), None
            )
        if target is None:
            continue
        out[target]["conflicted"] = list(group)
    return out


def mark_stalled_intents(
    ledger: Sequence[Dict[str, Any]], *, max_attempts: int
) -> List[Dict[str, Any]]:
    """Flag intents that have been queried ``max_attempts`` times and still not closed.

    This is the **efficacy** failure, not a truth failure: the intent may be
    perfectly well-formed while this framing of it keeps returning nothing. Without
    it the trigger only ever fires on ambiguity or retraction, and an intent that
    silently yields nothing for the whole run produces no signal at all — the plan
    is simply never revised.

    Records ``stall_reason`` so the replanner can be told which of the two it is:
    attempts that returned evidence but never resolved a referent need a
    *different question*; attempts that returned no evidence need a *different
    route*.
    """
    out = [dict(entry) for entry in ledger]
    for entry in out:
        if entry.get("status") == INTENT_CLOSED:
            continue
        attempts = list(entry.get("attempts") or [])
        # A contest that survived a *further hop* of evidence is not going to resolve:
        # retire it to ``undecided`` so it stops drawing retrieval, while staying live
        # for ``abstention_signal`` to hedge on.
        #
        # Distinct hops, not attempt count. The per-intent pooling cap allows 2
        # subquestions on one intent in a single hop, so an attempt-count test fires on
        # the very hop that detected the contest — before any new evidence could have
        # separated the rivals — and reports the intent as stalled rather than contested,
        # losing the competing surfaces the repair needs.
        #
        # Evaluated BEFORE the ``stalled`` early-return below, and not gated on it. An
        # intent commonly stalls while still merely open and only becomes contested on a
        # later hop; when the early-return covered this branch, that ordering meant it
        # could never retire. Measured on ``results/e4_plan_fixed``: 8 of the 13 intents
        # left contested had attempts across 3-4 distinct hops and still did not retire,
        # because ``stalled`` had already been set.
        if (
            entry.get("status") == INTENT_CONTESTED
            and len(attempts) >= max_attempts
            and len({a.get("hop") for a in attempts}) >= 2
        ):
            entry["status"] = INTENT_UNDECIDED
            entry["undecided_rivals"] = [
                str(b.get("surface") or "") for b in (entry.get("bindings") or [])
            ]
        if entry.get("stalled") or len(attempts) < max_attempts:
            continue
        yielded = sum(int(a.get("n_facts") or 0) for a in attempts)
        entry["stalled"] = True
        entry["stall_reason"] = (
            "no evidence returned across %d attempts — the route is unproductive"
            % len(attempts)
            if yielded == 0
            else "evidence returned across %d attempts but no referent resolved — "
            "the question may presuppose something false" % len(attempts)
        )
    return out


def _describe_failure(
    entry: Dict[str, Any], competing: Sequence[str]
) -> str:
    """State mechanically what went wrong, and prescribe the matching repair.

    Three failure kinds need three *different* repairs, and conflating them is how
    a replan loops:

    * **revision** (contested) — the question was fine, two referents answered it.
      Discriminate between them.
    * **revision** (falsified) — a premise was overturned. Re-establish it.
    * **contraction** (stalled) — the intent itself may be malformed: its
      presupposition may have no referent. Re-asking it in any wording will keep
      failing, so the instruction is to establish what *is* the case instead. This
      is the case the committed eval set is full of (30/62 rows rest on a definite
      description that can fail), and the one a plain tail-revision cannot fix.

    No diagnostic LLM sits in between: the planner is a model and can infer the
    cause from these signals, so an extra call would add cost and a place to
    confabulate a narrative.
    """
    intent = entry.get("intent")
    if entry.get("status") == INTENT_CONTESTED:
        return (
            f"The intent {intent!r} bound {len(competing)} different referents, so "
            "its closure is under-determined. Do not re-ask it as written — add a "
            "step that discriminates between the candidates (a distinguishing "
            "attribute, a date, or the criterion that decides the ranking)."
        )
    if entry.get("conflicted"):
        sides = list(entry.get("conflicted") or [])
        return (
            f"The evidence for {intent!r} disagrees with itself: retrieval returned "
            f"{len(sides)} mutually-contradicting sources that consolidation kept "
            "rather than adjudicating. Re-asking it returns the same two sources, so "
            "add a step that DISCRIMINATES BETWEEN THE SOURCES — which is more "
            "authoritative, more recent, or scoped to what the question asks."
        )
    if entry.get("falsified"):
        return (
            f"A fact this plan relied on was contradicted by retrieved evidence: "
            f"{entry.get('falsified')!r}. Re-establish it, and revise anything the "
            "plan built on top of it."
        )
    if entry.get("stalled"):
        return (
            f"The intent {intent!r} did not resolve: "
            f"{entry.get('stall_reason') or 'repeated attempts failed'}. Treat its "
            "presupposition as suspect — the thing it asks about may not exist, may "
            "not be unique, or may depend on a date or authority the question does "
            "not fix. Replace it with an intent that establishes what IS the case "
            "(e.g. whether such a thing exists at all, or which criterion decides "
            "it), rather than re-asking the same question in different words."
        )
    if entry.get("status") == INTENT_CLOSED:
        return (
            "Every intent in this plan is settled, yet the question still cannot be "
            "answered from what was found — so the plan was INSUFFICIENT, not wrong. "
            "Keep what is resolved and ADD the intents that are still missing; do "
            "not restate the ones already answered."
        )
    return (
        f"The intent {intent!r} could not be closed. Find a different route to the "
        "same information."
    )


def abstention_signal(ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How much the answer should hedge, given what the plan never established.

    An unmet intent is genuinely informative at answer time — but as a *confidence*
    signal, not an adjudication input. It never enters ``FINAL_ANSWER_SYNTHESIZER``'s
    prompt: there the only question is which candidate is true, and an unmet intent
    sitting beside candidate answers invites treating a correct answer as deficient.
    So it is computed here and surfaced in metadata/artifacts instead, where a caller
    can act on it.

    ``level`` is ordinal, not a probability: ``none`` (everything resolved) →
    ``low`` → ``high`` (nothing resolved, or a premise was contradicted).
    """
    live = [e for e in ledger if e.get("status") != INTENT_DEAD]
    if not live:
        return {"level": "none", "unmet": [], "resolved": 0, "total": 0, "reasons": []}
    resolved = [e for e in live if e.get("status") == INTENT_CLOSED and not e.get("falsified")]
    unmet = [e for e in live if e not in resolved]
    reasons = sorted(
        {
            "contested"
            if e.get("status") == INTENT_CONTESTED
            else "referent_ambiguous"
            if e.get("status") == INTENT_UNDECIDED
            else "sources_disagree"
            if e.get("conflicted")
            else "premise_contradicted"
            if e.get("falsified")
            else "stuck"
            if e.get("stalled")
            else "never_attempted"
            for e in unmet
        }
    )
    if not unmet:
        level = "none"
    elif len(resolved) == 0 or "premise_contradicted" in reasons:
        level = "high"
    else:
        level = "low"
    return {
        "level": level,
        "unmet": [str(e.get("intent") or "") for e in unmet],
        "resolved": len(resolved),
        "total": len(live),
        "reasons": reasons,
    }


def _dead_reason(entry: Dict[str, Any]) -> str:
    """Why an abandoned intent was dropped, phrased so the generator can use it."""
    if entry.get("status") == INTENT_CONTESTED:
        return "it bound two different referents and was replaced"
    if entry.get("conflicted"):
        return "its sources contradicted each other"
    if entry.get("falsified"):
        return "the premise it rested on was contradicted by evidence"
    if entry.get("stalled"):
        return entry.get("stall_reason") or "repeated attempts did not resolve it"
    return "superseded by a revised plan"


def _discharge_reason(
    ledger: Sequence[Dict[str, Any]], intent_idx: Optional[int], action: str
) -> str:
    """Name the branch that fired, for the fire-rate breakdown.

    Shared by both graphs' gates: duplicating the precedence chain let MCTS drift
    into reporting ``reason=None`` once already.
    """
    if intent_idx is None or not (0 <= intent_idx < len(ledger)):
        return action
    entry = ledger[intent_idx]
    if entry.get("status") == INTENT_CONTESTED:
        return "contested"
    if entry.get("conflicted"):
        return "conflicted"
    if entry.get("falsified"):
        return "falsified"
    if entry.get("stalled"):
        return "stalled"
    if action == PLAN_ACTION_REPLAN and entry.get("status") == INTENT_CLOSED:
        return "exhausted"
    return action


def apply_bindings(
    ledger: Sequence[Dict[str, Any]],
    candidates: Sequence[tuple[Optional[int], str]],
    label_to_qid: Dict[str, str],
    hop: int,
    retrieval_memory: Optional[Sequence[str]] = None,
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

    ``retrieval_memory`` is the ``[Retrieval]``-tagged memory lines. A binding whose
    surface is corroborated by one of them is marked ``grounded``, and only grounded
    bindings are shown in the rendered plan — the plan may *cite* a fact that came
    through the verified door, but must not present the model's own unverified
    inference as established.
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
    grounded_lines = _retrieval_lines(retrieval_memory)
    for intent_idx, text in candidates:
        if intent_idx is None or not (0 <= intent_idx < len(out)):
            continue
        if intent_idx in already_closed:
            continue
        entry = out[intent_idx]
        key = resolve_binding_key(text, label_to_qid)
        if key and not any(b.get("qid") == key for b in entry["bindings"]):
            entry["bindings"].append(
                {
                    "surface": text,
                    "qid": key,
                    "hop": hop,
                    "grounded": _is_corroborated(text, grounded_lines),
                }
            )
        distinct = count_rival_referents(entry["bindings"])
        if len(distinct) >= 2:
            entry["status"] = INTENT_CONTESTED
            entry["closed_at"] = None
        elif len(distinct) == 1:
            entry["status"] = INTENT_CLOSED
            entry["closed_at"] = hop
    return out


# A surface naming two or more linked entities, or shaped like a list, is an
# *enumeration* — "Congo, Yangtze, Danube, Zambezi and Hudson", "1. Burj Khalifa, 2. Tokyo
# Skytree", "Max Born and Walther Bothe". It must never absorb a rival by containment: it
# contains the rival because it lists it, and merging would silently pick one element of a
# set the question may be asking to rank.
_ENUMERATION_SHAPE = re.compile(
    r"(?:;)|(?:\b\d+\.\s+\w)|(?:\b[A-Z][\w'-]+\s+and\s+[A-Z][\w'-]+)"
)


def _is_enumeration(surface: str) -> bool:
    s = (surface or "").strip()
    return bool(_ENUMERATION_SHAPE.search(s)) or s.count(",") >= 2


def count_rival_referents(bindings: Sequence[Dict[str, Any]]) -> Set[str]:
    """Distinct QIDs on an intent, after merging surfaces that name the same referent.

    The contested test is "two or more distinct referents survive for one intent", and it
    was reading two *descriptions of one answer* as a contest. Measured on the shipped
    config (``results/e3_plan_logonly``), 17 of 38 contested fires had a numeral or
    sub-3-character rival, and the samples are unambiguous::

        ['The Danyang-Kunshan Grand Bridge was opened in 2011.', '2011']  # one fact, twice
        ['2012', '29 February 2012']                                     # one date, two granularities
        ['2009', '1989']                                                  # a genuine contest

    So when one surface is contained in another, they are treated as one referent and the
    longer, more specific one wins — **unless** the container is an enumeration
    (:func:`_is_enumeration`) or itself names two or more linked entities. Those keep both,
    because a list containing a candidate is evidence about a *set*, not a restatement of
    one member.

    Join-never-split: merging is only ever applied to collapse, never to invent a rival, so
    the worst case is a missed contest rather than a manufactured one — and a manufactured
    contest is the expensive error (it blocks closure and, with ``replan_max=0``, blocks it
    permanently).
    """
    keyed = [b for b in bindings if b.get("qid")]
    absorbed: Set[str] = set()
    for a in keyed:
        for b in keyed:
            if a is b or a.get("qid") == b.get("qid"):
                continue
            longer, shorter = str(b.get("surface") or ""), str(a.get("surface") or "")
            if len(longer) <= len(shorter):
                continue
            if shorter.strip().lower() not in longer.strip().lower():
                continue
            if _is_enumeration(longer):
                continue
            absorbed.add(str(a.get("qid")))
            logger.debug(
                "[bindings] %r absorbed by %r — one referent, not a contest",
                shorter[:60],
                longer[:60],
            )
    return {str(b["qid"]) for b in keyed if str(b["qid"]) not in absorbed}


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

    An **ungrounded** binding — one closed on the model's own inference rather than
    corroborated by a ``[Retrieval]`` line — is marked, not withheld.
    ``render_plan_for_prompt`` shows such an intent as ``[resolved, unverified]`` and
    hides its value, but this function had no ``grounded`` check at all, so the same
    binding the plan refused to state was broadcast here as the anchor for the next hop:
    15/39 hops in ``e1_cot_armed``, 18/44 in ``e2``, 4/43 in ``e3_plan_logonly``.
    Withholding it instead would be worse — this slot is the chaining anchor that
    ``roles.py``'s "do NOT re-ask what was already resolved" rule depends on, and
    suppressing it strands the next hop with no referent. So the value still goes out,
    labelled for what it is.
    """
    best: Optional[tuple[int, str, str, bool]] = None
    for entry in ledger:
        if entry.get("status") != INTENT_CLOSED:
            continue
        closed_at = entry.get("closed_at")
        if closed_at is None:
            continue
        binding = next(
            (b for b in (entry.get("bindings") or []) if b.get("qid")),
            None,
        )
        if binding is None or not binding.get("surface"):
            continue
        if best is None or int(closed_at) >= best[0]:
            best = (
                int(closed_at),
                str(entry.get("intent", "")),
                str(binding.get("surface")),
                bool(binding.get("grounded")),
            )
    if best is None:
        return None
    _, intent, surface, grounded = best
    value = surface if grounded else f"(unverified) {surface}"
    return f"{intent}\n→ {value}" if intent else value


# Similarity above which a fresh intent is treated as a restatement of an already
# closed one. Low, because the observed restatements sit at 0.63-0.84 ("determine the
# opening date of the longest bridge in the world" vs "determine the date on which the
# identified bridge was opened"). Safe at that level only because ``_fuzzy_closed_match``
# requires the prior closure to carry a retrieval-grounded binding — the worst case is
# closing an intent whose referent is already corroborated, against the alternative of
# paying for a whole extra hop to re-derive it.
_RESTATED_INTENT_THRESHOLD = 0.60


def _resolved_intent_lines(ledger: Sequence[Dict[str, Any]]) -> List[str]:
    """``intent -> bound referent`` for every settled intent, for the replan prompt.

    Only grounded bindings show their value, matching ``render_plan_for_prompt``: an
    intent closed on the model's own inference is reported as settled without
    asserting the referent it guessed.
    """
    lines: List[str] = []
    for entry in ledger:
        if entry.get("status") != INTENT_CLOSED or entry.get("falsified"):
            continue
        intent = (entry.get("intent") or "").strip()
        if not intent:
            continue
        bound = next(
            (
                b
                for b in (entry.get("bindings") or [])
                if b.get("qid") and b.get("grounded")
            ),
            None,
        )
        lines.append(
            f"{intent} -> {bound['surface']}" if bound else f"{intent} -> settled"
        )
    return lines


def _fuzzy_closed_match(
    intent: Optional[str],
    closed_by_intent: Dict[str, Dict[str, Any]],
    already_matched: Set[str],
) -> Optional[Dict[str, Any]]:
    """Best unclaimed closed intent that ``intent`` restates, or None.

    Only closures with a retrieval-grounded binding are eligible — see
    ``_RESTATED_INTENT_THRESHOLD`` for why that restriction is what makes a threshold
    this low safe.
    """
    key = _normalize_subq(intent or "")
    if not key:
        return None
    best: Optional[Dict[str, Any]] = None
    best_ratio = _RESTATED_INTENT_THRESHOLD
    for prior_key, prior in closed_by_intent.items():
        if prior_key in already_matched:
            continue
        if not any(
            b.get("qid") and b.get("grounded") for b in (prior.get("bindings") or [])
        ):
            continue
        ratio = difflib.SequenceMatcher(None, key, _normalize_subq(prior_key)).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best = prior
    if best is not None:
        logger.info(
            "[replan] fresh intent %r restates the resolved intent %r (sim %.2f); "
            "carrying its binding instead of re-opening it",
            (intent or "")[:70],
            (best.get("intent") or "")[:70],
            best_ratio,
        )
    return best



def render_plan_for_prompt(plan: str, ledger: Sequence[Dict[str, Any]]) -> str:
    """Annotate the prose plan with per-intent status and its resolved referent.

    The stored ``plan`` prose is never mutated — this view is **derived from the
    ledger on every call**, which is what makes it safe to include bound values.
    A binding here carries a QID, a hop, and an eviction path: a retracted premise
    marks the intent ``falsified``, and this function checks that flag *before*
    ``status`` so the value stops being shown from the next render onward. Writing
    the value into the plan text instead would strand a world-claim with none of
    those — nothing would ever un-assert it.

    Only ``grounded`` bindings — those corroborated by a ``[Retrieval]``-tagged
    memory line — are shown as resolved values. An intent closed on the model's own
    unverified inference renders as ``[resolved, unverified]`` without its value, so
    the plan never presents a guess as established.
    """
    text = (plan or "").strip()
    if not ledger:
        return text
    lines = []
    for entry in ledger:
        status = entry.get("status", INTENT_OPEN)
        intent = entry.get("intent", "")
        if entry.get("falsified"):
            # Checked BEFORE ``closed``: ``apply_retractions`` flags the intent but
            # deliberately leaves its status and bindings alone (the retraction was of
            # a *premise*, and the classifier still needs the binding history). If the
            # render trusted ``status`` first, a closed intent whose premise retrieval
            # just contradicted would keep asserting its value — the plan restating a
            # fact the evidence has overturned.
            lines.append(f"- [premise contradicted - re-establish] {intent}")
        elif status == INTENT_CLOSED:
            bound = next(
                (
                    b
                    for b in (entry.get("bindings") or [])
                    if b.get("qid") and b.get("grounded")
                ),
                None,
            )
            if bound is not None:
                lines.append(f"- [resolved: {bound['surface']}] {intent}")
            else:
                lines.append(f"- [resolved, unverified] {intent}")
        elif status == INTENT_CONTESTED:
            # Name the rivals. Told only that an intent is "ambiguous", the generator
            # re-issued the same query — 13 of 17 cross-hop repeats were contested
            # intents. Naming them makes a *discriminating* question writable.
            rivals = [
                str(b.get("surface") or "")[:60]
                for b in (entry.get("bindings") or [])
                if b.get("qid")
            ]
            lines.append(
                f"- [ambiguous between {' | '.join(rivals)} - ask what tells them "
                f"apart, do NOT re-ask this] {intent}"
                if len(rivals) >= 2
                else f"- [ambiguous - two candidate referents] {intent}"
            )
        elif status == INTENT_UNDECIDED:
            lines.append(
                f"- [UNRESOLVABLE - stop asking; the answer must hedge] {intent}"
            )
        elif status == INTENT_DEAD:
            why = entry.get("dead_reason") or "abandoned by a replan"
            lines.append(f"- [RULED OUT - do not re-propose: {why}] {intent}")
        elif entry.get("conflicted"):
            lines.append(f"- [sources disagree - needs adjudicating] {intent}")
        elif entry.get("stalled"):
            lines.append(f"- [stuck - this framing is not resolving it] {intent}")
        else:
            lines.append(f"- [open] {intent}")
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
    # Subquestions dropped because they still referenced an unresolved earlier
    # result. A non-zero count means the plan has a real dependency the generator
    # could not instantiate, which is precisely when ``intermediate_answer`` matters.
    n_uninstantiated: int = 0
    # Dropped as a reworded twin of one already pooled (see ``_NEAR_DUP_THRESHOLD``).
    n_near_duplicate: int = 0
    # Dropped because their plan intent had already reached ``_MAX_PER_INTENT``
    # attempts this hop.
    n_intent_capped: int = 0


def _normalize_subq(text: str) -> str:
    """Casing/punctuation/whitespace-insensitive form, for twin detection only."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


# Two subquestions above this similarity retrieve the same documents. Measured on
# the d1 run: reworded twins that differ only in tense or word order score 0.92-0.97
# and return *identical* fact counts, while subquestions that genuinely ask for
# different properties of one entity ("the height of X" vs "the completion year of
# X") score 0.84. The threshold sits above the second class deliberately — dropping
# a real question costs a whole extra hop, dropping a twin costs nothing.
_NEAR_DUP_THRESHOLD = 0.95

# Retrievals allowed per plan intent per hop. Two, not one, because an ordinal
# intent has two genuinely different retrieval targets — the specific item and the
# complete ranked list. Anything beyond that was, in every case measured, the same
# question reworded.
_MAX_PER_INTENT = 2


def pool_subquestions(outputs: Any) -> PooledSubquestions:
    """Pool ``n`` SUBQUESTION_GENERATOR completions into one decomposition.

    Only completions that judged the question *not* answerable contribute their
    subquestions; ``should_direct`` is the majority ``is_answerable`` vote over
    the completions that parsed. Preserves first-seen order and carries the
    parallel ``needs_kg`` / ``serves_intent`` arrays alongside.

    Two caps bound the union, because each surviving subquestion buys a full
    retrieval fan-out and the ``n`` completions cannot see each other's output —
    so no prompt rule can stop them proposing the same thing:

    * **near-duplicate** — a reworded twin of something already pooled.
    * **per-intent** — more than ``_MAX_PER_INTENT`` subquestions on one plan
      intent in one hop.

    Both are backstops, not the mechanism: the prompt is what should keep breadth
    pointed at *distinct intents*. They exist because a single over-eager prompt
    rule once turned 3 completions into 10 retrievals for one intent, and a cap
    cannot regress the way prose can.
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
    norms: List[str] = []
    per_intent: Dict[int, int] = {}
    answerable = 0
    uninstantiated = 0
    near_duplicate = 0
    intent_capped = 0
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
            if is_uninstantiated(sq):
                # The generator wanted an earlier hop's value and had none. Let the
                # intent stay open rather than retrieving on the placeholder text.
                uninstantiated += 1
                continue
            raw_intent = intent_list[i] if i < len(intent_list) else None
            # -1 (or any negative) is the generator's "advances no plan intent"
            # encoding; normalize it to None so attribution stays absent rather
            # than pointing at a bogus ledger slot.
            intent = (
                int(raw_intent)
                if isinstance(raw_intent, int) and not isinstance(raw_intent, bool)
                and raw_intent >= 0
                else None
            )
            norm = _normalize_subq(sq)
            twin = next(
                (
                    prev
                    for prev in norms
                    if difflib.SequenceMatcher(None, norm, prev).ratio()
                    >= _NEAR_DUP_THRESHOLD
                ),
                None,
            )
            if twin is not None:
                near_duplicate += 1
                logger.debug(
                    "[pool] dropped reworded twin: %r (≈ %r)", sq.strip(), twin
                )
                continue
            # Unattributed subquestions are NOT capped as a group: they carry no
            # intent, so a shared budget would make two unrelated gaps compete.
            if intent is not None and per_intent.get(intent, 0) >= _MAX_PER_INTENT:
                intent_capped += 1
                logger.debug(
                    "[pool] intent %d already has %d subquestions this hop; "
                    "dropping %r",
                    intent,
                    _MAX_PER_INTENT,
                    sq.strip(),
                )
                continue
            seen.add(key)
            norms.append(norm)
            if intent is not None:
                per_intent[intent] = per_intent.get(intent, 0) + 1
            subqs.append(sq.strip())
            flags.append(bool(kg_list[i]) if i < len(kg_list) else True)
            intents.append(intent)
    if near_duplicate or intent_capped:
        logger.info(
            "[pool] %d subquestions kept; dropped %d reworded twins and %d over the "
            "per-intent cap",
            len(subqs),
            near_duplicate,
            intent_capped,
        )
    should_direct = (answerable / len(survivors) > 0.5) if survivors else False
    return PooledSubquestions(
        subquestions=subqs,
        needs_kg=flags,
        serves_intent=intents,
        should_direct=should_direct,
        n_survivors=len(survivors),
        n_uninstantiated=uninstantiated,
        n_near_duplicate=near_duplicate,
        n_intent_capped=intent_capped,
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
    stall_after_attempts = max(
        1, int(getattr(plan_cfg, "stall_after_attempts", 2) or 2)
    )

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
        question = state.get("question", "")
        memory = list(state.get("text_memory") or [])
        outs, _ = await execute_role_lc(
            registry,
            PLANNER,
            PlanInput(question=question, context=_join_memory_context(state)),
            n=N_PLANS,
        )
        # ``select_plan`` is retained at N_PLANS=1 so raising the sample count stays a
        # one-constant change; with one completion it reduces to "take the one that
        # parsed", and the runners-up list is empty.
        chosen, _runners_up = select_plan(outs, question, memory)
        if chosen is None:
            logger.error(
                "[gen_plan] no PLANNER completion parsed; continuing without a plan "
                "(the loop degrades to plain decomposition)"
            )
            return {"plan": "", "plan_version": 0, "plan_ledger": []}
        plan_text = str(getattr(chosen, "plan", "") or "").strip()
        intents = list(getattr(chosen, "intents", None) or [])
        premises = list(getattr(chosen, "premises", None) or [])
        logger.info("[gen_plan] plan with %d intents", len(intents))
        return {
            "plan": plan_text,
            "plan_version": 1,
            "plan_ledger": build_plan_ledger(intents, premises),
            "plan_action": PLAN_ACTION_NONE,
        }

    async def gen_subq(state: CoTState) -> Dict[str, Any]:
        question = state.get("question", "")
        ctx = _join_memory_context(state)
        # Capture what UPDATE actually surfaced this hop. Without recording it, the
        # artifacts show that an intent closed but not whether its binding ever
        # reached a prompt — which is the only thing that makes UPDATE do any work.
        surfaced = latest_intermediate_answer(state.get("plan_ledger") or [])
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
            intermediate_answer=surfaced,
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
                "last_intermediate_answer": surfaced or "",
            }
        return {
            # "answerable" routes the loop to synthesis, same as the majority vote.
            "is_answerable": pooled.should_direct or not pooled.subquestions,
            "subquestions": pooled.subquestions,
            "subquestion_needs_kg": pooled.needs_kg,
            "subquestion_serves_intent": pooled.serves_intent,
            "subq_parse_failed": False,
            "last_intermediate_answer": surfaced or "",
            # Diagnostic: how many subquestions were dropped as un-instantiated, and
            # how wide the pooled decomposition ended up. Retrieval breadth per hop is
            # what the plan was measured to have collapsed (49% of hops fell to a
            # single subquestion), so both numbers have to be visible per hop.
            "last_n_uninstantiated": pooled.n_uninstantiated,
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
        confidence: List[str] = []
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
            confidence.append(
                str(getattr(r, "confidence_level", None) or "").strip().lower()
            )
        return {
            "current_subanswers": answers,
            "current_subanswers_concise": concise,
            "current_subanswer_confidence": confidence,
        }

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
            # Rule 6 of the consolidator keeps two conflicting [Retrieval] items
            # rather than adjudicating between them. That is a *second*, distinct
            # signal from a retraction: the evidence base disagrees with itself, so
            # no amount of re-asking the current intent resolves it — the plan needs
            # a step that discriminates between the two sources.
            "last_unresolved_conflicts": list(result.get("unresolved_conflicts") or []),
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
        confidences = list(state.get("current_subanswer_confidence") or [])
        candidates: List[tuple[Optional[int], str]] = []
        unattributed = 0
        low_confidence = 0
        for i, answer in enumerate(answers):
            if not (isinstance(answer, str) and answer.strip()):
                continue
            # A "low" self-reported confidence means the answerer is guessing.
            # Closing an intent on that would record a referent the reasoning does
            # not stand behind — and worse, a guess competing with a grounded answer
            # would read as genuine ambiguity. Let the intent stay open and stall.
            if i < len(confidences) and confidences[i] in _LOW_CONFIDENCE:
                low_confidence += 1
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

        # Pass the consolidated memory so a binding corroborated by a [Retrieval]
        # line is marked grounded — that is what licenses showing its value in the
        # rendered plan (a cited verified fact, not an originated claim).
        ledger = apply_bindings(
            ledger, candidates, label_to_qid, hop, state.get("text_memory") or []
        )
        ledger = apply_retractions(ledger, state.get("last_retractions") or [])
        ledger = mark_conflicted_intents(
            ledger, state.get("last_unresolved_conflicts") or []
        )
        # Negative record: what was asked and what it yielded. Memory cannot hold
        # this — "nothing was found for X" is not a fact about the world — and
        # without it a replanner given only (plan, memory) rewrites the same plan.
        n_facts = len(state.get("extracted_facts") or [])
        attempts_this_hop: List[Dict[str, Any]] = []
        for i, subq in enumerate(state.get("subquestions") or []):
            record = {"query": subq, "n_facts": n_facts, "hop": hop}
            # Every attempt goes to the run-level log, attributed or not: the
            # replanner needs the full "already tried" set, and an unattributed
            # query still tells it that framing yielded nothing.
            attempts_this_hop.append(record)
            intent_idx = intents[i] if i < len(intents) else None
            if intent_idx is None or not (0 <= intent_idx < len(ledger)):
                continue
            ledger[intent_idx]["attempts"] = list(
                ledger[intent_idx].get("attempts") or []
            ) + [record]

        # Efficacy check: an intent queried repeatedly without closing is stuck,
        # even when nothing surprising happened. This is the only branch that fires
        # on a plan that is quietly getting nowhere.
        ledger = mark_stalled_intents(ledger, max_attempts=stall_after_attempts)

        # Exhaustion needs care: ``is_answerable`` was judged by ``gen_subq`` at the
        # TOP of this hop, i.e. *before* this hop's retrieval. Reading it here alone
        # fired on rows where the plan had just closed every intent with fresh
        # evidence — the next ``gen_subq`` had not yet had a chance to see it. So also
        # require that nothing closed this hop: if progress was made, let the loop
        # re-judge answerability with the new memory before declaring the plan spent.
        progress_this_hop = any(
            e.get("status") == INTENT_CLOSED and e.get("closed_at") == hop
            for e in ledger
        )
        exhausted = not bool(state.get("is_answerable")) and not progress_this_hop
        action, intent_idx, competing = classify_discharge(ledger, exhausted=exhausted)
        entry = {
            "hop": hop,
            "action": action,
            "intent_index": intent_idx,
            "intent": ledger[intent_idx].get("intent") if intent_idx is not None else None,
            "competing_bindings": competing,
            "plan_version": int(state.get("plan_version", 0) or 0),
            # Attribution health, recorded per hop rather than only logged: if this
            # rate is high the gate is reading noise, and that has to be visible in
            # the fire-rate experiment rather than inferred from stderr.
            "answers_seen": len(candidates),
            "answers_unattributed": unattributed,
            "answers_low_confidence": low_confidence,
            # Which branch fired, so the fire-rate breakdown distinguishes ambiguity
            # from retraction from a stuck route.
            "reason": _discharge_reason(ledger, intent_idx, action),
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
            "plan_attempts_log": attempts_this_hop,
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
        failure = _describe_failure(entry, competing)
        # Read the run-level log, not the ledger: ``build_plan_ledger`` below starts
        # every intent with an empty ``attempts``, so a ledger-sourced list would be
        # empty on the second and later replans.
        attempts = [
            f"{a.get('query')} → {a.get('n_facts')} facts"
            for a in (state.get("plan_attempts_log") or [])
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
                # Settled work, named explicitly so the rewrite leaves it alone.
                # Without this the planner re-lists resolved intents in fresh wording
                # and the system pays to answer them again.
                resolved=_resolved_intent_lines(ledger) or None,
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
        # Carry forward closures the rewrite restated, so a replan does not re-open
        # work that is already done.
        closed_by_intent = {
            (e.get("intent") or "").strip().lower(): e
            for e in ledger
            if e.get("status") == INTENT_CLOSED
        }
        matched: set[str] = set()
        for fresh in new_ledger:
            key = (fresh.get("intent") or "").strip().lower()
            prior = closed_by_intent.get(key)
            if prior is None:
                # Exact text is not enough. The planner rewords, and a reworded
                # resolved intent used to be carried as closed AND re-listed here as
                # fresh and open — so it was asked a second time. Measured on the
                # armed 62-row run: 16 such pairs over 10 of the 20 replanned
                # questions, intents per question doubling with each replan
                # (2.17 -> 4.21 -> 8.17) and the closure rate falling to 32%.
                #
                # The prompt now names the settled intents and forbids re-listing
                # them, but the merge must not depend on the model obeying, so match
                # fuzzily as a backstop. Restricted to closures carrying a *grounded*
                # binding: a false positive then closes an intent whose referent
                # retrieval has already corroborated, which is far cheaper than
                # re-running a hop. Unverified closures still require exact text.
                prior = _fuzzy_closed_match(fresh.get("intent"), closed_by_intent, matched)
            if prior is not None:
                matched.add((prior.get("intent") or "").strip().lower())
                fresh["status"] = INTENT_CLOSED
                fresh["bindings"] = list(prior.get("bindings") or [])
                fresh["closed_at"] = prior.get("closed_at")
                fresh["attempts"] = list(prior.get("attempts") or [])
        # A resolved intent the planner reworded would otherwise vanish: its binding
        # would drop out of the rendered plan and ``intermediate_answer`` would fall
        # back to None, losing a referent that retrieval had already established.
        # Keep it as a carried entry — closed, so it is never re-asked or stalled.
        for key, prior in closed_by_intent.items():
            if key in matched:
                continue
            carried = dict(prior)
            carried["carried_from_version"] = int(state.get("plan_version", 0) or 0)
            new_ledger.append(carried)
        # Every *unresolved* intent the rewrite dropped is recorded DEAD rather than
        # deleted. A deleted intent is invisible to the generator, which is then free
        # to re-propose the exact framing this replan discarded — the loop the
        # attempt ledger and this marker jointly guard against.
        fresh_keys = {(e.get("intent") or "").strip().lower() for e in new_ledger}
        version = int(state.get("plan_version", 0) or 0)
        for prior in ledger:
            key = (prior.get("intent") or "").strip().lower()
            if not key or key in fresh_keys or prior.get("status") == INTENT_CLOSED:
                continue
            new_ledger.append(
                {
                    **prior,
                    "status": INTENT_DEAD,
                    "dead_reason": _dead_reason(prior),
                    "dead_at_version": version,
                }
            )
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
            history_entry["intermediate_answer"] = state.get(
                "last_intermediate_answer", ""
            )
            history_entry["n_uninstantiated"] = int(
                state.get("last_n_uninstantiated", 0) or 0
            )
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
            "current_subanswer_confidence": [],
            "retrieved_raw_context": Clear(),
            "retrieved_raw_triples": Clear(),
            "last_retractions": [],
            "last_unresolved_conflicts": [],
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
        out_state: Dict[str, Any] = {
            "final_answer": str(final_text),
            "concise_answer": str(concise),
            "reasoning": str(reasoning),
        }
        if plan_enabled:
            signal = abstention_signal(state.get("plan_ledger") or [])
            out_state["abstention"] = signal
            if signal["level"] != "none":
                logger.info(
                    "[gen_final] abstention=%s — %d/%d intents resolved; unmet: %s",
                    signal["level"],
                    signal["resolved"],
                    signal["total"],
                    signal["reasons"],
                )
        return out_state

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
    "resolve_primary_literal",
    "resolve_binding_key",
    "mark_stalled_intents",
    "apply_bindings",
    "apply_retractions",
    "classify_discharge",
    "latest_intermediate_answer",
    "render_plan_for_prompt",
]
