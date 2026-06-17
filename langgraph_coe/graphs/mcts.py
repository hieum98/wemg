"""MCTS strategy graph.

Monte Carlo Tree Search with CoT rollouts and shared text↔graph memory. The
search tree is stored as a dict in the graph state
(``Annotated[Dict, dict_merge]``).

Flow (one iteration = one superstep cycle)::

    START
      → select        (pUCT traversal root→leaf; seeds root if tree empty)
      → expand        (dispatch by leaf node_type → parallel generators)
      → simulate      (CoTGraph rollout from an expanded child; shared memory)
      → evaluate      (3 verifier views → reward ∈ [-1, 1])
      → backprop      (visits/value along current_path; bumps iteration)
      → mem_update    (MemoryUpdateGraph over expand+rollout+verifier text)
      → route:
            terminate? → synthesize → END
            else       → select   (loop)

Cross-iteration memory (``text_memory`` / ``graph_memory`` / ``entity_dict``)
is plain LastValue and is passed **by reference** into the rollout CoTGraph so
rollouts mutate the parent's memory directly (coe parity). Only ``mem_update``
replaces those channels with the consolidated result.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import (
    ANSWER_GENERATOR,
    FINAL_ANSWER_SYNTHESIZER,
    SELF_CORRECTOR,
    SUBQUESTION_GENERATOR,
    VERIFIER,
    AnswerGenerationInput,
    AnswerVerificationInput,
    FinalAnswerSynthesisInput,
    SelfCorrectionInput,
    SubquestionGenerationInput,
)
from ._memory_text import textualize_graph as _textualize_graph
from .cot import (
    Clear,
    N_SUBQUESTIONS,
    append_or_clear,
    build_cot_graph,
    gather_evidence,
    pool_subquestions,
)
from .kg_search import build_kg_search_graph
from .memory_update import (
    _is_retrieval_grounded,
    _strip_provenance_tag,
    build_memory_update_graph,
)
from .web_research import build_web_research_graph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Node types and priors
# ──────────────────────────────────────────────────────────────────────────────


class MCTSNodeType(str, Enum):
    USER_QUESTION = "user_question"  # root only
    SUB_QA = "sub_qa"
    SELF_CORRECTED = "self_corrected"
    FINAL_ANSWER = "final_answer"  # terminal


# Per-node-type prior used by pUCT selection. USER_QUESTION
# (root) has no prior — it is never a UCB candidate.
NODE_TYPE_PRIOR: Dict[MCTSNodeType, float] = {
    MCTSNodeType.SUB_QA: 0.60,
    MCTSNodeType.SELF_CORRECTED: 0.50,
    MCTSNodeType.FINAL_ANSWER: 0.30,
}


def dict_merge(
    left: Dict[str, "MCTSTreeNode"],
    right: Dict[str, "MCTSTreeNode"],
) -> Dict[str, "MCTSTreeNode"]:
    """Rightward dict union; right wins on key collisions.

    Covers both new-node injection (expand / simulate) and in-place visit/value
    updates from backprop, which re-emit the touched nodes as a partial dict.
    """
    return {**(left or {}), **(right or {})}


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────


class MCTSTreeNode(TypedDict, total=False):
    node_id: str
    parent_id: Optional[str]
    children_ids: List[str]
    node_type: MCTSNodeType
    content: Dict[str, Any]
    visits: int
    value: float
    prior: float


class MCTSState(TypedDict, total=False):
    # Inputs
    question: str
    max_iterations: int
    iteration: int

    # Tree
    tree: Annotated[Dict[str, MCTSTreeNode], dict_merge]
    root_id: str

    # Per-iteration traversal
    current_path: List[str]
    expanded_node_ids: List[str]
    simulation_result: Dict[str, Any]
    reward: float

    # Per-iteration retrieval accumulators (cleared each iteration)
    new_raw_triples: Annotated[List[Any], append_or_clear]
    # Retrieval-grounded facts from expansion (gather_evidence). Fed to
    # mem_update as [Retrieval]-provenance items, then cleared.
    new_retrieval_texts: Annotated[List[str], append_or_clear]

    # Per-iteration semantic-sufficiency flags (expand / rollout). Folded into
    # ``semantic_sufficiency_signals`` by backprop, then reset.
    expand_semantic_signal: bool
    rollout_semantic_signal: bool

    # Cross-iteration shared memory (rollouts mutate in place)
    text_memory: List[str]
    graph_memory: nx.DiGraph
    entity_dict: Dict[str, Any]

    # Memory facts already re-verified (recheck-on-retrieval); each distinct
    # fact is re-grounded once over the run to corroborate/gap-fill/evict it.
    reverified_facts: List[str]

    # Early termination tracking
    semantic_sufficiency_signals: int
    iterations_without_improvement: int
    best_value: float

    # Output
    final_answer: str
    concise_answer: str
    reasoning: str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _ntype(node: Dict[str, Any]) -> str:
    """Return a node's type as a plain string (handles enum or str storage)."""
    value = node.get("node_type")
    return getattr(value, "value", value)


def _is_terminal(node: Dict[str, Any]) -> bool:
    return _ntype(node) == MCTSNodeType.FINAL_ANSWER.value


def _make_node(
    node_type: MCTSNodeType, parent_id: Optional[str], content: Dict[str, Any]
) -> MCTSTreeNode:
    return {
        "node_id": _new_id(),
        "parent_id": parent_id,
        "children_ids": [],
        "node_type": node_type,
        "content": content,
        "visits": 0,
        "value": 0.0,
        "prior": NODE_TYPE_PRIOR.get(node_type, 1.0),
    }


def _make_root_node(root_id: str, question: str) -> MCTSTreeNode:
    """Build the USER_QUESTION root. Unlike :func:`_make_node`, the root keeps a
    caller-supplied ``node_id`` and a fixed prior of 1.0 (it is never a UCB
    candidate, so its prior is unused — see ``NODE_TYPE_PRIOR``)."""
    return {
        "node_id": root_id,
        "parent_id": None,
        "children_ids": [],
        "node_type": MCTSNodeType.USER_QUESTION,
        "content": {"question": question},
        "visits": 0,
        "value": 0.0,
        "prior": 1.0,
    }


def _format_text_memory(text_memory: Optional[List[str]]) -> str:
    items = [t for t in (text_memory or []) if isinstance(t, str) and t.strip()]
    return "\n".join(f"- {t}" for t in items)


def _join_memory_context(state: MCTSState) -> str:
    text = _format_text_memory(state.get("text_memory"))
    graph_text = _textualize_graph(state.get("graph_memory"))
    sections: List[str] = []
    if text:
        sections.append("Text memory:\n" + text)
    if graph_text:
        sections.append("Graph memory:\n" + graph_text)
    return "\n\n".join(sections) if sections else "Not provided"


def _answer_text(out: Any) -> str:
    return getattr(out, "concise_answer", None) or getattr(out, "answer", "") or ""


def _merge_ev(*evs: Dict[str, Any]) -> Dict[str, Any]:
    """Union the ``{facts, triples}`` evidence dicts from the expansion generators."""
    facts: List[str] = []
    triples: List[Any] = []
    for ev in evs:
        if not ev:
            continue
        facts.extend(ev.get("facts") or [])
        triples.extend(ev.get("triples") or [])
    return {"facts": facts, "triples": triples}


def _leaf_depth(path: List[str]) -> int:
    """Tree depth of the selected leaf (root = 0)."""
    return max(0, len(path) - 1)


def _should_gen_final(path: List[str], *, min_depth: int) -> bool:
    return _leaf_depth(path) >= min_depth


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────


def build_mcts_graph(registry: RoleModelRegistry, config: Optional[Any] = None):
    """Compile the MCTSGraph.

    ``config`` is the full :class:`LangGraphCoeConfig` in production (it carries
    ``search.mcts`` knobs plus ``memory`` for the rollout subgraphs). The test
    harness passes a minimal namespace exposing only ``search.mcts`` and patches
    the subgraph builders, so we read knobs defensively with ``getattr``.
    """
    mcts_cfg = getattr(getattr(config, "search", None), "mcts", None)
    sim_depth = int(getattr(mcts_cfg, "max_simulation_depth", 3) or 3)
    _final_min = getattr(mcts_cfg, "final_answer_min_depth", None)
    final_min_depth = 2 if _final_min is None else int(_final_min)
    high_conf = float(getattr(mcts_cfg, "high_confidence_threshold", 0.9) or 0.9)
    patience = int(getattr(mcts_cfg, "convergence_patience", 5) or 5)
    sem_count = int(getattr(mcts_cfg, "semantic_sufficiency_count", 3) or 3)
    explore = float(getattr(mcts_cfg, "exploration_weight", 2.0) or 2.0)
    min_iters = int(getattr(mcts_cfg, "min_iterations", 0) or 0)

    # Compile rollout + memory subgraphs once. These names are module globals so
    # the test harness can monkeypatch them before this builder runs.
    cot_graph = build_cot_graph(registry, config)
    memory_graph = build_memory_update_graph(
        registry, memory_cfg=getattr(config, "memory", None)
    )

    # Retrieval surfaces for expansion-time evidence gathering (gather_evidence).
    # Read knobs defensively — the test harness passes a minimal namespace.
    kg_graph = build_kg_search_graph(registry)
    web_enabled = bool(
        getattr(getattr(config, "web_search", None), "enabled", False)
    )
    web_graph = build_web_research_graph(registry) if web_enabled else None
    reranker_cfg = getattr(config, "reranker", None)
    rerank_top_k = max(1, int(getattr(reranker_cfg, "top_k", 10) or 10))
    extractor_max_chars = int(
        getattr(getattr(config, "memory", None), "extractor_max_input_chars", 24_000)
        or 24_000
    )

    # ── Expansion generators ────────────────────────────────────────────────

    async def _gen_subqa(
        question: str, parent_id: str, ctx: str, state: MCTSState
    ) -> tuple[List[MCTSTreeNode], bool, Dict[str, Any]]:
        """Decompose, retrieve evidence, and answer each subquestion.

        Returns ``(nodes, semantic_signal, evidence)``. ``semantic_signal`` is
        True when the generator deems the question answerable from context (coe
        parity: ``should_direct or not subquestions``); the caller then expands
        a FINAL_ANSWER child instead. ``evidence`` carries the retrieval facts/
        triples for memory persistence.
        """
        # coe parity: sample N completions, pool their decompositions (n=3).
        outs, _ = await execute_role_lc(
            registry,
            SUBQUESTION_GENERATOR,
            SubquestionGenerationInput(question=question, context=ctx),
            n=N_SUBQUESTIONS,
        )
        subqs, kg_flags, should_direct = pool_subquestions(outs)
        if should_direct or not subqs:
            return [], True, {}

        # Ground subanswers in retrieved evidence (corpus + gated KG fan-out →
        # per-subquestion rerank → EXTRACTOR), not memory/parametric knowledge.
        evidence = await gather_evidence(
            registry,
            question,
            subqs,
            needs_kg=kg_flags,
            memory_context=ctx,
            entity_dict=state.get("entity_dict"),
            kg_graph=kg_graph,
            web_graph=web_graph,
            web_enabled=web_enabled,
            reranker_cfg=reranker_cfg,
            rerank_top_k=rerank_top_k,
            extractor_max_chars=extractor_max_chars,
        )
        facts = [
            f
            for f in (evidence.get("extracted_facts") or [])
            if isinstance(f, str) and f.strip()
        ]
        answer_ctx = ctx
        if facts:
            evidence_block = "Retrieved evidence:\n" + "\n".join(
                f"- {f}" for f in facts
            )
            answer_ctx = f"{evidence_block}\n\n{ctx}" if ctx else evidence_block

        inputs = [
            AnswerGenerationInput(question=sq, context=answer_ctx) for sq in subqs
        ]
        answers, _ = await execute_role_lc(registry, ANSWER_GENERATOR, inputs)
        if not isinstance(answers, list):
            answers = [answers]
        nodes: List[MCTSTreeNode] = []
        for sq, ans in zip(subqs, answers):
            nodes.append(
                _make_node(
                    MCTSNodeType.SUB_QA,
                    parent_id,
                    {"sub_question": sq, "sub_answer": _answer_text(ans)},
                )
            )
        return nodes, False, {
            "facts": facts,
            "triples": list(evidence.get("raw_triples") or []),
        }

    async def _explore_for(question: str, target_q: str, ctx: str, state: MCTSState):
        """coe parity: one fresh retrieval pass (gather_evidence) for ``target_q``.

        Returns ``(answer_ctx, evidence)`` where ``answer_ctx`` prepends the
        distilled facts to the memory context and ``evidence`` is the
        ``{facts, triples}`` dict for persistence.
        """
        evidence = await gather_evidence(
            registry,
            question,
            [target_q],
            needs_kg=[True],
            memory_context=ctx,
            entity_dict=state.get("entity_dict"),
            kg_graph=kg_graph,
            web_graph=web_graph,
            web_enabled=web_enabled,
            reranker_cfg=reranker_cfg,
            rerank_top_k=rerank_top_k,
            extractor_max_chars=extractor_max_chars,
        )
        facts = [
            f
            for f in (evidence.get("extracted_facts") or [])
            if isinstance(f, str) and f.strip()
        ]
        answer_ctx = ctx
        if facts:
            block = "Retrieved evidence:\n" + "\n".join(f"- {f}" for f in facts)
            answer_ctx = f"{block}\n\n{ctx}" if ctx else block
        return answer_ctx, {
            "facts": facts,
            "triples": list(evidence.get("raw_triples") or []),
        }

    async def _reverify_memory(
        state: MCTSState, ctx: str
    ) -> tuple[str, Dict[str, Any], List[str]]:
        """Re-retrieve each unverified memory fact as its own query.

        Treats every ``[System Prediction]`` memory fact (provenance stripped) as
        a retrieval query and runs the standard ``gather_evidence`` pass (corpus +
        KG). ``[Retrieval]`` facts are skipped — they are already grounded in
        evidence, so re-fetching them is redundant. The fresh
        ``[Retrieval]`` evidence flows into the next consolidation, where the
        consolidator's conflict hierarchy can **corroborate** (b), **gap-fill**
        with adjacent facts (c), or **evict** a wrong model-inferred fact (a) —
        the KG re-fetch in particular returns an entity's true attributes
        regardless of what the stale fact claimed. Each distinct fact is
        re-verified once over the run (``reverified_facts``). Memory is tiny
        here (≈2-4 facts/question, max 12 observed), so the added cost is
        bounded. Returns ``(enriched_ctx, evidence, newly_reverified)``.
        """
        done = set(state.get("reverified_facts") or [])
        facts_q: List[str] = []
        for item in state.get("text_memory") or []:
            if not isinstance(item, str) or not item.strip():
                continue
            # Only re-verify [System Prediction] facts (the model's own
            # inferences). [Retrieval] facts are already grounded in evidence,
            # so re-retrieving them is redundant and was the main corpus+rerank
            # fan-out amplifier.
            if _is_retrieval_grounded(item):
                continue
            clean = _strip_provenance_tag(item)
            if clean and clean not in done and clean not in facts_q:
                facts_q.append(clean)
        if not facts_q:
            return ctx, {}, []

        evidence = await gather_evidence(
            registry,
            state.get("question", ""),
            facts_q,
            # Gate KG fan-out: don't force it on every fact. ``gather_evidence``
            # still fires KG when a fact mentions an already-linked entity
            # (``_subq_hits_known_entity``) — exactly where a re-fetch can
            # corroborate/evict — but skips Wikidata for facts with no
            # resolvable entity, cutting the public-API burst that caused the
            # ConnectTimeouts. Corpus re-verification (local FAISS) still runs.
            needs_kg=[False] * len(facts_q),
            memory_context=ctx,
            entity_dict=state.get("entity_dict"),
            kg_graph=kg_graph,
            web_graph=web_graph,
            web_enabled=web_enabled,
            reranker_cfg=reranker_cfg,
            rerank_top_k=rerank_top_k,
            extractor_max_chars=extractor_max_chars,
        )
        new_facts = [
            f
            for f in (evidence.get("extracted_facts") or [])
            if isinstance(f, str) and f.strip()
        ]
        enriched = ctx
        if new_facts:
            block = "Re-verification of prior memory (fresh evidence):\n" + "\n".join(
                f"- {f}" for f in new_facts
            )
            enriched = f"{block}\n\n{ctx}" if ctx and ctx != "Not provided" else block
        return (
            enriched,
            {"facts": new_facts, "triples": list(evidence.get("raw_triples") or [])},
            facts_q,
        )

    async def _gen_final(
        question: str,
        parent_id: str,
        ctx: str,
        state: MCTSState,
        *,
        should_explore: bool,
    ) -> tuple[List[MCTSTreeNode], Dict[str, Any]]:
        # coe parity (_generate_final_answer_nodes → generate_answer): the
        # terminal answer is a fresh retrieve-and-answer via ANSWER_GENERATOR,
        # not a synthesis over memory. Shallow expansions (depth < 2) retrieve;
        # deeper ones answer from the accumulated context.
        answer_ctx, evidence = ctx, {}
        if should_explore:
            answer_ctx, evidence = await _explore_for(question, question, ctx, state)
        out, _ = await execute_role_lc(
            registry,
            ANSWER_GENERATOR,
            AnswerGenerationInput(question=question, context=answer_ctx),
        )
        final_text = getattr(out, "answer", "") or ""
        concise = getattr(out, "concise_answer", "") or final_text
        reasoning = getattr(out, "reasoning", "") or ""
        node = _make_node(
            MCTSNodeType.FINAL_ANSWER,
            parent_id,
            {
                "final_answer": final_text,
                "concise_answer": concise,
                "reasoning": reasoning,
            },
        )
        return [node], evidence

    async def _gen_self_correct(
        leaf: MCTSTreeNode, ctx: str, question: str, state: MCTSState
    ) -> tuple[List[MCTSTreeNode], Dict[str, Any]]:
        content = leaf.get("content", {})
        sub_q = content.get("sub_question")
        sub_a = content.get("sub_answer")
        if not sub_q or not sub_a:
            return [], {}
        # coe parity (generate_self_correction → _explore): re-retrieve fresh
        # evidence for the sub-question before refining, not memory alone.
        correct_ctx, evidence = await _explore_for(question, sub_q, ctx, state)
        out, _ = await execute_role_lc(
            registry,
            SELF_CORRECTOR,
            SelfCorrectionInput(
                question=sub_q, proposed_answer=sub_a, context=correct_ctx
            ),
        )
        refined = getattr(out, "refined_answer", "") or sub_a
        node = _make_node(
            MCTSNodeType.SELF_CORRECTED,
            leaf["node_id"],
            {"sub_question": sub_q, "sub_answer": refined},
        )
        return [node], evidence

    # ── Nodes ────────────────────────────────────────────────────────────────

    async def select(state: MCTSState) -> Dict[str, Any]:
        tree = dict(state.get("tree") or {})
        root_id = state.get("root_id") or "root"
        tree_update: Dict[str, MCTSTreeNode] = {}

        if root_id not in tree:
            root = _make_root_node(root_id, state.get("question", ""))
            tree[root_id] = root
            tree_update[root_id] = root

        # pUCT traversal root → leaf.
        node_id = root_id
        path = [node_id]
        while True:
            node = tree[node_id]
            children = node.get("children_ids") or []
            if not children or _is_terminal(node):
                break
            parent_visits = node.get("visits", 0)
            best_id, best_score = None, -math.inf
            for cid in children:
                child = tree[cid]
                visits = child.get("visits", 0)
                q = (child.get("value", 0.0) / visits) if visits > 0 else 0.0
                u = (
                    explore
                    * child.get("prior", 1.0)
                    * math.sqrt(parent_visits + 1.0)
                    / (1.0 + visits)
                )
                score = q + u
                if score > best_score:
                    best_score, best_id = score, cid
            node_id = best_id
            path.append(node_id)

        out: Dict[str, Any] = {"current_path": path}
        if tree_update:
            out["tree"] = tree_update
        return out

    async def expand(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        path = state.get("current_path") or []
        if not path:
            return {"expanded_node_ids": [], "expand_semantic_signal": False}
        target = tree[path[-1]]
        target_path = list(path)

        # FINAL_ANSWER nodes are terminal — never expanded themselves. But a
        # previously-visited terminal leaf means selection found no unexplored
        # sibling: re-expand its PARENT so the tree keeps growing instead of
        # re-walking the same chain forever (coe parity: mcts_search picks
        # ``path[-2]`` in that case). A fresh terminal (visits == 0) is simply
        # evaluated this iteration.
        if _is_terminal(target):
            if int(target.get("visits", 0) or 0) > 0 and len(path) >= 2:
                target = tree[path[-2]]
                target_path = list(path[:-1])
            else:
                return {"expanded_node_ids": [], "expand_semantic_signal": False}

        ntype = _ntype(target)
        pid = target["node_id"]
        ctx = _join_memory_context(state)
        question = state.get("question", "")

        # Re-verify current memory facts (recheck-on-retrieval): fresh evidence
        # is prepended to the context all generators see this iteration, and
        # folded into ``evidence`` so consolidation can corroborate/gap-fill/evict.
        ctx, reverify_ev, reverified_now = await _reverify_memory(state, ctx)

        signal = False
        evidence: Dict[str, Any] = {}
        # coe parity: terminal answers from a shallow node (depth < 2) retrieve
        # fresh evidence; deeper ones answer from accumulated context.
        should_explore_final = _leaf_depth(target_path) < 2
        if ntype == MCTSNodeType.USER_QUESTION.value:
            subqa, signal, evidence = await _gen_subqa(question, pid, ctx, state)
            children = list(subqa)
            if signal or _should_gen_final(target_path, min_depth=final_min_depth):
                fin, ev_f = await _gen_final(
                    question, pid, ctx, state, should_explore=should_explore_final
                )
                children.extend(fin)
                evidence = _merge_ev(evidence, ev_f)
        elif ntype == MCTSNodeType.SUB_QA.value:
            (subqa, signal, ev_s), (sc, ev_c) = await asyncio.gather(
                _gen_subqa(question, pid, ctx, state),
                _gen_self_correct(target, ctx, question, state),
            )
            children = subqa + sc
            evidence = _merge_ev(ev_s, ev_c)
            if signal:
                fin, ev_f = await _gen_final(
                    question, pid, ctx, state, should_explore=should_explore_final
                )
                children.extend(fin)
                evidence = _merge_ev(evidence, ev_f)
        elif ntype == MCTSNodeType.SELF_CORRECTED.value:
            subqa, signal, evidence = await _gen_subqa(question, pid, ctx, state)
            children = list(subqa)
            if signal:
                fin, ev_f = await _gen_final(
                    question, pid, ctx, state, should_explore=should_explore_final
                )
                children.extend(fin)
                evidence = _merge_ev(evidence, ev_f)
        else:
            children = []

        # Fold the memory re-verification evidence into this iteration's facts
        # so it reaches consolidation, and record which facts were re-verified.
        evidence = _merge_ev(evidence, reverify_ev)
        out: Dict[str, Any] = {"expand_semantic_signal": signal}
        if reverified_now:
            out["reverified_facts"] = list(
                set(state.get("reverified_facts") or []) | set(reverified_now)
            )
        if evidence.get("facts"):
            out["new_retrieval_texts"] = list(evidence["facts"])
        if evidence.get("triples"):
            out["new_raw_triples"] = list(evidence["triples"])
        if target_path != list(path):
            # Redirected to the parent: backprop/simulate must work from there,
            # not through the stale terminal sibling.
            out["current_path"] = target_path

        if not children:
            out["expanded_node_ids"] = []
            return out

        updates: Dict[str, MCTSTreeNode] = {c["node_id"]: c for c in children}
        parent = dict(target)
        parent["children_ids"] = list(parent.get("children_ids") or []) + [
            c["node_id"] for c in children
        ]
        updates[pid] = parent
        out["tree"] = updates
        out["expanded_node_ids"] = [c["node_id"] for c in children]
        return out

    async def simulate(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        path = list(state.get("current_path") or [])
        expanded = state.get("expanded_node_ids") or []

        # Pick a non-terminal expanded child to roll out from. If expansion only
        # yielded a terminal FINAL_ANSWER (or nothing), there is no rollout: the
        # terminal becomes the evaluated node.
        start_id = next((cid for cid in expanded if not _is_terminal(tree[cid])), None)
        if start_id is None:
            if expanded:
                path = path + [expanded[0]]
            return {
                "current_path": path,
                "simulation_result": {"rollout_texts": []},
                "rollout_semantic_signal": False,
            }

        # Rollout through the compiled CoTGraph. LangGraph channel updates
        # REPLACE values rather than mutating them in place, so the rollout's
        # final memory channels are captured from ``cot_out`` and returned
        # below — otherwise every retrieval/consolidation the rollout performed
        # would be silently discarded (coe parity: the legacy rollout mutated
        # the shared WorkingMemory object).
        cot_out = await cot_graph.ainvoke(
            {
                "question": state.get("question", ""),
                "max_depth": sim_depth,
                "depth": 0,
                "text_memory": state.get("text_memory"),
                "graph_memory": state.get("graph_memory"),
                "entity_dict": state.get("entity_dict"),
            }
        )

        updates: Dict[str, MCTSTreeNode] = {}
        chain_ids: List[str] = []
        rollout_texts: List[str] = []

        # One SUB_QA chain node per (subquestion, subanswer) pair — packing a
        # whole CoT iteration into one node hides the decomposition from UCB
        # selection and from the evaluate() candidate text.
        for entry in cot_out.get("iteration_history") or []:
            subqs = entry.get("subquestions") or []
            subas = entry.get("subanswers") or []
            for i, sq in enumerate(subqs):
                sa = subas[i] if i < len(subas) else ""
                node = _make_node(
                    MCTSNodeType.SUB_QA,
                    None,
                    {"sub_question": sq, "sub_answer": sa},
                )
                chain_ids.append(node["node_id"])
                updates[node["node_id"]] = node
            rollout_texts.extend(a for a in subas if isinstance(a, str) and a.strip())

        final_answer = cot_out.get("final_answer")
        if final_answer:
            fnode = _make_node(
                MCTSNodeType.FINAL_ANSWER,
                None,
                {
                    "final_answer": final_answer,
                    "concise_answer": final_answer,
                    "reasoning": "",
                },
            )
            chain_ids.append(fnode["node_id"])
            updates[fnode["node_id"]] = fnode
            rollout_texts.append(str(final_answer))

        # Wire the chain linearly: start_id → chain[0] → chain[1] → …
        prev = start_id
        for cid in chain_ids:
            pnode = dict(updates.get(prev) or tree[prev])
            pnode["parent_id"] = pnode.get("parent_id")
            pnode["children_ids"] = list(pnode.get("children_ids") or []) + [cid]
            updates[prev] = pnode
            child = dict(updates[cid])
            child["parent_id"] = prev
            updates[cid] = child
            prev = cid

        new_path = path + [start_id] + chain_ids
        rollout_triples = [t for t in (cot_out.get("retrieved_raw_triples") or []) if t]
        out: Dict[str, Any] = {
            "tree": updates,
            "current_path": new_path,
            "new_raw_triples": rollout_triples,
            "simulation_result": {"rollout_texts": rollout_texts},
            # The rollout finalizing via "answerable from context" is a
            # semantic-sufficiency vote (coe parity: simulate's rollout_signal).
            "rollout_semantic_signal": bool(cot_out.get("is_answerable")),
        }
        # Persist the rollout's consolidated memory (see comment above).
        if cot_out.get("text_memory") is not None:
            out["text_memory"] = list(cot_out.get("text_memory") or [])
        if cot_out.get("graph_memory") is not None:
            out["graph_memory"] = cot_out.get("graph_memory")
        if cot_out.get("entity_dict") is not None:
            out["entity_dict"] = dict(cot_out.get("entity_dict") or {})
        return out

    async def evaluate(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        path = state.get("current_path") or []
        sr = dict(state.get("simulation_result") or {})
        if not path:
            return {
                "reward": 0.0,
                "simulation_result": {**sr, "verifier_critiques": []},
            }

        term = tree[path[-1]]
        content = term.get("content", {})
        if content.get("final_answer"):
            candidate = (
                f"Answer: {content.get('final_answer', '')}\n"
                f"Reasoning: {content.get('reasoning', '')}"
            )
        elif content.get("sub_answer"):
            candidate = str(content.get("sub_answer"))
        else:
            candidate = str(content)

        question = state.get("question", "")
        text_ctx = _format_text_memory(state.get("text_memory")) or "Not provided"
        graph_ctx = _textualize_graph(state.get("graph_memory")) or "Not provided"

        try:
            (r1, _), (r2, _), (r3, _) = await asyncio.gather(
                execute_role_lc(
                    registry,
                    VERIFIER,
                    AnswerVerificationInput(
                        question=question,
                        candidate_answer=candidate,
                        context="Not provided",
                    ),
                ),
                execute_role_lc(
                    registry,
                    VERIFIER,
                    AnswerVerificationInput(
                        question=question, candidate_answer=candidate, context=text_ctx
                    ),
                ),
                execute_role_lc(
                    registry,
                    VERIFIER,
                    AnswerVerificationInput(
                        question=question, candidate_answer=candidate, context=graph_ctx
                    ),
                ),
            )
            ratings = [float(r1.rating), float(r2.rating), float(r3.rating)]
            reward = (sum(ratings) / 3.0 - 5.0) / 5.0
            critiques = [
                f"Verifier (no context): {r1.reasoning}",
                f"Verifier (text memory): {r2.reasoning}",
                f"Verifier (graph memory): {r3.reasoning}",
            ]
        except Exception as e:  # noqa: BLE001 — neutral reward on parse failure (coe parity)
            logger.warning("Verifier reward computation failed; reward=0.0: %s", e)
            reward, critiques = 0.0, []

        return {
            "reward": reward,
            "simulation_result": {**sr, "verifier_critiques": critiques},
        }

    async def backprop(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        path = state.get("current_path") or []
        reward = float(state.get("reward", 0.0) or 0.0)

        # Snapshot BEFORE visit increments: convergence tracking must only
        # count newly created terminals (coe parity: ``is_new_terminal``),
        # otherwise re-walking an already-evaluated chain burns patience
        # without exploring anything.
        leaf = tree.get(path[-1]) if path else None
        leaf_is_terminal = leaf is not None and _is_terminal(leaf)
        is_new_terminal = leaf_is_terminal and int(leaf.get("visits", 0) or 0) == 0

        updates: Dict[str, MCTSTreeNode] = {}
        for nid in path:
            node = dict(updates.get(nid) or tree[nid])
            node["visits"] = node.get("visits", 0) + 1
            node["value"] = node.get("value", 0.0) + reward
            updates[nid] = node

        best = float(state.get("best_value", 0.0) or 0.0)
        no_imp = int(state.get("iterations_without_improvement", 0) or 0)
        if leaf_is_terminal:
            if reward > best:
                best, no_imp = reward, 0
            elif is_new_terminal:
                no_imp += 1

        signals = int(state.get("semantic_sufficiency_signals", 0) or 0)
        if state.get("expand_semantic_signal") or state.get("rollout_semantic_signal"):
            signals += 1

        return {
            "tree": updates,
            "iteration": int(state.get("iteration", 0) or 0) + 1,
            "best_value": best,
            "iterations_without_improvement": no_imp,
            "semantic_sufficiency_signals": signals,
            "expand_semantic_signal": False,
            "rollout_semantic_signal": False,
        }

    async def mem_update(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        sr = state.get("simulation_result") or {}

        # 1. Text from directly-expanded children.
        expanded_texts: List[str] = []
        for nid in state.get("expanded_node_ids") or []:
            node = tree.get(nid)
            if not node:
                continue
            c = node.get("content", {})
            if c.get("sub_answer"):
                expanded_texts.append(str(c["sub_answer"]))
            if c.get("final_answer"):
                expanded_texts.append(str(c["final_answer"]))

        # 2. Rollout subanswers + final. 3. Verifier critiques.
        rollout_texts = list(sr.get("rollout_texts") or [])
        critiques = list(sr.get("verifier_critiques") or [])

        # Generated sub-answers + rollouts share the expansion depth; verifier
        # critiques are tracked separately so they get the shallower hop below.
        new_text_items = [
            t
            for t in (expanded_texts + rollout_texts)
            if isinstance(t, str) and t.strip()
        ]
        new_critique_items = [
            t for t in critiques if isinstance(t, str) and t.strip()
        ]

        # Reasoning depth of this iteration's expansion. The new sub-answers and
        # retrieval facts were produced one level below the path tip, so
        # ``len(current_path)`` is their hop (legacy ``node.depth + 1``). Verifier
        # critiques assess the tip itself, so legacy tags them one level
        # shallower at ``node.depth`` (``mcts.py`` ``_self_correct_nodes``); we
        # mirror that with ``len(current_path) - 1``. ``None`` if no path.
        path_len = len(state.get("current_path") or [])
        gen_hop = path_len or None
        critique_hop = (path_len - 1) if path_len else None

        payload = {
            "question": state.get("question", ""),
            "new_text_items": new_text_items,
            "new_critique_items": new_critique_items,
            # Expansion-time retrieval facts carry [Retrieval] provenance so
            # consolidation can prefer them over generated answers/critiques.
            "new_retrieval_items": list(state.get("new_retrieval_texts") or []),
            "new_raw_triples": list(state.get("new_raw_triples") or []),
            "current_text_memory": list(state.get("text_memory") or []),
            "current_graph": state.get("graph_memory") or nx.DiGraph(),
            "entity_dict": dict(state.get("entity_dict") or {}),
            # Annotates [hop=N] for the consolidator's Hop Depth Filtering (RC-C).
            "hop_depth": gen_hop,
            "critique_hop_depth": critique_hop,
        }
        result = await memory_graph.ainvoke(payload)
        return {
            "text_memory": list(result.get("updated_text_memory") or []),
            "graph_memory": result.get("updated_graph") or nx.DiGraph(),
            "entity_dict": dict(result.get("updated_entity_dict") or {}),
            "new_raw_triples": Clear(),
            "new_retrieval_texts": Clear(),
        }

    async def synthesize(state: MCTSState) -> Dict[str, Any]:
        tree = state.get("tree") or {}
        candidates: List[str] = []
        scores: List[float] = []
        for node in tree.values():
            if _ntype(node) != MCTSNodeType.FINAL_ANSWER.value:
                continue
            c = node.get("content", {})
            candidates.append(c.get("final_answer") or c.get("concise_answer") or "")
            visits = node.get("visits", 0)
            scores.append((node.get("value", 0.0) / visits) if visits > 0 else 0.0)

        candidate_scores: Optional[List[float]] = scores
        if not candidates:
            candidates = list(state.get("text_memory") or []) or [
                "No answer available."
            ]
            candidate_scores = None

        out, _ = await execute_role_lc(
            registry,
            FINAL_ANSWER_SYNTHESIZER,
            FinalAnswerSynthesisInput(
                question=state.get("question", ""),
                candidate_answers=candidates,
                candidate_scores=candidate_scores,
                context=_join_memory_context(state),
            ),
        )
        final_text = (
            getattr(out, "final_answer", "") or getattr(out, "concise_answer", "") or ""
        )
        concise = getattr(out, "concise_answer", "") or final_text
        reasoning = getattr(out, "reasoning", "") or ""
        return {
            "final_answer": str(final_text),
            "concise_answer": str(concise),
            "reasoning": str(reasoning),
        }

    # ── Routing ────────────────────────────────────────────────────────────────

    def route_after_iteration(state: MCTSState) -> str:
        iteration = int(state.get("iteration", 0) or 0)
        # Hard cap always wins.
        if iteration >= int(state.get("max_iterations", 1) or 1):
            return "synthesize"
        # Floor: no early-termination condition may fire before min_iterations.
        if iteration < min_iters:
            return "select"
        if float(state.get("best_value", 0.0) or 0.0) >= high_conf:
            return "synthesize"
        if int(state.get("semantic_sufficiency_signals", 0) or 0) >= sem_count:
            return "synthesize"
        if int(state.get("iterations_without_improvement", 0) or 0) >= patience:
            return "synthesize"
        return "select"

    builder = StateGraph(MCTSState)
    builder.add_node("select", select)
    builder.add_node("expand", expand)
    builder.add_node("simulate", simulate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("backprop", backprop)
    builder.add_node("mem_update", mem_update)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "select")
    builder.add_edge("select", "expand")
    builder.add_edge("expand", "simulate")
    builder.add_edge("simulate", "evaluate")
    builder.add_edge("evaluate", "backprop")
    builder.add_edge("backprop", "mem_update")
    builder.add_conditional_edges(
        "mem_update", route_after_iteration, ["select", "synthesize"]
    )
    builder.add_edge("synthesize", END)

    return builder.compile()


__all__ = [
    "MCTSNodeType",
    "NODE_TYPE_PRIOR",
    "dict_merge",
    "MCTSTreeNode",
    "MCTSState",
    "_leaf_depth",
    "_should_gen_final",
    "build_mcts_graph",
]
