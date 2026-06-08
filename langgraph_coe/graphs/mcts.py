"""MCTSGraph (Phase 3).

Monte Carlo Tree Search with CoT rollouts and shared text↔graph memory. Ports
``coe/reasoning/mcts.py`` onto LangGraph, storing the search tree as a dict in
state (``Annotated[Dict, dict_merge]``). See ``implementation_plan.md`` §8.

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
from .cot import Clear, append_or_clear, build_cot_graph
from .memory_update import build_memory_update_graph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# §8.1 Node types and priors
# ──────────────────────────────────────────────────────────────────────────────


class MCTSNodeType(str, Enum):
    USER_QUESTION = "user_question"  # root only
    SUB_QA = "sub_qa"
    SELF_CORRECTED = "self_corrected"
    FINAL_ANSWER = "final_answer"  # terminal


# Ports legacy ``_NODE_TYPE_PRIOR`` (coe/reasoning/mcts.py:27). USER_QUESTION
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
# §8.2 State
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

    # Per-iteration retrieval accumulator (cleared each iteration)
    new_raw_triples: Annotated[List[Any], append_or_clear]

    # Cross-iteration shared memory (rollouts mutate in place)
    text_memory: List[str]
    graph_memory: nx.DiGraph
    entity_dict: Dict[str, Any]

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

    # ── Expansion generators ────────────────────────────────────────────────

    async def _gen_subqa(question: str, parent_id: str, ctx: str) -> List[MCTSTreeNode]:
        out, _ = await execute_role_lc(
            registry,
            SUBQUESTION_GENERATOR,
            SubquestionGenerationInput(question=question, context=ctx),
        )
        subqs = [
            s for s in (getattr(out, "subquestions", None) or []) if s and s.strip()
        ]
        if not subqs:
            return []
        inputs = [AnswerGenerationInput(question=sq, context=ctx) for sq in subqs]
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
        return nodes

    async def _gen_final(
        question: str, parent_id: str, ctx: str, state: MCTSState
    ) -> List[MCTSTreeNode]:
        candidates = list(state.get("text_memory") or []) or [
            "No prior reasoning available."
        ]
        out, _ = await execute_role_lc(
            registry,
            FINAL_ANSWER_SYNTHESIZER,
            FinalAnswerSynthesisInput(
                question=question, candidate_answers=candidates, context=ctx
            ),
        )
        final_text = getattr(out, "final_answer", "") or ""
        concise = getattr(out, "concise_answer", "") or final_text
        reasoning = getattr(out, "reasoning", "") or ""
        return [
            _make_node(
                MCTSNodeType.FINAL_ANSWER,
                parent_id,
                {
                    "final_answer": final_text,
                    "concise_answer": concise,
                    "reasoning": reasoning,
                },
            )
        ]

    async def _gen_self_correct(leaf: MCTSTreeNode, ctx: str) -> List[MCTSTreeNode]:
        content = leaf.get("content", {})
        sub_q = content.get("sub_question")
        sub_a = content.get("sub_answer")
        if not sub_q or not sub_a:
            return []
        out, _ = await execute_role_lc(
            registry,
            SELF_CORRECTOR,
            SelfCorrectionInput(question=sub_q, proposed_answer=sub_a, context=ctx),
        )
        refined = getattr(out, "refined_answer", "") or sub_a
        return [
            _make_node(
                MCTSNodeType.SELF_CORRECTED,
                leaf["node_id"],
                {"sub_question": sub_q, "sub_answer": refined},
            )
        ]

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
            return {"expanded_node_ids": []}
        leaf = tree[path[-1]]

        # FINAL_ANSWER nodes are terminal — never expanded (§8.3).
        if _is_terminal(leaf):
            return {"expanded_node_ids": []}

        ntype = _ntype(leaf)
        pid = leaf["node_id"]
        ctx = _join_memory_context(state)
        question = state.get("question", "")

        if ntype == MCTSNodeType.USER_QUESTION.value:
            subqa = await _gen_subqa(question, pid, ctx)
            children = list(subqa)
            if _should_gen_final(path, min_depth=final_min_depth):
                children.extend(await _gen_final(question, pid, ctx, state))
        elif ntype == MCTSNodeType.SUB_QA.value:
            subqa, sc = await asyncio.gather(
                _gen_subqa(question, pid, ctx),
                _gen_self_correct(leaf, ctx),
            )
            children = subqa + sc
        elif ntype == MCTSNodeType.SELF_CORRECTED.value:
            children = await _gen_subqa(question, pid, ctx)
        else:
            children = []

        if not children:
            return {"expanded_node_ids": []}

        updates: Dict[str, MCTSTreeNode] = {c["node_id"]: c for c in children}
        parent = dict(leaf)
        parent["children_ids"] = list(parent.get("children_ids") or []) + [
            c["node_id"] for c in children
        ]
        updates[pid] = parent
        return {"tree": updates, "expanded_node_ids": [c["node_id"] for c in children]}

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
            return {"current_path": path, "simulation_result": {"rollout_texts": []}}

        # Rollout through the compiled CoTGraph. text_memory / graph_memory /
        # entity_dict are passed BY REFERENCE so the rollout mutates the parent's
        # shared memory (coe parity).
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

        for entry in cot_out.get("iteration_history") or []:
            subqs = entry.get("subquestions") or []
            subas = entry.get("subanswers") or []
            concat_q = "\n".join(f"Sub Q{i + 1}: {q}" for i, q in enumerate(subqs))
            concat_a = "\n".join(f"Sub A{i + 1}: {a}" for i, a in enumerate(subas))
            node = _make_node(
                MCTSNodeType.SUB_QA,
                None,
                {"sub_question": concat_q, "sub_answer": concat_a},
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
        return {
            "tree": updates,
            "current_path": new_path,
            "new_raw_triples": rollout_triples,
            "simulation_result": {"rollout_texts": rollout_texts},
        }

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

        updates: Dict[str, MCTSTreeNode] = {}
        for nid in path:
            node = dict(updates.get(nid) or tree[nid])
            node["visits"] = node.get("visits", 0) + 1
            node["value"] = node.get("value", 0.0) + reward
            updates[nid] = node

        best = float(state.get("best_value", 0.0) or 0.0)
        no_imp = int(state.get("iterations_without_improvement", 0) or 0)
        if reward > best:
            best, no_imp = reward, 0
        else:
            no_imp += 1

        return {
            "tree": updates,
            "iteration": int(state.get("iteration", 0) or 0) + 1,
            "best_value": best,
            "iterations_without_improvement": no_imp,
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

        new_text_items = [
            t
            for t in (expanded_texts + rollout_texts + critiques)
            if isinstance(t, str) and t.strip()
        ]

        payload = {
            "question": state.get("question", ""),
            "new_text_items": new_text_items,
            "new_raw_triples": list(state.get("new_raw_triples") or []),
            "current_text_memory": list(state.get("text_memory") or []),
            "current_graph": state.get("graph_memory") or nx.DiGraph(),
            "entity_dict": dict(state.get("entity_dict") or {}),
        }
        result = await memory_graph.ainvoke(payload)
        return {
            "text_memory": list(result.get("updated_text_memory") or []),
            "graph_memory": result.get("updated_graph") or nx.DiGraph(),
            "entity_dict": dict(result.get("updated_entity_dict") or {}),
            "new_raw_triples": Clear(),
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
