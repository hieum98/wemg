"""Measure whether a shared plan stops MCTS rollouts re-decomposing the question.

This is the one setting where the plan's *cost* thesis is coherent. In CoT the
plan has almost nothing to save — the no-plan baseline already terminates in ~1.4
hops, so there is no long chain of redundant questions to prune. MCTS is
different: every iteration runs a whole CoT rollout, and without a shared plan
each expansion re-decomposes the question independently. Two consequences:

* the same subquestion is retrieved for repeatedly across sibling subtrees, and
* the visit statistics pUCT compares are not comparable, because siblings were
  answering different decompositions of the question.

So the measurement is duplication *across* the tree, not within one rollout:

``reuse``
    1 - distinct/total over every SUB_QA subquestion in the tree. High means the
    tree asked the same things repeatedly.
``sibling overlap``
    mean Jaccard over subquestion sets of sibling subtrees. This is the direct
    read on coordination: 0 means siblings explored disjoint decompositions, 1
    means they duplicated each other outright.

Usage::

    python -m langgraph_coe.scripts.mcts_rollout_cost --run results/m1 --run results/m0
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


def _walk(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    yield node
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk(child)


def _subqs(node: Dict[str, Any]) -> List[str]:
    """Subquestion text on a node, tolerating the several shapes content takes."""
    content = node.get("content") or {}
    out: List[str] = []
    for key in ("subquestion", "sub_question", "question"):
        v = content.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for key in ("subquestions", "sub_questions"):
        v = content.get(key)
        if isinstance(v, list):
            out.extend(str(x).strip() for x in v if str(x).strip())
    return out


def analyse(run: Path) -> Dict[str, Any]:
    trees = sorted((run / "artifacts").glob("*/search_tree.json"))
    per_q: List[Dict[str, Any]] = []
    for f in trees:
        try:
            root = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(root, list):
            root = root[0] if root else {}
        nodes = list(_walk(root)) if isinstance(root, dict) else []
        all_sq = [q for n in nodes for q in _subqs(n)]
        norm = [_norm(q) for q in all_sq]
        distinct = len(set(norm))
        # Sibling subtrees of the root: each is one line of search the tree pursued.
        subtrees: List[Set[str]] = []
        for child in (root.get("children") or []) if isinstance(root, dict) else []:
            if not isinstance(child, dict):
                continue
            s = {_norm(q) for n in _walk(child) for q in _subqs(n)}
            if s:
                subtrees.append(s)
        overlaps = [
            len(a & b) / len(a | b)
            for a, b in itertools.combinations(subtrees, 2)
            if (a | b)
        ]
        per_q.append(
            {
                "nodes": len(nodes),
                "subqs": len(all_sq),
                "distinct": distinct,
                "reuse": 1 - distinct / len(all_sq) if all_sq else 0.0,
                "subtrees": len(subtrees),
                "sibling_overlap": statistics.mean(overlaps) if overlaps else None,
            }
        )
    if not per_q:
        return {"questions": 0}
    ov = [q["sibling_overlap"] for q in per_q if q["sibling_overlap"] is not None]
    n = len(per_q)
    return {
        "questions": n,
        "nodes_per_q": sum(q["nodes"] for q in per_q) / n,
        "subqs_per_q": sum(q["subqs"] for q in per_q) / n,
        "distinct_per_q": sum(q["distinct"] for q in per_q) / n,
        "reuse": statistics.mean(q["reuse"] for q in per_q),
        "sibling_overlap": statistics.mean(ov) if ov else None,
        "questions_with_siblings": len(ov),
    }


def accuracy(run: Path) -> str:
    f = run / "evaluation_log.jsonl"
    if not f.is_file():
        return "n/a"
    rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return "n/a"
    got = sum(r.get("sub_em_short") or 0 for r in rows)
    return f"{got:.0f}/{len(rows)} = {got/len(rows):.1%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    print("=" * 82)
    print("MCTS rollout duplication — does a shared plan stop siblings re-decomposing?")
    print("=" * 82)
    for r in args.run:
        run = Path(r)
        st = analyse(run)
        print(f"\n── {run.name} ── sub-EM {accuracy(run)}")
        if not st.get("questions"):
            print("   no search_tree.json artifacts")
            continue
        ov = st["sibling_overlap"]
        print(f"   questions {st['questions']} | tree nodes/q {st['nodes_per_q']:.1f}")
        print(f"   subquestions/q {st['subqs_per_q']:.1f} "
              f"(distinct {st['distinct_per_q']:.1f})")
        print(f"   reuse (fraction re-asked)     {st['reuse']:.1%}")
        print(f"   mean sibling-subtree overlap  "
              + (f"{ov:.1%}  (over {st['questions_with_siblings']} questions)"
                 if ov is not None else "n/a — no question had 2+ root subtrees"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
