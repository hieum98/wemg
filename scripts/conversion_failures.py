#!/usr/bin/env python
"""Classify conversion failures by mechanism, so the fix targets the biggest bucket.

A conversion failure is a question whose memory holds the gold and whose answer is
still wrong. There are three ways that happens, and they need different fixes:

  A. NEVER BOUND    — some intent is still open/stalled. The referent was never
                      extracted from memory at all, so the downstream hop had nothing
                      to ground on and drifted. Fix: widen the candidate pool.
  B. WRONG REFERENT — every terminal intent closed, but on a surface that is not the
                      gold. Fix: adjudicate between rivals (what verification tried).
  C. SYNTHESIS      — a terminal intent closed *on the gold* and the final answer is
                      still wrong. Fix: the synthesis prompt / candidate ordering.

Bucket A is further split on whether memory contains a line that would plausibly have
resolved the stalled intent — i.e. whether the fix is "read memory" (cheap, deterministic)
or "retrieve harder" (expensive). That split is the whole point of the script: the
terminal-referent verification addressed B and moved paired conversion by exactly zero,
so the bucket sizes were never actually measured before it was built.

Usage:  python scripts/conversion_failures.py vf_on vf_off
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was",
    "city", "county", "state", "river", "town",
}


def _content(s: str) -> set:
    return {w for w in _norm(s).split() if w not in STOP and len(w) > 2}


def memory_lines(rec: dict) -> List[str]:
    p = (rec.get("artifacts") or {}).get("textual_memory_path")
    if not p or not os.path.exists(p):
        return []
    try:
        blob = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return []
    return [str(x) for x in blob] if isinstance(blob, list) else []


def plan_blob(rec: dict) -> dict:
    p = (rec.get("artifacts") or {}).get("plan_path")
    if not p or not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return {}


def retrieval_only(lines: List[str]) -> List[str]:
    """Lines the plan is allowed to bind from — [Retrieval], never [System Prediction].

    The same provenance rule the PLANNER prompt enforces, applied here so the bucket-A
    size is not inflated by gold that only ever appeared inside the model's own guess.
    """
    return [ln for ln in lines if "[Retrieval]" in ln]


def main(runs: List[str]) -> None:
    buckets: Counter = Counter()
    examples: Dict[str, List[str]] = {"A_readable": [], "A_absent": [], "B": [], "C": []}
    stall_reasons: Counter = Counter()
    total = gold_mem = 0

    for run in runs:
        path = f"results/{run}/evaluation_log.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            golds = [_norm(g) for g in (rec.get("correct_answer") or [])]
            lines = memory_lines(rec)
            ret = retrieval_only(lines)
            blob = " ".join(_norm(x) for x in ret)
            if not any(g and g in blob for g in golds):
                continue  # retrieval failure, not a conversion failure
            gold_mem += 1
            if float(rec.get("sub_em_short") or 0.0) > 0:
                continue  # converted fine
            # ---- this is a conversion failure; classify it
            pb = plan_blob(rec)
            ledger = pb.get("plan_ledger") or []
            dep = {e.get("depends_on") for e in ledger if isinstance(e.get("depends_on"), int)}
            terminals = [i for i in range(len(ledger)) if i not in dep]
            open_any = [
                e for e in ledger if e.get("status") not in ("closed",)
            ]
            q = (rec.get("question") or "")[:70]
            pred = rec.get("predicted_answer")
            gold = (rec.get("correct_answer") or [""])[0]

            term_surfaces = [
                _norm(b.get("surface"))
                for i in terminals
                for b in (ledger[i].get("bindings") or [])
                if b.get("surface")
            ]
            closed_on_gold = any(
                g and s and (g in s or s in g) for s in term_surfaces for g in golds
            )

            if open_any:
                # would reading memory have resolved a stalled intent?
                readable = False
                for e in open_any:
                    for r in e.get("attempts") or []:
                        pass
                    want = _content(e.get("intent") or "")
                    for ln in ret:
                        have = _content(ln)
                        if want and len(want & have) >= max(2, len(want) // 3):
                            readable = True
                            break
                    if e.get("stall_reason"):
                        stall_reasons[str(e["stall_reason"])[:60]] += 1
                    if readable:
                        break
                key = "A_readable" if readable else "A_absent"
                buckets[key] += 1
                if len(examples[key]) < 6:
                    examples[key].append(f"{q!r}\n        gold={gold!r} pred={pred!r}")
            elif closed_on_gold:
                buckets["C"] += 1
                if len(examples["C"]) < 6:
                    examples["C"].append(f"{q!r}\n        gold={gold!r} pred={pred!r}")
            else:
                buckets["B"] += 1
                if len(examples["B"]) < 6:
                    examples["B"].append(
                        f"{q!r}\n        gold={gold!r} pred={pred!r} bound={term_surfaces[:3]}"
                    )

    fails = sum(buckets.values())
    print(f"runs: {', '.join(runs)}")
    print(f"questions: {total}   gold in [Retrieval] memory: {gold_mem} "
          f"({100 * gold_mem / max(total, 1):.1f}%)")
    print(f"conversion failures: {fails} "
          f"({100 * fails / max(gold_mem, 1):.1f}% of those, "
          f"{100 * fails / max(total, 1):.1f}% of all questions)\n")
    labels = {
        "A_readable": "A. never bound, and memory HELD a matching line  <-- free to fix",
        "A_absent": "A. never bound, memory had nothing matching",
        "B": "B. wrong referent bound at a terminal intent",
        "C": "C. bound the gold, synthesis lost it",
    }
    for k in ("A_readable", "A_absent", "B", "C"):
        n = buckets[k]
        print(f"  {labels[k]:<52} {n:>3}  ({100 * n / max(fails, 1):.0f}%)")
    print()
    if stall_reasons:
        print("── stall reasons on open intents ─────────────────────────────────")
        for r, c in stall_reasons.most_common(6):
            print(f"  {c:>3}  {r}")
        print()
    for k in ("A_readable", "B", "C"):
        if examples[k]:
            print(f"── examples: {labels[k]} ──")
            for e in examples[k]:
                print(f"     {e}")
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
