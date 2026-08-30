"""Depth-stratified plan-vs-no-plan report for the MuSiQue hop strata.

The hypothesis this tests: the plan's benefit should **scale with chain depth**,
because it removes drift, and drift compounds. On `bamboogle_hardmix` the no-plan
baseline terminated in 1.45 hops — almost no chain to keep on track — and the plan
showed no established gain (36/62 vs 31/62, sign test p = 0.33). A widening gap
from 2hop to 4hop supports the regime explanation; a flat or shrinking gap refutes
it and says the mechanism, not the setting, is wrong.

Three things are reported per stratum, because accuracy alone cannot distinguish
"the plan did not help" from "the experiment could not see it":

**accuracy** with a paired sign test over discordant rows — the effect.

**cost** from the per-question meter (`calls`, `completions`, `prompt_tokens`) —
the other half of the claim, and the number the reviewers asked for. Falls back to
hop/subquestion proxies when the meter did not record.

**failure attribution** — of the questions that were wrong, how many attempts
returned *zero* facts. This separates the two failure modes that a depth test
confounds: a plan tells the system what to look for, so it cannot help when
retrieval comes back empty. If wrong answers are dominated by zero-fact attempts,
the stratum is retrieval-bound and the plan variable is unmeasurable there,
whatever the accuracy says.

Usage::

    python -m langgraph_coe.scripts.depth_report --plan results/dep_plan \\
        --noplan results/dep_noplan
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

STRATA = ["2hop", "3hop", "4hop"]


def load_rows(run: Path) -> List[Dict[str, Any]]:
    f = run / "evaluation_log.jsonl"
    if not f.is_file():
        return []
    return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def sign_test(wins_a: int, wins_b: int) -> float:
    """Two-sided exact sign test over discordant pairs only.

    Concordant rows carry no information about which arm is better, so the test is
    over the ``wins_a + wins_b`` rows where the arms disagree. Returns 1.0 when
    nothing is discordant.
    """
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = max(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _artifact_index(run: Path) -> Dict[str, Dict[str, Any]]:
    """Map a question-slug prefix → its ``plan.json``, for cost-proxy fallbacks."""
    out: Dict[str, Dict[str, Any]] = {}
    for f in (run / "artifacts").glob("*/plan.json"):
        m = re.match(r"q_\d+_[0-9a-f]+_(.*)$", os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        try:
            out[m.group(1)] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a partial run can hold a truncated file
            continue
    return out


def _slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (q or "").lower())[:48]


def _plan_for(idx: Dict[str, Dict[str, Any]], question: str) -> Optional[Dict[str, Any]]:
    s = _slug(question)
    if s in idx:
        return idx[s]
    return next((v for k, v in idx.items() if k.startswith(s[:40])), None)


def summarise(run: Path) -> Dict[str, Dict[str, Any]]:
    """Per-stratum accuracy, cost and zero-fact attempt counts for one arm."""
    idx = _artifact_index(run)
    acc: Dict[str, Dict[str, Any]] = {
        s: {"n": 0, "correct": 0, "calls": 0, "tokens": 0, "metered": 0,
            "hops": 0, "subqs": 0, "zero_fact": 0, "attempts": 0,
            "zero_fact_wrong": 0, "attempts_wrong": 0}
        for s in STRATA
    }
    for r in load_rows(run):
        s = r.get("level")
        if s not in acc:
            continue
        a = acc[s]
        a["n"] += 1
        correct = bool(r.get("sub_em_short"))
        a["correct"] += int(correct)
        cost = r.get("cost") or {}
        if cost.get("calls"):
            a["metered"] += 1
            a["calls"] += cost.get("calls") or 0
            a["tokens"] += cost.get("prompt_tokens") or 0
        pj = _plan_for(idx, r.get("question", ""))
        if pj:
            hist = pj.get("iteration_history") or []
            a["hops"] += len(hist)
            a["subqs"] += sum(len(h.get("subquestions") or []) for h in hist)
            att = pj.get("plan_attempts_log") or []
            zero = sum(1 for x in att if not (x.get("n_facts") or 0))
            a["attempts"] += len(att)
            a["zero_fact"] += zero
            if not correct:
                a["attempts_wrong"] += len(att)
                a["zero_fact_wrong"] += zero
    return acc


def report(plan_run: Path, noplan_run: Path) -> int:
    p_rows = {r["question"]: r for r in load_rows(plan_run)}
    n_rows = {r["question"]: r for r in load_rows(noplan_run)}
    if not p_rows or not n_rows:
        print("one or both runs have no rows yet")
        return 1
    shared = set(p_rows) & set(n_rows)

    p_sum, n_sum = summarise(plan_run), summarise(noplan_run)

    print("=" * 88)
    print("DEPTH HYPOTHESIS — does the plan's benefit scale with chain length?")
    print(f"paired on {len(shared)} questions answered by both arms")
    print("=" * 88)
    print(f"\n{'stratum':8} {'n':>3}  {'plan':>9} {'no-plan':>9}  {'gap':>6}  "
          f"{'p':>6}   {'plan calls':>10} {'noplan calls':>12}")
    print("-" * 88)

    by_depth = defaultdict(lambda: {"n": 0, "pw": 0, "nw": 0, "p": 0, "c": 0})
    for q in shared:
        s = p_rows[q].get("level")
        if s not in STRATA:
            continue
        d = by_depth[s]
        d["n"] += 1
        pc, nc = bool(p_rows[q].get("sub_em_short")), bool(n_rows[q].get("sub_em_short"))
        d["p"] += int(pc)
        d["c"] += int(nc)
        if pc and not nc:
            d["pw"] += 1
        elif nc and not pc:
            d["nw"] += 1

    gaps = []
    for s in STRATA:
        d = by_depth.get(s)
        if not d or not d["n"]:
            print(f"{s:8} {'-':>3}  (no paired rows yet)")
            continue
        gap = (d["p"] - d["c"]) / d["n"]
        gaps.append((s, gap))
        pv = sign_test(d["pw"], d["nw"])
        pc = p_sum[s]["calls"] / p_sum[s]["metered"] if p_sum[s]["metered"] else float("nan")
        nc = n_sum[s]["calls"] / n_sum[s]["metered"] if n_sum[s]["metered"] else float("nan")
        print(f"{s:8} {d['n']:>3}  {d['p']:>4}/{d['n']:<4} {d['c']:>4}/{d['n']:<4}  "
              f"{gap:>+6.1%}  {pv:>6.3f}   {pc:>10.1f} {nc:>12.1f}")

    if len(gaps) >= 2:
        print(f"\n  gap by depth: " + "  ".join(f"{s} {g:+.1%}" for s, g in gaps))
        trend = gaps[-1][1] - gaps[0][1]
        verdict = (
            "WIDENS with depth — consistent with the regime explanation"
            if trend > 0.05
            else "NARROWS with depth — refutes the regime explanation"
            if trend < -0.05
            else "FLAT across depth — no depth effect detected"
        )
        print(f"  {gaps[0][0]} → {gaps[-1][0]}: {trend:+.1%}  →  {verdict}")

    print("\n" + "-" * 88)
    print("failure attribution — can the plan variable even be seen at this depth?")
    print("-" * 88)
    for arm, summ in (("plan", p_sum), ("no-plan", n_sum)):
        for s in STRATA:
            a = summ[s]
            if not a["n"]:
                continue
            aw, zw = a["attempts_wrong"], a["zero_fact_wrong"]
            share = f"{zw/aw:.0%}" if aw else "n/a"
            print(f"  {arm:8} {s}: {a['correct']}/{a['n']} correct | "
                  f"on WRONG rows, {zw}/{aw} attempts returned zero facts ({share}) | "
                  f"hops/q {a['hops']/a['n']:.2f} subq/q {a['subqs']/a['n']:.2f}")
    print("\n  A high zero-fact share means the stratum is retrieval-bound: a plan says")
    print("  what to look for and cannot help when nothing comes back, so a null result")
    print("  there is uninformative about the plan rather than evidence against it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="results/dep_plan")
    ap.add_argument("--noplan", default="results/dep_noplan")
    args = ap.parse_args()
    return report(Path(args.plan), Path(args.noplan))


if __name__ == "__main__":
    raise SystemExit(main())
