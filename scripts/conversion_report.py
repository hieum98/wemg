#!/usr/bin/env python
"""Paired CONVERSION report: does the evidence in memory become the answer?

Conversion is the metric the terminal-referent check targets, and it is not accuracy.
Accuracy mixes two independent failures — *retrieval* (the gold was never found) and
*conversion* (the gold sat in memory and something else was answered). A fix aimed at
the second is diluted roughly 3x when read through accuracy, because ~60% of questions
never retrieve the gold at all and no amount of better binding can help them.

So the denominator here is only the questions whose textual memory contains the gold,
and it is computed PER ARM: a run that retrieves more gold gets a larger denominator,
which is the honest accounting rather than a free win.

Both a paired reading (same question, both arms hold the gold) and the two marginal
rates are reported. The paired one is the test; the marginals show whether the
denominators moved, which would make the marginals non-comparable.

Usage:  python scripts/conversion_report.py vf_on:vf_off
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def sign_test(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def load(run: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(f"results/{run}/evaluation_log.jsonl") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[_norm(r.get("question"))[:80]] = r
    return out


def memory_text(rec: dict) -> str:
    """Everything the answerer could have read, as one normalized blob.

    Both the consolidated textual memory and the graph triples count: a referent bound
    from a triple is just as available to synthesis as one from a text line.

    Read as bytes and decoded lossily rather than parsed: the textual memory is JSON but
    the graph is a pickle, and for a substring test the pickle's payload strings are
    already in the clear. Parsing would mean two loaders and a networkx import for no
    gain.
    """
    parts: List[str] = []
    arts = rec.get("artifacts") or {}
    for key in ("textual_memory_path", "graph_memory_path"):
        p = arts.get(key)
        if p and os.path.exists(p):
            try:
                parts.append(open(p, "rb").read().decode("utf-8", "ignore"))
            except OSError:
                continue
    return _norm(" ".join(parts))


def gold_in(text: str, golds: List[str]) -> bool:
    return any(g and g in text for g in (_norm(x) for x in golds))


def main(pairs: List[str]) -> None:
    for pair in pairs:
        na, nb = pair.split(":")
        A, B = load(na), load(nb)
        keys = sorted(set(A) & set(B))

        # marginals: each arm judged against its own retrieval
        marg: Dict[str, Tuple[int, int]] = {}
        for name, run in ((na, A), (nb, B)):
            den = num = 0
            for k in keys:
                r = run[k]
                golds = r.get("correct_answer") or []
                if not gold_in(memory_text(r), golds):
                    continue
                den += 1
                num += float(r.get("sub_em_short") or 0.0) > 0
            marg[name] = (num, den)

        # paired: only questions where BOTH arms hold the gold in memory
        both = [
            k
            for k in keys
            if gold_in(memory_text(A[k]), A[k].get("correct_answer") or [])
            and gold_in(memory_text(B[k]), B[k].get("correct_answer") or [])
        ]
        ca = cb = da = db = 0
        for k in both:
            ha = float(A[k].get("sub_em_short") or 0.0) > 0
            hb = float(B[k].get("sub_em_short") or 0.0) > 0
            ca += ha
            cb += hb
            if ha and not hb:
                da += 1
            elif hb and not ha:
                db += 1

        print(f"══ {na} vs {nb} ══  {len(keys)} paired questions\n")
        print("── RETRIEVAL (the denominator; must not differ much) ─────────────")
        for name in (na, nb):
            _, den = marg[name]
            print(f"  {name:<10} gold in memory on {den}/{len(keys)} = {100 * den / max(len(keys), 1):.1f}%")
        print()
        print("── CONVERSION, marginal (each arm on its own gold-in-memory set) ──")
        for name in (na, nb):
            num, den = marg[name]
            print(f"  {name:<10} {num}/{den} = {100 * num / max(den, 1):.1f}%")
        print()
        print("── CONVERSION, paired (both arms hold the gold) ───────────────────")
        n = len(both)
        if n:
            print(f"  measurable on {n} questions")
            print(f"  {na:<10} {ca}/{n} = {100 * ca / n:.1f}%")
            print(f"  {nb:<10} {cb}/{n} = {100 * cb / n:.1f}%   gap {ca - cb:+d}")
            d = da + db
            p = sign_test(da, db)
            theta = da / d if d else 0.5
            print(f"  discordant {d} ({da}/{db})  theta={theta:.3f}  sign test p = {p:.4f}"
                  f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
        else:
            print("  no questions with the gold in memory in both arms")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
