#!/usr/bin/env python
"""Paired report for the four plan fixes.

Reports the three quantities the goal is stated in — accuracy, cost, and the
intermediate-referent leak that the plan was previously *causing* — each as a paired
test over questions, because the unpaired mean is what produced two retracted claims
earlier in this project: a -18.5% cost saving and a -6.45 calls/question saving, both
heavy-right-tail artifacts whose paired medians sat at or above zero.

Usage:  python scripts/fix_report.py fix_plan_s1:fix_noplan_s1 fix_plan_s2:fix_noplan_s2
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def load(run: str) -> Dict[str, dict]:
    path = f"results/{run}/evaluation_log.jsonl"
    out: Dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[_norm(r.get("question"))[:80]] = r
    return out


def sign_test(a: int, b: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(diffs: List[float]) -> Tuple[float, int]:
    """Wilcoxon signed-rank on paired differences, normal approximation with ties.

    The test the cost claim actually needs. A bootstrap CI on the *mean* difference
    excluded zero twice in this project while the median sat at zero and "plan
    cheaper" ran at a coin flip — both times the mean was a heavy right tail, and both
    claims were retracted. Wilcoxon ranks the magnitudes, so one 300-call outlier
    cannot carry the result.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return (1.0, n)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sd_w == 0:
        return (1.0, n)
    z = (w_plus - mean_w) / sd_w
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (min(1.0, p), n)


def boot_ci(diffs: List[float], iters: int = 20000, seed: int = 0) -> Tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def leaked(rec: dict, ledger: List[dict]) -> Optional[bool]:
    """Did this answer name only a *scaffolding* referent?

    Needs the plan arm's ledger to know which referents are scaffolding, so the same
    ledger is applied to both arms — that is what makes it a paired measurement of the
    same error on the same question.
    """
    if len(ledger) < 2:
        return None
    dep = {e.get("depends_on") for e in ledger if isinstance(e.get("depends_on"), int)}

    def surfaces(idxs):
        return [
            _norm(b.get("surface"))
            for i in idxs
            for b in (ledger[i].get("bindings") or [])
            if b.get("surface")
        ]

    non_term = surfaces([i for i in range(len(ledger)) if i in dep])
    term = surfaces([i for i in range(len(ledger)) if i not in dep])
    ans = _norm(rec.get("predicted_answer"))
    if not ans:
        return None
    return any(s and s in ans for s in non_term) and not any(
        s and s in ans for s in term
    )


def main(pairs: List[str]) -> None:
    acc_p = acc_c = n = 0
    disc_p = disc_c = 0
    acc_diffs: List[float] = []
    cost_diffs: List[float] = []
    tok_diffs: List[float] = []
    leak_p = leak_c = leak_n = 0
    leak_disc_p = leak_disc_c = 0
    per_pair: List[str] = []
    hops_p: List[int] = []
    hops_c: List[int] = []

    for pair in pairs:
        pa, pc = pair.split(":")
        A, C = load(pa), load(pc)
        keys = sorted(set(A) & set(C))
        a_hit = c_hit = 0
        d_p = d_c = 0
        for k in keys:
            ra, rc = A[k], C[k]
            ea = float(ra.get("sub_em_short") or 0.0)
            ec = float(rc.get("sub_em_short") or 0.0)
            n += 1
            a_hit += ea > 0
            c_hit += ec > 0
            acc_p += ea > 0
            acc_c += ec > 0
            if ea > 0 and ec == 0:
                d_p += 1
            elif ec > 0 and ea == 0:
                d_c += 1
            acc_diffs.append((1.0 if ea > 0 else 0.0) - (1.0 if ec > 0 else 0.0))
            ca = (ra.get("cost") or {})
            cc = (rc.get("cost") or {})
            cost_diffs.append(ca.get("calls", 0) - cc.get("calls", 0))
            tok_diffs.append(ca.get("prompt_tokens", 0) - cc.get("prompt_tokens", 0))
            # leak, paired on the plan arm's ledger
            adir = (ra.get("artifacts") or {}).get("artifact_dir") or ""
            pj = os.path.join(adir, "plan.json")
            if os.path.exists(pj):
                blob = json.load(open(pj))
                led = blob.get("plan_ledger") or []
                att = [x.get("hop", 0) for e in led for x in (e.get("attempts") or [])]
                if att:
                    hops_p.append(max(att) + 1)
                la, lc = leaked(ra, led), leaked(rc, led)
                if la is not None and lc is not None:
                    leak_n += 1
                    leak_p += la
                    leak_c += lc
                    if la and not lc:
                        leak_disc_p += 1
                    elif lc and not la:
                        leak_disc_c += 1
        disc_p += d_p
        disc_c += d_c
        per_pair.append(
            f"  {pa:<16} {a_hit:>3}/{len(keys):<4} vs {pc:<16} {c_hit:>3}/{len(keys):<4}"
            f" gap {a_hit - c_hit:+3d}   discordant {d_p + d_c} ({d_p}/{d_c})"
        )

    print(f"paired questions: {n}\n")
    print("\n".join(per_pair))
    print()
    print("── ACCURACY ─────────────────────────────────────────────────────")
    print(f"  plan    {acc_p}/{n} = {100 * acc_p / n:.2f}%")
    print(f"  no-plan {acc_c}/{n} = {100 * acc_c / n:.2f}%   gap {acc_p - acc_c:+d}")
    d = disc_p + disc_c
    theta = disc_p / d if d else 0.5
    p = sign_test(disc_p, disc_c)
    lo, hi = boot_ci(acc_diffs)
    print(f"  discordant {d} ({disc_p}/{disc_c})  theta={theta:.3f}  sign test p = {p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    print(f"  bootstrap 95% CI on paired accuracy difference: [{lo:+.4f}, {hi:+.4f}]")
    print()
    print("── COST (paired, per question) ───────────────────────────────────")
    for name, ds in (("calls", cost_diffs), ("prompt tokens", tok_diffs)):
        ds_sorted = sorted(ds)
        med = ds_sorted[len(ds_sorted) // 2] if ds_sorted else 0
        lo, hi = boot_ci([float(x) for x in ds])
        cheaper = sum(1 for x in ds if x < 0)
        wp, wn = wilcoxon([float(x) for x in ds])
        print(f"  {name:<14} mean {sum(ds) / max(len(ds), 1):+,.1f}   median {med:+,.1f}"
              f"   plan cheaper on {cheaper}/{len(ds)} ({100 * cheaper / max(len(ds), 1):.0f}%)")
        print(f"  {'':<14} bootstrap CI on the MEAN [{lo:+,.1f}, {hi:+,.1f}]"
              f"  {'excludes 0' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else 'includes 0'}")
        print(f"  {'':<14} Wilcoxon signed-rank p = {wp:.4f} (n={wn} non-zero)"
              f"  {'SIGNIFICANT' if wp < 0.05 else 'not significant'}  <- the paired test")
        if (lo < 0 and hi < 0) and wp >= 0.05:
            print(f"  {'':<14} ** mean CI excludes 0 but Wilcoxon does not: heavy tail, "
                  f"not a per-question saving **")
    if hops_p:
        print(f"  mean hops run (plan arm): {sum(hops_p) / len(hops_p):.2f}")
    print()
    print("── INTERMEDIATE-REFERENT LEAK (the harm the fix targets) ─────────")
    if leak_n:
        print(f"  measurable on {leak_n} paired questions with a chained plan")
        print(f"  plan    {leak_p} ({100 * leak_p / leak_n:.1f}%)")
        print(f"  no-plan {leak_c} ({100 * leak_c / leak_n:.1f}%)")
        pl = sign_test(leak_disc_p, leak_disc_c)
        print(f"  discordant {leak_disc_p + leak_disc_c} ({leak_disc_p}/{leak_disc_c})"
              f"  sign test p = {pl:.4f}")
        print("  baseline before the fix: plan 10.6% vs no-plan 5.3%, 43 (31/12), p = 0.0054")
    else:
        print("  no chained plans found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
