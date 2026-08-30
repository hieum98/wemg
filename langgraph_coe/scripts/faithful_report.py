"""Report eval results against the dataset gold *and* against adjudicated gold.

`compute_sub_em` asks whether the gold string is a substring of the prediction. That
makes the score only as good as the gold, and this dataset's gold has three defect
classes that move the number for reasons that have nothing to do with the framework:

* **defective** — `"James Cameroon"` (misspelt; a correct "James Cameron" scores 0)
  and `"George VI\\n"` (trailing newline; unmatchable as stored). These rows are
  *unwinnable*, so they understate any system.
* **contestable** — the referent genuinely has more than one defensible answer
  ("the third fastest bird", "the most populous city in Punjab").
* **inflating** — a single-character gold (`"l"`) is a substring of nearly any
  English answer, so the row is scored free.

This script reports both numbers side by side and never silently substitutes one for
the other. Verdicts come from a hand-adjudicated file, built from web evidence by a
reader — not from a model, which would launder a guess into a ground truth.

Usage::

    python -m langgraph_coe.scripts.faithful_report \\
        --run results/a0 --run results/a1 --verdicts /tmp/gold_verdicts.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Verdicts that make a row's sub-EM uninformative, and why.
_UNWINNABLE = {"GOLD_DEFECTIVE", "GOLD_UNVERIFIABLE"}
_FREE = {"GOLD_INFLATING"}
_AMBIGUOUS = {"GOLD_CONTESTABLE"}

# Per-prediction verdicts (``--pred-verdicts``): how a *specific answer* stands up
# against reality, as opposed to against the gold string. The first group is
# factually correct and lost only on presentation, which is what makes raw sub-EM
# understate the system.
_PRED_CORRECT = {"FORMAT_ONLY", "ALIAS", "GOLD_DEFECTIVE"}
_PRED_DEFENSIBLE = {"CONTESTABLE"}
# Deliberately counted as failures: an answer that omits the day the question asked
# for is incomplete, and a date off by several days is wrong even if the calendar
# story is nearby.
_PRED_WRONG = {"WRONG", "UNDER_SPECIFIED", "CALENDAR"}


def _norm(s: str) -> str:
    """Loose match for adjudication: case, punctuation and spacing insensitive."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def sub_em(pred: str, gold: str) -> float:
    return 1.0 if gold and gold.lower() in (pred or "").lower() else 0.0


def lenient_em(pred: str, gold: str) -> float:
    """sub-EM after normalisation — recovers the two defective golds.

    Fixes exactly the mechanical defects (a stray newline, punctuation, casing), not
    the facts: `"George VI\\n"` matches "George VI", and `"James Cameroon"` still does
    NOT match "James Cameron", because that is a spelling error in the *gold* and only
    a human can rule on it.
    """
    return 1.0 if gold and _norm(gold) in _norm(pred) else 0.0


def load_rows(run: Path) -> List[Dict[str, Any]]:
    f = run / "evaluation_log.jsonl"
    if not f.is_file():
        return []
    return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def report(
    runs: List[Path],
    verdicts: Dict[str, Dict[str, str]],
    pred_verdicts: Dict[str, Dict[str, str]] | None = None,
) -> int:
    all_rows: Dict[str, List[Dict[str, Any]]] = {r.name: load_rows(r) for r in runs}
    if not any(all_rows.values()):
        print("no evaluation_log.jsonl found in any run")
        return 1

    print("=" * 96)
    print("FAITHFUL ASSESSMENT — dataset sub-EM vs adjudicated")
    print("=" * 96)

    for name, rows in all_rows.items():
        if not rows:
            print(f"\n{name}: no rows yet")
            continue
        n = len(rows)
        raw = sum(r.get("sub_em_short") or 0 for r in rows)
        lenient = 0.0
        counted = 0
        excluded: List[str] = []
        contested: List[str] = []
        for r in rows:
            q = r.get("question", "")
            gold = r.get("correct_answer") or ""
            if isinstance(gold, list) and gold:
                gold = gold[0]
            pred = r.get("predicted_answer") or r.get("full_answer") or ""
            verdict = (verdicts.get(q) or {}).get("verdict", "GOLD_CORRECT")
            if verdict in _UNWINNABLE:
                excluded.append(q)
                continue
            if verdict in _AMBIGUOUS:
                contested.append(q)
            counted += 1
            lenient += lenient_em(str(pred), str(gold))

        print(f"\n── {name} ({n} rows) ──")
        print(f"  dataset sub-EM (as scored by the harness) : {raw:.0f}/{n} = {raw/n:.1%}")
        print(
            f"  adjudicated  (defective/unverifiable rows excluded, gold normalised)"
            f" : {lenient:.0f}/{counted} = {lenient/counted:.1%}"
            if counted
            else "  adjudicated: no scorable rows"
        )
        print(f"  excluded as unwinnable : {len(excluded)}")
        for q in excluded:
            print(f"      - {q[:80]}  ({verdicts[q]['verdict']})")
        print(f"  rows with a contestable gold (kept, but the number is soft): {len(contested)}")

        errs = [r for r in rows if r.get("error")]
        if errs:
            print(f"  !! {len(errs)} rows errored")

        if pred_verdicts:
            from collections import Counter

            recovered = 0
            defensible = 0
            breakdown: Counter = Counter()
            for r in rows:
                if r.get("sub_em_short"):
                    continue
                pv = (pred_verdicts.get(r.get("question", "")) or {}).get("verdict")
                if not pv:
                    breakdown["(unadjudicated)"] += 1
                    continue
                breakdown[pv] += 1
                if pv in _PRED_CORRECT:
                    recovered += 1
                elif pv in _PRED_DEFENSIBLE:
                    defensible += 1
            strict = raw + recovered
            generous = strict + defensible
            print("\n  adjudicated against reality (misses read individually):")
            print(
                f"    factually correct, lost on presentation : +{recovered}"
                "   (date format, name alias, defective gold)"
            )
            print(f"    defensible alternative reading          : +{defensible}")
            print(f"    ADJUDICATED sub-EM  : {strict}/{n} = {strict/n:.1%}"
                  f"   (+{strict/n - raw/n:.1%} over the harness number)")
            print(f"    with defensible too : {generous}/{n} = {generous/n:.1%}")
            print("    miss breakdown: " + ", ".join(
                f"{k}={v}" for k, v in breakdown.most_common()))

    # Cross-run comparison, only over questions every run answered.
    names = [n for n, rs in all_rows.items() if rs]
    if len(names) >= 2:
        shared = set.intersection(
            *[{r.get("question") for r in all_rows[n]} for n in names]
        )
        print(f"\n── head-to-head on the {len(shared)} questions all runs answered ──")
        for n in names:
            idx = {r["question"]: r for r in all_rows[n]}
            got = sum(idx[q].get("sub_em_short") or 0 for q in shared)
            print(f"  {n:12} {got:.0f}/{len(shared)} = {got/len(shared):.1%}")
        # Where they differ — the rows worth reading a plan.json for.
        idxs = {n: {r["question"]: r for r in all_rows[n]} for n in names}
        diffs = [
            q
            for q in shared
            if len({(idxs[n][q].get("sub_em_short") or 0) for n in names}) > 1
        ]
        print(f"\n  {len(diffs)} questions where the runs disagree:")
        for q in sorted(diffs)[:20]:
            marks = " ".join(
                f"{n}={'Y' if (idxs[n][q].get('sub_em_short') or 0) else 'n'}" for n in names
            )
            print(f"      [{marks}] {q[:70]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="results dir")
    ap.add_argument("--verdicts", required=True, help="hand-adjudicated gold verdicts JSON")
    ap.add_argument(
        "--pred-verdicts",
        help="hand-adjudicated per-prediction verdicts JSON (FORMAT_ONLY/ALIAS/WRONG/...)",
    )
    args = ap.parse_args()
    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    preds = {}
    if args.pred_verdicts:
        preds = json.loads(Path(args.pred_verdicts).read_text(encoding="utf-8"))
        preds.pop("_doc", None)
    return report([Path(r) for r in args.run], verdicts, preds)


if __name__ == "__main__":
    raise SystemExit(main())
