"""Build a hop-stratified MuSiQue set for the depth hypothesis.

The plan channel showed no established benefit on `bamboogle_hardmix`, and the
measurements say why: that set is 2-hop, and the no-plan baseline already
terminates in **1.45 hops**. There is almost no chain to prune, and 0 of 17
cross-hop repeats touched an already-solved intent — so the specific waste a plan
removes is nearly absent. The hypothesis that survives is therefore about
*regime*, not about the mechanism: planning should pay where drift compounds, and
its benefit should scale with chain depth.

MuSiQue is the right instrument because it encodes the hop count in the record id
(``2hop__``, ``3hop1__``, ``4hop1__``) and composes each question from a verified
decomposition. That gives depth stratification **inside one dataset**, so a
depth trend is not confounded by dataset, annotator, or answer-format changes —
which is what comparing bamboogle against a separate 4-hop corpus would be.

The prediction is falsifiable and directional: the plan-minus-no-plan gap should
*widen* from 2-hop to 4-hop. A flat or shrinking gap refutes the regime story and
says the idea is wrong rather than mis-applied.

Two caveats to carry into any writeup:

* MuSiQue ships gold ``paragraphs``; they are deliberately **not** emitted. This
  system retrieves for itself, so the run is open-domain and absolute scores will
  sit well below published reading-comprehension numbers. Only the plan-vs-no-plan
  contrast within the same retrieval stack is meaningful here.
* ``answer_aliases`` are merged into the ``answer`` list because
  ``metrics.compute_sub_em`` scores 1.0 if *any* listed answer is a substring, and
  a system that says "Francisco Guterres" should not lose to a gold of "Guterres".

Usage::

    python -m langgraph_coe.scripts.build_musique_depth --per-stratum 40
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Fixed so the file is reproducible; the runner's own shuffle is separate.
SEED = 20260827


def depth_of(record_id: str) -> int:
    """``4hop2__123_456`` → 4. The id is MuSiQue's own hop-count label."""
    head = (record_id or "").split("__")[0]
    return int(head[0]) if head[:1].isdigit() else 0


def build(per_stratum: int, out_path: Path, split: str) -> int:
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split=split)

    # Group by depth, then by the finer sub-type (4hop1 / 4hop2 / 4hop3) so a
    # stratum is not silently dominated by whichever composition template happens
    # to be most frequent — those templates differ in shape, not just in length.
    by_depth: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in ds:
        if not r.get("answerable", True):
            continue
        d = depth_of(r.get("id", ""))
        if d in (2, 3, 4):
            by_depth[d][r["id"].split("__")[0]].append(r)

    rng = random.Random(SEED)
    rows: List[Dict[str, Any]] = []
    for depth in (2, 3, 4):
        subtypes = by_depth[depth]
        picked: List[Dict[str, Any]] = []
        # Proportional allocation across sub-types, largest first, so rounding
        # never starves a small sub-type out of the sample entirely.
        total = sum(len(v) for v in subtypes.values())
        order = sorted(subtypes.items(), key=lambda kv: -len(kv[1]))
        for i, (name, items) in enumerate(order):
            remaining_slots = per_stratum - len(picked)
            remaining_types = len(order) - i
            want = (
                remaining_slots
                if remaining_types == 1
                else max(1, round(per_stratum * len(items) / total))
            )
            want = min(want, remaining_slots, len(items))
            picked.extend(rng.sample(items, want))
        for r in picked:
            golds = [r["answer"]] + [
                a for a in (r.get("answer_aliases") or []) if a and a != r["answer"]
            ]
            rows.append(
                {
                    "question": r["question"],
                    "answer": golds,
                    # Read by ``level_column=level``; this is the stratifying variable.
                    "level": f"{depth}hop",
                    "musique_id": r["id"],
                    # Number of verified decomposition steps — an independent check
                    # that the id's hop label matches the annotated chain length.
                    "n_steps": len(r.get("question_decomposition") or []),
                }
            )

    rng.shuffle(rows)  # interleave depths so a partial run is still balanced
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out_path}")
    print("  by depth :", dict(sorted(Counter(r["level"] for r in rows).items())))
    print("  by subtype:", dict(sorted(Counter(r["musique_id"].split("__")[0] for r in rows).items())))
    mism = [r for r in rows if r["n_steps"] != int(r["level"][0])]
    print(f"  id-label vs annotated chain length mismatches: {len(mism)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="datasets/musique_depth.jsonl")
    args = ap.parse_args()
    return build(args.per_stratum, Path(args.out), args.split)


if __name__ == "__main__":
    raise SystemExit(main())
