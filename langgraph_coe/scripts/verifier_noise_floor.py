"""Experiment E5: measure the VERIFIER's own noise floor.

``search.plan.memory_disagreement_threshold`` is a *gap* between the closed-book
verifier rating and the lowest memory-grounded one, and MCTS treats a gap wider
than the threshold as "the evidence disagrees with the answer". That reading is
only sound if the gap exceeds what identical inputs already produce by chance —
the verifier runs at ``temperature: 0.7``, so the same context rated twice does
not return the same number.

This script establishes that floor: it rates each case ``--repeats`` times with
byte-identical input and reports the observed spread. Any threshold at or below
the measured spread fires on sampling noise, not on disagreement.

Cases are drawn from real predictions in a completed run so the ratings are over
the distribution the threshold actually sees, not over hand-written toys.

Usage::

    python -m langgraph_coe.scripts.verifier_noise_floor \\
        --config langgraph_coe/config.eval.yaml --run results/d1 --cases 6 --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from ..config import LangGraphCoeConfig
from ..llm import RoleModelRegistry, execute_role_lc, is_safe_default
from ..roles import VERIFIER, AnswerVerificationInput


def load_cases(run: Path, limit: int) -> List[Dict[str, Any]]:
    """Real (question, prediction, memory) triples from a finished run.

    Prefers rows the harness scored 0: a threshold matters most where the answer
    is wrong, since that is when a memory-grounded view *should* diverge.
    """
    log = run / "evaluation_log.jsonl"
    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows.sort(key=lambda r: (r.get("sub_em_short") or 0))
    cases = []
    for r in rows:
        pred = r.get("predicted_answer") or r.get("full_answer") or ""
        if not pred:
            continue
        mem = r.get("text_memory") or []
        if not mem:
            # The artifacts hold memory when the log does not.
            for art in (run / "artifacts").glob("*/memory.json"):
                pass
        cases.append(
            {
                "question": r.get("question", ""),
                "prediction": str(pred),
                "context": "\n".join(str(m) for m in mem[:40]) or "Not provided",
                "sub_em": r.get("sub_em_short") or 0,
            }
        )
        if len(cases) >= limit:
            break
    return cases


async def run(cfg_path: str, run_dir: str, n_cases: int, repeats: int) -> int:
    cfg = LangGraphCoeConfig.from_yaml(cfg_path)
    registry = RoleModelRegistry(cfg.llm)
    cases = load_cases(Path(run_dir), n_cases)
    if not cases:
        print("no usable cases found")
        return 1

    print(f"VERIFIER noise floor — {len(cases)} cases x {repeats} identical calls")
    print(f"model: {cfg.llm.role_tiers.get(VERIFIER.name, 'default')}\n")

    all_spreads: List[float] = []
    all_sd: List[float] = []
    for i, c in enumerate(cases):
        payload = AnswerVerificationInput(
            question=c["question"], candidate_answer=c["prediction"], context=c["context"]
        )
        # Byte-identical input, repeated. Sequential rather than n= so each call is
        # an independent request, matching how ``evaluate`` issues its three views.
        ratings: List[float] = []
        for _ in range(repeats):
            out, _ = await execute_role_lc(registry, VERIFIER, payload)
            if out is None or is_safe_default(out):
                continue
            r = getattr(out, "rating", None)
            if isinstance(r, (int, float)):
                ratings.append(float(r))
        if len(ratings) < 2:
            print(f"  case {i}: only {len(ratings)} parsed; skipped")
            continue
        spread = max(ratings) - min(ratings)
        sd = statistics.stdev(ratings)
        all_spreads.append(spread)
        all_sd.append(sd)
        print(
            f"  case {i} (sub_em={c['sub_em']:.0f}): ratings={ratings} "
            f"spread={spread:.1f} sd={sd:.2f}  | {c['question'][:52]}"
        )

    if not all_spreads:
        print("\nno case produced two parseable ratings")
        return 1
    mx = max(all_spreads)
    print("\n" + "=" * 74)
    print(f"mean spread {statistics.mean(all_spreads):.2f} | max spread {mx:.1f} "
          f"| mean sd {statistics.mean(all_sd):.2f}")
    print("=" * 74)
    print(
        f"A threshold at or below {mx:.1f} fires on identical inputs. "
        f"memory_disagreement_threshold must exceed the max spread to mean anything."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="langgraph_coe/config.eval.yaml")
    ap.add_argument("--run", default="results/d1", help="finished run to draw cases from")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    return asyncio.run(run(args.config, args.run, args.cases, args.repeats))


if __name__ == "__main__":
    raise SystemExit(main())
