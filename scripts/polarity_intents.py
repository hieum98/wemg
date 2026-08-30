#!/usr/bin/env python
"""Polarity (yes/no) intents: how many, and what damage do they do?

The PLANNER prompt asks for presuppositions to be hedged into conditionals —
*"determine whether she has a husband; if so, find his birthplace"* — so the ledger is
full of intents whose answer is a TRUTH VALUE, not a referent. Everything downstream of
the ledger assumes a referent:

  * ``apply_bindings`` resolves the answer to a QID/literal/phrase and calls that the
    intent's *referent*. For "Is Yangzhou a capital city?" the only linked entity in
    "No, Yangzhou is not a capital city" is Yangzhou — the intent's own input.
  * ``plan_target_resolved`` terminates the loop once every TERMINAL intent is closed and
    grounded. A terminal polarity intent closes on a boolean, so the loop can stop
    before the intent that actually answers the question has closed.
  * ``resolved_findings`` hands each closed terminal's surface to synthesis as a
    candidate answer, so a sentence-shaped boolean competes with the real answer.

This measures the population and each of those three harms.

Usage:  python scripts/polarity_intents.py vf_on vf_off ...
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import List

# Interrogative shapes whose answer is a truth value. Anchored at the start (after an
# optional conditional prefix) so "Determine the city that borders X" does not match on
# a stray "if".
_POLARITY = re.compile(
    r"^(?:\s*(?:if|once|after)\b[^,;]*[,;]\s*)?"
    r"(?:determine|establish|verify|check|confirm|ascertain|assess)\s+"
    r"(?:whether|if)\b"
    r"|^\s*(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would)\b"
    r"|\bwhether\s+(?:or\s+not\s+)?\b",
    re.I,
)
_AFFIRM = re.compile(r"^\s*(?:yes|no|correct|incorrect|true|false)\b", re.I)


def is_polarity(intent: str) -> bool:
    return bool(_POLARITY.search((intent or "").strip()))


def main(runs: List[str]) -> None:
    c: Counter = Counter()
    term_examples: List[str] = []
    for run in runs:
        path = f"results/{run}/evaluation_log.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            p = (rec.get("artifacts") or {}).get("plan_path")
            if not p or not os.path.exists(p):
                continue
            try:
                ledger = json.load(open(p)).get("plan_ledger") or []
            except (json.JSONDecodeError, OSError):
                continue
            if not ledger:
                continue
            c["questions"] += 1
            dep = {
                e.get("depends_on")
                for e in ledger
                if isinstance(e.get("depends_on"), int)
            }
            q_has_term_pol = False
            for i, e in enumerate(ledger):
                c["intents"] += 1
                pol = is_polarity(str(e.get("intent") or ""))
                terminal = i not in dep
                if not pol:
                    continue
                c["polarity"] += 1
                if terminal:
                    c["polarity_terminal"] += 1
                    q_has_term_pol = True
                binds = e.get("bindings") or []
                if e.get("status") == "closed" and binds:
                    c["polarity_closed"] += 1
                    surf = str(binds[0].get("surface") or "")
                    if _AFFIRM.match(surf):
                        c["polarity_closed_on_affirmation"] += 1
                    if len(surf.split()) > 8:
                        c["polarity_closed_on_sentence"] += 1
                    if terminal:
                        c["polarity_terminal_closed"] += 1
                        if len(term_examples) < 8:
                            term_examples.append(
                                f"{str(e.get('intent'))[:66]!r}\n"
                                f"         closed on: {surf[:72]!r}\n"
                                f"         gold={rec.get('correct_answer')} "
                                f"pred={str(rec.get('predicted_answer'))[:40]!r}"
                            )
            if q_has_term_pol:
                c["questions_with_terminal_polarity"] += 1
                if float(rec.get("sub_em_short") or 0.0) > 0:
                    c["qwtp_correct"] += 1
            else:
                c["questions_without"] += 1
                if float(rec.get("sub_em_short") or 0.0) > 0:
                    c["qwo_correct"] += 1

    q = max(c["questions"], 1)
    it = max(c["intents"], 1)
    pol = max(c["polarity"], 1)
    print(f"questions with a ledger: {c['questions']}   intents: {c['intents']}\n")
    print(f"  polarity (yes/no) intents                {c['polarity']:>5}"
          f"  ({100 * c['polarity'] / it:.1f}% of all intents)")
    print(f"  ...that are TERMINAL                     {c['polarity_terminal']:>5}"
          f"  ({100 * c['polarity_terminal'] / pol:.1f}% of polarity intents)")
    print(f"  ...closed with a binding                 {c['polarity_closed']:>5}")
    print(f"     of those, surface starts 'Yes/No/...' {c['polarity_closed_on_affirmation']:>5}"
          f"  <-- bound a truth value as a referent")
    print(f"     of those, surface is a full sentence  {c['polarity_closed_on_sentence']:>5}")
    print(f"  TERMINAL polarity intents closed         {c['polarity_terminal_closed']:>5}"
          f"  <-- can end the loop, and reaches synthesis as a candidate answer")
    print()
    a, an = c["questions_with_terminal_polarity"], c["questions_without"]
    print("── accuracy split by whether the plan has a TERMINAL polarity intent ──")
    print(f"  with    {c['qwtp_correct']}/{a} = {100 * c['qwtp_correct'] / max(a, 1):.1f}%")
    print(f"  without {c['qwo_correct']}/{an} = {100 * c['qwo_correct'] / max(an, 1):.1f}%")
    print("  (observational, not causal — hard questions may attract hedged plans)")
    if term_examples:
        print("\n── terminal polarity intents that closed ──")
        for e in term_examples:
            print(f"     {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
