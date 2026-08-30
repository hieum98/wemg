#!/usr/bin/env python
"""How often does a binding merely echo the intent's own input?

An intent like "Determine if the identified actor has a wife who is also an actress"
receives two candidate bindings:

    'Yes, Dennis Quaid is married to an actress.'   -> Dennis Quaid  (the INPUT)
    'Yes, Meg Ryan.'                               -> Meg Ryan      (the ANSWER)

Two distinct QIDs, so ``count_rival_referents`` calls the intent *contested* and it never
closes — at ``replan_max=0`` that is permanent. But Dennis Quaid is not a rival answer; he
is the referent the prerequisite intent already bound, restated by an answerer that
included the subject in its sentence. The same echo can also *close* an intent on the
wrong thing: a terminal whose only linked entity is its own input closes on the input.

This counts three things over the persisted ledgers:

  echo_contested — intents contested where >=1 binding echoes a transitive prerequisite's
                   referent, and removing the echoes would leave exactly one referent.
                   These are closures blocked by an artifact.
  echo_closed    — intents CLOSED on a binding that only echoes a prerequisite. These are
                   false closures, which terminate the loop early on the input.
  echo_any       — any binding that echoes, for the base rate.

Prerequisites are followed transitively through ``depends_on`` so a two-hop echo counts.

Usage:  python scripts/echo_rivals.py vf_on vf_off ...
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_coe.graphs.cot import count_rival_referents  # noqa: E402


def prereq_keys(ledger: List[dict], idx: int) -> Set[str]:
    """Every referent key bound by intents ``idx`` transitively depends on."""
    keys: Set[str] = set()
    seen: Set[int] = set()
    stack = [idx]
    while stack:
        i = stack.pop()
        if i in seen or not (0 <= i < len(ledger)):
            continue
        seen.add(i)
        dep = ledger[i].get("depends_on")
        if isinstance(dep, int) and dep not in seen:
            stack.append(dep)
            for b in ledger[dep].get("bindings") or []:
                if b.get("qid"):
                    keys.add(str(b["qid"]))
    return keys


def main(runs: List[str]) -> None:
    c: Counter = Counter()
    examples: List[str] = []
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
            for i, e in enumerate(ledger):
                binds = e.get("bindings") or []
                if not binds:
                    continue
                c["intents_with_bindings"] += 1
                pk = prereq_keys(ledger, i)
                if not pk:
                    continue
                echoes = [b for b in binds if str(b.get("qid")) in pk]
                rest = [b for b in binds if str(b.get("qid")) not in pk]
                if echoes:
                    c["echo_any"] += 1
                status = e.get("status")
                if status == "contested" and echoes:
                    c["echo_contested"] += 1
                    if len(count_rival_referents(rest)) == 1:
                        c["echo_contested_would_close"] += 1
                        if len(examples) < 8:
                            examples.append(
                                f"intent={str(e.get('intent'))[:70]!r}\n"
                                f"         echo={[str(b.get('surface'))[:52] for b in echoes]}\n"
                                f"         real={[str(b.get('surface'))[:52] for b in rest]}"
                            )
                if status == "closed" and echoes and not rest:
                    c["echo_closed_falsely"] += 1
    tot = max(c["intents_with_bindings"], 1)
    print(f"runs: {len(runs)}   intents with >=1 binding: {c['intents_with_bindings']}\n")
    print(f"  echoes a prerequisite referent            {c['echo_any']:>5}"
          f"  ({100 * c['echo_any'] / tot:.1f}%)")
    print(f"  contested WITH an echo                    {c['echo_contested']:>5}"
          f"  ({100 * c['echo_contested'] / tot:.1f}%)")
    print(f"    ...and would close if echoes dropped    {c['echo_contested_would_close']:>5}"
          f"  <-- closures blocked by an artifact")
    print(f"  CLOSED on nothing but an echo             {c['echo_closed_falsely']:>5}"
          f"  <-- false closure on the intent's own input")
    if examples:
        print("\n── examples of a contest that is really an echo ──")
        for e in examples:
            print(f"     {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
