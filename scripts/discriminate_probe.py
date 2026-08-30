#!/usr/bin/env python
"""Offline probe: would a *discrimination* query fetch the fact that separates two rivals?

§13.8 measured that in 47% of conversion failures the answered memory line matches the
question better than the gold line does — the system chose the better-supported candidate
and the discriminating fact was simply never retrieved. This asks whether one extra,
narrowly-aimed query would have retrieved it.

The query is built mechanically from things already in scope at synthesis time: the
question, the answered rival, and the gold rival. No LLM, so the probe measures the
*retrieval surface*, not a model's ability to phrase a query.

Run BEFORE building anything. A 120-row A/B costs two hours; this costs a few minutes, and
if the discriminating fact is not on the surface then no amount of wiring will find it.

Usage:  python scripts/discriminate_probe.py [n]
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_coe.config import LangGraphCoeConfig  # noqa: E402
from langgraph_coe.tools.web import init_web_search, web_search  # noqa: E402

STOP = set(
    "the a an of in on at to for and or is was were are what which who where when did "
    "do does has have had by with from that this it its his her their been being as".split()
)


def _n(s: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def words(s: str) -> set:
    return {w for w in _n(s).split() if w not in STOP and len(w) > 2}


def _strip_tags(line: str) -> str:
    out = re.sub(r"^\[hop=\d+\]\s*", "", line.strip())
    return re.sub(r"^\[[A-Za-z ]+\]\s*:?\s*", "", out).strip()


def collect(runs: List[str], limit: int) -> List[dict]:
    """Conversion failures where the answered line out-matches the gold line."""
    cases: List[dict] = []
    for run in runs:
        path = f"results/{run}/evaluation_log.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip() or len(cases) >= limit:
                continue
            rec = json.loads(line)
            tp = (rec.get("artifacts") or {}).get("textual_memory_path")
            if not (tp and os.path.exists(tp)):
                continue
            if float(rec.get("sub_em_short") or 0.0) > 0:
                continue
            mem = [str(x) for x in json.load(open(tp))]
            ret = [l for l in mem if "[Retrieval]" in l]
            golds = [_n(g) for g in (rec.get("correct_answer") or [])]
            if not any(g and g in _n(" ".join(ret)) for g in golds):
                continue
            pred = _n(rec.get("predicted_answer"))
            if not pred or len(pred) < 4:
                continue
            gl = pl = None
            for l in ret:
                ln = _n(l)
                if gl is None and any(g and g in ln for g in golds):
                    gl = l
                if pl is None and pred in ln:
                    pl = l
            if not (gl and pl) or gl == pl:
                continue
            q = words(rec.get("question", ""))
            gs = len(q & words(gl))
            ps = len(q & words(pl))
            if ps <= gs:
                continue  # only the "evidence favoured the rival" slice
            cases.append(
                {
                    "question": rec["question"],
                    "gold": (rec.get("correct_answer") or [""])[0],
                    "pred": str(rec.get("predicted_answer"))[:60],
                    "gold_line": _strip_tags(gl),
                    "pred_line": _strip_tags(pl),
                }
            )
    return cases


def discrimination_query(case: dict) -> str:
    """Superseded — kept so the failed first form stays on the record.

    The first attempt was ``f"{question} {rival_a} or {rival_b}"``. It returned text
    containing *neither* rival on 12 of 14 probes: a 20-word multi-hop question is a bad
    search query, and appending "A or B" does not rescue it. So that run measured the query,
    not the retrieval surface, and :func:`rival_probe` replaces it.
    """
    q = case["question"].rstrip("?")
    return f"{q} {case['pred'].strip()} or {case['gold'].strip()}"


async def rival_probe(rival: str, question: str) -> float:
    """Fetch ``rival``'s own page and score it against the question's constraint.

    This is the discrimination that matters: the two rivals are both plausible answers, so
    the separating evidence is on *their* pages ("Cayce borders Columbia", "Lexington County
    does not"), not in a query that mentions both. Score is content-word overlap with the
    question, which is the same measure §13.8 used to establish that the wrong rival looked
    better — so a rise here is directly comparable to that baseline.
    """
    try:
        results = await web_search.ainvoke({"query": rival})
    except Exception:  # noqa: BLE001
        return -1.0
    blob = " ".join(f"{r.get('title','')} {r.get('content','')}" for r in results)
    if not blob.strip():
        return -1.0
    q = words(question)
    return len(q & words(blob)) / max(len(q), 1)


async def main(limit: int) -> None:
    runs = "gd_on gd_off vf_on vf_off bd_input bd_conf".split()
    cases = collect(runs, limit)
    cfg = LangGraphCoeConfig.from_yaml("langgraph_coe/config.eval.yaml")
    init_web_search(cfg.web_search)

    resolved = unresolved = unusable = 0
    print(f"probing {len(cases)} cases where the evidence favoured the wrong rival\n")
    for i, case in enumerate(cases, 1):
        gs = await rival_probe(case["gold"], case["question"])
        ps = await rival_probe(case["pred"], case["question"])
        if gs < 0 or ps < 0:
            unusable += 1
            verdict = "a rival page could not be fetched"
        elif gs > ps:
            resolved += 1
            verdict = "GOLD page fits the question better -> discriminable"
        else:
            unresolved += 1
            verdict = "rival page still fits as well or better"
        print(f"{i:>2}. {verdict}   (gold {gs:.2f} / rival {ps:.2f})")
        print(f"    q     : {case['question'][:96]}")
        print(f"    gold  : {case['gold'][:60]!r}   answered: {case['pred'][:50]!r}")
    n = max(resolved + unresolved + unusable, 1)
    print(f"\n── outcome over {n} probes ──")
    print(f"  discriminable: fetching each rival's page separates them: {resolved} ({100*resolved/n:.0f}%)")
    print(f"  not separated by the rivals' own pages:                  {unresolved} ({100*unresolved/n:.0f}%)")
    print(f"  page unfetchable:                                        {unusable} ({100*unusable/n:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
