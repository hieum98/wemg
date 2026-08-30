"""Collect raw web evidence for dataset gold answers, for human adjudication.

Some `bamboogle_hardmix.jsonl` golds are stale or contestable — the census behind
`docs/plan_channel_status_and_plan.md` found 30/62 rows resting on a definite
description that can fail, including 8 whose referent is time-dependent (e.g. "the
*current* tallest wooden lattice tower", whose gold is 1935). Scoring a framework
against a wrong gold measures the gold, not the framework.

This script does **not** decide anything. It gathers snippets so a reader can judge,
and deliberately does not ask an LLM to re-answer: a model-derived gold is no more
trustworthy than the dataset's, and would launder a guess into a ground truth.

Usage::

    python -m langgraph_coe.scripts.gather_gold_evidence \\
        --out /tmp/gold_evidence.json                     # all 62 rows
    python -m langgraph_coe.scripts.gather_gold_evidence \\
        --questions-from /tmp/disagreements.txt --out /tmp/ev.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _REPO_ROOT / "datasets" / "bamboogle_hardmix.jsonl"

logger = logging.getLogger("gold")

# Markers of a referent that can drift: superlatives, ordinals, and explicit
# uniqueness or recency claims. These are the rows where a gold fixed at authoring
# time is most likely to have gone stale.
_VOLATILE = re.compile(
    r"\b(current|currently|now|today|tallest|largest|longest|highest|fastest|"
    r"biggest|deepest|oldest|newest|most\b|only\b|first\b|second|third|fourth|"
    r"fifth|latest|best[- ]selling|top\b)\b",
    re.I,
)


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    with _DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_of(row: Dict[str, Any]) -> str:
    g = row.get("answer")
    if isinstance(g, list) and g:
        g = g[0]
    return str(g)


async def gather(rows: List[Dict[str, Any]], top_k: int, crawl: bool) -> List[Dict[str, Any]]:
    from langgraph_coe.config import WebSearchConfig
    from langgraph_coe.tools.web import init_web_search, web_search

    init_web_search(
        WebSearchConfig(enabled=True, api_key=None, top_k=top_k, crawl_full_text=crawl)
    )

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        question, gold = row["question"], gold_of(row)
        record: Dict[str, Any] = {
            "question": question,
            "dataset_gold": gold,
            "category": row.get("category"),
            # Flagged, not judged: a volatile phrasing is a reason to *look*, not a
            # verdict that the gold is wrong.
            "volatile_phrasing": bool(_VOLATILE.search(question)),
            "evidence": [],
            "errors": [],
        }
        # Two queries: the question as asked, and the question paired with its gold.
        # The second surfaces pages that corroborate or contradict the gold directly,
        # which the bare question often does not.
        for query in (question, f"{question} {gold}"):
            try:
                results = await web_search.ainvoke({"query": query})
            except Exception as exc:  # noqa: BLE001 — record and continue
                record["errors"].append(f"{type(exc).__name__}: {exc}")
                continue
            for r in results or []:
                if not isinstance(r, dict):
                    continue
                record["evidence"].append(
                    {
                        "query": query,
                        "title": str(r.get("title", ""))[:200],
                        "url": str(r.get("url", "")),
                        "snippet": str(r.get("snippet", ""))[:600],
                        # A slice of the body, when crawled: enough to see a date or
                        # a ranking without dumping whole pages into the file.
                        "text": str(r.get("full_text", ""))[:1200],
                    }
                )
        out.append(record)
        print(
            f"[{i}/{len(rows)}] {len(record['evidence'])} snippets  "
            f"{'(volatile)' if record['volatile_phrasing'] else '          '}  "
            f"{question[:64]}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="JSON file to write")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--no-crawl", action="store_true", help="snippets only, faster")
    ap.add_argument(
        "--volatile-only",
        action="store_true",
        help="only rows whose phrasing can drift (superlatives, ordinals, 'current')",
    )
    ap.add_argument(
        "--questions-from",
        help="file of questions (one per line) to restrict to — e.g. the rows the "
        "system disagreed with",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "httpcore", "primp", "ddgs", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    rows = load_rows()
    if args.questions_from:
        wanted = {
            ln.strip()
            for ln in Path(args.questions_from).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        rows = [r for r in rows if r["question"].strip() in wanted]
    if args.volatile_only:
        rows = [r for r in rows if _VOLATILE.search(r["question"])]

    print(f"gathering evidence for {len(rows)} rows", flush=True)
    records = asyncio.run(gather(rows, args.top_k, not args.no_crawl))
    Path(args.out).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    print(
        f"{sum(1 for r in records if r['volatile_phrasing'])}/{len(records)} rows have "
        "volatile phrasing — read those first"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
