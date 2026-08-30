"""End-to-end smoke test for the plan channel.

Not an accuracy run — a wiring check. It answers a small number of real
questions against real services and asserts the things that would be silently
broken by a mis-wiring:

1. A plan is produced and injected into the subquestion prompt.
2. **No plan sentence reaches ``text_memory``** — the channel constraint. An
   interrogative in memory is picked up by ``_reverify_memory`` as a retrieval
   query, then reaches the verifier as grounding and the synthesizer as a
   candidate answer.
3. The ledger records bindings, and ``plan_action`` is computed each hop.
4. Consolidation reports retractions (the ``evicted`` field parses and resolves).
5. Both strategies terminate and return an answer.

Default environment (``config.smoke.yaml``): Qwen3-32B on Bedrock for every role,
Wikidata for the KG, web search instead of the local Wikipedia corpus, no
reranker.

Usage::

    python -m langgraph_coe.scripts.smoke_test                  # cot + mcts
    python -m langgraph_coe.scripts.smoke_test --strategy cot
    python -m langgraph_coe.scripts.smoke_test --questions 1
    COE_SPARQL_ENDPOINT=https://query.wikidata.org/sparql \\
        python -m langgraph_coe.scripts.smoke_test

Exit code is non-zero if any check fails, so this is usable in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "config.smoke.yaml"

logger = logging.getLogger("smoke")


_DATASET = _REPO_ROOT / "datasets" / "bamboogle_hardmix.jsonl"

# Real rows from the committed eval set, chosen to exercise each trigger branch
# rather than to measure accuracy (4 questions cannot). Selected by question text so
# the sample is stable if the file is reordered, and each carries the reason it is
# here — a smoke test that only ever hits the happy path tells you nothing about the
# branches you just added.
SMOKE_QUESTIONS: List[Tuple[str, str]] = [
    # Chained, unambiguous. The control: entity binding + UPDATE should close both
    # intents and the trigger should stay silent.
    (
        "Who was the father of the father of computer science?",
        "entity binding / UPDATE",
    ),
    # Ordinal-then-chained with a bare-year answer. The QID discriminator is blind
    # here, so this is the literal-binding path.
    ("In what year was the tallest lattice tower completed?", "literal (year) binding"),
    # "the only cruise line that flies the American flag" — an explicit uniqueness
    # claim, i.e. the presupposition-failure shape. Exercises stall/contraction.
    (
        "In what country was the only cruise line that flies the American flag "
        "incorporated in?",
        "presupposition risk / stall",
    ),
    # Rank with no canonical authority: the classic two-referents case, so the most
    # likely place for contested discharge to fire.
    ("In what country is the third largest stadium in the world?", "contested rank"),
    # Three chained hops (rank -> album -> release date), so hop 2 cannot be asked
    # until hop 1 resolves. Exercises whether UPDATE's ``intermediate_answer``
    # actually surfaces a closed binding into a *later* hop's prompt — the earlier
    # rows all closed every intent in one hop, which never tests that.
    (
        "When did Nirvana's second most selling studio album come out?",
        "multi-hop / intermediate_answer",
    ),
]


def _load_questions(limit: int) -> List[Tuple[str, str, str]]:
    """``(question, gold, why_selected)`` for the smoke sample, from the real file.

    Gold answers come from the dataset rather than being restated here, so a row
    whose gold changes cannot silently disagree with what this asserts.
    """
    if not _DATASET.is_file():
        raise SystemExit(f"dataset not found: {_DATASET}")
    by_question: Dict[str, Any] = {}
    with _DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_question[row["question"].strip()] = row

    out: List[Tuple[str, str, str]] = []
    for question, why in SMOKE_QUESTIONS[:limit]:
        row = by_question.get(question.strip())
        if row is None:
            raise SystemExit(
                f"question not found in {_DATASET.name}: {question!r}\n"
                "Update SMOKE_QUESTIONS if the dataset changed."
            )
        gold = row.get("answer")
        gold = gold[0] if isinstance(gold, list) and gold else gold
        label = f"{why}, category={row.get('category', '?')}"
        out.append((question, str(gold), label))
    return out


def _load_config(strategy: Optional[str]) -> Any:
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml(_SMOKE_CONFIG)

    # The local QEndpoint is deployment-specific; allow an override so the smoke
    # test is runnable from a box without the tunnel.
    endpoint = os.environ.get("COE_SPARQL_ENDPOINT")
    if endpoint:
        cfg.wikidata.sparql_endpoint = endpoint or None
    if strategy:
        cfg.search.strategy = strategy

    # LiteLLM reads AWS_REGION_NAME; AWS_REGION / ~/.aws/config are the more
    # common spellings, so bridge them rather than failing with an opaque error.
    if not os.environ.get("AWS_REGION_NAME"):
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            os.environ["AWS_REGION_NAME"] = region
    return cfg


def _preflight(cfg: Any) -> List[str]:
    """Fail fast with actionable messages rather than deep inside a graph."""
    problems: List[str] = []

    model = cfg.llm.tiers["heavy"].model_name
    if model.startswith("bedrock/"):
        try:
            import boto3  # noqa: F401
        except ImportError:
            problems.append(
                "boto3 is not installed; LiteLLM needs it to sign Bedrock "
                "requests. Install with: uv pip install boto3"
            )
        if not (
            os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_REGION")
            or (Path.home() / ".aws" / "config").is_file()
        ):
            problems.append("No AWS region configured (AWS_REGION / ~/.aws/config).")
        if cfg.llm.tiers["heavy"].api_base:
            problems.append(
                f"api_base must be null for Bedrock, got "
                f"{cfg.llm.tiers['heavy'].api_base!r} — LiteLLM would try to "
                "reach that host instead of the provider endpoint."
            )

    endpoint = cfg.wikidata.sparql_endpoint
    if endpoint:
        import httpx

        from langgraph_coe.tools.wikidata_backend import DEFAULT_USER_AGENT

        try:
            resp = httpx.get(
                endpoint,
                params={"query": "SELECT ?s WHERE {?s ?p ?o} LIMIT 1"},
                headers={
                    "Accept": "application/sparql-results+json",
                    # The public endpoint answers 403 to an unidentified client,
                    # so probe with the same UA the real backend sends — otherwise
                    # preflight reports a failure the app would not hit.
                    "User-Agent": DEFAULT_USER_AGENT,
                },
                timeout=15.0,
            )
            if resp.status_code >= 400:
                problems.append(
                    f"SPARQL endpoint {endpoint} returned HTTP {resp.status_code}"
                )
        except Exception as exc:
            problems.append(
                f"SPARQL endpoint {endpoint} unreachable ({type(exc).__name__}). "
                "Set COE_SPARQL_ENDPOINT, or null for the public endpoint."
            )
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────


def _check_plan_channel(state: Dict[str, Any]) -> List[str]:
    """The invariants that a mis-wiring would break silently."""
    failures: List[str] = []
    plan = str(state.get("plan") or "")
    if not plan.strip():
        failures.append("no plan was produced (PLANNER returned nothing usable)")
        return failures

    ledger = state.get("plan_ledger") or []
    if not ledger:
        failures.append("plan_ledger is empty — intents were never recorded")

    # THE channel constraint. Compare on sentence fragments rather than the whole
    # plan: consolidation rewrites items, so a leak would arrive paraphrased.
    memory_blob = "\n".join(str(m) for m in (state.get("text_memory") or [])).lower()
    fragments = [
        frag.strip().lower()
        for frag in plan.replace("\n", ". ").split(".")
        if len(frag.strip()) >= 40
    ]
    for frag in fragments:
        if frag in memory_blob:
            failures.append(
                f"PLAN LEAKED INTO text_memory: {frag[:90]!r} — an interrogative "
                "in memory becomes a retrieval query, then verifier grounding, "
                "then a synthesis candidate"
            )
            break
    return failures


def _dump_state(
    dump_dir: Path, strategy: str, question: str, result: Any, state: Dict[str, Any]
) -> None:
    """Write the plan, ledger, trajectory and tree for one question.

    Mirrors what ``runner._save_question_artifacts`` persists during an eval, so a
    single smoke question can be inspected without running the whole harness. The
    MCTS tree is flattened to ``node_id -> {parent, type, content, visits, value}``
    because the raw state holds it as a flat dict keyed by id.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in question.lower())[:50]
    path = dump_dir / f"{strategy}_{slug}.json"
    tree = state.get("tree") or {}
    payload = {
        "question": question,
        "strategy": strategy,
        "answer": result.concise_answer or result.answer,
        "plan": state.get("plan"),
        "plan_version": state.get("plan_version"),
        "plan_ledger": state.get("plan_ledger") or [],
        "plan_action_log": state.get("plan_action_log") or [],
        "plan_attempts_log": state.get("plan_attempts_log") or [],
        "iteration_history": state.get("iteration_history") or [],
        "text_memory": state.get("text_memory") or [],
        "tree": {
            nid: {
                "parent_id": n.get("parent_id"),
                "node_type": getattr(n.get("node_type"), "value", n.get("node_type")),
                "content": n.get("content"),
                "visits": n.get("visits"),
                "value": n.get("value"),
                "children_ids": n.get("children_ids"),
            }
            for nid, n in tree.items()
        },
        "current_path": state.get("current_path") or [],
        "root_id": state.get("root_id"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"    dumped -> {path}", flush=True)


def _summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    ledger = state.get("plan_ledger") or []
    action_log = state.get("plan_action_log") or []
    return {
        "plan_chars": len(str(state.get("plan") or "")),
        "plan_version": state.get("plan_version"),
        "intents": len(ledger),
        "closed": sum(1 for e in ledger if e.get("status") == "closed"),
        "contested": sum(1 for e in ledger if e.get("status") == "contested"),
        "bindings": sum(len(e.get("bindings") or []) for e in ledger),
        "attempts": sum(len(e.get("attempts") or []) for e in ledger),
        "stalled": sum(1 for e in ledger if e.get("stalled")),
        "falsified": sum(1 for e in ledger if e.get("falsified")),
        "gate_calls": len(action_log),
        "replan_signals": sum(1 for e in action_log if e.get("action") == "replan"),
        "updates": sum(1 for e in action_log if e.get("action") == "update"),
        # Which branch fired, so a fire is attributable rather than just counted.
        "reasons": sorted(
            {e.get("reason") for e in action_log if e.get("action") == "replan"}
        ),
        # Attribution health: a high unattributed rate means the gate reads noise.
        "unattributed": sum(int(e.get("answers_unattributed") or 0) for e in action_log),
        "low_conf": sum(int(e.get("answers_low_confidence") or 0) for e in action_log),
        "retractions": len(state.get("last_retractions") or []),
        "memory_items": len(state.get("text_memory") or []),
        "graph_nodes": getattr(state.get("graph_memory"), "number_of_nodes", lambda: 0)(),
        "linked_entities": len(state.get("entity_dict") or {}),
    }


async def _run_one(
    cfg: Any,
    question: str,
    expect: str,
    why: str,
    dump_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from langgraph_coe.system import answer

    started = time.monotonic()
    row: Dict[str, Any] = {
        "question": question,
        "strategy": cfg.search.strategy,
        "why": why,
    }
    try:
        result = await answer(question, cfg)
    except Exception as exc:  # noqa: BLE001 — the smoke test reports, never raises
        logger.exception("answer() raised")
        row.update(
            {
                "ok": False,
                "failures": [f"answer() raised {type(exc).__name__}: {exc}"],
                "elapsed_s": round(time.monotonic() - started, 1),
            }
        )
        return row

    state = (result.metadata or {}).get("_raw_state") or {}
    if dump_dir is not None:
        _dump_state(dump_dir, cfg.search.strategy, question, result, state)
    failures = _check_plan_channel(state)
    if not (result.concise_answer or result.answer).strip():
        failures.append("empty answer returned")

    row.update(
        {
            "ok": not failures,
            "failures": failures,
            "answer": result.concise_answer or result.answer,
            "expected_substring": expect,
            # Reported, never asserted: 2 questions cannot measure accuracy.
            "contains_expected": expect.lower()
            in (result.concise_answer or result.answer or "").lower(),
            "elapsed_s": round(time.monotonic() - started, 1),
            "metadata": {
                k: v for k, v in (result.metadata or {}).items() if k != "_raw_state"
            },
            "plan": str(state.get("plan") or ""),
            "summary": _summarize(state),
        }
    )
    return row


def _report(rows: List[Dict[str, Any]]) -> bool:
    print("\n" + "=" * 78)
    print("SMOKE TEST REPORT")
    print("=" * 78)
    all_ok = True
    for row in rows:
        status = "PASS" if row.get("ok") else "FAIL"
        if not row.get("ok"):
            all_ok = False
        print(f"\n[{status}] ({row['strategy']}, {row.get('elapsed_s')}s) {row['question']}")
        if row.get("why"):
            print(f"  selected for      : {row['why']}")
        if row.get("answer") is not None:
            print(f"  answer            : {str(row['answer'])[:160]}")
            print(
                f"  contains {row['expected_substring']!r:12}: "
                f"{row.get('contains_expected')}   (reported, not asserted)"
            )
        if row.get("plan"):
            first = str(row["plan"]).strip().splitlines()[0]
            print(f"  plan (first line) : {first[:150]}")
        summary = row.get("summary") or {}
        if summary:
            print("  " + "  ".join(f"{k}={v}" for k, v in summary.items()))
        meta = row.get("metadata") or {}
        if meta:
            print(f"  metadata          : {meta}")
        for failure in row.get("failures") or []:
            print(f"  !! {failure}")

    print("\n" + "-" * 78)
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"{n_ok}/{len(rows)} checks passed")
    # Accuracy is reported separately and never gates the smoke test.
    n_hit = sum(1 for r in rows if r.get("contains_expected"))
    print(f"{n_hit}/{len(rows)} answers contained the expected substring (informational)")
    print("-" * 78)
    return all_ok


async def _main_async(args: argparse.Namespace) -> int:
    strategies = ["cot", "mcts"] if args.strategy == "both" else [args.strategy]
    questions = _load_questions(args.questions)

    cfg0 = _load_config(strategies[0])
    problems = _preflight(cfg0)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  !! {p}")
        return 2
    print("preflight ok", flush=True)
    print(f"  model     : {cfg0.llm.tiers['heavy'].model_name}")
    print(f"  sparql    : {cfg0.wikidata.sparql_endpoint or 'public endpoint'}")
    print(f"  corpus    : {'on' if cfg0.retriever.enabled else 'off'}")
    print(f"  reranker  : {'on' if cfg0.reranker.enabled else 'off'}")
    print(f"  web search: {'on' if cfg0.web_search.enabled else 'off'}")
    print(
        f"  plan      : enabled={cfg0.search.plan.enabled} "
        f"replan_max={cfg0.search.plan.replan_max} "
        f"branch_local_memory={cfg0.search.mcts.branch_local_memory}"
    )

    rows: List[Dict[str, Any]] = []
    for strategy in strategies:
        cfg = _load_config(strategy)
        for question, expect, why in questions:
            print(f"\n>>> [{strategy}] {question}", flush=True)
            print(f"    ({why}; gold={expect!r})", flush=True)
            row = await _run_one(
                cfg, question, expect, why,
                Path(args.dump) if args.dump else None,
            )
            rows.append(row)
            print(
                f"    -> {'ok' if row.get('ok') else 'FAILED'} in "
                f"{row.get('elapsed_s')}s",
                flush=True,
            )

    return 0 if _report(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", choices=["cot", "mcts", "both"], default="both"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=len(SMOKE_QUESTIONS),
        help="how many of the dataset rows to run",
    )
    parser.add_argument(
        "--dump",
        metavar="DIR",
        help="write plan/ledger/trajectory/tree JSON per question",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # These are chatty at INFO and drown the report.
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
