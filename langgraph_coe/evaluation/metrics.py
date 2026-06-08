"""Evaluation metrics: Sub-EM, Acc (LLM-based), aggregates.

The pure scoring functions (``compute_sub_em``, ``compute_aggregate_metrics*``)
are byte-for-byte the same logic as ``coe.evaluation.metrics`` so the numbers
and the ``metrics.json`` shape are identical. Only the Acc judge differs: it runs
the ``langgraph_coe`` ``EVALUATOR`` role through a :class:`RoleModelRegistry`
instead of a ``coe`` ``LLMClient``. The rating→score mapping (``rating / 10.0``)
is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-EM (identical to coe.evaluation.metrics.compute_sub_em)
# ──────────────────────────────────────────────────────────────────────────────


def compute_sub_em(predicted: str, correct_answers: Union[str, List[str]]) -> float:
    """Substring exact match: 1.0 if any correct answer is a substring of predicted (case-insensitive)."""
    if not predicted or not correct_answers:
        return 0.0
    if isinstance(correct_answers, str):
        correct_answers = [correct_answers]
    predicted_lower = predicted.lower()
    for ans in correct_answers:
        if ans and ans.lower() in predicted_lower:
            return 1.0
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Acc — LLM judge via the langgraph_coe EVALUATOR role
# ──────────────────────────────────────────────────────────────────────────────


async def compute_acc(
    question: str,
    predicted: str,
    correct_answers: Union[str, List[str]],
    registry,
) -> float:
    """LLM-based accuracy via the ``EVALUATOR`` role; normalized score in [0, 1]."""
    from langgraph_coe.llm import execute_role_lc
    from langgraph_coe.roles import EVALUATOR, AnswerEvaluationInput

    correct_str = (
        "; ".join(correct_answers)
        if isinstance(correct_answers, list)
        else (correct_answers or "Not available")
    )
    try:
        out, _ = await execute_role_lc(
            registry,
            EVALUATOR,
            AnswerEvaluationInput(
                user_question=question,
                system_answer=predicted,
                correct_answer=correct_str,
            ),
            n=1,
        )
        return float(out.rating) / 10.0
    except Exception as e:  # noqa: BLE001 — neutral score on judge failure (coe parity)
        logger.error("Acc computation failed: %s", e)
        return 0.0


async def compute_acc_batch(
    tasks: List[Tuple[str, str, Union[str, List[str]]]],
    registry,
    max_concurrent: int = 10,
) -> List[float]:
    """Batch compute Acc scores, bounding concurrency with a semaphore.

    ``tasks``: list of ``(question, predicted, correct_answers)`` tuples. Unlike
    the legacy single batched-role call, items are scored as concurrent
    single-input role executions so many questions run in parallel up to
    ``max_concurrent`` (``execute_role_lc`` processes a list sequentially).
    """
    if not tasks:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def _one(
        question: str, predicted: str, correct_answers: Union[str, List[str]]
    ) -> float:
        async with sem:
            return await compute_acc(question, predicted, correct_answers, registry)

    return list(await asyncio.gather(*[_one(*t) for t in tasks]))


# ──────────────────────────────────────────────────────────────────────────────
# Aggregates (identical to coe.evaluation.metrics)
# ──────────────────────────────────────────────────────────────────────────────


def compute_aggregate_metrics(
    sub_ems: List[float],
    accs: List[float],
    pass_at_k_values: List[Optional[int]],
    max_k: int = 10,
) -> Dict:
    """Compute aggregate metrics from per-question results."""
    valid_sub_ems = [s for s in sub_ems if s is not None]
    valid_accs = [a for a in accs if a is not None]
    valid_pass = [p for p in pass_at_k_values if p is not None]

    metrics = {
        "mean_sub_em": sum(valid_sub_ems) / len(valid_sub_ems)
        if valid_sub_ems
        else 0.0,
        "mean_acc": sum(valid_accs) / len(valid_accs) if valid_accs else 0.0,
        "total_questions": len(sub_ems),
        "valid_questions": len(valid_sub_ems),
    }

    if valid_pass:
        metrics["overall_pass_rate"] = len(valid_pass) / len(sub_ems)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(
                sub_ems
            )

    return metrics


def compute_aggregate_metrics_both(
    sub_ems_short: List[float],
    sub_ems_long: List[float],
    accs_short: List[float],
    accs_long: List[float],
    pass_at_k_values: List[Optional[int]],
    max_k: int = 10,
) -> Dict:
    """Compute aggregate metrics for both short and long answer versions."""
    valid_sub_ems_short = [s for s in sub_ems_short if s is not None]
    valid_accs_short = [a for a in accs_short if a is not None]
    valid_pass = [p for p in pass_at_k_values if p is not None]

    metrics_short = {
        "mean_sub_em": sum(valid_sub_ems_short) / len(valid_sub_ems_short)
        if valid_sub_ems_short
        else 0.0,
        "mean_acc": sum(valid_accs_short) / len(valid_accs_short)
        if valid_accs_short
        else 0.0,
        "total_questions": len(sub_ems_short),
        "valid_questions": len(valid_sub_ems_short),
    }

    if valid_pass:
        metrics_short["overall_pass_rate"] = len(valid_pass) / len(sub_ems_short)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics_short[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(
                sub_ems_short
            )

    valid_sub_ems_long = [s for s in sub_ems_long if s is not None]
    valid_accs_long = [a for a in accs_long if a is not None]

    metrics_long = {
        "mean_sub_em": sum(valid_sub_ems_long) / len(valid_sub_ems_long)
        if valid_sub_ems_long
        else 0.0,
        "mean_acc": sum(valid_accs_long) / len(valid_accs_long)
        if valid_accs_long
        else 0.0,
        "total_questions": len(sub_ems_long),
        "valid_questions": len(valid_sub_ems_long),
    }

    if valid_pass:
        metrics_long["overall_pass_rate"] = len(valid_pass) / len(sub_ems_long)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics_long[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(
                sub_ems_long
            )

    return {
        "short_answer": metrics_short,
        "long_answer": metrics_long,
    }


def compute_aggregate_metrics_by_level(
    sub_ems_short: List[float],
    sub_ems_long: List[float],
    accs_short: List[Optional[float]],
    accs_long: List[Optional[float]],
    levels: List[str],
    pass_at_k_values: List[Optional[int]],
    max_k: int = 10,
) -> Dict:
    """Compute per-level metrics (overall + i.i.d. / compositional / zero-shot)."""
    known_levels = ["i.i.d.", "compositional", "zero-shot"]
    result: Dict = {}

    result["overall"] = compute_aggregate_metrics_both(
        sub_ems_short, sub_ems_long, accs_short, accs_long, pass_at_k_values, max_k
    )

    for lvl in known_levels:
        idxs = [i for i, lev in enumerate(levels) if lev == lvl]
        if not idxs:
            continue
        result[lvl] = compute_aggregate_metrics_both(
            [sub_ems_short[i] for i in idxs],
            [sub_ems_long[i] for i in idxs],
            [accs_short[i] for i in idxs],
            [accs_long[i] for i in idxs],
            [pass_at_k_values[i] for i in idxs],
            max_k,
        )

    return result
