"""Evaluation metrics: Sub-EM, Acc (LLM-based), Pass@k.

Pass@k values are computed from search metadata where ``pass_at_k`` marks
the first step/iteration with evaluator rating > 8.0 (Acc > 0.8).
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


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


async def compute_acc(
    question: str,
    predicted: str,
    correct_answers: Union[str, List[str]],
    client,
    interaction_memory=None,
) -> float:
    """LLM-based accuracy: evaluate predicted answer using Evaluator role, return normalized score [0, 1]."""
    from wemg.llm.roles import EVALUATOR, AnswerEvaluationInput, execute_role
    
    if isinstance(correct_answers, list):
        correct_str = "; ".join(correct_answers)
    else:
        correct_str = correct_answers or "Not available"
    
    eval_input = AnswerEvaluationInput(
        user_question=question,
        system_answer=predicted,
        correct_answer=correct_str,
    )
    
    try:
        results, _ = await execute_role(
            client=client, role=EVALUATOR, input_data=eval_input,
            interaction_memory=interaction_memory, n=1,
        )
        if results:
            return results[0].rating / 10.0
    except Exception as e:
        logger.error(f"Acc computation failed: {e}")
    return 0.0


async def compute_acc_batch(
    tasks: List[Tuple[str, str, Union[str, List[str]]]],
    client,
    max_concurrent: int = 10,
) -> List[float]:
    """Batch compute Acc scores with concurrency control.
    
    tasks: List of (question, predicted, correct_answers) tuples.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _one(question, predicted, correct):
        async with semaphore:
            return await compute_acc(question, predicted, correct, client)
    
    return await asyncio.gather(*[_one(q, p, c) for q, p, c in tasks])


def compute_aggregate_metrics(
    sub_ems: List[float],
    accs: List[float],
    pass_at_k_values: List[Optional[int]],
    max_k: int = 10,
) -> Dict:
    """Compute aggregate metrics from per-question results.

    ``pass_at_k_values`` contains the earliest step/iteration index where the
    evaluator-threshold pass condition was met for each question.
    """
    valid_sub_ems = [s for s in sub_ems if s is not None]
    valid_accs = [a for a in accs if a is not None]
    valid_pass = [p for p in pass_at_k_values if p is not None]
    
    metrics = {
        "mean_sub_em": sum(valid_sub_ems) / len(valid_sub_ems) if valid_sub_ems else 0.0,
        "mean_acc": sum(valid_accs) / len(valid_accs) if valid_accs else 0.0,
        "total_questions": len(sub_ems),
        "valid_questions": len(valid_sub_ems),
    }
    
    if valid_pass:
        metrics["overall_pass_rate"] = len(valid_pass) / len(sub_ems)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(sub_ems)
    
    return metrics


def compute_aggregate_metrics_both(
    sub_ems_short: List[float],
    sub_ems_long: List[float],
    accs_short: List[float],
    accs_long: List[float],
    pass_at_k_values: List[Optional[int]],
    max_k: int = 10,
) -> Dict:
    """Compute aggregate metrics for both short and long answer versions.
    
    Returns metrics dict with separate entries for short and long versions.
    """
    # Short answer metrics
    valid_sub_ems_short = [s for s in sub_ems_short if s is not None]
    valid_accs_short = [a for a in accs_short if a is not None]
    valid_pass = [p for p in pass_at_k_values if p is not None]
    
    metrics_short = {
        "mean_sub_em": sum(valid_sub_ems_short) / len(valid_sub_ems_short) if valid_sub_ems_short else 0.0,
        "mean_acc": sum(valid_accs_short) / len(valid_accs_short) if valid_accs_short else 0.0,
        "total_questions": len(sub_ems_short),
        "valid_questions": len(valid_sub_ems_short),
    }
    
    if valid_pass:
        metrics_short["overall_pass_rate"] = len(valid_pass) / len(sub_ems_short)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics_short[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(sub_ems_short)
    
    # Long answer metrics
    valid_sub_ems_long = [s for s in sub_ems_long if s is not None]
    valid_accs_long = [a for a in accs_long if a is not None]
    
    metrics_long = {
        "mean_sub_em": sum(valid_sub_ems_long) / len(valid_sub_ems_long) if valid_sub_ems_long else 0.0,
        "mean_acc": sum(valid_accs_long) / len(valid_accs_long) if valid_accs_long else 0.0,
        "total_questions": len(sub_ems_long),
        "valid_questions": len(valid_sub_ems_long),
    }
    
    if valid_pass:
        metrics_long["overall_pass_rate"] = len(valid_pass) / len(sub_ems_long)
        for k in range(1, min(max_k + 1, max(valid_pass) + 1)):
            metrics_long[f"pass_at_{k}"] = sum(1 for p in valid_pass if p <= k) / len(sub_ems_long)
    
    return {
        "short_answer": metrics_short,
        "long_answer": metrics_long,
    }
