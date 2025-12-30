"""Dataset evaluation module for computing metrics on question-answer datasets.

This module provides functionality to evaluate WEMG system performance on
HuggingFace datasets, computing Acc, Sub-EM, and Pass-at-k metrics.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datasets
import asyncio
import threading
from tqdm import tqdm

from wemg.main import WEMGSystem
from wemg.agents.roles.evaluator import Evaluator, AnswerEvaluationInput
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.interaction_memory import InteractionMemory
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


def check_answer_correctness(
    predicted_answer: str,
    correct_answers: Union[str, List[str]]
) -> bool:
    """Check if predicted answer contains any correct answer (case-insensitive substring match).
    
    Args:
        predicted_answer: The predicted answer string
        correct_answers: Single correct answer string or list of correct answer strings
        
    Returns:
        True if predicted answer contains any correct answer, False otherwise
    """
    if not predicted_answer or not correct_answers:
        return False
    
    # Normalize to list
    if isinstance(correct_answers, str):
        correct_answers = [correct_answers]
    
    predicted_lower = predicted_answer.lower()
    for correct in correct_answers:
        if correct and correct.lower() in predicted_lower:
            return True
    
    return False


def compute_sub_em(
    predicted_answer: str,
    correct_answers: Union[str, List[str]]
) -> bool:
    """Compute Sub-EM (Substring Exact Match) metric.
    
    Checks if predicted answer contains ANY of the correct answers
    (case-insensitive substring match).
    
    Args:
        predicted_answer: The predicted answer string
        correct_answers: Single correct answer string or list of correct answer strings
        
    Returns:
        True if predicted answer contains any correct answer, False otherwise
    """
    return check_answer_correctness(predicted_answer, correct_answers)


def _run_async_in_thread(coro):
    """Run async coroutine in a new event loop for thread-safe execution."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def compute_acc(
    question: str,
    predicted_answer: str,
    correct_answers: Union[str, List[str]],
    llm_agent: BaseLLMAgent,
    interaction_memory: Optional[InteractionMemory] = None
) -> float:
    """Compute Acc (Accuracy) metric using Evaluator role (synchronous).
    
    Uses the Evaluator role to rate the answer (0-10) and normalizes to 0-1.
    If multiple correct answers, uses the first one for evaluation.
    
    Args:
        question: The original question
        predicted_answer: The predicted answer
        correct_answers: Single correct answer string or list of correct answer strings
        llm_agent: LLM agent for evaluation
        interaction_memory: Optional interaction memory
        
    Returns:
        Accuracy score normalized to 0-1 (rating / 10.0)
    """
    # Use first correct answer if list
    if isinstance(correct_answers, list):
        correct_answer = correct_answers[0] if correct_answers else "Not available"
    else:
        correct_answer = correct_answers or "Not available"
    
    eval_input = AnswerEvaluationInput(
        user_question=question,
        system_answer=predicted_answer,
        correct_answer=correct_answer
    )
    
    try:
        # Run async execute_role in thread-safe manner
        eval_results, _ = _run_async_in_thread(
            execute_role(
                llm_agent=llm_agent,
                role=Evaluator(),
                input_data=eval_input,
                interaction_memory=interaction_memory,
                n=1
            )
        )
        
        if eval_results and len(eval_results) > 0:
            rating = eval_results[0].rating
            # Normalize from 0-10 to 0-1
            return rating / 10.0
        else:
            logger.warning("Evaluator returned no results, returning 0.0")
            return 0.0
    except Exception as e:
        logger.error(f"Error computing Acc metric: {e}")
        return 0.0


def compute_acc_batch(
    acc_tasks: List[Dict[str, Any]],
    llm_agent: BaseLLMAgent,
    max_workers: Optional[int] = None
) -> List[float]:
    """Compute Acc scores in parallel for multiple questions.
    
    Args:
        acc_tasks: List of dicts with keys: question, predicted_answer, correct_answers
        llm_agent: LLM agent for evaluation
        max_workers: Maximum number of parallel workers (None = use len(acc_tasks))
        
    Returns:
        List of accuracy scores (0-1) in same order as acc_tasks
    """
    if not acc_tasks:
        return []
    
    max_workers = max_workers or min(len(acc_tasks), 8)
    max_workers = max(1, min(max_workers, len(acc_tasks)))
    
    def compute_single(task):
        return compute_acc(
            question=task['question'],
            predicted_answer=task['predicted_answer'],
            correct_answers=task['correct_answers'],
            llm_agent=llm_agent,
            interaction_memory=None
        )
    
    results = [0.0] * len(acc_tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(compute_single, task): idx
            for idx, task in enumerate(acc_tasks)
        }
        
        # Progress bar compatible with parallel processing
        with tqdm(total=len(acc_tasks), desc="Computing Acc scores", unit="question") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error computing Acc for task {idx}: {e}")
                    results[idx] = 0.0
                pbar.update(1)
    
    return results


class DatasetEvaluator:
    """Evaluator for processing datasets and computing metrics."""
    
    def __init__(self, system: WEMGSystem):
        """Initialize the dataset evaluator.
        
        Args:
            system: WEMGSystem instance for answering questions
        """
        self.system = system
        self._log_lock = threading.Lock()
        # Ensure system is initialized
        self.system._initialize()
    
    def _normalize_answer_field(self, answer: Any) -> List[str]:
        """Normalize answer field to handle various input formats.
        
        Args:
            answer: Answer field from dataset (can be str, list, or other)
            
        Returns:
            Normalized answer as list of strings (empty list if None)
        """
        if answer is None:
            return []
        if isinstance(answer, str):
            return [answer]
        if isinstance(answer, list):
            return [str(a) for a in answer if a is not None]
        return [str(answer)]
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result dictionary with default values."""
        return {
            'predicted_answer': None,
            'concise_answer': None,
            'acc_score': None,
            'sub_em': None,
            'pass_at_k': None,
            'error': None,
            'metadata': None
        }
    
    def _process_single_question(
        self,
        question: str,
        correct_answers: Union[str, List[str]],
        question_id: Optional[str] = None,
        compute_acc_now: bool = False
    ) -> Dict[str, Any]:
        """Process a single question and compute metrics.
        
        Args:
            question: The question to answer
            correct_answers: Correct answer(s)
            question_id: Optional question ID
            compute_acc_now: Whether to compute Acc immediately (False for batch processing)
            
        Returns:
            Dictionary with results and metrics
        """
        result = self._create_empty_result()
        
        try:
            correct_answers = self._normalize_answer_field(correct_answers)
            if not correct_answers:
                result['error'] = "No correct answers provided"
                return result
            
            answer_result = self.system.answer(
                question=question,
                question_id=question_id,
                golden_answer=correct_answers[0]
            )
            
            result['predicted_answer'] = answer_result.answer
            result['concise_answer'] = answer_result.concise_answer
            result['metadata'] = answer_result.metadata or {}
            
            if answer_result.metadata and 'pass_at_k' in answer_result.metadata:
                result['pass_at_k'] = answer_result.metadata['pass_at_k']
            
            if result['concise_answer']:
                result['sub_em'] = compute_sub_em(result['concise_answer'], correct_answers)
            
            if compute_acc_now and result['predicted_answer']:
                result['acc_score'] = compute_acc(
                    question=question,
                    predicted_answer=result['predicted_answer'],
                    correct_answers=correct_answers,
                    llm_agent=self.system.llm_agent,
                    interaction_memory=None
                )
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def evaluate(
        self,
        dataset: datasets.Dataset,
        output_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        question_column: str = "question",
        answer_column: str = "answer",
        batch_size: int = 1,
        max_workers: Optional[int] = None
    ) -> datasets.Dataset:
        """Evaluate dataset and compute metrics with batch processing.
        
        Args:
            dataset: HuggingFace dataset with questions and answers
            output_path: Path to save results (directory)
            resume: Whether to resume from existing results
            question_column: Name of question column in dataset
            answer_column: Name of answer column in dataset
            batch_size: Number of questions to process in parallel (1 = sequential)
            max_workers: Maximum number of parallel workers (None = use batch_size)
            
        Returns:
            Dataset with added evaluation columns
        """
        # Validate columns exist in dataset
        if question_column not in dataset.column_names:
            raise ValueError(f"Column '{question_column}' not found in dataset. Available columns: {dataset.column_names}")
        if answer_column not in dataset.column_names:
            raise ValueError(f"Column '{answer_column}' not found in dataset. Available columns: {dataset.column_names}")
        
        output_path = Path(output_path) if output_path else None
        
        # Load logged results if resuming
        logged_results = {}
        log_path = None
        if output_path:
            log_path = self._get_log_path(output_path)
            if resume and log_path.exists():
                logged_results = self._load_logged_results(log_path)
                if logged_results:
                    logger.info(f"Resuming evaluation: {len(logged_results)} questions already processed")
        
        # Prepare results list
        results: List[Optional[Dict]] = [None] * len(dataset)
        
        # Load logged results into results array with validation
        for idx, result in logged_results.items():
            if 0 <= idx < len(results):
                # Validate that the logged question matches the dataset question at this index
                # This ensures indices are correct even if dataset order changed
                logged_question = result.get(question_column)
                dataset_question = dataset[idx].get(question_column, "")
                
                # Only validate if both questions exist
                if logged_question is not None and dataset_question:
                    # Normalize for comparison (strip whitespace, case-insensitive)
                    logged_normalized = logged_question.strip().lower()
                    dataset_normalized = dataset_question.strip().lower()
                    
                    if logged_normalized != dataset_normalized:
                        logger.warning(
                            f"Logged question at index {idx} doesn't match dataset question. "
                            f"Logged: '{logged_question[:50]}...', Dataset: '{dataset_question[:50]}...'. "
                            f"Skipping this logged result to avoid incorrect matching."
                        )
                        continue
                elif logged_question is None:
                    # If question not in log, log a warning but still use it (might be from old log format)
                    logger.debug(f"Logged result at index {idx} doesn't have '{question_column}' field. Using anyway.")
                
                results[idx] = result
            else:
                logger.warning(f"Logged result index {idx} is out of bounds for dataset size {len(results)}. Skipping.")
        
        # Prepare tasks (questions to process)
        tasks = []
        for idx, example in enumerate(dataset):
            if idx in logged_results:
                logged_result = logged_results[idx]
                # Double-check the result is actually in results array (validation passed)
                if results[idx] is not None:
                    if logged_result.get('predicted_answer') is not None and logged_result.get('error') is None:
                        logger.debug(f"Skipping question {idx} (already processed)")
                        continue
            
            tasks.append((
                idx,
                example,
                example.get(question_column, ""),
                example.get(answer_column, None),
                example.get('id', f"question_{idx}")
            ))
        
        # Process questions
        dataset_size = len(dataset)
        self._process_parallel(tasks, results, log_path, batch_size, max_workers, dataset_size)
        
        # Create result dataset
        result_dataset = datasets.Dataset.from_list(results)
        
        # Save final results
        if output_path:
            self._save_results(result_dataset, output_path)
        
        return result_dataset

    def score_dataset_from_predictions(
        self,
        dataset: datasets.Dataset,
        output_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
        question_column: str = "question",
        answer_column: str = "answer",
        predicted_answer_column: str = "predicted_answer",
        concise_answer_column: Optional[str] = "concise_answer",
        pass_at_k_column: str = "pass_at_k",
        batch_size: int = 8,
        max_workers: Optional[int] = None,
        compute_acc_scores: bool = True,
        overwrite_existing_scores: bool = False,
    ) -> datasets.Dataset:
        """Compute metrics for a dataset that already contains model predictions.

        This does NOT call `self.system.answer()`. It only reads an existing
        prediction column (default: `predicted_answer`) and computes:
        - `sub_em` (using `concise_answer` if present else `predicted_answer`)
        - `acc_score` (optional; uses Evaluator role via LLM)
        - keeps `pass_at_k` if present in input

        The output dataset always contains the canonical columns used by
        `compute_aggregate_metrics`: `predicted_answer`, `concise_answer`,
        `sub_em`, `acc_score`, `pass_at_k`, `error`, `metadata`.
        """
        # Validate required columns exist
        for col in (question_column, answer_column, predicted_answer_column):
            if col not in dataset.column_names:
                raise ValueError(
                    f"Column '{col}' not found in dataset. Available columns: {dataset.column_names}"
                )

        output_path = Path(output_path) if output_path else None
        log_path = None
        logged_results: Dict[int, Dict[str, Any]] = {}
        if output_path:
            log_path = output_path / "scoring_log.jsonl"
            if resume and log_path.exists():
                logged_results = self._load_logged_results(log_path)
                if logged_results:
                    logger.info(f"Resuming scoring: {len(logged_results)} questions already logged")

        results: List[Optional[Dict[str, Any]]] = [None] * len(dataset)

        # Seed results from scoring log if resuming
        if logged_results:
            for idx, r in logged_results.items():
                if 0 <= idx < len(results):
                    results[idx] = r

        # Build tasks for per-item metric computation (Sub-EM, etc.)
        tasks: List[tuple] = []
        for idx, example in enumerate(dataset):
            # If resuming and we already have what we need, skip
            if resume and results[idx] is not None and not overwrite_existing_scores:
                already_has_sub_em = results[idx].get("sub_em") is not None
                already_has_acc = results[idx].get("acc_score") is not None
                if already_has_sub_em and ((not compute_acc_scores) or already_has_acc):
                    continue

            tasks.append((idx, example))

        # First pass: compute Sub-EM synchronously and collect Acc tasks
        acc_tasks: List[Dict[str, Any]] = []
        for idx, example in tqdm(tasks, desc="Scoring (Sub-EM)", unit="question"):
            merged = dict(example)
            result = self._create_empty_result()

            try:
                correct_answers = self._normalize_answer_field(example.get(answer_column, None))
                if not correct_answers:
                    raise ValueError("No correct answers provided")

                predicted_answer = example.get(predicted_answer_column, None)
                if predicted_answer is None or str(predicted_answer).strip() == "":
                    raise ValueError(f"No predicted answer in column '{predicted_answer_column}'")

                concise_answer = None
                if concise_answer_column and concise_answer_column in example:
                    concise_answer = example.get(concise_answer_column, None)

                # Canonical output fields
                result["predicted_answer"] = str(predicted_answer)
                result["concise_answer"] = str(concise_answer) if concise_answer is not None else None
                result["pass_at_k"] = example.get(pass_at_k_column, None) if pass_at_k_column in example else None

                # Sub-EM: prefer concise answer if available, else predicted answer
                sub_em_text = result["concise_answer"] or result["predicted_answer"]
                result["sub_em"] = compute_sub_em(sub_em_text, correct_answers)

                # Acc: compute later in batch unless already present
                existing_acc = example.get("acc_score", None)
                if overwrite_existing_scores:
                    existing_acc = None
                if compute_acc_scores and existing_acc is None:
                    acc_tasks.append(
                        {
                            "idx": idx,
                            "question": example.get(question_column, ""),
                            "predicted_answer": result["predicted_answer"],
                            "correct_answers": correct_answers,
                        }
                    )
                else:
                    result["acc_score"] = existing_acc

            except Exception as e:
                result["error"] = str(e)

            merged = {**merged, **result}
            results[idx] = merged
            if log_path:
                self._log_result(idx, merged, log_path)

        # Second pass: compute Acc in parallel (optional)
        if compute_acc_scores and acc_tasks:
            logger.info(f"Computing Acc scores for {len(acc_tasks)} questions in parallel...")
            acc_scores = compute_acc_batch(
                [
                    {
                        "question": t["question"],
                        "predicted_answer": t["predicted_answer"],
                        "correct_answers": t["correct_answers"],
                    }
                    for t in acc_tasks
                ],
                self.system.llm_agent,
                max_workers=max_workers or min(batch_size, 8),
            )

            for task, score in zip(acc_tasks, acc_scores):
                idx = task["idx"]
                if results[idx] is not None:
                    results[idx]["acc_score"] = score
                    if log_path:
                        self._log_result(idx, results[idx], log_path)

        result_dataset = datasets.Dataset.from_list(results)
        if output_path:
            self._save_results(result_dataset, output_path)
        return result_dataset
    
    def _process_parallel(
        self,
        tasks: List[tuple],
        results: List[Optional[Dict]],
        log_path: Optional[Path],
        batch_size: int,
        max_workers: Optional[int],
        dataset_size: int
    ):
        """Process questions in parallel (or sequentially if batch_size=1)."""
        # For batch_size=1, use 1 worker (sequential processing)
        if batch_size == 1:
            max_workers = 1
        else:
            max_workers = max_workers or min(batch_size, len(tasks))
            max_workers = max(1, min(max_workers, len(tasks)))
        
        logger.info(f"Processing {len(tasks)} questions with {max_workers} workers")
        
        def process_task(task_data):
            idx, example, question, answer, question_id = task_data
            try:
                logger.info(f"Processing question {idx + 1}/{dataset_size}: {question[:50]}...")
                result = self._process_single_question(
                    question=question,
                    correct_answers=answer,
                    question_id=str(question_id),
                    compute_acc_now=False
                )
                return idx, {**example, **result}
            except Exception as e:
                logger.error(f"Error processing question {idx}: {e}", exc_info=True)
                error_result = {**example, **self._create_empty_result()}
                error_result['error'] = str(e)
                return idx, error_result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_task, task): task[0]
                for task in tasks
            }
            
            acc_tasks = []
            
            # Progress bar compatible with parallel processing
            with tqdm(total=len(tasks), desc="Evaluating", unit="question") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, merged_result = future.result()
                        results[idx] = merged_result
                        
                        if log_path:
                            self._log_result(idx, merged_result, log_path)
                        
                        if merged_result.get('predicted_answer') and not merged_result.get('error'):
                            original_task = next((t for t in tasks if t[0] == idx), None)
                            if original_task:
                                acc_tasks.append({
                                    'idx': idx,
                                    'question': original_task[2],
                                    'predicted_answer': merged_result['predicted_answer'],
                                    'correct_answers': self._normalize_answer_field(original_task[3])
                                })
                    except Exception as e:
                        idx = future_to_idx[future]
                        logger.error(f"Error getting result for question {idx}: {e}")
                        error_result = {
                            **next((t[1] for t in tasks if t[0] == idx), {}),
                            **self._create_empty_result(),
                            'error': str(e)
                        }
                        results[idx] = error_result
                        if log_path:
                            self._log_result(idx, error_result, log_path)
                    
                    # Update progress bar after each task completes
                    pbar.update(1)
        
        self._compute_and_update_acc_scores(acc_tasks, results, log_path, max_workers)
    
    def _compute_and_update_acc_scores(
        self,
        acc_tasks: List[Dict[str, Any]],
        results: List[Optional[Dict]],
        log_path: Optional[Path],
        max_workers: Optional[int]
    ):
        """Compute Acc scores in parallel and update results."""
        if not acc_tasks:
            return
        
        logger.info(f"Computing Acc scores for {len(acc_tasks)} questions in parallel...")
        acc_scores = compute_acc_batch(
            [{
                'question': t['question'],
                'predicted_answer': t['predicted_answer'],
                'correct_answers': t['correct_answers']
            } for t in acc_tasks],
            self.system.llm_agent,
            max_workers=max_workers
        )
        
        for task, score in zip(acc_tasks, acc_scores):
            if results[task['idx']] is not None:
                results[task['idx']]['acc_score'] = score
                if log_path:
                    self._log_result(task['idx'], results[task['idx']], log_path)
    
    def _get_log_path(self, output_path: Path) -> Path:
        """Get the path to the evaluation log file."""
        return output_path / "evaluation_log.jsonl"
    
    def _load_logged_results(self, log_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load results from log file.
        
        If multiple entries exist for the same question_index, the last one is used
        (allowing for updates like Acc score computation).
        
        Returns:
            Dictionary mapping question index to result dict
        """
        logged_results = {}
        if not log_path.exists():
            return logged_results
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        idx = result.get('question_index')
                        if idx is not None:
                            logged_results[idx] = result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse log line: {e}")
            logger.info(f"Loaded {len(logged_results)} results from log file")
        except Exception as e:
            logger.warning(f"Could not load log file: {e}")
        
        return logged_results
    
    def _log_result(self, question_index: int, result: Dict[str, Any], log_path: Path):
        """Log a single question result to the log file (thread-safe)."""
        with self._log_lock:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_entry = {'question_index': question_index, **result}
                with open(log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                    f.flush()
            except Exception as e:
                logger.error(f"Could not log result for question {question_index}: {e}")
    
    def _save_results(self, result_dataset: datasets.Dataset, output_path: Path):
        """Save results to disk."""
        output_path.mkdir(parents=True, exist_ok=True)
        result_dataset.save_to_disk(str(output_path))
        logger.info(f"Saved results to {output_path}")
    
    def compute_aggregate_metrics(
        self,
        result_dataset: datasets.Dataset,
        max_k: int = 10
    ) -> Dict[str, float]:
        """Compute aggregate metrics for the whole dataset.
        
        Args:
            result_dataset: Dataset with evaluation results
            max_k: Maximum k for Pass@k statistics
            
        Returns:
            Dictionary with aggregate metrics
        """
        # Filter out error entries
        valid_results = [
            r for r in result_dataset
            if r.get('error') is None and r.get('predicted_answer') is not None
        ]
        
        if not valid_results:
            logger.warning("No valid results to compute metrics")
            return {
                'mean_acc': 0.0,
                'mean_sub_em': 0.0,
                'overall_pass_rate': 0.0,
                **{f'pass_at_{k}': 0.0 for k in range(1, max_k + 1)}
            }
        
        # Compute mean Acc
        acc_scores = [r['acc_score'] for r in valid_results if r.get('acc_score') is not None]
        mean_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
        
        # Compute mean Sub-EM
        sub_em_scores = [r['sub_em'] for r in valid_results if r.get('sub_em') is not None]
        mean_sub_em = sum(sub_em_scores) / len(sub_em_scores) if sub_em_scores else 0.0
        
        # Compute Pass@k statistics
        pass_at_k_values = [r['pass_at_k'] for r in valid_results if r.get('pass_at_k') is not None]
        pass_at_k_stats = {
            f'pass_at_{k}': sum(1 for pk in pass_at_k_values if pk is not None and pk <= k) / len(valid_results)
            if valid_results else 0.0
            for k in range(1, max_k + 1)
        }
        
        overall_pass_rate = len(pass_at_k_values) / len(valid_results) if valid_results else 0.0
        
        return {
            'mean_acc': mean_acc,
            'mean_sub_em': mean_sub_em,
            'overall_pass_rate': overall_pass_rate,
            **pass_at_k_stats,
            'total_questions': len(result_dataset),
            'valid_questions': len(valid_results),
            'error_questions': len(result_dataset) - len(valid_results)
        }
    
    def save_metrics(
        self,
        metrics: Dict[str, float],
        output_path: Union[str, Path],
        save_summary: bool = True
    ):
        """Save aggregate metrics to file.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics (JSON file)
            save_summary: Whether to also save human-readable summary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")
        
        # Save summary
        if save_summary:
            summary_path = output_path.with_suffix('.txt')
            with open(summary_path, 'w') as f:
                f.write("Dataset Evaluation Metrics Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Questions: {metrics.get('total_questions', 0)}\n")
                f.write(f"Valid Questions: {metrics.get('valid_questions', 0)}\n")
                f.write(f"Error Questions: {metrics.get('error_questions', 0)}\n\n")
                f.write("Metrics:\n")
                f.write(f"  Mean Acc: {metrics.get('mean_acc', 0):.4f}\n")
                f.write(f"  Mean Sub-EM: {metrics.get('mean_sub_em', 0):.4f}\n")
                f.write(f"  Overall Pass Rate: {metrics.get('overall_pass_rate', 0):.4f}\n\n")
                f.write("Pass@k Statistics:\n")
                for k in range(1, 11):
                    key = f'pass_at_{k}'
                    if key in metrics:
                        f.write(f"  Pass@{k}: {metrics[key]:.4f}\n")
            logger.info(f"Saved metrics summary to {summary_path}")

