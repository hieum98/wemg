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


async def compute_acc_batch_async(
    acc_tasks: List[Dict[str, Any]],
    llm_agent: BaseLLMAgent,
    max_concurrent: Optional[int] = None
) -> List[float]:
    """Compute Acc scores concurrently for multiple questions using async.
    
    Args:
        acc_tasks: List of dicts with keys: question, predicted_answer, correct_answers
        llm_agent: LLM agent for evaluation
        max_concurrent: Maximum number of concurrent tasks (None = use min(8, len(acc_tasks)))
        
    Returns:
        List of accuracy scores (0-1) in same order as acc_tasks
    """
    if not acc_tasks:
        return []
    
    max_concurrent = max_concurrent or min(len(acc_tasks), 8)
    max_concurrent = max(1, min(max_concurrent, len(acc_tasks)))
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def compute_single_async(task, idx):
        async with semaphore:
            # Run sync compute_acc in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    None,
                    compute_acc,
                    task['question'],
                    task['predicted_answer'],
                    task['correct_answers'],
                    llm_agent,
                    None  # interaction_memory
                )
            except Exception as e:
                logger.error(f"Error computing Acc for task {idx}: {e}")
                return 0.0
    
    # Create tasks for all acc computations, wrapping to include index
    async def compute_with_idx(task, idx):
        try:
            result = await compute_single_async(task, idx)
            return idx, result
        except Exception as e:
            logger.error(f"Error computing Acc for task {idx}: {e}")
            return idx, 0.0
    
    tasks = [asyncio.create_task(compute_with_idx(task, idx)) for idx, task in enumerate(acc_tasks)]
    
    # Execute with progress tracking using as_completed for incremental updates
    results = [0.0] * len(acc_tasks)
    
    with tqdm(total=len(acc_tasks), desc="Computing Acc scores", unit="question") as pbar:
        # Process tasks as they complete (semaphore handles concurrency)
        for done_coro in asyncio.as_completed(tasks):
            try:
                idx, result = await done_coro
                results[idx] = result if not isinstance(result, Exception) else 0.0
            except Exception as e:
                logger.error(f"Error awaiting Acc computation: {e}")
            pbar.update(1)
    
    return results


def compute_acc_batch(
    acc_tasks: List[Dict[str, Any]],
    llm_agent: BaseLLMAgent,
    max_workers: Optional[int] = None
) -> List[float]:
    """Compute Acc scores in parallel for multiple questions (sync wrapper).
    
    Args:
        acc_tasks: List of dicts with keys: question, predicted_answer, correct_answers
        llm_agent: LLM agent for evaluation
        max_workers: Maximum number of parallel workers (None = use len(acc_tasks))
        
    Returns:
        List of accuracy scores (0-1) in same order as acc_tasks
    """
    # Use async version with single event loop
    return asyncio.run(compute_acc_batch_async(acc_tasks, llm_agent, max_workers))


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
        max_concurrent_questions: Optional[int] = None
    ) -> datasets.Dataset:
        """Evaluate dataset and compute metrics with async processing.
        
        Uses async processing with a single event loop for efficient I/O and proper
        rate limiting. The same entry point (system.answer()) is used for both
        inference and evaluation.
        
        Args:
            dataset: HuggingFace dataset with questions and answers
            output_path: Path to save results (directory)
            resume: Whether to resume from existing results
            question_column: Name of question column in dataset
            answer_column: Name of answer column in dataset
            max_concurrent_questions: Maximum concurrent questions (None = auto-detect from config, default: 8)
            
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
        
        # Process questions using async with single event loop
        dataset_size = len(dataset)
        # Determine max concurrent questions
        if max_concurrent_questions is None:
            max_concurrent_questions = min(8, len(tasks))
        
        # Use async processing with single event loop
        asyncio.run(self._process_parallel_async(
            tasks, results, log_path, max_concurrent_questions, dataset_size
        ))
        
        # Check for missing acc_scores in loaded results and compute them
        acc_tasks_to_compute = []
        for idx, result in enumerate(results):
            if result is not None:
                predicted_answer = result.get('predicted_answer')
                acc_score = result.get('acc_score')
                error = result.get('error')
                # If we have a predicted answer but no acc_score and no error, compute it
                if predicted_answer and acc_score is None and error is None:
                    example = dataset[idx]
                    question = example.get(question_column, "")
                    answer = example.get(answer_column, None)
                    if question and answer:
                        acc_tasks_to_compute.append({
                            'idx': idx,
                            'question': question,
                            'predicted_answer': predicted_answer,
                            'correct_answers': self._normalize_answer_field(answer)
                        })
        
        # Compute missing acc scores
        if acc_tasks_to_compute:
            logger.info(f"Computing missing Acc scores for {len(acc_tasks_to_compute)} questions...")
            if max_concurrent_questions is None:
                max_concurrent_questions = min(8, len(acc_tasks_to_compute))
            acc_scores = asyncio.run(compute_acc_batch_async(
                [{
                    'question': t['question'],
                    'predicted_answer': t['predicted_answer'],
                    'correct_answers': t['correct_answers']
                } for t in acc_tasks_to_compute],
                self.system.llm_agent,
                max_concurrent=max_concurrent_questions
            ))
            
            for task, score in zip(acc_tasks_to_compute, acc_scores):
                if results[task['idx']] is not None:
                    results[task['idx']]['acc_score'] = score
                    if log_path:
                        self._log_result(task['idx'], results[task['idx']], log_path)
        
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
        max_concurrent_questions: Optional[int] = None,
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

        # Second pass: compute Acc concurrently using async (optional)
        if compute_acc_scores and acc_tasks:
            logger.info(f"Computing Acc scores for {len(acc_tasks)} questions concurrently...")
            if max_concurrent_questions is None:
                # Get LLM concurrency from config if available
                try:
                    from omegaconf import OmegaConf
                    llm_concurrency = OmegaConf.select(self.system.cfg, "llm.concurrency") or 64
                except:
                    llm_concurrency = 64
                max_concurrent = min(8, llm_concurrency, len(acc_tasks))
            else:
                max_concurrent = max_concurrent_questions
            acc_scores = asyncio.run(compute_acc_batch_async(
                [
                    {
                        "question": t["question"],
                        "predicted_answer": t["predicted_answer"],
                        "correct_answers": t["correct_answers"],
                    }
                    for t in acc_tasks
                ],
                self.system.llm_agent,
                max_concurrent=max_concurrent,
            ))

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
    
    async def _process_parallel_async(
        self,
        tasks: List[tuple],
        results: List[Optional[Dict]],
        log_path: Optional[Path],
        max_concurrent: int,
        dataset_size: int
    ):
        """Process questions concurrently using async with single event loop.
        
        Args:
            tasks: List of tuples (idx, example, question, answer, question_id)
            results: List to store results (indexed by question idx)
            log_path: Optional path to log file
            max_concurrent: Maximum number of concurrent questions to process
            dataset_size: Total dataset size for logging
        """
        if not tasks:
            return
        
        logger.info(f"Processing {len(tasks)} questions with max_concurrent={max_concurrent} (async)")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        acc_tasks = []
        
        async def process_task_async(task_data):
            idx, example, question, answer, question_id = task_data
            async with semaphore:
                try:
                    logger.info(f"Processing question {idx + 1}/{dataset_size}: {question[:50]}...")
                    # Run sync _process_single_question in thread pool
                    # This allows I/O operations inside system.answer() to happen concurrently
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,  # Use default ThreadPoolExecutor
                        self._process_single_question,
                        question,
                        answer,
                        str(question_id),
                        False  # compute_acc_now
                    )
                    merged_result = {**example, **result}
                    return idx, merged_result
                except Exception as e:
                    logger.error(f"Error processing question {idx}: {e}", exc_info=True)
                    error_result = {**example, **self._create_empty_result()}
                    error_result['error'] = str(e)
                    return idx, error_result
        
        # Create all async tasks - semaphore will limit concurrency automatically
        # Tasks start immediately as semaphore slots become available
        async_tasks = [asyncio.create_task(process_task_async(task)) for task in tasks]
        
        # Process with progress tracking - no batching, process as tasks complete
        with tqdm(total=len(tasks), desc="Evaluating", unit="question") as pbar:
            # Use as_completed so we process results as they finish (no waiting for batches)
            # This allows new questions to start immediately when a slot opens
            for done_coro in asyncio.as_completed(async_tasks):
                try:
                    result = await done_coro
                    # Result is always a tuple (idx, merged_result) from process_task_async
                    # Even on error, process_task_async returns (idx, error_result)
                    result_idx, merged_result = result
                    idx = result_idx
                    
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
                    # If exception occurs, log it but don't crash the whole evaluation
                    logger.error(f"Error processing result: {e}", exc_info=True)
                    pbar.update(1)
        
        # Compute Acc scores asynchronously
        await self._compute_and_update_acc_scores_async(acc_tasks, results, log_path, max_concurrent)
    
    async def _compute_and_update_acc_scores_async(
        self,
        acc_tasks: List[Dict[str, Any]],
        results: List[Optional[Dict]],
        log_path: Optional[Path],
        max_concurrent: Optional[int]
    ):
        """Compute Acc scores concurrently using async and update results."""
        if not acc_tasks:
            return
        
        logger.info(f"Computing Acc scores for {len(acc_tasks)} questions concurrently...")
        acc_scores = await compute_acc_batch_async(
            [{
                'question': t['question'],
                'predicted_answer': t['predicted_answer'],
                'correct_answers': t['correct_answers']
            } for t in acc_tasks],
            self.system.llm_agent,
            max_concurrent=max_concurrent
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

