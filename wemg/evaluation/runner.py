"""Dataset evaluation runner."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """Evaluates WEMG on datasets with resume support."""
    
    def __init__(self, system):
        """
        Args:
            system: WEMGSystem instance
        """
        self.system = system
    
    def evaluate(
        self,
        dataset,
        output_path: str = "./results",
        resume: bool = True,
        question_column: str = "question",
        answer_column: str = "answer",
        max_concurrent: Optional[int] = None,
        log_batch_size: Optional[int] = None,
    ) -> Dict:
        """Run evaluation: generate answers + compute metrics.
        
        Args:
            dataset: HuggingFace Dataset with question and answer columns
            output_path: Directory to save results
            resume: Whether to resume from previous run
            question_column: Column name for questions
            answer_column: Column name for correct answers
            max_concurrent: Max concurrent workers per answer_batch chunk. None uses
                min(chunk size, llm.concurrency, 8), same as answer_questions_batch.
            log_batch_size: How many unanswered questions to run per answer_batch before
                appending to the log. None defaults to the same cap as max_concurrent
                (min(llm.concurrency, 8)). Smaller values checkpoint more often if the
                process stops mid-run.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "evaluation_log.jsonl"
        
        # Load existing results for resume
        completed = {}
        if resume and log_file.exists():
            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line)
                    completed[entry["question"]] = entry
            logger.info(f"Resuming: {len(completed)} questions already processed")
        
        from wemg.evaluation.metrics import compute_sub_em
        
        n = len(dataset)
        sub_ems: List[float] = [0.0] * n
        accs: List[Optional[float]] = [None] * n
        pass_at_k_values: List[Optional[int]] = [None] * n
        predictions: Dict[str, str] = {}
        for q, entry in completed.items():
            predictions[q] = entry.get("predicted_answer", "")
        
        pending: List[tuple] = []
        for i, example in enumerate(dataset):
            question = example[question_column]
            correct = example[answer_column]
            if question in completed:
                entry = completed[question]
                sub_ems[i] = float(entry.get("sub_em", 0.0))
                accs[i] = entry.get("acc")
                pass_at_k_values[i] = entry.get("pass_at_k")
            else:
                pending.append((i, question, correct))
        
        def append_logs(entries: List[Dict[str, Any]]) -> None:
            with open(log_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                f.flush()
        
        def process_answer_results(
            batch: List[Tuple[int, str, Any]],
            results: List,
        ) -> List[Dict[str, Any]]:
            """Build log entries and update sub_ems / predictions for one answer_batch."""
            out: List[Dict[str, Any]] = []
            for (i, question, correct), result in zip(batch, results):
                err = result.metadata.get("error") if result.metadata else None
                if err:
                    logger.error(f"Error processing question {i}: {err}")
                    err_text = f"Error: {err}"
                    entry = {
                        "question": question,
                        "correct_answer": correct,
                        "predicted_answer": err_text,
                        "sub_em": 0.0,
                        "acc": None,
                        "pass_at_k": None,
                        "error": str(err),
                    }
                    sub_ems[i] = 0.0
                    pass_at_k_values[i] = None
                    predictions[question] = err_text
                else:
                    try:
                        predicted = result.concise_answer or result.answer
                        sub_em = compute_sub_em(predicted, correct)
                        pass_at_k = result.metadata.get("pass_at_k") if result.metadata else None
                        entry = {
                            "question": question,
                            "correct_answer": correct,
                            "predicted_answer": predicted,
                            "full_answer": result.answer,
                            "sub_em": sub_em,
                            "pass_at_k": pass_at_k,
                            "acc": None,
                        }
                        sub_ems[i] = sub_em
                        pass_at_k_values[i] = pass_at_k
                        predictions[question] = predicted
                    except Exception as e:
                        logger.error(f"Error processing question {i}: {e}")
                        err_text = f"Error: {e}"
                        entry = {
                            "question": question,
                            "correct_answer": correct,
                            "predicted_answer": err_text,
                            "sub_em": 0.0,
                            "acc": None,
                            "pass_at_k": None,
                            "error": str(e),
                        }
                        sub_ems[i] = 0.0
                        pass_at_k_values[i] = None
                        predictions[question] = err_text
                out.append(entry)
                logger.info(f"[{i+1}/{n}] Sub-EM: {sub_ems[i]:.1f} | Q: {question[:60]}...")
            return out
        
        if pending:
            cfg_cap = min(self.system.cfg.llm.concurrency, 8)
            if log_batch_size is not None:
                chunk_size = max(1, log_batch_size)
            elif max_concurrent is not None:
                chunk_size = max(1, max_concurrent)
            else:
                chunk_size = max(1, cfg_cap)
            total_pending = len(pending)
            for chunk_start in range(0, total_pending, chunk_size):
                batch = pending[chunk_start : chunk_start + chunk_size]
                questions = [p[1] for p in batch]
                qids = [str(p[0]) for p in batch]
                golds = [
                    list(c) if isinstance(c, (list, tuple)) else (c if isinstance(c, str) else str(c))
                    for _, _, c in batch
                ]
                mw = max_concurrent
                if mw is not None:
                    mw = max(1, min(mw, len(batch)))
                results = self.system.answer_batch(
                    questions,
                    question_ids=qids,
                    golden_answers=golds,
                    max_workers=mw,
                )
                entries = process_answer_results(batch, results)
                append_logs(entries)
        
        # Compute Acc scores in batch (async concurrent)
        acc_task_rows: List[tuple] = []
        for i, example in enumerate(dataset):
            if accs[i] is not None:
                continue
            question = example[question_column]
            correct = example[answer_column]
            predicted = predictions.get(question, "")
            acc_task_rows.append((i, question, predicted, correct))
        
        if acc_task_rows and self.system.client:
            from wemg.evaluation.metrics import compute_acc_batch
            acc_max = max_concurrent if max_concurrent is not None else 10
            tasks = [(q, pred, cor) for _, q, pred, cor in acc_task_rows]
            try:
                acc_results = asyncio.run(
                    compute_acc_batch(tasks, self.system.client, max_concurrent=acc_max)
                )
                for (idx, _, _, _), acc in zip(acc_task_rows, acc_results):
                    accs[idx] = acc
            except Exception as e:
                logger.error(f"Acc batch failed: {e}")
                from wemg.evaluation.metrics import compute_acc
                for idx, question, predicted, correct in acc_task_rows:
                    try:
                        accs[idx] = asyncio.run(
                            compute_acc(question, predicted, correct, self.system.client)
                        )
                    except Exception as e2:
                        logger.error(f"Acc computation failed for question {idx}: {e2}")
                        accs[idx] = 0.0
        
        # Compute aggregate metrics
        from wemg.evaluation.metrics import compute_aggregate_metrics
        metrics = compute_aggregate_metrics(sub_ems, accs, pass_at_k_values)
        
        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("WEMG Evaluation Results\n")
            f.write("=" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        
        logger.info(f"Evaluation complete. Metrics saved to {metrics_file}")
        return metrics
    
    def score_from_predictions(
        self,
        dataset,
        output_path: str = "./results",
        question_column: str = "question",
        answer_column: str = "answer",
        prediction_column: str = "predicted_answer",
    ) -> Dict:
        """Score existing predictions without running WEMG."""
        from wemg.evaluation.metrics import compute_sub_em, compute_aggregate_metrics, compute_acc_batch
        
        sub_ems = []
        accs = []
        
        for example in dataset:
            correct = example[answer_column]
            predicted = example.get(prediction_column, "")
            sub_ems.append(compute_sub_em(predicted, correct))
            accs.append(None)
        
        # Compute Acc if client available
        if self.system.client:
            tasks = [
                (example[question_column], example.get(prediction_column, ""), example[answer_column])
                for example in dataset
            ]
            try:
                acc_results = asyncio.run(compute_acc_batch(tasks, self.system.client))
                for i, acc in enumerate(acc_results):
                    accs[i] = acc
            except Exception:
                from wemg.evaluation.metrics import compute_acc
                for i, example in enumerate(dataset):
                    try:
                        accs[i] = asyncio.run(compute_acc(
                            example[question_column], example.get(prediction_column, ""),
                            example[answer_column], self.system.client
                        ))
                    except Exception:
                        accs[i] = 0.0
        
        metrics = compute_aggregate_metrics(sub_ems, accs, [])
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
