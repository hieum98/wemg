"""Dataset evaluation runner."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        max_concurrent: int = 1,
    ) -> Dict:
        """Run evaluation: generate answers + compute metrics.
        
        Args:
            dataset: HuggingFace Dataset with question and answer columns
            output_path: Directory to save results
            resume: Whether to resume from previous run
            question_column: Column name for questions
            answer_column: Column name for correct answers
            max_concurrent: Number of questions to process concurrently
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
        
        # Process questions
        sub_ems = []
        accs = []
        pass_at_k_values = []
        
        from wemg.evaluation.metrics import compute_sub_em
        
        for i, example in enumerate(dataset):
            question = example[question_column]
            correct = example[answer_column]
            
            if question in completed:
                entry = completed[question]
                sub_ems.append(entry.get("sub_em", 0.0))
                accs.append(entry.get("acc"))
                pass_at_k_values.append(entry.get("pass_at_k"))
                continue
            
            try:
                result = self.system.answer(question, question_id=str(i), golden_answer=correct if isinstance(correct, str) else str(correct))
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
                
                sub_ems.append(sub_em)
                accs.append(None)
                pass_at_k_values.append(pass_at_k)
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                entry = {
                    "question": question,
                    "correct_answer": correct,
                    "predicted_answer": f"Error: {e}",
                    "sub_em": 0.0,
                    "acc": None,
                    "pass_at_k": None,
                    "error": str(e),
                }
                sub_ems.append(0.0)
                accs.append(None)
                pass_at_k_values.append(None)
            
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            logger.info(f"[{i+1}/{len(dataset)}] Sub-EM: {sub_ems[-1]:.1f} | Q: {question[:60]}...")
        
        # Compute Acc scores in batch
        acc_tasks = []
        for i, example in enumerate(dataset):
            if accs[i] is None:
                question = example[question_column]
                correct = example[answer_column]
                # Find predicted answer from log
                if question in completed:
                    predicted = completed[question].get("predicted_answer", "")
                else:
                    with open(log_file) as f:
                        for line in f:
                            e = json.loads(line)
                            if e["question"] == question:
                                predicted = e.get("predicted_answer", "")
                                break
                        else:
                            predicted = ""
                acc_tasks.append((i, question, predicted, correct))
        
        if acc_tasks and self.system.client:
            from wemg.evaluation.metrics import compute_acc
            for idx, question, predicted, correct in acc_tasks:
                try:
                    acc = asyncio.run(compute_acc(question, predicted, correct, self.system.client))
                    accs[idx] = acc
                except Exception as e:
                    logger.error(f"Acc computation failed for question {idx}: {e}")
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
        from wemg.evaluation.metrics import compute_sub_em, compute_aggregate_metrics
        
        sub_ems = []
        accs = []
        
        for example in dataset:
            correct = example[answer_column]
            predicted = example.get(prediction_column, "")
            sub_ems.append(compute_sub_em(predicted, correct))
            accs.append(None)
        
        # Compute Acc if client available
        if self.system.client:
            from wemg.evaluation.metrics import compute_acc
            for i, example in enumerate(dataset):
                try:
                    acc = asyncio.run(compute_acc(
                        example[question_column], example.get(prediction_column, ""),
                        example[answer_column], self.system.client
                    ))
                    accs[i] = acc
                except Exception:
                    accs[i] = 0.0
        
        metrics = compute_aggregate_metrics(sub_ems, accs, [])
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
