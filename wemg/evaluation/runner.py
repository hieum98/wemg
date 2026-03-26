"""Dataset evaluation runner."""

import asyncio
import json
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_slug(text: str, max_len: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if not cleaned:
        return "question"
    return cleaned[:max_len]


def _question_artifact_dir(base_dir: Path, question: str, index: int) -> Path:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
    slug = _safe_slug(question)
    return base_dir / f"q_{index:05d}_{digest}_{slug}"


def _serialize_search_tree(node: Any) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    payload: Dict[str, Any] = {
        "node_type": getattr(getattr(node, "node_type", None), "value", str(getattr(node, "node_type", "UNKNOWN"))),
        "content": dict(getattr(getattr(node, "node_state", None), "content", {}) or {}),
    }
    if hasattr(node, "visits"):
        payload["visits"] = int(getattr(node, "visits", 0))
    if hasattr(node, "value"):
        payload["value"] = float(getattr(node, "value", 0.0))
    children = list(getattr(node, "children", []) or [])
    payload["children"] = [_serialize_search_tree(child) for child in children]
    return payload


def _save_question_artifacts(
    artifacts_root: Path,
    index: int,
    question: str,
    result: Any,
) -> Dict[str, Any]:
    q_dir = _question_artifact_dir(artifacts_root, question, index)
    q_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {
        "artifact_dir": str(q_dir),
        "search_tree_path": None,
        "textual_memory_path": None,
        "graph_memory_path": None,
    }

    tree_payload = _serialize_search_tree(getattr(result, "search_tree", None))
    if tree_payload is not None:
        tree_path = q_dir / "search_tree.json"
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree_payload, f, indent=2, ensure_ascii=False)
        out["search_tree_path"] = str(tree_path)

    working_memory = getattr(result, "working_memory", None)
    if working_memory is not None:
        textual_path = q_dir / "working_memory_textual.json"
        with open(textual_path, "w", encoding="utf-8") as f:
            json.dump(list(getattr(working_memory, "textual_memory", []) or []), f, indent=2, ensure_ascii=False)
        out["textual_memory_path"] = str(textual_path)

        graph = getattr(working_memory, "graph_memory", None)
        if graph is not None:
            graph_path = q_dir / "working_memory_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(graph, f)
            out["graph_memory_path"] = str(graph_path)

    return out


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
        clear_kb_cache_every_n_batches: Optional[int] = 1,
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
            clear_kb_cache_every_n_batches: Periodically clear in-process Wikidata
                triple caches every N processed chunks to bound memory growth.
                Set to None or <=0 to disable periodic cache clears.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "evaluation_log.jsonl"
        artifacts_root = output_dir / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        
        # Load existing results for resume
        completed = {}
        if resume and log_file.exists():
            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line)
                    completed[entry["question"]] = entry
            print(f"Resuming: {len(completed)} questions already processed")
        
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
                    try:
                        entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                    except Exception as artifact_error:
                        logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                        entry["artifacts_error"] = str(artifact_error)
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
                        try:
                            entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                        except Exception as artifact_error:
                            logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                            entry["artifacts_error"] = str(artifact_error)
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
                        try:
                            entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                        except Exception as artifact_error:
                            logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                            entry["artifacts_error"] = str(artifact_error)
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
            cache_clear_every = (
                int(clear_kb_cache_every_n_batches)
                if clear_kb_cache_every_n_batches is not None
                else 0
            )
            if cache_clear_every < 0:
                cache_clear_every = 0
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

                # Bound process memory in long-running eval by clearing Wikidata
                # triple caches periodically between chunks.
                chunk_index = (chunk_start // chunk_size) + 1
                if cache_clear_every > 0 and (chunk_index % cache_clear_every == 0):
                    wikidata_client = getattr(self.system, "wikidata_client", None)
                    if wikidata_client is not None and hasattr(
                        wikidata_client, "clear_triple_caches"
                    ):
                        try:
                            wikidata_client.clear_triple_caches()
                            logger.info(
                                "Cleared Wikidata triple caches after chunk %d",
                                chunk_index,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to clear Wikidata triple caches at chunk %d: %s",
                                chunk_index,
                                e,
                            )
        
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
