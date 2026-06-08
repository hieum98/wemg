"""Dataset evaluation runner."""

import asyncio
import json
import logging
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

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
    global_knowledge = getattr(result, "global_knowledge", None)

    # MCTS: global_knowledge absorbs all branch discoveries — use it as the source of truth.
    # CoT: no global_knowledge, so fall back to working_memory directly.
    if global_knowledge is not None:
        textual_items = list(getattr(global_knowledge, "confirmed_facts", []) or [])
        graph = getattr(global_knowledge, "graph", None)
    else:
        textual_items = list(getattr(working_memory, "textual_memory", []) or []) if working_memory else []
        graph = getattr(working_memory, "graph_memory", None) if working_memory else None

    if textual_items:
        textual_path = q_dir / "working_memory_textual.json"
        with open(textual_path, "w", encoding="utf-8") as f:
            json.dump(textual_items, f, indent=2, ensure_ascii=False)
        out["textual_memory_path"] = str(textual_path)

    if graph is not None:
        graph_path = q_dir / "working_memory_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        out["graph_memory_path"] = str(graph_path)

    return out


def _load_freebase_subgraph_cache_index(cache_root: Path) -> Dict[str, str]:
    """Load question -> entry_path map from split cache index.jsonl."""
    index_path = cache_root / "index.jsonl"
    if not cache_root.is_dir() or not index_path.is_file():
        raise ValueError(
            "freebase_subgraph cache must be a directory with index.jsonl "
            f"(got: {cache_root})"
        )

    out: Dict[str, str] = {}
    with open(index_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                logger.warning("Invalid cache index JSON at line %d in %s", line_no, index_path)
                continue
            question = row.get("question")
            entry_path = row.get("entry_path")
            if not question or not entry_path:
                logger.warning(
                    "Skipping malformed cache index row at line %d in %s",
                    line_no,
                    index_path,
                )
                continue
            out[str(question)] = str(entry_path)
    return out


def _load_freebase_subgraph_cache_entry(cache_root: Path, entry_relpath: str) -> Optional[Dict[str, Any]]:
    """Load one split cache entry JSON by index-relative path."""
    entry_path = cache_root / entry_relpath
    if not entry_path.is_file():
        logger.warning("Subgraph cache entry not found: %s", entry_path)
        return None
    try:
        with open(entry_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning("Could not load subgraph cache entry %s: %s", entry_path, e)
        return None
    if not isinstance(payload, dict):
        logger.warning("Subgraph cache entry is not a JSON object: %s", entry_path)
        return None
    return payload


class DatasetEvaluator:
    """Evaluates COE on datasets with resume support."""
    
    def __init__(self, system):
        """
        Args:
            system: COESystem instance
        """
        self.system = system
    
    def evaluate(
        self,
        dataset,
        output_path: str = "./results",
        resume: bool = True,
        score_only: bool = False,
        question_column: str = "question",
        answer_column: str = "answer",
        level_column: str = "level",
        max_concurrent: Optional[int] = None,
        log_batch_size: Optional[int] = None,
        clear_kb_cache_every_n_batches: Optional[int] = 1,
    ) -> Dict:
        """Run evaluation: generate answers + compute metrics.
        
        Args:
            dataset: HuggingFace Dataset with question and answer columns
            output_path: Directory to save results
            resume: Whether to resume from previous run
            score_only: If True, only recompute scores from existing logfile (skip answer generation)
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
        
        # Load existing results for resume or score_only
        completed = {}
        completed_rows: List[Dict[str, Any]] = []
        if (resume or score_only) and log_file.exists():
            with open(log_file) as f:
                for line in f:
                    entry = json.loads(line)
                    if 'error' in entry:
                        logger.warning(f"Existing log entry for question with error: {entry.get('question', 'unknown')[:60]}... Error: {entry['error']}. Ignoring this entry for resume/score_only purposes.")
                        continue
                    completed_rows.append(entry)
                    completed[entry["question"]] = entry
            print(f"Loaded: {len(completed)} questions from logfile")
        
        from coe.evaluation.metrics import compute_sub_em
        
        n = len(dataset)
        sub_ems_short: List[float] = [0.0] * n
        sub_ems_long: List[float] = [0.0] * n
        accs_short: List[Optional[float]] = [None] * n
        accs_long: List[Optional[float]] = [None] * n
        pass_at_k_values: List[Optional[int]] = [None] * n
        levels: List[str] = ["unknown"] * n
        predictions_short: Dict[str, str] = {}
        predictions_long: Dict[str, str] = {}
        
        # Load predictions from completed entries
        for q, entry in completed.items():
            predictions_short[q] = entry.get("predicted_answer", "")
            predictions_long[q] = entry.get("full_answer", "")
        
        # ------------------------------------------------------------------
        # Load subgraph cache (freebase_subgraph mode only)
        # ------------------------------------------------------------------
        subgraph_cache_index_by_question: Dict[str, str] = {}
        subgraph_cache_entry_by_relpath: Dict[str, Dict[str, Any]] = {}
        subgraph_cache_root: Optional[Path] = None
        _ng = getattr(getattr(self.system, "cfg", None), "node_generation", None)
        _kb_source = getattr(_ng, "kb_source", "wikidata")
        _cache_path = getattr(_ng, "freebase_subgraph_cache", None)

        if _kb_source == "freebase_subgraph":
            if not _cache_path:
                raise ValueError(
                    "kb_source=freebase_subgraph requires node_generation.freebase_subgraph_cache "
                    "to point to a split cache directory"
                )
            subgraph_cache_root = Path(_cache_path)
            if subgraph_cache_root.is_file():
                raise ValueError(
                    "Monolithic JSONL cache is no longer supported for freebase_subgraph evaluation. "
                    "Please migrate to split cache directory format (index.jsonl + entries/*.json)."
                )
            subgraph_cache_index_by_question = _load_freebase_subgraph_cache_index(subgraph_cache_root)
            logger.info(
                "Loaded freebase_subgraph cache index with %d entries from %s",
                len(subgraph_cache_index_by_question),
                subgraph_cache_root,
            )

        pending: List[tuple] = []
        scored_indices: List[int] = []
        matched_by_question = 0
        matched_by_index = 0
        for i, example in enumerate(dataset):
            question = example[question_column]
            correct = example[answer_column]
            level = example.get(level_column, "unknown") if hasattr(example, "get") else "unknown"
            levels[i] = level
            entry = completed.get(question)
            if entry is not None:
                matched_by_question += 1
            elif score_only and i < len(completed_rows):
                # In score_only mode, allow positional alignment as a fallback when
                # question text differs slightly (e.g., whitespace/punctuation changes).
                entry = completed_rows[i]
                matched_by_index += 1

            if entry is not None:
                scored_indices.append(i)
                # In score_only mode, recompute all scores fresh
                if score_only:
                    # Recompute sub_ems
                    predicted_short = entry.get("predicted_answer", "")
                    predicted_long = entry.get("full_answer", "")
                    sub_ems_short[i] = compute_sub_em(predicted_short, correct)
                    sub_ems_long[i] = compute_sub_em(predicted_long, correct)
                    predictions_short[question] = predicted_short
                    predictions_long[question] = predicted_long
                    # Reset accs to recompute
                    accs_short[i] = None
                    accs_long[i] = None
                else:
                    # Reuse existing sub_ems when not score_only
                    sub_ems_short[i] = float(entry.get("sub_em_short", 0.0))
                    sub_ems_long[i] = float(entry.get("sub_em_long", 0.0))
                    accs_short[i] = entry.get("acc_short")
                    accs_long[i] = entry.get("acc_long")
                pass_at_k_values[i] = entry.get("pass_at_k")
            else:
                if not score_only:
                    cache_entry_relpath = subgraph_cache_index_by_question.get(question)
                    pending.append((i, question, correct, cache_entry_relpath, level))

        if score_only:
            if not completed_rows:
                raise ValueError(
                    f"score_only=true but logfile not found or empty: {log_file}"
                )
            aligned_total = matched_by_question + matched_by_index
            if aligned_total == 0:
                raise ValueError(
                    "score_only=true could not align dataset rows with evaluation_log.jsonl "
                    "(no question matches and no positional overlap)."
                )
            logger.info(
                "score_only alignment: matched_by_question=%d, matched_by_index=%d, total=%d",
                matched_by_question,
                matched_by_index,
                aligned_total,
            )

        scored_index_set = set(scored_indices)

        initial_done = n - len(pending)
        progress_bar = tqdm(
            total=n,
            initial=initial_done,
            desc="Evaluating",
            unit="q",
            dynamic_ncols=True,
            leave=True,
        )
        
        def append_logs(entries: List[Dict[str, Any]]) -> None:
            with open(log_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                f.flush()
        
        def process_answer_results(
            batch: List[tuple],
            results: List,
        ) -> List[Dict[str, Any]]:
            """Build log entries and update sub_ems / predictions for one answer_batch."""
            out: List[Dict[str, Any]] = []
            for pending_item, result in zip(batch, results):
                i, question, correct = pending_item[0], pending_item[1], pending_item[2]
                level = pending_item[4] if len(pending_item) > 4 else "unknown"
                err = result.metadata.get("error") if result.metadata else None
                if err:
                    logger.error(f"Error processing question {i}: {err}")
                    err_text = f"Error: {err}"
                    entry = {
                        "question": question,
                        "correct_answer": correct,
                        "predicted_answer": err_text,
                        "sub_em_short": 0.0,
                        "sub_em_long": 0.0,
                        "acc_short": None,
                        "acc_long": None,
                        "pass_at_k": None,
                        "level": level,
                        "error": str(err),
                    }
                    try:
                        entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                    except Exception as artifact_error:
                        logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                        entry["artifacts_error"] = str(artifact_error)
                    sub_ems_short[i] = 0.0
                    sub_ems_long[i] = 0.0
                    pass_at_k_values[i] = None
                    predictions_short[question] = err_text
                    predictions_long[question] = err_text
                else:
                    try:
                        predicted_short = result.concise_answer or result.answer
                        predicted_long = result.answer
                        sub_em_short = compute_sub_em(predicted_short, correct)
                        sub_em_long = compute_sub_em(predicted_long, correct)
                        pass_at_k = result.metadata.get("pass_at_k") if result.metadata else None
                        entry = {
                            "question": question,
                            "correct_answer": correct,
                            "predicted_answer": predicted_short,
                            "full_answer": predicted_long,
                            "sub_em_short": sub_em_short,
                            "sub_em_long": sub_em_long,
                            "pass_at_k": pass_at_k,
                            "acc_short": None,
                            "acc_long": None,
                            "level": level,
                        }
                        try:
                            entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                        except Exception as artifact_error:
                            logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                            entry["artifacts_error"] = str(artifact_error)
                        sub_ems_short[i] = sub_em_short
                        sub_ems_long[i] = sub_em_long
                        pass_at_k_values[i] = pass_at_k
                        predictions_short[question] = predicted_short
                        predictions_long[question] = predicted_long
                    except Exception as e:
                        logger.error(f"Error processing question {i}: {e}")
                        err_text = f"Error: {e}"
                        entry = {
                            "question": question,
                            "correct_answer": correct,
                            "predicted_answer": err_text,
                            "sub_em_short": 0.0,
                            "sub_em_long": 0.0,
                            "acc_short": None,
                            "acc_long": None,
                            "pass_at_k": None,
                            "level": level,
                            "error": str(e),
                        }
                        try:
                            entry["artifacts"] = _save_question_artifacts(artifacts_root, i, question, result)
                        except Exception as artifact_error:
                            logger.warning(f"Could not persist artifacts for question {i}: {artifact_error}")
                            entry["artifacts_error"] = str(artifact_error)
                        sub_ems_short[i] = 0.0
                        sub_ems_long[i] = 0.0
                        pass_at_k_values[i] = None
                        predictions_short[question] = err_text
                        predictions_long[question] = err_text
                out.append(entry)
                logger.info(f"[{i+1}/{n}] Sub-EM Short: {sub_ems_short[i]:.1f} | Sub-EM Long: {sub_ems_long[i]:.1f} | Q: {question[:60]}...")
            return out
        
        try:
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
                        for _, _, c, *_ in batch
                    ]
                    cache_entry_relpaths = [p[3] if len(p) > 3 else None for p in batch]
                    cache_entries: List[Optional[Dict[str, Any]]] = []
                    for relpath in cache_entry_relpaths:
                        if not relpath or subgraph_cache_root is None:
                            cache_entries.append(None)
                            continue
                        loaded_entry = subgraph_cache_entry_by_relpath.get(relpath)
                        if loaded_entry is None:
                            loaded_entry = _load_freebase_subgraph_cache_entry(
                                subgraph_cache_root,
                                relpath,
                            )
                            if loaded_entry is not None:
                                subgraph_cache_entry_by_relpath[relpath] = loaded_entry
                        cache_entries.append(loaded_entry)
                    use_cache = any(e is not None for e in cache_entries)
                    mw = max_concurrent
                    if mw is not None:
                        mw = max(1, min(mw, len(batch)))
                    chunk_index = (chunk_start // chunk_size) + 1
                    results = self.system.answer_batch(
                        questions,
                        question_ids=qids,
                        golden_answers=golds,
                        subgraph_cache_entries=cache_entries if use_cache else None,
                        max_workers=mw,
                    )
                    entries = process_answer_results(batch, results)
                    append_logs(entries)
                    progress_bar.update(len(batch))

                    # Bound process memory in long-running eval by clearing Wikidata
                    # triple caches periodically between chunks.
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
        finally:
            progress_bar.close()
        
        # Compute Acc scores in batch (async concurrent) for both short and long answers
        acc_task_rows_short: List[tuple] = []
        acc_task_rows_long: List[tuple] = []
        for i, example in enumerate(dataset):
            if score_only and i not in scored_index_set:
                continue
            if accs_short[i] is not None and accs_long[i] is not None:
                continue
            question = example[question_column]
            correct = example[answer_column]
            if accs_short[i] is None:
                predicted_short = predictions_short.get(question, "")
                acc_task_rows_short.append((i, question, predicted_short, correct))
            if accs_long[i] is None:
                predicted_long = predictions_long.get(question, "")
                acc_task_rows_long.append((i, question, predicted_long, correct))
        
        # In score_only mode we may skip answer generation, so force lazy
        # system initialization to make sure evaluator client exists.
        if self.system.client is None and hasattr(self.system, "_initialize"):
            try:
                self.system._initialize()
            except Exception as e:
                logger.error(f"Failed to initialize system for Acc scoring: {e}")

        if self.system.client:
            from coe.evaluation.metrics import compute_acc_batch
            acc_max = max_concurrent if max_concurrent is not None else 10
            acc_chunk_size = max(1, acc_max)
            total_acc_tasks = len(acc_task_rows_short) + len(acc_task_rows_long)
            acc_progress = tqdm(
                total=total_acc_tasks,
                desc="Scoring Acc",
                unit="q",
                dynamic_ncols=True,
                leave=True,
            )

            def _compute_acc_rows_in_chunks(task_rows: List[tuple], target: List[Optional[float]], label: str) -> None:
                if not task_rows:
                    return
                try:
                    for start in range(0, len(task_rows), acc_chunk_size):
                        chunk_rows = task_rows[start : start + acc_chunk_size]
                        chunk_tasks = [(q, pred, cor) for _, q, pred, cor in chunk_rows]
                        chunk_scores = asyncio.run(
                            compute_acc_batch(chunk_tasks, self.system.client, max_concurrent=acc_max)
                        )
                        for (idx, _, _, _), acc in zip(chunk_rows, chunk_scores):
                            target[idx] = acc
                        acc_progress.update(len(chunk_rows))
                except Exception as e:
                    logger.error(f"Acc {label} batch failed: {e}")
                    from coe.evaluation.metrics import compute_acc
                    for idx, question, predicted, correct in task_rows:
                        try:
                            target[idx] = asyncio.run(
                                compute_acc(question, predicted, correct, self.system.client)
                            )
                        except Exception as e2:
                            logger.error(f"Acc {label} computation failed for question {idx}: {e2}")
                            target[idx] = 0.0
                        finally:
                            acc_progress.update(1)

            try:
                _compute_acc_rows_in_chunks(acc_task_rows_short, accs_short, "short")
                _compute_acc_rows_in_chunks(acc_task_rows_long, accs_long, "long")
            finally:
                acc_progress.close()
        
        # Compute aggregate metrics for both short and long answers
        from coe.evaluation.metrics import compute_aggregate_metrics_both, compute_aggregate_metrics_by_level
        if score_only:
            metric_sub_ems_short = [sub_ems_short[i] for i in scored_indices]
            metric_sub_ems_long = [sub_ems_long[i] for i in scored_indices]
            metric_accs_short = [accs_short[i] for i in scored_indices]
            metric_accs_long = [accs_long[i] for i in scored_indices]
            metric_pass_at_k_values = [pass_at_k_values[i] for i in scored_indices]
            metric_levels = [levels[i] for i in scored_indices]
        else:
            metric_sub_ems_short = sub_ems_short
            metric_sub_ems_long = sub_ems_long
            metric_accs_short = accs_short
            metric_accs_long = accs_long
            metric_pass_at_k_values = pass_at_k_values
            metric_levels = levels

        metrics = compute_aggregate_metrics_both(
            metric_sub_ems_short,
            metric_sub_ems_long,
            metric_accs_short,
            metric_accs_long,
            metric_pass_at_k_values,
        )

        # Per-level breakdown (populated when level field is present, e.g. GrailQA)
        if any(l != "unknown" for l in metric_levels):
            metrics["by_level"] = compute_aggregate_metrics_by_level(
                metric_sub_ems_short,
                metric_sub_ems_long,
                metric_accs_short,
                metric_accs_long,
                metric_levels,
                metric_pass_at_k_values,
            )
        
        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("COE Evaluation Results (Short and Long Answer Versions)\n")
            f.write("=" * 50 + "\n\n")
            f.write("SHORT ANSWER METRICS:\n")
            f.write("-" * 30 + "\n")
            for k, v in metrics["short_answer"].items():
                f.write(f"{k}: {v}\n")
            f.write("\nLONG ANSWER METRICS:\n")
            f.write("-" * 30 + "\n")
            for k, v in metrics["long_answer"].items():
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
        long_prediction_column: Optional[str] = None,
    ) -> Dict:
        """Score existing predictions without running COE.
        
        If long_prediction_column is provided, computes metrics for both short and long versions.
        Otherwise, uses prediction_column for both.
        """
        from coe.evaluation.metrics import (
            compute_sub_em, compute_aggregate_metrics_both, compute_acc_batch
        )
        
        sub_ems_short = []
        sub_ems_long = []
        accs_short = []
        accs_long = []
        
        # If no long_prediction_column provided, use prediction_column for both
        if long_prediction_column is None:
            long_prediction_column = prediction_column
        
        for example in dataset:
            correct = example[answer_column]
            predicted_short = example.get(prediction_column, "")
            predicted_long = example.get(long_prediction_column, "")
            
            sub_ems_short.append(compute_sub_em(predicted_short, correct))
            sub_ems_long.append(compute_sub_em(predicted_long, correct))
            accs_short.append(None)
            accs_long.append(None)
        
        # Compute Acc if client available
        if self.system.client:
            # Short answer acc
            tasks_short = [
                (example[question_column], example.get(prediction_column, ""), example[answer_column])
                for example in dataset
            ]
            try:
                acc_results_short = asyncio.run(compute_acc_batch(tasks_short, self.system.client))
                for i, acc in enumerate(acc_results_short):
                    accs_short[i] = acc
            except Exception as e:
                logger.error(f"Acc short batch failed: {e}")
                from coe.evaluation.metrics import compute_acc
                for i, example in enumerate(dataset):
                    try:
                        accs_short[i] = asyncio.run(compute_acc(
                            example[question_column], example.get(prediction_column, ""),
                            example[answer_column], self.system.client
                        ))
                    except Exception as e2:
                        logger.error(f"Acc short failed for {i}: {e2}")
                        accs_short[i] = 0.0
            
            # Long answer acc
            tasks_long = [
                (example[question_column], example.get(long_prediction_column, ""), example[answer_column])
                for example in dataset
            ]
            try:
                acc_results_long = asyncio.run(compute_acc_batch(tasks_long, self.system.client))
                for i, acc in enumerate(acc_results_long):
                    accs_long[i] = acc
            except Exception as e:
                logger.error(f"Acc long batch failed: {e}")
                from coe.evaluation.metrics import compute_acc
                for i, example in enumerate(dataset):
                    try:
                        accs_long[i] = asyncio.run(compute_acc(
                            example[question_column], example.get(long_prediction_column, ""),
                            example[answer_column], self.system.client
                        ))
                    except Exception as e2:
                        logger.error(f"Acc long failed for {i}: {e2}")
                        accs_long[i] = 0.0
        
        metrics = compute_aggregate_metrics_both(sub_ems_short, sub_ems_long, accs_short, accs_long, [])
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

