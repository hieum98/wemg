"""Dataset evaluation runner for the ``langgraph_coe`` system.

Produces the **same output** as ``coe.evaluation.runner.DatasetEvaluator``:

  - ``evaluation_log.jsonl`` — one JSON row per question with the identical keys
    (``question``, ``correct_answer``, ``predicted_answer``, ``full_answer``,
    ``sub_em_short``, ``sub_em_long``, ``pass_at_k``, ``acc_short``, ``acc_long``,
    ``level``, ``artifacts`` / ``error``).
  - ``metrics.json`` — ``{"short_answer": …, "long_answer": …, "by_level"?: …}``.
  - ``summary.txt`` — the same human-readable block.
  - ``artifacts/q_XXXXX_<digest>_<slug>/`` — ``search_tree.json`` (MCTS),
    ``working_memory_textual.json``, ``working_memory_graph.pkl``.

Resume / ``score_only`` semantics match the legacy runner. The difference is the
generation backend: instead of a persistent ``COESystem``, this driver wires the
``langgraph_coe`` runtime **once** (so the 99 GB FAISS index loads a single time),
builds the configured CoT/MCTS graph once, and invokes it per question inside one
event loop — then adapts each final graph state through
``langgraph_coe.system.AnswerResult.from_state`` (the same envelope ``answer()``
returns).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from tqdm.auto import tqdm

from langgraph_coe.config import LangGraphCoeConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Artifact paths (identical slug/dir scheme to the legacy runner)
# ──────────────────────────────────────────────────────────────────────────────


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


def _ntype(node: Dict[str, Any]) -> str:
    value = node.get("node_type")
    return getattr(value, "value", value)


def _nested_search_tree(raw_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert the MCTS flat ``{node_id: node}`` tree into a nested JSON payload.

    Shape mirrors the legacy ``search_tree.json`` (``node_type`` / ``content`` /
    ``visits`` / ``value`` / ``children``) so the same ``artifacts.py`` viewers
    can load it. CoT has no tree → returns ``None``.
    """
    tree = raw_state.get("tree")
    root_id = raw_state.get("root_id")
    if not isinstance(tree, dict) or not tree or root_id not in tree:
        return None

    def _build(node_id: str, seen: set) -> Optional[Dict[str, Any]]:
        if node_id in seen or node_id not in tree:
            return None
        seen.add(node_id)
        node = tree[node_id]
        payload: Dict[str, Any] = {
            "node_type": _ntype(node),
            "content": dict(node.get("content", {}) or {}),
            "visits": int(node.get("visits", 0) or 0),
            "value": float(node.get("value", 0.0) or 0.0),
        }
        children = [
            child
            for cid in (node.get("children_ids") or [])
            if (child := _build(cid, seen)) is not None
        ]
        payload["children"] = children
        return payload

    return _build(root_id, set())


def _save_question_artifacts(
    artifacts_root: Path,
    index: int,
    question: str,
    result: Any,
) -> Dict[str, Any]:
    """Persist a question's search tree + working memory, returning their paths.

    Reads the raw final graph state stashed on ``result.metadata['_raw_state']``
    (text memory, graph memory, and — for MCTS — the search tree). When the
    result carries nothing to save, the dir is **not** created and all paths are
    ``None`` (the row still gets an ``artifacts`` key for schema parity).
    """
    q_dir = _question_artifact_dir(artifacts_root, question, index)
    out: Dict[str, Any] = {
        "artifact_dir": str(q_dir),
        "search_tree_path": None,
        "textual_memory_path": None,
        "graph_memory_path": None,
        "plan_path": None,
        "retrieval_log_path": None,
    }

    raw_state = (getattr(result, "metadata", None) or {}).get("_raw_state") or {}
    tree_payload = _nested_search_tree(raw_state)
    textual_items = [
        t
        for t in (raw_state.get("text_memory") or [])
        if isinstance(t, str) and t.strip()
    ]
    graph = raw_state.get("graph_memory")
    has_graph = (
        isinstance(graph, (nx.Graph, nx.DiGraph)) and graph.number_of_nodes() > 0
    )
    # Plan trajectory. ``iteration_history`` was never persisted before, and it is
    # the only record of what was asked per hop — required to measure the
    # cross-iteration re-ask rate that the UPDATE write is meant to remove.
    plan_payload: Optional[Dict[str, Any]] = None
    # ``iteration_history`` is the CoT trajectory — what was asked and answered per
    # hop — and it is the only record that makes two arms comparable. It was written
    # only when a plan was active, which left the plan-disabled baseline with no
    # trajectory at all and made "did the plan change the questions asked?"
    # unanswerable from artifacts.
    if raw_state.get("plan") or raw_state.get("iteration_history"):
        plan_payload = {
            "plan": raw_state.get("plan"),
            "plan_version": raw_state.get("plan_version"),
            "plan_ledger": raw_state.get("plan_ledger") or [],
            "plan_action_log": raw_state.get("plan_action_log") or [],
            "plan_attempts_log": raw_state.get("plan_attempts_log") or [],
            "abstention": raw_state.get("abstention") or {},
            "iteration_history": raw_state.get("iteration_history") or [],
        }

    # Pre-consolidation evidence. Persisted separately from the consolidated memory so
    # "the gold was retrieved and then dropped by consolidation" is answerable after a run;
    # ``extracted_facts`` is cleared every hop, so nothing else preserves it.
    retrieval_items = [
        t
        for t in (raw_state.get("retrieval_log") or [])
        if isinstance(t, str) and t.strip()
    ]

    if not (tree_payload or textual_items or has_graph or plan_payload or retrieval_items):
        return out

    q_dir.mkdir(parents=True, exist_ok=True)

    if retrieval_items:
        rl_path = q_dir / "retrieval_log.json"
        with open(rl_path, "w", encoding="utf-8") as f:
            json.dump(retrieval_items, f, indent=2, ensure_ascii=False)
        out["retrieval_log_path"] = str(rl_path)

    if plan_payload is not None:
        plan_path = q_dir / "plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_payload, f, indent=2, ensure_ascii=False, default=str)
        out["plan_path"] = str(plan_path)

    if tree_payload is not None:
        tree_path = q_dir / "search_tree.json"
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree_payload, f, indent=2, ensure_ascii=False)
        out["search_tree_path"] = str(tree_path)

    if textual_items:
        textual_path = q_dir / "working_memory_textual.json"
        with open(textual_path, "w", encoding="utf-8") as f:
            json.dump(textual_items, f, indent=2, ensure_ascii=False)
        out["textual_memory_path"] = str(textual_path)

    if has_graph:
        graph_path = q_dir / "working_memory_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        out["graph_memory_path"] = str(graph_path)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────


class DatasetEvaluator:
    """Evaluate the ``langgraph_coe`` system on datasets, with resume support.

    Args:
        cfg: a resolved :class:`LangGraphCoeConfig`. ``cfg.search.strategy`` picks
            the graph (``cot`` or ``mcts``); the rest configures the live runtime.
    """

    def __init__(self, cfg: LangGraphCoeConfig):
        self.cfg = cfg
        self._registry = None  # lazily built RoleModelRegistry for Acc scoring

    # -- runtime ----------------------------------------------------------------

    def _registry_for_scoring(self):
        if self._registry is None:
            from langgraph_coe.llm import RoleModelRegistry

            self._registry = RoleModelRegistry(self.cfg.llm)
        return self._registry

    async def _answer_one(
        self,
        graph: Any,
        strategy: str,
        question: str,
        recursion_limit: int,
        sem: asyncio.Semaphore,
    ) -> Any:
        """Invoke the compiled graph for one question → an ``AnswerResult``.

        Per-question ContextVar resets isolate sibling tasks (the documented
        ``reset_*_session`` pattern). On failure the error is captured into
        ``metadata['error']`` exactly like the legacy per-question error rows.
        The raw final state is stashed under ``metadata['_raw_state']`` for
        artifact saving (never serialized into the JSONL row).
        """
        from langgraph_coe.system import (
            AnswerResult,
            _initial_cot_state,
            _initial_mcts_state,
        )
        from langgraph_coe.llm import read_cost_meter, start_cost_meter
        from langgraph_coe.tools.web import reset_web_research_session
        from langgraph_coe.tools.wikidata import reset_wikidata_session

        async with sem:
            reset_wikidata_session()
            reset_web_research_session()
            # Same per-question ContextVar discipline as the session resets above: the
            # meter must not blend concurrent questions. Without it no cost claim about
            # this system is provable — hops and subquestions per question are the only
            # other proxies, and they miss every call inside the retrieval subgraphs.
            start_cost_meter()
            if strategy == "mcts":
                initial = _initial_mcts_state(question, self.cfg)
            else:
                initial = _initial_cot_state(question, self.cfg)
            try:
                final = await graph.ainvoke(
                    initial, config={"recursion_limit": recursion_limit}
                )
                result = AnswerResult.from_state(
                    {**(final or {}), "strategy": strategy}
                )
                result.metadata["_raw_state"] = final or {}
                result.metadata["cost"] = read_cost_meter() or {}
                return result
            except Exception as e:  # noqa: BLE001 — captured as an error row (coe parity)
                # Many failing exceptions (timeouts, connection errors) have an
                # empty str(), so log the type + repr and a traceback to make the
                # cause visible instead of a bare "Error answering question: ".
                logger.error(
                    "Error answering question %r: %s: %r",
                    question[:60],
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                return AnswerResult(
                    question=question,
                    answer="",
                    concise_answer="",
                    reasoning="",
                    metadata={"error": f"{type(e).__name__}: {e}"},
                )

    async def _generate_and_log(
        self,
        pending: List[tuple],
        *,
        log_file: Path,
        artifacts_root: Path,
        process_answer_results,
        append_logs,
        progress_bar,
        chunk_size: int,
        max_concurrent: Optional[int],
    ) -> None:
        """Wire runtime once, then answer all pending questions in one event loop."""
        from langgraph_coe.graphs import build_cot_graph, build_mcts_graph
        from langgraph_coe.llm import RoleModelRegistry
        from langgraph_coe.system import _init_runtime
        from langgraph_coe.tools.web import reset_web_research_session
        from langgraph_coe.tools.wikidata import reset_wikidata_session

        cfg = self.cfg
        strategy = getattr(cfg.search, "strategy", "cot")

        _init_runtime(cfg)  # loads the FAISS index + wikidata client ONCE
        reset_wikidata_session()
        reset_web_research_session()

        registry = RoleModelRegistry(cfg.llm)
        self._registry = registry  # reuse for Acc scoring
        if strategy == "mcts":
            graph = build_mcts_graph(registry, cfg)
            recursion_limit = int(cfg.search.mcts.recursion_limit)
        else:
            graph = build_cot_graph(registry, cfg)
            recursion_limit = int(cfg.search.cot.recursion_limit)

        workers = max_concurrent if max_concurrent is not None else chunk_size
        sem = asyncio.Semaphore(max(1, workers))

        try:
            for chunk_start in range(0, len(pending), chunk_size):
                batch = pending[chunk_start : chunk_start + chunk_size]
                results = await asyncio.gather(
                    *[
                        self._answer_one(graph, strategy, p[1], recursion_limit, sem)
                        for p in batch
                    ]
                )
                entries = process_answer_results(batch, results)
                append_logs(entries)
                progress_bar.update(len(batch))
        finally:
            await self._aclose_runtime()

    async def _aclose_runtime(self) -> None:
        """Best-effort close of the Wikidata httpx client on the active loop."""
        from langgraph_coe.tools import wikidata as wd_mod

        client = getattr(wd_mod, "_wikidata_client", None)
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass

    # -- main entry -------------------------------------------------------------

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
    ) -> Dict:
        """Run evaluation: generate answers + compute metrics. Mirrors legacy output.

        Args:
            dataset: HuggingFace Dataset with question/answer (and optional level) columns.
            output_path: directory for the log, metrics, summary, and artifacts.
            resume: reuse answered questions from an existing ``evaluation_log.jsonl``.
            score_only: recompute scores from the existing log without generating.
            question_column / answer_column / level_column: dataset field names.
            max_concurrent: max concurrent graph invocations / Acc judge calls.
            log_batch_size: questions per generation chunk before appending to the log.
        """
        from langgraph_coe.evaluation.metrics import (
            compute_acc_batch,
            compute_aggregate_metrics_both,
            compute_aggregate_metrics_by_level,
            compute_sub_em,
            compute_sub_em_relaxed,
        )

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "evaluation_log.jsonl"
        artifacts_root = output_dir / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume / score_only.
        completed: Dict[str, Any] = {}
        completed_rows: List[Dict[str, Any]] = []
        if (resume or score_only) and log_file.exists():
            with open(log_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if "error" in entry:
                        logger.warning(
                            "Existing log entry with error for %r... (%s). Ignoring for resume.",
                            str(entry.get("question", "unknown"))[:60],
                            entry["error"],
                        )
                        continue
                    completed_rows.append(entry)
                    completed[entry["question"]] = entry
            print(f"Loaded: {len(completed)} questions from logfile")

        n = len(dataset)
        sub_ems_short: List[float] = [0.0] * n
        sub_ems_long: List[float] = [0.0] * n
        accs_short: List[Optional[float]] = [None] * n
        accs_long: List[Optional[float]] = [None] * n
        pass_at_k_values: List[Optional[int]] = [None] * n
        levels: List[str] = ["unknown"] * n
        predictions_short: Dict[str, str] = {}
        predictions_long: Dict[str, str] = {}

        for q, entry in completed.items():
            predictions_short[q] = entry.get("predicted_answer", "")
            predictions_long[q] = entry.get("full_answer", "")

        pending: List[tuple] = []
        scored_indices: List[int] = []
        matched_by_question = 0
        matched_by_index = 0
        for i, example in enumerate(dataset):
            question = example[question_column]
            correct = example[answer_column]
            level = (
                example.get(level_column, "unknown")
                if hasattr(example, "get")
                else "unknown"
            )
            levels[i] = level
            entry = completed.get(question)
            if entry is not None:
                matched_by_question += 1
            elif score_only and i < len(completed_rows):
                entry = completed_rows[i]
                matched_by_index += 1

            if entry is not None:
                scored_indices.append(i)
                if score_only:
                    predicted_short = entry.get("predicted_answer", "")
                    predicted_long = entry.get("full_answer", "")
                    sub_ems_short[i] = compute_sub_em(predicted_short, correct)
                    sub_ems_long[i] = compute_sub_em(predicted_long, correct)
                    predictions_short[question] = predicted_short
                    predictions_long[question] = predicted_long
                    accs_short[i] = None
                    accs_long[i] = None
                else:
                    sub_ems_short[i] = float(entry.get("sub_em_short", 0.0))
                    sub_ems_long[i] = float(entry.get("sub_em_long", 0.0))
                    accs_short[i] = entry.get("acc_short")
                    accs_long[i] = entry.get("acc_long")
                pass_at_k_values[i] = entry.get("pass_at_k")
            else:
                if not score_only:
                    pending.append((i, question, correct, level))

        if score_only:
            if not completed_rows:
                raise ValueError(
                    f"score_only=true but logfile not found or empty: {log_file}"
                )
            if matched_by_question + matched_by_index == 0:
                raise ValueError(
                    "score_only=true could not align dataset rows with evaluation_log.jsonl "
                    "(no question matches and no positional overlap)."
                )

        scored_index_set = set(scored_indices)

        progress_bar = tqdm(
            total=n,
            initial=n - len(pending),
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

        def _error_entry(
            i: int, question: str, correct: Any, level: str, err: Any
        ) -> Dict[str, Any]:
            """Build an error row and zero out this question's metric slots."""
            logger.error("Error processing question %d: %s", i, err)
            err_text = f"Error: {err}"
            sub_ems_short[i] = 0.0
            sub_ems_long[i] = 0.0
            pass_at_k_values[i] = None
            predictions_short[question] = err_text
            predictions_long[question] = err_text
            return {
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

        def process_answer_results(
            batch: List[tuple], results: List[Any]
        ) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for pending_item, result in zip(batch, results):
                i, question, correct = pending_item[0], pending_item[1], pending_item[2]
                level = pending_item[3] if len(pending_item) > 3 else "unknown"
                err = result.metadata.get("error") if result.metadata else None
                if err:
                    entry = _error_entry(i, question, correct, level, err)
                else:
                    try:
                        predicted_short = result.concise_answer or result.answer
                        predicted_long = result.answer
                        sub_em_short = compute_sub_em(predicted_short, correct)
                        sub_em_long = compute_sub_em(predicted_long, correct)
                        # Recorded alongside, never in place of, sub-EM. A gold answer that
                        # arrives wrapped ("at the city of Cairo, Illinois") cannot be
                        # matched verbatim by a correctly concise answer, and that scored
                        # 3 questions of this dataset wrong in EVERY run — +1.79 points
                        # understated, and up to 3 questions kept out of every paired
                        # comparison's discordant pool. See ``compute_sub_em_relaxed``.
                        sub_em_short_relaxed = compute_sub_em_relaxed(
                            predicted_short, correct
                        )
                        pass_at_k = (
                            result.metadata.get("pass_at_k")
                            if result.metadata
                            else None
                        )
                        entry = {
                            "question": question,
                            "correct_answer": correct,
                            "predicted_answer": predicted_short,
                            "full_answer": predicted_long,
                            "sub_em_short": sub_em_short,
                            "sub_em_long": sub_em_long,
                            "sub_em_short_relaxed": sub_em_short_relaxed,
                            "pass_at_k": pass_at_k,
                            "acc_short": None,
                            "acc_long": None,
                            "level": level,
                            # Per-question LLM spend, so a cost claim is checkable
                            # rather than inferred from hop and subquestion counts
                            # (which miss every call inside the retrieval subgraphs
                            # and weight an n=3 role the same as an n=1 one).
                            "cost": (result.metadata or {}).get("cost") or {},
                        }
                        try:
                            entry["artifacts"] = _save_question_artifacts(
                                artifacts_root, i, question, result
                            )
                        except Exception as artifact_error:  # noqa: BLE001
                            logger.warning(
                                "Could not persist artifacts for question %d: %s",
                                i,
                                artifact_error,
                            )
                            entry["artifacts_error"] = str(artifact_error)
                        sub_ems_short[i] = sub_em_short
                        sub_ems_long[i] = sub_em_long
                        pass_at_k_values[i] = pass_at_k
                        predictions_short[question] = predicted_short
                        predictions_long[question] = predicted_long
                    except Exception as e:  # noqa: BLE001
                        entry = _error_entry(i, question, correct, level, e)
                out.append(entry)
                logger.info(
                    "[%d/%d] Sub-EM Short: %.1f | Sub-EM Long: %.1f | Q: %s...",
                    i + 1,
                    n,
                    sub_ems_short[i],
                    sub_ems_long[i],
                    question[:60],
                )
            return out

        # Determine chunking (legacy used min(llm.concurrency, 8); we have no
        # concurrency knob, so default to 8 unless overridden).
        if log_batch_size is not None:
            chunk_size = max(1, log_batch_size)
        elif max_concurrent is not None:
            chunk_size = max(1, max_concurrent)
        else:
            chunk_size = 8

        try:
            if pending:
                asyncio.run(
                    self._generate_and_log(
                        pending,
                        log_file=log_file,
                        artifacts_root=artifacts_root,
                        process_answer_results=process_answer_results,
                        append_logs=append_logs,
                        progress_bar=progress_bar,
                        chunk_size=chunk_size,
                        max_concurrent=max_concurrent,
                    )
                )
        finally:
            progress_bar.close()

        # ── Acc scoring (LLM judge) ───────────────────────────────────────────
        acc_task_rows_short: List[tuple] = []
        acc_task_rows_long: List[tuple] = []
        for i, example in enumerate(dataset):
            if score_only and i not in scored_index_set:
                continue
            if accs_short[i] is not None and accs_long[i] is not None:
                continue
            question = example[question_column]
            if accs_short[i] is None:
                acc_task_rows_short.append(
                    (i, question, predictions_short.get(question, ""))
                )
            if accs_long[i] is None:
                acc_task_rows_long.append(
                    (i, question, predictions_long.get(question, ""))
                )

        total_acc_tasks = len(acc_task_rows_short) + len(acc_task_rows_long)
        if total_acc_tasks:
            registry = self._registry_for_scoring()
            acc_max = max_concurrent if max_concurrent is not None else 10
            acc_chunk = max(1, acc_max)
            acc_progress = tqdm(
                total=total_acc_tasks,
                desc="Scoring Acc",
                unit="q",
                dynamic_ncols=True,
                leave=True,
            )

            def _correct_for(i: int) -> Any:
                return dataset[i][answer_column]

            def _score_rows(
                task_rows: List[tuple], target: List[Optional[float]], label: str
            ) -> None:
                if not task_rows:
                    return
                try:
                    for start in range(0, len(task_rows), acc_chunk):
                        chunk_rows = task_rows[start : start + acc_chunk]
                        tasks = [
                            (q, pred, _correct_for(idx)) for idx, q, pred in chunk_rows
                        ]
                        scores = asyncio.run(
                            compute_acc_batch(tasks, registry, max_concurrent=acc_max)
                        )
                        for (idx, _, _), acc in zip(chunk_rows, scores):
                            target[idx] = acc
                        acc_progress.update(len(chunk_rows))
                except Exception as e:  # noqa: BLE001
                    logger.error("Acc %s batch failed: %s", label, e)
                    for idx, _, _ in task_rows:
                        if target[idx] is None:
                            target[idx] = 0.0
                            acc_progress.update(1)

            try:
                _score_rows(acc_task_rows_short, accs_short, "short")
                _score_rows(acc_task_rows_long, accs_long, "long")
            finally:
                acc_progress.close()

        # ── Aggregate + write ─────────────────────────────────────────────────
        if score_only:
            sel = scored_indices
            metric_sub_ems_short = [sub_ems_short[i] for i in sel]
            metric_sub_ems_long = [sub_ems_long[i] for i in sel]
            metric_accs_short = [accs_short[i] for i in sel]
            metric_accs_long = [accs_long[i] for i in sel]
            metric_pass = [pass_at_k_values[i] for i in sel]
            metric_levels = [levels[i] for i in sel]
        else:
            metric_sub_ems_short = sub_ems_short
            metric_sub_ems_long = sub_ems_long
            metric_accs_short = accs_short
            metric_accs_long = accs_long
            metric_pass = pass_at_k_values
            metric_levels = levels

        metrics = compute_aggregate_metrics_both(
            metric_sub_ems_short,
            metric_sub_ems_long,
            metric_accs_short,
            metric_accs_long,
            metric_pass,
        )

        if any(lev != "unknown" for lev in metric_levels):
            metrics["by_level"] = compute_aggregate_metrics_by_level(
                metric_sub_ems_short,
                metric_sub_ems_long,
                metric_accs_short,
                metric_accs_long,
                metric_levels,
                metric_pass,
            )

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

        logger.info("Evaluation complete. Metrics saved to %s", metrics_file)
        return metrics
