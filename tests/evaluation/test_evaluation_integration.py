"""Evaluation module: real-system runs on small dataset slices (slow / integration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_corpus_paths, requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output
from wemg.evaluation.datasets import load_dataset_any
from wemg.evaluation.evaluate import split_eval_overrides
from wemg.evaluation.runner import DatasetEvaluator
from wemg.system import WEMGSystem

# Small slice for CI / local smoke (full WEMG answer + optional Acc LLM calls per row).
EVAL_MAX_EXAMPLES = 16


def test_split_eval_overrides_routes_keys():
    """Fast: evaluation overrides are peeled off; rest go to WEMG config."""
    eval_params, cfg = split_eval_overrides(
        [
            "+dataset_name_or_path=bamboogle",
            "max_examples=16",
            "resume=false",
            "llm.concurrency=2",
            "search.strategy=cot",
        ]
    )
    assert eval_params["dataset_name_or_path"] == "bamboogle"
    assert eval_params["max_examples"] == 16
    assert eval_params["resume"] is False
    assert cfg == ["llm.concurrency=2", "search.strategy=cot"]


def _load_eval_slice(dataset_key: str, n: int):
    pytest.importorskip("datasets")
    try:
        return load_dataset_any(dataset_key, max_examples=n, shuffle=False)
    except Exception as exc:  # noqa: BLE001 — hub/network/private dataset
        pytest.skip(f"Could not load dataset {dataset_key!r}: {exc}")


def _assert_qa_columns(dataset) -> tuple[str, str]:
    cols = dataset.column_names
    if "question" not in cols or "answer" not in cols:
        pytest.skip(f"Dataset missing question/answer columns; got {cols!r}")
    return "question", "answer"


@pytest.mark.requires_llm
@pytest.mark.slow_integration
@pytest.mark.integration
@pytest.mark.parametrize("dataset_key", ["qald_10", "bamboogle"])
def test_evaluate_dataset_real_system_small_slice(
    tmp_path: Path,
    wemg_config_path: Path,
    wemg_config,
    smoke_system_overrides: list,
    dataset_key: str,
):
    """Run ``DatasetEvaluator.evaluate`` on up to 16 examples (CoT + smoke search limits)."""
    requires_llm_credentials(wemg_config)
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)

    ds = _load_eval_slice(dataset_key, EVAL_MAX_EXAMPLES)
    q_col, a_col = _assert_qa_columns(ds)
    n = len(ds)
    assert 1 <= n <= EVAL_MAX_EXAMPLES

    overrides = list(smoke_system_overrides) + [
        "search.strategy=cot",
        "llm.concurrency=2",
    ]
    system = WEMGSystem(config_path=wemg_config_path, config_overrides=overrides)
    out_dir = tmp_path / f"eval_{dataset_key}"
    try:
        metrics = DatasetEvaluator(system).evaluate(
            ds,
            output_path=str(out_dir),
            resume=False,
            question_column=q_col,
            answer_column=a_col,
            max_concurrent=2,
            log_batch_size=4,
        )
        print_slow_integration_output(
            f"test_evaluate_dataset_real_system_small_slice[{dataset_key}]",
            metrics=metrics,
            output_dir=str(out_dir),
        )

        assert metrics["total_questions"] == n
        assert metrics["valid_questions"] == n
        assert 0.0 <= metrics["mean_sub_em"] <= 1.0
        assert "mean_acc" in metrics
        assert 0.0 <= metrics["mean_acc"] <= 1.0

        metrics_path = out_dir / "metrics.json"
        assert metrics_path.is_file()
        loaded = json.loads(metrics_path.read_text())
        assert loaded["total_questions"] == n

        log_path = out_dir / "evaluation_log.jsonl"
        assert log_path.is_file()
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == n
        first = json.loads(lines[0])
        assert "question" in first and "predicted_answer" in first and "sub_em" in first
    finally:
        system.close()
