"""CLI: run dataset evaluation (``python -m wemg.evaluation.evaluate``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from wemg.config import WEMGConfig, _parse_override_value, get_default_config_path
from wemg.evaluation.datasets import load_dataset_any
from wemg.evaluation.runner import DatasetEvaluator
from wemg.system import WEMGSystem

logger = logging.getLogger(__name__)

# Parsed from ``key=value`` overrides (Hydra-style ``+key=`` is accepted).
EVAL_OVERRIDE_KEYS = frozenset(
    {
        "dataset_name_or_path",
        "output_path",
        "resume",
        "max_examples",
        "shuffle",
        "question_column",
        "answer_column",
        "max_concurrent",
        "log_batch_size",
        "score_only",
        "prediction_column",
    }
)


def split_eval_overrides(
    overrides: Optional[List[str]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Separate evaluation kwargs from WEMG config overrides."""
    eval_params: Dict[str, Any] = {}
    config_overrides: List[str] = []
    for override in overrides or []:
        key, _, raw_value = override.partition("=")
        key_stripped = key.strip().lstrip("+")
        if key_stripped in EVAL_OVERRIDE_KEYS:
            eval_params[key_stripped] = _parse_override_value(raw_value.strip())
        else:
            config_overrides.append(f"{key_stripped}={raw_value.strip()}")
    return eval_params, config_overrides


def _write_resolved_config_to_output(output_dir: Path, cfg: WEMGConfig) -> None:
    """Persist the resolved WEMG config (after env + overrides) for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    data = cfg.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run WEMG on a dataset and write metrics (JSONL log + metrics.json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to config YAML (default: {get_default_config_path()})",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Overrides: evaluation keys (dataset_name_or_path, output_path, resume, max_examples, "
            "shuffle, question_column, answer_column, max_concurrent, log_batch_size, score_only, "
            "prediction_column) plus WEMG config keys (e.g. llm.model_name=..., search.strategy=mcts). "
            "Hydra-style +prefix on keys is ignored."
        ),
    )
    args = parser.parse_args(argv)

    eval_params, config_overrides = split_eval_overrides(args.overrides)

    dataset_ref = eval_params.get("dataset_name_or_path")
    output_path = eval_params.get("output_path", "./results")
    resume = eval_params.get("resume", True)
    if not isinstance(resume, bool):
        resume = bool(resume)
    max_examples = eval_params.get("max_examples")
    if max_examples is not None and not isinstance(max_examples, int):
        try:
            max_examples = int(max_examples)
        except (TypeError, ValueError):
            logger.warning("max_examples not an int; ignoring")
            max_examples = None
    shuffle = eval_params.get("shuffle", False)
    if not isinstance(shuffle, bool):
        shuffle = bool(shuffle)
    question_column = eval_params.get("question_column", "question")
    answer_column = eval_params.get("answer_column", "answer")
    max_concurrent = eval_params.get("max_concurrent")
    if max_concurrent is not None and not isinstance(max_concurrent, int):
        try:
            max_concurrent = int(max_concurrent)
        except (TypeError, ValueError):
            max_concurrent = None
    log_batch_size = eval_params.get("log_batch_size")
    if log_batch_size is not None and not isinstance(log_batch_size, int):
        try:
            log_batch_size = int(log_batch_size)
        except (TypeError, ValueError):
            log_batch_size = None
    score_only = eval_params.get("score_only", False)
    if not isinstance(score_only, bool):
        score_only = bool(score_only)
    prediction_column = eval_params.get("prediction_column", "predicted_answer")

    if not dataset_ref:
        parser.print_help()
        print(
            "\nError: set dataset_name_or_path=... (e.g. bamboogle or /path/to/data.jsonl).\n",
            file=sys.stderr,
        )
        return 1

    system = WEMGSystem(config_path=args.config, config_overrides=config_overrides)
    _write_resolved_config_to_output(Path(str(output_path)), system.cfg)
    try:
        ds = load_dataset_any(
            str(dataset_ref),
            max_examples=max_examples,
            shuffle=shuffle,
        )
        evaluator = DatasetEvaluator(system)
        if score_only:
            metrics = evaluator.score_from_predictions(
                ds,
                output_path=str(output_path),
                question_column=str(question_column),
                answer_column=str(answer_column),
                prediction_column=str(prediction_column),
            )
        else:
            metrics = evaluator.evaluate(
                ds,
                output_path=str(output_path),
                resume=resume,
                question_column=str(question_column),
                answer_column=str(answer_column),
                max_concurrent=max_concurrent,
                log_batch_size=log_batch_size,
            )
        print("Done. Metrics:", metrics)
        return 0
    finally:
        system.close()


if __name__ == "__main__":
    raise SystemExit(main())
