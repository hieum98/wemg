"""CLI: run dataset evaluation for the langgraph_coe system.

    python -m langgraph_coe.evaluation.evaluate \
        dataset_name_or_path=bamboogle output_path=./results/lgc_bamboogle \
        search.strategy=mcts

Accepts the same evaluation keys as ``wemg.evaluation.evaluate`` plus dotted
``langgraph_coe`` config overrides (e.g. ``search.strategy=cot``,
``search.mcts.num_iterations=8``, ``llm.tiers.heavy.api_base=...``). Writes the
resolved config to ``<output_path>/config.yaml`` for reproducibility, then runs
:class:`langgraph_coe.evaluation.runner.DatasetEvaluator`, which emits the same
``evaluation_log.jsonl`` / ``metrics.json`` / ``summary.txt`` as the legacy
system. Hydra-style ``+key=`` prefixes are accepted and ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.evaluation.datasets import load_dataset_any
from langgraph_coe.evaluation.runner import DatasetEvaluator

logger = logging.getLogger(__name__)

# Load the repo-root ``.env`` so ``API_KEY`` / ``OPENAI_API_KEY`` are available to
# ``from_yaml`` and litellm — same as the integration tests' bootstrap. Without
# this the LLM calls fail with ``litellm.AuthenticationError`` (the api_key never
# reaches the ChatLiteLLM models). ``langgraph_coe/evaluation/evaluate.py`` →
# parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    pass


def _ensure_api_keys(cfg: LangGraphCoeConfig) -> None:
    """Guarantee a non-empty api_key reaches every model (LLM tiers, embedder, reranker).

    ``RoleModelRegistry`` builds each ChatLiteLLM with ``tier.api_key or
    llm.api_key`` (``llm.py``); when both are unset, litellm raises
    ``AuthenticationError`` before the request is sent. SGLang ignores the value,
    so ``"EMPTY"`` is a safe last-resort default that still satisfies the client.
    """
    key = (
        cfg.llm.api_key
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    cfg.llm.api_key = key
    for tier in cfg.llm.tiers.values():
        if not tier.api_key:
            tier.api_key = key
    if not cfg.retriever.corpus.embedder.api_key:
        cfg.retriever.corpus.embedder.api_key = key
    if not cfg.reranker.api_key:
        cfg.reranker.api_key = key
    # Some litellm code paths read OPENAI_API_KEY from the environment directly.
    os.environ.setdefault("OPENAI_API_KEY", key)


EVAL_OVERRIDE_KEYS = frozenset(
    {
        "dataset_name_or_path",
        "output_path",
        "resume",
        "max_examples",
        "shuffle",
        "question_column",
        "answer_column",
        "level_column",
        "max_concurrent",
        "log_batch_size",
        "score_only",
    }
)


def _parse_override_value(raw: str) -> Any:
    """Parse a ``key=value`` RHS into bool / None / int / float / JSON / str."""
    text = raw.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text and text[0] in '[{"':
        try:
            return json.loads(text)
        except Exception:
            pass
    return text


def split_eval_overrides(
    overrides: Optional[List[str]],
) -> Tuple[Dict[str, Any], List[Tuple[str, Any]]]:
    """Separate evaluation kwargs from dotted config overrides."""
    eval_params: Dict[str, Any] = {}
    config_overrides: List[Tuple[str, Any]] = []
    for override in overrides or []:
        key, _, raw_value = override.partition("=")
        key_stripped = key.strip().lstrip("+")
        value = _parse_override_value(raw_value)
        if key_stripped in EVAL_OVERRIDE_KEYS:
            eval_params[key_stripped] = value
        else:
            config_overrides.append((key_stripped, value))
    return eval_params, config_overrides


def _apply_config_overrides(
    cfg: LangGraphCoeConfig, overrides: List[Tuple[str, Any]]
) -> None:
    """Set dotted attributes on a Pydantic config in place (e.g. ``search.strategy``)."""
    for dotted, value in overrides:
        parts = dotted.split(".")
        target: Any = cfg
        for part in parts[:-1]:
            if isinstance(target, dict):
                target = target[part]
            else:
                target = getattr(target, part)
        leaf = parts[-1]
        if isinstance(target, dict):
            target[leaf] = value
        else:
            setattr(target, leaf, value)


def _redact_api_keys(data: Any) -> Any:
    """Recursively null out any ``api_key`` field so secrets never hit disk."""
    if isinstance(data, dict):
        return {
            k: (None if k == "api_key" else _redact_api_keys(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_redact_api_keys(v) for v in data]
    return data


def _write_resolved_config_to_output(output_dir: Path, cfg: LangGraphCoeConfig) -> None:
    """Persist the resolved config (after env + overrides) for reproducibility.

    ``api_key`` fields are redacted to ``null`` so the resolved key (loaded from
    ``.env``) is never written into a shared results directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _redact_api_keys(cfg.model_dump(mode="json")),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _as_bool(value: Any, default: bool) -> bool:
    return bool(value) if value is not None else default


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Run langgraph_coe on a dataset and write metrics (JSONL log + metrics.json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: langgraph_coe/config.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Overrides: evaluation keys (dataset_name_or_path, output_path, resume, "
            "max_examples, shuffle, question_column, answer_column, level_column, "
            "max_concurrent, log_batch_size, score_only) plus dotted config keys "
            "(e.g. search.strategy=mcts, search.mcts.num_iterations=8). "
            "Hydra-style +prefix on keys is ignored."
        ),
    )
    args = parser.parse_args(argv)

    eval_params, config_overrides = split_eval_overrides(args.overrides)

    dataset_ref = eval_params.get("dataset_name_or_path")
    output_path = eval_params.get("output_path", "./results")
    resume = _as_bool(eval_params.get("resume"), True)
    score_only = _as_bool(eval_params.get("score_only"), False)
    shuffle = _as_bool(eval_params.get("shuffle"), False)
    max_examples = _as_int(eval_params.get("max_examples"))
    question_column = str(eval_params.get("question_column", "question"))
    answer_column = str(eval_params.get("answer_column", "answer"))
    level_column = str(eval_params.get("level_column", "level"))
    max_concurrent = _as_int(eval_params.get("max_concurrent"))
    log_batch_size = _as_int(eval_params.get("log_batch_size"))

    if not dataset_ref:
        parser.print_help()
        print(
            "\nError: set dataset_name_or_path=... (e.g. bamboogle or /path/to/data.jsonl).\n",
            file=sys.stderr,
        )
        return 1

    cfg = LangGraphCoeConfig.from_yaml(args.config)
    _apply_config_overrides(cfg, config_overrides)
    _ensure_api_keys(cfg)
    _write_resolved_config_to_output(Path(str(output_path)), cfg)

    ds = load_dataset_any(str(dataset_ref), max_examples=max_examples, shuffle=shuffle)
    evaluator = DatasetEvaluator(cfg)
    metrics = evaluator.evaluate(
        ds,
        output_path=str(output_path),
        resume=resume,
        score_only=score_only,
        question_column=question_column,
        answer_column=answer_column,
        level_column=level_column,
        max_concurrent=max_concurrent,
        log_batch_size=log_batch_size,
    )
    print("Done. Metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
