from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wemg.evaluation.runner import DatasetEvaluator
from wemg.system import AnswerResult


class _FakeSystem:
    def __init__(self, cache_path: str):
        self.client = None
        self.cfg = SimpleNamespace(
            llm=SimpleNamespace(concurrency=1),
            node_generation=SimpleNamespace(
                kb_source="freebase_subgraph",
                freebase_subgraph_cache=cache_path,
            ),
        )
        self.received_cache_entries = []

    def answer_batch(
        self,
        questions,
        question_ids=None,
        golden_answers=None,
        subgraph_cache_entries=None,
        max_workers=None,
    ):
        self.received_cache_entries.extend(subgraph_cache_entries or [])
        return [
            AnswerResult(
                question=q,
                answer="dummy-answer",
                concise_answer="dummy-answer",
                metadata={},
            )
            for q in questions
        ]


def _write_split_cache(cache_root: Path, questions: list[str]) -> None:
    entries_dir = cache_root / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for idx, q in enumerate(questions):
        filename = f"entries/q_{idx}.json"
        payload = {
            "id": str(idx),
            "question": q,
            "q_entity_qids": [],
            "entity_dict": {},
            "property_dict": {},
            "triples": [],
            "embeddings": {},
        }
        (cache_root / filename).write_text(json.dumps(payload), encoding="utf-8")
        index_rows.append({"question": q, "id": str(idx), "entry_path": filename})

    with open(cache_root / "index.jsonl", "w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row) + "\n")


def test_evaluate_freebase_subgraph_only_loads_pending_questions(tmp_path: Path):
    cache_root = tmp_path / "split_cache"
    _write_split_cache(cache_root, ["q1", "q2", "q3"])

    # Mark q1 as already completed; evaluator should only load q2 from split cache.
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_entry = {
        "question": "q1",
        "correct_answer": "a1",
        "predicted_answer": "a1",
        "full_answer": "a1",
        "sub_em_short": 1.0,
        "sub_em_long": 1.0,
        "acc_short": None,
        "acc_long": None,
        "pass_at_k": None,
    }
    with open(output_dir / "evaluation_log.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(completed_entry) + "\n")

    system = _FakeSystem(str(cache_root))
    evaluator = DatasetEvaluator(system)
    dataset = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]

    metrics = evaluator.evaluate(dataset, output_path=str(output_dir), resume=True)

    assert metrics["short_answer"]["total_questions"] == 2
    assert metrics["long_answer"]["total_questions"] == 2
    assert len(system.received_cache_entries) == 1
    assert system.received_cache_entries[0]["question"] == "q2"


def test_evaluate_freebase_subgraph_rejects_monolithic_jsonl_cache(tmp_path: Path):
    monolithic_cache = tmp_path / "cache.jsonl"
    monolithic_cache.write_text('{"question": "q", "triples": []}\n', encoding="utf-8")

    system = _FakeSystem(str(monolithic_cache))
    evaluator = DatasetEvaluator(system)

    with pytest.raises(ValueError, match="Monolithic JSONL cache"):
        evaluator.evaluate(
            [{"question": "q", "answer": "a"}],
            output_path=str(tmp_path / "out2"),
            resume=False,
        )
