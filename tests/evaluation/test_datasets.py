"""Tests for dataset loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from wemg.evaluation import datasets as datasets_mod


def test_extract_grailqa_answers_handles_local_schema():
    raw_answer = [
        {"answer_type": "Entity", "entity_name": "Set Designer", "answer_argument": "m.0b787yg"},
        {"answer_type": "Entity", "entity_name": "Set Designer", "answer_argument": "m.0b787yg"},
        {"answer_type": "Value", "answer_argument": "literal-value"},
        {"answer_type": "Entity", "entity_name": ""},
    ]

    answers = datasets_mod._extract_grailqa_answers(raw_answer)

    assert answers == ["Set Designer", "literal-value"]


def test_get_known_dataset_grailqa_prefers_local_dev(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    datasets = pytest.importorskip("datasets")

    local_dev = tmp_path / "grailqa_v1.0_dev.json"
    local_dev.write_text("[]", encoding="utf-8")

    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return datasets.Dataset.from_list(
            [
                {
                    "question": "who",
                    "answer": [{"entity_name": "Paris", "answer_argument": "m.paris"}],
                    "level": "i.i.d.",
                    "extra": 1,
                }
            ]
        )

    monkeypatch.setattr(datasets_mod, "_get_local_grailqa_dev_path", lambda: local_dev)
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    ds = datasets_mod._get_known_dataset("grailqa")

    assert calls
    assert calls[0][0] == ("json",)
    assert calls[0][1]["data_files"] == str(local_dev)
    assert calls[0][1]["split"] == "train"
    assert ds.column_names == ["question", "answer", "level"]
    assert ds[0]["answer"] == ["Paris"]
