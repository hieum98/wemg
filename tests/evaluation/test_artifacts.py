"""Tests for evaluation artifact persistence and loading helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import networkx as nx

from wemg.evaluation.artifacts import (
    load_graph_memory,
    load_question_artifacts,
    print_saved_search_tree,
    visualize_graph_memory,
)
from wemg.evaluation.runner import DatasetEvaluator
from wemg.reasoning.memory import WorkingMemory
from wemg.reasoning.nodes import CoTNode, NodeState, NodeType
from wemg.system import AnswerResult


def _build_search_tree(question: str) -> CoTNode:
    root = CoTNode(
        node_state=NodeState(
            node_type=NodeType.USER_QUESTION,
            content={"user_question": question},
        ),
        max_depth=4,
    )
    CoTNode(
        node_state=NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={
                "user_question": question,
                "final_answer": "Paris",
                "concise_answer": "Paris",
            },
        ),
        parent=root,
        max_depth=4,
    )
    return root


def _build_working_memory() -> WorkingMemory:
    wm = WorkingMemory()
    wm.textual_memory = ["[System Prediction]: France has capital Paris."]
    wm.graph_memory = nx.DiGraph()
    wm.graph_memory.add_node("France", data="France")
    wm.graph_memory.add_node("Paris", data="Paris")
    wm.graph_memory.add_edge("France", "Paris", relation={"has capital"})
    return wm


class _FakeSystem:
    def __init__(self, result: AnswerResult):
        self._result = result
        self.client = None
        self.cfg = SimpleNamespace(llm=SimpleNamespace(concurrency=1))

    def answer_batch(self, questions, question_ids=None, golden_answers=None, max_workers=None):
        return [self._result for _ in questions]


def test_evaluate_persists_per_question_artifacts(tmp_path: Path):
    question = "What is the capital of France?"
    result = AnswerResult(
        question=question,
        answer="Final Answer: Paris",
        concise_answer="Paris",
        search_tree=_build_search_tree(question),
        metadata={"strategy": "cot"},
        working_memory=_build_working_memory(),
    )
    evaluator = DatasetEvaluator(_FakeSystem(result))
    dataset = [{"question": question, "answer": "Paris"}]

    metrics = evaluator.evaluate(dataset, output_path=str(tmp_path), resume=False)
    assert metrics["total_questions"] == 1

    log_path = tmp_path / "evaluation_log.jsonl"
    line = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()][0]
    entry = json.loads(line)
    artifacts = entry["artifacts"]
    assert Path(artifacts["artifact_dir"]).is_dir()
    assert Path(artifacts["search_tree_path"]).is_file()
    assert Path(artifacts["textual_memory_path"]).is_file()
    assert Path(artifacts["graph_memory_path"]).is_file()


def test_artifact_helpers_load_and_visualize(tmp_path: Path):
    question = "What is the capital of France?"
    result = AnswerResult(
        question=question,
        answer="Final Answer: Paris",
        concise_answer="Paris",
        search_tree=_build_search_tree(question),
        metadata={"strategy": "cot"},
        working_memory=_build_working_memory(),
    )
    evaluator = DatasetEvaluator(_FakeSystem(result))
    evaluator.evaluate([{"question": question, "answer": "Paris"}], output_path=str(tmp_path), resume=False)

    loaded = load_question_artifacts(tmp_path, index=0)
    assert loaded["search_tree"]["node_type"] == NodeType.USER_QUESTION.value
    assert isinstance(loaded["textual_memory"], list)
    assert isinstance(loaded["graph_memory"], nx.DiGraph)
    assert loaded["graph_memory"].number_of_edges() == 1

    graph_path = loaded["artifact_paths"]["graph_memory_path"]
    graph = load_graph_memory(graph_path)
    assert isinstance(graph, nx.DiGraph)

    img_path = tmp_path / "graph.png"
    visualize_graph_memory(graph_path, title="Test Graph", save_path=img_path)
    assert img_path.is_file()


def test_print_saved_search_tree(tmp_path: Path, capsys):
    question = "What is the capital of France?"
    result = AnswerResult(
        question=question,
        answer="Final Answer: Paris",
        concise_answer="Paris",
        search_tree=_build_search_tree(question),
        metadata={"strategy": "cot"},
        working_memory=_build_working_memory(),
    )
    evaluator = DatasetEvaluator(_FakeSystem(result))
    evaluator.evaluate([{"question": question, "answer": "Paris"}], output_path=str(tmp_path), resume=False)

    loaded = load_question_artifacts(tmp_path, index=0)
    tree_path = loaded["artifact_paths"]["search_tree_path"]
    rendered = print_saved_search_tree(tree_path)
    out = capsys.readouterr().out
    assert "USER_QUESTION" in rendered
    assert "FINAL_ANSWER" in rendered
    assert rendered in out

