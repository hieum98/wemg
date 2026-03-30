"""Unit tests for pass_at_k evaluator-threshold semantics."""

from __future__ import annotations

import pytest

from wemg.reasoning.nodes import CoTNode, MCTSNode, NodeState, NodeType


class _DummyWorkingMemory:
    def add_textual_memory(self, *args, **kwargs):
        return None

    async def synchronize_memory(self, *args, **kwargs):
        return None


class _FakeNodeGenerator:
    def __init__(self, *args, working_memory=None, interaction_memory=None, **kwargs):
        self.working_memory = working_memory or _DummyWorkingMemory()
        self.interaction_memory = interaction_memory

    def update_working_memory(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_cot_pass_at_k_when_acc_above_threshold(monkeypatch):
    from wemg.reasoning import cot as cot_module

    monkeypatch.setattr(cot_module, "NodeGenerator", _FakeNodeGenerator)

    async def _fake_next_step(current, generator, working_memory, interaction_memory=None):
        return CoTNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    "user_question": current.user_question,
                    "final_answer": "Paris",
                    "concise_answer": "Paris",
                },
            ),
            parent=current,
            max_depth=current.max_depth,
        )

    async def _fake_compute_acc(*args, **kwargs):
        return 0.81

    monkeypatch.setattr(cot_module, "generate_next_step", _fake_next_step)
    monkeypatch.setattr(cot_module, "compute_acc", _fake_compute_acc)

    terminal_content, reasoning_path, pass_at_k = await cot_module.cot_search(
        question="What is the capital of France?",
        client=object(),
        retriever=None,
        wikidata_client=None,
        working_memory=_DummyWorkingMemory(),
        correct_answers=["Paris"],
        max_depth=2,
    )

    assert terminal_content is not None
    assert reasoning_path[-1].is_terminal()
    assert pass_at_k == 1


@pytest.mark.asyncio
async def test_cot_pass_at_k_requires_strict_threshold(monkeypatch):
    from wemg.reasoning import cot as cot_module

    monkeypatch.setattr(cot_module, "NodeGenerator", _FakeNodeGenerator)

    async def _fake_next_step(current, generator, working_memory, interaction_memory=None):
        return CoTNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    "user_question": current.user_question,
                    "final_answer": "Paris",
                    "concise_answer": "Paris",
                },
            ),
            parent=current,
            max_depth=current.max_depth,
        )

    async def _fake_compute_acc(*args, **kwargs):
        return 0.8

    monkeypatch.setattr(cot_module, "generate_next_step", _fake_next_step)
    monkeypatch.setattr(cot_module, "compute_acc", _fake_compute_acc)

    _, _, pass_at_k = await cot_module.cot_search(
        question="What is the capital of France?",
        client=object(),
        retriever=None,
        wikidata_client=None,
        working_memory=_DummyWorkingMemory(),
        correct_answers=["Paris"],
        max_depth=2,
    )

    assert pass_at_k is None


@pytest.mark.asyncio
async def test_mcts_pass_at_k_when_acc_above_threshold(monkeypatch):
    from wemg.reasoning import mcts as mcts_module

    monkeypatch.setattr(mcts_module, "NodeGenerator", _FakeNodeGenerator)

    def _fake_select(root, exploration_weight=2.0):
        return [root]

    async def _fake_expand(node, generator):
        terminal = MCTSNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    "user_question": node.user_question,
                    "final_answer": "Paris",
                    "concise_answer": "Paris",
                },
            ),
            parent=node,
            max_depth=node.max_depth,
        )
        return [terminal], False

    async def _fake_evaluate(node, client, working_memory, golden_answer=None):
        return 0.5

    async def _fake_compute_acc(*args, **kwargs):
        return 0.81

    monkeypatch.setattr(mcts_module, "select", _fake_select)
    monkeypatch.setattr(mcts_module, "expand", _fake_expand)
    monkeypatch.setattr(mcts_module, "evaluate", _fake_evaluate)
    monkeypatch.setattr(mcts_module, "compute_acc", _fake_compute_acc)

    _, _, pass_at_k = await mcts_module.mcts_search(
        question="What is the capital of France?",
        client=object(),
        retriever=None,
        wikidata_client=None,
        working_memory=_DummyWorkingMemory(),
        correct_answers=["Paris"],
        num_iterations=1,
    )

    assert pass_at_k == 1


@pytest.mark.asyncio
async def test_mcts_pass_at_k_requires_strict_threshold(monkeypatch):
    from wemg.reasoning import mcts as mcts_module

    monkeypatch.setattr(mcts_module, "NodeGenerator", _FakeNodeGenerator)

    def _fake_select(root, exploration_weight=2.0):
        return [root]

    async def _fake_expand(node, generator):
        terminal = MCTSNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    "user_question": node.user_question,
                    "final_answer": "Paris",
                    "concise_answer": "Paris",
                },
            ),
            parent=node,
            max_depth=node.max_depth,
        )
        return [terminal], False

    async def _fake_evaluate(node, client, working_memory, golden_answer=None):
        return 0.5

    async def _fake_compute_acc(*args, **kwargs):
        return 0.8

    monkeypatch.setattr(mcts_module, "select", _fake_select)
    monkeypatch.setattr(mcts_module, "expand", _fake_expand)
    monkeypatch.setattr(mcts_module, "evaluate", _fake_evaluate)
    monkeypatch.setattr(mcts_module, "compute_acc", _fake_compute_acc)

    _, _, pass_at_k = await mcts_module.mcts_search(
        question="What is the capital of France?",
        client=object(),
        retriever=None,
        wikidata_client=None,
        working_memory=_DummyWorkingMemory(),
        correct_answers=["Paris"],
        num_iterations=1,
    )

    assert pass_at_k is None
