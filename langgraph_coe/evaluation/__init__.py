"""Dataset evaluation for the ``langgraph_coe`` system.

Mirrors ``coe.evaluation`` one-for-one in **output**: the same
``evaluation_log.jsonl`` row schema, the same ``metrics.json`` /
``summary.txt`` / ``config.yaml`` files, and the same per-question ``artifacts``
layout — so results are directly comparable against the legacy system. Only the
answer-generation backend differs: this package drives the LangGraph CoT/MCTS
graphs (``langgraph_coe.system``) instead of ``COESystem``.

Entry point: ``python -m langgraph_coe.evaluation.evaluate``.
"""
