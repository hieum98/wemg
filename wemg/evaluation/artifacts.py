"""Utilities to load and inspect per-question evaluation artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import networkx as nx

from wemg.utils.graph import visualize_graph


def _read_jsonl_entries(log_path: Path) -> list[dict[str, Any]]:
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_artifacts_entry(
    output_path: Union[str, Path],
    *,
    index: Optional[int] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """Find one question log entry containing artifact references."""
    if index is None and question is None:
        raise ValueError("Provide either index or question.")

    log_path = Path(output_path) / "evaluation_log.jsonl"
    entries = _read_jsonl_entries(log_path)
    if index is not None:
        if index < 0 or index >= len(entries):
            raise IndexError(f"index {index} out of range for {len(entries)} entries")
        return entries[index]
    for entry in entries:
        if entry.get("question") == question:
            return entry
    raise ValueError(f"No entry found for question: {question!r}")


def load_search_tree_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved search tree JSON payload."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summarize_tree_node(node: Dict[str, Any]) -> str:
    node_type = str(node.get("node_type", "UNKNOWN"))
    content = node.get("content", {}) or {}
    if node_type == "USER_QUESTION":
        return f"User: {str(content.get('user_question', ''))[:80]}"
    if node_type == "FINAL_ANSWER":
        return f"Final: {str(content.get('final_answer', ''))[:80]}"
    if node_type == "SUBQUESTION":
        sub_q = str(content.get("sub_question", ""))[:60]
        sub_a = str(content.get("sub_answer", ""))[:60]
        return f"Sub_Q: {sub_q} - Sub_A: {sub_a}"
    if node_type == "REPHRASE_QUESTION":
        return f"Rephrase: {str(content.get('sub_question', ''))[:80]}"
    if node_type == "SELF_CORRECT":
        return f"Self_corrected: {str(content.get('sub_answer', ''))[:80]}"
    if node_type == "SYNTHESIS":
        return f"Synthesis: {str(content.get('synthesized_reasoning', ''))[:80]}"
    return str(content)[:80]


def _format_tree_lines(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> list[str]:
    node_type = str(node.get("node_type", "UNKNOWN"))
    summary = _summarize_tree_node(node).replace("\n", " ").replace("\r", " ")
    marker = "└── " if is_last else "├── "
    lines = [f"{prefix}{marker}{node_type} {summary}"]

    children = [c for c in (node.get("children", []) or []) if isinstance(c, dict)]
    if children:
        child_prefix = prefix + ("    " if is_last else "│   ")
        for idx, child in enumerate(children):
            lines.extend(
                _format_tree_lines(
                    child,
                    prefix=child_prefix,
                    is_last=(idx == len(children) - 1),
                )
            )
    return lines


def print_saved_search_tree(
    tree_or_path: Union[Dict[str, Any], str, Path],
) -> str:
    """Print a saved search tree in a compact hierarchy similar to system output."""
    tree = (
        load_search_tree_json(tree_or_path)
        if isinstance(tree_or_path, (str, Path))
        else tree_or_path
    )
    if not isinstance(tree, dict):
        raise TypeError(f"Expected dict search tree payload, got: {type(tree)}")
    lines = _format_tree_lines(tree, prefix="", is_last=True)
    rendered = "\n".join(lines)
    print(rendered)
    return rendered


def load_graph_memory(path: Union[str, Path]) -> nx.DiGraph:
    """Load a pickled networkx graph from disk."""
    with open(path, "rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError(f"Expected networkx graph, got: {type(graph)}")
    return graph


def load_question_artifacts(
    output_path: Union[str, Path],
    *,
    index: Optional[int] = None,
    question: Optional[str] = None,
    entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load one question's saved artifacts by log index, question text, or log entry."""
    selected = entry if entry is not None else find_artifacts_entry(output_path, index=index, question=question)
    artifacts = selected.get("artifacts", {}) or {}

    tree_path = artifacts.get("search_tree_path")
    text_path = artifacts.get("textual_memory_path")
    graph_path = artifacts.get("graph_memory_path")

    return {
        "entry": selected,
        "artifact_paths": artifacts,
        "search_tree": load_search_tree_json(tree_path) if tree_path else None,
        "textual_memory": json.loads(Path(text_path).read_text(encoding="utf-8")) if text_path else None,
        "graph_memory": load_graph_memory(graph_path) if graph_path else None,
    }


def visualize_graph_memory(
    graph_or_path: Union[nx.DiGraph, str, Path],
    *,
    title: str = "Graph Memory",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Notebook helper: visualize graph memory from object or saved path."""
    graph = (
        load_graph_memory(graph_or_path)
        if isinstance(graph_or_path, (str, Path))
        else graph_or_path
    )
    visualize_graph(
        graph,
        title=title,
        save_path=str(save_path) if save_path is not None else None,
    )

