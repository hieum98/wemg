"""Utilities to load and inspect per-question evaluation artifacts."""

from __future__ import annotations

import json
import pickle
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional, Union

import networkx as nx

from coe.utils.graph import visualize_graph, visualize_graph_interactive


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
    """Full serialized node content on one line (newlines collapsed)."""
    node_type = str(node.get("node_type", "UNKNOWN"))
    content = node.get("content", {}) or {}

    def nl(s: str) -> str:
        return str(s).replace("\n", " ").replace("\r", " ")

    if node_type == "USER_QUESTION":
        gt = content.get("golden_answer", "N/A")
        return nl(f"User: {content.get('user_question', '')} - GT: {gt}")
    if node_type == "FINAL_ANSWER":
        parts = [f"Final: {content.get('final_answer', '')}"]
        if content.get("reasoning"):
            parts.append(f"Reasoning: {content['reasoning']}")
        if content.get("concise_answer"):
            parts.append(f"Concise: {content['concise_answer']}")
        return nl(" | ".join(parts))
    if node_type == "SUBQUESTION":
        parts = [
            f"Sub_Q: {content.get('sub_question', '')}",
            f"Sub_A: {content.get('sub_answer', '')}",
        ]
        if content.get("reasoning"):
            parts.append(f"Reasoning: {content['reasoning']}")
        return nl(" | ".join(parts))
    if node_type == "REPHRASE_QUESTION":
        return nl(f"Rephrase: {content.get('sub_question', '')}")
    if node_type == "SELF_CORRECT":
        parts = [
            f"Sub_Q: {content.get('sub_question', '')}",
            f"Sub_A: {content.get('sub_answer', '')}",
        ]
        if content.get("reasoning"):
            parts.append(f"Reasoning: {content['reasoning']}")
        return nl(" | ".join(parts))
    if node_type == "SYNTHESIS":
        return nl(f"Synthesis: {content.get('synthesized_reasoning', '')}")
    return nl(str(content))


def _format_tree_lines(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> list[str]:
    node_type = str(node.get("node_type", "UNKNOWN"))
    summary = _summarize_tree_node(node).replace("\n", " ").replace("\r", " ")
    mcts: list[str] = []
    if "visits" in node:
        mcts.append(f"visits={node['visits']}")
    if "value" in node:
        mcts.append(f"value={node['value']}")
    if mcts:
        summary = f"{summary} | {' '.join(mcts)}"
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
    """Print a saved search tree hierarchy (full node text, like system ``print_tree``)."""
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


def visualize_graph_memory_interactive(
    graph_or_path: Union[nx.DiGraph, str, Path],
    *,
    title: str = "Graph Memory",
    save_path: Optional[Union[str, Path]] = None,
    open_in_browser: bool = False,
    notebook_mode: bool = True,
    physics: bool = True,
    max_nodes: Optional[int] = None,
    neat_layout: bool = True,
) -> Optional[str]:
    """Notebook/browser helper: render graph memory as interactive HTML."""
    graph = (
        load_graph_memory(graph_or_path)
        if isinstance(graph_or_path, (str, Path))
        else graph_or_path
    )
    html_path = visualize_graph_interactive(
        graph,
        title=title,
        save_path=str(save_path) if save_path is not None else None,
        notebook=notebook_mode,
        physics=physics,
        max_nodes=max_nodes,
        neat_layout=neat_layout,
    )
    if html_path and open_in_browser:
        webbrowser.open(f"file://{Path(html_path).resolve()}")
    if html_path and notebook_mode:
        try:
            from IPython.display import IFrame, display

            display(IFrame(src=str(Path(html_path).resolve()), width="100%", height=780))
        except Exception:
            # Keep notebook helper resilient in non-IPython environments.
            pass
    return html_path

