"""Utilities to load and inspect per-question evaluation artifacts."""

from __future__ import annotations

import json
import pickle
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx

from wemg.utils.graph import visualize_graph, visualize_graph_interactive


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


# ---------------------------------------------------------------------------
# Note artifacts
# ---------------------------------------------------------------------------

def load_notes(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load notes.json, returning a list of note dicts (each is a to_json() string parsed)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Each element is a JSON string (from MemoryNote.to_json())
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(json.loads(item))
        else:
            result.append(item)
    return result


def load_entity_graph(path: Union[str, Path]) -> nx.MultiDiGraph:
    """Load a pickled nx.MultiDiGraph entity structural index from disk."""
    with open(path, "rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(f"Expected nx.MultiDiGraph, got: {type(graph)}")
    return graph


def build_note_graph(notes: List[Dict[str, Any]]) -> nx.DiGraph:
    """Reconstruct a DiGraph of note-to-note links from a list of note dicts.

    Node attrs: ``label`` (first 60 chars of content), ``note_type``, ``confidence``.
    Edge attrs: ``link_type``.
    """
    g = nx.DiGraph()
    for note in notes:
        nid = note["id"]
        label = (note.get("content") or "")[:60].replace("\n", " ")
        g.add_node(
            nid,
            label=label,
            note_type=note.get("note_type", "fleeting"),
            confidence=note.get("confidence", 0.5),
        )
    for note in notes:
        nid = note["id"]
        for lnk in note.get("links", []):
            tid = lnk.get("target_id")
            lt = lnk.get("link_type", "")
            if tid and g.has_node(tid):
                g.add_edge(nid, tid, link_type=lt)
    return g


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
    notes_path = artifacts.get("notes_path")
    entity_graph_path = artifacts.get("entity_graph_path")

    notes = load_notes(notes_path) if notes_path else None
    entity_graph = load_entity_graph(entity_graph_path) if entity_graph_path else None

    return {
        "entry": selected,
        "artifact_paths": artifacts,
        "search_tree": load_search_tree_json(tree_path) if tree_path else None,
        "textual_memory": json.loads(Path(text_path).read_text(encoding="utf-8")) if text_path else None,
        "notes": notes,
        "entity_graph": entity_graph,
    }


# ---------------------------------------------------------------------------
# Note-graph visualization
# ---------------------------------------------------------------------------

def _build_note_viz_graph(notes_or_path: Union[List[Dict[str, Any]], str, Path]) -> nx.DiGraph:
    """Load notes (if path) and build a visualization DiGraph."""
    if isinstance(notes_or_path, (str, Path)):
        notes = load_notes(notes_or_path)
    else:
        notes = notes_or_path
    return build_note_graph(notes)


_NOTE_TYPE_COLORS = {
    "structure": "#e15759",
    "permanent": "#4e79a7",
    "literature": "#76b7b2",
    "fleeting": "#bab0ab",
}


def visualize_note_connections(
    notes_or_path: Union[List[Dict[str, Any]], str, Path],
    *,
    title: str = "Note Connections",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize note-to-note link graph (matplotlib)."""
    g = _build_note_viz_graph(notes_or_path)
    visualize_graph(
        g,
        title=title,
        save_path=str(save_path) if save_path is not None else None,
    )


def visualize_note_connections_interactive(
    notes_or_path: Union[List[Dict[str, Any]], str, Path],
    *,
    title: str = "Note Connections",
    save_path: Optional[Union[str, Path]] = None,
    open_in_browser: bool = False,
    notebook_mode: bool = True,
    physics: bool = True,
    max_nodes: Optional[int] = None,
    neat_layout: bool = True,
) -> Optional[str]:
    """Render note-to-note links as interactive HTML (pyvis).

    Node color reflects note_type: structure=red, permanent=blue,
    literature=teal, fleeting=grey.
    Edge label = link_type.
    """
    if isinstance(notes_or_path, (str, Path)):
        notes = load_notes(notes_or_path)
    else:
        notes = notes_or_path

    g = build_note_graph(notes)

    # Inject note-type color and link_type edge label into graph attrs
    # so that visualize_graph_interactive can pick them up.
    # We temporarily monkey-patch node "data" using a simple namespace.
    import types
    for nid, attrs in g.nodes(data=True):
        nt = attrs.get("note_type", "fleeting")
        label = attrs.get("label", nid[:40])
        node_ns = types.SimpleNamespace(
            label=label,
            note_type=nt,
            confidence=attrs.get("confidence", 0.5),
        )
        g.nodes[nid]["data"] = node_ns
        g.nodes[nid]["_note_type"] = nt

    for u, v, data in g.edges(data=True):
        data["relation"] = {data.get("link_type", "")}

    # Use a custom wrapper to colour nodes by note_type
    html_path = _visualize_note_graph_interactive(
        g,
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
            pass
    return html_path


def _visualize_note_graph_interactive(
    g: nx.DiGraph,
    title: str,
    save_path: Optional[str],
    notebook: bool,
    physics: bool,
    max_nodes: Optional[int],
    neat_layout: bool,
) -> Optional[str]:
    """Internal: render a note DiGraph with note_type-based colours."""
    import json as _json
    import os

    if len(g.nodes) == 0:
        print(f"{title}: Empty graph (no notes)")
        return None

    try:
        from pyvis.network import Network
    except ImportError:
        print("Warning: pyvis not available. Install pyvis for interactive visualization.")
        return None

    from wemg.utils.graph import _get_graph_output_filepath

    selected = g
    if max_nodes is not None and max_nodes > 0 and len(g.nodes) > max_nodes:
        ranked = sorted(g.nodes(), key=lambda n: g.in_degree(n) + g.out_degree(n), reverse=True)[:max_nodes]
        selected = g.subgraph(ranked).copy()

    net = Network(height="750px", width="100%", directed=True, notebook=notebook, cdn_resources="in_line")
    options = {
        "interaction": {"dragNodes": True, "dragView": True, "zoomView": True, "hover": True, "navigationButtons": True},
        "physics": {
            "enabled": bool(physics),
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -90, "centralGravity": 0.01, "springLength": 140, "springConstant": 0.05, "damping": 0.75, "avoidOverlap": 1.0},
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 900, "updateInterval": 25},
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "smooth": {"enabled": True, "type": "dynamic"},
            "color": {"opacity": 0.6},
            "font": {"size": 11, "strokeWidth": 0},
        },
        "nodes": {"font": {"size": 13}, "shape": "dot"},
    }
    if neat_layout:
        options["layout"] = {"improvedLayout": True, "randomSeed": 17}
    net.set_options(_json.dumps(options))

    for nid in selected.nodes():
        attrs = selected.nodes[nid]
        nt = attrs.get("_note_type", "fleeting")
        color = _NOTE_TYPE_COLORS.get(nt, "#bab0ab")
        label = attrs.get("label", str(nid)[:45])
        degree = selected.in_degree(nid) + selected.out_degree(nid)
        size = max(14, min(60, 14 + int(degree * 2)))
        title_text = f"[{nt}] conf={attrs.get('confidence', 0.5):.2f}\n{label}"
        net.add_node(str(nid), label=str(label)[:45], title=title_text, size=size, color=color)

    for u, v, data in selected.edges(data=True):
        lt = data.get("link_type", "")
        net.add_edge(str(u), str(v), label=lt, title=lt)

    output_path = (
        _get_graph_output_filepath(str(save_path), title, "html")
        if save_path is not None
        else _get_graph_output_filepath("./tmp", title, "html")
    )
    net.save_graph(output_path)
    print(f"Note connections visualization saved to {os.path.abspath(output_path)}")
    if notebook:
        try:
            from IPython.display import IFrame, display
            display(IFrame(src=str(Path(output_path).resolve()), width="100%", height=780))
        except Exception:
            pass
    return output_path


# ---------------------------------------------------------------------------
# Entity-graph visualization
# ---------------------------------------------------------------------------

def visualize_entity_connections_interactive(
    graph_or_path: Union[nx.MultiDiGraph, str, Path],
    *,
    title: str = "Entity Graph",
    save_path: Optional[Union[str, Path]] = None,
    open_in_browser: bool = False,
    notebook_mode: bool = True,
    physics: bool = True,
    max_nodes: Optional[int] = None,
    neat_layout: bool = True,
) -> Optional[str]:
    """Render the entity structural index as interactive HTML (pyvis).

    Node colour: QID entities = blue, literal scalars = orange.
    Edge label = rel_label.
    """
    import json as _json
    import os

    graph = (
        load_entity_graph(graph_or_path)
        if isinstance(graph_or_path, (str, Path))
        else graph_or_path
    )

    if len(graph.nodes) == 0:
        print(f"{title}: Empty entity graph (no nodes)")
        return None

    try:
        from pyvis.network import Network
    except ImportError:
        print("Warning: pyvis not available. Install pyvis for interactive visualization.")
        return None

    from wemg.utils.graph import _get_graph_output_filepath

    selected = graph
    if max_nodes is not None and max_nodes > 0 and len(graph.nodes) > max_nodes:
        ranked = sorted(graph.nodes(), key=lambda n: graph.in_degree(n) + graph.out_degree(n), reverse=True)[:max_nodes]
        selected = graph.subgraph(ranked).copy()

    net = Network(height="750px", width="100%", directed=True, notebook=notebook_mode, cdn_resources="in_line")
    options = {
        "interaction": {"dragNodes": True, "dragView": True, "zoomView": True, "hover": True, "navigationButtons": True},
        "physics": {
            "enabled": bool(physics),
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -90, "centralGravity": 0.01, "springLength": 140, "springConstant": 0.05, "damping": 0.75, "avoidOverlap": 1.0},
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 900, "updateInterval": 25},
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "smooth": {"enabled": True, "type": "dynamic"},
            "color": {"opacity": 0.6},
            "font": {"size": 11, "strokeWidth": 0},
        },
        "nodes": {"font": {"size": 13}, "shape": "dot"},
    }
    if neat_layout:
        options["layout"] = {"improvedLayout": True, "randomSeed": 17}
    net.set_options(_json.dumps(options))

    for nid in selected.nodes():
        attrs = selected.nodes[nid]
        is_literal = attrs.get("node_type") == "literal"
        color = "#f28e2b" if is_literal else "#4e79a7"
        label = attrs.get("label", str(nid))
        desc = attrs.get("description", "")
        degree = selected.in_degree(nid) + selected.out_degree(nid)
        size = max(14, min(60, 14 + int(degree * 2)))
        title_text = f"{'[literal] ' if is_literal else ''}{label}"
        if desc:
            title_text += f"\n{desc}"
        net.add_node(str(nid), label=str(label)[:45], title=title_text, size=size, color=color)

    for u, v, key, data in selected.edges(data=True, keys=True):
        rel = data.get("rel_label", "")
        net.add_edge(str(u), str(v), label=rel, title=rel)

    output_path = (
        _get_graph_output_filepath(str(save_path), title, "html")
        if save_path is not None
        else _get_graph_output_filepath("./tmp", title, "html")
    )
    net.save_graph(output_path)
    print(f"Entity graph visualization saved to {os.path.abspath(output_path)}")

    if open_in_browser:
        webbrowser.open(f"file://{Path(output_path).resolve()}")
    if notebook_mode:
        try:
            from IPython.display import IFrame, display
            display(IFrame(src=str(Path(output_path).resolve()), width="100%", height=780))
        except Exception:
            pass
    return output_path
