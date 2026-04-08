"""Graph utilities for textualization and visualization."""

import os
import json
from pathlib import Path
from collections import deque
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import networkx as nx


def get_node_id(entity: Any) -> str:
    """Get a string ID for an entity. Uses .qid for WikidataEntity, .id for Entity, else str()."""
    if hasattr(entity, "qid") and entity.qid:
        return entity.qid
    elif hasattr(entity, "id") and entity.id:
        return str(entity.id)
    return str(entity)


def get_densest_node(
    component: Set,
    graph: Union[nx.DiGraph, nx.Graph],
    filter_func: Callable[[Any], bool] = None,
) -> Optional[str]:
    """Get node with highest degree in a component, optionally filtered by a function.
    Returns None if the component is empty or no node passes the filter."""
    max_degree = -1
    densest_node = None

    for node in component:
        if filter_func:
            node_data = graph.nodes[node].get("data")
            if not filter_func(node_data):
                continue

        degree = graph.in_degree(node) + graph.out_degree(node)
        if degree > max_degree:
            max_degree = degree
            densest_node = node

    return densest_node


def textualize_graph(
    component: Set,
    graph: Union[nx.DiGraph, nx.Graph],
    method: str = "dfs",
    annotate_steps: bool = False,
) -> Tuple[List[str], str]:
    """Convert a graph component to textual triples via DFS or BFS traversal.

    Triples are sorted by source_step ascending (foundational facts first,
    more specific later-step facts after). Triples without provenance sort last.
    """
    if not component:
        return [], ""

    start_node = get_densest_node(component, graph)
    if start_node is None:
        return [], ""

    traversers = {
        "dfs": _dfs_textualize,
        "bfs": _bfs_textualize,
    }

    traverser = traversers.get(method)
    if not traverser:
        raise ValueError(f"Unknown textualization method: {method}")

    raw: List[Tuple[Optional[int], str]] = traverser(graph, start_node, annotate_step=annotate_steps)

    # Sort by source_step ascending; None-step triples sort last. Stable sort preserves traversal order for ties.
    raw.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0))

    all_triples = [text for _, text in raw]

    cluster_text = "\n-----------------------\n".join(
        f"{i}. {triple}" for i, triple in enumerate(all_triples, 1)
    )
    return all_triples, f"Cluster Information:\n{cluster_text}"


def _format_node_description(node: str, graph: Union[nx.DiGraph, nx.Graph]) -> str:
    """Format a node into a readable description using duck typing."""
    node_data = graph.nodes[node].get("data")
    if not node_data:
        return str(node)
    if hasattr(node_data, "to_context"):
        return node_data.to_context(include_wiki_page=False)
    if hasattr(node_data, "label"):
        return node_data.label
    return str(node_data)


def _format_triple(
    source: str,
    target: str,
    graph: Union[nx.DiGraph, nx.Graph],
    edge_data: dict,
    annotate_step: bool = False,
) -> List[Tuple[Optional[int], str]]:
    """Format a graph edge into readable triple strings, paired with source_step for sorting."""
    source_data = graph.nodes[source].get("data")
    target_data = graph.nodes[target].get("data")
    relations = edge_data.get("relation", set())

    if not source_data or not target_data or not relations:
        return []

    step: Optional[int] = edge_data.get("provenance", {}).get("source_step")
    prefix = f"[Step {step}] " if (annotate_step and step is not None) else ""

    triples = []
    for relation in relations:
        rel_label = relation.label if hasattr(relation, "label") else str(relation)
        triple = f"{prefix}Subject: {source_data} - Relation: {rel_label} - Object: {target_data}"
        triples.append((step, triple))
    return triples


def _get_neighbors(
    node: str,
    graph: Union[nx.DiGraph, nx.Graph],
    visited: Set,
) -> List[Tuple[str, str, str]]:
    """Get unvisited neighbors (incoming and outgoing).

    Returns list of (source_node, target_node, edge_direction).
    """
    neighbors = []

    for neighbor in graph.successors(node):
        if neighbor not in visited:
            neighbors.append((node, neighbor, "out"))

    for neighbor in graph.predecessors(node):
        if neighbor not in visited:
            neighbors.append((neighbor, node, "in"))

    return neighbors


def _dfs_textualize(
    graph: Union[nx.DiGraph, nx.Graph],
    start_node: str,
    annotate_step: bool = False,
) -> List[Tuple[Optional[int], str]]:
    """DFS traversal to textualize graph."""
    visited = set()
    all_triples: List[Tuple[Optional[int], str]] = []
    stack = [(start_node, None, None)]

    while stack:
        current, edge_source, edge_target = stack.pop()

        if current in visited:
            continue
        visited.add(current)

        if edge_source is not None and edge_target is not None:
            edge_data = graph.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph, edge_data, annotate_step=annotate_step))

        for source, target, _ in reversed(_get_neighbors(current, graph, visited)):
            next_node = target if source == current else source
            if next_node not in visited:
                stack.append((next_node, source, target))

    if not all_triples and start_node:
        node_description = _format_node_description(start_node, graph)
        if node_description:
            all_triples.append((None, node_description))

    return all_triples


def _bfs_textualize(
    graph: Union[nx.DiGraph, nx.Graph],
    start_node: str,
    annotate_step: bool = False,
) -> List[Tuple[Optional[int], str]]:
    """BFS traversal to textualize graph."""
    visited = set()
    all_triples: List[Tuple[Optional[int], str]] = []
    queue = deque([(start_node, None, None)])

    while queue:
        current, edge_source, edge_target = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        if edge_source is not None and edge_target is not None:
            edge_data = graph.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph, edge_data, annotate_step=annotate_step))

        for source, target, _ in _get_neighbors(current, graph, visited):
            next_node = target if source == current else source
            if next_node not in visited:
                queue.append((next_node, source, target))

    if not all_triples and start_node:
        node_description = _format_node_description(start_node, graph)
        if node_description:
            all_triples.append((None, node_description))

    return all_triples


def _normalize_relation_labels(relation: Any) -> List[str]:
    """Return relation labels as a sorted list of strings."""
    if relation is None:
        return []
    if isinstance(relation, (set, list, tuple)):
        labels = [
            item.label if hasattr(item, "label") else str(item)
            for item in relation
            if item is not None
        ]
        return sorted([label for label in labels if label])
    if hasattr(relation, "label"):
        return [str(relation.label)]
    return [str(relation)]


def _get_graph_output_filepath(base_path: str, fig_title: str, extension: str) -> str:
    """Resolve output file path for graph visualizations."""
    import time

    safe_title = "".join(c for c in fig_title if c.isalnum() or c in (" ", "-", "_")).rstrip()
    safe_title = safe_title.replace(" ", "_").lower() or "graph"
    timestamp = int(time.time())
    filename = f"graph_{safe_title}_{timestamp}.{extension.lstrip('.')}"

    base_path = str(base_path).replace("\\", "/")

    if base_path.endswith("/"):
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, filename)
    if os.path.isdir(base_path):
        return os.path.join(base_path, filename)
    if os.path.dirname(base_path) and os.path.dirname(base_path) != ".":
        dir_path = os.path.dirname(base_path)
        os.makedirs(dir_path, exist_ok=True)
        if os.path.splitext(base_path)[1]:
            return base_path
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, filename)
    if base_path.endswith((".png", ".jpg", ".jpeg", ".pdf", ".html")):
        os.makedirs("./tmp", exist_ok=True)
        return os.path.join("./tmp", base_path)
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, filename)


def visualize_graph_interactive(
    graph: nx.DiGraph,
    title: str = "Graph Memory",
    save_path: Optional[Union[str, Path]] = None,
    notebook: bool = True,
    *,
    physics: bool = True,
    max_nodes: Optional[int] = None,
    neat_layout: bool = True,
) -> Optional[str]:
    """Visualize graph memory as interactive HTML (drag/zoom/node-size slider)."""
    if len(graph.nodes) == 0:
        print(f"{title}: Empty graph (no nodes)")
        return None

    try:
        from pyvis.network import Network
    except ImportError:
        print("Warning: pyvis is not available. Install pyvis to use interactive graph visualization.")
        return None

    selected_graph = graph
    if max_nodes is not None and max_nodes > 0 and len(graph.nodes) > max_nodes:
        ranked_nodes = sorted(
            graph.nodes(),
            key=lambda n: graph.in_degree(n) + graph.out_degree(n),
            reverse=True,
        )[:max_nodes]
        selected_graph = graph.subgraph(ranked_nodes).copy()

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=notebook,
        cdn_resources="in_line",
    )
    options = {
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "navigationButtons": True,
            "keyboard": {"enabled": True},
        },
        "physics": {
            "enabled": bool(physics),
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -90,
                "centralGravity": 0.01,
                "springLength": 140,
                "springConstant": 0.05,
                "damping": 0.75,
                "avoidOverlap": 1.0,
            },
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
    net.set_options(json.dumps(options))

    for node in selected_graph.nodes():
        node_data = selected_graph.nodes[node].get("data")
        node_id = str(node)
        if node_data is None:
            label = node_id[:40]
            title_text = node_id
        else:
            label = getattr(node_data, "label", None) or getattr(node_data, "name", None) or node_id
            title_text = str(node_data)
        degree = selected_graph.in_degree(node) + selected_graph.out_degree(node)
        size = max(14, min(60, 14 + int(degree * 2)))
        color = "#4e79a7" if degree >= 6 else "#76b7b2" if degree >= 3 else "#bab0ab"
        net.add_node(node_id, label=str(label)[:45], title=title_text, size=size, color=color)

    for source, target, data in selected_graph.edges(data=True):
        labels = _normalize_relation_labels(data.get("relation"))
        edge_label = ", ".join(labels[:2]) if labels else ""
        edge_title = "<br>".join(labels) if labels else ""
        net.add_edge(str(source), str(target), label=edge_label, title=edge_title)

    output_path = (
        _get_graph_output_filepath(str(save_path), title, "html")
        if save_path is not None
        else _get_graph_output_filepath("./tmp", title, "html")
    )
    net.save_graph(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    inject = """
<div style="margin: 8px 0; font-family: sans-serif;">
  <label for="nodeSizeScale"><strong>Node size</strong></label>
  <input id="nodeSizeScale" type="range" min="0.5" max="3" step="0.1" value="1.0" style="width: 260px; margin-left: 8px;" />
  <span id="nodeSizeValue">1.0x</span>
  <button id="freezeLayoutBtn" style="margin-left: 14px;">Freeze layout</button>
  <button id="relaxLayoutBtn" style="margin-left: 6px;">Relax layout</button>
  <span style="margin-left: 10px; color: #666;">Tip: drag nodes, then freeze</span>
</div>
<script>
(function() {
  const slider = document.getElementById("nodeSizeScale");
  const valueEl = document.getElementById("nodeSizeValue");
  const freezeBtn = document.getElementById("freezeLayoutBtn");
  const relaxBtn = document.getElementById("relaxLayoutBtn");
  if (!slider || !window.nodes) return;
  const baseSizes = {};
  window.nodes.forEach(function(node) {
    baseSizes[node.id] = node.size || 20;
  });
  function applyScale(scale) {
    const updates = [];
    window.nodes.forEach(function(node) {
      const base = baseSizes[node.id] || 20;
      updates.push({id: node.id, size: Math.max(8, base * scale)});
    });
    window.nodes.update(updates);
    valueEl.textContent = scale.toFixed(1) + "x";
  }
  slider.addEventListener("input", function() {
    applyScale(parseFloat(slider.value));
  });
  applyScale(parseFloat(slider.value));

  if (window.network) {
    // Let physics settle once, then auto-freeze so labels stay readable.
    window.network.once("stabilizationIterationsDone", function() {
      window.network.setOptions({physics: {enabled: false}});
      if (freezeBtn) freezeBtn.textContent = "Layout frozen";
    });

    if (freezeBtn) {
      freezeBtn.addEventListener("click", function() {
        window.network.setOptions({physics: {enabled: false}});
        freezeBtn.textContent = "Layout frozen";
      });
    }
    if (relaxBtn) {
      relaxBtn.addEventListener("click", function() {
        window.network.setOptions({physics: {enabled: true}});
        window.network.stabilize(200);
        if (freezeBtn) freezeBtn.textContent = "Freeze layout";
      });
    }
  }
})();
</script>
"""
    marker = '<div id="mynetwork" class="card-body"></div>'
    if marker in html:
        html = html.replace(marker, inject + "\n" + marker, 1)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    print(f"Interactive graph visualization saved to {os.path.abspath(output_path)}")
    return output_path


def visualize_graph(
    graph: nx.DiGraph,
    title: str = "Graph Memory",
    save_path: Optional[str] = "./tmp",
):
    """Visualize a networkx DiGraph using matplotlib."""
    use_agg = os.getenv("DISPLAY") is None and os.name != "nt"

    try:
        import matplotlib
        if use_agg:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Warning: matplotlib not available. Cannot visualize {title}")
        return
    except Exception as e:
        print(f"Warning: Could not initialize matplotlib: {e}")
        return

    if len(graph.nodes) == 0:
        print(f"{title}: Empty graph (no nodes)")
        return

    plt.figure(figsize=(12, 8))

    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except Exception:
        pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=1000, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.6)

    labels = {}
    for node in graph.nodes():
        node_data = graph.nodes[node].get("data", None)
        if node_data:
            if hasattr(node_data, "label"):
                labels[node] = node_data.label
            elif hasattr(node_data, "name"):
                labels[node] = node_data.name
            else:
                labels[node] = str(node)[:20]
        else:
            labels[node] = str(node)[:20]

    nx.draw_networkx_labels(graph, pos, labels, font_size=8)

    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        labels = _normalize_relation_labels(data.get("relation"))
        edge_labels[(u, v)] = ", ".join(labels[:2])

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        filepath = _get_graph_output_filepath(save_path, title, "png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Graph visualization saved to {os.path.abspath(filepath)}")
    else:
        if use_agg:
            filepath = _get_graph_output_filepath("./tmp", title, "png")
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Graph visualization saved to {os.path.abspath(filepath)} (no display available)")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display graph interactively: {e}")
                filepath = _get_graph_output_filepath("./tmp", title, "png")
                plt.savefig(filepath, dpi=150, bbox_inches="tight")
                print(f"Graph visualization saved to {os.path.abspath(filepath)} (fallback)")

    plt.close()
