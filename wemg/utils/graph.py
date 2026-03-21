"""Graph utilities for textualization and visualization."""

import os
from collections import deque
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import networkx as nx


def get_node_id(entity: Any) -> str:
    """Get a string ID for an entity. Uses .qid for WikidataEntity, .id for Entity, else str()."""
    if hasattr(entity, "qid"):
        return entity.qid
    elif hasattr(entity, "id"):
        return str(entity.id) if entity.id else str(entity)
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
) -> Tuple[List[str], str]:
    """Convert a graph component to textual triples via DFS or BFS traversal."""
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

    all_triples = traverser(graph, start_node)

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
) -> List[str]:
    """Format a graph edge into readable triple strings."""
    source_data = graph.nodes[source].get("data")
    target_data = graph.nodes[target].get("data")
    relations = edge_data.get("relation", set())

    if not source_data or not target_data or not relations:
        return []

    triples = []
    for relation in relations:
        rel_label = relation.label if hasattr(relation, "label") else str(relation)
        triple = f"Subject: {source_data}\nRelation: {rel_label}\nObject: {target_data}"
        triples.append(triple)
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
) -> List[str]:
    """DFS traversal to textualize graph."""
    visited = set()
    all_triples = []
    stack = [(start_node, None, None)]

    while stack:
        current, edge_source, edge_target = stack.pop()

        if current in visited:
            continue
        visited.add(current)

        if edge_source is not None and edge_target is not None:
            edge_data = graph.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph, edge_data))

        for source, target, _ in reversed(_get_neighbors(current, graph, visited)):
            next_node = target if source == current else source
            if next_node not in visited:
                stack.append((next_node, source, target))

    if not all_triples and start_node:
        node_description = _format_node_description(start_node, graph)
        if node_description:
            all_triples.append(node_description)

    return all_triples


def _bfs_textualize(
    graph: Union[nx.DiGraph, nx.Graph],
    start_node: str,
) -> List[str]:
    """BFS traversal to textualize graph."""
    visited = set()
    all_triples = []
    queue = deque([(start_node, None, None)])

    while queue:
        current, edge_source, edge_target = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        if edge_source is not None and edge_target is not None:
            edge_data = graph.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph, edge_data))

        for source, target, _ in _get_neighbors(current, graph, visited):
            next_node = target if source == current else source
            if next_node not in visited:
                queue.append((next_node, source, target))

    if not all_triples and start_node:
        node_description = _format_node_description(start_node, graph)
        if node_description:
            all_triples.append(node_description)

    return all_triples


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
        relation = data.get("relation", {})
        if isinstance(relation, set):
            rel_labels = []
            for prop in relation:
                if hasattr(prop, "label"):
                    rel_labels.append(prop.label)
                else:
                    rel_labels.append(str(prop))
            edge_labels[(u, v)] = ", ".join(rel_labels[:2])
        elif hasattr(relation, "label"):
            edge_labels[(u, v)] = relation.label
        else:
            edge_labels[(u, v)] = str(relation)[:15]

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    def _get_filepath(base_path: str, fig_title: str) -> str:
        import time
        safe_title = "".join(c for c in fig_title if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_title = safe_title.replace(" ", "_").lower()
        timestamp = int(time.time())
        filename = f"graph_{safe_title}_{timestamp}.png"

        base_path = base_path.replace("\\", "/")

        if base_path.endswith("/"):
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, filename)
        elif os.path.isdir(base_path):
            return os.path.join(base_path, filename)
        elif os.path.dirname(base_path) and os.path.dirname(base_path) != ".":
            dir_path = os.path.dirname(base_path)
            os.makedirs(dir_path, exist_ok=True)
            return base_path
        elif base_path.endswith((".png", ".jpg", ".pdf")):
            os.makedirs("./tmp", exist_ok=True)
            return os.path.join("./tmp", base_path)
        else:
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, filename)

    if save_path:
        filepath = _get_filepath(save_path, title)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Graph visualization saved to {os.path.abspath(filepath)}")
    else:
        if use_agg:
            filepath = _get_filepath("./tmp", title)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Graph visualization saved to {os.path.abspath(filepath)} (no display available)")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display graph interactively: {e}")
                filepath = _get_filepath("./tmp", title)
                plt.savefig(filepath, dpi=150, bbox_inches="tight")
                print(f"Graph visualization saved to {os.path.abspath(filepath)} (fallback)")

    plt.close()
