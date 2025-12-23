"""Graph utility functions for working memory operations."""
import os
from collections import deque
from typing import Any, List, Optional, Set, Tuple, Union

import networkx as nx


def get_densest_node(
    component: Set, 
    graph_memory: Union[nx.DiGraph, nx.Graph], 
    node_type: Any = None
) -> str:
    """Get the node with highest degree in a component that optionally matches a type."""
    max_degree = -1
    densest_node = None
    
    for node in component:
        if node_type:
            node_data = graph_memory.nodes[node].get('data')
            if not isinstance(node_data, node_type):
                continue
        
        degree = graph_memory.in_degree(node) + graph_memory.out_degree(node)
        if degree > max_degree:
            max_degree = degree
            densest_node = node
    
    return densest_node


def textualize_graph(
    component: Set, 
    graph_memory: Union[nx.DiGraph, nx.Graph],
    method: str = 'dfs'
) -> Tuple[List[str], str]:
    """Convert a graph component to textual triples in traversal order."""
    if not component:
        return [], ""
    
    start_node = get_densest_node(component, graph_memory)
    if start_node is None:
        return [], ""
    
    # Choose traversal method
    traversers = {
        'dfs': _dfs_textualize,
        'bfs': _bfs_textualize,
    }
    
    traverser = traversers.get(method)
    if not traverser:
        raise ValueError(f"Unknown textualization method: {method}")
    
    all_triples = traverser(graph_memory, start_node)
    
    cluster_text = "\n-----------------------\n".join(
        f"{i}. {triple}" for i, triple in enumerate(all_triples, 1)
    )
    return all_triples, f"Cluster Information:\n{cluster_text}"


def _format_triple(
    source_node: str, 
    target_node: str, 
    graph_memory: Union[nx.DiGraph, nx.Graph], 
    edge_data: dict
) -> List[str]:
    """Format a triple into natural language sentences."""
    # Lazy import to avoid circular dependencies
    from wemg.agents import roles
    
    source_data = graph_memory.nodes[source_node].get('data')
    target_data = graph_memory.nodes[target_node].get('data')
    relations = edge_data.get('relation', set())
    
    if not source_data or not target_data or not relations:
        return []
    
    triples = []
    for relation in relations:
        triple = roles.open_ie.Relation(
            subject=str(source_data),
            relation=str(relation),
            object=str(target_data),
            evidence=None 
        )
        triples.append(str(triple))
    return triples


def _get_neighbors(
    node: str, 
    graph_memory: Union[nx.DiGraph, nx.Graph], 
    visited: Set
) -> List[Tuple[str, str, str]]:
    """Get unvisited neighbors (incoming and outgoing).
    
    Returns list of (source_node, target_node, edge_direction).
    """
    neighbors = []
    
    # Outgoing: current -> neighbor
    for neighbor in graph_memory.successors(node):
        if neighbor not in visited:
            neighbors.append((node, neighbor, 'out'))
    
    # Incoming: neighbor -> current
    for neighbor in graph_memory.predecessors(node):
        if neighbor not in visited:
            neighbors.append((neighbor, node, 'in'))
    
    return neighbors


def _dfs_textualize(
    graph_memory: Union[nx.DiGraph, nx.Graph], 
    start_node: str
) -> List[str]:
    """DFS traversal to textualize graph."""
    visited = set()
    all_triples = []
    stack = [(start_node, None, None)]  # (node, edge_source, edge_target)
    
    while stack:
        current, edge_source, edge_target = stack.pop()
        
        if current in visited:
            continue
        visited.add(current)
        
        # Add triples for the edge that led here
        if edge_source is not None and edge_target is not None:
            edge_data = graph_memory.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph_memory, edge_data))
        
        # Get neighbors and add to stack (reversed for natural order)
        for source, target, _ in reversed(_get_neighbors(current, graph_memory, visited)):
            next_node = target if source == current else source
            if next_node not in visited:
                stack.append((next_node, source, target))
    
    return all_triples


def _bfs_textualize(
    graph_memory: Union[nx.DiGraph, nx.Graph], 
    start_node: str
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
        
        # Add triples for the edge that led here
        if edge_source is not None and edge_target is not None:
            edge_data = graph_memory.edges[edge_source, edge_target]
            all_triples.extend(_format_triple(edge_source, edge_target, graph_memory, edge_data))
        
        # Get neighbors and add to queue
        for source, target, _ in _get_neighbors(current, graph_memory, visited):
            next_node = target if source == current else source
            if next_node not in visited:
                queue.append((next_node, source, target))
    
    return all_triples


def visualize_graph(graph: nx.DiGraph, title: str = "Graph Memory", save_path: Optional[str] = './tmp'):
    """Visualize a networkx DiGraph using matplotlib.
    
    Args:
        graph: The networkx DiGraph to visualize
        title: Title for the plot
        save_path: Path to save the figure. Can be a directory (will create filename) 
                   or a full file path. If None, tries to display interactively, 
                   or saves to './tmp' if display is not available.
    """
    # Check if we need to use non-interactive backend (no display available)
    use_agg = os.getenv('DISPLAY') is None and os.name != 'nt'
    
    try:
        import matplotlib
        # Use non-interactive backend if no display is available (must be set before pyplot import)
        if use_agg:
            matplotlib.use('Agg')
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
    
    # Use a layout algorithm
    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except:
        # Fallback to simple layout if spring_layout fails
        pos = nx.circular_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, alpha=0.6)
    
    # Draw labels
    labels = {}
    for node in graph.nodes():
        node_data = graph.nodes[node].get('data', None)
        if node_data:
            # Try to get a label from the data object
            if hasattr(node_data, 'label'):
                labels[node] = node_data.label
            elif hasattr(node_data, 'name'):
                labels[node] = node_data.name
            else:
                labels[node] = str(node)[:20]  # Truncate long node IDs
        else:
            labels[node] = str(node)[:20]
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    # Draw edge labels (relations)
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        relation = data.get('relation', {})
        if isinstance(relation, set):
            # If relation is a set of properties, get labels
            rel_labels = []
            for prop in relation:
                if hasattr(prop, 'label'):
                    rel_labels.append(prop.label)
                else:
                    rel_labels.append(str(prop))
            edge_labels[(u, v)] = ', '.join(rel_labels[:2])  # Limit to 2 relations
        elif hasattr(relation, 'label'):
            edge_labels[(u, v)] = relation.label
        else:
            edge_labels[(u, v)] = str(relation)[:15]
    
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Helper function to generate file path
    def _get_filepath(base_path: str, title: str) -> str:
        """Generate a file path, creating directory if needed."""
        import time
        # Create a safe filename from the title
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_').lower()
        timestamp = int(time.time())
        filename = f"graph_{safe_title}_{timestamp}.png"
        
        # Normalize path separators
        base_path = base_path.replace('\\', '/')
        
        # Check if base_path is a directory or a file path
        if base_path.endswith('/'):
            # It's explicitly a directory (ends with /)
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, filename)
        elif os.path.isdir(base_path):
            # It's an existing directory
            return os.path.join(base_path, filename)
        elif os.path.dirname(base_path) and os.path.dirname(base_path) != '.':
            # It's a file path with directory component
            dir_path = os.path.dirname(base_path)
            os.makedirs(dir_path, exist_ok=True)
            return base_path
        elif base_path.endswith('.png') or base_path.endswith('.jpg') or base_path.endswith('.pdf'):
            # It's a filename with extension, save to ./tmp
            os.makedirs('./tmp', exist_ok=True)
            return os.path.join('./tmp', base_path)
        else:
            # Treat as directory name, create it and save file there
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, filename)
    
    # Determine if we should save or show
    if save_path:
        filepath = _get_filepath(save_path, title)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to {os.path.abspath(filepath)}")
    else:
        # If using Agg backend (no display), always save to file
        if use_agg:
            filepath = _get_filepath('./tmp', title)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Graph visualization saved to {os.path.abspath(filepath)} (no display available)")
        else:
            # Try to show interactively
            try:
                plt.show()
            except Exception as e:
                # If show() fails, save to a file instead
                print(f"Could not display graph interactively: {e}")
                filepath = _get_filepath('./tmp', title)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Graph visualization saved to {os.path.abspath(filepath)} (fallback)")
    
    plt.close()
