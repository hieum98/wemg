"""Graph utility functions for working memory operations."""
from collections import deque
from typing import Any, List, Set, Tuple, Union

import networkx as nx

from wemg.agents import roles


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
