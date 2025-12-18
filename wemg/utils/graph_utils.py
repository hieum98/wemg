from typing import Any, List, Tuple, Union
import networkx as nx

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles


def get_densest_node(component: set, graph_memory: Union[nx.DiGraph, nx.Graph], node_type: Any=None) -> str:
    """Get the node with highest degree (in + out) in a component of the graph memory that optionally matches a specific type."""
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
        component: set, 
        graph_memory: Union[nx.DiGraph, nx.Graph],
        method: str = 'dfs'
        ) -> Tuple[List[str], str]:
    """
    Given a weakly connected component of the graph memory, return all textual triples in the component in the order of traversal.
    """
    if not component:
        return [], ""
    
    # Start from the densest node
    start_node = get_densest_node(component, graph_memory)
    if start_node is None:
        return [], ""
    
    # Choose traversal method
    if method == 'dfs':
        all_triples = _dfs_textualize(component, graph_memory, start_node)
    elif method == 'bfs':
        all_triples = _bfs_textualize(component, graph_memory, start_node)
    elif method == 'semantic-dfs':
        # TODO: implement semantic DFS  
        raise NotImplementedError("Semantic DFS not implemented yet.") 
    else:
        raise ValueError(f"Unknown textualization method: {method}")

    cluster_text = [f"{i}. {triple}" for i, triple in enumerate(all_triples, start=1)]
    cluster_text = "\n-----------------------\n".join(cluster_text)
    cluster_text = f"Cluster Information:\n{cluster_text}"
    return all_triples, cluster_text


def _format_triple(
        source_node, 
        target_node, 
        graph_memory: Union[nx.DiGraph, nx.Graph], 
        edge_data: dict
    ) -> str:
    """Format a triple (source, relation, target) into a natural language sentence."""
    assert 'relation' in edge_data, "Edge data must contain 'relation' key"
    assert 'data' in graph_memory.nodes[source_node], "Source node must have 'data'"
    assert 'data' in graph_memory.nodes[target_node], "Target node must have 'data'"

    source_data = graph_memory.nodes[source_node].get('data')
    target_data = graph_memory.nodes[target_node].get('data')
    relations = edge_data.get('relation', set())
    
    assert (not source_data) and (not target_data) and (not relations), "Edge and node data must present"

    all_triples = []
    for relation in relations:
        # Convert to strings
        triple = roles.open_ie.Relation(
            subject=str(source_data),
            relation=relation,
            object=str(target_data),
            evidence=None 
        )
        all_triples.append(str(triple))
    return all_triples


def _get_neighbors(node, graph_memory: Union[nx.DiGraph, nx.Graph], visited: set) -> list:
    """Get all unvisited neighbors (both incoming and outgoing) of a node.
    
    Returns list of tuples: (source_node, target_node, edge_direction)
    where edge_direction is 'out' for outgoing edges and 'in' for incoming edges.
    The source and target are ordered according to the actual edge direction in the graph.
    """
    neighbors = []
    
    # Get outgoing neighbors (current -> neighbor)
    for neighbor in graph_memory.successors(node):
        if neighbor not in visited:
            neighbors.append((node, neighbor, 'out'))
    
    # Get incoming neighbors (neighbor -> current) 
    # We can traverse backward, but we preserve the edge direction for textualization
    for neighbor in graph_memory.predecessors(node):
        if neighbor not in visited:
            neighbors.append((neighbor, node, 'in'))
    
    return neighbors


def _dfs_textualize(graph_memory: Union[nx.DiGraph, nx.Graph], start_node: str) -> List[str]:
    """Perform depth-first search traversal to textualize the graph."""
    visited = set()
    all_triples = []
    stack = [(start_node, None, None)]  # (node, edge_source, edge_target)
    
    while stack:
        current, edge_source, edge_target = stack.pop()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Add sentence for the edge that led to this node (if any)
        if edge_source is not None and edge_target is not None:
            edge_data = graph_memory.edges[edge_source, edge_target]
            triples = _format_triple(edge_source, edge_target, graph_memory, edge_data)
            all_triples.extend(triples)
        
        # Get all neighbors (both directions)
        neighbors = _get_neighbors(current, graph_memory, visited)
        
        # Add neighbors to stack (reverse order for DFS to maintain natural order)
        for source, target, direction in reversed(neighbors):
            # Determine which node to visit next (the one that's not current)
            next_node = target if source == current else source
            if next_node not in visited:
                stack.append((next_node, source, target))
    return all_triples


def _bfs_textualize(graph_memory: Union[nx.DiGraph, nx.Graph], start_node: str) -> List[str]:
    """Perform breadth-first search traversal to textualize the graph."""
    from collections import deque
    
    visited = set()
    all_triples = []
    queue = deque([(start_node, None, None)])  # (node, edge_source, edge_target)
    
    while queue:
        current, edge_source, edge_target = queue.popleft()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Add sentence for the edge that led to this node (if any)
        if edge_source is not None and edge_target is not None:
            edge_data = graph_memory.edges[edge_source, edge_target]
            triples = _format_triple(edge_source, edge_target, graph_memory, edge_data)
            all_triples.extend(triples)
        
        # Get all neighbors (both directions)
        neighbors = _get_neighbors(current, graph_memory, visited)
        
        # Add neighbors to queue
        for source, target, direction in neighbors:
            # Determine which node to visit next (the one that's not current)
            next_node = target if source == current else source
            if next_node not in visited:
                queue.append((next_node, source, target))
    return all_triples


