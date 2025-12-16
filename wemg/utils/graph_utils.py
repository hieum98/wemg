from typing import Any, Union
import networkx as nx


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


