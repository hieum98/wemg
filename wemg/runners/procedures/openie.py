import logging
import os
from typing import List, Optional
import networkx as nx
import asyncio

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.utils.preprocessing import get_node_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


def parse_graph_from_text(llm_agent: BaseLLMAgent, text: str, interaction_memory: Optional[InteractionMemory] = None):
    """Parse a graph (entities and relations) from the given text using OpenIE."""
    
    re_input = roles.open_ie.RelationExtractionInput(text=text)
    triples, re_log = asyncio.run(
        execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=re_input,
            interaction_memory=interaction_memory,
            n=1
        )
    )
    relation_triples = triples[0][0] # bs=1, n=1
    if relation_triples and isinstance(relation_triples, roles.open_ie.RelationExtractionOutput):
        relation_triples = relation_triples.relations
    else:
        logger.error(f"Failed to extract relations from text. Got {type(relation_triples)}")
        relation_triples = []
    
    graph = nx.DiGraph()
    for triple in relation_triples:
        subject_id = get_node_id(triple.subject)
        object_id = get_node_id(triple.object)
        if not graph.has_node(subject_id):
            graph.add_node(subject_id, data=triple.subject)
        if not graph.has_node(object_id):
            graph.add_node(object_id, data=triple.object)
        if not graph.has_edge(subject_id, object_id):
            graph.add_edge(subject_id, object_id, relation=set())
        graph.edges[subject_id, object_id]['relation'].add(triple.relation)
    return graph, re_log
