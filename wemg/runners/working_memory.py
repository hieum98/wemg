from collections import defaultdict
from enum import Enum
import logging
import os
from typing import Any, Dict, List, Tuple, Union, Set
import networkx as nx

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools import wikidata
from wemg.agents.tools.web_search import WebSearchTool
from wemg.runners.procerduces.extraction import extract_relations_from_text
from wemg.utils.graph_utils import get_densest_node
from wemg.utils.preprocessing import get_node_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

class Memory:
    def __init__(
            self,
            textual_memory: List[str] = [],
            graph_memory: Union[nx.DiGraph, None] = None,
            parsed_graph_memory: Union[nx.DiGraph, None] = None,
            max_textual_memory_tokens: int = 8192,
    ):  
        self.entity_dict: Dict[roles.open_ie.Entity, wikidata.WikidataEntity] = {} # mapping from open_ie.Entity to wikidata.WikidataEntity
        self.property_dict: Dict[str, wikidata.WikidataProperty] = {} # mapping from property name to wikidata.WikidataProperty
        self.id_dict: Dict[str, wikidata.WikidataEntity] = {} # mapping from node_id to wikidata.WikidataEntity
        self.textual_memory: List[str] = textual_memory
        if graph_memory is None:
            self.graph_memory: nx.DiGraph = nx.DiGraph()
            self.parsed_graph_memory: nx.DiGraph = nx.DiGraph()
        else:
            self.graph_memory: nx.DiGraph = graph_memory
            self.parsed_graph_memory: nx.DiGraph = parsed_graph_memory

        self.max_textual_memory_tokens = max_textual_memory_tokens
    
    def add_textual_memory(self, text: str, source: roles.extractor.SourceType=roles.extractor.SourceType.SYSTEM_PREDICTION):
        """Add text to textual memory if not already present."""
        text = text.strip()
        if source == roles.extractor.SourceType.SYSTEM_PREDICTION:
            text = f"[System Prediction]: {text}"
        elif source == roles.extractor.SourceType.RETRIEVAL:
            text = f"[Retrieval]: {text}"
        if text not in self.textual_memory:
            self.textual_memory.append(text)

    def format_textual_memory(self) -> str:
        """Format the textual memory as a single string."""
        memory = [f"- {text.strip()}" for text in self.textual_memory]
        return "\n".join(memory)

    def consolidate_textual_memory(
            self, 
            llm_agent: BaseLLMAgent, 
            question: str, 
            in_context_examples: List[Dict[str, str]] = None,
            ):
        """Consolidate the textual memory with respect to the question."""
        memory_consolidation_role = roles.extractor.MemoryConsolidationRole()
        memory_consolidation_input = roles.extractor.MemoryConsolidationInput(
            question=question,
            memory=self.format_textual_memory()
        )
        memory_consolidation_messages = memory_consolidation_role.format_messages(input_data=memory_consolidation_input, history=in_context_examples)
        consolidation_kwargs = {
            'output_schema': roles.extractor.MemoryConsolidationOutput,
            'n': 1,
            'max_tokens': self.max_textual_memory_tokens
        }
        consolidation_response = llm_agent.generator_role_execute(memory_consolidation_messages, **consolidation_kwargs)[0][0]
        consolidation_output: roles.extractor.MemoryConsolidationOutput = memory_consolidation_role.parse_response(consolidation_response)
        for item in consolidation_output.consolidated_memory:
            if item.provenance not in [roles.extractor.SourceType.SYSTEM_PREDICTION.value, roles.extractor.SourceType.RETRIEVAL.value]:
                self.add_textual_memory(item.content, source=roles.extractor.SourceType.SYSTEM_PREDICTION)
            else:
                self.add_textual_memory(item.content, source=item.provenance)
    
    def consolidate_textual_memory_rerank(self):
        pass

    def remove_textual_memory(self):
        pass
    
    def update_textual_memory(self):
        pass

    def parse_graph_memory_from_textual_memory(self, llm_agent: BaseLLMAgent):
        textual_memory = self.format_textual_memory()
        relation_triples = extract_relations_from_text(llm_agent=llm_agent, text=textual_memory)
        for triple in relation_triples:
            subject_id = get_node_id(triple.subject)
            object_id = get_node_id(triple.object)
            if not self.parsed_graph_memory.has_node(subject_id):
                self.parsed_graph_memory.add_node(subject_id, data=triple.subject)
            if not self.parsed_graph_memory.has_node(object_id):
                self.parsed_graph_memory.add_node(object_id, data=triple.object)
            if not self.parsed_graph_memory.has_edge(subject_id, object_id):
                self.parsed_graph_memory.add_edge(
                    subject_id,
                    object_id,
                    relation={triple.relation}
                    )
            else:
                # append relation to the existing set
                self.parsed_graph_memory.edges[subject_id, object_id]['relation'].add(triple.relation)

    def add_edge_to_graph_memory(self, wiki_triple: wikidata.WikiTriple):
        subject_id = get_node_id(wiki_triple.subject)
        object_id = get_node_id(wiki_triple.object)
        if not self.graph_memory.has_node(subject_id):
            self.graph_memory.add_node(subject_id, data=wiki_triple.subject)
        if not self.graph_memory.has_node(object_id):
            self.graph_memory.add_node(object_id, data=wiki_triple.object)
        if not self.graph_memory.has_edge(subject_id, object_id):
            self.graph_memory.add_edge(
                subject_id,
                object_id,
                relation={wiki_triple.relation}
                )
        else:
            # append relation to the existing set
            self.graph_memory.edges[subject_id, object_id]['relation'].add(wiki_triple.relation)
    
    def connect_graph_memory(
            self,
            wikidata_path_finder: "wikidata.WikidataPathFindingTool" = None,
            max_hops: int = 3,
    ) -> bool:
        """Check for connectivity in the graph memory and try to connect disconnected components.
        
        This method:
        1. Checks if the graph is connected
        2. If not, identifies all weakly connected components
        3. Finds the densest node (highest degree) in each component
        4. Attempts to find paths between densest nodes using Wikidata
        5. Adds the connecting paths to the graph
        Returns:
            bool: True if the graph is now connected, False otherwise.
        """
        # Check if graph is empty or has only one node
        if self.graph_memory.number_of_nodes() <= 1:
            return True
        
        # Get weakly connected components (for directed graph)
        components = list(nx.weakly_connected_components(self.graph_memory))
        
        if len(components) <= 1:
            logger.info("Graph memory is already connected")
            return True
        
        logger.info(f"Found {len(components)} disconnected components in graph memory")
        
        # Initialize path finder if not provided
        if wikidata_path_finder is None:
            wikidata_path_finder = wikidata.WikidataPathFindingTool()
        
        # Get densest nodes for all components
        densest_nodes = [get_densest_node(comp, self.graph_memory, node_type=wikidata.WikidataEntity) for comp in components]
        logger.info(f"Densest nodes in each component: {densest_nodes}")
        
        # Try to connect components by finding paths between their densest nodes
        for i in range(len(densest_nodes) - 1):
            source_node = densest_nodes[i]
            target_node = densest_nodes[i + 1]
            
            # Get entity data from graph nodes
            source_data = self.graph_memory.nodes[source_node].get('data')
            target_data = self.graph_memory.nodes[target_node].get('data')
            
            # Determine QIDs
            source_qid = None
            target_qid = None
            
            if isinstance(source_data, wikidata.WikidataEntity):
                source_qid = source_data.qid
            if isinstance(target_data, wikidata.WikidataEntity):
                target_qid = target_data.qid
            
            if not source_qid or not target_qid:
                logger.warning(f"Cannot find path: source ({source_node}) or target ({target_node}) is not a Wikidata entity")
                continue
            
            logger.info(f"Searching for path between {source_qid} and {target_qid}")
            
            # Find path between entities
            path_result: wikidata.WikidataPathBetweenEntities= wikidata_path_finder.invoke(
                {
                    "source_qid": source_qid,
                    "target_qid": target_qid,
                    "max_hops": max_hops,
                }
            )
            
            if path_result and path_result.path:
                logger.info(f"Found path with {len(path_result.path)} hops: {path_result}")
                # Add the path triples to the graph
                for triple in path_result.path:
                    self.add_edge_to_graph_memory(triple)
            else:
                logger.warning(f"No path found between {source_qid} and {target_qid}")
        
        # Check if graph is now connected
        new_components = list(nx.weakly_connected_components(self.graph_memory))
        is_connected = len(new_components) == 1
        
        if is_connected:
            logger.info("Graph memory is now fully connected")
        else:
            logger.info(f"Graph memory still has {len(new_components)} disconnected components")
        
        return is_connected

    def update_graph_memory(self):
        # Update all nodes with full WikidataEntity details if missing
        id_to_fetch: Set[str] = set()
        for node_id in self.graph_memory.nodes:
            node_data = self.graph_memory.nodes[node_id].get('data')
            if isinstance(node_data, wikidata.WikidataEntity):
                if not node_data.description or not node_data.wikipedia_content:
                    if node_data.qid in self.id_dict:
                        full_entity = self.id_dict[node_data.qid]
                        self.graph_memory.nodes[node_id]['data'] = full_entity
                    else:
                        id_to_fetch.add(node_data.qid)
        if id_to_fetch:
            logger.info(f"Fetching details for {len(id_to_fetch)} Wikidata entities to update graph memory.")
            entity_retriever = wikidata.WikidataEntityRetrievalTool()
            retrieval_results = entity_retriever.invoke(
                {
                    "query": list(id_to_fetch),
                    "num_entities": 1,
                    "get_details": True,
                }
            )
            all_fetched_entities: List[wikidata.WikidataEntity] = sum(retrieval_results, [])
            for ent in all_fetched_entities:
                self.id_dict[ent.qid] = ent
                node_id = get_node_id(ent)
                if self.graph_memory.has_node(node_id):
                    self.graph_memory.nodes[node_id]['data'] = ent
                    logger.info(f"Updated node {node_id} with full entity details.")

    def consolidate_graph_memory(self):
        pass

    def synchronize_memories(self):
        relation_triples = []
        for subject_id, object_id, edge_data in self.parsed_graph_memory.edges(data=True):
            subject_entity = self.parsed_graph_memory.nodes[subject_id].get('data')
            object_entity = self.parsed_graph_memory.nodes[object_id].get('data')
            for relation_name in edge_data.get('relation', []):
                relation_triples.append(
                    roles.open_ie.Relation(
                        subject=subject_entity,
                        relation=relation_name,
                        object=object_entity,
                    )
                )
        # Identify entities and properties to fetch from Wikidata
        to_fetch_entities: Set[roles.open_ie.Entity] = set()
        to_fetch_properties: Set[str] = set()
        for triple in relation_triples:
            if triple.subject not in self.entity_dict:
                to_fetch_entities.add(triple.subject)
            if triple.object not in self.entity_dict:
                to_fetch_entities.add(triple.object)
            if triple.relation not in self.property_dict:
                to_fetch_properties.add(triple.relation)
        # Fetch entities and properties from Wikidata
        if to_fetch_entities:
            logger.info(f"Fetching {len(to_fetch_entities)} entities from Wikidata for synchronization.")
            entity_retriever = wikidata.WikidataEntityRetrievalTool()
            retrieval_results = entity_retriever.invoke(
                {
                    "query": [ent.name for ent in to_fetch_entities],
                    "num_entities": 1,
                    "get_details": True,
                }
            )
            for item, result in zip(to_fetch_entities, retrieval_results):
                if result and isinstance(result[0], wikidata.WikidataEntity): # take the top-1 entity
                    self.entity_dict[item] = result[0]
                    logger.info(f"Synchronized entity {item.name} to Wikidata entity {result[0].qid}")
        
        if to_fetch_properties:
            logger.info(f"Fetching {len(to_fetch_properties)} properties from Wikidata for synchronization.")
            property_retriever = wikidata.WikidataPropertyRetrievalTool()
            retrieval_results = property_retriever.invoke(
                {
                    "query": list(to_fetch_properties),
                    "num_properties": 1,
                }
            )
            for prop_name, result in zip(to_fetch_properties, retrieval_results):
                if result and isinstance(result[0], wikidata.WikidataProperty): # take the top-1 property
                    self.property_dict[prop_name] = result[0]
                    logger.info(f"Synchronized property {prop_name} to Wikidata property {result[0].pid}")
        
        # Convert relation triples to WikiTriples and add to graph memory
        for triple in relation_triples:
            subject_entity = self.entity_dict.get(triple.subject)
            object_entity = self.entity_dict.get(triple.object)
            property_entity = self.property_dict.get(triple.relation)
            if subject_entity and object_entity and property_entity:
                wiki_triple = wikidata.WikiTriple(
                    subject=subject_entity,
                    relation=property_entity,
                    object=object_entity
                )
                self.add_edge_to_graph_memory(wiki_triple)
                logger.info(f"Added triple to graph memory: {wiki_triple}")
        # Make graph memory connected
        self.connect_graph_memory()
        self.update_graph_memory() # ensure all nodes have full details, mostly for newly added nodes via connectivity

        # Sync graph memory to textual memory

    
    

