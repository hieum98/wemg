from collections import defaultdict
import copy
from enum import Enum
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import networkx as nx
import asyncio

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools import wikidata
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.procerduces.base_role_excercution import execute_role
from wemg.runners.procerduces.openie import parse_graph_from_text
from wemg.utils.graph_utils import get_densest_node, textualize_graph
from wemg.utils.preprocessing import get_node_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

class WorkingMemory:
    def __init__(
            self,
            textual_memory: List[str] = [],
            graph_memory: Union[nx.DiGraph, None] = None,
            parsed_graph_memory: Union[nx.DiGraph, None] = None,
            max_textual_memory_tokens: int = 8192,
    ):  
        assert isinstance(textual_memory, list), "textual_memory must be a list of strings"
        if not textual_memory:
            textual_memory = []
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

    @staticmethod
    def format_memory_item(content: str, provenance: roles.extractor.SourceType) -> str:
        """Format a memory item with its provenance tag."""
        if provenance == roles.extractor.SourceType.SYSTEM_PREDICTION:
            return f"[System Prediction]: {content.strip()}"
        elif provenance == roles.extractor.SourceType.RETRIEVAL:
            return f"[Retrieval]: {content.strip()}"
        else:
            return content.strip()
    
    def add_textual_memory(self, text: str, source: roles.extractor.SourceType=roles.extractor.SourceType.SYSTEM_PREDICTION):
        """Add text to textual memory if not already present."""
        text = self.format_memory_item(text, source)
        if text not in self.textual_memory:
            self.textual_memory.append(text)

    def format_textual_memory(self) -> str:
        """Format the textual memory as a single string."""
        memory = [f"- {text.strip()}" for text in self.textual_memory]
        return "\n".join(memory)
    
    @staticmethod
    def memory_consolidation(
        llm_agent: BaseLLMAgent, 
        question: str, 
        raw_memory: str,
        interaction_memory: InteractionMemory = None,
        max_consolidation_tokens: int = 8192,
        ):
        consolidation_input = roles.extractor.MemoryConsolidationInput(question=question, memory=raw_memory)
        response, consolidation_log = asyncio.run(
            execute_role(
                llm_agent=llm_agent,
                role=roles.extractor.MemoryConsolidationRole(),
                input_data=consolidation_input,
                interaction_memory=interaction_memory,
                n=1,
                max_tokens=max_consolidation_tokens
            )
        )
        consolidation_output: roles.extractor.MemoryConsolidationOutput = response[0]
        return consolidation_output, consolidation_log

    def consolidate_textual_memory(
            self, 
            llm_agent: BaseLLMAgent, 
            question: str, 
            interaction_memory: InteractionMemory = None,
            ):
        """Consolidate the textual memory with respect to the question."""
        consolidation_output, consolidation_log = self.memory_consolidation(
            llm_agent=llm_agent,
            question=question,
            raw_memory=self.format_textual_memory(),
            interaction_memory=interaction_memory,
            max_consolidation_tokens=self.max_textual_memory_tokens,
        )
        for item in consolidation_output.consolidated_memory:
            if item.provenance not in [roles.extractor.SourceType.SYSTEM_PREDICTION.value, roles.extractor.SourceType.RETRIEVAL.value]:
                self.add_textual_memory(item.content, source=roles.extractor.SourceType.SYSTEM_PREDICTION)
            else:
                self.add_textual_memory(item.content, source=item.provenance)
        
        # Process logs
        if consolidation_log and interaction_memory:
            for k, v in consolidation_log.items():
                model_input, model_output = zip(*v)
                interaction_memory.log_turn(
                    role=k,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )

    def parse_graph_memory_from_textual_memory(self, llm_agent: BaseLLMAgent, interaction_memory: InteractionMemory = None):
        textual_memory = self.format_textual_memory()
        self.parsed_graph_memory, re_log = parse_graph_from_text(llm_agent=llm_agent, text=textual_memory, interaction_memory=interaction_memory)
        if re_log and interaction_memory:
            for k, v in re_log.items():
                model_input, model_output = zip(*v)
                interaction_memory.log_turn(
                    role=k,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )

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

    def merge_graph_memory(self, other_graph: nx.DiGraph):
        relation_triples: List[roles.open_ie.Relation] = []
        for subject_id, object_id, edge_data in other_graph.edges(data=True):
            subject_entity = other_graph.nodes[subject_id].get('data')
            if not subject_entity:
                continue
            object_entity = other_graph.nodes[object_id].get('data')
            if not object_entity:
                continue
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
    
    async def _process_cluster_async(
        self,
        cluster_raw_text: str,
        llm_agent: BaseLLMAgent,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None
    ) -> tuple[nx.DiGraph, Dict[str, List[tuple[str, str]]]]:
        """Process a single cluster asynchronously: consolidate memory and parse graph."""
        # Run memory consolidation in thread pool to avoid blocking
        consolidation_output, consolidation_log = await asyncio.to_thread(
            self.memory_consolidation,
            llm_agent=llm_agent,
            question=question,
            raw_memory=cluster_raw_text,
            interaction_memory=interaction_memory,
            max_consolidation_tokens=self.max_textual_memory_tokens,
        )
        
        consolidated_texts = []
        for item in consolidation_output.consolidated_memory:
            consolidated_texts.append(
                self.format_memory_item(item.content, item.provenance)
            )
        consolidated_cluster_text = "\n".join([f"- {text}" for text in consolidated_texts])
        
        # Parse graph from consolidated cluster text
        cluster_graph = await asyncio.to_thread(
            parse_graph_from_text,
            llm_agent=llm_agent,
            text=consolidated_cluster_text
        )
        
        return cluster_graph, consolidation_log
    
    def consolidate_graph_memory(self, 
                                 llm_agent: BaseLLMAgent, 
                                 question: str, 
                                 interaction_memory: InteractionMemory = None
                                 ):
        components = list(nx.weakly_connected_components(self.graph_memory))
        self.graph_memory = nx.DiGraph()  # reset graph memory
        all_cluster_text = []
        for comp in components:
            all_triples, _ = textualize_graph(comp, self.graph_memory, method='dfs')
            all_triples = [
                self.format_memory_item(triple, roles.extractor.SourceType.RETRIEVAL) for triple in all_triples
            ]
            cluster_raw_text = [f"- {triple}" for triple in all_triples]
            cluster_raw_text = "\n".join(cluster_raw_text)
            all_cluster_text.append(cluster_raw_text)
        
        # Process all clusters concurrently
        consolidated_graphs, consolidation_logs = asyncio.run(
            asyncio.gather(*[
                self._process_cluster_async(cluster_text, llm_agent, question, interaction_memory=interaction_memory)
                for cluster_text in all_cluster_text
            ])
        )
        
        for cluster_graph in consolidated_graphs:
            # Merge cluster graph into main graph memory
            self.merge_graph_memory(cluster_graph)
        
        # Process logs
        all_log_keys = set(sum([list(log.keys()) for log in consolidation_logs], []))
        to_log_data = {}
        for k in all_log_keys:
            all_values = sum([log.get(k, []) for log in consolidation_logs], [])
            to_log_data[k] = list(set(all_values))
        if consolidation_logs and interaction_memory:
            for k, v in to_log_data.items():
                model_input, model_output = zip(*v)
                interaction_memory.log_turn(
                    role=k,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )

    def synchronize_memory(self, 
                           llm_agent: BaseLLMAgent,
                           question: str,
                           interaction_memory: Optional[InteractionMemory] = None
                           ):
        """Sync graph memory to textual memory and vice versa."""

        # Consolidate graph memory
        self.consolidate_graph_memory(llm_agent=llm_agent, question=question, interaction_memory=interaction_memory)

        # Sync textual memory to graph memory
        if self.textual_memory:
            if not self.parsed_graph_memory or self.parsed_graph_memory.number_of_nodes() == 0:
                self.parse_graph_memory_from_textual_memory(llm_agent=llm_agent, interaction_memory=interaction_memory)
            textual_memory_graph = copy.deepcopy(self.parsed_graph_memory)
            self.merge_graph_memory(textual_memory_graph)
        
        # Sync graph memory to textual memory
        if self.graph_memory and self.graph_memory.number_of_nodes() > 0:
            components = list(nx.weakly_connected_components(self.graph_memory))
            for comp in components:
                all_triples, _ = textualize_graph(comp, self.graph_memory, method='dfs')
                for triple in all_triples:
                    self.add_textual_memory(triple, source=roles.extractor.SourceType.RETRIEVAL)
        
        # Consolidate textual memory
        self.consolidate_textual_memory(llm_agent=llm_agent, question=question, interaction_memory=interaction_memory)
    
    

