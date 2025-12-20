"""Working Memory for reasoning processes.

This module manages textual and graph-based memory during reasoning,
including consolidation, synchronization, and retrieval.
"""
import asyncio
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools import wikidata
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.procedures.openie import parse_graph_from_text
from wemg.utils.graph_utils import get_densest_node, textualize_graph
from wemg.utils.preprocessing import get_node_id
from wemg.utils.common import log_to_interaction_memory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class WorkingMemory:
    """Manages working memory including textual facts and knowledge graph."""
    
    def __init__(
        self,
        textual_memory: Optional[List[str]] = None,
        graph_memory: Optional[nx.DiGraph] = None,
        parsed_graph_memory: Optional[nx.DiGraph] = None,
        max_textual_memory_tokens: int = 8192,
    ):
        self.textual_memory: List[str] = textual_memory or []
        self.graph_memory: nx.DiGraph = graph_memory or nx.DiGraph()
        self.parsed_graph_memory: nx.DiGraph = parsed_graph_memory or nx.DiGraph()
        self.max_textual_memory_tokens = max_textual_memory_tokens
        
        # Entity/property mappings
        self.entity_dict: Dict[roles.open_ie.Entity, wikidata.WikidataEntity] = {}
        self.property_dict: Dict[str, wikidata.WikidataProperty] = {}
        self.id_dict: Dict[str, wikidata.WikidataEntity] = {}

    # =========================================================================
    # Textual Memory Operations
    # =========================================================================
    
    @staticmethod
    def format_memory_item(content: str, provenance: roles.extractor.SourceType) -> str:
        """Format a memory item with its provenance tag."""
        tag = {
            roles.extractor.SourceType.SYSTEM_PREDICTION: "[System Prediction]",
            roles.extractor.SourceType.RETRIEVAL: "[Retrieval]",
        }.get(provenance, "")
        return f"{tag}: {content.strip()}" if tag else content.strip()
    
    def add_textual_memory(
        self, 
        text: str, 
        source: roles.extractor.SourceType = roles.extractor.SourceType.SYSTEM_PREDICTION
    ) -> None:
        """Add text to textual memory if not already present."""
        formatted = self.format_memory_item(text, source)
        if formatted not in self.textual_memory:
            self.textual_memory.append(formatted)

    def format_textual_memory(self) -> str:
        """Format textual memory as a single string."""
        return "\n".join(f"- {text.strip()}" for text in self.textual_memory)
    
    def consolidate_textual_memory(
        self, 
        llm_agent: BaseLLMAgent, 
        question: str, 
        interaction_memory: Optional[InteractionMemory] = None,
    ) -> None:
        """Consolidate textual memory with respect to the question."""
        output, log = self._run_consolidation(
            llm_agent, question, self.format_textual_memory(), interaction_memory
        )
        
        for item in output.consolidated_memory:
            provenance = (
                roles.extractor.SourceType.SYSTEM_PREDICTION 
                if item.provenance not in [
                    roles.extractor.SourceType.SYSTEM_PREDICTION.value,
                    roles.extractor.SourceType.RETRIEVAL.value
                ]
                else item.provenance
            )
            self.add_textual_memory(item.content, source=provenance)
        
        log_to_interaction_memory(interaction_memory, log)

    def _run_consolidation(
        self,
        llm_agent: BaseLLMAgent,
        question: str,
        raw_memory: str,
        interaction_memory: Optional[InteractionMemory] = None,
    ):
        """Run memory consolidation role."""
        consolidation_input = roles.extractor.MemoryConsolidationInput(
            question=question, 
            memory=raw_memory
        )
        response, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.MemoryConsolidationRole(),
            input_data=consolidation_input,
            interaction_memory=interaction_memory,
            n=1,
            max_tokens=self.max_textual_memory_tokens
        ))
        return response[0], log

    # =========================================================================
    # Graph Memory Operations
    # =========================================================================
    
    def add_edge_to_graph_memory(self, wiki_triple: wikidata.WikiTriple) -> None:
        """Add a triple to graph memory."""
        subject_id = get_node_id(wiki_triple.subject)
        object_id = get_node_id(wiki_triple.object)
        
        # Add nodes if missing
        for node_id, data in [(subject_id, wiki_triple.subject), (object_id, wiki_triple.object)]:
            if not self.graph_memory.has_node(node_id):
                self.graph_memory.add_node(node_id, data=data)
        
        # Add or update edge
        if not self.graph_memory.has_edge(subject_id, object_id):
            self.graph_memory.add_edge(subject_id, object_id, relation={wiki_triple.relation})
        else:
            self.graph_memory.edges[subject_id, object_id]['relation'].add(wiki_triple.relation)
    
    def parse_graph_memory_from_textual_memory(
        self, 
        llm_agent: BaseLLMAgent, 
        interaction_memory: Optional[InteractionMemory] = None
    ) -> None:
        """Parse graph from textual memory using OpenIE."""
        self.parsed_graph_memory, log = parse_graph_from_text(
            llm_agent=llm_agent, 
            text=self.format_textual_memory(), 
            interaction_memory=interaction_memory
        )
        log_to_interaction_memory(interaction_memory, log)

    def connect_graph_memory(
        self,
        wikidata_path_finder: Optional["wikidata.WikidataPathFindingTool"] = None,
        max_hops: int = 3,
    ) -> bool:
        """Connect disconnected components in graph memory using Wikidata paths."""
        if self.graph_memory.number_of_nodes() <= 1:
            return True
        
        components = list(nx.weakly_connected_components(self.graph_memory))
        if len(components) <= 1:
            logger.info("Graph memory is already connected")
            return True
        
        logger.info(f"Found {len(components)} disconnected components")
        
        if wikidata_path_finder is None:
            wikidata_path_finder = wikidata.WikidataPathFindingTool()
        
        # Get densest nodes per component
        densest_nodes = [
            get_densest_node(comp, self.graph_memory, node_type=wikidata.WikidataEntity) 
            for comp in components
        ]
        
        # Connect adjacent components
        for i in range(len(densest_nodes) - 1):
            self._connect_nodes(
                densest_nodes[i], densest_nodes[i + 1], 
                wikidata_path_finder, max_hops
            )
        
        new_components = list(nx.weakly_connected_components(self.graph_memory))
        is_connected = len(new_components) == 1
        logger.info(f"Graph {'now connected' if is_connected else f'has {len(new_components)} components'}")
        return is_connected

    def _connect_nodes(
        self, 
        source_id: str, 
        target_id: str,
        path_finder: "wikidata.WikidataPathFindingTool",
        max_hops: int
    ) -> None:
        """Find and add path between two nodes."""
        source_data = self.graph_memory.nodes[source_id].get('data')
        target_data = self.graph_memory.nodes[target_id].get('data')
        
        source_qid = source_data.qid if isinstance(source_data, wikidata.WikidataEntity) else None
        target_qid = target_data.qid if isinstance(target_data, wikidata.WikidataEntity) else None
        
        if not source_qid or not target_qid:
            logger.warning(f"Cannot find path: source or target is not a Wikidata entity")
            return
        
        logger.info(f"Searching for path between {source_qid} and {target_qid}")
        path_result = path_finder.invoke({
            "source_qid": source_qid,
            "target_qid": target_qid,
            "max_hops": max_hops,
        })
        
        if path_result and path_result.path:
            logger.info(f"Found path with {len(path_result.path)} hops")
            for triple in path_result.path:
                self.add_edge_to_graph_memory(triple)
        else:
            logger.warning(f"No path found between {source_qid} and {target_qid}")

    def update_graph_memory(self) -> None:
        """Update nodes with full WikidataEntity details."""
        ids_to_fetch: Set[str] = set()
        
        for node_id in self.graph_memory.nodes:
            node_data = self.graph_memory.nodes[node_id].get('data')
            if isinstance(node_data, wikidata.WikidataEntity):
                if not node_data.description or not node_data.wikipedia_content:
                    if node_data.qid in self.id_dict:
                        self.graph_memory.nodes[node_id]['data'] = self.id_dict[node_data.qid]
                    else:
                        ids_to_fetch.add(node_data.qid)
        
        if ids_to_fetch:
            self._fetch_and_update_entities(list(ids_to_fetch))

    def _fetch_and_update_entities(self, qids: List[str]) -> None:
        """Fetch entity details and update graph memory."""
        logger.info(f"Fetching details for {len(qids)} Wikidata entities")
        entity_retriever = wikidata.WikidataEntityRetrievalTool()
        results = entity_retriever.invoke({
            "query": qids,
            "num_entities": 1,
            "get_details": True,
        })
        
        for entity in sum(results, []):
            self.id_dict[entity.qid] = entity
            node_id = get_node_id(entity)
            if self.graph_memory.has_node(node_id):
                self.graph_memory.nodes[node_id]['data'] = entity
                logger.info(f"Updated node {node_id}")

    def merge_graph_memory(self, other_graph: nx.DiGraph) -> None:
        """Merge another graph into graph memory, syncing with Wikidata."""
        # Collect triples
        triples = self._extract_triples_from_graph(other_graph)
        
        # Fetch missing entities and properties
        self._fetch_missing_entities_and_properties(triples)
        
        # Convert and add triples
        for triple in triples:
            wiki_triple = self._convert_to_wiki_triple(triple)
            if wiki_triple:
                self.add_edge_to_graph_memory(wiki_triple)
                logger.info(f"Added triple: {wiki_triple}")
        
        self.connect_graph_memory()
        self.update_graph_memory()

    def _extract_triples_from_graph(self, graph: nx.DiGraph) -> List[roles.open_ie.Relation]:
        """Extract relation triples from a graph."""
        triples = []
        for s_id, o_id, edge_data in graph.edges(data=True):
            s_entity = graph.nodes[s_id].get('data')
            o_entity = graph.nodes[o_id].get('data')
            if s_entity and o_entity:
                for rel in edge_data.get('relation', []):
                    triples.append(roles.open_ie.Relation(
                        subject=s_entity, relation=rel, object=o_entity
                    ))
        return triples

    def _fetch_missing_entities_and_properties(
        self, 
        triples: List[roles.open_ie.Relation]
    ) -> None:
        """Fetch entities and properties not in cache."""
        to_fetch_entities: Set[roles.open_ie.Entity] = set()
        to_fetch_props: Set[str] = set()
        
        for triple in triples:
            if triple.subject not in self.entity_dict:
                to_fetch_entities.add(triple.subject)
            if triple.object not in self.entity_dict:
                to_fetch_entities.add(triple.object)
            if triple.relation not in self.property_dict:
                to_fetch_props.add(triple.relation)
        
        # Fetch entities
        if to_fetch_entities:
            logger.info(f"Fetching {len(to_fetch_entities)} entities from Wikidata")
            retriever = wikidata.WikidataEntityRetrievalTool()
            results = retriever.invoke({
                "query": [e.name for e in to_fetch_entities],
                "num_entities": 1,
                "get_details": True,
            })
            for item, result in zip(to_fetch_entities, results):
                if result and isinstance(result[0], wikidata.WikidataEntity):
                    self.entity_dict[item] = result[0]
        
        # Fetch properties
        if to_fetch_props:
            logger.info(f"Fetching {len(to_fetch_props)} properties from Wikidata")
            retriever = wikidata.WikidataPropertyRetrievalTool()
            results = retriever.invoke({
                "query": list(to_fetch_props),
                "num_properties": 1,
            })
            for prop, result in zip(to_fetch_props, results):
                if result and isinstance(result[0], wikidata.WikidataProperty):
                    self.property_dict[prop] = result[0]

    def _convert_to_wiki_triple(
        self, 
        triple: roles.open_ie.Relation
    ) -> Optional[wikidata.WikiTriple]:
        """Convert a Relation to WikiTriple if all entities/properties exist."""
        subject = self.entity_dict.get(triple.subject)
        obj = self.entity_dict.get(triple.object)
        prop = self.property_dict.get(triple.relation)
        
        if subject and obj and prop:
            return wikidata.WikiTriple(subject=subject, relation=prop, object=obj)
        return None

    # =========================================================================
    # Graph Consolidation
    # =========================================================================
    
    def consolidate_graph_memory(
        self, 
        llm_agent: BaseLLMAgent, 
        question: str, 
        interaction_memory: Optional[InteractionMemory] = None
    ) -> None:
        """Consolidate graph memory by processing each component."""
        components = list(nx.weakly_connected_components(self.graph_memory))
        self.graph_memory = nx.DiGraph()  # Reset
        
        # Process components concurrently
        cluster_texts = []
        for comp in components:
            triples, _ = textualize_graph(comp, self.graph_memory, method='dfs')
            formatted = [
                self.format_memory_item(t, roles.extractor.SourceType.RETRIEVAL) 
                for t in triples
            ]
            cluster_texts.append("\n".join(f"- {t}" for t in formatted))
        
        results = asyncio.run(asyncio.gather(*[
            self._process_cluster_async(text, llm_agent, question, interaction_memory)
            for text in cluster_texts
        ]))
        
        for cluster_graph, log in results:
            self.merge_graph_memory(cluster_graph)
            log_to_interaction_memory(interaction_memory, log)

    async def _process_cluster_async(
        self,
        cluster_text: str,
        llm_agent: BaseLLMAgent,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None
    ):
        """Process a cluster: consolidate and parse graph."""
        output, log = await asyncio.to_thread(
            self._run_consolidation,
            llm_agent, question, cluster_text, interaction_memory
        )
        
        consolidated = [
            self.format_memory_item(item.content, item.provenance)
            for item in output.consolidated_memory
        ]
        consolidated_text = "\n".join(f"- {t}" for t in consolidated)
        
        cluster_graph = await asyncio.to_thread(
            parse_graph_from_text, llm_agent, consolidated_text
        )
        
        return cluster_graph, log

    # =========================================================================
    # Memory Synchronization
    # =========================================================================
    
    def synchronize_memory(
        self, 
        llm_agent: BaseLLMAgent,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None
    ) -> None:
        """Synchronize graph and textual memory bidirectionally."""
        # Consolidate graph memory
        self.consolidate_graph_memory(llm_agent, question, interaction_memory)

        # Sync textual → graph
        if self.textual_memory:
            if not self.parsed_graph_memory or self.parsed_graph_memory.number_of_nodes() == 0:
                self.parse_graph_memory_from_textual_memory(llm_agent, interaction_memory)
            self.merge_graph_memory(copy.deepcopy(self.parsed_graph_memory))
        
        # Sync graph → textual
        if self.graph_memory.number_of_nodes() > 0:
            for comp in nx.weakly_connected_components(self.graph_memory):
                triples, _ = textualize_graph(comp, self.graph_memory, method='dfs')
                for triple in triples:
                    self.add_textual_memory(triple, source=roles.extractor.SourceType.RETRIEVAL)
        
        # Consolidate textual memory
        self.consolidate_textual_memory(llm_agent, question, interaction_memory)
