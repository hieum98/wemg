"""Working Memory for reasoning processes.

This module manages textual and graph-based memory during reasoning,
including consolidation, synchronization, and retrieval.
"""
import asyncio
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools import wikidata
from wemg.agents.tools.web_search import WebSearchTool
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
        max_textual_memory_tokens: int = 16384,
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
        content = content.strip()
        if content.startswith(("[System Prediction]", "[Retrieval]")):
            return content
        return f"{tag}: {content}" if tag else content
    
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
        raw_memory = self.format_textual_memory()
        output, log = asyncio.run(self._run_consolidation(
            llm_agent=llm_agent,
            question=question,
            raw_memory=raw_memory,
            interaction_memory=interaction_memory
        ))
        output: roles.extractor.MemoryConsolidationOutput
        self.textual_memory = [] # Reset textual memory before adding new items because we don't want to keep the old items
        for item in output.consolidated_memory:
            if item.provenance == roles.extractor.SourceType.SYSTEM_PREDICTION.value:
                provenance = roles.extractor.SourceType.SYSTEM_PREDICTION
            elif item.provenance == roles.extractor.SourceType.RETRIEVAL.value:
                provenance = roles.extractor.SourceType.RETRIEVAL
            else:
                provenance = roles.extractor.SourceType.SYSTEM_PREDICTION
            logger.info(f"Adding item to textual memory: {item} with provenance: {provenance}")
            self.add_textual_memory(item.content, source=provenance)
        
        log_to_interaction_memory(interaction_memory, log)

    async def _run_consolidation(
        self,
        llm_agent: BaseLLMAgent,
        question: str,
        raw_memory: str,
        interaction_memory: Optional[InteractionMemory] = None,
    ) -> Tuple[roles.extractor.MemoryConsolidationOutput, Dict]:
        """Run memory consolidation role."""
        consolidation_input = roles.extractor.MemoryConsolidationInput(
            question=question, 
            memory=raw_memory
        )
        response, log = await execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.MemoryConsolidationRole(),
            input_data=consolidation_input,
            interaction_memory=interaction_memory,
            n=1,
            max_tokens=self.max_textual_memory_tokens
        )
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
        self.parsed_graph_memory, log = asyncio.run(parse_graph_from_text(
            llm_agent=llm_agent, 
            text=self.format_textual_memory(), 
            interaction_memory=interaction_memory
        ))
        log_to_interaction_memory(interaction_memory, log)

    def connect_graph_memory(self, max_hops: int = 2,) -> bool:
        """Connect disconnected components in graph memory using Wikidata paths."""
        if self.graph_memory.number_of_nodes() <= 1:
            return True
        
        components = list(nx.weakly_connected_components(self.graph_memory))
        if len(components) <= 1:
            logger.info("Graph memory is already connected")
            return True
        
        logger.info(f"Found {len(components)} disconnected components")
        
        # Get densest nodes per component
        densest_nodes = [
            get_densest_node(comp, self.graph_memory, node_type=wikidata.WikidataEntity) 
            for comp in components
        ]
        densest_qids = [self.graph_memory.nodes[node_id].get('data').qid for node_id in densest_nodes]
        paths: List[Optional[wikidata.WikidataPathBetweenEntities]] = asyncio.run(asyncio.gather(*[
            self._connect_nodes(densest_qids[i], densest_qids[i + 1], max_hops)
            for i in range(len(densest_nodes) - 1)
        ]))
        for path in paths:
            if path:
                for triple in path.path:
                    self.add_edge_to_graph_memory(triple)
        
        new_components = list(nx.weakly_connected_components(self.graph_memory))
        is_connected = len(new_components) == 1
        logger.info(f"Graph {'now connected' if is_connected else f'has {len(new_components)} components'}")
        return is_connected

    async def _connect_nodes(
        self, 
        source_qid: str, 
        target_qid: str,
        max_hops: int = 2,
    ) -> Optional[wikidata.WikidataPathBetweenEntities]:
        """Find path between two nodes."""
        wikidata_props = wikidata.DEFAULT_PROPERTIES + [item.pid for item in self.property_dict.values()]
        wikidata_props = list(set(wikidata_props))
        wikidata_props_with_labels = {item.pid: {'label': item.label, 'description': item.description} for item in self.property_dict.values()}
        wikidata_props_with_labels = {**wikidata.PROPERTY_LABELS, **wikidata_props_with_labels}
        wikidata_wrapper = wikidata.CustomWikidataAPIWrapper(
            wikidata_props=wikidata_props,
            wikidata_props_with_labels=wikidata_props_with_labels
        )
        path_finder = wikidata.WikidataPathFindingTool(wikidata_wrapper=wikidata_wrapper)
        
        if not source_qid or not target_qid:
            logger.warning(f"Cannot find path: source or target is not a Wikidata entity")
            return
        
        logger.info(f"Searching for path between {source_qid} and {target_qid}")
        path_result: Optional[wikidata.WikidataPathBetweenEntities] = await path_finder.ainvoke({
            "source_qid": source_qid,
            "target_qid": target_qid,
            "max_hops": max_hops,
        })
        
        if path_result and path_result.path:
            logger.info(f"Found path with {len(path_result.path)} hops")
            return path_result
        else:
            logger.warning(f"No path found between {source_qid} and {target_qid}")

    def update_graph_memory(self) -> None:
        """Update nodes with full WikidataEntity details."""
        ids_to_fetch: Set[str] = set()
        
        for node_id in self.graph_memory.nodes:
            node_data = self.graph_memory.nodes[node_id].get('data')
            if isinstance(node_data, wikidata.WikidataEntity):
                if not node_data.description or not node_data.wikipedia_url:
                    if node_data.qid in self.id_dict:
                        self.graph_memory.nodes[node_id]['data'] = self.id_dict[node_data.qid]
                    else:
                        ids_to_fetch.add(node_data.qid)
        
        if ids_to_fetch:
            self._fetch_and_update_entities(list(ids_to_fetch))

    def _fetch_and_update_entities(self, qids: List[str]) -> None:
        """Fetch entity details and update graph memory."""
        to_fetch_entities: Set[str] = set()
        for qid in qids:
            if qid in self.id_dict:
                self.graph_memory.nodes[get_node_id(self.id_dict[qid])]['data'] = self.id_dict[qid]
            else:
                to_fetch_entities.add(qid)
        if to_fetch_entities:
            logger.info(f"Fetching details for {len(to_fetch_entities)} Wikidata entities")
            entity_retriever = wikidata.WikidataEntityRetrievalTool()
            results = entity_retriever.invoke({
                "query": list(to_fetch_entities),
                "num_entities": 1,
                "get_details": True,
            })
            
            for entity in sum(results, []):
                self.id_dict[entity.qid] = entity
                node_id = get_node_id(entity)
                if self.graph_memory.has_node(node_id):
                    self.graph_memory.nodes[node_id]['data'] = entity
                    logger.info(f"Updated node {node_id}")
        
        # Update self.id_dict with wikipedia content
        to_fetch_wikipedia: Set[str] = set()
        for qid in self.id_dict:
            # wikipedia content is not available, but wikipedia url is available (fetched from last step)
            if not self.id_dict[qid].wikipedia_content and self.id_dict[qid].wikipedia_url:
                to_fetch_wikipedia.add(qid)
        if to_fetch_wikipedia:
            logger.info(f"Fetching {len(to_fetch_wikipedia)} Wikipedia content")
            urls = [self.id_dict[qid].wikipedia_url for qid in to_fetch_wikipedia]
            contents = WebSearchTool.crawl_web_pages(urls)
            for qid, content in zip(to_fetch_wikipedia, contents):
                self.id_dict[qid].wikipedia_content = content
                logger.info(f"Updated Wikipedia content for {qid}")

    @staticmethod
    def extract_triples_from_graph(graph: nx.DiGraph) -> List[roles.open_ie.Relation]:
        """Extract relation triples from a graph."""
        triples = []
        for s_id, o_id, edge_data in graph.edges(data=True):
            s_entity = str(graph.nodes[s_id].get('data'))
            o_entity = str(graph.nodes[o_id].get('data'))
            if s_entity and o_entity:
                for rel in edge_data.get('relation', []):
                    triples.append(roles.open_ie.Relation(subject=s_entity, relation=rel, object=o_entity))
        return triples

    def merge_graph_memory(self, other_graph: nx.DiGraph) -> None:
        """Merge another graph into graph memory, syncing with Wikidata."""
        # Collect triples
        triples = WorkingMemory.extract_triples_from_graph(other_graph)
        
        # Fetch missing entities and properties
        self._fetch_missing_entities_and_properties(triples)
        
        # Batch convert and add triples (optimized for performance)
        wiki_triples = self._batch_convert_and_add_triples(triples)
        for wiki_triple in wiki_triples:
            self.add_edge_to_graph_memory(wiki_triple)

        self.connect_graph_memory()
        self.update_graph_memory()
    
    def _batch_convert_and_add_triples(
        self, 
        triples: List[roles.open_ie.Relation], 
        fetch_missing: bool = True,
    ) -> List[wikidata.WikiTriple]:
        """Batch convert triples to WikiTriples.
        """
        all_triples: List[wikidata.WikiTriple] = []
        if fetch_missing:
            self._fetch_missing_entities_and_properties(triples)
        
        to_verify_triples: List[wikidata.WikiTriple] = []
        to_connect_triples: List[wikidata.WikiTriple] = []
        for triple in triples:
            subject = self.entity_dict.get(triple.subject, None)
            relation = self.property_dict.get(triple.relation, None)
            object = self.entity_dict.get(triple.object, None)
            if subject and object and isinstance(subject, wikidata.WikidataEntity) and isinstance(object, wikidata.WikidataEntity):
                if relation:
                    wiki_triple = wikidata.WikiTriple(
                        subject=subject,
                        relation=relation,
                        object=object
                    )
                    to_verify_triples.append(wiki_triple)
                else:
                    wiki_triple = wikidata.WikiTriple(
                        subject=subject,
                        relation=wikidata.WikidataProperty(pid="", label=None, description=None),
                        object=object
                    )
                    to_connect_triples.append(wiki_triple)
        # Verify triples in Wikidata
        if to_verify_triples:
            qids = [triple.subject.qid for triple in to_verify_triples]
            qids.extend([triple.object.qid for triple in to_verify_triples])
            qids = list(set(qids))
            # Here we need retrieve for default properties and properties in self.property_dict
            wikidata_props = wikidata.DEFAULT_PROPERTIES + [item.pid for item in self.property_dict.values()]
            wikidata_props = list(set(wikidata_props))
            wikidata_props_with_labels = {item.pid: {'label': item.label, 'description': item.description} for item in self.property_dict.values()}
            wikidata_props_with_labels = {**wikidata.PROPERTY_LABELS, **wikidata_props_with_labels}
            wikidata_wrapper = wikidata.CustomWikidataAPIWrapper(
                wikidata_props=wikidata_props,
                wikidata_props_with_labels=wikidata_props_with_labels
            )
            retriever = wikidata.WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)
            results = retriever.invoke({
                "query": qids,
                "is_qids": True,
                'k': 1,
            })
            retrieved_triples: List[wikidata.WikiTriple] = sum(results, [])
            id_with_triple: Dict[str, wikidata.WikiTriple] = {}
            for triple in retrieved_triples:
                id = (get_node_id(triple.subject), get_node_id(triple.object))
                id_with_triple[id] = triple
            for triple in to_verify_triples:
                id = (get_node_id(triple.subject), get_node_id(triple.object))
                if id in id_with_triple:
                    all_triples.append(id_with_triple[id])
                else:
                    to_connect_triples.append(triple)
        
        # Connect triples
        tasks = []
        for triple in to_connect_triples:
            tasks.append(
                self._connect_nodes(
                    triple.subject.qid,
                    triple.object.qid,
                    max_hops=2,
                )
            )
        paths: List[Optional[wikidata.WikidataPathBetweenEntities]] = asyncio.run(asyncio.gather(*tasks))
        for path in paths:
            if path:
                all_triples.extend(path.path)
        return all_triples

    def _fetch_missing_entities_and_properties(self, triples: List[roles.open_ie.Relation]):
        """Fetch entities and properties not in cache."""
        to_fetch_entities: Set[str] = set()
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
                "query": list(to_fetch_entities),
                "num_entities": 1,
                "get_details": False,
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
                "top_k_results": 1,
            })
            for prop, result in zip(to_fetch_props, results):
                if result and isinstance(result[0], wikidata.WikidataProperty):
                    self.property_dict[prop] = result[0]

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
        self.update_graph_memory() # Update nodes with full WikidataEntity details before processing
        components = list(nx.weakly_connected_components(self.graph_memory))
        
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
        
        self.graph_memory = nx.DiGraph()  # Reset
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
        output, log = await self._run_consolidation(
            llm_agent=llm_agent,
            question=question,
            raw_memory=cluster_text,
            interaction_memory=interaction_memory
        )
        
        consolidated = [
            self.format_memory_item(item.content, item.provenance)
            for item in output.consolidated_memory
        ]
        consolidated_text = "\n".join(f"- {t}" for t in consolidated)
        
        cluster_graph, _ = await parse_graph_from_text(llm_agent, consolidated_text)
        
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
        # Consolidate memory
        self.consolidate_textual_memory(llm_agent, question, interaction_memory)
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
        
        # Consolidate textual memory after synchronization
        self.consolidate_textual_memory(llm_agent, question, interaction_memory)
