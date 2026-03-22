"""Working memory: textual and graph-based reasoning memory.

This module contains:
- Graph parsing helper (`parse_graph_from_text`)
- `WorkingMemory` class for textual + graph memory
"""

import asyncio
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from wemg.llm.roles import Relation
from wemg.reasoning.generator import merge_logs
from wemg.retrieval.entity_linking import link_entities_azure, link_entities_llm
from wemg.retrieval.wikidata import WikiTriple, WikidataClient, WikidataEntity, WikidataProperty

from .interaction_memory import InteractionMemory, log_to_interaction_memory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


# =============================================================================
# Graph parsing helper
# =============================================================================


async def parse_graph_from_text(
    client,
    text: str,
    interaction_memory: Optional[InteractionMemory] = None,
    known_entities: Optional[List[Any]] = None,
):
    """Parse text into a networkx DiGraph using relation extraction."""
    from wemg.llm.roles import RELATION_EXTRACTOR, RelationExtractionInput, execute_role

    re_input = RelationExtractionInput(text=text, known_entities=known_entities)
    triples, re_log = await execute_role(
        client=client,
        role=RELATION_EXTRACTOR,
        input_data=re_input,
        interaction_memory=interaction_memory,
        n=1,
    )

    relation_triples = triples[0] if triples else None
    if relation_triples and hasattr(relation_triples, "relations"):
        relation_triples = relation_triples.relations
    else:
        logger.error(f"Failed to extract relations from text. Got {type(relation_triples)}")
        relation_triples = []
    return relation_triples, re_log


async def _gather_awaitables(tasks):
    return list(await asyncio.gather(*tasks))


# =============================================================================
# WorkingMemory
# =============================================================================


class WorkingMemory:
    """Manages working memory including textual facts and knowledge graph."""

    def __init__(
        self,
        textual_memory: Optional[List[str]] = None,
        graph_memory: Optional[nx.DiGraph] = None,
        max_textual_memory_tokens: int = 16384,
        wikidata_client=None,
    ):
        self.textual_memory: List[str] = textual_memory or []
        self.graph_memory: nx.DiGraph = graph_memory or nx.DiGraph()
        self.max_textual_memory_tokens = max_textual_memory_tokens
        self._wikidata_client: Optional[WikidataClient] = wikidata_client

        self.entity_dict: Dict[str, WikidataEntity] = {}

    # -----------------------------------------------------------------
    # Textual Memory Operations
    # -----------------------------------------------------------------

    @staticmethod
    def format_memory_item(content: str, provenance) -> str:
        """Format a memory item with its provenance tag."""
        from wemg.llm.roles import SourceType

        tag = {
            SourceType.SYSTEM_PREDICTION: "[System Prediction]",
            SourceType.RETRIEVAL: "[Retrieval]",
        }.get(provenance, "")
        content = content.strip()
        if content.startswith(("[System Prediction]", "[Retrieval]")):
            return content
        return f"{tag}: {content}" if tag else content

    def add_textual_memory(self, text: str, source=None) -> None:
        """Add text to textual memory if not already present."""
        if source is None:
            from wemg.llm.roles import SourceType

            source = SourceType.SYSTEM_PREDICTION
        formatted = self.format_memory_item(text, source)
        if formatted not in self.textual_memory:
            self.textual_memory.append(formatted)

    def format_textual_memory(self) -> str:
        return "\n".join(f"- {text.strip()}" for text in self.textual_memory)

    def consolidate_textual_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
    ) -> None:
        """Consolidate textual memory with respect to the question."""
        from wemg.llm.roles import SourceType

        raw_memory = self.format_textual_memory()
        output, log = asyncio.run(
            self._run_consolidation(
                client=client,
                question=question,
                raw_memory=raw_memory,
                interaction_memory=interaction_memory,
            )
        )
        self.textual_memory = []
        for item in output.consolidated_memory:
            if item.provenance == SourceType.SYSTEM_PREDICTION.value:
                provenance = SourceType.SYSTEM_PREDICTION
            elif item.provenance == SourceType.RETRIEVAL.value:
                provenance = SourceType.RETRIEVAL
            else:
                logger.warning(f"Invalid provenance: {item.provenance}")
                provenance = SourceType.SYSTEM_PREDICTION
            logger.info(f"Adding item to textual memory: {item} with provenance: {provenance}")
            self.add_textual_memory(item.content, source=provenance)

        log_to_interaction_memory(interaction_memory, log)

    async def _run_consolidation(
        self,
        client,
        question: str,
        raw_memory: str,
        interaction_memory: Optional[InteractionMemory] = None,
    ):
        from wemg.llm.roles import (
            MEMORY_CONSOLIDATOR,
            MemoryConsolidationInput,
            execute_role,
        )

        consolidation_input = MemoryConsolidationInput(
            question=question,
            memory=raw_memory,
        )
        response, log = await execute_role(
            client=client,
            role=MEMORY_CONSOLIDATOR,
            input_data=consolidation_input,
            interaction_memory=interaction_memory,
            n=1,
            max_tokens=self.max_textual_memory_tokens,
        )
        if not response:
            logger.warning("Memory consolidation returned empty response; returning input as-is.")
            from wemg.llm.roles import MemoryConsolidationOutput, MemoryItem
            fallback = MemoryConsolidationOutput(consolidated_memory=[
                MemoryItem(content=raw_memory, provenance="system_prediction")
            ])
            return fallback, log
        return response[0], log

    # -----------------------------------------------------------------
    # Graph Memory Operations
    # -----------------------------------------------------------------

    def add_node_to_graph_memory(self, node) -> None:
        """Add a node to graph memory."""
        from wemg.utils.graph import get_node_id
        from wemg.llm.roles import Entity as OpenIEEntity
        from wemg.retrieval.wikidata import WikidataEntity

        if isinstance(node, OpenIEEntity):
            if node.id and self.entity_dict.get(node.id, None):
                node = self.entity_dict[node.id]
            elif node.name: # Known entity name but don't have id or this is not a known entity, retrieve the entity from Wikidata using the WikidataClient
                assert self._wikidata_client is not None, "WikidataClient must be provided"
                entity_name = node.id if node.id else node.name
                result = self._wikidata_client.search_entities(entity_name, num_results=1, get_details=False)
                node = result[0] if result else None
                if node is None:
                    logger.warning(f"No entity found for {entity_name}")
                    return
            else:
                logger.warning(f"Empty entity: {node}")
                return

        if isinstance(node, WikidataEntity):
            if node.qid and node.label:
                node_data = OpenIEEntity(id=node.qid, name=node.label, description=node.description)
                if node.qid not in self.entity_dict:
                    self.entity_dict[node.qid] = node

            elif node.label: # Scalar value (like dates, numbers, etc.)
                node_data = OpenIEEntity(id=None, name=node.label, description=node.description)
            else: # Known enity id but don't have label or description
                if self.entity_dict.get(node.qid, None):
                    node = self.entity_dict[node.qid]
                # Retrieve the entity from Wikidata using the WikidataClient and its qid
                assert self._wikidata_client is not None, "WikidataClient must be provided"
                if self._wikidata_client._is_entity_enriched(node):
                    node_data = OpenIEEntity(id=node.qid, name=node.label, description=node.description)
                else:
                    node = self._wikidata_client.enrich_entities([node], get_details=False)[0]
                    if node is None or not node.qid or not node.label:
                        logger.warning(f"No entity found for {node}")
                        return
                    node_data = OpenIEEntity(id=node.qid, name=node.label, description=node.description)
                    if node.qid not in self.entity_dict:
                        self.entity_dict[node.qid] = node
        else:
            raise ValueError(f"Invalid node type: {type(node)}")
        
        node_id = get_node_id(node_data)
        if not self.graph_memory.has_node(node_id):
            self.graph_memory.add_node(node_id, data=node_data)
        return node_data

    def add_edge_to_graph_memory(self, triple) -> None:
        """Add a triple to graph memory."""
        from wemg.utils.graph import get_node_id
        from wemg.llm.roles import Relation as OpenIERelation, Entity as OpenIEEntity
        from wemg.retrieval.wikidata import WikiTriple

        if isinstance(triple, OpenIERelation):
            subject = OpenIEEntity(id=triple.subject_id, name=triple.subject, description=None)
            object = OpenIEEntity(id=triple.object_id, name=triple.object, description=None)
            subject_data = self.add_node_to_graph_memory(subject)
            object_data = self.add_node_to_graph_memory(object)
            relation_label = str(triple.relation) if triple.relation else None
        elif isinstance(triple, WikiTriple):
            if isinstance(triple.subject, WikidataEntity):
                subject = OpenIEEntity(id=triple.subject.qid, name=triple.subject.label, description=triple.subject.description)
                subject_data = self.add_node_to_graph_memory(subject)
            else:
                subject_data = str(triple.subject)
            if isinstance(triple.object, WikidataEntity):
                object = OpenIEEntity(id=triple.object.qid, name=triple.object.label, description=triple.object.description)
                object_data = self.add_node_to_graph_memory(object)
            else:
                object_data = str(triple.object)
            assert isinstance(triple.relation, WikidataProperty), "Relation must be a WikidataProperty for a WikiTriple"
            relation_label = triple.relation.label if triple.relation.label else None
        else:
            raise ValueError(f"Invalid triple type: {type(triple)}")

        if subject_data is None or object_data is None or not relation_label:
            logger.warning(f"Skipping invalid triple for graph memory: {triple}")
            return

        subject_id = get_node_id(subject_data)
        object_id = get_node_id(object_data)
        # check all nodes in the triple are in the graph memory
        if subject_id not in self.graph_memory or object_id not in self.graph_memory:
            logger.warning(f"Skipping invalid triple for graph memory: {triple}")
            return

        logger.info(f"Adding edge to graph memory: {str(triple)}")

        if not self.graph_memory.has_edge(subject_id, object_id):
            self.graph_memory.add_edge(subject_id, object_id, relation={relation_label})
        else:
            rel_set = self.graph_memory.edges[subject_id, object_id].setdefault("relation", set())
            rel_set.add(relation_label)
    
    def format_graph_memory(self) -> str:
        """Format graph memory as a single string."""
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        components = list(nx.weakly_connected_components(self.graph_memory))

        cluster_texts = []
        for comp in components:
            triples, _ = textualize_graph(comp, self.graph_memory, method="dfs")
            formatted = [
                self.format_memory_item(t, SourceType.SYSTEM_PREDICTION) for t in triples
            ]
            cluster_texts.append("\n".join(f"- {t}" for t in formatted))

        cluster_text = "**Information**\n".join(
            [f"{i}. {text}" for i, text in enumerate(cluster_texts, 1)]
        )
        return cluster_text

    def connect_graph_memory(self, max_hops: int = 1) -> bool:
        """Connect disconnected components in graph memory using Wikidata paths."""
        from wemg.retrieval.wikidata import WikidataPathBetweenEntities
        from wemg.utils.graph import get_densest_node

        if self.graph_memory.number_of_nodes() <= 1:
            return True

        components = list(nx.weakly_connected_components(self.graph_memory))
        if len(components) <= 1:
            logger.info("Graph memory is already connected")
            return True

        logger.info(f"Found {len(components)} disconnected components")

        densest_nodes = [
            get_densest_node(comp, self.graph_memory, lambda x: hasattr(x, "qid") and x.qid)
            for comp in components
        ]
        densest_qids: List[Optional[str]] = []
        for node_id in densest_nodes:
            if node_id is None:
                densest_qids.append(None)
            else:
                data = self.graph_memory.nodes[node_id].get("data")
                if data is not None and hasattr(data, "qid") and data.qid:
                    densest_qids.append(data.qid)
                else:
                    densest_qids.append(None)

        assert self._wikidata_client is not None, "WikidataClient must be provided"
        valid_pairs = [
            i for i in range(len(densest_qids) - 1)
            if densest_qids[i] is not None and densest_qids[i + 1] is not None
        ]
        tasks = [
            self._wikidata_client.afind_path(densest_qids[i], densest_qids[i + 1], max_hops=max_hops)
            for i in valid_pairs
        ]
        paths = asyncio.run(_gather_awaitables(tasks))
        for i, path in zip(valid_pairs, paths):
            if path and isinstance(path, WikidataPathBetweenEntities) and path.path:
                for triple in path.path:
                    self.add_edge_to_graph_memory(triple)
            else:
                logger.warning(f"No path found between {densest_qids[i]} and {densest_qids[i + 1]}")

        new_components = list(nx.weakly_connected_components(self.graph_memory))
        is_connected = len(new_components) == 1
        logger.info(
            f"Graph {'now connected' if is_connected else f'has {len(new_components)} components'}"
        )
        return is_connected

    def _enhance_triples(self, triples: List[Relation]) -> List[WikiTriple]:
        """Enhance triples with entity and property information."""
        to_retrieve = []
        for triple in triples:
            if not triple.subject_id:
                to_retrieve.append(triple.subject)
            elif triple.subject_id not in self.entity_dict:
                to_retrieve.append(triple.subject_id)
            if not triple.object_id:
                to_retrieve.append(triple.object)
            elif triple.object_id not in self.entity_dict:
                to_retrieve.append(triple.object_id)

        to_retrieve = list(set(to_retrieve))
        retrieved_results = {}
        if to_retrieve:
            assert self._wikidata_client is not None, "WikidataClient must be provided"
            entities = self._wikidata_client.search_entities(to_retrieve, num_results=1, get_details=False)
            for i, entity in enumerate(entities):
                first = entity[0] if isinstance(entity, list) and entity else None
                if first is not None and isinstance(first, WikidataEntity) and first.qid:
                    self.entity_dict[first.qid] = first
                    retrieved_results[to_retrieve[i]] = first.qid
                else:
                    logger.warning(f"Invalid or empty entity result: {entity} for {to_retrieve[i]}")
        
        enhanced_triples: List[WikiTriple] = []
        for triple in triples:
            s = None
            o = None
            r = None
            if triple.subject_id:
                subject_key = retrieved_results.get(triple.subject_id, triple.subject_id)
                s = self.entity_dict.get(subject_key, None)
            if triple.object_id:
                object_key = retrieved_results.get(triple.object_id, triple.object_id)
                o = self.entity_dict.get(object_key, None)
            if triple.relation:
                r = WikidataProperty(pid="", label=triple.relation, description=None)
            if s and o and r:
                enhanced_triples.append(WikiTriple(subject=s, relation=r, object=o))

        return enhanced_triples

    async def _link_entities_async(
        self,
        client,
        text: str,
        known_entities: Optional[List[WikidataEntity]],
        interaction_memory: Optional[InteractionMemory] = None,
        *,
        entity_linking_method: str = "llm",
        top_k_entities: int = 1,
        reranker=None,
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
    ) -> Tuple[List[WikidataEntity], Dict]:
        """Link entities in ``text``; returns resolved entities and role log only (no cache writes)."""
        if not (text and text.strip()) or self._wikidata_client is None:
            return [], {}
        if entity_linking_method == "azure":
            wikidata_entities, _, link_log = await link_entities_azure(
                text,
                self._wikidata_client,
                endpoint=azure_endpoint,
                key=azure_key,
            )
        else:
            wikidata_entities, _, link_log = await link_entities_llm(
                client,
                text,
                self._wikidata_client,
                top_k_entities=top_k_entities,
                interaction_memory=interaction_memory,
                known_entities=known_entities,
                reranker=reranker,
            )
        flat: List[WikidataEntity] = []
        for e in wikidata_entities:
            if isinstance(e, WikidataEntity) and e.qid:
                flat.append(e)
        return flat, (link_log if isinstance(link_log, dict) else {})

    def _merge_linked_entities_into_cache(self, entities: List[WikidataEntity]) -> None:
        """Apply linking results to ``entity_dict`` (call only from sync code, after awaitables finish)."""
        for e in entities:
            if e.qid:
                self.entity_dict[e.qid] = e

    @staticmethod
    def _entity_link_kwargs_from_call(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_linking_method": kwargs.get("entity_linking_method", "llm"),
            "top_k_entities": kwargs.get("top_k_entities", 1),
            "reranker": kwargs.get("reranker"),
            "azure_endpoint": kwargs.get("azure_endpoint"),
            "azure_key": kwargs.get("azure_key"),
        }

    # -----------------------------------------------------------------
    # Graph Consolidation
    # -----------------------------------------------------------------

    def consolidate_graph_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        """Consolidate graph memory by processing each component."""
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        textualized_graph = self.format_graph_memory()
        if not textualized_graph:
            return
        

        consolidated_output, consolidation_log = asyncio.run(self._run_consolidation(
            client=client,
            question=question,
            raw_memory=textualized_graph,
            interaction_memory=interaction_memory,
        ))
        consolidated = [
            self.format_memory_item(item.content, item.provenance)
            for item in consolidated_output.consolidated_memory
        ]
        consolidated_text = "\n".join(f"- {t}" for t in consolidated)
        known_entities = list(self.entity_dict.values())
        triples, parse_log = asyncio.run(parse_graph_from_text(client, consolidated_text, interaction_memory=interaction_memory, known_entities=known_entities))
        log = merge_logs(consolidation_log, parse_log)
        log_to_interaction_memory(interaction_memory, log)
        enhanced_triples = self._enhance_triples(triples)

        # Update the graph memory with the enhanced triples
        self.graph_memory = nx.DiGraph()
        for triple in enhanced_triples:
            self.add_edge_to_graph_memory(triple)
        # Connect the graph memory
        self.connect_graph_memory()

    # -----------------------------------------------------------------
    # Memory Synchronization
    # -----------------------------------------------------------------

    def synchronize_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        """Synchronize graph and textual memory bidirectionally."""
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        if self.graph_memory.number_of_nodes() > 0:
            for comp in nx.weakly_connected_components(self.graph_memory):
                triples, _ = textualize_graph(comp, self.graph_memory, method="dfs")
                for triple in triples:
                    self.add_textual_memory(triple, source=SourceType.SYSTEM_PREDICTION)
        self.consolidate_textual_memory(client, question, interaction_memory)

        if self.textual_memory:
            textual_memory = self.format_textual_memory()
            known_snapshot = [
                e for e in self.entity_dict.values() if isinstance(e, WikidataEntity)
            ] or None
            link_kw = self._entity_link_kwargs_from_call(kwargs)
            linked_entities, link_log = asyncio.run(
                self._link_entities_async(
                    client,
                    textual_memory,
                    known_snapshot,
                    interaction_memory,
                    **link_kw,
                )
            )
            self._merge_linked_entities_into_cache(linked_entities)
            known_entities = list(self.entity_dict.values())
            triples, parse_log = asyncio.run(parse_graph_from_text(client, textual_memory, interaction_memory=interaction_memory, known_entities=known_entities))
            log_to_interaction_memory(interaction_memory, merge_logs(link_log, parse_log))
            enhanced_triples = self._enhance_triples(triples)
            for triple in enhanced_triples:
                self.add_edge_to_graph_memory(triple)
            self.connect_graph_memory() 

__all__ = [
    "parse_graph_from_text",
    "WorkingMemory",
]

