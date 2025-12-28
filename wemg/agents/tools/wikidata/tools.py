"""Wikidata tools - BaseTool classes for entity, property, and triple retrieval."""

import asyncio
import copy
import logging
from typing import List, Dict, Optional, Set, Union, Tuple

import pydantic
from langchain_core.tools import BaseTool

from wemg.agents.tools.wikidata.models import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    WikidataPathBetweenEntities,
)
from wemg.agents.tools.wikidata.api_wrapper import CustomWikidataAPIWrapper
from wemg.agents.tools.wikidata.enrichment import EnrichmentCollector
from wemg.agents.tools.wikidata.utils import (
    normalize_single_or_list,
    unwrap_single_result,
    flatten_and_map_ids,
    build_result_map,
)
from wemg.agents.tools.wikidata.constants import DEFAULT_PROPERTIES, PROPERTY_LABELS


logger = logging.getLogger(__name__)


class WikidataEntityRetrievalTool(BaseTool):
    """Tool for retrieving Wikidata entities based on a textual query."""
    
    name: str = "Wikidata Entity Retrieval Tool"
    description: str = (
        "A tool to retrieve Wikidata entities based on a textual query. "
        "Given a query, it returns a list of Wikidata entities with their details."
    )
    wikidata_wrapper: CustomWikidataAPIWrapper = pydantic.Field(
        default_factory=lambda: CustomWikidataAPIWrapper(lang="en", top_k_results=3),
        description="An instance of CustomWikidataAPIWrapper for querying Wikidata."
    )

    def _run(
        self, 
        query: Union[str, List[str]], 
        num_entities: int = 3, 
        is_qids: bool = False, 
        get_details: bool = False
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Retrieve Wikidata entities based on a textual query or QIDs."""
        is_single, query_list = normalize_single_or_list(query)
        
        if is_qids:
            qids_per_query = [[qid] for qid in query_list]
        else:
            search_results = self.wikidata_wrapper._get_id(query_list)
            # _get_id returns List[List[str]] when given a list, so extract results per query
            qids_per_query = [results[:num_entities] if results else [] for results in search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids, qid_to_query_idx = flatten_and_map_ids(qids_per_query)
        
        if not all_qids:
            return unwrap_single_result([[] for _ in qids_per_query], is_single)
        
        # Use batch _get_item for efficient retrieval
        results = self.wikidata_wrapper._get_item(all_qids, get_details=get_details)
        
        # Build QID to entity map
        qid_to_entity = build_result_map(results, all_qids)
        
        # Build output aligned with input queries
        output: List[List[WikidataEntity]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid in qid_to_entity:
                    output[query_idx].append(qid_to_entity[qid])
        
        return unwrap_single_result(output, is_single)

    async def _arun(
        self, 
        query: Union[str, List[str]], 
        num_entities: int = 3, 
        is_qids: bool = False, 
        get_details: bool = False
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Async version of entity retrieval with batch support."""
        is_single, query_list = normalize_single_or_list(query)
        
        if is_qids:
            qids_per_query = [[qid] for qid in query_list]
        else:
            search_results = await asyncio.to_thread(self.wikidata_wrapper._get_id, query_list)
            # _get_id returns List[List[str]] when given a list, so extract results per query
            qids_per_query = [results[:num_entities] if results else [] for results in search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids, qid_to_query_idx = flatten_and_map_ids(qids_per_query)
        
        if not all_qids:
            return unwrap_single_result([[] for _ in qids_per_query], is_single)
        
        # Use batch _get_item_async for efficient retrieval
        try:
            results = await self.wikidata_wrapper._get_item_async(all_qids, get_details=get_details)
        except Exception as e:
            logger.error(f"Error fetching entities: {e}")
            return unwrap_single_result([[] for _ in qids_per_query], is_single)
        
        # Build QID to entity map
        qid_to_entity = build_result_map(results, all_qids)
        
        # Build output aligned with input queries
        output: List[List[WikidataEntity]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid in qid_to_entity:
                    output[query_idx].append(qid_to_entity[qid])
        
        return unwrap_single_result(output, is_single)


class WikidataPropertyRetrievalTool(BaseTool):
    """Tool for retrieving Wikidata properties based on a textual query."""
    
    name: str = "Wikidata Property Retrieval Tool"
    description: str = (
        "A tool to retrieve Wikidata properties based on a textual query. "
        "Given a query, it returns a list of Wikidata properties with their details."
    )
    wikidata_wrapper: CustomWikidataAPIWrapper = pydantic.Field(
        default_factory=lambda: CustomWikidataAPIWrapper(lang="en", top_k_results=3),
        description="An instance of CustomWikidataAPIWrapper for querying Wikidata."
    )

    def _run(
        self, 
        query: Union[str, List[str]], 
        top_k_results: int = 3
    ) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        """Retrieve Wikidata properties based on a textual query."""
        is_single, query_list = normalize_single_or_list(query)
        pids = self.wikidata_wrapper._get_id(query_list, id_type="property")
        pids = [pid_list[:top_k_results] if pid_list else [] for pid_list in pids]
        all_pids = set(sum(pids, []))
        all_properties: Dict[str, WikidataProperty] = {}
        if all_pids:
            fetched_properties = self.wikidata_wrapper._get_property(list(all_pids))
            all_properties = build_result_map(fetched_properties, list(all_pids))
        output: List[List[WikidataProperty]] = []
        for pid_list in pids:    
            properties = [all_properties[pid] for pid in pid_list if pid in all_properties]
            output.append(properties)
        return unwrap_single_result(output, is_single)
    
    async def _arun(
        self, 
        query: Union[str, List[str]], 
        top_k_results: int = 3
    ) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        """Async version of property retrieval."""
        return await asyncio.to_thread(self._run, query, top_k_results)


class WikidataKHopTriplesRetrievalTool(BaseTool):
    """Tool for retrieving k-hop Wikidata triples based on a textual query."""
    
    name: str = "Wikidata k-Hop Triples Retrieval Tool"
    description: str = (
        "A tool to retrieve k-hop Wikidata triples for entities matching a textual query. "
        "Given a query, it returns a list of Wikidata triples representing (subject, relation, object)."
    )
    wikidata_wrapper: CustomWikidataAPIWrapper = pydantic.Field(
        default_factory=lambda: CustomWikidataAPIWrapper(lang="en", top_k_results=3),
        description="An instance of CustomWikidataAPIWrapper for querying Wikidata."
    )

    def enrich_triples_tool(self, triples: List[WikiTriple], get_details: bool = False) -> List[WikiTriple]:
        """Enrich all entities and properties in triples using batch API calls.
        
        This is the centralized enrichment that was moved from wrapper methods.
        Uses EnrichmentCollector for efficient batch enrichment.
        """
        if not triples:
            return triples
        wikidata_props = DEFAULT_PROPERTIES + self.wikidata_wrapper.wikidata_props
        wikidata_props = list(set(wikidata_props))
        wikidata_props_with_labels = copy.deepcopy(self.wikidata_wrapper.wikidata_props_with_labels)
        wikidata_props_with_labels = {**PROPERTY_LABELS, **wikidata_props_with_labels}
        wikidata_wrapper = CustomWikidataAPIWrapper(
            wikidata_props=wikidata_props,
            wikidata_props_with_labels=wikidata_props_with_labels
        )
        enrichment_collector = EnrichmentCollector(wikidata_wrapper)
        enrichment_collector.collect_from_triples(triples)
        enrichment_collector.enrich_all(get_details=get_details)
        return enrichment_collector.enrich_triples(triples)

    def _run(
        self, 
        query: Union[str, List[str]], 
        is_qids: bool = False,
        k: int = 1, 
        num_entities: int = 3, 
        bidirectional: bool = False,
        prop: Optional[str] = None,
        enrich: bool = True,
        get_details: bool = False,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop triples for entities matching query with batch support.
        
        Args:
            query: Text query or QID(s) to search for
            is_qids: If True, treat query as QID(s) directly
            k: Number of hops to traverse
            num_entities: Max entities to retrieve per query
            bidirectional: If True, traverse both incoming and outgoing edges
            prop: Optional property ID to filter by
            enrich: If True, enrich entity/property labels using batch API calls
            get_details: If True, fetch full entity details (wikipedia content, etc.)
        """
        is_single, query_list = normalize_single_or_list(query)
        if is_qids:
            qids_per_query: List[List[str]] = [[qid] for qid in query_list]
        else:
            search_results = self.wikidata_wrapper._get_id(query_list)
            # _get_id returns List[List[str]] when given a list, so extract results per query
            qids_per_query: List[List[str]] = [results[:num_entities] if results else [] for results in search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids, qid_to_query_indices = flatten_and_map_ids(qids_per_query)
        if not all_qids:
            return unwrap_single_result([[] for _ in qids_per_query], is_single)
        
        if prop and prop not in self.wikidata_wrapper.wikidata_props:
            logger.warning(f"Property {prop} not in the specified wikidata_props.")
            return unwrap_single_result([[] for _ in qids_per_query], is_single)
        
        # Use batch k-hop methods for efficient retrieval
        if bidirectional:
            results = self.wikidata_wrapper._get_k_hop_bidirectional(all_qids, k=k, prop=prop)
        else:
            results = self.wikidata_wrapper._get_k_hop_outgoing(all_qids, k=k, prop=prop)

        all_triples: List[WikiTriple] = sum(results, []) if isinstance(results[0], list) else results
        unique_triples = set(all_triples)

        # Enrich entities and properties with labels (centralized enrichment)
        if enrich:
            logger.info(f"Enriching {len(unique_triples)} triples")
            all_triples = self.enrich_triples_tool(unique_triples, get_details=get_details)
            logger.info(f"Enriched {len(all_triples)} triples")
        batch_output: List[List[WikiTriple]] = [[] for _ in qids_per_query]
        for triple in all_triples:
            query_idx = qid_to_query_indices.get(triple.subject.qid, [])
            if bidirectional and isinstance(triple.object, WikidataEntity):
                query_idx.extend(qid_to_query_indices.get(triple.object.qid, []))
            for i in query_idx:
                batch_output[i].append(triple)
        return unwrap_single_result(batch_output, is_single)

    async def _arun(
        self, 
        query: Union[str, List[str]],
        is_qids: bool = False,
        k: int = 1, 
        num_entities: int = 3, 
        bidirectional: bool = False,
        prop: Optional[str] = None,
        enrich: bool = True,
        get_details: bool = False,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Async version of k-hop triples retrieval with batch support."""
        return await asyncio.to_thread(
            self._run,
            query,
            is_qids,
            k,
            num_entities,
            bidirectional,
            prop,
            enrich,
            get_details
        )


class WikidataPathFindingTool(BaseTool):
    """Tool for finding paths between two Wikidata entities using bidirectional BFS."""
    
    name: str = "Wikidata Path Finding Tool"
    description: str = (
        "A tool to find paths between two Wikidata entities. "
        "Given two entity QIDs, it returns the shortest path(s) connecting them."
    )
    wikidata_wrapper: CustomWikidataAPIWrapper = pydantic.Field(
        default_factory=lambda: CustomWikidataAPIWrapper(lang="en", top_k_results=3),
        description="An instance of CustomWikidataAPIWrapper for querying Wikidata."
    )
    
    def enrich_path(self, path: List[WikiTriple]) -> List[WikiTriple]:
        """Enrich entities and properties in the path using batch API calls."""
        if not path:
            return path
        
        wikidata_props = DEFAULT_PROPERTIES + self.wikidata_wrapper.wikidata_props
        wikidata_props = list(set(wikidata_props))
        wikidata_props_with_labels = copy.deepcopy(self.wikidata_wrapper.wikidata_props_with_labels)
        wikidata_props_with_labels = {**PROPERTY_LABELS, **wikidata_props_with_labels}
        wikidata_wrapper = CustomWikidataAPIWrapper(
            wikidata_props=wikidata_props,
            wikidata_props_with_labels=wikidata_props_with_labels
        )
        enrichment_collector = EnrichmentCollector(wikidata_wrapper)
        enrichment_collector.collect_from_triples(path)
        enrichment_collector.enrich_all(get_details=False)
        return enrichment_collector.enrich_triples(path)
    
    def _find_path_bidirectional_bfs(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 2,
        enrich: bool = True,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Find path between two entities using bidirectional BFS.
        
        Args:
            source_qid: Source entity QID
            target_qid: Target entity QID
            max_hops: Maximum number of hops to search
            enrich: If True, enrich entity/property labels in the final path
        """
        if source_qid == target_qid:
            source_entity = self.wikidata_wrapper._get_item(source_qid, get_details=False)
            if isinstance(source_entity, list):
                source_entity = source_entity[0] if source_entity else None
            if source_entity:
                return WikidataPathBetweenEntities(
                    source=source_entity,
                    target=source_entity,
                    path=[],
                    path_length=0
                )
            return None
        
        # Forward search from source
        forward_visited: Dict[str, Tuple[str, WikiTriple]] = {source_qid: (None, None)}
        forward_frontier: Set[str] = {source_qid}
        
        # Backward search from target
        backward_visited: Dict[str, Tuple[str, WikiTriple]] = {target_qid: (None, None)}
        backward_frontier: Set[str] = {target_qid}
        
        meeting_qid: Optional[str] = None
        
        for hop in range(max_hops):
            if not forward_frontier and not backward_frontier:
                break
                
            # Expand forward frontier
            if forward_frontier and (not backward_frontier or len(forward_frontier) <= len(backward_frontier)):
                new_forward_frontier: Set[str] = set()
                
                for entity_qid in forward_frontier:
                    triples = self.wikidata_wrapper._get_k_hop_bidirectional(entity_qid, k=1)
                    if isinstance(triples, list) and triples and isinstance(triples[0], list):
                        triples = triples[0]
                    
                    for triple in triples:
                        if triple.subject.qid == entity_qid and isinstance(triple.object, WikidataEntity):
                            neighbor_qid = triple.object.qid
                        elif isinstance(triple.object, WikidataEntity) and triple.object.qid == entity_qid:
                            neighbor_qid = triple.subject.qid
                        else:
                            continue
                        
                        if neighbor_qid not in forward_visited:
                            forward_visited[neighbor_qid] = (entity_qid, triple)
                            new_forward_frontier.add(neighbor_qid)
                            
                            if neighbor_qid in backward_visited:
                                meeting_qid = neighbor_qid
                                break
                    
                    if meeting_qid:
                        break
                
                forward_frontier = new_forward_frontier
                
            # Expand backward frontier
            else:
                new_backward_frontier: Set[str] = set()
                
                for entity_qid in backward_frontier:
                    triples = self.wikidata_wrapper._get_k_hop_bidirectional(entity_qid, k=1)
                    if isinstance(triples, list) and triples and isinstance(triples[0], list):
                        triples = triples[0]
                    
                    for triple in triples:
                        if triple.subject.qid == entity_qid and isinstance(triple.object, WikidataEntity):
                            neighbor_qid = triple.object.qid
                        elif isinstance(triple.object, WikidataEntity) and triple.object.qid == entity_qid:
                            neighbor_qid = triple.subject.qid
                        else:
                            continue
                        
                        if neighbor_qid not in backward_visited:
                            backward_visited[neighbor_qid] = (entity_qid, triple)
                            new_backward_frontier.add(neighbor_qid)
                            
                            if neighbor_qid in forward_visited:
                                meeting_qid = neighbor_qid
                                break
                    
                    if meeting_qid:
                        break
                
                backward_frontier = new_backward_frontier
            
            if meeting_qid:
                break
        
        if not meeting_qid:
            logger.info(f"No path found between {source_qid} and {target_qid} within {max_hops} hops")
            return None
        
        # Reconstruct path from source to meeting point
        forward_path: List[WikiTriple] = []
        current = meeting_qid
        while forward_visited[current][0] is not None:
            parent_qid, triple = forward_visited[current]
            forward_path.append(triple)
            current = parent_qid
        forward_path.reverse()
        
        # Reconstruct path from meeting point to target
        backward_path: List[WikiTriple] = []
        current = meeting_qid
        while backward_visited[current][0] is not None:
            parent_qid, triple = backward_visited[current]
            backward_path.append(triple)
            current = parent_qid
        
        # Combine paths
        full_path = forward_path + backward_path
        
        # Enrich the final path (not during search for performance)
        if enrich and full_path:
            full_path = self.enrich_path(full_path)
        
        # Get entity details
        source_entity = self.wikidata_wrapper._get_item(source_qid, get_details=False)
        target_entity = self.wikidata_wrapper._get_item(target_qid, get_details=False)
        
        if isinstance(source_entity, list):
            source_entity = source_entity[0] if source_entity else None
        if isinstance(target_entity, list):
            target_entity = target_entity[0] if target_entity else None
        
        if not source_entity or not target_entity:
            logger.warning(f"Could not retrieve entity details for source {source_qid} or target {target_qid}")
            return None
        
        return WikidataPathBetweenEntities(
            source=source_entity,
            target=target_entity,
            path=full_path,
            path_length=len(full_path)
        )
    
    def _run(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 2,
        enrich: bool = True,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Find path between two Wikidata entities.
        
        Args:
            source_qid: Source entity QID
            target_qid: Target entity QID
            max_hops: Maximum number of hops to search
            enrich: If True, enrich entity/property labels in the final path
        """
        return self._find_path_bidirectional_bfs(source_qid, target_qid, max_hops, enrich)
    
    async def _arun(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 2,
        enrich: bool = True,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Async version - delegates to sync version using asyncio.to_thread."""
        return await asyncio.to_thread(
            self._find_path_bidirectional_bfs,
            source_qid,
            target_qid,
            max_hops,
            enrich
        )

