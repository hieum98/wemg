"""Maintainable working-memory layer for reasoning.

This module exposes:
- ``parse_graph_from_text``: relation extraction helper.
- ``WorkingMemory``: local text/graph/entity memory with bidirectional synchronization.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

from wemg.llm.roles import Relation
from wemg.reasoning.generator import merge_logs, _PRUNE_TRIPLES_BATCH_SIZE
from wemg.retrieval.entity_linking import link_entities_azure, link_entities_llm
from wemg.retrieval.wikidata import (
    WikiTriple,
    WikidataClient,
    WikidataEntity,
    WikidataProperty,
    filter_entities_relevant_to_text,
)

from .interaction_memory import InteractionMemory, log_to_interaction_memory

logger = logging.getLogger(__name__)


class _GraphStore:
    """Thin wrapper around a NetworkX graph with dedup helpers."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph: nx.DiGraph = graph or nx.DiGraph()

    def merge_same_qid_nodes(self) -> None:
        groups: Dict[str, List[str]] = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            qid = getattr(data.get("data"), "id", None)
            if qid:
                groups[qid].append(node_id)
        for _, node_ids in groups.items():
            if len(node_ids) > 1:
                self._merge_nodes(node_ids[0], node_ids[1:])

    def _merge_nodes(self, canonical_id: str, duplicate_ids: List[str]) -> None:
        for dup_id in duplicate_ids:
            if not self.graph.has_node(dup_id):
                continue
            for pred in list(self.graph.predecessors(dup_id)):
                if pred == canonical_id:
                    continue
                edge = dict(self.graph.edges[pred, dup_id])
                if self.graph.has_edge(pred, canonical_id):
                    self.graph.edges[pred, canonical_id].setdefault("relation", set()).update(
                        edge.get("relation", set())
                    )
                else:
                    self.graph.add_edge(pred, canonical_id, **edge)
            for succ in list(self.graph.successors(dup_id)):
                if succ == canonical_id:
                    continue
                edge = dict(self.graph.edges[dup_id, succ])
                if self.graph.has_edge(canonical_id, succ):
                    self.graph.edges[canonical_id, succ].setdefault("relation", set()).update(
                        edge.get("relation", set())
                    )
                else:
                    self.graph.add_edge(canonical_id, succ, **edge)
            self.graph.remove_node(dup_id)


async def parse_graph_from_text(
    client,
    text: str,
    interaction_memory: Optional[InteractionMemory] = None,
    known_entities: Optional[List[Any]] = None,
):
    """Parse text into relation triples extracted by the RELATION_EXTRACTOR role."""
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
        return relation_triples.relations, re_log
    logger.error("Failed relation extraction from text; using empty relation list.")
    return [], re_log


async def _aprune_triple_strings_llm(
    client,
    question: str,
    triple_strings: List[str],
    interaction_memory=None,
    batch_size: int = _PRUNE_TRIPLES_BATCH_SIZE,
) -> Set[int]:
    """Stage-B LLM pruning. Returns set of indices (into triple_strings) to keep."""
    from wemg.llm.roles import TRIPLE_PRUNER, TriplePruneInput, execute_role

    if not triple_strings:
        return set(range(len(triple_strings)))
    chunks = [triple_strings[i : i + batch_size] for i in range(0, len(triple_strings), batch_size)]
    offsets = [i * batch_size for i in range(len(chunks))]
    inputs = [TriplePruneInput(question=question, triples=chunk) for chunk in chunks]
    responses, _ = await execute_role(
        client=client,
        role=TRIPLE_PRUNER,
        input_data=inputs,
        interaction_memory=interaction_memory,
        n=1,
    )
    kept: Set[int] = set()
    for chunk, offset, out_list in zip(chunks, offsets, responses):
        out = out_list[0] if out_list else None
        if not out or not hasattr(out, "keep_indices"):
            kept.update(range(offset, offset + len(chunk)))
        else:
            kept.update(offset + i for i in out.keep_indices if i < len(chunk))
    return kept


async def _aprune_graph_edges(
    client,
    question: str,
    graph: nx.DiGraph,
    new_edges: Set[Tuple],
    interaction_memory=None,
) -> None:
    """Remove irrelevant new edges from *graph* in-place using TRIPLE_PRUNER (Stage B).

    Only the edges in *new_edges* are evaluated — previously-pruned edges are not re-examined.
    """
    if not new_edges:
        return
    flat: List[Tuple] = []  # (src, tgt, rel_str, triple_str)
    for src, tgt in new_edges:
        if not graph.has_edge(src, tgt):
            continue
        data = graph.edges[src, tgt]
        src_label = str(getattr(graph.nodes[src].get("data"), "name", src))
        tgt_label = str(getattr(graph.nodes[tgt].get("data"), "name", tgt))
        for rel in data.get("relation") or set():
            rel_str = rel.label if hasattr(rel, "label") else str(rel)
            flat.append((src, tgt, rel_str, f"Subject: {src_label}\nRelation: {rel_str}\nObject: {tgt_label}"))
    if not flat:
        return
    kept_idx = await _aprune_triple_strings_llm(client, question, [t[3] for t in flat], interaction_memory)
    # Collect edges whose ALL relations were pruned
    edge_kept_rels: Dict[Tuple, Set[str]] = {}
    for i, (src, tgt, rel_str, _) in enumerate(flat):
        key = (src, tgt)
        if i in kept_idx:
            edge_kept_rels.setdefault(key, set()).add(rel_str)
    for src, tgt in new_edges:
        if not graph.has_edge(src, tgt):
            continue
        kept_rels = edge_kept_rels.get((src, tgt))
        if not kept_rels:
            graph.remove_edge(src, tgt)
        else:
            graph.edges[src, tgt]["relation"] = kept_rels


class _MemoryFormatter:
    @staticmethod
    def format_item(content: str, provenance) -> str:
        from wemg.llm.roles import SourceType

        tag = {
            SourceType.SYSTEM_PREDICTION: "[System Prediction]",
            SourceType.RETRIEVAL: "[Retrieval]",
        }.get(provenance, "")
        normalized = content.strip()
        if normalized.startswith(("[System Prediction]", "[Retrieval]")):
            return normalized
        return f"{tag}: {normalized}" if tag else normalized

    @staticmethod
    def format_lines(items: Sequence[str]) -> str:
        return "\n".join(f"- {i.strip()}" for i in items if i and i.strip())


class WorkingMemory:
    """Local working memory used by CoT and MCTS branch execution.

    Text↔graph sync is bidirectional:
    - text → graph: entity linking + relation extraction
    - graph → text: textualize graph triples → add as RETRIEVAL items

    Entity linking is always redone on all current text items each sync call.
    Idempotency is guaranteed by graph deduplication.
    """

    def __init__(
        self,
        textual_memory: Optional[List[str]] = None,
        graph_memory: Optional[nx.DiGraph] = None,
        max_textual_memory_tokens: int = 16384,
        wikidata_client: Optional[WikidataClient] = None,
        annotate_steps: bool = False,
    ):
        self.textual_memory: List[str] = list(textual_memory or [])
        self.graph_store = _GraphStore(graph_memory)
        self.graph_memory = self.graph_store.graph
        self.max_textual_memory_tokens = max_textual_memory_tokens
        self.annotate_steps = annotate_steps
        self.entity_dict: Dict[str, WikidataEntity] = {}

        self._wikidata_client = wikidata_client
        self._qid_to_node_id: Dict[str, str] = {}

    @staticmethod
    def format_memory_item(content: str, provenance) -> str:
        return _MemoryFormatter.format_item(content, provenance)

    def add_textual_memory(self, text: str, source=None, hop_depth: int = None) -> None:
        from wemg.llm.roles import SourceType

        if source is None:
            source = SourceType.SYSTEM_PREDICTION
        formatted = _MemoryFormatter.format_item(text, source)
        if hop_depth is not None:
            formatted = f"[hop={hop_depth}] {formatted}"
        if formatted not in self.textual_memory:
            self.textual_memory.append(formatted)

    def format_textual_memory(self) -> str:
        return _MemoryFormatter.format_lines(self.textual_memory)

    async def _arun_consolidation(self, client, question: str, raw_memory: str, interaction_memory=None):
        from wemg.llm.roles import MEMORY_CONSOLIDATOR, MemoryConsolidationInput, MemoryConsolidationOutput, MemoryItem, SourceType, execute_role

        responses, log = await execute_role(
            client=client,
            role=MEMORY_CONSOLIDATOR,
            input_data=MemoryConsolidationInput(question=question, memory=raw_memory),
            interaction_memory=interaction_memory,
            n=1,
            max_tokens=self.max_textual_memory_tokens,
        )
        if not responses:
            return MemoryConsolidationOutput(
                consolidated_memory=[MemoryItem(content=raw_memory, provenance=SourceType.SYSTEM_PREDICTION.value)]
            ), log
        return responses[0], log

    async def _aconsolidate_textual_memory(self, client, question: str, interaction_memory=None) -> None:
        from wemg.llm.roles import SourceType

        raw_memory = self.format_textual_memory()
        if not raw_memory:
            return
        output, log = await self._arun_consolidation(client, question, raw_memory, interaction_memory)
        self.textual_memory = []
        for item in output.consolidated_memory:
            prov = SourceType.SYSTEM_PREDICTION
            if item.provenance == SourceType.RETRIEVAL.value:
                prov = SourceType.RETRIEVAL
            self.add_textual_memory(item.content, source=prov, hop_depth=item.hop_depth)
        log_to_interaction_memory(interaction_memory, log)

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
        if not text.strip() or self._wikidata_client is None:
            return [], {}
        if entity_linking_method == "azure":
            entities, _, link_log = await link_entities_azure(
                text,
                self._wikidata_client,
                endpoint=azure_endpoint,
                key=azure_key,
            )
        else:
            entities, _, link_log = await link_entities_llm(
                client,
                text,
                self._wikidata_client,
                top_k_entities=top_k_entities,
                interaction_memory=interaction_memory,
                known_entities=known_entities,
                reranker=reranker,
            )
        resolved = [e for e in entities if isinstance(e, WikidataEntity) and e.qid]
        return resolved, (link_log if isinstance(link_log, dict) else {})

    @staticmethod
    def _entity_link_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_linking_method": kwargs.get("entity_linking_method", "llm"),
            "top_k_entities": kwargs.get("top_k_entities", 1),
            "reranker": kwargs.get("reranker"),
            "azure_endpoint": kwargs.get("azure_endpoint"),
            "azure_key": kwargs.get("azure_key"),
        }

    def _enhance_relations(self, relations: List[Relation]) -> List[Union[WikiTriple, Relation]]:
        to_lookup: List[str] = []
        for rel in relations:
            if rel.subject_id and rel.subject_id not in self.entity_dict:
                to_lookup.append(rel.subject_id)
            elif not rel.subject_id and rel.subject:
                to_lookup.append(rel.subject)
            if rel.object_id and rel.object_id not in self.entity_dict:
                to_lookup.append(rel.object_id)
            elif not rel.object_id and rel.object:
                to_lookup.append(rel.object)
        lookup = list(set(to_lookup))
        resolved: Dict[str, str] = {}
        if lookup and self._wikidata_client is not None:
            results = self._wikidata_client.search_entities(lookup, num_results=1, get_details=False)
            for query, result in zip(lookup, results):
                first = result[0] if isinstance(result, list) and result else None
                if isinstance(first, WikidataEntity) and first.qid:
                    self.entity_dict[first.qid] = first
                    resolved[query] = first.qid

        triples: List[Union[WikiTriple, Relation]] = []
        for rel in relations:
            if not rel.relation:
                continue
            sub_key = rel.subject_id or resolved.get(rel.subject or "", "")
            obj_key = rel.object_id or resolved.get(rel.object or "", "")
            subject = self.entity_dict.get(sub_key)
            object_ = self.entity_dict.get(obj_key)
            if subject and object_:
                triples.append(
                    WikiTriple(
                        subject=subject,
                        relation=WikidataProperty(pid="", label=str(rel.relation), description=None),
                        object=object_,
                    )
                )
            elif rel.subject and rel.object:
                # Fall back to string-based relation when QID resolution fails
                triples.append(rel)
        return triples

    def _textualize_graph_to_text(self) -> List[str]:
        """Return graph triples as plain strings for adding to textual memory."""
        from wemg.utils.graph import textualize_graph

        items = []
        components = list(nx.weakly_connected_components(self.graph_memory))
        for comp in components:
            triples, _ = textualize_graph(comp, self.graph_memory, method="dfs")
            items.extend(triples)
        return items

    async def synchronize_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        """Bidirectional text↔graph sync.

        Step 1: Consolidate textual memory.
        Step 2: text → graph (entity link + relation extraction + prune + dedup).
        Step 3: graph → text (textualize graph triples → add as RETRIEVAL items).
        Step 4: Consolidate again if new graph triples were added (text now enriched).
        """
        if self.textual_memory:
            # Step 1: Consolidate
            await self._aconsolidate_textual_memory(client, question=question, interaction_memory=interaction_memory)

        # Step 2: text → graph
        if self.textual_memory:
            text_blob = self.format_textual_memory()
            known = filter_entities_relevant_to_text(
                [e for e in self.entity_dict.values() if isinstance(e, WikidataEntity)],
                text_blob,
            )
            link_kwargs = self._entity_link_kwargs(kwargs)
            link_result, parse_result = await asyncio.gather(
                self._link_entities_async(client, text_blob, known, interaction_memory, **link_kwargs),
                parse_graph_from_text(
                    client,
                    text_blob,
                    interaction_memory=interaction_memory,
                    known_entities=list(known) if known else None,
                ),
            )
            linked_entities, link_log = link_result
            relations, parse_log = parse_result
            for entity in linked_entities:
                self.entity_dict[entity.qid] = entity
            edges_before = set(self.graph_memory.edges())
            for triple in self._enhance_relations(relations):
                self.add_edge_to_graph_memory(triple)
            log_to_interaction_memory(interaction_memory, merge_logs(link_log, parse_log))
            new_edges = set(self.graph_memory.edges()) - edges_before
            if new_edges:
                await _aprune_graph_edges(client, question, self.graph_memory, new_edges, interaction_memory)

        if self.graph_memory.number_of_nodes() > 0:
            self.deduplicate_graph()

        # Step 3: graph → text (bidirectional enrichment)
        from wemg.llm.roles import SourceType
        n_before = len(self.textual_memory)
        for item in self._textualize_graph_to_text():
            self.add_textual_memory(item, source=SourceType.SYSTEM_PREDICTION)

        # Step 4: Consolidate again if graph triples enriched the text
        if len(self.textual_memory) > n_before:
            await self._aconsolidate_textual_memory(client, question=question, interaction_memory=interaction_memory)

    def add_node_to_graph_memory(self, node):
        from wemg.llm.roles import Entity as OpenIEEntity
        from wemg.utils.graph import get_node_id

        entity: Optional[WikidataEntity] = None
        node_data = None
        if isinstance(node, OpenIEEntity):
            if node.id and node.id in self.entity_dict:
                entity = self.entity_dict[node.id]
            elif node.name and self._wikidata_client is not None:
                results = self._wikidata_client.search_entities(node.id or node.name, num_results=1, get_details=False)
                entity = results[0] if results else None
            if entity is None:
                if node.name:
                    # Fall back to a string-only node (no Wikidata QID).
                    # Use the name directly as the node ID so it's human-readable
                    # and can be merged later if the entity is resolved to a QID.
                    node_data = OpenIEEntity(id=None, name=node.name, description=None)
                    node_id = node.name
                    if not self.graph_memory.has_node(node_id):
                        self.graph_memory.add_node(node_id, data=node_data)
                    return node_data
                return None
        elif isinstance(node, WikidataEntity):
            entity = node
        else:
            raise ValueError(f"Invalid node type: {type(node)}")

        if entity.qid and entity.label:
            node_data = OpenIEEntity(id=entity.qid, name=entity.label, description=entity.description)
            self.entity_dict[entity.qid] = entity
        elif entity.label:
            node_data = OpenIEEntity(id=None, name=entity.label, description=entity.description)
        else:
            return None
        node_id = get_node_id(node_data)
        if not self.graph_memory.has_node(node_id):
            self.graph_memory.add_node(node_id, data=node_data)
        if entity.qid:
            self._qid_to_node_id[entity.qid] = node_id
        return node_data

    def add_edge_to_graph_memory(
        self,
        triple,
        *,
        source_step: Optional[int] = None,
        timestamp: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> None:
        from wemg.llm.roles import Entity as OpenIEEntity, Relation as OpenIERelation
        from wemg.utils.graph import get_node_id

        relation_label = None
        if isinstance(triple, OpenIERelation):
            subject = self.add_node_to_graph_memory(OpenIEEntity(id=triple.subject_id, name=triple.subject, description=None))
            object_ = self.add_node_to_graph_memory(OpenIEEntity(id=triple.object_id, name=triple.object, description=None))
            relation_label = str(triple.relation) if triple.relation else None
        elif isinstance(triple, WikiTriple):
            subject = self.add_node_to_graph_memory(triple.subject) if isinstance(triple.subject, WikidataEntity) else None
            if isinstance(triple.object, WikidataEntity):
                object_ = self.add_node_to_graph_memory(triple.object)
            elif triple.object is not None:
                # Scalar literal (date, number, string) — store as a string-only node.
                # Node key = scalar_str; get_node_id will recover it via str(entity) -> entity.name.
                scalar_str = str(triple.object).strip()
                if scalar_str:
                    node_data = OpenIEEntity(id=None, name=scalar_str, description=None)
                    if not self.graph_memory.has_node(scalar_str):
                        self.graph_memory.add_node(scalar_str, data=node_data)
                    object_ = node_data
                else:
                    object_ = None
            else:
                object_ = None
            if isinstance(triple.relation, WikidataProperty):
                relation_label = triple.relation.label
        else:
            raise ValueError(f"Invalid triple type: {type(triple)}")
        if subject is None or object_ is None or not relation_label:
            return
        subject_id = get_node_id(subject)
        object_id = get_node_id(object_)

        provenance: Dict[str, Any] = {}
        if source_step is not None:
            provenance["source_step"] = source_step
        if timestamp is not None:
            provenance["timestamp"] = timestamp
        if reward is not None:
            provenance["reward"] = reward

        if not self.graph_memory.has_edge(subject_id, object_id):
            payload: Dict[str, Any] = {"relation": {relation_label}}
            if provenance:
                payload["provenance"] = provenance
            self.graph_memory.add_edge(subject_id, object_id, **payload)
            return
        edge = self.graph_memory.edges[subject_id, object_id]
        edge.setdefault("relation", set()).add(relation_label)
        if provenance:
            existing = edge.get("provenance", {})
            merged: Dict[str, Any] = dict(existing)
            new_step = provenance.get("source_step")
            old_step = existing.get("source_step")
            if new_step is not None and (old_step is None or new_step < old_step):
                merged["source_step"] = new_step
            new_ts = provenance.get("timestamp")
            old_ts = existing.get("timestamp")
            if new_ts is not None and (old_ts is None or new_ts < old_ts):
                merged["timestamp"] = new_ts
            if provenance.get("reward") is not None:
                merged["reward"] = provenance["reward"]
            edge["provenance"] = merged

    def _remove_isolated_graph_nodes(self) -> None:
        isolated_nodes = [
            node_id
            for node_id in list(self.graph_memory.nodes())
            if self.graph_memory.in_degree(node_id) == 0 and self.graph_memory.out_degree(node_id) == 0
        ]
        if isolated_nodes:
            self.graph_memory.remove_nodes_from(isolated_nodes)

    def _merge_string_nodes_with_qid_nodes(self) -> None:
        """Merge string-only nodes (id=None) into QID nodes that share the same label."""
        label_to_qid_node: Dict[str, str] = {}
        for node_id, data in self.graph_memory.nodes(data=True):
            node_data = data.get("data")
            qid = getattr(node_data, "id", None)
            label = getattr(node_data, "name", None)
            if qid and label:
                label_to_qid_node[label.lower()] = node_id

        to_merge: List[Tuple[str, str]] = []  # (string_node_id, qid_node_id)
        for node_id, data in self.graph_memory.nodes(data=True):
            node_data = data.get("data")
            if getattr(node_data, "id", None) is not None:
                continue
            label = getattr(node_data, "name", None)
            if label and label.lower() in label_to_qid_node:
                canonical = label_to_qid_node[label.lower()]
                if canonical != node_id:
                    to_merge.append((canonical, node_id))

        for canonical_id, dup_id in to_merge:
            self.graph_store._merge_nodes(canonical_id, [dup_id])

    def deduplicate_graph(self) -> None:
        self.graph_store.merge_same_qid_nodes()
        self._merge_string_nodes_with_qid_nodes()
        self._remove_isolated_graph_nodes()
        # Rebuild cache: merging/removal may change node IDs.
        self._qid_to_node_id = {}
        for node_id, data in self.graph_memory.nodes(data=True):
            qid = getattr(data.get("data"), "id", None)
            if qid:
                self._qid_to_node_id[qid] = node_id

    def format_graph_memory(self) -> str:
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        components = list(nx.weakly_connected_components(self.graph_memory))
        sections = []
        for comp in components:
            triples, _ = textualize_graph(comp, self.graph_memory, method="dfs", annotate_steps=self.annotate_steps)
            triples = [_MemoryFormatter.format_item(t, SourceType.SYSTEM_PREDICTION) for t in triples]
            sections.append(_MemoryFormatter.format_lines(triples))
        return "\n\n".join(f"**Information {idx}**\n{text}" for idx, text in enumerate(sections, 1))

    def format_combined_memory(self) -> str:
        """Format both textual and graph memory into a single string."""
        textual = self.format_textual_memory()
        graph = self.format_graph_memory()
        parts = [p for p in (textual, graph) if p]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Graph intelligence helpers
    # ------------------------------------------------------------------

    def get_known_entity_qids(self) -> Set[str]:
        """Return QIDs of all entities currently represented in the graph."""
        qids: Set[str] = set()
        for _, data in self.graph_memory.nodes(data=True):
            qid = getattr(data.get("data"), "id", None)
            if qid:
                qids.add(qid)
        return qids

    def get_well_connected_qids(self, min_degree: int = 4) -> Set[str]:
        """Return QIDs of graph entities whose degree >= *min_degree*."""
        well: Set[str] = set()
        for node_id, data in self.graph_memory.nodes(data=True):
            degree = self.graph_memory.in_degree(node_id) + self.graph_memory.out_degree(node_id)
            if degree >= min_degree:
                qid = getattr(data.get("data"), "id", None)
                if qid:
                    well.add(qid)
        return well

    def get_underexplored_neighbor_qids(
        self,
        seed_qids: Sequence[str],
        max_degree: int = 2,
        max_results: int = 5,
    ) -> List[str]:
        """Return QIDs of graph neighbours of *seed_qids* with low degree.

        These represent entities we've seen but haven't fully explored yet.
        """
        node_for_qid: Dict[str, str] = {}
        for node_id, data in self.graph_memory.nodes(data=True):
            qid = getattr(data.get("data"), "id", None)
            if qid:
                node_for_qid[qid] = node_id

        seed_nodes = {node_for_qid[q] for q in seed_qids if q in node_for_qid}
        if not seed_nodes:
            return []

        neighbor_qids: List[str] = []
        seen: Set[str] = set(seed_qids)
        for seed in seed_nodes:
            neighbours = set(self.graph_memory.successors(seed)) | set(
                self.graph_memory.predecessors(seed)
            )
            for nbr in neighbours:
                degree = self.graph_memory.in_degree(nbr) + self.graph_memory.out_degree(nbr)
                if degree > max_degree:
                    continue
                qid = getattr(self.graph_memory.nodes[nbr].get("data"), "id", None)
                if qid and qid not in seen:
                    seen.add(qid)
                    neighbor_qids.append(qid)
                    if len(neighbor_qids) >= max_results:
                        return neighbor_qids
        return neighbor_qids

    def is_triple_known(self, subject_qid: str, relation_label: str, object_qid: str) -> bool:
        """Return True if the graph already contains this (s, r, o) edge."""
        s_node = self._qid_to_node_id.get(subject_qid)
        o_node = self._qid_to_node_id.get(object_qid)
        if s_node is None or o_node is None:
            return False
        if not self.graph_memory.has_edge(s_node, o_node):
            return False
        return relation_label in self.graph_memory.edges[s_node, o_node].get("relation", set())


__all__ = ["parse_graph_from_text", "WorkingMemory"]
