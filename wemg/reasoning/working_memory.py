"""Maintainable working-memory layer for reasoning.

This module exposes:
- ``parse_graph_from_text``: relation extraction helper.
- ``MemoryDelta``: branch delta object for MCTS promotion.
- ``GlobalKnowledge``: reward-gated shared knowledge for MCTS.
- ``WorkingMemory``: local text/graph/entity memory with synchronization.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from wemg.llm.roles import Relation
from wemg.reasoning.generator import merge_logs
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


@dataclass
class MemoryDelta:
    """Items discovered by a branch since its snapshot."""

    new_textual_items: List[str] = field(default_factory=list)
    new_triples: List[WikiTriple] = field(default_factory=list)
    new_entities: Dict[str, WikidataEntity] = field(default_factory=dict)


@dataclass
class _SnapshotMarker:
    text_len: int
    entity_keys: Set[str]
    edge_set: Set[Tuple[str, str]]


@dataclass
class _TextStore:
    items: List[str] = field(default_factory=list)
    processed_texts: Set[str] = field(default_factory=set)
    dirty: bool = False
    items_since_sync: int = 0


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


_PRUNE_TRIPLES_BATCH_SIZE = 16


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
    # Remove nodes that became isolated only because their edges were just pruned
    for node in [n for n in list(graph.nodes()) if graph.degree(n) == 0]:
        graph.remove_node(node)


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


class GlobalKnowledge:
    """Shared append-only knowledge store used by MCTS branches."""

    def __init__(
        self,
        wikidata_client: Optional[WikidataClient] = None,
        max_textual_memory_tokens: int = 16384,
    ):
        self.confirmed_facts: List[str] = []
        self.graph_store = _GraphStore()
        self.graph = self.graph_store.graph
        self.entity_dict: Dict[str, WikidataEntity] = {}

        self._wikidata_client = wikidata_client
        self._max_textual_memory_tokens = max_textual_memory_tokens

        self._items_since_consolidation = 0
        self._confirmed_facts_set: Set[str] = set()
        self._processed_texts: Set[str] = set()

    def absorb(self, delta: MemoryDelta, reward: float, min_reward: float = 0.0) -> bool:
        """Promote branch delta into global store only if reward >= min_reward.

        Returns True if anything was actually absorbed.
        """
        if reward < min_reward:
            return False
        absorbed = False
        for text in delta.new_textual_items:
            if text not in self._confirmed_facts_set:
                self.confirmed_facts.append(text)
                self._confirmed_facts_set.add(text)
                self._items_since_consolidation += 1
                absorbed = True
        self.entity_dict.update(delta.new_entities)
        for triple in delta.new_triples:
            self._add_wikitriple(triple, reward=reward)
            absorbed = True
        return absorbed

    def _add_wikitriple(self, triple: WikiTriple, reward: Optional[float]) -> None:
        from wemg.llm.roles import Entity as OpenIEEntity
        from wemg.utils.graph import get_node_id

        if not isinstance(triple.subject, WikidataEntity) or not isinstance(triple.object, WikidataEntity):
            return
        if not triple.subject.qid or not triple.object.qid:
            return
        if not triple.subject.label or not triple.object.label:
            return
        relation_label = triple.relation.label if hasattr(triple.relation, "label") else str(triple.relation)
        if not relation_label:
            return

        subject_data = OpenIEEntity(
            id=triple.subject.qid, name=triple.subject.label, description=triple.subject.description
        )
        object_data = OpenIEEntity(
            id=triple.object.qid, name=triple.object.label, description=triple.object.description
        )
        subject_id = get_node_id(subject_data)
        object_id = get_node_id(object_data)
        if not self.graph.has_node(subject_id):
            self.graph.add_node(subject_id, data=subject_data)
        if not self.graph.has_node(object_id):
            self.graph.add_node(object_id, data=object_data)

        provenance: Dict[str, Any] = {"timestamp": time.time()}
        if reward is not None:
            provenance["reward"] = reward
        if not self.graph.has_edge(subject_id, object_id):
            self.graph.add_edge(subject_id, object_id, relation={relation_label}, provenance=provenance)
        else:
            edge = self.graph.edges[subject_id, object_id]
            edge.setdefault("relation", set()).add(relation_label)
            edge["provenance"] = provenance
        self.entity_dict[triple.subject.qid] = triple.subject
        self.entity_dict[triple.object.qid] = triple.object

    async def aconsolidate_if_needed(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        if self._items_since_consolidation > 0 and self.confirmed_facts:
            await self._aconsolidate_text(client, question, interaction_memory)
            self._items_since_consolidation = 0
        edges_before = set(self.graph.edges())
        await self._aextract_new_text_to_graph(client, interaction_memory, **kwargs)
        self.graph_store.merge_same_qid_nodes()
        new_edges = set(self.graph.edges()) - edges_before
        if new_edges:
            await _aprune_graph_edges(client, question, self.graph, new_edges, interaction_memory)

    async def afinalize(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        """Unconditional final consolidation + graph dedup pass."""
        if self.confirmed_facts:
            await self._aconsolidate_text(client, question, interaction_memory)
        edges_before = set(self.graph.edges())
        await self._aextract_new_text_to_graph(client, interaction_memory, **kwargs)
        self.graph_store.merge_same_qid_nodes()
        new_edges = set(self.graph.edges()) - edges_before
        if new_edges:
            await _aprune_graph_edges(client, question, self.graph, new_edges, interaction_memory)
        self._items_since_consolidation = 0

    async def _aconsolidate_text(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
    ) -> None:
        from wemg.llm.roles import MEMORY_CONSOLIDATOR, MemoryConsolidationInput, SourceType, execute_role

        raw_memory = _MemoryFormatter.format_lines(self.confirmed_facts)
        if not raw_memory:
            return
        responses, log = await execute_role(
            client=client,
            role=MEMORY_CONSOLIDATOR,
            input_data=MemoryConsolidationInput(question=question, memory=raw_memory),
            interaction_memory=interaction_memory,
            n=1,
            max_tokens=self._max_textual_memory_tokens,
        )
        log_to_interaction_memory(interaction_memory, log)
        if not responses:
            return
        self.confirmed_facts = []
        self._confirmed_facts_set = set()
        for item in responses[0].consolidated_memory:
            prov = SourceType.SYSTEM_PREDICTION
            if item.provenance == SourceType.RETRIEVAL.value:
                prov = SourceType.RETRIEVAL
            formatted = _MemoryFormatter.format_item(item.content, prov)
            if formatted not in self._confirmed_facts_set:
                self.confirmed_facts.append(formatted)
                self._confirmed_facts_set.add(formatted)

    async def _aextract_new_text_to_graph(
        self,
        client,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        del kwargs
        unprocessed = [text for text in self.confirmed_facts if text not in self._processed_texts]
        if not unprocessed:
            return
        text_blob = _MemoryFormatter.format_lines(unprocessed)
        known_entities = filter_entities_relevant_to_text(list(self.entity_dict.values()), text_blob)
        relations, parse_log = await parse_graph_from_text(
            client, text_blob, interaction_memory=interaction_memory, known_entities=known_entities
        )
        log_to_interaction_memory(interaction_memory, parse_log)
        for relation in relations:
            if hasattr(relation, "subject_id") and hasattr(relation, "object_id"):
                self._add_relation(relation)
        self._processed_texts.update(unprocessed)

    def _add_relation(self, relation: Relation) -> None:
        from wemg.llm.roles import Entity as OpenIEEntity
        from wemg.utils.graph import get_node_id

        def _resolve_entity(key: Optional[str]) -> Optional[WikidataEntity]:
            if not key:
                return None
            entity = self.entity_dict.get(key)
            if entity is None and self._wikidata_client:
                results = self._wikidata_client.search_entities(key, num_results=1, get_details=False)
                if results and isinstance(results[0], WikidataEntity) and results[0].qid:
                    entity = results[0]
                    self.entity_dict[entity.qid] = entity
            if entity is None or not entity.qid or not entity.label:
                return None
            return entity

        rel_label = str(relation.relation) if relation.relation else None
        if not rel_label:
            return

        subject = _resolve_entity(relation.subject_id or relation.subject)
        if subject is None:
            return

        subject_data = OpenIEEntity(id=subject.qid, name=subject.label, description=subject.description)

        object_ = _resolve_entity(relation.object_id or relation.object)
        if object_ is not None:
            object_data = OpenIEEntity(id=object_.qid, name=object_.label, description=object_.description)
        elif relation.object:
            # Keep non-entity literals (e.g., dates/numbers) as object nodes.
            object_data = OpenIEEntity(id=relation.object_id, name=str(relation.object), description=None)
        else:
            return

        subject_id = get_node_id(subject_data)
        object_id = get_node_id(object_data)
        if not self.graph.has_node(subject_id):
            self.graph.add_node(subject_id, data=subject_data)
        if not self.graph.has_node(object_id):
            self.graph.add_node(object_id, data=object_data)
        if not self.graph.has_edge(subject_id, object_id):
            self.graph.add_edge(subject_id, object_id, relation={rel_label})
        else:
            self.graph.edges[subject_id, object_id].setdefault("relation", set()).add(rel_label)

    def format_textual_memory(self) -> str:
        return _MemoryFormatter.format_lines(self.confirmed_facts)

    def format_graph_memory(self) -> str:
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        components = list(nx.weakly_connected_components(self.graph))
        sections = []
        for comp in components:
            triples, _ = textualize_graph(comp, self.graph, method="dfs")
            triples = [_MemoryFormatter.format_item(t, SourceType.SYSTEM_PREDICTION) for t in triples]
            sections.append(_MemoryFormatter.format_lines(triples))
        return "\n\n".join(f"**Information {idx}**\n{text}" for idx, text in enumerate(sections, 1))

    def deduplicate_graph(self) -> None:
        self.graph_store.merge_same_qid_nodes()


class WorkingMemory:
    """Local working memory used by CoT and MCTS branch execution."""

    def __init__(
        self,
        textual_memory: Optional[List[str]] = None,
        graph_memory: Optional[nx.DiGraph] = None,
        max_textual_memory_tokens: int = 16384,
        wikidata_client: Optional[WikidataClient] = None,
        global_knowledge: Optional[GlobalKnowledge] = None,
    ):
        self.text_store = _TextStore(items=list(textual_memory or []))
        self.graph_store = _GraphStore(graph_memory)
        self.graph_memory = self.graph_store.graph
        self.max_textual_memory_tokens = max_textual_memory_tokens
        self.entity_dict: Dict[str, WikidataEntity] = {}
        self.global_knowledge = global_knowledge

        self._wikidata_client = wikidata_client
        self._qid_to_node_id: Dict[str, str] = {}
        self._absorbed_global_facts: Set[str] = set()
        self._snapshot = _SnapshotMarker(
            text_len=len(self.text_store.items),
            entity_keys=set(self.entity_dict.keys()),
            edge_set=set(self.graph_memory.edges()),
        )

    @property
    def textual_memory(self) -> List[str]:
        return self.text_store.items

    @textual_memory.setter
    def textual_memory(self, values: List[str]) -> None:
        self.text_store.items = list(values)

    @staticmethod
    def format_memory_item(content: str, provenance) -> str:
        return _MemoryFormatter.format_item(content, provenance)

    def snapshot(self) -> "WorkingMemory":
        wm = WorkingMemory(
            textual_memory=list(self.text_store.items),
            graph_memory=self.graph_memory.copy(),
            max_textual_memory_tokens=self.max_textual_memory_tokens,
            wikidata_client=self._wikidata_client,
            global_knowledge=self.global_knowledge,
        )
        wm.entity_dict = dict(self.entity_dict)
        wm._qid_to_node_id = dict(self._qid_to_node_id)
        wm.text_store.processed_texts = set(self.text_store.processed_texts)
        wm._absorbed_global_facts = set(self._absorbed_global_facts)
        wm._snapshot = _SnapshotMarker(
            text_len=len(wm.text_store.items),
            entity_keys=set(wm.entity_dict.keys()),
            edge_set=set(wm.graph_memory.edges()),
        )
        return wm

    def get_delta(self) -> MemoryDelta:
        new_text = self.text_store.items[self._snapshot.text_len :]
        new_entities = {k: v for k, v in self.entity_dict.items() if k not in self._snapshot.entity_keys}
        new_triples: List[WikiTriple] = []
        for source, target in self.graph_memory.edges():
            if (source, target) in self._snapshot.edge_set:
                continue
            source_data = self.graph_memory.nodes[source].get("data")
            target_data = self.graph_memory.nodes[target].get("data")
            rel_labels = self.graph_memory.edges[source, target].get("relation", set())
            subject = self.entity_dict.get(getattr(source_data, "id", None))
            object_ = self.entity_dict.get(getattr(target_data, "id", None))
            if not subject or not object_:
                continue
            for rel_label in rel_labels:
                new_triples.append(
                    WikiTriple(
                        subject=subject,
                        relation=WikidataProperty(pid="", label=rel_label, description=None),
                        object=object_,
                    )
                )
        return MemoryDelta(
            new_textual_items=new_text,
            new_triples=new_triples,
            new_entities=new_entities,
        )

    def add_textual_memory(self, text: str, source=None) -> None:
        from wemg.llm.roles import SourceType

        if source is None:
            source = SourceType.SYSTEM_PREDICTION
        formatted = _MemoryFormatter.format_item(text, source)
        if formatted not in self.text_store.items:
            self.text_store.items.append(formatted)
            self.text_store.dirty = True
            self.text_store.items_since_sync += 1

    def format_textual_memory(self) -> str:
        merged = list(self.text_store.items)
        if self.global_knowledge is not None:
            for fact in self.global_knowledge.confirmed_facts:
                if fact not in merged and fact not in self._absorbed_global_facts:
                    merged.insert(0, fact)
        return _MemoryFormatter.format_lines(merged)

    async def _arun_consolidation(self, client, question: str, raw_memory: str, interaction_memory=None):
        from wemg.llm.roles import MEMORY_CONSOLIDATOR, MemoryConsolidationInput, MemoryConsolidationOutput, MemoryItem, execute_role

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
                consolidated_memory=[MemoryItem(content=raw_memory, provenance="system_prediction")]
            ), log
        return responses[0], log

    async def _aconsolidate_textual_memory(self, client, question: str, interaction_memory=None) -> None:
        from wemg.llm.roles import SourceType

        if self.global_knowledge is not None:
            self._absorbed_global_facts.update(self.global_knowledge.confirmed_facts)
        output, log = await self._arun_consolidation(client, question, self.format_textual_memory(), interaction_memory)
        self.text_store.items = []
        # Don't reset processed_texts — content-based tracking survives consolidation.
        for item in output.consolidated_memory:
            prov = SourceType.SYSTEM_PREDICTION
            if item.provenance == SourceType.RETRIEVAL.value:
                prov = SourceType.RETRIEVAL
            self.add_textual_memory(item.content, source=prov)
        log_to_interaction_memory(interaction_memory, log)

    def should_synchronize(self) -> bool:
        return self.text_store.dirty

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

    def _enhance_relations(self, relations: List[Relation]) -> List[WikiTriple]:
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

        triples: List[WikiTriple] = []
        for rel in relations:
            sub_key = rel.subject_id or resolved.get(rel.subject or "", "")
            obj_key = rel.object_id or resolved.get(rel.object or "", "")
            subject = self.entity_dict.get(sub_key)
            object_ = self.entity_dict.get(obj_key)
            if not subject or not object_ or not rel.relation:
                continue
            triples.append(
                WikiTriple(
                    subject=subject,
                    relation=WikidataProperty(pid="", label=str(rel.relation), description=None),
                    object=object_,
                )
            )
        return triples

    async def asynchronize_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        t_sync = time.perf_counter()
        if not self.text_store.dirty:
            logger.info(
                "PROFSTEP wm stage=skip_sync reason=no_new_items question=%s",
                question[:120].replace("\n", " "),
            )
            return

        logger.info(
            "PROFSTEP wm stage=sync_start text_items=%d items_since_sync=%d graph_nodes=%d question=%s",
            len(self.text_store.items),
            self.text_store.items_since_sync,
            self.graph_memory.number_of_nodes(),
            question[:120].replace("\n", " "),
        )
        t0 = time.perf_counter()
        await self._aconsolidate_textual_memory(client, question=question, interaction_memory=interaction_memory)
        logger.info(
            "PROFSTEP wm stage=consolidate_done elapsed_ms=%.1f text_items=%d",
            (time.perf_counter() - t0) * 1000.0,
            len(self.text_store.items),
        )
        unprocessed = [text for text in self.text_store.items if text not in self.text_store.processed_texts]
        if unprocessed:
            text_blob = _MemoryFormatter.format_lines(unprocessed)
            known = filter_entities_relevant_to_text(
                [e for e in self.entity_dict.values() if isinstance(e, WikidataEntity)],
                text_blob,
            )
            link_kwargs = self._entity_link_kwargs(kwargs)
            t0 = time.perf_counter()
            link_result, parse_result = await asyncio.gather(
                self._link_entities_async(client, text_blob, known, interaction_memory, **link_kwargs),
                parse_graph_from_text(
                    client,
                    text_blob,
                    interaction_memory=interaction_memory,
                    known_entities=list(known) if known else None,
                ),
            )
            logger.info(
                "PROFSTEP wm stage=link_parse_done elapsed_ms=%.1f unprocessed_items=%d",
                (time.perf_counter() - t0) * 1000.0,
                len(unprocessed),
            )
            linked_entities, link_log = link_result
            relations, parse_log = parse_result
            for entity in linked_entities:
                self.entity_dict[entity.qid] = entity
            edges_before = set(self.graph_memory.edges())
            for triple in self._enhance_relations(relations):
                self.add_edge_to_graph_memory(triple)
            self.text_store.processed_texts.update(unprocessed)
            log_to_interaction_memory(interaction_memory, merge_logs(link_log, parse_log))
            logger.info(
                "PROFSTEP wm stage=graph_update_done linked_entities=%d relations=%d graph_nodes=%d graph_edges=%d",
                len(linked_entities),
                len(relations),
                self.graph_memory.number_of_nodes(),
                self.graph_memory.number_of_edges(),
            )
            new_edges = set(self.graph_memory.edges()) - edges_before
            if new_edges:
                t0 = time.perf_counter()
                await _aprune_graph_edges(client, question, self.graph_memory, new_edges, interaction_memory)
                logger.info(
                    "PROFSTEP wm stage=graph_prune_done elapsed_ms=%.1f graph_nodes=%d graph_edges=%d",
                    (time.perf_counter() - t0) * 1000.0,
                    self.graph_memory.number_of_nodes(),
                    self.graph_memory.number_of_edges(),
                )

        t0 = time.perf_counter()
        self.deduplicate_graph()
        logger.info(
            "PROFSTEP wm stage=dedup_done elapsed_ms=%.1f graph_nodes=%d graph_edges=%d",
            (time.perf_counter() - t0) * 1000.0,
            self.graph_memory.number_of_nodes(),
            self.graph_memory.number_of_edges(),
        )

        self.text_store.dirty = False
        self.text_store.items_since_sync = 0
        logger.info(
            "PROFSTEP wm stage=sync_done elapsed_ms=%.1f",
            (time.perf_counter() - t_sync) * 1000.0,
        )

    async def afinalize(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs,
    ) -> None:
        """Unconditional final consolidation + graph extraction + dedup pass."""
        await self._aconsolidate_textual_memory(client, question=question, interaction_memory=interaction_memory)
        unprocessed = [t for t in self.text_store.items if t not in self.text_store.processed_texts]
        if unprocessed:
            text_blob = _MemoryFormatter.format_lines(unprocessed)
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
            self.text_store.processed_texts.update(unprocessed)
            log_to_interaction_memory(interaction_memory, merge_logs(link_log, parse_log))
            new_edges = set(self.graph_memory.edges()) - edges_before
            if new_edges:
                await _aprune_graph_edges(client, question, self.graph_memory, new_edges, interaction_memory)
        self.deduplicate_graph()
        self.text_store.dirty = False
        self.text_store.items_since_sync = 0

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
            object_ = self.add_node_to_graph_memory(triple.object) if isinstance(triple.object, WikidataEntity) else None
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
            edge["provenance"] = provenance

    def deduplicate_graph(self) -> None:
        self.graph_store.merge_same_qid_nodes()
        # Rebuild cache: merging may have removed duplicate node IDs.
        self._qid_to_node_id = {}
        for node_id, data in self.graph_memory.nodes(data=True):
            qid = getattr(data.get("data"), "id", None)
            if qid:
                self._qid_to_node_id[qid] = node_id

    def format_graph_memory(self) -> str:
        from wemg.llm.roles import SourceType
        from wemg.utils.graph import textualize_graph

        graph = self.graph_memory
        if self.global_knowledge is not None and self.global_knowledge.graph.number_of_nodes() > 0:
            composed = nx.compose(self.global_knowledge.graph, self.graph_memory)
            gs = _GraphStore(composed)
            gs.merge_same_qid_nodes()
            graph = gs.graph
        components = list(nx.weakly_connected_components(graph))
        sections = []
        for comp in components:
            triples, _ = textualize_graph(comp, graph, method="dfs")
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
        if self.global_knowledge is not None:
            for _, data in self.global_knowledge.graph.nodes(data=True):
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



__all__ = ["parse_graph_from_text", "GlobalKnowledge", "MemoryDelta", "WorkingMemory"]

