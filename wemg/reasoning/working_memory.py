"""Maintainable working-memory layer for reasoning.

This module exposes:
- ``WorkingMemory``: Zettelkasten-inspired note-based memory with a structural entity
  index and bidirectional note lifecycle.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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


# =============================================================================
# Zettelkasten data model
# =============================================================================

class NoteType(str, Enum):
    FLEETING = "fleeting"
    LITERATURE = "literature"
    PERMANENT = "permanent"
    STRUCTURE = "structure"


class LinkType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    GAP_FOR = "gap_for"
    ANSWERS = "answers"
    ELABORATES = "elaborates"
    PRECEDES = "precedes"
    REFINES = "refines"
    ANALOGOUS_TO = "analogous_to"
    CO_OCCURS_WITH = "co_occurs_with"


@dataclass
class NoteLink:
    target_id: str
    link_type: LinkType


@dataclass
class MemoryNote:
    id: str
    content: str
    note_type: NoteType
    provenance: Any  # SourceType — avoid circular import at module level
    confidence: float
    links: List[NoteLink] = field(default_factory=list)
    hop_depth: Optional[int] = None
    created_at_step: int = 0
    promoted_at: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "content": self.content,
            "note_type": self.note_type.value,
            "provenance": self.provenance.value if hasattr(self.provenance, "value") else str(self.provenance),
            "confidence": self.confidence,
            "links": [{"target_id": lnk.target_id, "link_type": lnk.link_type.value} for lnk in self.links],
            "hop_depth": self.hop_depth,
            "created_at_step": self.created_at_step,
            "promoted_at": self.promoted_at,
        })

    @classmethod
    def from_json(cls, s: str) -> "MemoryNote":
        from wemg.llm.roles import SourceType
        d = json.loads(s)
        prov_raw = d.get("provenance", "system_prediction")
        try:
            provenance = SourceType(prov_raw)
        except (ValueError, KeyError):
            provenance = SourceType.SYSTEM_PREDICTION
        links = [
            NoteLink(target_id=lnk["target_id"], link_type=LinkType(lnk["link_type"]))
            for lnk in d.get("links", [])
        ]
        return cls(
            id=d["id"],
            content=d["content"],
            note_type=NoteType(d["note_type"]),
            provenance=provenance,
            confidence=d.get("confidence", 0.5),
            links=links,
            hop_depth=d.get("hop_depth"),
            created_at_step=d.get("created_at_step", 0),
            promoted_at=d.get("promoted_at"),
        )


# =============================================================================
# Module-level helpers
# =============================================================================

async def _extract_relations(
    client,
    text: str,
    interaction_memory: Optional[InteractionMemory] = None,
    known_entities: Optional[List[Any]] = None,
):
    """Extract relation triples from text using the RELATION_EXTRACTOR role."""
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


async def _aprune_relations(
    client,
    question: str,
    triples: List[Union[WikiTriple, Relation]],
    interaction_memory=None,
) -> List[Union[WikiTriple, Relation]]:
    """Prune irrelevant relation triples using TRIPLE_PRUNER (Stage B).

    Works on List[Union[WikiTriple, Relation]] — replaces the old graph-edge-based pruner.
    """
    if not triples:
        return triples
    triple_strings: List[str] = []
    for t in triples:
        if isinstance(t, WikiTriple):
            subj = t.subject.label if hasattr(t.subject, "label") and t.subject.label else str(t.subject)
            obj = t.object.label if hasattr(t.object, "label") and t.object.label else str(t.object)
            rel = t.relation.label if hasattr(t.relation, "label") and t.relation.label else str(t.relation)
        else:  # Relation (string-only)
            subj = t.subject or ""
            obj = t.object or ""
            rel = str(t.relation) if t.relation else ""
        triple_strings.append(f"Subject: {subj}\nRelation: {rel}\nObject: {obj}")
    kept_idx = await _aprune_triple_strings_llm(client, question, triple_strings, interaction_memory)
    return [t for i, t in enumerate(triples) if i in kept_idx]


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


# =============================================================================
# WorkingMemory
# =============================================================================

# Priority for sorting notes (lower = higher priority)
_NOTE_TYPE_PRIORITY: Dict[NoteType, int] = {
    NoteType.STRUCTURE: 0,
    NoteType.PERMANENT: 1,
    NoteType.LITERATURE: 2,
    NoteType.FLEETING: 3,
}


class WorkingMemory:
    """Zettelkasten-inspired working memory for CoT and MCTS reasoning.

    Notes enter as FLEETING, survive consolidation as LITERATURE, gain promotion
    to PERMANENT via corroboration, and are synthesized into STRUCTURE notes that
    emit gap questions to steer the reasoning process.

    The ``_entity_graph`` (nx.MultiDiGraph) is a pure structural index for
    triple dedup and KB frontier expansion — it never feeds the LLM directly.
    The ``_note_graph`` (nx.DiGraph) carries semantic note-to-note links.
    All LLM context is served via ``format_textual_memory()``.

    ``synchronize_memory()`` returns ``List[str]`` gap questions.
    """

    def __init__(
        self,
        textual_memory: Optional[List[str]] = None,
        graph_memory=None,  # accepted but ignored — kept for call-site compat
        max_textual_memory_tokens: int = 16384,
        wikidata_client: Optional[WikidataClient] = None,
        annotate_steps: bool = False,
        # Note lifecycle config
        promotion_corroboration_count: int = 2,
        structure_note_trigger_m: int = 5,
        retrieval_k_min: int = 3,
        note_store=None,  # Optional[NoteVectorStore]
    ):
        # --- Note system ---
        self._notes: Dict[str, MemoryNote] = {}
        self._note_graph: nx.DiGraph = nx.DiGraph()  # note_id → note_id with LinkType
        self._entity_to_notes: Dict[str, Set[str]] = defaultdict(set)  # QID → {note_id}
        self._step_counter: int = 0
        self._permanents_since_last_structure: int = 0
        self._note_store = note_store
        self._promotion_corroboration_count = promotion_corroboration_count
        self._structure_note_trigger_m = structure_note_trigger_m
        self._retrieval_k_min = retrieval_k_min

        # --- Structural entity index (pure structural, never feeds LLM) ---
        self._entity_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.max_textual_memory_tokens = max_textual_memory_tokens
        self.annotate_steps = annotate_steps
        self.entity_dict: Dict[str, WikidataEntity] = {}
        self._wikidata_client = wikidata_client

        # Seed from pre-existing textual_memory list (backward compat)
        for item in (textual_memory or []):
            self.add_textual_memory(item)

    # ------------------------------------------------------------------
    # Backward-compat property
    # ------------------------------------------------------------------

    @property
    def textual_memory(self) -> List[str]:
        """Return note contents in priority order (STRUCTURE > PERMANENT > LITERATURE > FLEETING)."""
        notes = sorted(
            self._notes.values(),
            key=lambda n: (_NOTE_TYPE_PRIORITY.get(n.note_type, 4), -n.confidence, -n.created_at_step),
        )
        return [n.content for n in notes]

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    @staticmethod
    def format_memory_item(content: str, provenance) -> str:
        return _MemoryFormatter.format_item(content, provenance)

    def add_textual_memory(self, text: str, source=None, hop_depth: Optional[int] = None) -> None:
        from wemg.llm.roles import SourceType

        if source is None:
            source = SourceType.SYSTEM_PREDICTION
        formatted = _MemoryFormatter.format_item(text, source)
        if hop_depth is not None:
            formatted = f"[hop={hop_depth}] {formatted}"
        # Content-based deduplication
        if any(n.content == formatted for n in self._notes.values()):
            return
        self._step_counter += 1
        note = MemoryNote(
            id=uuid.uuid4().hex,
            content=formatted,
            note_type=NoteType.FLEETING,
            provenance=source,
            confidence=0.3,
            hop_depth=hop_depth,
            created_at_step=self._step_counter,
        )
        self._notes[note.id] = note
        if self._note_store is not None:
            self._note_store.add_or_update(note)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def format_textual_memory(self, query: Optional[str] = None) -> str:
        notes = sorted(
            self._notes.values(),
            key=lambda n: (_NOTE_TYPE_PRIORITY.get(n.note_type, 4), -n.confidence, -n.created_at_step),
        )
        return _MemoryFormatter.format_lines([n.content for n in notes])

    async def retrieve_relevant_notes(
        self,
        entities: List[str],
        question: str,
        k_min: Optional[int] = None,
    ) -> List[MemoryNote]:
        """Graph-proximity retrieval with semantic fallback.

        1. Find notes linked to active entity QIDs.
        2. Expand 1 hop via SUPPORTS / DERIVED_FROM / ELABORATES / REFINES / ANSWERS links.
        3. If still < k_min notes: semantic fallback via NoteVectorStore.
        4. Rank: PERMANENT > STRUCTURE > LITERATURE > FLEETING, then confidence, then recency.
        """
        k_min = k_min if k_min is not None else self._retrieval_k_min
        candidate_ids: Set[str] = set()

        for qid in entities:
            candidate_ids.update(self._entity_to_notes.get(qid, set()))

        expansion_types = {
            LinkType.SUPPORTS, LinkType.DERIVED_FROM, LinkType.ELABORATES,
            LinkType.REFINES, LinkType.ANSWERS,
        }
        for nid in list(candidate_ids):
            if self._note_graph.has_node(nid):
                for _, nbr, data in self._note_graph.out_edges(nid, data=True):
                    if data.get("link_type") in expansion_types:
                        candidate_ids.add(nbr)
                for pred, _, data in self._note_graph.in_edges(nid, data=True):
                    if data.get("link_type") in expansion_types:
                        candidate_ids.add(pred)

        if len(candidate_ids) < k_min and self._note_store is not None:
            for n in self._note_store.search(question, k=10):
                candidate_ids.add(n.id)

        candidates = [self._notes[nid] for nid in candidate_ids if nid in self._notes]
        candidates.sort(
            key=lambda n: (_NOTE_TYPE_PRIORITY.get(n.note_type, 4), -n.confidence, -n.created_at_step)
        )
        return candidates

    # ------------------------------------------------------------------
    # Entity graph: structural index
    # ------------------------------------------------------------------

    def register_retrieved_triples(
        self,
        triples: List[WikiTriple],
        entities: List[WikidataEntity],
    ) -> None:
        """Update entity structural index and create LITERATURE notes for new triples.

        - Adds/updates QID nodes in ``_entity_graph`` for each entity.
        - For each triple: skips if already known (earlier-wins dedup); else adds edge
          and creates a LITERATURE note ("subj → rel → obj").
        - Object node: QID node if ``triple.object`` has a qid, else literal node.
        """
        from wemg.llm.roles import SourceType

        # Update entity dict + ensure QID nodes exist
        for entity in entities:
            if not entity.qid:
                continue
            self.entity_dict[entity.qid] = entity
            if self._entity_graph.has_node(entity.qid):
                self._entity_graph.nodes[entity.qid]["label"] = entity.label or entity.qid
                self._entity_graph.nodes[entity.qid]["description"] = entity.description or ""
            else:
                self._entity_graph.add_node(
                    entity.qid,
                    label=entity.label or entity.qid,
                    description=entity.description or "",
                )

        existing_contents = {n.content for n in self._notes.values()}

        for triple in triples:
            subj = triple.subject
            if not isinstance(subj, WikidataEntity) or not subj.qid:
                continue
            subj_id = subj.qid
            subj_label = subj.label or subj_id

            # Ensure subject node exists
            if not self._entity_graph.has_node(subj_id):
                self._entity_graph.add_node(subj_id, label=subj_label, description=subj.description or "")

            rel_label = (
                triple.relation.label
                if hasattr(triple.relation, "label") and triple.relation.label
                else str(triple.relation)
            )
            if not rel_label:
                continue

            obj = triple.object
            if isinstance(obj, WikidataEntity) and obj.qid:
                obj_id = obj.qid
                obj_label = obj.label or obj_id
                if not self._entity_graph.has_node(obj_id):
                    self._entity_graph.add_node(obj_id, label=obj_label, description=obj.description or "")
            else:
                # Literal node (date, number, string scalar)
                obj_val = obj.label if hasattr(obj, "label") and obj.label else str(obj)
                obj_id = obj_val
                if not self._entity_graph.has_node(obj_id):
                    self._entity_graph.add_node(obj_id, label=obj_val, node_type="literal")
                obj_label = obj_val

            # Earlier-wins dedup
            if self.is_triple_known(subj_id, rel_label, obj_id):
                continue

            self._entity_graph.add_edge(subj_id, obj_id, rel_label=rel_label)

            # Create LITERATURE note
            note_content = f"{subj_label} → {rel_label} → {obj_label}"
            formatted = _MemoryFormatter.format_item(note_content, SourceType.RETRIEVAL)
            if formatted not in existing_contents:
                self._step_counter += 1
                note = MemoryNote(
                    id=uuid.uuid4().hex,
                    content=formatted,
                    note_type=NoteType.LITERATURE,
                    provenance=SourceType.RETRIEVAL,
                    confidence=0.5,
                    created_at_step=self._step_counter,
                )
                self._notes[note.id] = note
                existing_contents.add(formatted)
                if self._note_store is not None:
                    self._note_store.add_or_update(note)

    def is_triple_known(self, subj_id: str, rel_label: str, obj_id: str) -> bool:
        """Return True if (subj_id, rel_label, obj_id) is already in the entity graph."""
        return (
            self._entity_graph.has_edge(subj_id, obj_id)
            and any(
                d["rel_label"] == rel_label
                for d in self._entity_graph[subj_id][obj_id].values()
            )
        )

    def get_underexplored_neighbor_qids(
        self,
        seed_qids: Sequence[str],
        max_degree: int = 2,
        max_results: int = 5,
    ) -> List[str]:
        """Return QID neighbors of seed_qids that have low degree (underexplored frontier).

        Filters out literal nodes and nodes already in seed_qids.
        """
        neighbor_qids: List[str] = []
        seen: Set[str] = set(seed_qids)
        for q in seed_qids:
            if not self._entity_graph.has_node(q):
                continue
            neighbours = (
                set(self._entity_graph.successors(q))
                | set(self._entity_graph.predecessors(q))
            )
            for nbr in neighbours:
                if nbr in seen:
                    continue
                if self._entity_graph.nodes[nbr].get("node_type") == "literal":
                    continue
                degree = self._entity_graph.in_degree(nbr) + self._entity_graph.out_degree(nbr)
                if degree > max_degree:
                    continue
                seen.add(nbr)
                neighbor_qids.append(nbr)
                if len(neighbor_qids) >= max_results:
                    return neighbor_qids
        return neighbor_qids

    # ------------------------------------------------------------------
    # Internal: consolidation (step 2)
    # ------------------------------------------------------------------

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

    async def _aconsolidate_fleeting_notes(
        self, cheap_client, question: str, interaction_memory=None
    ) -> None:
        """Consolidate all FLEETING notes; surviving content becomes LITERATURE."""
        from wemg.llm.roles import SourceType

        fleeting = [n for n in self._notes.values() if n.note_type == NoteType.FLEETING]
        if not fleeting:
            return

        raw_memory = _MemoryFormatter.format_lines([n.content for n in fleeting])
        output, log = await self._arun_consolidation(cheap_client, question, raw_memory, interaction_memory)

        # Remove old FLEETING notes
        for n in fleeting:
            self._notes.pop(n.id, None)
            if self._note_graph.has_node(n.id):
                self._note_graph.remove_node(n.id)
            if self._note_store is not None:
                self._note_store.delete(n.id)

        # Create LITERATURE notes from consolidated output
        for item in output.consolidated_memory:
            prov = SourceType.SYSTEM_PREDICTION
            if hasattr(item, "provenance") and item.provenance == SourceType.RETRIEVAL.value:
                prov = SourceType.RETRIEVAL
            hop = getattr(item, "hop_depth", None)
            content = _MemoryFormatter.format_item(item.content, prov)
            if hop is not None:
                content = f"[hop={hop}] {content}"
            # Skip if exact content already exists
            if any(n.content == content for n in self._notes.values()):
                continue
            self._step_counter += 1
            note = MemoryNote(
                id=uuid.uuid4().hex,
                content=content,
                note_type=NoteType.LITERATURE,
                provenance=prov,
                confidence=0.4,
                hop_depth=hop,
                created_at_step=self._step_counter,
            )
            self._notes[note.id] = note
            if self._note_store is not None:
                self._note_store.add_or_update(note)

        log_to_interaction_memory(interaction_memory, log)

    # ------------------------------------------------------------------
    # Internal: entity linking + note linking (step 3)
    # ------------------------------------------------------------------

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
                text, self._wikidata_client, endpoint=azure_endpoint, key=azure_key,
            )
        else:
            entities, _, link_log = await link_entities_llm(
                client, text, self._wikidata_client,
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
                triples.append(rel)
        return triples

    def _update_entity_to_notes(self) -> None:
        """Rebuild QID → note-ID index based on entity name occurrence in note content."""
        self._entity_to_notes = defaultdict(set)
        for qid, entity in self.entity_dict.items():
            label = getattr(entity, "label", None) or ""
            if not label:
                continue
            label_lower = label.lower()
            for nid, note in self._notes.items():
                if label_lower in note.content.lower():
                    self._entity_to_notes[qid].add(nid)

    def _create_entity_based_links(self) -> None:
        """Add CO_OCCURS_WITH links between notes that mention the same entity."""
        for note_ids in self._entity_to_notes.values():
            ids = list(note_ids)
            for i, nid_a in enumerate(ids):
                for nid_b in ids[i + 1:]:
                    if not self._note_graph.has_edge(nid_a, nid_b):
                        self._note_graph.add_edge(nid_a, nid_b, link_type=LinkType.CO_OCCURS_WITH)
                    note_a = self._notes.get(nid_a)
                    note_b = self._notes.get(nid_b)
                    if note_a and not any(lnk.target_id == nid_b for lnk in note_a.links):
                        note_a.links.append(NoteLink(nid_b, LinkType.CO_OCCURS_WITH))
                    if note_b and not any(lnk.target_id == nid_a for lnk in note_b.links):
                        note_b.links.append(NoteLink(nid_a, LinkType.CO_OCCURS_WITH))

    async def _alink_entities_and_update_notes(
        self,
        cheap_client,
        question: str,
        interaction_memory,
        reranker=None,
        **kwargs,
    ) -> None:
        """Entity linking + relation extraction → entity graph update + LITERATURE notes.

        Step 3 of synchronize_memory():
        - Entity linking: text → WikidataEntity QIDs (populates entity_dict).
        - Relation extraction: text → List[Relation].
        - Enhance relations (try to resolve string subjects/objects to WikidataEntity).
        - Prune irrelevant relations with TRIPLE_PRUNER.
        - Register kept WikiTriple relations via register_retrieved_triples() (dedup + note creation).
        - String-only Relation objects get LITERATURE notes directly.
        - Rebuild entity_to_notes index and CO_OCCURS_WITH note links.
        """
        from wemg.llm.roles import SourceType

        non_structure = [n for n in self._notes.values() if n.note_type != NoteType.STRUCTURE]
        if not non_structure:
            return

        text_blob = _MemoryFormatter.format_lines([n.content for n in non_structure])
        known = filter_entities_relevant_to_text(
            [e for e in self.entity_dict.values() if isinstance(e, WikidataEntity)],
            text_blob,
        )
        link_kwargs = self._entity_link_kwargs(kwargs)
        t0 = time.perf_counter()
        link_result, parse_result = await asyncio.gather(
            self._link_entities_async(cheap_client, text_blob, known, interaction_memory, **link_kwargs),
            _extract_relations(
                cheap_client, text_blob,
                interaction_memory=interaction_memory,
                known_entities=list(known) if known else None,
            ),
        )
        logger.info(
            "PROFSTEP wm stage=link_parse_done elapsed_ms=%.1f",
            (time.perf_counter() - t0) * 1000.0,
        )

        linked_entities, link_log = link_result
        relations, parse_log = parse_result
        for entity in linked_entities:
            self.entity_dict[entity.qid] = entity
        log_to_interaction_memory(interaction_memory, merge_logs(link_log, parse_log))

        enhanced = self._enhance_relations(relations)

        if enhanced:
            t0 = time.perf_counter()
            kept = await _aprune_relations(cheap_client, question, enhanced, interaction_memory)
            logger.info(
                "PROFSTEP wm stage=relation_prune_done elapsed_ms=%.1f relations=%d",
                (time.perf_counter() - t0) * 1000.0,
                len(kept),
            )

            # WikiTriple → entity graph + LITERATURE notes via register_retrieved_triples
            wiki_triples = [t for t in kept if isinstance(t, WikiTriple)]
            all_triple_entities: List[WikidataEntity] = []
            for t in wiki_triples:
                if isinstance(t.subject, WikidataEntity):
                    all_triple_entities.append(t.subject)
                if isinstance(t.object, WikidataEntity):
                    all_triple_entities.append(t.object)
            if wiki_triples:
                self.register_retrieved_triples(wiki_triples, all_triple_entities)

            # String-only Relation → LITERATURE notes directly
            existing_contents = {n.content for n in self._notes.values()}
            for rel in kept:
                if isinstance(rel, WikiTriple):
                    continue
                if not rel.subject or not rel.object or not rel.relation:
                    continue
                note_content = f"{rel.subject} → {rel.relation} → {rel.object}"
                formatted = _MemoryFormatter.format_item(note_content, SourceType.RETRIEVAL)
                if formatted not in existing_contents:
                    self._step_counter += 1
                    note = MemoryNote(
                        id=uuid.uuid4().hex,
                        content=formatted,
                        note_type=NoteType.LITERATURE,
                        provenance=SourceType.RETRIEVAL,
                        confidence=0.5,
                        created_at_step=self._step_counter,
                    )
                    self._notes[note.id] = note
                    existing_contents.add(formatted)
                    if self._note_store is not None:
                        self._note_store.add_or_update(note)

            logger.info(
                "PROFSTEP wm stage=graph_update_done linked=%d relations=%d graph_nodes=%d graph_edges=%d",
                len(linked_entities), len(relations),
                self._entity_graph.number_of_nodes(), self._entity_graph.number_of_edges(),
            )

        self._update_entity_to_notes()
        self._create_entity_based_links()

    # ------------------------------------------------------------------
    # Internal: promotion (step 4)
    # ------------------------------------------------------------------

    async def _apromote_notes(self, client, question: str, interaction_memory=None) -> int:
        """Run PROMOTION_EVALUATOR to move LITERATURE → PERMANENT. Returns new-permanent count."""
        from wemg.llm.roles import (
            PROMOTION_EVALUATOR, PromotionEvalInput, NoteForPromotion, execute_role, SourceType
        )

        literature = [n for n in self._notes.values() if n.note_type == NoteType.LITERATURE]
        permanents = [n for n in self._notes.values() if n.note_type == NoteType.PERMANENT]
        if not literature:
            return 0

        def _prov_label(n: MemoryNote) -> str:
            v = n.provenance.value if hasattr(n.provenance, "value") else str(n.provenance)
            return "Retrieval" if v == "retrieval" else "System Prediction"

        candidates = [NoteForPromotion(idx=i, content=n.content, provenance=_prov_label(n)) for i, n in enumerate(literature)]
        perm_list = [NoteForPromotion(idx=i, content=n.content, provenance=_prov_label(n)) for i, n in enumerate(permanents)]

        results, log = await execute_role(
            client=client,
            role=PROMOTION_EVALUATOR,
            input_data=PromotionEvalInput(question=question, candidate_notes=candidates, permanent_notes=perm_list),
            interaction_memory=interaction_memory,
            n=1,
        )
        log_to_interaction_memory(interaction_memory, log)
        if not results:
            return 0

        result = results[0]
        new_perms = 0

        for idx in result.promote_indices:
            if 0 <= idx < len(literature):
                note = literature[idx]
                note.note_type = NoteType.PERMANENT
                note.confidence = min(note.confidence + 0.2, 1.0)
                note.promoted_at = self._step_counter
                new_perms += 1
                if self._note_store is not None:
                    self._note_store.add_or_update(note)

        for dem in result.demotions:
            pidx, cidx = dem.permanent_idx, dem.contradicting_candidate_idx
            if 0 <= pidx < len(permanents):
                perm_note = permanents[pidx]
                perm_note.note_type = NoteType.LITERATURE
                perm_note.confidence = max(perm_note.confidence - 0.2, 0.1)
                if 0 <= cidx < len(literature):
                    contra = literature[cidx]
                    if not any(lnk.target_id == contra.id for lnk in perm_note.links):
                        perm_note.links.append(NoteLink(contra.id, LinkType.CONTRADICTS))
                    if not any(lnk.target_id == perm_note.id for lnk in contra.links):
                        contra.links.append(NoteLink(perm_note.id, LinkType.CONTRADICTS))
                    if not self._note_graph.has_edge(perm_note.id, contra.id):
                        self._note_graph.add_edge(perm_note.id, contra.id, link_type=LinkType.CONTRADICTS)
                if self._note_store is not None:
                    self._note_store.add_or_update(perm_note)

        return new_perms

    # ------------------------------------------------------------------
    # Internal: structure notes + gap questions (step 5)
    # ------------------------------------------------------------------

    async def _agenerate_structure_notes(
        self, client, question: str, interaction_memory=None
    ) -> List[str]:
        """Generate a STRUCTURE note from PERMANENT notes. Returns gap_questions list."""
        from wemg.llm.roles import (
            STRUCTURE_NOTE_GENERATOR, StructureNoteInput, execute_role, SourceType
        )

        permanents = [n for n in self._notes.values() if n.note_type == NoteType.PERMANENT]
        if not permanents:
            return []

        structures = [n for n in self._notes.values() if n.note_type == NoteType.STRUCTURE]
        existing_summaries = [n.content for n in structures]
        permanent_contents = [n.content for n in permanents]

        results, log = await execute_role(
            client=client,
            role=STRUCTURE_NOTE_GENERATOR,
            input_data=StructureNoteInput(
                question=question,
                permanent_notes=permanent_contents,
                existing_structure_summaries=existing_summaries,
            ),
            interaction_memory=interaction_memory,
            n=1,
        )
        log_to_interaction_memory(interaction_memory, log)
        if not results:
            return []

        result = results[0]
        struct_content = f"[Structure: {result.theme}] {result.summary}"
        self._step_counter += 1
        struct_note = MemoryNote(
            id=uuid.uuid4().hex,
            content=struct_content,
            note_type=NoteType.STRUCTURE,
            provenance=SourceType.SYSTEM_PREDICTION,
            confidence=0.9,
            created_at_step=self._step_counter,
        )

        for idx in result.covered_note_indices:
            if 0 <= idx < len(permanents):
                perm_note = permanents[idx]
                struct_note.links.append(NoteLink(perm_note.id, LinkType.DERIVED_FROM))
                if not self._note_graph.has_edge(struct_note.id, perm_note.id):
                    self._note_graph.add_edge(struct_note.id, perm_note.id, link_type=LinkType.DERIVED_FROM)

        self._notes[struct_note.id] = struct_note
        if self._note_store is not None:
            self._note_store.add_or_update(struct_note)

        logger.info(
            "PROFSTEP wm stage=structure_note theme=%s gap_questions=%d",
            result.theme, len(result.gap_questions),
        )
        return result.gap_questions

    # ------------------------------------------------------------------
    # Main synchronize_memory — 6-step Zettelkasten pipeline
    # ------------------------------------------------------------------

    async def synchronize_memory(
        self,
        client,
        question: str,
        interaction_memory: Optional[InteractionMemory] = None,
        cheap_client=None,
        reranker=None,
        **kwargs,
    ) -> List[str]:
        """Note lifecycle management with entity structural index sync.

        Step 1: Notes already entered as FLEETING via add_textual_memory().
        Step 2: Consolidate FLEETING → LITERATURE (cheap client).
        Step 3: Entity linking + relation extraction → entity graph update + LITERATURE notes + entity-based note links.
        Step 4: Promote LITERATURE → PERMANENT via PROMOTION_EVALUATOR (expensive).
        Step 5: Generate STRUCTURE notes + gap questions if threshold reached (expensive).
        Step 6: Retrieval happens on-demand via retrieve_relevant_notes() / format_textual_memory().

        Returns list of gap questions emitted by new STRUCTURE notes.
        """
        cheap = cheap_client or client
        gap_questions: List[str] = []

        if not self._notes:
            return gap_questions

        t_sync = time.perf_counter()
        logger.info(
            "PROFSTEP wm stage=sync_start total_notes=%d entity_graph_nodes=%d question=%s",
            len(self._notes),
            self._entity_graph.number_of_nodes(),
            question[:120].replace("\n", " "),
        )

        # Step 2: Consolidate FLEETING → LITERATURE
        t0 = time.perf_counter()
        await self._aconsolidate_fleeting_notes(cheap, question, interaction_memory)
        logger.info(
            "PROFSTEP wm stage=consolidate_done elapsed_ms=%.1f lit=%d perm=%d",
            (time.perf_counter() - t0) * 1000.0,
            sum(1 for n in self._notes.values() if n.note_type == NoteType.LITERATURE),
            sum(1 for n in self._notes.values() if n.note_type == NoteType.PERMANENT),
        )

        # Step 3: Entity linking + relation extraction + entity graph + LITERATURE notes
        t0 = time.perf_counter()
        await self._alink_entities_and_update_notes(cheap, question, interaction_memory, reranker=reranker, **kwargs)
        logger.info(
            "PROFSTEP wm stage=entity_notes_sync_done elapsed_ms=%.1f total_notes=%d",
            (time.perf_counter() - t0) * 1000.0, len(self._notes),
        )

        # Step 4: Promote LITERATURE → PERMANENT (expensive client)
        literature = [n for n in self._notes.values() if n.note_type == NoteType.LITERATURE]
        if literature:
            t0 = time.perf_counter()
            new_perms = await self._apromote_notes(client, question, interaction_memory)
            self._permanents_since_last_structure += new_perms
            logger.info(
                "PROFSTEP wm stage=promotion_done elapsed_ms=%.1f new_perms=%d total_perms=%d",
                (time.perf_counter() - t0) * 1000.0,
                new_perms,
                sum(1 for n in self._notes.values() if n.note_type == NoteType.PERMANENT),
            )

        # Step 5: Structure notes + gap questions (expensive client)
        permanents = [n for n in self._notes.values() if n.note_type == NoteType.PERMANENT]
        has_structure = any(n.note_type == NoteType.STRUCTURE for n in self._notes.values())
        should_gen_structure = (
            self._permanents_since_last_structure >= self._structure_note_trigger_m
            or (permanents and not has_structure)
        )
        if should_gen_structure and permanents:
            t0 = time.perf_counter()
            new_gaps = await self._agenerate_structure_notes(client, question, interaction_memory)
            gap_questions.extend(new_gaps)
            self._permanents_since_last_structure = 0
            logger.info(
                "PROFSTEP wm stage=structure_done elapsed_ms=%.1f gap_questions=%d",
                (time.perf_counter() - t0) * 1000.0, len(new_gaps),
            )

        logger.info(
            "PROFSTEP wm stage=sync_done elapsed_ms=%.1f total_notes=%d gap_questions=%d",
            (time.perf_counter() - t_sync) * 1000.0,
            len(self._notes), len(gap_questions),
        )
        return gap_questions

    def close(self) -> None:
        """Release auxiliary resources used by working memory."""
        if self._note_store is not None:
            try:
                self._note_store.close()
            except Exception:
                pass


__all__ = [
    "NoteType", "LinkType", "NoteLink", "MemoryNote",
    "WorkingMemory",
]
