"""Memory-update graph.

Synchronizes text↔graph working memory as an explicit ``StateGraph`` with a
single OpenIE extraction pass, batched
``triple_pruner`` (size 16) on **newly proposed edges only**, and an optional
post-merge consolidation pass that folds newly-textualised graph triples back into
prose memory.

Flow::

    START
      → consolidate_pre
      → open_ie                                        # entities + relations
      → link_entities                                  # Wikidata tool only
      → merge_and_prune
      → textualize_graph
      → route:
            kept_triples non-empty → consolidate_post
            else                 → finalize_memory
      → END
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..config import MemoryConfig
from ..llm import RoleModelRegistry, execute_role_lc
from ..roles import (
    MEMORY_CONSOLIDATOR,
    OPEN_IE,
    TRIPLE_PRUNER,
    Entity,
    MemoryConsolidationInput,
    MemoryConsolidationOutput,
    OpenIEInput,
    Relation,
    SourceType,
    TriplePruneInput,
    WikidataEntity,
)
from ..tools.wikidata import link_entities
from ._memory_text import format_triple_line

logger = logging.getLogger(__name__)


class MemoryUpdateState(TypedDict, total=False):
    # Inputs
    question: str
    new_text_items: List[str]
    # Retrieval-grounded facts (e.g. EXTRACTOR output over reranked passages).
    # Tagged ``[Retrieval]`` in consolidation so provenance-aware rules
    # (conflict resolution, provenance audit) can prefer them over
    # ``[System Prediction]`` items from ``new_text_items``.
    new_retrieval_items: List[str]
    new_raw_triples: List[Any]
    current_text_memory: List[str]
    current_graph: nx.DiGraph
    entity_dict: Dict[str, Any]
    # Reasoning-tree depth at which this iteration's new items were produced.
    # Prefixed as ``[hop=N]`` on new items so the consolidator's Hop Depth
    # Filtering rule can prefer the lowest-hop item that fully answers the
    # question. ``None`` disables annotation.
    #
    # Matches legacy ``WorkingMemory.add_textual_memory(..., hop_depth=...)``:
    # hop is assigned *per producing node*, not uniformly per iteration.
    # ``hop_depth`` tags generated sub-answers/rollouts and retrieval facts
    # (legacy ``node.depth + 1``); ``critique_hop_depth`` tags verifier /
    # self-correction items, which legacy emits one level shallower at
    # ``node.depth`` (see ``mcts.py`` ``_self_correct_nodes`` assessments).
    hop_depth: int
    new_critique_items: List[str]
    critique_hop_depth: int

    # Intermediates
    consolidated_memory: List[str]
    extracted_entities: List[Entity]
    extracted_relations: List[Relation]
    linked_entities: Dict[str, str]
    kept_triples: List[str]
    # Stringified forms of kept triples that originated from ``new_raw_triples``
    # (KG retrieval) rather than OpenIE over generated text — these carry
    # [Retrieval] provenance when textualized.
    kg_triple_strings: List[str]

    # Outputs
    updated_text_memory: List[str]
    updated_graph: nx.DiGraph
    updated_entity_dict: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _format_memory_item(content: str, provenance: SourceType) -> str:
    tag = {
        SourceType.SYSTEM_PREDICTION: "[System Prediction]",
        SourceType.RETRIEVAL: "[Retrieval]",
    }.get(provenance, "")
    normalized = (content or "").strip()
    # Pass through items that already carry a provenance tag or a leading
    # ``[hop=N]`` reasoning-depth annotation (see RC-C / hop-depth filtering).
    if normalized.startswith(("[System Prediction]", "[Retrieval]", "[hop=")):
        return normalized
    return f"{tag}: {normalized}" if tag else normalized


def _format_lines(items: Sequence[str]) -> str:
    return "\n".join(f"- {i.strip()}" for i in items if i and i.strip())


_PROVENANCE_TAGS = ("[System Prediction]", "[Retrieval]")


def _strip_provenance_tag(text: str) -> str:
    stripped = (text or "").strip()
    # Strip an optional leading ``[hop=N]`` annotation before the provenance tag
    # so neither leaks into entity/relation surface forms or re-tagged content.
    if stripped.startswith("[hop="):
        end = stripped.find("]")
        if end != -1:
            stripped = stripped[end + 1 :].lstrip()
    for tag in _PROVENANCE_TAGS:
        if stripped.startswith(tag):
            stripped = stripped[len(tag) :].lstrip(": ").strip()
            break
    return stripped


def _is_retrieval_grounded(text: str) -> bool:
    """True if a memory line carries the ``[Retrieval]`` provenance tag.

    Skips an optional leading ``[hop=N]`` annotation first (same order as
    :func:`_strip_provenance_tag`). ``[Retrieval]`` facts came from evidence and
    are already grounded; re-verification targets ``[System Prediction]`` facts
    (the model's own inferences), so callers use this to exclude grounded items.
    """
    stripped = (text or "").strip()
    if stripped.startswith("[hop="):
        end = stripped.find("]")
        if end != -1:
            stripped = stripped[end + 1 :].lstrip()
    return stripped.startswith("[Retrieval]")


def _consolidated_to_text(output: MemoryConsolidationOutput) -> List[str]:
    """Render consolidated items as provenance-tagged strings.

    Tags must survive the round-trip into ``text_memory``: downstream prompts
    (FINAL_ANSWER_SYNTHESIZER adjudication, the consolidator's own provenance
    audit on later passes) key on ``[Retrieval]`` vs ``[System Prediction]``.
    Dropping them here would silently demote all retrieval evidence.
    """
    items: List[str] = []
    for item in output.consolidated_memory:
        prov = (
            SourceType.RETRIEVAL
            if (item.provenance or "").strip().lower() == "retrieval"
            else SourceType.SYSTEM_PREDICTION
        )
        formatted = _format_memory_item(_strip_provenance_tag(item.content), prov)
        # Re-attach the consolidator's hop_depth so it survives into text_memory
        # and the next consolidation pass can apply Hop Depth Filtering again.
        hop = getattr(item, "hop_depth", None)
        if hop is not None:
            formatted = f"[hop={hop}] {formatted}"
        items.append(formatted)
    return items


def _stringify_triple(rel: Relation) -> str:
    return format_triple_line(rel.subject, rel.relation, rel.object)


def _node_key(name: str, qid: Optional[str]) -> str:
    return qid if qid else name


def _collect_unlinked_entity_names(
    entities: Sequence[Entity],
    relations: Sequence[Relation],
    entity_dict: Dict[str, Any],
) -> List[str]:
    known_labels = {
        (e.label or "").lower(): e.qid
        for e in entity_dict.values()
        if isinstance(e, WikidataEntity) and e.qid
    }
    candidates: List[str] = []
    seen: set[str] = set()

    def maybe_add(name: str) -> None:
        normalized = (name or "").strip()
        if not normalized or normalized in seen:
            return
        if normalized.lower() in known_labels:
            return
        seen.add(normalized)
        candidates.append(normalized)

    for entity in entities:
        ent_qid = getattr(entity, "id", None)
        if ent_qid and ent_qid in entity_dict:
            continue
        maybe_add(getattr(entity, "name", "") or "")

    for rel in relations:
        if not rel.subject_id:
            maybe_add(rel.subject)
        if not rel.object_id:
            maybe_add(rel.object)

    return candidates


def _add_triple_to_graph(
    graph: nx.DiGraph,
    rel: Relation,
    entity_dict: Dict[str, Any],
) -> None:
    sub_key = _node_key(rel.subject, rel.subject_id)
    obj_key = _node_key(rel.object, rel.object_id)
    if not sub_key or not obj_key or not rel.relation:
        return
    if not graph.has_node(sub_key):
        graph.add_node(
            sub_key,
            name=rel.subject,
            qid=rel.subject_id,
        )
    if not graph.has_node(obj_key):
        graph.add_node(
            obj_key,
            name=rel.object,
            qid=rel.object_id,
        )
    if graph.has_edge(sub_key, obj_key):
        relations = graph.edges[sub_key, obj_key].setdefault("relation", set())
        if not isinstance(relations, set):
            relations = set(relations) if relations else set()
            graph.edges[sub_key, obj_key]["relation"] = relations
        relations.add(rel.relation)
    else:
        graph.add_edge(sub_key, obj_key, relation={rel.relation})


def _relation_already_in_graph(graph: nx.DiGraph, rel: Relation) -> bool:
    """Edge with the same relation already present?

    Match on either QID-keyed nodes or surface-form node ids; treat existing
    edge ``relation`` (set or scalar) uniformly.
    """
    candidates_src = [k for k in (rel.subject_id, rel.subject) if k]
    candidates_dst = [k for k in (rel.object_id, rel.object) if k]
    for src in candidates_src:
        for dst in candidates_dst:
            if not graph.has_edge(src, dst):
                continue
            existing = graph.edges[src, dst].get("relation")
            if existing is None:
                continue
            if isinstance(existing, (set, list, tuple)):
                if rel.relation in existing:
                    return True
            elif existing == rel.relation:
                return True
    return False


def _merge_same_qid_nodes(graph: nx.DiGraph) -> None:
    """Collapse multiple nodes sharing the same QID into a single canonical node."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        qid = data.get("qid")
        if qid:
            groups[qid].append(node_id)

    for qid, node_ids in groups.items():
        if len(node_ids) <= 1:
            continue
        # Prefer the node already keyed by QID, else the first occurrence.
        canonical = next((n for n in node_ids if n == qid), node_ids[0])
        for dup in node_ids:
            if dup == canonical:
                continue
            for pred in list(graph.predecessors(dup)):
                if pred == canonical:
                    continue
                edge = dict(graph.edges[pred, dup])
                rels = edge.get("relation", set()) or set()
                if graph.has_edge(pred, canonical):
                    existing = graph.edges[pred, canonical].setdefault(
                        "relation", set()
                    )
                    if not isinstance(existing, set):
                        existing = set(existing) if existing else set()
                        graph.edges[pred, canonical]["relation"] = existing
                    existing.update(rels)
                else:
                    graph.add_edge(pred, canonical, relation=set(rels))
            for succ in list(graph.successors(dup)):
                if succ == canonical:
                    continue
                edge = dict(graph.edges[dup, succ])
                rels = edge.get("relation", set()) or set()
                if graph.has_edge(canonical, succ):
                    existing = graph.edges[canonical, succ].setdefault(
                        "relation", set()
                    )
                    if not isinstance(existing, set):
                        existing = set(existing) if existing else set()
                        graph.edges[canonical, succ]["relation"] = existing
                    existing.update(rels)
                else:
                    graph.add_edge(canonical, succ, relation=set(rels))
            graph.remove_node(dup)


def _coerce_raw_triple_to_relation(triple: Any) -> Optional[Relation]:
    """Best-effort coercion of a heterogeneous raw triple into a ``Relation``.

    Accepts ``Relation`` instances directly, dicts shaped like a relation, and
    KG ``WikiTriple``-style objects carrying ``subject``/``relation``/``object``
    attributes whose nested ``.qid`` / ``.label`` fields can be unpacked.
    """
    if triple is None:
        return None
    if isinstance(triple, Relation):
        return triple
    if isinstance(triple, str):
        # Readable KG triple ``subject [Qs] -- relation -- object [Qo]`` emitted
        # by the wikidata tool. Split on ` -- `; pull the optional ``[Q…]`` id
        # off each entity end so graph edges keep their QIDs.
        parts = [p.strip() for p in triple.split(" -- ")]
        if len(parts) != 3 or not all(parts):
            return None

        def _split_id(text: str) -> tuple[str, Optional[str]]:
            m = re.search(r"\s*\[(Q\d+)\]\s*$", text)
            if m:
                return text[: m.start()].strip(), m.group(1)
            return text.strip(), None

        subj, subj_id = _split_id(parts[0])
        obj, obj_id = _split_id(parts[2])
        try:
            return Relation(
                subject=subj,
                subject_id=subj_id,
                relation=parts[1],
                object=obj,
                object_id=obj_id,
                context=None,
            )
        except Exception:
            return None
    if isinstance(triple, dict):
        try:
            return Relation(
                subject=str(triple.get("subject", "")),
                subject_id=triple.get("subject_id"),
                relation=str(triple.get("relation", "")),
                object=str(triple.get("object", "")),
                object_id=triple.get("object_id"),
                context=triple.get("context"),
            )
        except Exception:
            return None

    subject = getattr(triple, "subject", None)
    relation_obj = getattr(triple, "relation", None)
    object_ = getattr(triple, "object", None)
    if subject is None or relation_obj is None or object_ is None:
        return None
    try:
        return Relation(
            subject=str(
                getattr(subject, "label", None)
                or getattr(subject, "name", None)
                or subject
            ),
            subject_id=getattr(subject, "qid", None) or getattr(subject, "id", None),
            relation=str(getattr(relation_obj, "label", None) or relation_obj),
            object=str(
                getattr(object_, "label", None)
                or getattr(object_, "name", None)
                or object_
            ),
            object_id=getattr(object_, "qid", None) or getattr(object_, "id", None),
            context=None,
        )
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Node implementations
# ──────────────────────────────────────────────────────────────────────────────


def build_memory_update_graph(
    registry: RoleModelRegistry, *, memory_cfg: Optional[MemoryConfig] = None
):
    """Compile the MemoryUpdateGraph. *registry* supplies role-tier LLM clients."""

    cfg = memory_cfg or MemoryConfig()
    batch_size = cfg.prune_batch_size

    async def consolidate_pre(state: MemoryUpdateState) -> Dict[str, Any]:
        current = list(state.get("current_text_memory") or [])
        new_items = list(state.get("new_text_items") or [])
        new_critiques = list(state.get("new_critique_items") or [])
        new_retrieval = list(state.get("new_retrieval_items") or [])
        if not (current or new_items or new_critiques or new_retrieval):
            return {"consolidated_memory": []}

        # ``current`` items already carry tags (and any ``[hop=N]`` prefix) from
        # a previous consolidation pass (``_format_memory_item`` passes pre-tagged
        # strings through); untagged strings default to [System Prediction].
        # Retrieval-grounded facts are tagged [Retrieval] so the consolidator's
        # provenance audit and conflict-resolution rules can anchor on them.
        # NEW items get a ``[hop=N]`` prefix (when a depth is supplied) so Hop
        # Depth Filtering can discard deeper items that overshoot a question
        # already answered at a shallower hop. Mirroring legacy
        # ``add_textual_memory(..., hop_depth=...)``, the hop is assigned per
        # producing node: generated sub-answers/rollouts and retrieval facts at
        # ``hop_depth``; verifier/self-correction critiques one level shallower
        # at ``critique_hop_depth``.
        def _hop_prefix(value: Optional[int]) -> str:
            return f"[hop={value}] " if value is not None else ""

        gen_prefix = _hop_prefix(state.get("hop_depth"))
        crit_prefix = _hop_prefix(state.get("critique_hop_depth"))
        tagged = (
            [_format_memory_item(item, SourceType.SYSTEM_PREDICTION) for item in current]
            + [
                gen_prefix + _format_memory_item(item, SourceType.SYSTEM_PREDICTION)
                for item in new_items
            ]
            + [
                crit_prefix + _format_memory_item(item, SourceType.SYSTEM_PREDICTION)
                for item in new_critiques
            ]
            + [
                gen_prefix + _format_memory_item(item, SourceType.RETRIEVAL)
                for item in new_retrieval
            ]
        )

        raw_blob = _format_lines(tagged)
        inp = MemoryConsolidationInput(
            question=state.get("question", ""), memory=raw_blob
        )
        out, _ = await execute_role_lc(registry, MEMORY_CONSOLIDATOR, inp)
        return {"consolidated_memory": _consolidated_to_text(out)}

    async def open_ie(state: MemoryUpdateState) -> Dict[str, Any]:
        consolidated = list(state.get("consolidated_memory") or [])
        if not consolidated:
            return {"extracted_relations": []}

        entity_dict = dict(state.get("entity_dict") or {})
        known_entities: List[WikidataEntity] = [
            e for e in entity_dict.values() if isinstance(e, WikidataEntity)
        ]

        # Strip provenance tags before extraction so "[Retrieval]" /
        # "[System Prediction]" never leak into entity/relation surface forms.
        inp = OpenIEInput(
            text="\n".join(_strip_provenance_tag(c) for c in consolidated),
            known_entities=known_entities or None,
        )
        out, _ = await execute_role_lc(registry, OPEN_IE, inp)
        relations = list(getattr(out, "relations", []) or [])
        entities = list(getattr(out, "entities", []) or [])
        return {
            "extracted_entities": entities,
            "extracted_relations": relations,
        }

    async def link_entities_node(state: MemoryUpdateState) -> Dict[str, Any]:
        entity_dict = dict(state.get("entity_dict") or {})
        entities = list(state.get("extracted_entities") or [])
        relations = list(state.get("extracted_relations") or [])

        candidate_names = _collect_unlinked_entity_names(
            entities, relations, entity_dict
        )
        linked_map: Dict[str, str] = {}
        if candidate_names:
            results = await link_entities.ainvoke({"entity_names": candidate_names})
            for row in results or []:
                if not isinstance(row, dict):
                    continue
                qid = row.get("qid")
                name = row.get("name")
                if not qid or not name:
                    continue
                linked_map[name] = qid
                if qid not in entity_dict:
                    entity_dict[qid] = WikidataEntity(
                        qid=qid,
                        label=name,
                        description=row.get("description", "") or "",
                    )

        return {"linked_entities": linked_map, "updated_entity_dict": entity_dict}

    async def merge_and_prune(state: MemoryUpdateState) -> Dict[str, Any]:
        question = state.get("question", "")
        relations = list(state.get("extracted_relations") or [])
        kg_strings: set[str] = set()
        for raw in state.get("new_raw_triples") or []:
            coerced = _coerce_raw_triple_to_relation(raw)
            if coerced is not None:
                relations.append(coerced)
                kg_strings.add(_stringify_triple(coerced))

        linked_map = dict(state.get("linked_entities") or {})
        entity_dict = dict(
            state.get("updated_entity_dict") or state.get("entity_dict") or {}
        )

        # Enrich relations with newly linked QIDs.
        for rel in relations:
            if not rel.subject_id:
                rel.subject_id = linked_map.get(rel.subject)
            if not rel.object_id:
                rel.object_id = linked_map.get(rel.object)

        # Work on a copy — never mutate the caller's graph.
        source_graph = state.get("current_graph") or nx.DiGraph()
        new_graph: nx.DiGraph = source_graph.copy()

        # Filter to **newly proposed** relations only — pruner is expensive and
        # should never re-examine edges already in the graph.
        relations = [
            rel
            for rel in relations
            if rel.relation and not _relation_already_in_graph(source_graph, rel)
        ]

        if relations:
            triple_strings = [_stringify_triple(rel) for rel in relations]
            chunks = [
                triple_strings[i : i + batch_size]
                for i in range(0, len(triple_strings), batch_size)
            ]
            chunk_relations = [
                relations[i : i + batch_size]
                for i in range(0, len(relations), batch_size)
            ]
            kept_relations: List[Relation] = []
            inputs = [
                TriplePruneInput(question=question, triples=chunk) for chunk in chunks
            ]

            if inputs:
                outputs, _ = await execute_role_lc(registry, TRIPLE_PRUNER, inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                for batch, out in zip(chunk_relations, outputs):
                    if out is None or not hasattr(out, "keep_indices"):
                        raise RuntimeError(
                            f"triple_pruner returned no keep_indices for batch of {len(batch)} relations"
                        )
                    keep = {idx for idx in out.keep_indices if 0 <= idx < len(batch)}
                    kept_relations.extend(
                        batch[i] for i in range(len(batch)) if i in keep
                    )
            else:
                kept_relations = list(relations)

            for rel in kept_relations:
                _add_triple_to_graph(new_graph, rel, entity_dict)

            _merge_same_qid_nodes(new_graph)
            kept_triple_strings = [_stringify_triple(r) for r in kept_relations]
        else:
            kept_triple_strings = []

        return {
            "updated_graph": new_graph,
            "updated_entity_dict": entity_dict,
            "kept_triples": kept_triple_strings,
            "kg_triple_strings": sorted(kg_strings),
        }

    async def textualize_graph(state: MemoryUpdateState) -> Dict[str, Any]:
        consolidated = list(state.get("consolidated_memory") or [])
        kept_triples: List[str] = list(state.get("kept_triples") or [])
        if not kept_triples:
            return {"consolidated_memory": consolidated}

        # Triples fetched from the knowledge graph are retrieval evidence;
        # tagging them [System Prediction] would let consolidation's
        # conflict-resolution rule discard a true Wikidata fact in favor of a
        # conflicting [Retrieval]-tagged extractor claim.
        kg_set = set(state.get("kg_triple_strings") or [])
        appended = consolidated + [
            _format_memory_item(
                t,
                SourceType.RETRIEVAL if t in kg_set else SourceType.SYSTEM_PREDICTION,
            )
            for t in kept_triples
        ]
        return {"consolidated_memory": appended}

    async def consolidate_post(state: MemoryUpdateState) -> Dict[str, Any]:
        consolidated = list(state.get("consolidated_memory") or [])
        if not consolidated:
            return {"updated_text_memory": []}

        raw_blob = _format_lines(consolidated)
        inp = MemoryConsolidationInput(
            question=state.get("question", ""), memory=raw_blob
        )
        out, _ = await execute_role_lc(registry, MEMORY_CONSOLIDATOR, inp)
        return {"updated_text_memory": _consolidated_to_text(out)}

    async def finalize_memory(state: MemoryUpdateState) -> Dict[str, Any]:
        return {"updated_text_memory": list(state.get("consolidated_memory") or [])}

    def route_after_textualize(state: MemoryUpdateState) -> str:
        if state.get("kept_triples"):
            return "consolidate_post"
        return "finalize_memory"

    builder = StateGraph(MemoryUpdateState)
    builder.add_node("consolidate_pre", consolidate_pre)
    builder.add_node("open_ie", open_ie)
    builder.add_node("link_entities", link_entities_node)
    builder.add_node("merge_and_prune", merge_and_prune)
    builder.add_node("textualize_graph", textualize_graph)
    builder.add_node("consolidate_post", consolidate_post)
    builder.add_node("finalize_memory", finalize_memory)

    builder.add_edge(START, "consolidate_pre")
    builder.add_edge("consolidate_pre", "open_ie")
    builder.add_edge("open_ie", "link_entities")
    builder.add_edge("link_entities", "merge_and_prune")
    builder.add_edge("merge_and_prune", "textualize_graph")
    builder.add_conditional_edges(
        "textualize_graph",
        route_after_textualize,
        ["consolidate_post", "finalize_memory"],
    )
    builder.add_edge("consolidate_post", END)
    builder.add_edge("finalize_memory", END)

    return builder.compile()


__all__ = ["MemoryUpdateState", "build_memory_update_graph"]
