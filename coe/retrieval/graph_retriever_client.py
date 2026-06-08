"""GraphRetrieverClient — serves a pre-sliced Freebase subgraph as a KB.

Drop-in replacement for WikidataClient in freebase_subgraph evaluation mode.
Loads a preprocessed cache entry produced by freebase_preprocess.py and
exposes the same async interface used by NodeGenerator and entity_linking.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from coe.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple

logger = logging.getLogger(__name__)

# Regex to detect Freebase MIDs (CVT and regular entity nodes stored as MIDs).
_MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")


def _is_mid(s: str) -> bool:
    return bool(_MID_RE.match(s))


class GraphRetrieverClient:
    """Serves a per-question Freebase subgraph as a KB.

    Usage:
        client = GraphRetrieverClient(reranker=reranker, embed_client=embed_client)
        seed_qids = client.load_from_cache(cache_entry)
        triples   = await client.aget_k_hop_triples(seed_qids[:1], k=1)
    """

    def __init__(self, reranker=None, embed_client=None):
        """
        Args:
            reranker: optional Reranker used to re-score entity candidates in asearch_entities.
            embed_client: optional LLMClient(is_embedding=True) used to embed text queries
                for cosine-similarity entity search.  Falls back to string matching if None.
        """
        self._reranker = reranker
        self._embed_client = embed_client

        # Populated by load_from_cache()
        self._entities_by_qid: Dict[str, WikidataEntity] = {}
        self._entities_by_label: Dict[str, str] = {}  # lowercase label -> qid
        self._properties_by_pid: Dict[str, WikidataProperty] = {}
        self._adj_out: Dict[str, List[Tuple[str, str]]] = {}  # qid -> [(pid, obj_qid_or_str)]
        self._adj_in: Dict[str, List[Tuple[str, str]]] = {}   # qid -> [(pid, subj_qid)]
        self._embeddings: Dict[str, List[float]] = {}          # qid -> embedding vector
        self._cvt_qids: Set[str] = set()                       # Q-IDs of CVT record nodes
        self._q_entity_qids: List[str] = []

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def load_from_cache(self, cache_entry: dict) -> List[str]:
        """Load a preprocessed cache entry; return seed Q-IDs for q_entity texts.

        Args:
            cache_entry: one JSON object from the subgraph_cache.jsonl file.

        Returns:
            List of Q-IDs corresponding to the question's seed entities.
        """
        self._entities_by_qid = {}
        self._entities_by_label = {}
        self._properties_by_pid = {}
        self._adj_out = {}
        self._adj_in = {}
        self._embeddings = {}
        self._cvt_qids = set()

        # Load entities
        for qid, entity_data in cache_entry.get("entity_dict", {}).items():
            is_cvt = entity_data.pop("is_cvt_record", False)
            entity = WikidataEntity(
                qid=entity_data.get("qid", qid),
                label=entity_data.get("label"),
                description=entity_data.get("description", ""),
            )
            self._entities_by_qid[qid] = entity
            if is_cvt:
                self._cvt_qids.add(qid)
            elif entity.label:
                self._entities_by_label[entity.label.lower()] = qid

        # Load properties
        for pid, prop_data in cache_entry.get("property_dict", {}).items():
            self._properties_by_pid[pid] = WikidataProperty(
                pid=prop_data.get("pid", pid),
                label=prop_data.get("label"),
                description=prop_data.get("description", ""),
            )

        # Load triples and build adjacency lists
        for triple in cache_entry.get("triples", []):
            if len(triple) != 3:
                continue
            subj_qid, prop_pid, obj_id = triple

            if subj_qid not in self._adj_out:
                self._adj_out[subj_qid] = []
            self._adj_out[subj_qid].append((prop_pid, obj_id))

            if obj_id in self._entities_by_qid:
                if obj_id not in self._adj_in:
                    self._adj_in[obj_id] = []
                self._adj_in[obj_id].append((prop_pid, subj_qid))

        # Load pre-computed embeddings (qid -> vector)
        self._embeddings = {
            qid: vec
            for qid, vec in cache_entry.get("embeddings", {}).items()
        }

        self._q_entity_qids = cache_entry.get("q_entity_qids", [])
        return list(self._q_entity_qids)

    # ------------------------------------------------------------------
    # Synchronous k-hop traversal
    # ------------------------------------------------------------------

    def get_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Return k-hop WikiTriple lists from seed Q-IDs (local graph only)."""
        is_single = isinstance(qids, str)
        raw_list = [qids] if is_single else list(qids)

        all_results: List[List[WikiTriple]] = []
        for seed_qid in raw_list:
            collected: List[WikiTriple] = []
            visited: Set[str] = set()
            frontier: Set[str] = {seed_qid}

            for _ in range(k):
                next_frontier: Set[str] = set()
                for qid in frontier:
                    if qid in visited:
                        continue
                    visited.add(qid)

                    # Outgoing triples
                    for pid, obj_id in self._adj_out.get(qid, []):
                        subj = self._entities_by_qid.get(qid, WikidataEntity(qid=qid))
                        prop = self._properties_by_pid.get(pid, WikidataProperty(pid=pid))
                        if obj_id in self._entities_by_qid:
                            obj: Any = self._entities_by_qid[obj_id]
                            # Only expand non-CVT entity nodes into next hop
                            if obj_id not in visited and obj_id not in self._cvt_qids:
                                next_frontier.add(obj_id)
                        else:
                            obj = obj_id  # string literal value
                        collected.append(WikiTriple(subject=subj, relation=prop, object=obj))

                    if bidirectional:
                        # Incoming triples
                        for pid, subj_id in self._adj_in.get(qid, []):
                            subj = self._entities_by_qid.get(subj_id, WikidataEntity(qid=subj_id))
                            prop = self._properties_by_pid.get(pid, WikidataProperty(pid=pid))
                            obj = self._entities_by_qid.get(qid, WikidataEntity(qid=qid))
                            if subj_id not in visited and subj_id not in self._cvt_qids:
                                next_frontier.add(subj_id)
                            collected.append(WikiTriple(subject=subj, relation=prop, object=obj))

                frontier = next_frontier
                if not frontier:
                    break

            # Deduplicate by hash
            seen_hashes: Set[int] = set()
            dedup: List[WikiTriple] = []
            for t in collected:
                h = hash(t)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    dedup.append(t)

            all_results.append(dedup)

        return all_results[0] if is_single else all_results

    # ------------------------------------------------------------------
    # Synchronous entity search
    # ------------------------------------------------------------------

    def search_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
        is_qids: bool = False,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Find entities in the loaded subgraph.

        * ``is_qids=True``: direct lookup by Q-ID (from Azure entity linking).
        * Otherwise: cosine-similarity (if embed_client) or string match.
        """
        is_single = isinstance(query, str)
        queries = [query] if is_single else list(query)

        results: List[List[WikidataEntity]] = []
        for q in queries:
            if is_qids:
                entity = self._entities_by_qid.get(q)
                results.append([entity] if entity else [])
            elif self._embeddings and self._embed_client is not None:
                results.append(self._embed_search(q, num_results))
            else:
                results.append(self._string_search(q, num_results))

        return results[0] if is_single else results

    def _embed_search(self, query: str, num_results: int) -> List[WikidataEntity]:
        """Cosine-similarity search using pre-computed entity embeddings."""
        import numpy as np

        q_vec = self._embed_client.get_embeddings(query)
        if q_vec is None:
            return self._string_search(query, num_results)

        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm == 0:
            return self._string_search(query, num_results)
        q_arr = q_arr / q_norm

        scored: List[Tuple[float, str]] = []
        for qid, vec in self._embeddings.items():
            if qid in self._cvt_qids:
                continue
            e_arr = np.array(vec, dtype=np.float32)
            e_norm = np.linalg.norm(e_arr)
            if e_norm == 0:
                continue
            sim = float(np.dot(q_arr, e_arr / e_norm))
            scored.append((sim, qid))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [
            self._entities_by_qid[qid]
            for _, qid in scored[:num_results]
            if qid in self._entities_by_qid
        ]

        if self._reranker and len(candidates) > 1:
            docs = [str(c) for c in candidates]
            try:
                ranked_indices, _, _ = self._reranker.client.get_reranking_scores(query, docs)
                candidates = [candidates[i] for i in ranked_indices[:num_results]]
            except Exception as e:
                logger.debug("Reranker failed in embed_search: %s", e)

        return candidates

    def _string_search(self, query: str, num_results: int) -> List[WikidataEntity]:
        """Fuzzy string match against entity labels."""
        q_lower = query.lower().strip()

        # Exact match first
        if q_lower in self._entities_by_label:
            qid = self._entities_by_label[q_lower]
            return [self._entities_by_qid[qid]]

        # Score by token overlap
        q_tokens = set(q_lower.split())
        scored: List[Tuple[float, str]] = []
        for label_lower, qid in self._entities_by_label.items():
            if qid in self._cvt_qids:
                continue
            l_tokens = set(label_lower.split())
            if not l_tokens:
                continue
            inter = len(q_tokens & l_tokens)
            union = len(q_tokens | l_tokens)
            score = inter / union if union else 0.0
            # Bonus for contains
            if q_lower in label_lower or label_lower in q_lower:
                score += 0.5
            if score > 0:
                scored.append((score, qid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            self._entities_by_qid[qid]
            for _, qid in scored[:num_results]
            if qid in self._entities_by_qid
        ]

    # ------------------------------------------------------------------
    # Entity enrichment (labels already loaded — no-op lookup)
    # ------------------------------------------------------------------

    def enrich_entities(
        self, entities: List[WikidataEntity], get_details: bool = False,
    ) -> List[WikidataEntity]:
        """Return entities enriched with labels from the loaded subgraph."""
        enriched = []
        for e in entities:
            known = self._entities_by_qid.get(e.qid)
            enriched.append(known if known is not None else e)
        return enriched

    # ------------------------------------------------------------------
    # Cache management (no-op — nothing to clear)
    # ------------------------------------------------------------------

    def clear_triple_caches(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def aget_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        return await asyncio.to_thread(self.get_k_hop_triples, qids, k, bidirectional, enrich)

    async def asearch_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
        is_qids: bool = False,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        return await asyncio.to_thread(
            self.search_entities, query, num_results, get_details, is_qids,
        )

    async def aenrich_entities(
        self, entities: List[WikidataEntity], get_details: bool = False,
    ) -> List[WikidataEntity]:
        return await asyncio.to_thread(self.enrich_entities, entities, get_details)
