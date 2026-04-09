"""FreebaseClient — live Freebase SPARQL client for freebase_live evaluation mode.

Drop-in replacement for WikidataClient / GraphRetrieverClient.
Queries a running Virtuoso endpoint (http://n0387:3001/sparql) and performs
text-to-MID entity search via Wikidata search + local QID->MID mapping, with
SPARQL CONTAINS as fallback.

CVT nodes are handled by on-the-fly flattening, matching the output format of
freebase_preprocess.py so the LLM sees consistent triple representations
regardless of which Freebase access mode is used.
"""

import asyncio
import json
import logging
import pickle
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON

from wemg.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FB_NS = "http://rdf.freebase.com/ns/"

# Predicate URI prefixes (after stripping FB_NS) to skip — schema/meta relations.
SKIP_PREFIXES = (
    "rdf-schema#",
    "type.object.name",
    "type.object.type",
    "type.type.",
    "kg.",
    "freebase.",
    "common.topic.webpage",
    "base.",
    "user.",
    "dataworld.",
)

# Rate limiter for Virtuoso SPARQL (be polite; Virtuoso handles concurrent well but
# we keep a modest ceiling to avoid hammering it during large evals).
_SPARQL_RATE_LOCK = threading.Lock()
_SPARQL_LAST_REQUEST: List[float] = [0.0]
_SPARQL_MIN_INTERVAL = 0.01  # 100 RPS max
_SPARQL_VALUES_BATCH_SIZE = 200


def _sparql_rate_limit() -> None:
    with _SPARQL_RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _SPARQL_LAST_REQUEST[0]
        if elapsed < _SPARQL_MIN_INTERVAL:
            time.sleep(_SPARQL_MIN_INTERVAL - elapsed)
        _SPARQL_LAST_REQUEST[0] = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_ns(uri: str) -> str:
    """Strip the Freebase namespace prefix from a URI, return the local name."""
    return uri.replace(FB_NS, "")


def _is_mid(s: str) -> bool:
    """Return True if s looks like a Freebase MID (m.xxx or g.xxx)."""
    return (s.startswith("m.") or s.startswith("g.")) and len(s) > 2


def _normalize_mid(mid: str) -> Optional[str]:
    """Normalize MID strings to dot format (m.xxx / g.xxx)."""
    if not isinstance(mid, str):
        return None
    value = mid.strip()
    if not value:
        return None
    if value.startswith("/m/"):
        value = f"m.{value[3:]}"
    elif value.startswith("/g/"):
        value = f"g.{value[3:]}"
    return value if _is_mid(value) else None


def _humanize(freebase_path: str) -> str:
    """Convert a Freebase relation path to a human-readable label.

    Matches freebase_preprocess._humanize exactly so triple labels are
    consistent between subgraph and live modes.

    Examples:
        "baseball.baseball_team.roster"  → "roster"
        "people.person.place_of_birth"   → "place of birth"
        "p1::p2"                         → "label1 label2"
    """
    if "::" in freebase_path:
        r1, r2 = freebase_path.split("::", 1)
        return f"{_humanize(r1)} {_humanize(r2)}"
    last = freebase_path.split(".")[-1]
    return last.replace("_", " ")


def _should_skip(rel: str) -> bool:
    """Return True if this Freebase relation should be filtered out."""
    return any(rel.startswith(p) for p in SKIP_PREFIXES)


def _chunked(items: List[str], size: int) -> List[List[str]]:
    """Return deterministic chunks for VALUES-style SPARQL batching."""
    return [items[i:i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# FreebaseClient
# ---------------------------------------------------------------------------

class FreebaseClient:
    """Live Freebase SPARQL client — drop-in for WikidataClient/GraphRetrieverClient.

    Args:
        sparql_url: Virtuoso HTTP SPARQL endpoint, e.g. "http://localhost:8890/sparql".
    """

    def __init__(
        self,
        sparql_url: str = "http://n0387:3001/sparql",
        wikidata_client=None,
        qid_to_mid_map_path: Optional[str] = None,
        qid_to_mid_candidates: int = 5,
    ) -> None:
        self._sparql_url = sparql_url
        self._wikidata_client = wikidata_client
        self._qid_to_mid_candidates = max(1, int(qid_to_mid_candidates))
        self._sparql = SPARQLWrapper(sparql_url)
        self._sparql.setReturnFormat(SPARQL_JSON)

        # In-memory caches (per-instance; cleared by clear_triple_caches)
        self._triple_cache: Dict[str, List[WikiTriple]] = {}
        self._label_cache: Dict[str, Optional[str]] = {}
        self._cvt_cache: Dict[str, bool] = {}
        self._qid_to_mids = self._load_qid_to_mid_map(qid_to_mid_map_path)

    @staticmethod
    def _load_qid_to_mid_map(path: Optional[str]) -> Dict[str, List[str]]:
        """Load QID->MID map from local JSON/JSONL/PKL file.

        Supported formats:
        - .pkl/.pickle: dict[str, str|list[str]]
        - .json: dict[str, str|list[str]]
        - .jsonl: one object per line with keys {qid, mid} or {qid, mids}
        """
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            logger.warning("QID->MID map file not found: %s", p)
            return {}

        raw: Dict[str, Any] = {}
        try:
            suffix = p.suffix.lower()
            if suffix in (".pkl", ".pickle"):
                with p.open("rb") as f:
                    loaded = pickle.load(f)
                raw = loaded if isinstance(loaded, dict) else {}
            elif suffix == ".json":
                with p.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                raw = loaded if isinstance(loaded, dict) else {}
            elif suffix == ".jsonl":
                entries: Dict[str, Any] = {}
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        qid = str(row.get("qid") or "").strip().upper()
                        if not qid.startswith("Q"):
                            continue
                        if "mids" in row and isinstance(row["mids"], list):
                            entries[qid] = row["mids"]
                        elif "mid" in row:
                            entries[qid] = row["mid"]
                raw = entries
            else:
                logger.warning("Unsupported QID->MID map format for %s", p)
                return {}
        except Exception as e:
            logger.warning("Failed to load QID->MID map from %s: %s", p, e)
            return {}

        normalized: Dict[str, List[str]] = {}
        for qid_raw, mids_raw in raw.items():
            qid = str(qid_raw).strip().upper()
            if not qid.startswith("Q"):
                continue
            if isinstance(mids_raw, list):
                mids = [_normalize_mid(m) for m in mids_raw]
            else:
                mids = [_normalize_mid(mids_raw)]
            mids = [m for m in mids if m]
            if not mids:
                continue
            normalized[qid] = sorted(set(mids))

        logger.info("Loaded QID->MID map: %d QIDs from %s", len(normalized), p)
        return normalized

    # ------------------------------------------------------------------
    # Internal SPARQL helpers
    # ------------------------------------------------------------------

    def _sparql_query(self, sparql: str, retries: int = 3) -> List[Dict]:
        """Execute a SPARQL SELECT query and return result bindings."""
        _sparql_rate_limit()
        for attempt in range(retries):
            try:
                self._sparql.setQuery(sparql)
                results = self._sparql.query().convert()
                return results.get("results", {}).get("bindings", [])
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.warning("SPARQL query failed after %d retries: %s\nQuery: %s", retries, e, sparql[:200])
                    return []
        return []

    def _sparql_ask(self, sparql: str, retries: int = 3) -> bool:
        """Execute a SPARQL ASK query and return the boolean result."""
        _sparql_rate_limit()
        for attempt in range(retries):
            try:
                self._sparql.setQuery(sparql)
                results = self._sparql.query().convert()
                return bool(results.get("boolean", False))
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.warning("SPARQL ASK failed after %d retries: %s", retries, e)
                    return False
        return False

    def _get_label(self, mid: str) -> Optional[str]:
        """Fetch type.object.name (English) for a MID; cached."""
        if mid in self._label_cache:
            return self._label_cache[mid]
        sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT ?name WHERE {{
  ns:{mid} ns:type.object.name ?name .
  FILTER(LANG(?name) = 'en')
}}
LIMIT 1
"""
        bindings = self._sparql_query(sparql)
        label = bindings[0]["name"]["value"] if bindings else None
        self._label_cache[mid] = label
        return label

    def _get_labels_batch(self, mids: List[str]) -> Dict[str, Optional[str]]:
        """Fetch English labels for many MIDs using VALUES batching."""
        valid_mids = sorted({_normalize_mid(mid) for mid in mids if _normalize_mid(mid)})
        if not valid_mids:
            return {}

        to_fetch = [mid for mid in valid_mids if mid not in self._label_cache]
        for chunk in _chunked(to_fetch, _SPARQL_VALUES_BATCH_SIZE):
            values = " ".join(f"ns:{mid}" for mid in chunk)
            sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT ?mid ?name WHERE {{
  VALUES ?mid {{ {values} }}
  ?mid ns:type.object.name ?name .
  FILTER(LANG(?name) = 'en')
}}
"""
            bindings = self._sparql_query(sparql)
            seen = set()
            for b in bindings:
                mid_key = _strip_ns(b["mid"]["value"])
                if mid_key in seen:
                    continue
                seen.add(mid_key)
                self._label_cache[mid_key] = b["name"]["value"]
            for mid in chunk:
                self._label_cache.setdefault(mid, None)

        return {mid: self._label_cache.get(mid) for mid in valid_mids}

    def _is_cvt(self, mid: str) -> bool:
        """Return True if mid is a CVT node (has no type.object.name)."""
        if mid in self._cvt_cache:
            return self._cvt_cache[mid]
        sparql = f"""
PREFIX ns: <{FB_NS}>
ASK {{
  ns:{mid} ns:type.object.name ?name .
  FILTER(LANG(?name) = 'en')
}}
"""
        has_name = self._sparql_ask(sparql)
        is_cvt = not has_name
        self._cvt_cache[mid] = is_cvt
        return is_cvt

    def _is_cvt_batch(self, mids: List[str]) -> Dict[str, bool]:
        """Return CVT flags for many MIDs using VALUES batching."""
        valid_mids = sorted({_normalize_mid(mid) for mid in mids if _normalize_mid(mid)})
        if not valid_mids:
            return {}

        to_fetch = [mid for mid in valid_mids if mid not in self._cvt_cache]
        for chunk in _chunked(to_fetch, _SPARQL_VALUES_BATCH_SIZE):
            values = " ".join(f"ns:{mid}" for mid in chunk)
            sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT DISTINCT ?mid WHERE {{
  VALUES ?mid {{ {values} }}
  ?mid ns:type.object.name ?name .
  FILTER(LANG(?name) = 'en')
}}
"""
            bindings = self._sparql_query(sparql)
            mids_with_name = {_strip_ns(b["mid"]["value"]) for b in bindings}
            for mid in chunk:
                self._cvt_cache[mid] = mid not in mids_with_name

        return {mid: self._cvt_cache[mid] for mid in valid_mids}

    def _get_outgoing(self, mid: str) -> List[Tuple[str, str]]:
        """Return [(rel_local, obj_value)] for all outgoing edges from mid.

        rel_local has FB_NS stripped. obj_value is the raw IRI local name
        (for entity nodes) or the literal string (for values).
        Filtered by SKIP_PREFIXES.
        """
        return self._get_outgoing_batch([mid]).get(mid, [])

    def _get_outgoing_batch(self, mids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        """Return outgoing edges for many MIDs via VALUES batching."""
        valid_mids = sorted({_normalize_mid(mid) for mid in mids if _normalize_mid(mid)})
        if not valid_mids:
            return {}

        out: Dict[str, List[Tuple[str, str]]] = {mid: [] for mid in valid_mids}
        for chunk in _chunked(valid_mids, _SPARQL_VALUES_BATCH_SIZE):
            values = " ".join(f"ns:{mid}" for mid in chunk)
            sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT ?s ?p ?o WHERE {{
  VALUES ?s {{ {values} }}
  ?s ?p ?o .
}}
"""
            bindings = self._sparql_query(sparql)
            for b in bindings:
                subj = _strip_ns(b["s"]["value"])
                p_uri = b["p"]["value"]
                # Only keep Freebase namespace predicates. Virtuoso may also return
                # RDF/OWL schema predicates (e.g., rdf:type) that are not useful
                # for QA retrieval and would surface as raw IDs in output.
                if not p_uri.startswith(FB_NS):
                    continue
                rel = _strip_ns(p_uri)
                if _should_skip(rel):
                    continue

                obj_binding = b["o"]
                if obj_binding["type"] == "uri":
                    obj_val = _strip_ns(obj_binding["value"])
                else:
                    # Keep English or untagged literals; drop other languages to
                    # avoid noisy multilingual name variants in retrieval output.
                    lang = (obj_binding.get("xml:lang") or "").lower()
                    if lang and lang != "en":
                        continue
                    obj_val = obj_binding["value"]
                out.setdefault(subj, []).append((rel, obj_val))

        return out

    def _expand_cvt(self, cvt_mid: str) -> List[Tuple[str, str]]:
        """Return [(rel_local, obj_value)] for all outgoing edges from a CVT node."""
        return self._get_outgoing(cvt_mid)

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------

    def search_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
        is_qids: bool = False,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Search for Freebase entities.

        Args:
            query: Entity name(s) or MID(s).
            num_results: Max results per query.
            get_details: Unused (kept for interface compatibility).
            is_qids: If True, query is already a MID — do direct lookup.
        """
        is_single = isinstance(query, str)
        queries = [query] if is_single else list(query)

        all_results: List[List[WikidataEntity]] = []
        for q in queries:
            if is_qids:
                all_results.append(self._lookup_by_mid(q))
            elif self._qid_to_mids and self._wikidata_client is not None:
                mapped = self._wikidata_search_then_map(q, num_results)
                if mapped:
                    all_results.append(mapped)
                else:
                    raise RuntimeError(
                        "Freebase text search fallback is disabled. "
                        f"No QID->MID mapping result for query {q!r}. "
                        "Provide valid qid_to_mid_map_path coverage or query by MID using is_qids=True."
                    )
            else:
                raise RuntimeError(
                    "Freebase text search fallback is disabled. "
                    "search_entities requires wikidata_client + qid_to_mid_map_path for text queries, "
                    "or use MID lookups with is_qids=True."
                )

        return all_results[0] if is_single else all_results

    def _lookup_by_mid(self, mid: str) -> List[WikidataEntity]:
        """Direct MID → WikidataEntity lookup."""
        normalized = _normalize_mid(mid)
        if not normalized:
            return []
        label = self._get_label(normalized)
        return [WikidataEntity(qid=normalized, label=label)]

    def _wikidata_search_then_map(self, text: str, num_results: int) -> List[WikidataEntity]:
        """Find Wikidata entities by text, then map candidate QIDs to Freebase MIDs."""
        try:
            wd_entities = self._wikidata_client.search_entities(
                text,
                num_results=self._qid_to_mid_candidates,
                get_details=False,
            )
        except Exception as e:
            logger.warning("Wikidata search failed for %r: %s", text, e)
            return []

        if not isinstance(wd_entities, list):
            return []

        seen: set = set()
        ordered_mids: List[str] = []
        fallback_labels: Dict[str, Optional[str]] = {}
        for entity in wd_entities:
            qid = getattr(entity, "qid", None)
            if not isinstance(qid, str):
                continue
            mids = self._qid_to_mids.get(qid.upper(), [])
            for mid in mids:
                if mid in seen:
                    continue
                seen.add(mid)
                ordered_mids.append(mid)
                fallback_labels[mid] = getattr(entity, "label", None)
                if len(ordered_mids) >= num_results:
                    break
            if len(ordered_mids) >= num_results:
                break

        if not ordered_mids:
            return []

        label_map = self._get_labels_batch(ordered_mids)
        return [
            WikidataEntity(qid=mid, label=label_map.get(mid) or fallback_labels.get(mid) or mid)
            for mid in ordered_mids
        ]

    def get_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = False,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Return k-hop triples from seed MIDs with on-the-fly CVT flattening.

        CVT nodes are transparently expanded: for each (entity, p, cvt) edge,
        we emit:
          - compound triples: (entity, p::p2, o2) for each (cvt, p2, o2)
          - one synthesized string triple: (entity, p, "p2a: val; p2b: val2")

        This matches freebase_preprocess.py's output format exactly.
        """
        is_single = isinstance(qids, str)
        raw_list = [qids] if is_single else list(qids)

        all_results: List[List[WikiTriple]] = []
        for seed_mid in raw_list:
            if seed_mid in self._triple_cache:
                all_results.append(self._triple_cache[seed_mid])
                continue
            triples = self._get_triples_for_mid(seed_mid)
            self._triple_cache[seed_mid] = triples
            all_results.append(triples)

        return all_results[0] if is_single else all_results

    def _get_triples_for_mid(self, mid: str) -> List[WikiTriple]:
        """Fetch 1-hop triples for a MID with full CVT flattening."""
        if not _is_mid(mid):
            return []

        subj_label = self._get_labels_batch([mid]).get(mid)
        subj_entity = WikidataEntity(qid=mid, label=subj_label)

        outgoing = self._get_outgoing(mid)
        triples: List[WikiTriple] = []
        seen: set = set()

        object_mids = sorted({obj_val for _, obj_val in outgoing if _is_mid(obj_val)})
        cvt_map = self._is_cvt_batch(object_mids)
        cvt_mids = [obj_mid for obj_mid in object_mids if cvt_map.get(obj_mid, False)]
        cvt_outgoing = self._get_outgoing_batch(cvt_mids) if cvt_mids else {}

        regular_entity_mids = [obj_mid for obj_mid in object_mids if not cvt_map.get(obj_mid, False)]
        cvt_entity_mids = sorted({
            p2_obj
            for attrs in cvt_outgoing.values()
            for _, p2_obj in attrs
            if _is_mid(p2_obj)
        })
        label_map = self._get_labels_batch(regular_entity_mids + cvt_entity_mids)

        for rel, obj_val in outgoing:
            if _is_mid(obj_val):
                # Check if CVT
                if cvt_map.get(obj_val, False):
                    # Expand CVT transparently
                    cvt_attrs = cvt_outgoing.get(obj_val, [])
                    # Build synthesized string
                    synth_parts = []
                    for p2_rel, p2_obj in cvt_attrs:
                        p2_label = _humanize(p2_rel)
                        if _is_mid(p2_obj):
                            p2_obj_label = label_map.get(p2_obj) or p2_obj
                        else:
                            p2_obj_label = p2_obj
                        synth_parts.append(f"{p2_label}: {p2_obj_label}")

                        # Compound triple: entity --[p::p2]--> o2
                        compound_rel = f"{rel}::{p2_rel}"
                        prop = WikidataProperty(pid=compound_rel, label=_humanize(compound_rel))
                        if _is_mid(p2_obj):
                            obj_label = label_map.get(p2_obj) or p2_obj
                            obj = WikidataEntity(qid=p2_obj, label=obj_label)
                        else:
                            obj = p2_obj
                        key = (mid, compound_rel, str(obj))
                        if key not in seen:
                            seen.add(key)
                            triples.append(WikiTriple(subject=subj_entity, relation=prop, object=obj))

                    # Synthesized CVT string triple
                    if synth_parts:
                        synth_str = "; ".join(synth_parts)
                        prop = WikidataProperty(pid=rel, label=_humanize(rel))
                        key = (mid, rel, synth_str)
                        if key not in seen:
                            seen.add(key)
                            triples.append(WikiTriple(subject=subj_entity, relation=prop, object=synth_str))
                else:
                    # Regular entity
                    obj_label = label_map.get(obj_val) or obj_val
                    obj_entity = WikidataEntity(qid=obj_val, label=obj_label)
                    prop = WikidataProperty(pid=rel, label=_humanize(rel))
                    key = (mid, rel, obj_val)
                    if key not in seen:
                        seen.add(key)
                        triples.append(WikiTriple(subject=subj_entity, relation=prop, object=obj_entity))
            else:
                # Literal value
                prop = WikidataProperty(pid=rel, label=_humanize(rel))
                key = (mid, rel, obj_val)
                if key not in seen:
                    seen.add(key)
                    triples.append(WikiTriple(subject=subj_entity, relation=prop, object=obj_val))

        return triples

    def enrich_entities(
        self,
        entities: List[WikidataEntity],
        get_details: bool = False,
    ) -> List[WikidataEntity]:
        """Enrich entities with Freebase labels and descriptions."""
        mids = [e.qid for e in entities if _is_mid(e.qid)]
        label_map = self._get_labels_batch(mids)
        desc_map = self._get_descriptions_batch(mids) if get_details else {}

        enriched = []
        for e in entities:
            if not _is_mid(e.qid):
                enriched.append(e)
                continue
            label = e.label if e.label is not None else label_map.get(e.qid)
            description = desc_map.get(e.qid) if get_details else None
            enriched.append(WikidataEntity(
                qid=e.qid,
                label=label,
                description=description or e.description,
            ))
        return enriched

    def _get_description(self, mid: str) -> Optional[str]:
        """Fetch common.topic.description for a MID."""
        sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT ?desc WHERE {{
  ns:{mid} ns:common.topic.description ?desc .
  FILTER(LANG(?desc) = 'en')
}}
LIMIT 1
"""
        bindings = self._sparql_query(sparql)
        return bindings[0]["desc"]["value"] if bindings else None

    def _get_descriptions_batch(self, mids: List[str]) -> Dict[str, Optional[str]]:
        """Fetch English descriptions for many MIDs using VALUES batching."""
        valid_mids = sorted({_normalize_mid(mid) for mid in mids if _normalize_mid(mid)})
        if not valid_mids:
            return {}

        descriptions: Dict[str, Optional[str]] = {mid: None for mid in valid_mids}
        for chunk in _chunked(valid_mids, _SPARQL_VALUES_BATCH_SIZE):
            values = " ".join(f"ns:{mid}" for mid in chunk)
            sparql = f"""
PREFIX ns: <{FB_NS}>
SELECT ?mid ?desc WHERE {{
  VALUES ?mid {{ {values} }}
  ?mid ns:common.topic.description ?desc .
  FILTER(LANG(?desc) = 'en')
}}
"""
            bindings = self._sparql_query(sparql)
            for b in bindings:
                mid_key = _strip_ns(b["mid"]["value"])
                descriptions[mid_key] = b["desc"]["value"]

        return descriptions

    def clear_triple_caches(self) -> None:
        """Clear fast-changing caches while keeping stable label/CVT caches."""
        self._triple_cache.clear()

    # ------------------------------------------------------------------
    # Async wrappers (same pattern as GraphRetrieverClient)
    # ------------------------------------------------------------------

    async def asearch_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
        is_qids: bool = False,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        return await asyncio.to_thread(
            self.search_entities, query, num_results, get_details, is_qids
        )

    async def aget_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = False,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        return await asyncio.to_thread(
            self.get_k_hop_triples, qids, k, bidirectional, enrich
        )

    async def aenrich_entities(
        self,
        entities: List[WikidataEntity],
        get_details: bool = False,
    ) -> List[WikidataEntity]:
        return await asyncio.to_thread(self.enrich_entities, entities, get_details)
