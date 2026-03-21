"""Consolidated Wikidata module for entity/property retrieval, SPARQL queries,
k-hop triple traversal, and path finding."""

import asyncio
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

import pydantic
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    from wikibase_rest_api_client import Client as WikibaseClient
    from wikibase_rest_api_client.api.items import get_item as wb_get_item
    from wikibase_rest_api_client.api.search import search_entities as wb_search
    WIKIBASE_AVAILABLE = True
except ImportError:
    WIKIBASE_AVAILABLE = False

try:
    from mediawikiapi import MediaWikiAPI
    MEDIAWIKI_AVAILABLE = True
except ImportError:
    MEDIAWIKI_AVAILABLE = False

# Rate limit for Wikipedia page fetches (same default as web crawl to avoid destination rate limits).
DEFAULT_MAX_WIKIPEDIA_REQUESTS_PER_SECOND = 2.0
_wiki_rate_lock = threading.Lock()
_wiki_last_request_time: List[float] = [0.0]


def _wikipedia_rate_limit(max_requests_per_second: float) -> None:
    """Wait if needed so the next Wikipedia request does not exceed the given rate."""
    with _wiki_rate_lock:
        now = time.monotonic()
        min_interval = 1.0 / max(0.1, min(10.0, max_requests_per_second))
        elapsed = now - _wiki_last_request_time[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _wiki_last_request_time[0] = time.monotonic()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class WikidataEntity(pydantic.BaseModel):
    qid: str
    label: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = pydantic.Field(default_factory=list)
    url: Optional[str] = None
    wikipedia_url: Optional[str] = None
    wikipedia_content: Optional[str] = None

    def __hash__(self):
        return hash(self.qid)

    def __eq__(self, other):
        if not isinstance(other, WikidataEntity):
            return False
        return self.qid == other.qid

    def __str__(self):
        parts = [self.label or self.qid]
        if self.description:
            parts.append(f"({self.description})")
        return " ".join(parts)

    def to_context(self, include_wiki_page: bool = False) -> str:
        parts = [f"{self.label or self.qid}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.aliases:
            parts.append(f"Also known as: {', '.join(self.aliases)}")
        if include_wiki_page and self.wikipedia_content:
            parts.append(f"Wikipedia: {self.wikipedia_content[:500]}")
        return "\n".join(parts)


class WikidataProperty(pydantic.BaseModel):
    pid: str
    label: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self):
        return hash(self.pid)

    def __eq__(self, other):
        if not isinstance(other, WikidataProperty):
            return False
        return self.pid == other.pid

    def __str__(self):
        if self.label and self.label.strip():
            return self.label
        if self.description and self.description.strip():
            return self.description
        return self.pid


class WikiTriple(pydantic.BaseModel):
    subject: Any
    relation: Any
    object: Any

    def __hash__(self):
        s = self.subject.qid if hasattr(self.subject, "qid") else str(self.subject)
        r = self.relation.pid if hasattr(self.relation, "pid") else str(self.relation)
        o = self.object.qid if hasattr(self.object, "qid") else str(self.object)
        return hash((s, r, o))

    def __eq__(self, other):
        if not isinstance(other, WikiTriple):
            return False
        return hash(self) == hash(other)

    def __str__(self):
        return f"Subject: {self.subject}\nRelation: {self.relation}\nObject: {self.object}"


class WikidataPathBetweenEntities(pydantic.BaseModel):
    source: WikidataEntity
    target: WikidataEntity
    path: List[WikiTriple] = pydantic.Field(default_factory=list)
    path_length: int = 0

    def __str__(self):
        if not self.path:
            return f"No path found between {self.source} and {self.target}."
        triples_str = "\n---\n".join(
            f"{i + 1}.\n{t}" for i, t in enumerate(self.path)
        )
        return f"Path from {self.source} to {self.target}:\n{triples_str}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "WEMG/0.2.0"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
LIMIT_PER_QUERY = 100
MAX_ENTITIES_PER_HOP = 500
BATCH_SIZE = 25

DEFAULT_PROPERTIES = [
    "P31", "P279", "P27", "P361", "P527", "P495", "P17", "P585", "P106",
    "P569", "P570", "P577", "P50", "P571", "P641", "P625", "P19", "P69",
    "P108", "P136", "P39", "P161", "P20", "P101", "P179", "P175", "P7937",
    "P57", "P607", "P509", "P800", "P449", "P580", "P582", "P276", "P112",
    "P740", "P159", "P452", "P102", "P1142", "P1387", "P1576", "P140",
    "P178", "P287", "P25", "P22", "P40", "P185", "P802", "P1416", "P26",
    "P3373", "P451", "P1038", "P184", "P166", "P512", "P463", "P127",
    "P749", "P355", "P488", "P169", "P131", "P706", "P150", "P36", "P30",
    "P6", "P35", "P1313", "P54", "P1344", "P1532", "P170", "P86", "P162",
    "P58", "P144", "P921", "P407", "P264", "P1411", "P1082", "P2044",
    "P2046",
]

PROPERTY_LABELS: Dict[str, Dict[str, str]] = {
    "P6": {"label": "head of government", "description": "head of the executive power of this governmental body"},
    "P17": {"label": "country", "description": "sovereign state that this item is in"},
    "P19": {"label": "place of birth", "description": "most specific known birth location of a person"},
    "P20": {"label": "place of death", "description": "most specific known death location of a person"},
    "P22": {"label": "father", "description": "male parent of the subject"},
    "P25": {"label": "mother", "description": "female parent of the subject"},
    "P26": {"label": "spouse", "description": "the subject has the object as their spouse"},
    "P27": {"label": "country of citizenship", "description": "the object is a country that recognizes the subject as its citizen"},
    "P30": {"label": "continent", "description": "continent of which the subject is a part"},
    "P31": {"label": "instance of", "description": "that class of which this subject is a particular example and member"},
    "P35": {"label": "head of state", "description": "official with the highest formal authority in a country/state"},
    "P36": {"label": "capital", "description": "seat of government of a country, province, state or other administrative territorial entity"},
    "P39": {"label": "position held", "description": "subject currently or formerly holds the object position or public office"},
    "P40": {"label": "child", "description": "subject has object as child"},
    "P50": {"label": "author", "description": "main creator(s) of a written work"},
    "P54": {"label": "member of sports team", "description": "sports teams or clubs that the subject represents"},
    "P57": {"label": "director", "description": "director(s) of film, TV-series, stageplay, video game or similar"},
    "P58": {"label": "screenwriter", "description": "person(s) who wrote the script for subject item"},
    "P69": {"label": "educated at", "description": "educational institution attended by subject"},
    "P86": {"label": "composer", "description": "person(s) who wrote the music"},
    "P101": {"label": "field of work", "description": "specialization of a person or organization"},
    "P102": {"label": "member of political party", "description": "the political party of which a person is or has been a member"},
    "P106": {"label": "occupation", "description": "occupation of a person"},
    "P108": {"label": "employer", "description": "person or organization for which the subject works or worked"},
    "P112": {"label": "founded by", "description": "founder or co-founder of this organization, religion, place or entity"},
    "P127": {"label": "owned by", "description": "owner of the subject"},
    "P131": {"label": "located in the administrative territorial entity", "description": "the item is located on the territory of the following administrative entity"},
    "P136": {"label": "genre", "description": "creative work's genre or an artist's field of work"},
    "P140": {"label": "religion or worldview", "description": "religion of a person, organization or religious building"},
    "P144": {"label": "based on", "description": "the work(s) used as the basis for subject item"},
    "P150": {"label": "contains the administrative territorial entity", "description": "direct subdivisions of an administrative territorial entity"},
    "P159": {"label": "headquarters location", "description": "city or town where an organization's headquarters is situated"},
    "P161": {"label": "cast member", "description": "actor in the subject production"},
    "P162": {"label": "producer", "description": "person(s) who produced the film, musical work, theatrical production, etc."},
    "P166": {"label": "award received", "description": "award or recognition received by a person, organization or creative work"},
    "P169": {"label": "chief executive officer", "description": "highest-ranking corporate officer appointed as the CEO"},
    "P170": {"label": "creator", "description": "maker of this creative work or other object"},
    "P175": {"label": "performer", "description": "actor, musician, band or other performer associated with this work"},
    "P178": {"label": "developer", "description": "organization or person that developed the item"},
    "P179": {"label": "part of the series", "description": "series which contains the subject"},
    "P184": {"label": "doctoral advisor", "description": "person who supervised the doctorate or PhD thesis"},
    "P185": {"label": "doctoral student", "description": "doctoral student(s) of a professor"},
    "P264": {"label": "record label", "description": "brand and trademark for marketing of music recordings"},
    "P276": {"label": "location", "description": "location of the object, structure or event"},
    "P279": {"label": "subclass of", "description": "this subject is a subclass of that class"},
    "P287": {"label": "designed by", "description": "person or organization which designed the object"},
    "P355": {"label": "has subsidiary", "description": "child organization/unit of an organization/unit"},
    "P361": {"label": "part of", "description": "object of which the subject is a part"},
    "P407": {"label": "language of work or name", "description": "language associated with this creative work or a name"},
    "P449": {"label": "original broadcaster", "description": "network(s) that originally broadcast a radio or television program"},
    "P451": {"label": "unmarried partner", "description": "someone with whom the person is in a relationship without being married"},
    "P452": {"label": "industry", "description": "specific industry of company or organization"},
    "P463": {"label": "member of", "description": "organization, club or musical group to which the subject belongs"},
    "P488": {"label": "chairperson", "description": "presiding member of an organization, group or body"},
    "P495": {"label": "country of origin", "description": "country of origin of this item"},
    "P509": {"label": "cause of death", "description": "underlying or immediate cause of death"},
    "P512": {"label": "academic degree", "description": "academic degree that the person holds"},
    "P527": {"label": "has part(s)", "description": "part of this subject; inverse of 'part of' (P361)"},
    "P569": {"label": "date of birth", "description": "date on which the subject was born"},
    "P570": {"label": "date of death", "description": "date on which the subject died"},
    "P571": {"label": "inception", "description": "time when an entity begins to exist"},
    "P577": {"label": "publication date", "description": "date or point in time when a work was first published or released"},
    "P580": {"label": "start time", "description": "time an entity begins to exist or a statement starts being valid"},
    "P582": {"label": "end time", "description": "moment when an entity ceases to exist or a statement stops being valid"},
    "P585": {"label": "point in time", "description": "date something took place, existed or a statement was true"},
    "P607": {"label": "participated in conflict", "description": "battles, wars or other military engagements in which the subject participated"},
    "P625": {"label": "coordinate location", "description": "geocoordinates of the subject"},
    "P641": {"label": "sport", "description": "sport that the subject participates in or is associated with"},
    "P706": {"label": "located in/on physical feature", "description": "located on the specified geophysical feature"},
    "P740": {"label": "location of formation", "description": "location where a group or organization was formed"},
    "P749": {"label": "parent organization", "description": "parent organization or unit of an organization"},
    "P800": {"label": "notable work", "description": "notable scientific, artistic or literary work of the subject"},
    "P802": {"label": "student", "description": "notable student(s) of the subject individual"},
    "P921": {"label": "main subject", "description": "primary topic of a work or act of communication"},
    "P1038": {"label": "relative", "description": "family member (qualify with kinship to subject P1039)"},
    "P1082": {"label": "population", "description": "number of people inhabiting the place"},
    "P1142": {"label": "political ideology", "description": "political ideology of an organization or person"},
    "P1313": {"label": "office held by head of government", "description": "political office fulfilled by the head of government"},
    "P1344": {"label": "participant in", "description": "event in which a person or organization was a participant"},
    "P1387": {"label": "political alignment", "description": "political position within the left-right political spectrum"},
    "P1411": {"label": "nominated for", "description": "award nomination received by a person, organization or creative work"},
    "P1416": {"label": "affiliation", "description": "organization that a person is affiliated with"},
    "P1532": {"label": "country for sport", "description": "country a person or team represents when playing a sport"},
    "P1576": {"label": "lifestyle", "description": "typical way of life of an individual, group, or culture"},
    "P2044": {"label": "elevation above sea level", "description": "height of the item above a fixed reference point"},
    "P2046": {"label": "area", "description": "area of an entity"},
    "P3373": {"label": "sibling", "description": "the subject and the object have at least one common parent"},
    "P7937": {"label": "form of creative work", "description": "structure of a creative work"},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_qid(qid: str) -> Optional[str]:
    if not qid or not isinstance(qid, str):
        return None
    normalized = qid.strip().upper()
    if re.fullmatch(r"Q\d+", normalized):
        return normalized
    return None


def _normalize_pid(pid: str) -> Optional[str]:
    if not pid or not isinstance(pid, str):
        return None
    normalized = pid.strip().upper()
    if re.fullmatch(r"P\d+", normalized):
        return normalized
    return None


def _extract_id_from_uri(uri: str) -> Optional[str]:
    if not isinstance(uri, str) or not uri:
        return None
    if "/entity/" in uri:
        candidate = uri.rsplit("/", 1)[-1]
    else:
        candidate = uri
    candidate = candidate.strip()
    if re.fullmatch(r"[QP]\d+", candidate, flags=re.IGNORECASE):
        return candidate.upper()
    return None


# ---------------------------------------------------------------------------
# WikidataClient
# ---------------------------------------------------------------------------


class WikidataClient:
    """Unified Wikidata access client combining SPARQL queries, entity/property
    retrieval, k-hop triples, and path finding."""

    def __init__(
        self,
        properties: Optional[List[str]] = None,
        property_labels: Optional[Dict[str, Dict[str, str]]] = None,
        max_wikipedia_requests_per_second: Optional[float] = None,
    ):
        self.properties = properties or list(DEFAULT_PROPERTIES)
        self.property_labels = dict(property_labels or PROPERTY_LABELS)
        self._max_wikipedia_rps = (
            max_wikipedia_requests_per_second
            if max_wikipedia_requests_per_second is not None
            else DEFAULT_MAX_WIKIPEDIA_REQUESTS_PER_SECOND
        )
        self._semaphore = threading.Semaphore(10)
        self._wikibase_client = None
        if WIKIBASE_AVAILABLE:
            try:
                self._wikibase_client = WikibaseClient(
                    base_url="https://www.wikidata.org/w/rest.php/wikibase/v0"
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SPARQL execution
    # ------------------------------------------------------------------

    def _sparql_query(self, sparql: str) -> List[Dict]:
        with self._semaphore:
            client = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
            client.setQuery(sparql)
            client.setReturnFormat(JSON)
            client.addCustomHttpHeader("User-Agent", USER_AGENT)
            for attempt in range(MAX_RETRIES):
                try:
                    results = client.query().convert()
                    return results.get("results", {}).get("bindings", [])
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"SPARQL query failed after {MAX_RETRIES} attempts: {e}")
                        return []
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
        return []

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_property_text(
        pid: str,
        label: Optional[str],
        description: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Normalize Wikidata property label/description.

        - Strip whitespace.
        - Treat PID-like labels (e.g. "P36") as missing.
        - If label is missing but description is present, reuse description
          as a human-facing label.
        """
        label = label.strip() if isinstance(label, str) else None
        description = description.strip() if isinstance(description, str) else None

        if label and re.fullmatch(r"P\d+", label.upper()):
            label = None

        if not label and description:
            label = description

        return (label or None), (description or None)

    def _get_property_label(self, pid: str) -> Tuple[Optional[str], Optional[str]]:
        info = self.property_labels.get(pid, {})
        label, desc = self._normalize_property_text(
            pid,
            info.get("label"),
            info.get("description"),
        )
        return label, desc

    def _is_entity_enriched(self, entity: WikidataEntity) -> bool:
        if not entity or not entity.label:
            return False
        label = entity.label.strip()
        if not label or (label.startswith("Q") and label[1:].isdigit()):
            return False
        return bool(entity.description and entity.description.strip())

    def _is_property_enriched(self, prop: WikidataProperty) -> bool:
        if not prop or not prop.label:
            return False
        label = prop.label.strip()
        return bool(label and not (label.startswith("P") and label[1:].isdigit()))

    def _has_fake_property_label(self, prop: Any) -> bool:
        """Return True if the property has no real label (missing or equals PID)."""
        if prop is None or not hasattr(prop, "pid") or not hasattr(prop, "label"):
            return True
        label = prop.label
        if label is None or not str(label).strip():
            return True
        label = str(label).strip()
        if label.upper() == getattr(prop, "pid", "").upper():
            return True
        if len(label) > 1 and label[0].upper() == "P" and label[1:].isdigit():
            return True
        return False

    # ------------------------------------------------------------------
    # Entity search / retrieval
    # ------------------------------------------------------------------

    def _search_entity_by_text(self, text: str, num_results: int = 1) -> List[WikidataEntity]:
        if WIKIBASE_AVAILABLE and self._wikibase_client:
            try:
                response = wb_search.sync(
                    client=self._wikibase_client,
                    search=text,
                    language="en",
                    limit=num_results,
                )
                items = (
                    response
                    if isinstance(response, list)
                    else getattr(response, "results", []) or []
                )
                entities: List[WikidataEntity] = []
                for item in items:
                    qid = getattr(item, "id", None) or (
                        item.get("id") if isinstance(item, dict) else None
                    )
                    if qid:
                        entities.append(WikidataEntity(
                            qid=qid,
                            label=getattr(item, "label", None)
                            or (item.get("label") if isinstance(item, dict) else None),
                            description=getattr(item, "description", None)
                            or (item.get("description") if isinstance(item, dict) else None),
                            url=f"https://www.wikidata.org/wiki/{qid}",
                        ))
                if entities:
                    return entities[:num_results]
            except Exception as e:
                logger.debug(f"wikibase_rest_api_client search failed: {e}")

        text_escaped = json.dumps(text)
        sparql = f"""
        SELECT ?entity ?ordinal WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                            wikibase:api "EntitySearch" ;
                            mwapi:search {text_escaped} ;
                            mwapi:language "en" ;
                            mwapi:type "item" ;
                            mwapi:limit {num_results} .
            ?entity wikibase:apiOutputItem mwapi:item .
            ?ordinal wikibase:apiOrdinal true .
          }}
        }}
        ORDER BY ASC(?ordinal)
        """
        bindings = self._sparql_query(sparql)
        entities = []
        for row in bindings:
            uri = row.get("entity", {}).get("value", "")
            qid = _extract_id_from_uri(uri)
            if qid and qid.startswith("Q"):
                entities.append(WikidataEntity(
                    qid=qid, url=f"https://www.wikidata.org/wiki/{qid}",
                ))
        return entities[:num_results]

    def _get_entity_by_qid(
        self, qid: str, get_details: bool = False
    ) -> Optional[WikidataEntity]:
        entities = self._get_entities_batch([qid], get_details=get_details)
        return entities.get(qid)

    def _get_entities_batch(
        self, qids: List[str], get_details: bool = False
    ) -> Dict[str, WikidataEntity]:
        if not qids:
            return {}

        entity_map: Dict[str, WikidataEntity] = {}

        for start in range(0, len(qids), BATCH_SIZE):
            batch = qids[start : start + BATCH_SIZE]
            values_clause = " ".join(f"wd:{q}" for q in batch)
            sparql = f"""
            SELECT ?entity ?entityLabel ?entityDescription ?entityAltLabel ?article
            WHERE {{
              VALUES ?entity {{ {values_clause} }}
              OPTIONAL {{
                ?article schema:about ?entity ;
                         schema:isPartOf <https://en.wikipedia.org/> .
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
            }}
            """
            for row in self._sparql_query(sparql):
                uri = row.get("entity", {}).get("value", "")
                if "/entity/" not in uri:
                    continue
                qid_val = uri.split("/")[-1].upper()

                label = row.get("entityLabel", {}).get("value", "")
                if label.startswith("Q") and label[1:].isdigit():
                    label = ""

                description = row.get("entityDescription", {}).get("value", "")

                alt_label = row.get("entityAltLabel", {}).get("value", "")
                aliases = [a.strip() for a in alt_label.split(",") if a.strip()] if alt_label else []

                wikipedia_url = row.get("article", {}).get("value")

                if qid_val not in entity_map:
                    entity_map[qid_val] = WikidataEntity(
                        qid=qid_val,
                        label=label or None,
                        description=description or None,
                        aliases=aliases,
                        url=f"https://www.wikidata.org/wiki/{qid_val}",
                        wikipedia_url=wikipedia_url,
                    )

        if get_details and MEDIAWIKI_AVAILABLE:
            wiki = MediaWikiAPI()
            for qid_val, entity in list(entity_map.items()):
                if entity.wikipedia_url:
                    try:
                        _wikipedia_rate_limit(self._max_wikipedia_rps)
                        title = entity.wikipedia_url.split("/wiki/")[-1]
                        page = wiki.page(title)
                        entity.wikipedia_content = page.content
                    except Exception:
                        pass

        return entity_map

    def search_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Search for Wikidata entities by text query or QID(s)."""
        is_single = isinstance(query, str)
        queries = [query] if is_single else list(query)

        # Classify: (index, qid) for QID lookups, (index, text) for text search
        qid_entries: List[Tuple[int, str]] = []
        text_entries: List[Tuple[int, str]] = []
        for i, q in enumerate(queries):
            qid = _normalize_qid(q)
            if qid:
                qid_entries.append((i, qid))
            else:
                text_entries.append((i, q))

        # Batch fetch all QIDs in one call (or few if over BATCH_SIZE)
        qid_batch: Dict[str, WikidataEntity] = {}
        if qid_entries:
            all_qids = [qid for _, qid in qid_entries]
            qid_batch = self._get_entities_batch(all_qids, get_details=get_details)

        # Text searches (sequential; no batch search API)
        text_results: List[Tuple[int, List[WikidataEntity]]] = []
        details_qids: List[str] = []
        for i, q in text_entries:
            found = self._search_entity_by_text(q, num_results=num_results)
            if get_details and found:
                details_qids.extend(e.qid for e in found)
            text_results.append((i, found))

        # One batch for details of all text-search results
        details_batch: Dict[str, WikidataEntity] = {}
        if get_details and details_qids:
            details_batch = self._get_entities_batch(
                details_qids, get_details=get_details
            )

        # Build results in query order
        results: List[List[WikidataEntity]] = [[] for _ in range(len(queries))]
        for i, qid in qid_entries:
            entity = qid_batch.get(qid)
            results[i] = [entity] if entity else []
        for i, found in text_results:
            if get_details and found and details_batch:
                found = [details_batch.get(e.qid, e) for e in found]
            results[i] = found

        if not is_single:
            assert len(results) == len(queries), "Number of results does not match number of queries"
        return results[0] if is_single else results

    # ------------------------------------------------------------------
    # Property search / retrieval
    # ------------------------------------------------------------------

    def _get_properties_batch(self, pids: List[str]) -> Dict[str, WikidataProperty]:
        result: Dict[str, WikidataProperty] = {}
        to_fetch: List[str] = []

        for pid in pids:
            if pid in self.property_labels:
                info = self.property_labels[pid]
                label, desc = self._normalize_property_text(
                    pid,
                    info.get("label"),
                    info.get("description"),
                )
                result[pid] = WikidataProperty(
                    pid=pid,
                    label=label,
                    description=desc,
                )
            else:
                to_fetch.append(pid)

        if not to_fetch:
            return result

        for start in range(0, len(to_fetch), BATCH_SIZE):
            batch = to_fetch[start : start + BATCH_SIZE]
            values_clause = " ".join(f"wd:{p}" for p in batch)
            sparql = f"""
            SELECT ?property ?propertyLabel ?propertyDescription WHERE {{
              VALUES ?property {{ {values_clause} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
            }}
            """
            for row in self._sparql_query(sparql):
                uri = row.get("property", {}).get("value", "")
                if "/entity/" not in uri:
                    continue
                pid_val = uri.split("/")[-1].upper()
                raw_label = row.get("propertyLabel", {}).get("value", "")
                raw_desc = row.get("propertyDescription", {}).get("value", "")
                label, desc = self._normalize_property_text(pid_val, raw_label, raw_desc)
                prop = WikidataProperty(pid=pid_val, label=label, description=desc)
                result[pid_val] = prop
                self.property_labels[pid_val] = {
                    "label": prop.label,
                    "description": prop.description,
                }

        return result

    def search_properties(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
    ) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        """Search for Wikidata properties by text query or PID(s)."""
        is_single = isinstance(query, str)
        queries = [query] if is_single else list(query)

        results: List[List[WikidataProperty]] = []
        for q in queries:
            pid = _normalize_pid(q)
            if pid:
                # First try cached labels/descriptions with normalization
                info = self.property_labels.get(pid, {})
                label, desc = self._normalize_property_text(
                    pid,
                    info.get("label"),
                    info.get("description"),
                )

                # If cache is missing or uninformative, fetch from Wikidata
                if not (label or desc):
                    batch = self._get_properties_batch([pid])
                    prop = batch.get(pid)
                    if prop:
                        label, desc = prop.label, prop.description

                # If still nothing meaningful, return an empty result for this PID
                if not (label or desc):
                    results.append([])
                else:
                    prop = WikidataProperty(pid=pid, label=label, description=desc)
                    results.append([prop])
            else:
                text_escaped = json.dumps(q)
                sparql = f"""
                SELECT ?entity ?ordinal WHERE {{
                  SERVICE wikibase:mwapi {{
                    bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                                    wikibase:api "EntitySearch" ;
                                    mwapi:search {text_escaped} ;
                                    mwapi:language "en" ;
                                    mwapi:type "property" ;
                                    mwapi:limit {num_results} .
                    ?entity wikibase:apiOutputItem mwapi:item .
                    ?ordinal wikibase:apiOrdinal true .
                  }}
                }}
                ORDER BY ASC(?ordinal)
                """
                bindings = self._sparql_query(sparql)
                pids_found = []
                for row in bindings:
                    uri = row.get("entity", {}).get("value", "")
                    found_pid = _extract_id_from_uri(uri)
                    if found_pid and found_pid.startswith("P"):
                        pids_found.append(found_pid)

                props: List[WikidataProperty] = []
                if pids_found:
                    prop_map = self._get_properties_batch(pids_found[:num_results])
                    for p in pids_found[:num_results]:
                        prop = prop_map.get(p)
                        # Filter out completely uninformative properties
                        if prop and (prop.label or prop.description):
                            props.append(prop)
                results.append(props)

        return results[0] if is_single else results

    # ------------------------------------------------------------------
    # Triple retrieval
    # ------------------------------------------------------------------

    def _get_outgoing_triples(self, qid: str) -> List[WikiTriple]:
        prop_uris = " ".join(
            f"<http://www.wikidata.org/prop/direct/{p}>" for p in self.properties
        )
        sparql = f"""
        SELECT ?relation ?object WHERE {{
          wd:{qid} ?relation ?object .
          VALUES ?relation {{ {prop_uris} }}
        }}
        LIMIT {LIMIT_PER_QUERY}
        """
        bindings = self._sparql_query(sparql)
        triples: List[WikiTriple] = []
        subject = WikidataEntity(qid=qid, url=f"https://www.wikidata.org/wiki/{qid}")

        for row in bindings:
            rel_uri = row.get("relation", {}).get("value", "")
            if "/prop/direct/" not in rel_uri:
                continue
            pid = rel_uri.split("/")[-1]
            label, desc = self._get_property_label(pid)
            prop = WikidataProperty(pid=pid, label=label, description=desc)

            obj_data = row.get("object", {})
            obj_type = obj_data.get("type", "")
            obj_value = obj_data.get("value", "")

            if obj_type == "uri" and "/entity/" in obj_value:
                obj_qid = obj_value.split("/")[-1].upper()
                if obj_qid.startswith("Q") and obj_qid[1:].isdigit():
                    obj_entity = WikidataEntity(
                        qid=obj_qid, url=f"https://www.wikidata.org/wiki/{obj_qid}"
                    )
                    triples.append(WikiTriple(subject=subject, relation=prop, object=obj_entity))
                else:
                    triples.append(WikiTriple(subject=subject, relation=prop, object=obj_value))
            else:
                triples.append(WikiTriple(subject=subject, relation=prop, object=obj_value))

        return triples

    def _get_bidirectional_triples(self, qid: str) -> List[WikiTriple]:
        triples = self._get_outgoing_triples(qid)

        prop_uris = " ".join(
            f"<http://www.wikidata.org/prop/direct/{p}>" for p in self.properties
        )
        sparql = f"""
        SELECT ?subject ?relation WHERE {{
          ?subject ?relation wd:{qid} .
          VALUES ?relation {{ {prop_uris} }}
        }}
        LIMIT {LIMIT_PER_QUERY}
        """
        bindings = self._sparql_query(sparql)
        obj_entity = WikidataEntity(qid=qid, url=f"https://www.wikidata.org/wiki/{qid}")

        for row in bindings:
            subj_uri = row.get("subject", {}).get("value", "")
            rel_uri = row.get("relation", {}).get("value", "")
            if "/entity/" not in subj_uri or "/prop/direct/" not in rel_uri:
                continue
            subj_qid = subj_uri.split("/")[-1].upper()
            if not (subj_qid.startswith("Q") and subj_qid[1:].isdigit()):
                continue
            pid = rel_uri.split("/")[-1]
            label, desc = self._get_property_label(pid)
            prop = WikidataProperty(pid=pid, label=label, description=desc)
            subj = WikidataEntity(qid=subj_qid, url=f"https://www.wikidata.org/wiki/{subj_qid}")
            triples.append(WikiTriple(subject=subj, relation=prop, object=obj_entity))

        return triples

    @staticmethod
    def _deduplicate_triples(triples: List[WikiTriple]) -> List[WikiTriple]:
        seen: Set[Tuple] = set()
        unique: List[WikiTriple] = []
        for t in triples:
            if isinstance(t.object, WikidataEntity):
                key = (t.subject.qid, t.relation.pid, t.object.qid)
            else:
                key = (t.subject.qid, t.relation.pid, str(t.object))
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    def get_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Get k-hop triples from entity QIDs via SPARQL."""
        is_single = isinstance(qids, str)
        qid_list = [qids] if is_single else list(qids)

        all_results: List[List[WikiTriple]] = []
        for seed in qid_list:
            seed_qid = _normalize_qid(seed)
            if not seed_qid:
                all_results.append([])
                continue

            collected: List[WikiTriple] = []
            visited: Set[str] = set()
            frontier: Set[str] = {seed_qid}

            for _hop in range(k):
                next_frontier: Set[str] = set()
                for entity_qid in frontier:
                    if entity_qid in visited:
                        continue
                    visited.add(entity_qid)

                    if bidirectional:
                        hop_triples = self._get_bidirectional_triples(entity_qid)
                    else:
                        hop_triples = self._get_outgoing_triples(entity_qid)

                    collected.extend(hop_triples)

                    for t in hop_triples:
                        if isinstance(t.object, WikidataEntity) and t.object.qid not in visited:
                            next_frontier.add(t.object.qid)
                        if bidirectional and t.subject.qid not in visited:
                            next_frontier.add(t.subject.qid)

                if len(next_frontier) > MAX_ENTITIES_PER_HOP:
                    next_frontier = set(list(next_frontier)[:MAX_ENTITIES_PER_HOP])
                frontier = next_frontier
                if not frontier:
                    break

            collected = self._deduplicate_triples(collected)
            if enrich:
                collected = self._enrich_triples(collected)
            collected = [
                t for t in collected
                if not self._has_fake_property_label(getattr(t, "relation", None))
            ]
            all_results.append(collected)

        return all_results[0] if is_single else all_results

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def find_path(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 2,
        enrich: bool = True,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Find path between two entities using bidirectional BFS on SPARQL."""
        src = _normalize_qid(source_qid)
        tgt = _normalize_qid(target_qid)
        if not src or not tgt:
            return None

        if src == tgt:
            entity = self._get_entity_by_qid(src)
            if entity:
                return WikidataPathBetweenEntities(
                    source=entity, target=entity, path=[], path_length=0,
                )
            return None

        forward_visited: Dict[str, Tuple[Optional[str], Optional[WikiTriple]]] = {src: (None, None)}
        forward_frontier: Set[str] = {src}
        backward_visited: Dict[str, Tuple[Optional[str], Optional[WikiTriple]]] = {tgt: (None, None)}
        backward_frontier: Set[str] = {tgt}
        meeting_qid: Optional[str] = None

        for _hop in range(max_hops):
            if not forward_frontier and not backward_frontier:
                break

            expand_forward = forward_frontier and (
                not backward_frontier or len(forward_frontier) <= len(backward_frontier)
            )
            if expand_forward:
                meeting_qid, forward_frontier = self._expand_frontier(
                    forward_frontier, forward_visited, backward_visited,
                )
            else:
                meeting_qid, backward_frontier = self._expand_frontier(
                    backward_frontier, backward_visited, forward_visited,
                )
            if meeting_qid:
                break

        if not meeting_qid:
            logger.info(f"No path found between {src} and {tgt} within {max_hops} hops")
            return None

        forward_path = self._reconstruct_path(forward_visited, meeting_qid)
        forward_path.reverse()
        backward_path = self._reconstruct_path(backward_visited, meeting_qid)
        full_path = forward_path + backward_path

        if enrich and full_path:
            full_path = self._enrich_triples(full_path)
            full_path = [
                t for t in full_path
                if not self._has_fake_property_label(getattr(t, "relation", None))
            ]

        source_entity = self._get_entity_by_qid(src)
        target_entity = self._get_entity_by_qid(tgt)
        if not source_entity or not target_entity:
            return None

        return WikidataPathBetweenEntities(
            source=source_entity,
            target=target_entity,
            path=full_path,
            path_length=len(full_path),
        )

    def _expand_frontier(
        self,
        frontier: Set[str],
        own_visited: Dict[str, Tuple[Optional[str], Optional[WikiTriple]]],
        other_visited: Dict[str, Tuple[Optional[str], Optional[WikiTriple]]],
    ) -> Tuple[Optional[str], Set[str]]:
        new_frontier: Set[str] = set()
        for entity_qid in frontier:
            triples = self._get_bidirectional_triples(entity_qid)
            for t in triples:
                neighbors: List[str] = []
                if t.subject.qid == entity_qid and isinstance(t.object, WikidataEntity):
                    neighbors.append(t.object.qid)
                elif isinstance(t.object, WikidataEntity) and t.object.qid == entity_qid:
                    neighbors.append(t.subject.qid)

                for nq in neighbors:
                    if nq not in own_visited:
                        own_visited[nq] = (entity_qid, t)
                        new_frontier.add(nq)
                        if nq in other_visited:
                            return nq, new_frontier
        return None, new_frontier

    @staticmethod
    def _reconstruct_path(
        visited: Dict[str, Tuple[Optional[str], Optional[WikiTriple]]],
        end_qid: str,
    ) -> List[WikiTriple]:
        path: List[WikiTriple] = []
        current = end_qid
        while visited[current][0] is not None:
            parent_qid, triple = visited[current]
            path.append(triple)
            current = parent_qid
        return path

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich_entities(
        self, entities: List[WikidataEntity], get_details: bool = False,
    ) -> List[WikidataEntity]:
        """Batch enrich entities with labels, descriptions, aliases."""
        qids_to_enrich = [
            e.qid for e in entities if e and not self._is_entity_enriched(e)
        ]
        if not qids_to_enrich:
            return entities

        enriched_map = self._get_entities_batch(
            list(set(qids_to_enrich)), get_details=get_details,
        )
        return [enriched_map.get(e.qid, e) if e else e for e in entities]

    def enrich_properties(
        self, properties: List[WikidataProperty],
    ) -> List[WikidataProperty]:
        """Batch enrich properties with labels and descriptions."""
        pids_to_enrich = [
            p.pid for p in properties if p and not self._is_property_enriched(p)
        ]
        if not pids_to_enrich:
            return properties

        enriched_map = self._get_properties_batch(list(set(pids_to_enrich)))
        return [enriched_map.get(p.pid, p) if p else p for p in properties]

    def _enrich_triples(
        self,
        triples: List[WikiTriple],
        get_details: bool = False,
    ) -> List[WikiTriple]:
        entity_qids: Set[str] = set()
        property_pids: Set[str] = set()

        for t in triples:
            if hasattr(t.subject, "qid") and not self._is_entity_enriched(t.subject):
                entity_qids.add(t.subject.qid)
            if isinstance(t.object, WikidataEntity) and not self._is_entity_enriched(t.object):
                entity_qids.add(t.object.qid)
            if hasattr(t.relation, "pid") and not self._is_property_enriched(t.relation):
                property_pids.add(t.relation.pid)

        entity_map: Dict[str, WikidataEntity] = {}
        if entity_qids:
            entity_map = self._get_entities_batch(list(entity_qids), get_details=get_details)

        prop_map: Dict[str, WikidataProperty] = {}
        if property_pids:
            prop_map = self._get_properties_batch(list(property_pids))

        enriched: List[WikiTriple] = []
        for t in triples:
            subject = entity_map.get(t.subject.qid, t.subject) if hasattr(t.subject, "qid") else t.subject
            relation = prop_map.get(t.relation.pid, t.relation) if hasattr(t.relation, "pid") else t.relation
            if isinstance(t.object, WikidataEntity):
                obj = entity_map.get(t.object.qid, t.object)
            else:
                obj = t.object
            enriched.append(WikiTriple(subject=subject, relation=relation, object=obj))

        return enriched

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------

    async def asearch_entities(
        self,
        query: Union[str, List[str]],
        num_results: int = 1,
        get_details: bool = True,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        return await asyncio.to_thread(
            self.search_entities, query, num_results, get_details,
        )

    async def aget_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        return await asyncio.to_thread(
            self.get_k_hop_triples, qids, k, bidirectional, enrich,
        )

    async def afind_path(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 2,
        enrich: bool = True,
    ) -> Optional[WikidataPathBetweenEntities]:
        return await asyncio.to_thread(
            self.find_path, source_qid, target_qid, max_hops, enrich,
        )
