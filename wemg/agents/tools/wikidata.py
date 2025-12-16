"""Wikidata tool for querying entity information from Wikidata knowledge base."""
import asyncio
import logging
import time
import random
from threading import Semaphore
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import pydantic
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper
from langchain_core.tools import BaseTool

from wemg.agents.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

WIKIDATA_MAX_QUERY_LENGTH = 500
# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = 4  # Maximum concurrent requests to Wikidata
REQUEST_DELAY = 0.1  # Base delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_BASE_DELAY = 1.0  # Base delay for exponential backoff
USER_AGENT = "WEMG-Bot/1.0 (https://github.com/uonlp/wemg; contact@example.com) Python/3.x"

# Global semaphore for rate limiting (sync)
_wikidata_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
# Async semaphore will be created per event loop
_async_semaphore: Optional[asyncio.Semaphore] = None

def _get_async_semaphore() -> asyncio.Semaphore:
    """Get or create async semaphore for the current event loop."""
    global _async_semaphore
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if _async_semaphore is None or loop is None:
        _async_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _async_semaphore
DEFAULT_PROPERTIES = [
    "P31",
    "P279",
    "P27",
    "P361",
    "P527",
    "P495",
    "P17",
    "P585",
    "P131",
    "P106",
    "P21",
    "P569",
    "P570",
    "P577",
    "P50",
    "P571",
    "P641",
    "P625",
    "P19",
    "P69",
    "P108",
    "P136",
    "P39",
    "P161",
    "P20",
    "P101",
    "P179",
    "P175",
    "P7937",
    "P57",
    "P607",
    "P509",
    "P800",
    "P449",
    "P580",
    "P582",
    "P276",
    "P112",
    "P740",
    "P159",
    "P452",
    "P102",
    "P1142",
    "P1387",
    "P1576",
    "P140",
    "P178",
    "P287",
    "P25",
    "P22",
    "P40",
    "P185",
    "P802",
    "P1416",
]

PROPERTY_LABELS = {
    'P1142': {
        'label': 'political ideology',
        'description': 'political ideology of an organization or person or of a work (such as a newspaper)'
    },
    'P69': {
        'label': 'educated at',
        'description': 'educational institution attended by subject'
    },
    'P108': {
        'label': 'employer',
        'description': 'person or organization for which the subject works or worked'
    },
    'P136': {
        'label': 'genre',
        'description': "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic"
    },
    'P50': {
        'label': 'author',
        'description': 'main creator(s) of a written work (use on works, not humans); use P2093 (author name string) when Wikidata item is unknown or does not exist'
    },
    'P22': {
        'label': 'father',
        'description': 'male parent of the subject. For stepfather, use "stepparent" (P3448)'
    },
    'P19': {
        'label': 'place of birth',
        'description': 'most specific known birth location of a person, animal or fictional character'
    },
    'P112': {
        'label': 'founded by',
        'description': 'founder or co-founder of this organization, religion, place or entity'
    },
    'P495': {
        'label': 'country of origin',
        'description': 'country of origin of this item (creative work, food, phrase, product, etc.)'
    },
    'P27': {
        'label': 'country of citizenship',
        'description': 'the object is a country that recognizes the subject as its citizen'
    },
    'P509': {
        'label': 'cause of death',
        'description': "underlying or immediate cause of death. Underlying cause (e.g. car accident, stomach cancer) preferred. Use 'manner of death' (P1196) for broadest category, e.g. natural causes, accident, homicide, suicide"
    },
    'P31': {
        'label': 'instance of',
        'description': 'type to which this subject corresponds/belongs. Different from P279 (subclass of); for example: K2 is an instance of mountain; volcano is a subclass of mountain'
    },
    'P287': {
        'label': 'designed by',
        'description': 'person or organization which designed the object. For buildings use "architect" (Property:P84)'
    },
    'P159': {
        'label': 'headquarters location',
        'description': "city or town where an organization's headquarters is or has been situated. Use P276 qualifier for specific building"
    },
    'P571': {
        'label': 'inception',
        'description': 'time when an entity begins to exist; for date of official opening use P1619'
    },
    'P279': {
        'label': 'subclass of',
        'description': 'this item is a subclass (subset) of that item; ALL instances of this item are instances of that item; different from P31 (instance of), e.g.: volcano is a subclass of mountain; Everest is an instance of mountain'
    },
    'P25': {
        'label': 'mother',
        'description': 'female parent of the subject. For stepmother, use "stepparent" (P3448)'
    },
    'P106': {
        'label': 'occupation',
        'description': 'occupation of a person. See also "field of work" (Property:P101), "position held" (Property:P39). Not for groups of people. There, use "field of work" (Property:P101), "industry" (Property:P452), "members have occupation" (Property:P3989).'
    },
    'P57': {
        'label': 'director',
        'description': 'director(s) of film, TV-series, stageplay, video game or similar'
    },
    'P102': {
        'label': 'member of political party',
        'description': 'the political party of which a person is or has been a member or otherwise affiliated'
    },
    'P607': {
        'label': 'participated in conflict',
        'description': 'battles, wars or other military engagements in which the person or item participated'
    },
    'P179': {
        'label': 'part of the series',
        'description': 'series which contains the subject'
    },
    'P40': {
        'label': 'child',
        'description': 'subject has object as child. Do not use for stepchildren—use "relative" (P1038), qualified with "type of kinship" (P1039)'
    },
    'P1416': {
        'label': 'affiliation',
        'description': 'organization that a person or organization is affiliated with (not necessarily member of or employed by)'
    },
    'P361': {
        'label': 'part of',
        'description': 'object of which the subject is a part (if this subject is already part of object A which is a part of object B, then please only make the subject part of object A), inverse property of "has part" (P527, see also "has parts of the class" (P2670))'
    },
    'P527': {
        'label': 'has part(s)',
        'description': 'part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).'
    },
    'P449': {
        'label': 'original broadcaster',
        'description': 'network(s) or service(s) that originally broadcast a radio or television program'
    },
    'P101': {
        'label': 'field of work',
        'description': 'specialization of a person or organization; see P106 for the occupation'
    },
    'P7937': {
        'label': 'form of creative work',
        'description': 'structure of a creative work'
    },
    'P178': {
        'label': 'developer',
        'description': 'organization or person that developed the item'
    },
    'P17': {
        'label': 'country',
        'description': 'sovereign state that this item is in (not to be used for human beings)'
    },
    'P39': {
        'label': 'position held',
        'description': 'subject currently or formerly holds the object position or public office'
    },
    'P175': {
        'label': 'performer',
        'description': 'actor, musician, band or other performer associated with this role or musical work'
    },
    'P585': {
        'label': 'point in time',
        'description': 'date something took place, existed or a statement was true; for providing time use the "refine date" property (P4241)'
    },
    'P577': {
        'label': 'publication date',
        'description': 'date or point in time when a work or product was first published or released'
    },
    'P802': {
        'label': 'student',
        'description': 'notable student(s) of the subject individual'
    },
    'P21': {
        'label': 'sex or gender',
        'description': 'sex or gender identity of human or animal. For human: male, female, non-binary, intersex, transgender female, transgender male, agender, etc. For animal: male organism, female organism. Groups of same gender use subclass of (P279)'
    },
    'P140': {
        'label': 'religion or worldview',
        'description': 'religion of a person, organization or religious building, or associated with this subject'
    },
    'P740': {
        'label': 'location of formation',
        'description': 'location where a group or organization was formed'
    },
    'P20': {
        'label': 'place of death',
        'description': 'most specific known (e.g. city instead of country, or hospital instead of city) death location of a person, animal or fictional character'
    },
    'P131': {
        'label': 'located in the administrative territorial entity',
        'description': 'the item is located on the territory of the following administrative entity. Use P276 for specifying locations that are non-administrative places and for items about events. Use P1382 if the item falls only partially into the administrative entity'
    },
    'P800': {
        'label': 'notable work',
        'description': "notable scientific, artistic or literary work, or other work of significance among subject's works"
    },
    'P570': {
        'label': 'date of death',
        'description': 'date on which the subject died'
    },
    'P580': {
        'label': 'start time',
        'description': 'time an entity begins to exist or a statement starts being valid'
    },
    'P582': {
        'label': 'end time',
        'description': 'moment when an entity ceases to exist and a statement stops being entirely valid or no longer be true'
    },
    'P569': {
        'label': 'date of birth',
        'description': 'date on which the subject was born'
    },
    'P276': {
        'label': 'location',
        'description': 'location of the object, structure or event; use P131 to indicate the containing administrative entity, P8138 for statistical entities, or P706 for geographic entities; use P7153 for locations associated with the object'
    },
    'P1387': {
        'label': 'political alignment',
        'description': 'political position within the left–right political spectrum'
    },
    'P185': {
        'label': 'doctoral student',
        'description': 'doctoral student(s) of a professor'
    },
    'P641': {
        'label': 'sport',
        'description': 'sport that the subject participates or participated in or is associated with'
    },
    'P452': {
        'label': 'industry',
        'description': 'specific industry of company or organization'
    },
    'P625': {
        'label': 'coordinate location',
        'description': 'geocoordinates of the subject. For Earth, please note that only the WGS84 geodetic datum is currently supported'
    },
    'P1576': {
        'label': 'lifestyle',
        'description': 'typical way of life of an individual, group, or culture'
    },
    'P161': {
        'label': 'cast member',
        'description': 'actor in the subject production [use "character role" (P453) and/or "name of the character role" (P4633) as qualifiers] [use "voice actor" (P725) for voice-only role] - [use "recorded participant" (P11108) for non-fiction productions]'
    },
}


class WikidataEntity(pydantic.BaseModel):
    """Represents a single Wikidata entity with structured information."""
    
    qid: str = pydantic.Field(..., description="Wikidata QID (e.g., Q7251)")
    label: Optional[str] = pydantic.Field("", description="The entity label/name")
    description: Optional[str] = pydantic.Field("", description="Brief description of the entity")
    aliases: Optional[List[str]] = pydantic.Field(default_factory=list, description="Alternative names for the entity")
    url: str = pydantic.Field(None, description="Wikidata URL for the entity")
    wikipedia_url: Optional[str] = pydantic.Field(None, description="Wikipedia URL for the entity")
    wikidata_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikidata")
    wikipedia_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikipedia")

    def to_context(self, include_wiki_page: bool = False) -> str:
        """Return an LLM-friendly, natural-language summary of this entity."""
        label = (self.label or "").strip() or self.qid
        description = (self.description or "").strip()
        header = label
        if description:
            header = f"{header}: {description}"
        lines: List[str] = [header]
        if include_wiki_page and self.wikipedia_content:
            lines.append(f"Wikipedia Content:\n{self.wikipedia_content.strip()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        label = (self.label or "").strip() or self.qid
        description = (self.description or "").strip()
        return f"{label} - {description}" if description else label
    
    def __hash__(self):
        return hash(self.qid)

class WikidataProperty(pydantic.BaseModel):
    """Represents a property of a Wikidata entity."""
    
    pid: str = pydantic.Field(..., description="Wikidata Property ID (e.g., P31)")
    label: Optional[str] = pydantic.Field("", description="The property label/name")
    description: Optional[str] = pydantic.Field(..., description="The description associated with the property")

    def __hash__(self):
        return hash(self.pid)
    
    def __str__(self) -> str:
        label = (self.label or "").strip() or self.pid
        description = (self.description or "").strip()
        return f"{label}: {description}" if description else label

class WikiTriple(pydantic.BaseModel):
    """Represents a single triple (subject, predicate, object) from Wikidata."""
    
    subject: WikidataEntity = pydantic.Field(..., description="The subject entity of the triple")
    relation: WikidataProperty = pydantic.Field(..., description="The property/relation of the triple")
    object: Any = pydantic.Field(..., description="The object/value of the triple")

    def __str__(self) -> str:
        subject_str = str(self.subject)
        relation_str = str(self.relation)
        object_str = str(self.object)
        return f"Subject: {subject_str}\nRelation: {relation_str}\nObject: {object_str}"

    def __hash__(self):
        if isinstance(self.object, WikidataEntity):
            return hash((self.subject.qid, self.relation.pid, self.object.qid))
        else:
            return hash((self.subject.qid, self.relation.pid, str(self.object)))

class WikidataPathBetweenEntities(pydantic.BaseModel):
    """Represents a path between two Wikidata entities."""
    
    source: WikidataEntity = pydantic.Field(..., description="The source entity of the path")
    target: WikidataEntity = pydantic.Field(..., description="The target entity of the path")
    path: List[WikiTriple] = pydantic.Field(default_factory=list, description="List of triples forming the path from source to target")
    path_length: int = pydantic.Field(0, description="Length of the path (number of hops)")
    
    def __str__(self) -> str:
        if not self.path:
            return f"No path found between {self.source} and {self.target}."
        all_triples = []
        for i, triple in enumerate(self.path):
            triple_str = f"{i + 1}.\n{str(triple)}"
            all_triples.append(triple_str)
        path_str = "\n--------------\n".join(all_triples)
        return f"Path from {self.source} to {self.target}:\n{path_str}"

    def __hash__(self):
        # sort triples by their subject.qid, relation.pid and object.qid/str for consistent hashing
        sorted_triples = sorted(
            self.path,
            key=lambda t: (t.subject.qid, t.relation.pid, t.object.qid if isinstance(t.object, WikidataEntity) else str(t.object))
        )
        hashable_path = tuple(
            (triple.subject.qid, triple.relation.pid, triple.object.qid if isinstance(triple.object, WikidataEntity) else str(triple.object))
            for triple in sorted_triples
        )
        return hash(hashable_path) # this ensures that the hash is order-independent

class CustomWikidataAPIWrapper(WikidataAPIWrapper):
    """Custom Wikidata API Wrapper with enhanced item retrieval."""

    # add wikidata_props_with_labels property
    wikidata_props_with_labels: Dict[str, Dict[str, Optional[str]]] = {pid: PROPERTY_LABELS[pid] for pid in PROPERTY_LABELS}
    wikidata_props: List[str] = DEFAULT_PROPERTIES

    @staticmethod
    def _execute_sparql_with_retry(
        query: str,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY
    ) -> Optional[Dict]:
        """Execute a SPARQL query with retry logic and rate limiting. """
        from SPARQLWrapper import SPARQLWrapper, JSON, POST
        
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)
        sparql.setMethod(POST)
        sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
        sparql.setQuery(query)
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Acquire semaphore to limit concurrent requests
                with _wikidata_semaphore:
                    # Add small random delay to avoid thundering herd
                    if attempt > 0 or REQUEST_DELAY > 0:
                        delay = REQUEST_DELAY + random.uniform(0, 0.1)
                        time.sleep(delay)
                    
                    results = sparql.query().convert()
                    return results
                    
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit or server error
                is_rate_limit = any(x in error_str for x in [
                    "429", "too many requests", "rate limit",
                    "503", "service unavailable", "timeout",
                    "500", "internal server error"
                ])
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limited or server error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                elif attempt < max_retries - 1:
                    # For other errors, still retry but with shorter delay
                    delay = base_delay * (attempt + 1)
                    logger.warning(
                        f"Query failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
        
        return None

    @staticmethod
    async def _execute_sparql_with_retry_async(
        query: str,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY
    ) -> Optional[Dict]:
        """Execute a SPARQL query asynchronously with retry logic and rate limiting."""
        import aiohttp
        
        url = "https://query.wikidata.org/sparql"
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        semaphore = _get_async_semaphore()
        
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    # Add small random delay to avoid thundering herd
                    if attempt > 0 or REQUEST_DELAY > 0:
                        delay = REQUEST_DELAY + random.uniform(0, 0.1)
                        await asyncio.sleep(delay)
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            data={"query": query},
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                            elif response.status in [429, 503, 500]:
                                raise Exception(f"HTTP {response.status}: {await response.text()}")
                            else:
                                response.raise_for_status()
                                
            except Exception as e:
                error_str = str(e).lower()
                
                is_rate_limit = any(x in error_str for x in [
                    "429", "too many requests", "rate limit",
                    "503", "service unavailable", "timeout",
                    "500", "internal server error"
                ])
                
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limited or server error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                elif attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    logger.warning(
                        f"Query failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
        
        return None

    async def _get_item_async(self, qid: Union[str, List[str]], get_details: bool = False) -> Union[Optional[WikidataEntity], List[Optional[WikidataEntity]]]:
        """Async wrapper for `_get_item` (supports batch).

        This keeps async call sites working while avoiding duplicated async
        implementations.
        """
        return await asyncio.to_thread(self._get_item, qid, get_details=get_details)

    async def _get_k_hop_bidirectional_async(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Async wrapper for `_get_k_hop_bidirectional` (supports batch)."""
        return await asyncio.to_thread(self._get_k_hop_bidirectional, qid, k, prop)

    async def _get_k_hop_outgoing_async(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Async wrapper for `_get_k_hop_outgoing` (supports batch)."""
        return await asyncio.to_thread(self._get_k_hop_outgoing, qid, k, prop)

    def model_post_init(self, __context) -> None:
        """Initialize wikidata_props_with_labels after model initialization."""
        super().model_post_init(__context)
        
        # Populate wikidata_props_with_labels with labels and descriptions for all wikidata_props
        self.wikidata_props = list(set(self.wikidata_props))  # Ensure uniqueness
        
        # Reinitialize with existing labels from PROPERTY_LABELS if available to avoid redundant fetches
        for pid in self.wikidata_props:
            if pid in PROPERTY_LABELS:
                self.wikidata_props_with_labels[pid] = PROPERTY_LABELS[pid]
        
        if self.wikidata_props:
            # Filter properties that need to be fetched
            props_to_fetch = [p for p in self.wikidata_props if p not in self.wikidata_props_with_labels]
            
            if props_to_fetch:
                logger.info(f"Fetching labels for {len(props_to_fetch)} Wikidata properties...")
                # Rate-limited parallel fetch of property labels
                def fetch_property(prop_id: str) -> Tuple[str, Optional[Dict[str, Optional[str]]]]:
                    with _wikidata_semaphore:
                        time.sleep(REQUEST_DELAY + random.uniform(0, 0.05))
                        prop_data = self._get_property(prop_id)
                        if prop_data:
                            return prop_id, {"label": prop_data.label, "description": prop_data.description}
                        return prop_id, None
                
                with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(props_to_fetch))) as executor:
                    futures = {executor.submit(fetch_property, pid): pid for pid in props_to_fetch}
                    for future in as_completed(futures):
                        prop_id = futures[future]
                        try:
                            pid, result = future.result()
                            if result:
                                self.wikidata_props_with_labels[pid] = result
                            else:
                                self.wikidata_props_with_labels[pid] = {"label": pid, "description": None}
                                logger.warning(f"Could not load property {pid}, using ID as label")
                        except Exception as e:
                            self.wikidata_props_with_labels[prop_id] = {"label": prop_id, "description": None}
                            logger.warning(f"Error loading property {prop_id}: {e}")

    def _get_id(self, query: Union[str, List[str]], id_type: str = "item") -> Union[List[str], List[List[str]]]:
        """Search for Wikidata IDs for a single query or a batch.

        - If ``query`` is a string, returns ``List[str]`` (possibly empty).
        - If ``query`` is a list of strings, returns ``List[List[str]]`` aligned
          to the input order.

        The search uses WDQS SPARQL with ``SERVICE wikibase:mwapi`` and
        ``wikibase:api "EntitySearch"``.

        Args:
            query: Text query (str) or batch of queries (List[str]).
            id_type: Either "item" (QIDs, default) or "property" (PIDs).

        Returns:
            For a string input: a list of IDs (QIDs/PIDs depending on ``id_type``).
            For a list input: list-of-lists of IDs aligned with the input.
        """
        is_single = False
        if isinstance(query, str):
            query = [query]
            is_single = True

        # Batch mode
        if isinstance(query, list):
            if not all(isinstance(q, str) or q is None for q in query):
                raise TypeError("query must be a string or a list of strings")

            normalized: List[str] = [((q or "").strip()) for q in query]
            output: List[List[str]] = [[] for _ in normalized]

            pending: List[Tuple[int, str]] = []
            for idx, q in enumerate(normalized):
                if not q:
                    continue

                if re.fullmatch(r"[QP]\d+", q, flags=re.IGNORECASE):
                    direct = q.upper()
                    if id_type == "property" and direct.startswith("P"):
                        output[idx] = [direct]
                    elif id_type != "property" and direct.startswith("Q"):
                        output[idx] = [direct]
                    else:
                        output[idx] = []
                    continue

                pending.append((idx, q[:WIKIDATA_MAX_QUERY_LENGTH]))

            if not pending:
                return output[0] if is_single else output

            batch_size = 25
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                chunk_queries = [q for _, q in chunk]
                try:
                    results_map = self._search_ids_via_sparql(
                        chunk_queries,
                        id_type=id_type,
                        limit=self.top_k_results,
                    )
                except Exception as e:
                    logger.warning(f"Wikidata ID batch search failed (type={id_type}): {e}")
                    results_map = {q: [] for q in chunk_queries}

                for idx, q in chunk:
                    output[idx] = results_map.get(q, [])[: self.top_k_results]

            return output[0] if is_single else output

    def _search_ids_via_sparql(self, queries: List[str], *, id_type: str, limit: int) -> Dict[str, List[str]]:
        """Internal: search IDs for multiple queries via SPARQL mwapi EntitySearch."""
        if not queries:
            return {}

        # SPARQL string literal encoding
        values = " ".join(json.dumps(q) for q in queries)
        entity_type = "property" if id_type == "property" else "item"
        language = (self.lang or "en").strip() or "en"
        limit = max(1, int(limit))

        sparql = f"""
        SELECT ?search ?entity ?ordinal WHERE {{
          VALUES ?search {{ {values} }}
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org" ;
                            wikibase:api "EntitySearch" ;
                            wikibase:limit "once" ;
                            mwapi:search ?search ;
                            mwapi:language {json.dumps(language)} ;
                            mwapi:type {json.dumps(entity_type)} ;
                            mwapi:limit {limit} .
            ?entity wikibase:apiOutputItem mwapi:item .
            ?ordinal wikibase:apiOrdinal true .
          }}
        }}
        ORDER BY ?search ASC(?ordinal)
        """

        data = self._execute_sparql_with_retry(sparql)
        if not data or not isinstance(data, dict):
            return {q: [] for q in queries}

        bindings = (((data.get("results") or {}).get("bindings")) or [])
        # Collect per query with ordinal sorting
        tmp: Dict[str, List[Tuple[int, str]]] = {q: [] for q in queries}

        def extract_id(uri: str) -> Optional[str]:
            if not isinstance(uri, str) or not uri:
                return None
            # Typical: http://www.wikidata.org/entity/Q42
            if "/entity/" in uri:
                candidate = uri.rsplit("/", 1)[-1]
            else:
                candidate = uri
            candidate = candidate.strip()
            if re.fullmatch(r"[QP]\d+", candidate, flags=re.IGNORECASE):
                return candidate.upper()
            return None

        for row in bindings:
            if not isinstance(row, dict):
                continue
            search_val = (row.get("search") or {}).get("value")
            entity_uri = (row.get("entity") or {}).get("value")
            ordinal_raw = (row.get("ordinal") or {}).get("value")
            if not isinstance(search_val, str) or search_val not in tmp:
                continue
            entity_id = extract_id(entity_uri)
            if not entity_id:
                continue
            if id_type == "property" and not entity_id.startswith("P"):
                continue
            if id_type != "property" and not entity_id.startswith("Q"):
                continue
            try:
                ordinal = int(float(ordinal_raw)) if ordinal_raw is not None else 0
            except Exception:
                ordinal = 0
            tmp[search_val].append((ordinal, entity_id))

        out: Dict[str, List[str]] = {}
        for q, pairs in tmp.items():
            pairs.sort(key=lambda t: t[0])
            seen: Set[str] = set()
            ids: List[str] = []
            for _, entity_id in pairs:
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                ids.append(entity_id)
            out[q] = ids

        return out

    def _get_item(self, qid: Union[str, List[str]], get_details: bool = False) -> Union[Optional[WikidataEntity], List[Optional[WikidataEntity]]]:
        """Retrieve Wikidata entities by QID(s) using optimized batch SPARQL queries.
        
        Args:
            qid: A single QID (str) or list of QIDs (List[str]) to retrieve.
            
        Returns:
            - If qid is a string: Optional[WikidataEntity] (single entity or None)
            - If qid is a list: List[Optional[WikidataEntity]] aligned with input order
        """
        # Handle single QID input
        is_single = isinstance(qid, str)
        if is_single:
            qids = [qid]
        else:
            qids = list(qid)
        
        # Filter out empty/invalid QIDs and track their positions
        valid_qids: List[str] = []
        qid_to_indices: Dict[str, List[int]] = {}  # Map QID to list indices (handles duplicates)
        
        for idx, q in enumerate(qids):
            if q and isinstance(q, str):
                normalized = q.strip().upper()
                if re.fullmatch(r"Q\d+", normalized):
                    valid_qids.append(normalized)
                    if normalized not in qid_to_indices:
                        qid_to_indices[normalized] = []
                    qid_to_indices[normalized].append(idx)
        
        # Initialize results with None for all positions
        results: List[Optional[WikidataEntity]] = [None] * len(qids)
        
        if not valid_qids:
            return results[0] if is_single else results
        
        # Deduplicate QIDs for efficient querying
        unique_qids = list(set(valid_qids))
        
        # Process in batches to avoid query limits
        batch_size = 25
        entity_map: Dict[str, WikidataEntity] = {}
        
        for batch_start in range(0, len(unique_qids), batch_size):
            batch_qids = unique_qids[batch_start:batch_start + batch_size]
            batch_entities = self._get_items_batch(batch_qids, get_details=get_details)
            entity_map.update(batch_entities)
        
        # Map results back to original positions
        for qid_key, indices in qid_to_indices.items():
            entity = entity_map.get(qid_key)
            for idx in indices:
                results[idx] = entity
        
        return results[0] if is_single else results

    def _get_items_batch(self, qids: List[str], get_details: bool = False) -> Dict[str, WikidataEntity]:
        """Retrieve multiple Wikidata entities in a single SPARQL query.
        
        Args:
            qids: List of QIDs to retrieve (should be deduplicated and validated).
            
        Returns:
            Dictionary mapping QID to WikidataEntity for successfully retrieved entities.
        """
        if not qids:
            return {}
        
        # Build property filter for optimized query
        if self.wikidata_props:
            prop_filters = " || ".join([
                f'?prop = wdt:{prop}'
                for prop in self.wikidata_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = "FILTER(STRSTARTS(STR(?prop), 'http://www.wikidata.org/prop/direct/'))"
        
        # Build VALUES clause for batch query
        values_clause = " ".join([f"wd:{qid}" for qid in qids])
        
        # Single comprehensive SPARQL query for all entities
        query = f"""
        SELECT ?entity ?label ?description ?alias ?wikipedia ?prop ?value ?valueLabel
        WHERE {{
          VALUES ?entity {{ {values_clause} }}
          
          # Get label
          OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "{self.lang}") }}
          
          # Get description
          OPTIONAL {{ ?entity schema:description ?description . FILTER(LANG(?description) = "{self.lang}") }}
          
          # Get aliases
          OPTIONAL {{ ?entity skos:altLabel ?alias . FILTER(LANG(?alias) = "{self.lang}") }}
          
          # Get Wikipedia URL
          OPTIONAL {{
            ?wikipedia schema:about ?entity ;
                      schema:inLanguage "{self.lang}" ;
                      schema:isPartOf <https://{self.lang}.wikipedia.org/> .
          }}
          
          # Get all properties and their values with labels
          OPTIONAL {{
            ?entity ?prop ?value .
            {property_filter}
            OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "{self.lang}") }}
          }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
        }}
        """
        
        try:
            results = self._execute_sparql_with_retry(query)
            
            if not results or not results["results"]["bindings"]:
                logger.warning(f"Could not find any items for QIDs: {qids}")
                return {}
            
            # Parse results into per-entity data structures
            entity_data: Dict[str, Dict[str, Any]] = {}
            
            for row in results["results"]["bindings"]:
                # Extract entity QID from URI
                entity_uri = row.get("entity", {}).get("value", "")
                if "/entity/" not in entity_uri:
                    continue
                entity_qid = entity_uri.split("/")[-1].upper()
                
                # Initialize entity data if not exists
                if entity_qid not in entity_data:
                    entity_data[entity_qid] = {
                        "label": "",
                        "description": "",
                        "aliases": [],
                        "wikipedia_url": None,
                        "properties": {}
                    }
                
                data = entity_data[entity_qid]
                
                # Extract label (same for all rows of this entity)
                if not data["label"] and "label" in row:
                    data["label"] = row["label"]["value"]
                
                # Extract description (same for all rows of this entity)
                if not data["description"] and "description" in row:
                    data["description"] = row["description"]["value"]
                
                # Extract aliases (can be multiple)
                if "alias" in row:
                    alias_val = row["alias"]["value"]
                    if alias_val not in data["aliases"]:
                        data["aliases"].append(alias_val)
                
                # Extract Wikipedia URL (same for all rows of this entity)
                if not data["wikipedia_url"] and "wikipedia" in row:
                    data["wikipedia_url"] = row["wikipedia"]["value"]
                
                # Extract properties and values
                if "prop" in row and "value" in row:
                    prop_uri = row["prop"]["value"]
                    if "/prop/direct/" in prop_uri:
                        prop_id = prop_uri.split("/")[-1]
                        
                        # Get property label from cache
                        prop_label = prop_id
                        if prop_id in self.wikidata_props_with_labels:
                            prop_label = self.wikidata_props_with_labels[prop_id]["label"]
                        
                        # Get value (use label if available, otherwise raw value)
                        value_data = row["value"]
                        if value_data["type"] == "uri" and "/entity/" in value_data["value"]:
                            value_str = row.get("valueLabel", {}).get("value", value_data["value"].split("/")[-1])
                        else:
                            value_str = value_data["value"]
                        
                        # Group multiple values for same property
                        if prop_id not in data["properties"]:
                            data["properties"][prop_id] = {"label": prop_label, "values": []}
                        if value_str not in data["properties"][prop_id]["values"]:
                            data["properties"][prop_id]["values"].append(value_str)
            
            # Build WikidataEntity objects from parsed data
            entity_map: Dict[str, WikidataEntity] = {}
            
            # Collect Wikipedia URLs for batch fetching
            wikipedia_urls: Dict[str, str] = {}  # QID -> URL
            for qid_key, data in entity_data.items():
                if data["wikipedia_url"]:
                    wikipedia_urls[qid_key] = data["wikipedia_url"]
            
            # Batch fetch Wikipedia content using ThreadPoolExecutor
            wikipedia_content_map: Dict[str, Optional[str]] = {}
            if wikipedia_urls and get_details:
                def fetch_wikipedia(qid_url: Tuple[str, str]) -> Tuple[str, Optional[str]]:
                    qid_key, url = qid_url
                    try:
                        content = WebSearchTool.crawl_web_pages(url)
                        return qid_key, content
                    except Exception as e:
                        logger.warning(f"Failed to fetch Wikipedia content for {qid_key}: {e}")
                        return qid_key, None
                
                with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(wikipedia_urls))) as executor:
                    futures = [executor.submit(fetch_wikipedia, item) for item in wikipedia_urls.items()]
                    for future in as_completed(futures):
                        try:
                            qid_key, content = future.result()
                            wikipedia_content_map[qid_key] = content
                        except Exception as e:
                            logger.warning(f"Error fetching Wikipedia content: {e}")
            
            for qid_key, data in entity_data.items():
                # Build wikidata_content from properties
                wikidata_content_lines = []
                if data["label"]:
                    wikidata_content_lines.append(f"Label: {data['label']}")
                if data["description"]:
                    wikidata_content_lines.append(f"Description: {data['description']}")
                if data["aliases"]:
                    wikidata_content_lines.append(f"Aliases: {', '.join(data['aliases'])}")
                
                # Add properties with their labels
                for prop_id, prop_data in data["properties"].items():
                    if prop_id not in self.wikidata_props_with_labels:
                        continue
                    prop_label = prop_data["label"]
                    values_str = ", ".join(prop_data["values"])
                    wikidata_content_lines.append(f"{prop_label}: {values_str}")
                
                wikidata_content = "\n".join(wikidata_content_lines)
                
                # Use Wikipedia content if available (replaces wikidata_content per original logic)
                wikipedia_url = data["wikipedia_url"]
                wikipedia_content = wikipedia_content_map.get(qid_key)
                if wikipedia_content:
                    wikidata_content = wikipedia_content
                
                entity_map[qid_key] = WikidataEntity(
                    qid=qid_key,
                    label=data["label"],
                    description=data["description"],
                    aliases=data["aliases"],
                    wikidata_content=wikidata_content,
                    wikipedia_content=wikipedia_content,
                    url=f"https://www.wikidata.org/wiki/{qid_key}",
                    wikipedia_url=wikipedia_url
                )
            
            return entity_map
            
        except Exception as e:
            logger.error(f"Error fetching items {qids} via SPARQL: {e}")
            return {}

    def _get_property(self, pid: Union[str, List[str]]) -> Union[Optional[WikidataProperty], List[Optional[WikidataProperty]]]:
        """Retrieve Wikidata properties by PID(s) using SPARQL batch queries.
        
        Args:
            pid: A single PID (str) or list of PIDs (List[str]) to retrieve.
            
        Returns:
            - If pid is a string: Optional[WikidataProperty] (single property or None)
            - If pid is a list: List[Optional[WikidataProperty]] aligned with input order
        """
        # Handle single PID input
        is_single = isinstance(pid, str)
        if is_single:
            pids = [pid]
        else:
            pids = list(pid)
        
        # Filter out empty/invalid PIDs and track their positions
        valid_pids: List[str] = []
        pid_to_indices: Dict[str, List[int]] = {}
        property_map: Dict[str, WikidataProperty] = {}
        
        for idx, p in enumerate(pids):
            if p and isinstance(p, str):
                normalized = p.strip().upper()
                if re.fullmatch(r"P\d+", normalized):
                    valid_pids.append(normalized)
                    if normalized not in pid_to_indices:
                        pid_to_indices[normalized] = []
                    pid_to_indices[normalized].append(idx)
        
        # Initialize results with None for all positions
        results: List[Optional[WikidataProperty]] = [None] * len(pids)
        
        if not valid_pids:
            return results[0] if is_single else results
        
        # only fetch properties not already in PROPERTY_LABELS
        to_fetch_pids = [p for p in valid_pids if p not in PROPERTY_LABELS]
        for pid_key in valid_pids:
            if pid_key in PROPERTY_LABELS:
                prop_info = PROPERTY_LABELS[pid_key]
                property_map[pid_key] = WikidataProperty(
                    pid=pid_key,
                    label=prop_info.get("label", pid_key),
                    description=prop_info.get("description")
                )

        # Deduplicate PIDs for efficient querying
        unique_pids = list(set(to_fetch_pids))
        
        # Process in batches
        batch_size = 50
        
        for batch_start in range(0, len(unique_pids), batch_size):
            batch_pids = unique_pids[batch_start:batch_start + batch_size]
            batch_props = self._get_properties_batch(batch_pids)
            property_map.update(batch_props)
        
        # Map results back to original positions
        for pid_key, indices in pid_to_indices.items():
            prop = property_map.get(pid_key)
            for idx in indices:
                results[idx] = prop
        
        return results[0] if is_single else results

    def _get_properties_batch(self, pids: List[str]) -> Dict[str, WikidataProperty]:
        """Retrieve multiple Wikidata properties in a single SPARQL query.
        
        Args:
            pids: List of PIDs to retrieve (should be deduplicated and validated).
            
        Returns:
            Dictionary mapping PID to WikidataProperty for successfully retrieved properties.
        """
        if not pids:
            return {}
        
        # Build VALUES clause for batch query
        values_clause = " ".join([f"wd:{pid}" for pid in pids])
        
        query = f"""
        SELECT ?property ?label ?description
        WHERE {{
          VALUES ?property {{ {values_clause} }}
          
          OPTIONAL {{ ?property rdfs:label ?label . FILTER(LANG(?label) = "{self.lang}") }}
          OPTIONAL {{ ?property schema:description ?description . FILTER(LANG(?description) = "{self.lang}") }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
        }}
        """
        
        try:
            results = self._execute_sparql_with_retry(query)
            
            if not results or not results["results"]["bindings"]:
                logger.warning(f"Could not find any properties for PIDs: {pids}")
                return {}
            
            property_map: Dict[str, WikidataProperty] = {}
            
            for row in results["results"]["bindings"]:
                prop_uri = row.get("property", {}).get("value", "")
                if "/entity/" not in prop_uri:
                    continue
                
                prop_id = prop_uri.split("/")[-1].upper()
                label = row.get("label", {}).get("value", "")
                description = row.get("description", {}).get("value", "")
                
                if prop_id not in property_map:
                    property_map[prop_id] = WikidataProperty(
                        pid=prop_id,
                        label=label,
                        description=description
                    )
            
            return property_map
            
        except Exception as e:
            logger.error(f"Error fetching properties {pids} via SPARQL: {e}")
            return {}

    def _get_k_hop_bidirectional(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop bidirectional triples for given entity QID(s).
        
        Args:
            qid: A single QID (str) or list of QIDs (List[str]) to retrieve triples for.
            k: Number of hops to traverse (default: 1)
            prop: Optional property ID to filter by
            
        Returns:
            - If qid is a string: List[WikiTriple] for that entity
            - If qid is a list: List[List[WikiTriple]] aligned with input order
        """
        # Handle single QID input
        is_single = isinstance(qid, str)
        if is_single:
            qids = [qid]
        else:
            qids = list(qid)
        
        # Normalize and validate QIDs
        normalized_qids: List[str] = []
        qid_to_indices: Dict[str, List[int]] = {}
        
        for idx, q in enumerate(qids):
            if q and isinstance(q, str):
                normalized = q.strip().upper()
                if re.fullmatch(r"Q\d+", normalized):
                    normalized_qids.append(normalized)
                    if normalized not in qid_to_indices:
                        qid_to_indices[normalized] = []
                    qid_to_indices[normalized].append(idx)
        
        # Initialize results
        results: List[List[WikiTriple]] = [[] for _ in qids]
        
        if not normalized_qids:
            return results[0] if is_single else results
        
        unique_qids = list(set(normalized_qids))
        
        # Build property filter for SPARQL query
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{p}>'
                for p in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        # Track triples per source QID
        qid_triples: Dict[str, List[WikiTriple]] = {q: [] for q in unique_qids}
        visited_entities: Dict[str, Set[str]] = {q: set() for q in unique_qids}
        current_level: Dict[str, Set[str]] = {q: {q} for q in unique_qids}
        
        for hop in range(k):
            # Collect all entities to query at this level
            all_entities_this_hop: Set[str] = set()
            entity_to_source: Dict[str, Set[str]] = {}  # entity -> source QIDs that need it
            
            for source_qid in unique_qids:
                entities_to_query = current_level[source_qid] - visited_entities[source_qid]
                visited_entities[source_qid].update(entities_to_query)
                all_entities_this_hop.update(entities_to_query)
                for e in entities_to_query:
                    if e not in entity_to_source:
                        entity_to_source[e] = set()
                    entity_to_source[e].add(source_qid)
            
            if not all_entities_this_hop:
                break
            
            # Execute batch query for all entities at this level
            hop_triples, next_qids_map = self._execute_bidirectional_batch(
                list(all_entities_this_hop), property_filter
            )
            
            # Distribute triples to their source QIDs
            for entity_qid, triples in hop_triples.items():
                source_qids = entity_to_source.get(entity_qid, set())
                for source_qid in source_qids:
                    qid_triples[source_qid].extend(triples)
            
            # Update next level for each source QID
            if hop < k - 1:
                for source_qid in unique_qids:
                    next_level: Set[str] = set()
                    for entity_qid in current_level[source_qid]:
                        if entity_qid in next_qids_map:
                            next_level.update(next_qids_map[entity_qid])
                    current_level[source_qid] = next_level
        
        # Deduplicate and enrich triples for each source QID
        for source_qid in unique_qids:
            unique_triples = self._deduplicate_and_enrich_triples(qid_triples[source_qid])
            qid_triples[source_qid] = unique_triples
            logger.info(f"Retrieved {len(unique_triples)} unique triples for {source_qid} with {k}-hop bidirectional traversal")
        
        # Map results back to original positions
        for qid_key, indices in qid_to_indices.items():
            triples = qid_triples.get(qid_key, [])
            for idx in indices:
                results[idx] = triples
        
        return results[0] if is_single else results

    def _execute_bidirectional_batch(
        self,
        entity_qids: List[str],
        property_filter: str
    ) -> Tuple[Dict[str, List[WikiTriple]], Dict[str, Set[str]]]:
        """Execute bidirectional SPARQL query for multiple entities in batch.
        
        Args:
            entity_qids: List of entity QIDs to query
            property_filter: SPARQL property filter string
            
        Returns:
            Tuple of:
            - Dict mapping entity QID to list of triples
            - Dict mapping entity QID to set of next-level QIDs
        """
        if not entity_qids:
            return {}, {}
        
        # Process in batches to avoid query limits
        batch_size = 10
        all_triples: Dict[str, List[WikiTriple]] = {q: [] for q in entity_qids}
        all_next_qids: Dict[str, Set[str]] = {q: set() for q in entity_qids}
        
        for batch_start in range(0, len(entity_qids), batch_size):
            batch_qids = entity_qids[batch_start:batch_start + batch_size]
            values_clause = " ".join([f"wd:{qid}" for qid in batch_qids])
            
            query = f"""
            SELECT ?sourceEntity ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel 
                   ?object ?objectLabel ?objectDesc ?direction
            WHERE {{
              VALUES ?sourceEntity {{ {values_clause} }}
              {{
                # Outgoing: entity as subject
                BIND(?sourceEntity AS ?subject)
                ?sourceEntity ?relation ?object .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("outgoing" AS ?direction)
              }}
              UNION
              {{
                # Incoming: entity as object
                BIND(?sourceEntity AS ?object)
                ?subject ?relation ?sourceEntity .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("incoming" AS ?direction)
              }}
              
              OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 2000
            """
            
            try:
                results = self._execute_sparql_with_retry(query)
                
                if not results or not results["results"]["bindings"]:
                    continue
                
                for row in results["results"]["bindings"]:
                    source_uri = row.get("sourceEntity", {}).get("value", "")
                    if "/entity/" not in source_uri:
                        continue
                    source_qid = source_uri.split("/")[-1].upper()
                    
                    triple, next_qid, direction = self._parse_triple_row(row)
                    if triple:
                        all_triples[source_qid].append(triple)
                        if next_qid:
                            if direction == "outgoing":
                                all_next_qids[source_qid].add(next_qid)
                            elif direction == "incoming":
                                # For incoming, the subject is the next entity to explore
                                all_next_qids[source_qid].add(triple.subject.qid)
                                
            except Exception as e:
                logger.warning(f"Error executing bidirectional batch query: {e}")
        
        return all_triples, all_next_qids

    def _parse_triple_row(self, row: Dict) -> Tuple[Optional[WikiTriple], Optional[str], str]:
        """Parse a SPARQL result row into a WikiTriple.
        
        Returns:
            Tuple of (triple, next_qid for traversal, direction)
        """
        relation_uri = row.get("relation", {}).get("value", "")
        if not relation_uri or "/prop/direct/" not in relation_uri:
            return None, None, ""
        
        pid = relation_uri.split("/")[-1]
        relation_label = row.get("relationLabel", {}).get("value", "")
        direction = row.get("direction", {}).get("value", "")
        
        subject_uri = row.get("subject", {}).get("value", "")
        if "/entity/" not in subject_uri:
            return None, None, ""
        subject_qid = subject_uri.split("/")[-1]
        subject_label = row.get("subjectLabel", {}).get("value", subject_qid)
        subject_desc = row.get("subjectDesc", {}).get("value", "")
        
        object_uri = row.get("object", {}).get("value", "")
        object_type = row.get("object", {}).get("type", "")
        
        subject_entity = WikidataEntity(
            qid=subject_qid,
            label=subject_label,
            description=subject_desc,
            aliases=[],
            url=f"https://www.wikidata.org/wiki/{subject_qid}"
        )
        
        next_qid = None
        if object_type == "uri" and "/entity/" in object_uri:
            object_qid = object_uri.split("/")[-1]
            object_label = row.get("objectLabel", {}).get("value", object_qid)
            object_desc = row.get("objectDesc", {}).get("value", "")
            
            object_entity = WikidataEntity(
                qid=object_qid,
                label=object_label,
                description=object_desc,
                aliases=[],
                url=f"https://www.wikidata.org/wiki/{object_qid}"
            )
            
            if pid:
                triple = WikiTriple(
                    subject=subject_entity,
                    relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                    object=object_entity
                )
                next_qid = object_qid
                return triple, next_qid, direction
        else:
            object_value = row.get("object", {}).get("value", "")
            if pid:
                triple = WikiTriple(
                    subject=subject_entity,
                    relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                    object=object_value
                )
                return triple, None, direction
        
        return None, None, ""

    def _deduplicate_and_enrich_triples(self, triples: List[WikiTriple]) -> List[WikiTriple]:
        """Deduplicate triples and enrich relation labels from cache."""
        seen: Set[Tuple] = set()
        unique_triples: List[WikiTriple] = []
        missing_pids: Set[str] = set()
        
        # First pass: collect missing PIDs
        for triple in triples:
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)
            
            pid = triple.relation.pid
            if pid not in self.wikidata_props_with_labels and not triple.relation.label:
                missing_pids.add(pid)
        
        # Batch fetch missing property labels
        if missing_pids:
            fetched_props = self._get_property(list(missing_pids))
            if isinstance(fetched_props, list):
                for prop in fetched_props:
                    if prop:
                        self.wikidata_props_with_labels[prop.pid] = {
                            "label": prop.label,
                            "description": prop.description
                        }
        
        # Second pass: build unique triples with enriched relations
        seen.clear()
        for triple in triples:
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)
            
            relation = triple.relation
            pid = relation.pid
            
            if pid in self.wikidata_props_with_labels:
                relation = WikidataProperty(
                    pid=pid,
                    label=self.wikidata_props_with_labels[pid].get("label", relation.label),
                    description=self.wikidata_props_with_labels[pid].get("description", relation.description)
                )
            
            updated_triple = WikiTriple(
                subject=triple.subject,
                object=triple.object,
                relation=relation
            )
            unique_triples.append(updated_triple)
        
        return unique_triples

    def _get_k_hop_outgoing(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop outgoing triples for given entity QID(s).
        
        Args:
            qid: A single QID (str) or list of QIDs (List[str]) to retrieve triples for.
            k: Number of hops to traverse (default: 1)
            prop: Optional property ID to filter by
            
        Returns:
            - If qid is a string: List[WikiTriple] for that entity
            - If qid is a list: List[List[WikiTriple]] aligned with input order
        """
        # Handle single QID input
        is_single = isinstance(qid, str)
        if is_single:
            qids = [qid]
        else:
            qids = list(qid)
        
        # Normalize and validate QIDs
        normalized_qids: List[str] = []
        qid_to_indices: Dict[str, List[int]] = {}
        
        for idx, q in enumerate(qids):
            if q and isinstance(q, str):
                normalized = q.strip().upper()
                if re.fullmatch(r"Q\d+", normalized):
                    normalized_qids.append(normalized)
                    if normalized not in qid_to_indices:
                        qid_to_indices[normalized] = []
                    qid_to_indices[normalized].append(idx)
        
        # Initialize results
        results: List[List[WikiTriple]] = [[] for _ in qids]
        
        if not normalized_qids:
            return results[0] if is_single else results
        
        unique_qids = list(set(normalized_qids))
        
        # Build property filter for SPARQL query
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{p}>'
                for p in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        # Track triples per source QID
        qid_triples: Dict[str, List[WikiTriple]] = {q: [] for q in unique_qids}
        visited_entities: Dict[str, Set[str]] = {q: set() for q in unique_qids}
        current_level: Dict[str, Set[str]] = {q: {q} for q in unique_qids}
        
        for hop in range(k):
            # Collect all entities to query at this level
            all_entities_this_hop: Set[str] = set()
            entity_to_source: Dict[str, Set[str]] = {}
            
            for source_qid in unique_qids:
                entities_to_query = current_level[source_qid] - visited_entities[source_qid]
                visited_entities[source_qid].update(entities_to_query)
                all_entities_this_hop.update(entities_to_query)
                for e in entities_to_query:
                    if e not in entity_to_source:
                        entity_to_source[e] = set()
                    entity_to_source[e].add(source_qid)
            
            if not all_entities_this_hop:
                break
            
            # Execute batch query for all entities at this level
            hop_triples, next_qids_map = self._execute_outgoing_batch(
                list(all_entities_this_hop), property_filter
            )
            
            # Distribute triples to their source QIDs
            for entity_qid, triples in hop_triples.items():
                source_qids = entity_to_source.get(entity_qid, set())
                for source_qid in source_qids:
                    qid_triples[source_qid].extend(triples)
            
            # Update next level for each source QID
            if hop < k - 1:
                for source_qid in unique_qids:
                    next_level: Set[str] = set()
                    for entity_qid in current_level[source_qid]:
                        if entity_qid in next_qids_map:
                            next_level.update(next_qids_map[entity_qid])
                    current_level[source_qid] = next_level
        
        # Deduplicate and enrich triples for each source QID
        for source_qid in unique_qids:
            unique_triples = self._deduplicate_and_enrich_triples(qid_triples[source_qid])
            qid_triples[source_qid] = unique_triples
            logger.info(f"Retrieved {len(unique_triples)} unique outgoing triples for {source_qid} with {k}-hop traversal")
        
        # Map results back to original positions
        for qid_key, indices in qid_to_indices.items():
            triples = qid_triples.get(qid_key, [])
            for idx in indices:
                results[idx] = triples
        
        return results[0] if is_single else results

    def _execute_outgoing_batch(
        self,
        entity_qids: List[str],
        property_filter: str
    ) -> Tuple[Dict[str, List[WikiTriple]], Dict[str, Set[str]]]:
        """Execute outgoing SPARQL query for multiple entities in batch.
        
        Args:
            entity_qids: List of entity QIDs to query
            property_filter: SPARQL property filter string
            
        Returns:
            Tuple of:
            - Dict mapping entity QID to list of triples
            - Dict mapping entity QID to set of next-level QIDs
        """
        if not entity_qids:
            return {}, {}
        
        # Process in batches to avoid query limits
        batch_size = 10
        all_triples: Dict[str, List[WikiTriple]] = {q: [] for q in entity_qids}
        all_next_qids: Dict[str, Set[str]] = {q: set() for q in entity_qids}
        
        for batch_start in range(0, len(entity_qids), batch_size):
            batch_qids = entity_qids[batch_start:batch_start + batch_size]
            values_clause = " ".join([f"wd:{qid}" for qid in batch_qids])
            
            query = f"""
            SELECT ?sourceEntity ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel 
                   ?object ?objectLabel ?objectDesc
            WHERE {{
              VALUES ?sourceEntity {{ {values_clause} }}
              
              BIND(?sourceEntity AS ?subject)
              ?sourceEntity ?relation ?object .
              FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
              {property_filter}
              
              OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 2000
            """
            
            try:
                results = self._execute_sparql_with_retry(query)
                
                if not results or not results["results"]["bindings"]:
                    continue
                
                for row in results["results"]["bindings"]:
                    source_uri = row.get("sourceEntity", {}).get("value", "")
                    if "/entity/" not in source_uri:
                        continue
                    source_qid = source_uri.split("/")[-1].upper()
                    
                    # Add direction for compatibility with _parse_triple_row
                    row["direction"] = {"value": "outgoing"}
                    triple, next_qid, _ = self._parse_triple_row(row)
                    if triple:
                        all_triples[source_qid].append(triple)
                        if next_qid:
                            all_next_qids[source_qid].add(next_qid)
                                
            except Exception as e:
                logger.warning(f"Error executing outgoing batch query: {e}")
        
        return all_triples, all_next_qids


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

    def _run(self, query: Union[str, List[str]], num_entities: int = 3, is_qids: bool = False, get_details: bool = False) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Retrieve Wikidata entities based on a textual query or QIDs.
        
        Args:
            query: A search query string or list of queries/QIDs
            num_entities: Maximum number of entities to return per query
            is_qids: If True, treat query as QID(s) directly
            
        Returns:
            - If query is a string: List[WikidataEntity]
            - If query is a list: List[List[WikidataEntity]] aligned with input order
        """
        is_single = isinstance(query, str)
        
        if is_qids:
            if is_single:
                qids_per_query = [[query]]
            else:
                qids_per_query = [[qid] for qid in query]
        else:
            if is_single:
                search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [search_results[:num_entities] if search_results else []]
            else:
                all_search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [results[:num_entities] if results else [] for results in all_search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids = []
        qid_to_query_idx: Dict[str, List[int]] = {}
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid not in qid_to_query_idx:
                    qid_to_query_idx[qid] = []
                    all_qids.append(qid)
                qid_to_query_idx[qid].append(query_idx)
        
        if not all_qids:
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Use batch _get_item for efficient retrieval
        results = self.wikidata_wrapper._get_item(all_qids, get_details=get_details)
        
        # Build QID to entity map
        qid_to_entity: Dict[str, WikidataEntity] = {}
        if isinstance(results, list):
            for qid, entity in zip(all_qids, results):
                if entity:
                    qid_to_entity[qid] = entity
        elif results:
            qid_to_entity[all_qids[0]] = results
        
        # Build output aligned with input queries
        output: List[List[WikidataEntity]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid in qid_to_entity:
                    output[query_idx].append(qid_to_entity[qid])
        
        return output[0] if is_single else output

    async def _arun(self, query: Union[str, List[str]], num_entities: int = 3, is_qids: bool = False, get_details: bool = False) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        """Async version of entity retrieval with batch support.
        
        Args:
            query: A search query string or list of queries/QIDs
            num_entities: Maximum number of entities to return per query
            is_qids: If True, treat query as QID(s) directly
            
        Returns:
            - If query is a string: List[WikidataEntity]
            - If query is a list: List[List[WikidataEntity]] aligned with input order
        """
        is_single = isinstance(query, str)
        
        if is_qids:
            if is_single:
                qids_per_query = [[query]]
            else:
                qids_per_query = [[qid] for qid in query]
        else:
            if is_single:
                search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [search_results[:num_entities] if search_results else []]
            else:
                all_search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [results[:num_entities] if results else [] for results in all_search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids = []
        qid_to_query_idx: Dict[str, List[int]] = {}
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid not in qid_to_query_idx:
                    qid_to_query_idx[qid] = []
                    all_qids.append(qid)
                qid_to_query_idx[qid].append(query_idx)
        
        if not all_qids:
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Use batch _get_item_async for efficient retrieval
        try:
            results = await self.wikidata_wrapper._get_item_async(all_qids, get_details=get_details)
        except Exception as e:
            logger.error(f"Error fetching entities: {e}")
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Build QID to entity map
        qid_to_entity: Dict[str, WikidataEntity] = {}
        if isinstance(results, list):
            for qid, entity in zip(all_qids, results):
                if entity:
                    qid_to_entity[qid] = entity
        elif results:
            qid_to_entity[all_qids[0]] = results
        
        # Build output aligned with input queries
        output: List[List[WikidataEntity]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid in qid_to_entity:
                    output[query_idx].append(qid_to_entity[qid])
        
        return output[0] if is_single else output

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

    def _run(self, query: Union[str, List[str]], top_k_results: int = 3) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        """Retrieve Wikidata properties based on a textual query.
        
        Args:
            query: A search query string or list of queries
            num_properties: Maximum number of properties to return per query
            
        Returns:
            - If query is a string: List[WikidataProperty]
            - If query is a list: List[List[WikidataProperty]] aligned with input order
        """
        is_single = False
        if isinstance(query, str):
            is_single = True
            query = [query]
        pids = self.wikidata_wrapper._get_id(query, id_type="property")
        pids = [pid_list[:top_k_results] if pid_list else [] for pid_list in pids]
        all_pids = set(sum(pids, []))
        all_properties: Dict[str, WikidataProperty] = {}
        if all_pids:
            fetched_properties = self.wikidata_wrapper._get_property(list(all_pids))
            for prop in fetched_properties:
                if prop:
                    all_properties[prop.pid] = prop
        output: List[List[WikidataProperty]] = []
        for pid_list in pids:    
            properties = [all_properties[pid] for pid in pid_list if pid in all_properties]
            output.append(properties)
        return output[0] if is_single else output
    
    async def _arun(self, query: Union[str, List[str]], top_k_results: int = 3) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        return asyncio.to_thread(self._run, query, top_k_results)
        
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

    @staticmethod
    def dedup_triples_tool(triples: List[WikiTriple]) -> List[WikiTriple]:
        seen = set()
        unique_triples = []
        for triple in triples:
            # Create unique identifier based on subject, relation, and object
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id not in seen:
                seen.add(triple_id)
                unique_triples.append(triple)
        return unique_triples
    
    def get_entity_details_tool(self, qids: Set[str]) -> Dict[str, WikidataEntity]:
        """Fetch entity details for multiple QIDs using batch processing.
        
        Args:
            qids: Set of QIDs to fetch details for
            
        Returns:
            Dictionary mapping QID to WikidataEntity
        """
        entity_details: Dict[str, WikidataEntity] = {}
        
        if not qids:
            return entity_details
        
        qid_list = list(qids)
        
        # Use batch _get_item for efficient retrieval
        try:
            results = self.wikidata_wrapper._get_item(qid_list, get_details=True)
            
            if isinstance(results, list):
                for qid, entity in zip(qid_list, results):
                    if entity:
                        entity_details[qid] = entity
            elif results:  # Single result
                entity_details[qid_list[0]] = results
        except Exception as e:
            logger.error(f"Error fetching entity details: {e}")
        
        return entity_details
    
    def update_triples_with_details_tool(self, all_triples: List[WikiTriple]) -> List[WikiTriple]:
        # Update the triples with full details
        updated_triples: List[WikiTriple] = []
        all_entity_qids: Set[str] = set()
        for triple in all_triples:
            all_entity_qids.add(triple.subject.qid)
            if hasattr(triple.object, 'qid'):
                all_entity_qids.add(triple.object.qid)
        entity_details = self.get_entity_details_tool(all_entity_qids)
        for triple in all_triples:
            subject_detail = entity_details.get(triple.subject.qid)
            if hasattr(triple.object, 'qid'):
                object_detail = entity_details.get(triple.object.qid)
            else:
                object_detail = triple.object
            # Skip if details not found
            if not subject_detail or not object_detail:
                continue
            updated_triple = WikiTriple(
                subject=subject_detail,
                relation=triple.relation,
                object=object_detail
            )
            updated_triples.append(updated_triple)
        return updated_triples

    def _run(
        self, 
        query: Union[str, List[str]], 
        is_qids: bool = False,
        k: int = 1, 
        num_entities: int = 3, 
        bidirectional: bool = False,
        prop: Optional[str] = None,
        update_with_details: bool = False,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop triples for entities matching query with batch support.
        
        Args:
            query: Search query string, list of queries, or QID(s) if is_qids=True
            is_qids: If True, treat query as QID(s) directly
            k: Number of hops to traverse
            num_entities: Maximum number of entities per query
            bidirectional: If True, traverse both incoming and outgoing edges
            prop: Optional property ID to filter by
            update_with_details: If True, enrich triples with full entity details
            
        Returns:
            - If query is a string: List[WikiTriple]
            - If query is a list: List[List[WikiTriple]] aligned with input order
        """
        is_single = isinstance(query, str)
        
        if is_qids:
            if is_single:
                qids_per_query = [[query]]
            else:
                qids_per_query = [[qid] for qid in query]
        else:
            if is_single:
                search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [search_results[:num_entities] if search_results else []]
            else:
                all_search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [results[:num_entities] if results else [] for results in all_search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids = []
        qid_to_query_indices: Dict[str, List[int]] = {}
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid not in qid_to_query_indices:
                    qid_to_query_indices[qid] = []
                    all_qids.append(qid)
                qid_to_query_indices[qid].append(query_idx)
        
        if not all_qids:
            return [] if is_single else [[] for _ in qids_per_query]
        
        if prop and prop not in self.wikidata_wrapper.wikidata_props:
            logger.warning(f"Property {prop} not in the specified wikidata_props.")
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Use batch k-hop methods for efficient retrieval
        if bidirectional:
            results = self.wikidata_wrapper._get_k_hop_bidirectional(all_qids, k=k, prop=prop)
        else:
            results = self.wikidata_wrapper._get_k_hop_outgoing(all_qids, k=k, prop=prop)
        
        # Build QID to triples map
        qid_to_triples: Dict[str, List[WikiTriple]] = {}
        if isinstance(results, list):
            if results and isinstance(results[0], list):
                # Batch results: List[List[WikiTriple]]
                for qid, triples_list in zip(all_qids, results):
                    qid_to_triples[qid] = triples_list if triples_list else []
            else:
                # Single result wrapped: List[WikiTriple]
                qid_to_triples[all_qids[0]] = results
        
        # Build output aligned with input queries
        output: List[List[WikiTriple]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            query_triples: List[WikiTriple] = []
            for qid in qids:
                if qid in qid_to_triples:
                    query_triples.extend(qid_to_triples[qid])
            # Deduplicate triples per query
            output[query_idx] = self.dedup_triples_tool(query_triples)
        
        if update_with_details:
            # Collect all entity QIDs across all outputs
            all_entity_qids: Set[str] = set()
            for triples_list in output:
                for triple in triples_list:
                    all_entity_qids.add(triple.subject.qid)
                    if hasattr(triple.object, 'qid'):
                        all_entity_qids.add(triple.object.qid)
            
            entity_details = self.get_entity_details_tool(all_entity_qids)
            
            # Update each query's triples
            for i, triples_list in enumerate(output):
                updated_triples: List[WikiTriple] = []
                for triple in triples_list:
                    subject_detail = entity_details.get(triple.subject.qid)
                    if hasattr(triple.object, 'qid'):
                        object_detail = entity_details.get(triple.object.qid)
                    else:
                        object_detail = triple.object
                    if not subject_detail or not object_detail:
                        continue
                    updated_triple = WikiTriple(
                        subject=subject_detail,
                        relation=triple.relation,
                        object=object_detail
                    )
                    updated_triples.append(updated_triple)
                output[i] = updated_triples
        
        return output[0] if is_single else output

    async def get_entity_details_tool_async(self, qids: Set[str]) -> Dict[str, WikidataEntity]:
        """Async version of entity details fetching with batch support.
        
        Args:
            qids: Set of QIDs to fetch details for
            
        Returns:
            Dictionary mapping QID to WikidataEntity
        """
        return await asyncio.to_thread(self.get_entity_details_tool, qids)
    
    async def update_triples_with_details_tool_async(self, all_triples: List[WikiTriple]) -> List[WikiTriple]:
        """Async version of updating triples with full entity details."""
        return await asyncio.to_thread(self.update_triples_with_details_tool, all_triples)

    async def _arun(
        self, 
        query: Union[str, List[str]],
        is_qids: bool = False,
        k: int = 1, 
        num_entities: int = 3, 
        bidirectional: bool = False,
        prop: Optional[str] = None,
        update_with_details: bool = False,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Async version of k-hop triples retrieval with batch support.
        
        Args:
            query: Search query string, list of queries, or QID(s) if is_qids=True
            is_qids: If True, treat query as QID(s) directly
            k: Number of hops to traverse
            num_entities: Maximum number of entities per query
            bidirectional: If True, traverse both incoming and outgoing edges
            prop: Optional property ID to filter by
            update_with_details: If True, enrich triples with full entity details
            
        Returns:
            - If query is a string: List[WikiTriple]
            - If query is a list: List[List[WikiTriple]] aligned with input order
        """
        is_single = isinstance(query, str)
        
        if is_qids:
            if is_single:
                qids_per_query = [[query]]
            else:
                qids_per_query = [[qid] for qid in query]
        else:
            if is_single:
                search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [search_results[:num_entities] if search_results else []]
            else:
                all_search_results = self.wikidata_wrapper._get_id(query)
                qids_per_query = [results[:num_entities] if results else [] for results in all_search_results]
        
        # Flatten all QIDs for batch retrieval
        all_qids = []
        qid_to_query_indices: Dict[str, List[int]] = {}
        for query_idx, qids in enumerate(qids_per_query):
            for qid in qids:
                if qid not in qid_to_query_indices:
                    qid_to_query_indices[qid] = []
                    all_qids.append(qid)
                qid_to_query_indices[qid].append(query_idx)
        
        if not all_qids:
            return [] if is_single else [[] for _ in qids_per_query]
        
        if prop and prop not in self.wikidata_wrapper.wikidata_props:
            logger.warning(f"Property {prop} not in the specified wikidata_props.")
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Use batch k-hop async methods for efficient retrieval
        try:
            if bidirectional:
                results = await self.wikidata_wrapper._get_k_hop_bidirectional_async(all_qids, k=k, prop=prop)
            else:
                results = await self.wikidata_wrapper._get_k_hop_outgoing_async(all_qids, k=k, prop=prop)
        except Exception as e:
            logger.error(f"Error fetching triples: {e}")
            return [] if is_single else [[] for _ in qids_per_query]
        
        # Build QID to triples map
        qid_to_triples: Dict[str, List[WikiTriple]] = {}
        if isinstance(results, list):
            if results and isinstance(results[0], list):
                for qid, triples_list in zip(all_qids, results):
                    qid_to_triples[qid] = triples_list if triples_list else []
            else:
                qid_to_triples[all_qids[0]] = results
        
        # Build output aligned with input queries
        output: List[List[WikiTriple]] = [[] for _ in qids_per_query]
        for query_idx, qids in enumerate(qids_per_query):
            query_triples: List[WikiTriple] = []
            for qid in qids:
                if qid in qid_to_triples:
                    query_triples.extend(qid_to_triples[qid])
            output[query_idx] = self.dedup_triples_tool(query_triples)
        
        if update_with_details:
            all_entity_qids: Set[str] = set()
            for triples_list in output:
                for triple in triples_list:
                    all_entity_qids.add(triple.subject.qid)
                    if hasattr(triple.object, 'qid'):
                        all_entity_qids.add(triple.object.qid)
            
            entity_details = await self.get_entity_details_tool_async(all_entity_qids)
            
            for i, triples_list in enumerate(output):
                updated_triples: List[WikiTriple] = []
                for triple in triples_list:
                    subject_detail = entity_details.get(triple.subject.qid)
                    if hasattr(triple.object, 'qid'):
                        object_detail = entity_details.get(triple.object.qid)
                    else:
                        object_detail = triple.object
                    if not subject_detail or not object_detail:
                        continue
                    updated_triple = WikiTriple(
                        subject=subject_detail,
                        relation=triple.relation,
                        object=object_detail
                    )
                    updated_triples.append(updated_triple)
                output[i] = updated_triples
        
        return output[0] if is_single else output

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
    
    def _find_path_bidirectional_bfs(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 3,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Find path between two entities using bidirectional BFS."""
        if source_qid == target_qid:
            # Same entity - return empty path
            source_entity = self.wikidata_wrapper._get_item(source_qid, get_details=False)
            if source_entity:
                return WikidataPathBetweenEntities(
                    source=source_entity,
                    target=source_entity,
                    path=[],
                    path_length=0
                )
            return None
        
        # Forward search from source
        forward_visited: Dict[str, Tuple[str, WikiTriple]] = {source_qid: (None, None)}  # qid -> (parent_qid, triple)
        forward_frontier: Set[str] = {source_qid}
        
        # Backward search from target
        backward_visited: Dict[str, Tuple[str, WikiTriple]] = {target_qid: (None, None)}
        backward_frontier: Set[str] = {target_qid}
        
        meeting_qid: Optional[str] = None
        
        for hop in range(max_hops):
            if not forward_frontier and not backward_frontier:
                break
                
            # Expand forward frontier (smaller direction)
            if forward_frontier and (not backward_frontier or len(forward_frontier) <= len(backward_frontier)):
                new_forward_frontier: Set[str] = set()
                
                for entity_qid in forward_frontier:
                    triples = self.wikidata_wrapper._get_k_hop_bidirectional(entity_qid, k=1)
                    
                    for triple in triples:
                        # Get the neighboring QID
                        if triple.subject.qid == entity_qid and hasattr(triple.object, 'qid'):
                            neighbor_qid = triple.object.qid
                        elif hasattr(triple.object, 'qid') and triple.object.qid == entity_qid:
                            neighbor_qid = triple.subject.qid
                        else:
                            continue
                        
                        if neighbor_qid not in forward_visited:
                            forward_visited[neighbor_qid] = (entity_qid, triple)
                            new_forward_frontier.add(neighbor_qid)
                            
                            # Check if we've met the backward search
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
                    
                    for triple in triples:
                        # Get the neighboring QID
                        if triple.subject.qid == entity_qid and hasattr(triple.object, 'qid'):
                            neighbor_qid = triple.object.qid
                        elif hasattr(triple.object, 'qid') and triple.object.qid == entity_qid:
                            neighbor_qid = triple.subject.qid
                        else:
                            continue
                        
                        if neighbor_qid not in backward_visited:
                            backward_visited[neighbor_qid] = (entity_qid, triple)
                            new_backward_frontier.add(neighbor_qid)
                            
                            # Check if we've met the forward search
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
            # Need to potentially reverse the triple direction
            backward_path.append(triple)
            current = parent_qid
        
        # Combine paths
        full_path = forward_path + backward_path
        
        # Get entity details
        source_entity = self.wikidata_wrapper._get_item(source_qid, get_details=False)
        target_entity = self.wikidata_wrapper._get_item(target_qid, get_details=False)
        
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
        max_hops: int = 3,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Find path between two Wikidata entities.
        
        Args:
            source_qid: QID of the source entity (e.g., 'Q42')
            target_qid: QID of the target entity (e.g., 'Q937')
            max_hops: Maximum path length to search (default: 3)
            
        Returns:
            WikidataPathBetweenEntities object if path found, None otherwise
        """
        return self._find_path_bidirectional_bfs(source_qid, target_qid, max_hops)
    
    async def _arun(
        self,
        source_qid: str,
        target_qid: str,
        max_hops: int = 3,
    ) -> Optional[WikidataPathBetweenEntities]:
        """Async version - delegates to sync version using asyncio.to_thread."""
        # Use asyncio.to_thread (Python 3.9+) to run blocking code in thread pool
        return await asyncio.to_thread(
            self._find_path_bidirectional_bfs,
            source_qid,
            target_qid,
            max_hops
        )
