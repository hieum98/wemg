"""Wikidata tool for querying entity information from Wikidata knowledge base."""
import asyncio
import logging
import time
import random
from threading import Semaphore
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import pydantic
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


class WikidataEntity(pydantic.BaseModel):
    """Represents a single Wikidata entity with structured information."""
    
    qid: str = pydantic.Field(..., description="Wikidata QID (e.g., Q7251)")
    label: Optional[str] = pydantic.Field("", description="The entity label/name")
    description: Optional[str] = pydantic.Field("", description="Brief description of the entity")
    aliases: Optional[List[str]] = pydantic.Field(default_factory=list, description="Alternative names for the entity")
    url: str = pydantic.Field(..., description="Wikidata URL for the entity")
    wikipedia_url: Optional[str] = pydantic.Field(None, description="Wikipedia URL for the entity")
    wikidata_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikidata")
    wikipedia_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikipedia")


class WikidataProperty(pydantic.BaseModel):
    """Represents a property of a Wikidata entity."""
    
    pid: str = pydantic.Field(..., description="Wikidata Property ID (e.g., P31)")
    label: Optional[str] = pydantic.Field("", description="The property label/name")
    description: Optional[str] = pydantic.Field(..., description="The description associated with the property")

    def __hash__(self):
        return hash(self.pid)


class WikiTriple(pydantic.BaseModel):
    """Represents a single triple (subject, predicate, object) from Wikidata."""
    
    subject: WikidataEntity = pydantic.Field(..., description="The subject entity of the triple")
    relation: WikidataProperty = pydantic.Field(..., description="The property/relation of the triple")
    object: Any = pydantic.Field(..., description="The object/value of the triple")


class CustomWikidataAPIWrapper(WikidataAPIWrapper):
    """Custom Wikidata API Wrapper with enhanced item retrieval."""

    # add wikidata_props_with_labels property
    wikidata_props_with_labels: Dict[str, Dict[str, Optional[str]]] = pydantic.Field(
        default_factory=dict,
        description="Dictionary of Wikidata property IDs mapped to their labels and descriptions"
    )
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

    async def _get_item_async(self, qid: str) -> Optional[WikidataEntity]:
        """Retrieve a Wikidata entity by its QID asynchronously."""
        if self.wikidata_props:
            prop_filters = " || ".join([
                f'?prop = wdt:{prop}'
                for prop in self.wikidata_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = "FILTER(STRSTARTS(STR(?prop), 'http://www.wikidata.org/prop/direct/'))"
        
        query = f"""
        SELECT ?label ?description ?alias ?wikipedia ?prop ?value ?valueLabel
        WHERE {{
          OPTIONAL {{ wd:{qid} rdfs:label ?label . FILTER(LANG(?label) = "{self.lang}") }}
          OPTIONAL {{ wd:{qid} schema:description ?description . FILTER(LANG(?description) = "{self.lang}") }}
          OPTIONAL {{ wd:{qid} skos:altLabel ?alias . FILTER(LANG(?alias) = "{self.lang}") }}
          OPTIONAL {{
            ?wikipedia schema:about wd:{qid} ;
                      schema:inLanguage "{self.lang}" ;
                      schema:isPartOf <https://{self.lang}.wikipedia.org/> .
          }}
          OPTIONAL {{
            wd:{qid} ?prop ?value .
            {property_filter}
            OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "{self.lang}") }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
        }}
        """
        
        try:
            results = await self._execute_sparql_with_retry_async(query)
            
            if not results or not results["results"]["bindings"]:
                logger.warning(f"Could not find item {qid} in Wikidata")
                return None
            
            label = ""
            description = ""
            aliases = []
            wikipedia_url = None
            properties = {}
            
            for result in results["results"]["bindings"]:
                if not label and "label" in result:
                    label = result["label"]["value"]
                if not description and "description" in result:
                    description = result["description"]["value"]
                if "alias" in result:
                    alias_val = result["alias"]["value"]
                    if alias_val not in aliases:
                        aliases.append(alias_val)
                if not wikipedia_url and "wikipedia" in result:
                    wikipedia_url = result["wikipedia"]["value"]
                if "prop" in result and "value" in result:
                    prop_uri = result["prop"]["value"]
                    if "/prop/direct/" in prop_uri:
                        prop_id = prop_uri.split("/")[-1]
                        prop_label = prop_id
                        if prop_id in self.wikidata_props_with_labels:
                            prop_label = self.wikidata_props_with_labels[prop_id]["label"]
                        value_data = result["value"]
                        if value_data["type"] == "uri" and "/entity/" in value_data["value"]:
                            value_str = result.get("valueLabel", {}).get("value", value_data["value"].split("/")[-1])
                        else:
                            value_str = value_data["value"]
                        if prop_id not in properties:
                            properties[prop_id] = {"label": prop_label, "values": []}
                        if value_str not in properties[prop_id]["values"]:
                            properties[prop_id]["values"].append(value_str)
            
            wikidata_content_lines = []
            if label:
                wikidata_content_lines.append(f"Label: {label}")
            if description:
                wikidata_content_lines.append(f"Description: {description}")
            if aliases:
                wikidata_content_lines.append(f"Aliases: {', '.join(aliases)}")
            for prop_id, prop_data in properties.items():
                if prop_id not in self.wikidata_props_with_labels:
                    continue
                prop_label = prop_data["label"]
                values_str = ", ".join(prop_data["values"])
                wikidata_content_lines.append(f"{prop_label}: {values_str}")
            
            wikidata_content = "\n".join(wikidata_content_lines)
            
            # Note: Wikipedia content fetching is kept sync for simplicity
            wikipedia_content = None
            if wikipedia_url:
                wikidata_content = WebSearchTool.crawl_web_pages(wikipedia_url)
            
            return WikidataEntity(
                qid=qid,
                label=label,
                description=description,
                aliases=aliases,
                wikidata_content=wikidata_content,
                wikipedia_content=wikipedia_content,
                url=f"https://www.wikidata.org/wiki/{qid}",
                wikipedia_url=wikipedia_url
            )
            
        except Exception as e:
            logger.error(f"Error fetching item {qid} via async SPARQL: {e}")
            return None

    async def _get_k_hop_bidirectional_async(self, qid: str, k: int = 1, prop: Optional[str] = None) -> List[WikiTriple]:
        """Retrieve k-hop bidirectional triples asynchronously."""
        all_triples: List[WikiTriple] = []
        visited_entities: Set[str] = set()
        current_level: Set[str] = {qid}
        
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{p}>'
                for p in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        async def execute_bidirectional_query_async(entity_qid: str) -> Tuple[str, List[WikiTriple], Set[str]]:
            triples: List[WikiTriple] = []
            next_qids: Set[str] = set()
            
            bidirectional_query = f"""
            SELECT ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel ?object ?objectLabel ?objectDesc ?direction WHERE {{
              {{
                BIND(wd:{entity_qid} AS ?subject)
                wd:{entity_qid} ?relation ?object .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("outgoing" AS ?direction)
                OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
                OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              }}
              UNION
              {{
                BIND(wd:{entity_qid} AS ?object)
                ?subject ?relation wd:{entity_qid} .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("incoming" AS ?direction)
                OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
                OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 200
            """
            
            try:
                results = await CustomWikidataAPIWrapper._execute_sparql_with_retry_async(bidirectional_query)
                if not results:
                    return entity_qid, triples, next_qids
                
                for result in results["results"]["bindings"]:
                    relation_uri = result.get("relation", {}).get("value", "")
                    if not relation_uri or "/prop/direct/" not in relation_uri:
                        continue
                    
                    pid = relation_uri.split("/")[-1]
                    relation_label = result.get("relationLabel", {}).get("value", "")
                    direction = result.get("direction", {}).get("value", "")
                    
                    subject_uri = result.get("subject", {}).get("value", "")
                    if "/entity/" not in subject_uri:
                        continue
                    subject_qid = subject_uri.split("/")[-1]
                    subject_label = result.get("subjectLabel", {}).get("value", subject_qid)
                    subject_desc = result.get("subjectDesc", {}).get("value", "")
                    
                    object_uri = result.get("object", {}).get("value", "")
                    object_type = result.get("object", {}).get("type", "")
                    
                    subject_entity = WikidataEntity(
                        qid=subject_qid, label=subject_label, description=subject_desc,
                        aliases=[], url=f"https://www.wikidata.org/wiki/{subject_qid}"
                    )
                    
                    if object_type == "uri" and "/entity/" in object_uri:
                        object_qid = object_uri.split("/")[-1]
                        object_label = result.get("objectLabel", {}).get("value", object_qid)
                        object_desc = result.get("objectDesc", {}).get("value", "")
                        object_entity = WikidataEntity(
                            qid=object_qid, label=object_label, description=object_desc,
                            aliases=[], url=f"https://www.wikidata.org/wiki/{object_qid}"
                        )
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_entity
                            )
                            triples.append(triple)
                            if direction == "outgoing":
                                next_qids.add(object_qid)
                            elif direction == "incoming":
                                next_qids.add(subject_qid)
                    else:
                        object_value = result.get("object", {}).get("value", "")
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_value
                            )
                            triples.append(triple)
            except Exception as e:
                logger.warning(f"Error executing async bidirectional query for {entity_qid}: {e}")
            
            return entity_qid, triples, next_qids
        
        for hop in range(k):
            next_level: Set[str] = set()
            entities_to_query = [e for e in current_level if e not in visited_entities]
            visited_entities.update(entities_to_query)
            
            if not entities_to_query:
                break
            
            # Execute queries concurrently with asyncio.gather
            tasks = [execute_bidirectional_query_async(eq) for eq in entities_to_query]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error in async query: {result}")
                    continue
                entity_qid, triples, next_qids = result
                all_triples.extend(triples)
                if hop < k - 1:
                    next_level.update(next_qids)
            
            current_level = next_level
            if not current_level:
                break
        
        # Deduplicate triples
        seen = set()
        unique_triples = []
        for triple in all_triples:
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)
            
            relation = triple.relation
            if relation.pid in self.wikidata_props_with_labels:
                relation.label = self.wikidata_props_with_labels[relation.pid]["label"]
                relation.description = self.wikidata_props_with_labels[relation.pid]["description"]
            
            updated_triple = WikiTriple(subject=triple.subject, object=triple.object, relation=relation)
            unique_triples.append(updated_triple)
        
        logger.info(f"Retrieved {len(unique_triples)} unique triples async for {qid}")
        return unique_triples

    async def _get_k_hop_outgoing_async(self, qid: str, k: int = 1, prop: Optional[str] = None) -> List[WikiTriple]:
        """Retrieve k-hop outgoing triples asynchronously."""
        all_triples: List[WikiTriple] = []
        visited_entities: Set[str] = set()
        current_level: Set[str] = {qid}
        
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{p}>'
                for p in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        async def execute_outgoing_query_async(entity_qid: str) -> Tuple[str, List[WikiTriple], Set[str]]:
            triples: List[WikiTriple] = []
            next_qids: Set[str] = set()
            
            outgoing_query = f"""
            SELECT ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel ?object ?objectLabel ?objectDesc WHERE {{
              BIND(wd:{entity_qid} AS ?subject)
              wd:{entity_qid} ?relation ?object .
              FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
              {property_filter}
              OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 200
            """
            
            try:
                results = await CustomWikidataAPIWrapper._execute_sparql_with_retry_async(outgoing_query)
                if not results:
                    return entity_qid, triples, next_qids
                
                for result in results["results"]["bindings"]:
                    relation_uri = result.get("relation", {}).get("value", "")
                    if not relation_uri or "/prop/direct/" not in relation_uri:
                        continue
                    
                    pid = relation_uri.split("/")[-1]
                    relation_label = result.get("relationLabel", {}).get("value", "")
                    
                    subject_uri = result.get("subject", {}).get("value", "")
                    if "/entity/" not in subject_uri:
                        continue
                    subject_qid = subject_uri.split("/")[-1]
                    subject_label = result.get("subjectLabel", {}).get("value", subject_qid)
                    subject_desc = result.get("subjectDesc", {}).get("value", "")
                    
                    object_uri = result.get("object", {}).get("value", "")
                    object_type = result.get("object", {}).get("type", "")
                    
                    subject_entity = WikidataEntity(
                        qid=subject_qid, label=subject_label, description=subject_desc,
                        aliases=[], url=f"https://www.wikidata.org/wiki/{subject_qid}"
                    )
                    
                    if object_type == "uri" and "/entity/" in object_uri:
                        object_qid = object_uri.split("/")[-1]
                        object_label = result.get("objectLabel", {}).get("value", object_qid)
                        object_desc = result.get("objectDesc", {}).get("value", "")
                        object_entity = WikidataEntity(
                            qid=object_qid, label=object_label, description=object_desc,
                            aliases=[], url=f"https://www.wikidata.org/wiki/{object_qid}"
                        )
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_entity
                            )
                            triples.append(triple)
                            next_qids.add(object_qid)
                    else:
                        object_value = result.get("object", {}).get("value", "")
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_value
                            )
                            triples.append(triple)
            except Exception as e:
                logger.warning(f"Error executing async outgoing query for {entity_qid}: {e}")
            
            return entity_qid, triples, next_qids
        
        for hop in range(k):
            next_level: Set[str] = set()
            entities_to_query = [e for e in current_level if e not in visited_entities]
            visited_entities.update(entities_to_query)
            
            if not entities_to_query:
                break
            
            tasks = [execute_outgoing_query_async(eq) for eq in entities_to_query]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error in async query: {result}")
                    continue
                entity_qid, triples, next_qids = result
                all_triples.extend(triples)
                if hop < k - 1:
                    next_level.update(next_qids)
            
            current_level = next_level
            if not current_level:
                break
        
        # Deduplicate triples
        seen = set()
        unique_triples = []
        for triple in all_triples:
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)
            
            relation = triple.relation
            if relation.pid in self.wikidata_props_with_labels:
                relation.label = self.wikidata_props_with_labels[relation.pid]["label"]
                relation.description = self.wikidata_props_with_labels[relation.pid]["description"]
            
            updated_triple = WikiTriple(subject=triple.subject, object=triple.object, relation=relation)
            unique_triples.append(updated_triple)
        
        logger.info(f"Retrieved {len(unique_triples)} unique outgoing triples async for {qid}")
        return unique_triples

    def model_post_init(self, __context) -> None:
        """Initialize wikidata_props_with_labels after model initialization."""
        super().model_post_init(__context)
        
        # Populate wikidata_props_with_labels with labels and descriptions for all wikidata_props
        self.wikidata_props = list(set(self.wikidata_props))  # Ensure uniqueness
        if self.wikidata_props:
            # Filter properties that need to be fetched
            props_to_fetch = [p for p in self.wikidata_props if p not in self.wikidata_props_with_labels]
            
            if props_to_fetch:
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

    def _get_id(self, query: str) -> Optional[List[str]]:
        """Retrieve a list of Wikidata QIDs based on a textual query."""
        clipped_query = query[:WIKIDATA_MAX_QUERY_LENGTH]
        items = self.wikidata_mw.search(clipped_query, results=self.top_k_results)
        if items:
            return items[: self.top_k_results]
        else:
            return None
        
    def _get_item(self, qid: str) -> Optional[WikidataEntity]:
        """Retrieve a Wikidata entity by its QID using optimized SPARQL queries."""
        # Build property filter for optimized query
        if self.wikidata_props:
            prop_filters = " || ".join([
                f'?prop = wdt:{prop}'
                for prop in self.wikidata_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            # If no specific properties, get all direct properties
            property_filter = "FILTER(STRSTARTS(STR(?prop), 'http://www.wikidata.org/prop/direct/'))"
        
        # Single comprehensive SPARQL query to get all entity information
        query = f"""
        SELECT ?label ?description ?alias ?wikipedia ?prop ?value ?valueLabel
        WHERE {{
          # Get label
          OPTIONAL {{ wd:{qid} rdfs:label ?label . FILTER(LANG(?label) = "{self.lang}") }}
          
          # Get description
          OPTIONAL {{ wd:{qid} schema:description ?description . FILTER(LANG(?description) = "{self.lang}") }}
          
          # Get aliases
          OPTIONAL {{ wd:{qid} skos:altLabel ?alias . FILTER(LANG(?alias) = "{self.lang}") }}
          
          # Get Wikipedia URL
          OPTIONAL {{
            ?wikipedia schema:about wd:{qid} ;
                      schema:inLanguage "{self.lang}" ;
                      schema:isPartOf <https://{self.lang}.wikipedia.org/> .
          }}
          
          # Get all properties and their values with labels
          OPTIONAL {{
            wd:{qid} ?prop ?value .
            {property_filter}
            OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "{self.lang}") }}
          }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
        }}
        """
        
        try:
            results = self._execute_sparql_with_retry(query)
            
            if not results or not results["results"]["bindings"]:
                logger.warning(f"Could not find item {qid} in Wikidata")
                return None
            
            # Parse results
            label = ""
            description = ""
            aliases = []
            wikipedia_url = None
            properties = {}
            
            for result in results["results"]["bindings"]:
                # Extract label (same for all rows)
                if not label and "label" in result:
                    label = result["label"]["value"]
                
                # Extract description (same for all rows)
                if not description and "description" in result:
                    description = result["description"]["value"]
                
                # Extract aliases (can be multiple)
                if "alias" in result:
                    alias_val = result["alias"]["value"]
                    if alias_val not in aliases:
                        aliases.append(alias_val)
                
                # Extract Wikipedia URL (same for all rows)
                if not wikipedia_url and "wikipedia" in result:
                    wikipedia_url = result["wikipedia"]["value"]
                
                # Extract properties and values
                if "prop" in result and "value" in result:
                    prop_uri = result["prop"]["value"]
                    if "/prop/direct/" in prop_uri:
                        prop_id = prop_uri.split("/")[-1]
                        # Get property label from wikidata_props_with_labels cache
                        prop_label = prop_id
                        if prop_id in self.wikidata_props_with_labels:
                            prop_label = self.wikidata_props_with_labels[prop_id]["label"]
                        
                        # Get value (use label if available, otherwise raw value)
                        value_data = result["value"]
                        if value_data["type"] == "uri" and "/entity/" in value_data["value"]:
                            # It's an entity reference
                            value_str = result.get("valueLabel", {}).get("value", value_data["value"].split("/")[-1])
                        else:
                            # It's a literal value
                            value_str = value_data["value"]
                        
                        # Group multiple values for same property
                        if prop_id not in properties:
                            properties[prop_id] = {"label": prop_label, "values": []}
                        if value_str not in properties[prop_id]["values"]:
                            properties[prop_id]["values"].append(value_str)
            
            # Build wikidata_content from properties
            wikidata_content_lines = []
            if label:
                wikidata_content_lines.append(f"Label: {label}")
            if description:
                wikidata_content_lines.append(f"Description: {description}")
            if aliases:
                wikidata_content_lines.append(f"Aliases: {', '.join(aliases)}")
            
            # Add properties with their labels
            for prop_id, prop_data in properties.items():
                if prop_id not in self.wikidata_props_with_labels:
                    continue  # Skip properties not in the specified list
                prop_label = prop_data["label"]
                values_str = ", ".join(prop_data["values"])
                wikidata_content_lines.append(f"{prop_label}: {values_str}")
            
            wikidata_content = "\n".join(wikidata_content_lines)
            
            wikipedia_content = None
            if wikipedia_url:
                wikidata_content = WebSearchTool.crawl_web_pages(wikipedia_url)
            return WikidataEntity(
                qid=qid,
                label=label,
                description=description,
                aliases=aliases,
                wikidata_content=wikidata_content,
                wikipedia_content=wikipedia_content,
                url=f"https://www.wikidata.org/wiki/{qid}",
                wikipedia_url=wikipedia_url
            )
            
        except Exception as e:
            logger.error(f"Error fetching item {qid} via SPARQL: {e}")
            return None

    def _get_property(self, pid: str) -> Optional[WikidataProperty]:
        """Retrieve a Wikidata property by its PID."""
        from wikibase_rest_api_client.api.properties import get_property
        
        resp = get_property.sync_detailed(pid, client=self.wikidata_rest)

        try:
            property_data = resp.parsed
            if property_data:
                assert hasattr(property_data, "labels"), f"Retrieved property data missing labels for PID {pid}"
                assert hasattr(property_data, "descriptions"), f"Retrieved property data missing descriptions for PID {pid}"
                label = property_data.labels[self.lang]
                description = property_data.descriptions[self.lang]
                assert label, f"Label for property {pid} not found in language {self.lang}"
                return WikidataProperty(
                    pid=pid,
                    label=label,
                    description=description,
                )
        except Exception as e:
            logger.warning(f"Could not find property {pid} in Wikidata: {e}")
            return None

    def _get_k_hop_bidirectional(self, qid: str, k: int = 1, prop: Optional[str] = None) -> List[WikiTriple]:
        """Retrieve k-hop bidirectional triples for a given entity QID.
        """
        all_triples: List[WikiTriple] = []
        visited_entities: Set[str] = set()
        current_level: Set[str] = {qid}
        
        # Build property filter for SPARQL query using direct property URIs
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            # Create filter for specific direct properties
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{prop}>'
                for prop in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        def execute_bidirectional_query(entity_qid: str) -> Tuple[str, List[WikiTriple], Set[str]]:
            """Execute bidirectional query for a single entity. Returns (qid, triples, next_level_qids)."""
            triples: List[WikiTriple] = []
            next_qids: Set[str] = set()
            
            bidirectional_query = f"""
            SELECT ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel ?object ?objectLabel ?objectDesc ?direction WHERE {{
              {{
                # Outgoing: entity as subject
                BIND(wd:{entity_qid} AS ?subject)
                wd:{entity_qid} ?relation ?object .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("outgoing" AS ?direction)
                
                OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
                OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              }}
              UNION
              {{
                # Incoming: entity as object
                BIND(wd:{entity_qid} AS ?object)
                ?subject ?relation wd:{entity_qid} .
                FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                {property_filter}
                BIND("incoming" AS ?direction)
                
                OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
                OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              }}
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 200
            """
            
            try:
                results = CustomWikidataAPIWrapper._execute_sparql_with_retry(bidirectional_query)
                
                if not results:
                    return entity_qid, triples, next_qids
                
                for result in results["results"]["bindings"]:
                    relation_uri = result.get("relation", {}).get("value", "")
                    if not relation_uri or "/prop/direct/" not in relation_uri:
                        continue
                    
                    pid = relation_uri.split("/")[-1]
                    relation_label = result.get("relationLabel", {}).get("value", "")
                    direction = result.get("direction", {}).get("value", "")
                    
                    subject_uri = result.get("subject", {}).get("value", "")
                    if "/entity/" not in subject_uri:
                        continue
                    subject_qid = subject_uri.split("/")[-1]
                    subject_label = result.get("subjectLabel", {}).get("value", subject_qid)
                    subject_desc = result.get("subjectDesc", {}).get("value", "")
                    
                    object_uri = result.get("object", {}).get("value", "")
                    object_type = result.get("object", {}).get("type", "")
                    
                    subject_entity = WikidataEntity(
                        qid=subject_qid,
                        label=subject_label,
                        description=subject_desc,
                        aliases=[],
                        url=f"https://www.wikidata.org/wiki/{subject_qid}"
                    )
                    
                    if object_type == "uri" and "/entity/" in object_uri:
                        object_qid = object_uri.split("/")[-1]
                        object_label = result.get("objectLabel", {}).get("value", object_qid)
                        object_desc = result.get("objectDesc", {}).get("value", "")
                        
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
                            triples.append(triple)
                            
                            if direction == "outgoing":
                                next_qids.add(object_qid)
                            elif direction == "incoming":
                                next_qids.add(subject_qid)
                    else:
                        object_value = result.get("object", {}).get("value", "")
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_value
                            )
                            triples.append(triple)
                            
            except Exception as e:
                logger.warning(f"Error executing bidirectional query for {entity_qid}: {e}")
            
            return entity_qid, triples, next_qids
        
        for hop in range(k):
            next_level: Set[str] = set()
            
            # Filter entities to query (not yet visited)
            entities_to_query = [e for e in current_level if e not in visited_entities]
            visited_entities.update(entities_to_query)
            
            if not entities_to_query:
                break
            
            # Parallel execution of queries for all entities at this level
            with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(entities_to_query))) as executor:
                futures = {executor.submit(execute_bidirectional_query, eq): eq for eq in entities_to_query}
                
                for future in as_completed(futures):
                    try:
                        entity_qid, triples, next_qids = future.result()
                        all_triples.extend(triples)
                        if hop < k - 1:
                            next_level.update(next_qids)
                    except Exception as e:
                        eq = futures[future]
                        logger.warning(f"Error processing entity {eq}: {e}")
            
            # Move to next level
            current_level = next_level
            if not current_level:
                break
        
        # Deduplicate triples at the end
        seen = set()
        unique_triples = []
        for triple in all_triples:
            # Create unique identifier based on subject, relation, and object
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)

            # Update relation with label and description if available
            relation = triple.relation
            if relation.pid in self.wikidata_props_with_labels:
                relation.label = self.wikidata_props_with_labels[relation.pid]["label"]
                relation.description = self.wikidata_props_with_labels[relation.pid]["description"]
            # Fall back to fetching property details if label is missing
            if not relation.label:
                relation = self._get_property(relation.pid)
                if not relation:
                    continue
                self.wikidata_props_with_labels[relation.pid] = {
                    "label": relation.label,
                    "description": relation.description
                }

            updated_triple = WikiTriple(subject=triple.subject, object=triple.object, relation=relation)
            unique_triples.append(updated_triple)

        logger.info(f"Retrieved {len(unique_triples)} unique triples (from {len(all_triples)} total) for {qid} with {k}-hop traversal")
        return unique_triples

    def _get_k_hop_outgoing(self, qid: str, k: int = 1, prop: Optional[str] = None) -> List[WikiTriple]:
        """Retrieve k-hop outgoing triples for a given entity QID.
        
        Args:
            qid: The Wikidata QID of the entity (e.g., 'Q42')
            k: Number of hops to traverse (default: 1)
        Returns:
            List of WikiTriple objects representing (subject, relation, object) triples
        """
        all_triples: List[WikiTriple] = []
        visited_entities: Set[str] = set()
        current_level: Set[str] = {qid}
        
        # Build property filter for SPARQL query using direct property URIs
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_filters = " || ".join([
                f'?relation = <http://www.wikidata.org/prop/direct/{prop}>'
                for prop in to_use_props
            ])
            property_filter = f"FILTER({prop_filters})"
        else:
            property_filter = ""
        
        def execute_outgoing_query(entity_qid: str) -> Tuple[str, List[WikiTriple], Set[str]]:
            """Execute outgoing query for a single entity. Returns (qid, triples, next_level_qids)."""
            triples: List[WikiTriple] = []
            next_qids: Set[str] = set()
            
            outgoing_query = f"""
            SELECT ?subject ?subjectLabel ?subjectDesc ?relation ?relationLabel ?object ?objectLabel ?objectDesc WHERE {{
              BIND(wd:{entity_qid} AS ?subject)
              wd:{entity_qid} ?relation ?object .
              FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
              {property_filter}
              
              OPTIONAL {{ ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?subject schema:description ?subjectDesc . FILTER(LANG(?subjectDesc) = "{self.lang}") }}
              OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "{self.lang}") }}
              OPTIONAL {{ ?object schema:description ?objectDesc . FILTER(LANG(?objectDesc) = "{self.lang}") }}
              
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.lang}" . }}
            }}
            LIMIT 200
            """
            
            try:
                results = CustomWikidataAPIWrapper._execute_sparql_with_retry(outgoing_query)
                
                if not results:
                    return entity_qid, triples, next_qids
                
                for result in results["results"]["bindings"]:
                    relation_uri = result.get("relation", {}).get("value", "")
                    if not relation_uri or "/prop/direct/" not in relation_uri:
                        continue
                    
                    pid = relation_uri.split("/")[-1]
                    relation_label = result.get("relationLabel", {}).get("value", "")
                    
                    subject_uri = result.get("subject", {}).get("value", "")
                    if "/entity/" not in subject_uri:
                        continue
                    subject_qid = subject_uri.split("/")[-1]
                    subject_label = result.get("subjectLabel", {}).get("value", subject_qid)
                    subject_desc = result.get("subjectDesc", {}).get("value", "")
                    
                    object_uri = result.get("object", {}).get("value", "")
                    object_type = result.get("object", {}).get("type", "")
                    
                    subject_entity = WikidataEntity(
                        qid=subject_qid,
                        label=subject_label,
                        description=subject_desc,
                        aliases=[],
                        url=f"https://www.wikidata.org/wiki/{subject_qid}"
                    )
                    
                    if object_type == "uri" and "/entity/" in object_uri:
                        object_qid = object_uri.split("/")[-1]
                        object_label = result.get("objectLabel", {}).get("value", object_qid)
                        object_desc = result.get("objectDesc", {}).get("value", "")
                        
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
                            triples.append(triple)
                            next_qids.add(object_qid)
                    else:
                        object_value = result.get("object", {}).get("value", "")
                        if pid:
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=relation_label, description=""),
                                object=object_value
                            )
                            triples.append(triple)
                            
            except Exception as e:
                logger.warning(f"Error executing outgoing query for {entity_qid}: {e}")
            
            return entity_qid, triples, next_qids
        
        for hop in range(k):
            next_level: Set[str] = set()
            
            # Filter entities to query (not yet visited)
            entities_to_query = [e for e in current_level if e not in visited_entities]
            visited_entities.update(entities_to_query)
            
            if not entities_to_query:
                break
            
            # Parallel execution of queries for all entities at this level
            with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(entities_to_query))) as executor:
                futures = {executor.submit(execute_outgoing_query, eq): eq for eq in entities_to_query}
                
                for future in as_completed(futures):
                    try:
                        entity_qid, triples, next_qids = future.result()
                        all_triples.extend(triples)
                        if hop < k - 1:
                            next_level.update(next_qids)
                    except Exception as e:
                        eq = futures[future]
                        logger.warning(f"Error processing entity {eq}: {e}")
            
            # Move to next level
            current_level = next_level
            if not current_level:
                break
        
        # Deduplicate triples at the end
        seen = set()
        unique_triples = []
        for triple in all_triples:
            # Create unique identifier based on subject, relation, and object
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id in seen:
                continue
            seen.add(triple_id)

            # Update relation with label and description if available
            relation = triple.relation
            if relation.pid in self.wikidata_props_with_labels:
                relation.label = self.wikidata_props_with_labels[relation.pid]["label"]
                relation.description = self.wikidata_props_with_labels[relation.pid]["description"]
            # Fall back to fetching property details if label is missing
            if not relation.label:
                relation = self._get_property(relation.pid)
                if not relation:
                    continue
                self.wikidata_props_with_labels[relation.pid] = {
                    "label": relation.label,
                    "description": relation.description
                }

            updated_triple = WikiTriple(subject=triple.subject, object=triple.object, relation=relation)
            unique_triples.append(updated_triple)

        logger.info(f"Retrieved {len(unique_triples)} unique outgoing triples (from {len(all_triples)} total) for {qid} with {k}-hop traversal")
        return unique_triples


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

    def _run(self, query: str, num_entities: int = 3, num_workers: int = 4) -> List[WikidataEntity]:
        search_results = self.wikidata_wrapper._get_id(query)
        if not search_results:
            return []
        
        qids_to_fetch = search_results[:num_entities]
        if not qids_to_fetch:
            return []
        
        entities: List[WikidataEntity] = []
        
        def fetch_entity(qid: str) -> Optional[WikidataEntity]:
            try:
                return self.wikidata_wrapper._get_item(qid)
            except Exception as e:
                logger.error(f"Error fetching entity {qid}: {e}")
                return None
        
        # Parallel fetch entities
        with ThreadPoolExecutor(max_workers=min(num_workers, len(qids_to_fetch))) as executor:
            futures = {executor.submit(fetch_entity, qid): qid for qid in qids_to_fetch}
            # Collect results preserving order
            results_map = {}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    entity = future.result()
                    if entity:
                        results_map[qid] = entity
                except Exception as e:
                    logger.error(f"Error fetching entity {qid}: {e}")
            
            # Preserve original order
            for qid in qids_to_fetch:
                if qid in results_map:
                    entities.append(results_map[qid])
        
        return entities

    async def _arun(self, query: str, num_entities: int = 3) -> List[WikidataEntity]:
        """Async version of entity retrieval."""
        search_results = self.wikidata_wrapper._get_id(query)
        if not search_results:
            return []
        
        qids_to_fetch = search_results[:num_entities]
        if not qids_to_fetch:
            return []
        
        # Fetch entities concurrently using asyncio.gather
        tasks = [self.wikidata_wrapper._get_item_async(qid) for qid in qids_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results preserving order
        entities: List[WikidataEntity] = []
        for qid, result in zip(qids_to_fetch, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching entity {qid}: {result}")
                continue
            if result:
                entities.append(result)
        
        return entities


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
    
    def get_entity_details_tool(self, qids: Set[str], num_workers: int = 4) -> Dict[str, WikidataEntity]:
        entity_details: Dict[str, WikidataEntity] = {}
        
        if not qids:
            return entity_details
        
        def fetch_entity(qid: str) -> Tuple[str, Optional[WikidataEntity]]:
            try:
                entity = self.wikidata_wrapper._get_item(qid)
                return qid, entity
            except Exception as e:
                logger.error(f"Error fetching details for {qid}: {e}")
                return qid, None

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=min(num_workers, len(qids))) as executor:
            # Submit all tasks
            futures = {executor.submit(fetch_entity, qid): qid for qid in qids}
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    qid_result, entity = future.result()
                    if entity:
                        entity_details[qid_result] = entity
                except Exception as e:
                    qid = futures[future]
                    logger.error(f"Error fetching details for {qid}: {e}")
        
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
        update_with_details: bool = True,
        ) -> List[WikiTriple]:
        # wikidata_wrapper = CustomWikidataAPIWrapper(lang=lang, top_k_results=num_entities)
        # Add assertion for the case is_qids is False and query is list
        assert isinstance(query, list) and not is_qids, "When is_qids is False, query must be a string."
        if is_qids:
            search_results = query if isinstance(query, list) else [query]
        else:
            search_results = self.wikidata_wrapper._get_id(query)
        if not search_results:
            return None
        if prop:
            if prop not in self.wikidata_wrapper.wikidata_props:
                logger.warning(f"Property {prop} not in the specified wikidata_props.")
                return None
        all_triples: List[WikiTriple] = []
        for qid in search_results[: num_entities]:
            if bidirectional:
                triples = self.wikidata_wrapper._get_k_hop_bidirectional(qid, k=k)
            else:
                triples = self.wikidata_wrapper._get_k_hop_outgoing(qid, k=k)
            if triples:
                all_triples.extend(triples)
    
        # Deduplicate triples
        all_triples = self.dedup_triples_tool(all_triples)
        if update_with_details:
            # Update triples with full entity details
            all_triples = self.update_triples_with_details_tool(all_triples)
        return all_triples

    async def get_entity_details_tool_async(self, qids: Set[str]) -> Dict[str, WikidataEntity]:
        """Async version of entity details fetching."""
        entity_details: Dict[str, WikidataEntity] = {}
        
        if not qids:
            return entity_details
        
        tasks = [self.wikidata_wrapper._get_item_async(qid) for qid in qids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for qid, result in zip(qids, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching details for {qid}: {result}")
                continue
            if result:
                entity_details[qid] = result
        
        return entity_details
    
    async def update_triples_with_details_tool_async(self, all_triples: List[WikiTriple]) -> List[WikiTriple]:
        """Async version of updating triples with full entity details."""
        updated_triples: List[WikiTriple] = []
        all_entity_qids: Set[str] = set()
        for triple in all_triples:
            all_entity_qids.add(triple.subject.qid)
            if hasattr(triple.object, 'qid'):
                all_entity_qids.add(triple.object.qid)
        
        entity_details = await self.get_entity_details_tool_async(all_entity_qids)
        
        for triple in all_triples:
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
        return updated_triples

    async def _arun(
        self, 
        query: Union[str, List[str]],
        is_qids: bool = False,
        k: int = 1, 
        num_entities: int = 3, 
        bidirectional: bool = False,
        prop: Optional[str] = None,
        update_with_details: bool = True,
    ) -> List[WikiTriple]:
        """Async version of k-hop triples retrieval."""
        assert isinstance(query, list) and not is_qids, "When is_qids is False, query must be a string."
        if is_qids:
            search_results = query if isinstance(query, list) else [query]
        else:
            search_results = self.wikidata_wrapper._get_id(query)
        if not search_results:
            return []
        
        if prop and prop not in self.wikidata_wrapper.wikidata_props:
            logger.warning(f"Property {prop} not in the specified wikidata_props.")
            return []
        
        # Fetch triples for all search results concurrently
        if bidirectional:
            tasks = [
                self.wikidata_wrapper._get_k_hop_bidirectional_async(qid, k=k, prop=prop)
                for qid in search_results[:num_entities]
            ]
        else:
            tasks = [
                self.wikidata_wrapper._get_k_hop_outgoing_async(qid, k=k, prop=prop)
                for qid in search_results[:num_entities]
            ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_triples: List[WikiTriple] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching triples: {result}")
                continue
            if result:
                all_triples.extend(result)
        
        # Deduplicate triples
        all_triples = self.dedup_triples_tool(all_triples)
        
        if update_with_details:
            all_triples = await self.update_triples_with_details_tool_async(all_triples)
        
        return all_triples
    
