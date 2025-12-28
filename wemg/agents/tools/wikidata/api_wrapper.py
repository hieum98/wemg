"""Custom Wikidata API Wrapper with enhanced item retrieval and centralized enrichment."""

import asyncio
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from langchain_community.tools.wikidata.tool import WikidataAPIWrapper

from wemg.agents.tools.web_search import WebSearchTool
from wemg.agents.tools.wikidata.constants import (
    WIKIDATA_MAX_QUERY_LENGTH,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_DELAY,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    USER_AGENT,
    DEFAULT_PROPERTIES,
    PROPERTY_LABELS,
    BATCH_SIZE_ENTITY_SEARCH,
    BATCH_SIZE_ENTITY_RETRIEVAL,
    BATCH_SIZE_PROPERTY_RETRIEVAL,
    BATCH_SIZE_TRIPLE_QUERY,
    LIMIT_PER_QUERY,
    MAX_ENTITIES_PER_HOP,
)
from wemg.agents.tools.wikidata.models import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
)
from wemg.agents.tools.wikidata.rate_limiter import (
    get_sync_semaphore,
    get_async_semaphore,
)
from wemg.agents.tools.wikidata.utils import (
    normalize_and_validate_qid,
    normalize_and_validate_pid,
    extract_id_from_uri,
    normalize_single_or_list,
    map_results_to_indices,
    unwrap_single_result,
    flatten_and_map_ids,
    build_result_map,
    create_minimal_entity,
    create_minimal_property,
    build_property_filter,
    build_property_values_clause,
    validate_and_normalize_ids,
)

logger = logging.getLogger(__name__)


class CustomWikidataAPIWrapper(WikidataAPIWrapper):
    """Custom Wikidata API Wrapper with enhanced item retrieval and centralized enrichment."""

    wikidata_props_with_labels: Dict[str, Dict[str, Optional[str]]] = {}
    wikidata_props: List[str] = DEFAULT_PROPERTIES

    @staticmethod
    def _execute_sparql_with_retry(
        query: str,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY,
        timeout: int = 12
    ) -> Optional[Dict]:
        """Execute a SPARQL query with retry logic and rate limiting."""
        url = "https://query.wikidata.org/sparql"
        data = urllib.parse.urlencode({'query': query}).encode('utf-8')
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/sparql-results+json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        semaphore = get_sync_semaphore()
        
        for attempt in range(max_retries):
            try:
                with semaphore:
                    # Add small random delay to avoid thundering herd
                    if attempt > 0:
                        delay = REQUEST_DELAY + random.uniform(0, 0.1)
                        time.sleep(delay)
                    elif REQUEST_DELAY > 0.05:
                        delay = REQUEST_DELAY * 0.5 + random.uniform(0, 0.05)
                        time.sleep(delay)
                    
                    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        results = json.loads(response.read().decode('utf-8'))
                        return results
                    
            except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
                error_str = str(e).lower()
                
                is_rate_limit = any(x in error_str for x in [
                    "429", "too many requests", "rate limit",
                    "503", "service unavailable", "timeout",
                    "504", "gateway timeout", "gateway",
                    "500", "internal server error",
                    "read operation timed out", "timed out"
                ])
                
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limited or server error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                elif attempt < max_retries - 1:
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
        
        semaphore = get_async_semaphore()
        
        for attempt in range(max_retries):
            try:
                async with semaphore:
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

    def model_post_init(self, __context) -> None:
        """Initialize wikidata_props_with_labels after model initialization."""
        super().model_post_init(__context)
        
        self.wikidata_props = list(set(self.wikidata_props))
        
        # Load from PROPERTY_LABELS first
        for pid in self.wikidata_props:
            if pid in PROPERTY_LABELS:
                self.wikidata_props_with_labels[pid] = PROPERTY_LABELS[pid]
        
        # Fetch missing properties
        props_to_fetch = [p for p in self.wikidata_props if p not in self.wikidata_props_with_labels]
        
        if props_to_fetch:
            logger.info(f"Fetching labels for {len(props_to_fetch)} Wikidata properties...")
            
            def fetch_property(prop_id: str) -> Tuple[str, Optional[Dict[str, Optional[str]]]]:
                semaphore = get_sync_semaphore()
                with semaphore:
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
        """Search for Wikidata IDs for a single query or a batch."""
        is_single, query_list = normalize_single_or_list(query)
        
        if not all(isinstance(q, str) or q is None for q in query_list):
            raise TypeError("query must be a string or a list of strings")

        normalized: List[str] = [((q or "").strip()) for q in query_list]
        output: List[List[str]] = [[] for _ in normalized]

        pending: List[Tuple[int, str]] = []
        for idx, q in enumerate(normalized):
            if not q:
                continue

            # Check if it's already a valid ID
            if id_type == "property":
                direct = normalize_and_validate_pid(q)
            else:
                direct = normalize_and_validate_qid(q)
            
            if direct:
                output[idx] = [direct]
                continue

            pending.append((idx, q[:WIKIDATA_MAX_QUERY_LENGTH]))

        if not pending:
            return unwrap_single_result(output, is_single)

        for start in range(0, len(pending), BATCH_SIZE_ENTITY_SEARCH):
            chunk = pending[start : start + BATCH_SIZE_ENTITY_SEARCH]
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

        return unwrap_single_result(output, is_single)

    def _search_ids_via_sparql(self, queries: List[str], *, id_type: str, limit: int) -> Dict[str, List[str]]:
        """Internal: search IDs for multiple queries via SPARQL mwapi EntitySearch."""
        if not queries:
            return {}

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
        tmp: Dict[str, List[Tuple[int, str]]] = {q: [] for q in queries}

        for row in bindings:
            if not isinstance(row, dict):
                continue
            search_val = (row.get("search") or {}).get("value")
            entity_uri = (row.get("entity") or {}).get("value")
            ordinal_raw = (row.get("ordinal") or {}).get("value")
            if not isinstance(search_val, str) or search_val not in tmp:
                continue
            entity_id = extract_id_from_uri(entity_uri)
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
        """Retrieve Wikidata entities by QID(s) using optimized batch SPARQL queries."""
        is_single, qids = normalize_single_or_list(qid)
        
        valid_qids, qid_to_indices = validate_and_normalize_ids(qids, id_type="item")
        
        if not valid_qids:
            results: List[Optional[WikidataEntity]] = [None] * len(qids)
            return unwrap_single_result(results, is_single)
        
        unique_qids = list(set(valid_qids))
        entity_map: Dict[str, WikidataEntity] = {}
        
        for batch_start in range(0, len(unique_qids), BATCH_SIZE_ENTITY_RETRIEVAL):
            batch_qids = unique_qids[batch_start:batch_start + BATCH_SIZE_ENTITY_RETRIEVAL]
            batch_entities = self._get_items_batch(batch_qids, get_details=get_details)
            entity_map.update(batch_entities)
        
        results = map_results_to_indices(entity_map, qid_to_indices, len(qids))
        return unwrap_single_result(results, is_single)

    def _get_items_batch(self, qids: List[str], get_details: bool = False) -> Dict[str, WikidataEntity]:
        """Retrieve multiple Wikidata entities in a single SPARQL query."""
        if not qids:
            return {}
        
        values_clause = " ".join([f"wd:{qid}" for qid in qids])
        
        # Build property VALUES clause for filtering
        if self.wikidata_props:
            prop_values = " ".join([f"wd:{pid}" for pid in self.wikidata_props])
            property_values_clause = f"VALUES ?property {{ {prop_values} }}"
        else:
            property_values_clause = ""
        
        # Query using wikibase:directClaim pattern for reliable label resolution
        # Property/object retrieval is OPTIONAL so entities without matching properties still return labels
        query = f"""
        SELECT ?entity ?entityLabel ?entityDescription ?entityArticle
               ?property ?propertyLabel ?object ?objectLabel ?objectDescription
        WHERE {{
          VALUES ?entity {{ {values_clause} }}
          
          # Optional: Get the statement using direct claim pattern
          OPTIONAL {{
            ?entity ?p ?object .
            
            # Map predicate to property ID
            ?property wikibase:directClaim ?p .
            
            # Filter to specific properties if configured
            {property_values_clause}
          }}
          
          # Get Wikipedia URL for the entity
          OPTIONAL {{
            ?entityArticle schema:about ?entity ;
                           schema:isPartOf <https://{self.lang}.wikipedia.org/> .
          }}
          
          # Label service with language fallbacks
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "{self.lang},en,en-gb,mul" . 
          }}
        }}
        """
        
        try:
            results = self._execute_sparql_with_retry(query)
            if not results or not results["results"]["bindings"]:
                logger.warning(f"Could not find any items for QIDs: {qids}")
                return {}
            
            entity_data: Dict[str, Dict[str, Any]] = {}
            
            for row in results["results"]["bindings"]:
                # Extract entity QID
                entity_uri = row.get("entity", {}).get("value", "")
                if "/entity/" not in entity_uri:
                    continue
                entity_qid = entity_uri.split("/")[-1].upper()
                
                if entity_qid not in entity_data:
                    entity_data[entity_qid] = {
                        "label": "",
                        "description": "",
                        "aliases": [],
                        "wikipedia_url": None,
                        "properties": {}
                    }
                
                data = entity_data[entity_qid]
                
                # Get entity label from label service
                if not data["label"] and "entityLabel" in row:
                    label_val = row["entityLabel"]["value"]
                    # Avoid using QID as label
                    if not (label_val.startswith("Q") and label_val[1:].isdigit()):
                        data["label"] = label_val
                
                # Get entity description from label service
                if not data["description"] and "entityDescription" in row:
                    data["description"] = row["entityDescription"]["value"]
                
                # Get Wikipedia URL
                if not data["wikipedia_url"] and "entityArticle" in row:
                    data["wikipedia_url"] = row["entityArticle"]["value"]
                
                # Extract property and object
                if "property" in row and "object" in row:
                    prop_uri = row["property"]["value"]
                    if "/entity/" in prop_uri:
                        prop_id = prop_uri.split("/")[-1].upper()
                        
                        # Get property label from label service or cache
                        prop_label = row.get("propertyLabel", {}).get("value", prop_id)
                        if prop_id in self.wikidata_props_with_labels:
                            cached_label = self.wikidata_props_with_labels[prop_id].get("label")
                            if cached_label:
                                prop_label = cached_label
                        
                        # Get object value
                        object_data = row["object"]
                        if object_data["type"] == "uri" and "/entity/" in object_data["value"]:
                            # Object is an entity - use objectLabel
                            value_str = row.get("objectLabel", {}).get("value", object_data["value"].split("/")[-1])
                        else:
                            # Object is a literal value
                            value_str = object_data["value"]
                        
                        if prop_id not in data["properties"]:
                            data["properties"][prop_id] = {"label": prop_label, "values": []}
                        if value_str not in data["properties"][prop_id]["values"]:
                            data["properties"][prop_id]["values"].append(value_str)
            
            entity_map: Dict[str, WikidataEntity] = {}
            wikipedia_urls: Dict[str, str] = {}
            for qid_key, data in entity_data.items():
                if data["wikipedia_url"]:
                    wikipedia_urls[qid_key] = data["wikipedia_url"]
            
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
                wikidata_content_lines = []
                if data["label"]:
                    wikidata_content_lines.append(f"Label: {data['label']}")
                if data["description"]:
                    wikidata_content_lines.append(f"Description: {data['description']}")
                if data["aliases"]:
                    wikidata_content_lines.append(f"Aliases: {', '.join(data['aliases'])}")
                
                for prop_id, prop_data in data["properties"].items():
                    if prop_id not in self.wikidata_props_with_labels:
                        continue
                    prop_label = prop_data["label"]
                    values_str = ", ".join(prop_data["values"])
                    wikidata_content_lines.append(f"{prop_label}: {values_str}")
                
                wikidata_content = "\n".join(wikidata_content_lines)
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

    async def _get_item_async(self, qid: Union[str, List[str]], get_details: bool = False) -> Union[Optional[WikidataEntity], List[Optional[WikidataEntity]]]:
        """Async wrapper for `_get_item` (supports batch)."""
        return await asyncio.to_thread(self._get_item, qid, get_details=get_details)

    def _get_property(self, pid: Union[str, List[str]]) -> Union[Optional[WikidataProperty], List[Optional[WikidataProperty]]]:
        """Retrieve Wikidata properties by PID(s) using SPARQL batch queries."""
        is_single, pids = normalize_single_or_list(pid)
        
        valid_pids, pid_to_indices = validate_and_normalize_ids(pids, id_type="property")
        
        if not valid_pids:
            results: List[Optional[WikidataProperty]] = [None] * len(pids)
            return unwrap_single_result(results, is_single)
        
        property_map: Dict[str, WikidataProperty] = {}
        
        # Use cached properties from PROPERTY_LABELS
        to_fetch_pids = [p for p in valid_pids if p not in PROPERTY_LABELS]
        for pid_key in valid_pids:
            if pid_key in PROPERTY_LABELS:
                prop_info = PROPERTY_LABELS[pid_key]
                property_map[pid_key] = WikidataProperty(
                    pid=pid_key,
                    label=prop_info.get("label", pid_key),
                    description=prop_info.get("description")
                )
        
        unique_pids = list(set(to_fetch_pids))
        
        for batch_start in range(0, len(unique_pids), BATCH_SIZE_PROPERTY_RETRIEVAL):
            batch_pids = unique_pids[batch_start:batch_start + BATCH_SIZE_PROPERTY_RETRIEVAL]
            batch_props = self._get_properties_batch(batch_pids)
            property_map.update(batch_props)
        
        results = map_results_to_indices(property_map, pid_to_indices, len(pids))
        return unwrap_single_result(results, is_single)

    def _get_properties_batch(self, pids: List[str]) -> Dict[str, WikidataProperty]:
        """Retrieve multiple Wikidata properties in a single SPARQL query."""
        if not pids:
            return {}
        
        values_clause = " ".join([f"wd:{pid}" for pid in pids])
        
        # Query using label service for reliable label resolution
        query = f"""
        SELECT ?property ?propertyLabel ?propertyDescription
        WHERE {{
          VALUES ?property {{ {values_clause} }}
          
          # Label service with language fallbacks
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "{self.lang},en,en-gb,mul" . 
          }}
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
                
                # Get label from label service
                label = row.get("propertyLabel", {}).get("value", "")
                # Avoid using PID as label
                if label.startswith("P") and label[1:].isdigit():
                    label = ""
                
                # Get description from label service
                description = row.get("propertyDescription", {}).get("value", "")
                
                if prop_id not in property_map:
                    property_map[prop_id] = WikidataProperty(
                        pid=prop_id,
                        label=label if label else prop_id,
                        description=description
                    )
            
            return property_map
            
        except Exception as e:
            logger.error(f"Error fetching properties {pids} via SPARQL: {e}")
            return {}

    def _get_property_label_and_description(self, pid: str, fallback_label: str = "") -> Tuple[str, str]:
        """Get property label and description from wikidata_props_with_labels if available."""
        if pid in self.wikidata_props_with_labels:
            prop_info = self.wikidata_props_with_labels.get(pid, {})
            if not prop_info:
                prop_info = PROPERTY_LABELS.get(pid, {})
            label = prop_info.get("label") if prop_info.get("label") is not None else fallback_label
            description = prop_info.get("description") if prop_info.get("description") is not None else ""
            return label, description
        return fallback_label, ""

    def _parse_triple_row(self, row: Dict) -> Tuple[Optional[WikiTriple], Optional[str], str]:
        """Parse a SPARQL result row into a WikiTriple.
        
        Returns:
            Tuple of (triple, next_qid for traversal, direction)
        """
        relation_uri = row.get("relation", {}).get("value", "")
        if not relation_uri or "/prop/direct/" not in relation_uri:
            return None, None, ""
        
        pid = relation_uri.split("/")[-1]
        relation_label = row.get("relationLabel", {}).get("value", "") if "relationLabel" in row else ""
        relation_label, relation_description = self._get_property_label_and_description(pid, relation_label)
        direction = row.get("direction", {}).get("value", "")
        
        subject_uri = row.get("subject", {}).get("value", "")
        if "/entity/" not in subject_uri:
            return None, None, ""
        subject_qid = subject_uri.split("/")[-1]
        subject_label = row.get("subjectLabel", {}).get("value", subject_qid) if "subjectLabel" in row else subject_qid
        subject_desc = row.get("subjectDesc", {}).get("value", "") if "subjectDesc" in row else ""
        
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
            object_qid = object_uri.split("/")[-1].upper()
            # Only create WikidataEntity for actual entities (Q...), not properties (P...)
            if object_qid.startswith("Q") and object_qid[1:].isdigit():
                object_label = row.get("objectLabel", {}).get("value", object_qid) if "objectLabel" in row else object_qid
                object_desc = row.get("objectDesc", {}).get("value", "") if "objectDesc" in row else ""
                
                object_entity = WikidataEntity(
                    qid=object_qid,
                    label=object_label,
                    description=object_desc,
                    aliases=[],
                    url=f"https://www.wikidata.org/wiki/{object_qid}"
                )
                
                if pid:
                    # Use label from SPARQL if available, otherwise from cache
                    prop = create_minimal_property(pid, self)
                    if relation_label:
                        prop = WikidataProperty(pid=pid, label=relation_label, description=relation_description)
                    triple = WikiTriple(
                        subject=subject_entity,
                        relation=prop,
                        object=object_entity
                    )
                    next_qid = object_qid
                    return triple, next_qid, direction
            else:
                # Property or other non-entity URI - treat as literal
                object_value = object_uri
                if pid:
                    prop = create_minimal_property(pid, self)
                    if relation_label:
                        prop = WikidataProperty(pid=pid, label=relation_label, description=relation_description)
                    triple = WikiTriple(
                        subject=subject_entity,
                        relation=prop,
                        object=object_value
                    )
                    return triple, None, direction
        else:
            object_value = row.get("object", {}).get("value", "")
            if pid:
                # Use label from SPARQL if available, otherwise from cache
                prop = create_minimal_property(pid, self)
                if relation_label:
                    prop = WikidataProperty(pid=pid, label=relation_label, description=relation_description)
                triple = WikiTriple(
                    subject=subject_entity,
                    relation=prop,
                    object=object_value
                )
                return triple, None, direction
        
        return None, None, ""

    def _deduplicate_triples(self, triples: List[WikiTriple]) -> List[WikiTriple]:
        """Deduplicate triples based on (subject, relation, object) tuple.
        
        Note: This only deduplicates. Enrichment is done at the tool level (see tools.py).
        """
        seen: Set[Tuple] = set()
        unique_triples: List[WikiTriple] = []
        
        for triple in triples:
            if isinstance(triple.object, WikidataEntity):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            if triple_id not in seen:
                seen.add(triple_id)
                unique_triples.append(triple)
        
        return unique_triples

    def _execute_bidirectional_batch(
        self,
        entity_qids: List[str],
        property_filter: str,
        to_use_props: Optional[List[str]] = None
    ) -> Tuple[Dict[str, List[WikiTriple]], Dict[str, Set[str]]]:
        """Execute bidirectional SPARQL query for multiple entities.
        
        Processes entities ONE AT A TIME to avoid timeouts. Splits outgoing and incoming
        into separate queries (no UNION) for maximum reliability.
        Uses simple queries without label service for efficiency (labels enriched later).
        """
        if not entity_qids:
            return {}, {}
        
        all_triples: Dict[str, List[WikiTriple]] = {q: [] for q in entity_qids}
        all_next_qids: Dict[str, Set[str]] = {q: set() for q in entity_qids}
        
        # Use VALUES clause for properties if reasonable number (up to 100 properties)
        use_values_for_props = to_use_props and len(to_use_props) <= 100
        if use_values_for_props and to_use_props:
            prop_values = " ".join([f"<http://www.wikidata.org/prop/direct/{p}>" for p in to_use_props])
            prop_values_clause = f"VALUES ?relation {{ {prop_values} }}"
            props_set = set(to_use_props)
        else:
            prop_values_clause = ""
            props_set = set(to_use_props) if to_use_props else None
        
        for qid in entity_qids:
            # Query 1: Outgoing triples (simple, no label service)
            if use_values_for_props:
                outgoing_query = f"""
                SELECT ?relation ?object
                WHERE {{
                  wd:{qid} ?relation ?object .
                  {prop_values_clause}
                }}
                LIMIT {LIMIT_PER_QUERY}
                """
            else:
                outgoing_query = f"""
                SELECT ?relation ?object
                WHERE {{
                  wd:{qid} ?relation ?object .
                  FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                }}
                LIMIT {LIMIT_PER_QUERY}
                """
            
            try:
                results = self._execute_sparql_with_retry(outgoing_query, timeout=10)
                if results and results.get("results", {}).get("bindings"):
                    for row in results["results"]["bindings"]:
                        relation_uri = row.get("relation", {}).get("value", "")
                        object_uri = row.get("object", {}).get("value", "")
                        object_type = row.get("object", {}).get("type", "")
                        
                        if "/prop/direct/" not in relation_uri:
                            continue
                        
                        pid = relation_uri.split("/")[-1]
                        if props_set and pid not in props_set:
                            continue
                        
                        subject_entity = create_minimal_entity(qid)
                        prop_label, prop_description = self._get_property_label_and_description(pid, "")
                        
                        if object_type == "uri" and "/entity/" in object_uri:
                            object_qid = object_uri.split("/")[-1].upper()
                            # Only create WikidataEntity for actual entities (Q...), not properties (P...)
                            if object_qid.startswith("Q") and object_qid[1:].isdigit():
                                object_entity = create_minimal_entity(object_qid)
                                triple = WikiTriple(
                                    subject=subject_entity,
                                    relation=WikidataProperty(pid=pid, label=prop_label, description=prop_description),
                                    object=object_entity
                                )
                                all_triples[qid].append(triple)
                                all_next_qids[qid].add(object_qid)
                            else:
                                # Property or other non-entity URI - treat as literal
                                triple = WikiTriple(
                                    subject=subject_entity,
                                    relation=WikidataProperty(pid=pid, label=prop_label, description=prop_description),
                                    object=object_uri
                                )
                                all_triples[qid].append(triple)
                        else:
                            object_value = object_uri if object_uri else str(row.get("object", {}).get("value", ""))
                            triple = WikiTriple(
                                subject=subject_entity,
                                relation=WikidataProperty(pid=pid, label=prop_label, description=prop_description),
                                object=object_value
                            )
                            all_triples[qid].append(triple)
            except Exception as e:
                logger.warning(f"Outgoing query failed for {qid}: {e}")
            
            time.sleep(0.01)  # Reduced from 0.15 for better performance
            
            # Query 2: Incoming triples (simple, no label service)
            if use_values_for_props:
                incoming_query = f"""
                SELECT ?subject ?relation
                WHERE {{
                  ?subject ?relation wd:{qid} .
                  {prop_values_clause}
                }}
                LIMIT {LIMIT_PER_QUERY}
                """
            else:
                incoming_query = f"""
                SELECT ?subject ?relation
                WHERE {{
                  ?subject ?relation wd:{qid} .
                  FILTER(STRSTARTS(STR(?relation), "http://www.wikidata.org/prop/direct/"))
                }}
                LIMIT {LIMIT_PER_QUERY}
                """
            
            try:
                results = self._execute_sparql_with_retry(incoming_query, timeout=10)
                if results and results.get("results", {}).get("bindings"):
                    for row in results["results"]["bindings"]:
                        relation_uri = row.get("relation", {}).get("value", "")
                        subject_uri = row.get("subject", {}).get("value", "")
                        
                        if "/prop/direct/" not in relation_uri or "/entity/" not in subject_uri:
                            continue
                        
                        pid = relation_uri.split("/")[-1]
                        if props_set and pid not in props_set:
                            continue
                        
                        subject_qid = subject_uri.split("/")[-1].upper()
                        # Only create entities for actual entities (Q...), not properties (P...)
                        if not (subject_qid.startswith("Q") and subject_qid[1:].isdigit()):
                            continue
                        
                        object_qid = qid
                        
                        subject_entity = create_minimal_entity(subject_qid)
                        object_entity = create_minimal_entity(object_qid)
                        prop_label, prop_description = self._get_property_label_and_description(pid, "")
                        
                        triple = WikiTriple(
                            subject=subject_entity,
                            relation=WikidataProperty(pid=pid, label=prop_label, description=prop_description),
                            object=object_entity
                        )
                        all_triples[qid].append(triple)
                        all_next_qids[qid].add(subject_qid)
            except Exception as e:
                logger.warning(f"Incoming query failed for {qid}: {e}")
            
            time.sleep(0.01)  # Reduced from 0.1 for better performance
        
        return all_triples, all_next_qids

    def _get_k_hop_bidirectional(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop bidirectional triples for given entity QID(s)."""
        is_single, qids = normalize_single_or_list(qid)
        
        normalized_qids, qid_to_indices = validate_and_normalize_ids(qids, id_type="item")
        
        results: List[List[WikiTriple]] = [[] for _ in qids]
        
        if not normalized_qids:
            return unwrap_single_result(results, is_single)
        
        unique_qids = list(set(normalized_qids))
        
        # Determine properties to use (used by _execute_bidirectional_batch)
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        
        qid_triples: Dict[str, List[WikiTriple]] = {q: [] for q in unique_qids}
        visited_entities: Dict[str, Set[str]] = {q: set() for q in unique_qids}
        current_level: Dict[str, Set[str]] = {q: {q} for q in unique_qids}
        
        for hop in range(k):
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
            
            if len(all_entities_this_hop) > MAX_ENTITIES_PER_HOP:
                logger.warning(f"Limiting entities at hop {hop + 1} from {len(all_entities_this_hop)} to {MAX_ENTITIES_PER_HOP}")
                entity_priority = [(len(sources), e) for e, sources in entity_to_source.items()]
                entity_priority.sort(reverse=True)
                all_entities_this_hop = {e for _, e in entity_priority[:MAX_ENTITIES_PER_HOP]}
                entity_to_source = {e: entity_to_source[e] for e in all_entities_this_hop if e in entity_to_source}
            
            hop_triples, next_qids_map = self._execute_bidirectional_batch(
                list(all_entities_this_hop), "", to_use_props=to_use_props
            )
            
            for entity_qid, triples in hop_triples.items():
                source_qids = entity_to_source.get(entity_qid, set())
                for source_qid in source_qids:
                    qid_triples[source_qid].extend(triples)
            
            if hop < k - 1:
                for source_qid in unique_qids:
                    next_level: Set[str] = set()
                    for entity_qid in current_level[source_qid]:
                        if entity_qid in next_qids_map:
                            next_level.update(next_qids_map[entity_qid])
                    current_level[source_qid] = next_level
        
        # Deduplicate triples (enrichment is done at tool level for performance)
        for source_qid in unique_qids:
            unique_triples = self._deduplicate_triples(qid_triples[source_qid])
            qid_triples[source_qid] = unique_triples
            logger.info(f"Retrieved {len(unique_triples)} unique triples for {source_qid} with {k}-hop bidirectional traversal")
        
        for qid_key, indices in qid_to_indices.items():
            triples = qid_triples.get(qid_key, [])
            for idx in indices:
                results[idx] = triples
        
        return unwrap_single_result(results, is_single)

    def _execute_outgoing_batch(
        self,
        entity_qids: List[str],
        property_filter: str
    ) -> Tuple[Dict[str, List[WikiTriple]], Dict[str, Set[str]]]:
        """Execute outgoing SPARQL query for multiple entities in batch."""
        if not entity_qids:
            return {}, {}
        
        all_triples: Dict[str, List[WikiTriple]] = {q: [] for q in entity_qids}
        all_next_qids: Dict[str, Set[str]] = {q: set() for q in entity_qids}
        
        for batch_start in range(0, len(entity_qids), BATCH_SIZE_TRIPLE_QUERY):
            batch_qids = entity_qids[batch_start:batch_start + BATCH_SIZE_TRIPLE_QUERY]
            values_clause = " ".join([f"wd:{qid}" for qid in batch_qids])
            
            # Use wikibase:directClaim pattern for reliable label resolution
            query = f"""
            SELECT ?subject ?subjectLabel ?subjectDescription 
                   ?property ?propertyLabel
                   ?object ?objectLabel ?objectDescription
            WHERE {{
              VALUES ?subject {{ {values_clause} }}
              
              # Get the statement using direct claim pattern
              ?subject ?p ?object .
              
              # Map predicate to property ID
              ?property wikibase:directClaim ?p .
              
              {property_filter}
              
              # Label service with language fallbacks
              SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "{self.lang},en,en-gb,mul" . 
              }}
            }}
            LIMIT 2000
            """
            
            try:
                results = self._execute_sparql_with_retry(query)
                
                if not results or not results["results"]["bindings"]:
                    continue
                
                for row in results["results"]["bindings"]:
                    # Extract subject QID
                    subject_uri = row.get("subject", {}).get("value", "")
                    if "/entity/" not in subject_uri:
                        continue
                    subject_qid = subject_uri.split("/")[-1].upper()
                    
                    # Get subject label and description
                    subject_label = row.get("subjectLabel", {}).get("value", subject_qid)
                    if subject_label.startswith("Q") and subject_label[1:].isdigit():
                        subject_label = subject_qid
                    subject_desc = row.get("subjectDescription", {}).get("value", "")
                    
                    subject_entity = WikidataEntity(
                        qid=subject_qid,
                        label=subject_label,
                        description=subject_desc,
                        aliases=[],
                        url=f"https://www.wikidata.org/wiki/{subject_qid}"
                    )
                    
                    # Extract property
                    prop_uri = row.get("property", {}).get("value", "")
                    if "/entity/" not in prop_uri:
                        continue
                    prop_id = prop_uri.split("/")[-1].upper()
                    prop_label = row.get("propertyLabel", {}).get("value", prop_id)
                    if prop_label.startswith("P") and prop_label[1:].isdigit():
                        # Use cached label if available
                        if prop_id in self.wikidata_props_with_labels:
                            cached = self.wikidata_props_with_labels[prop_id].get("label")
                            if cached:
                                prop_label = cached
                    
                    prop = WikidataProperty(pid=prop_id, label=prop_label, description="")
                    
                    # Extract object
                    object_data = row.get("object", {})
                    object_type = object_data.get("type", "")
                    object_value = object_data.get("value", "")
                    
                    if object_type == "uri" and "/entity/" in object_value:
                        object_qid = object_value.split("/")[-1].upper()
                        # Only create WikidataEntity for actual entities (Q...), not properties (P...)
                        if object_qid.startswith("Q") and object_qid[1:].isdigit():
                            object_label = row.get("objectLabel", {}).get("value", object_qid)
                            if object_label.startswith("Q") and object_label[1:].isdigit():
                                object_label = object_qid
                            object_desc = row.get("objectDescription", {}).get("value", "")
                            
                            object_entity = WikidataEntity(
                                qid=object_qid,
                                label=object_label,
                                description=object_desc,
                                aliases=[],
                                url=f"https://www.wikidata.org/wiki/{object_qid}"
                            )
                            
                            triple = WikiTriple(subject=subject_entity, relation=prop, object=object_entity)
                            all_triples[subject_qid].append(triple)
                            all_next_qids[subject_qid].add(object_qid)
                        else:
                            # Property or other non-entity URI - treat as literal
                            triple = WikiTriple(subject=subject_entity, relation=prop, object=object_value)
                            all_triples[subject_qid].append(triple)
                    else:
                        # Object is a literal value
                        triple = WikiTriple(subject=subject_entity, relation=prop, object=object_value)
                        all_triples[subject_qid].append(triple)
                                
            except Exception as e:
                logger.warning(f"Error executing outgoing batch query: {e}")
        
        return all_triples, all_next_qids

    def _get_k_hop_outgoing(
        self,
        qid: Union[str, List[str]],
        k: int = 1,
        prop: Optional[str] = None
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        """Retrieve k-hop outgoing triples for given entity QID(s)."""
        is_single, qids = normalize_single_or_list(qid)
        
        normalized_qids, qid_to_indices = validate_and_normalize_ids(qids, id_type="item")
        
        results: List[List[WikiTriple]] = [[] for _ in qids]
        
        if not normalized_qids:
            return unwrap_single_result(results, is_single)
        
        unique_qids = list(set(normalized_qids))
        
        # Build property VALUES clause for filtering (uses ?property from wikibase:directClaim)
        to_use_props = [prop] if prop and (prop in self.wikidata_props) else self.wikidata_props
        if to_use_props:
            prop_values = " ".join([f"wd:{p}" for p in to_use_props])
            property_filter = f"VALUES ?property {{ {prop_values} }}"
        else:
            property_filter = ""
        
        qid_triples: Dict[str, List[WikiTriple]] = {q: [] for q in unique_qids}
        visited_entities: Dict[str, Set[str]] = {q: set() for q in unique_qids}
        current_level: Dict[str, Set[str]] = {q: {q} for q in unique_qids}
        
        for hop in range(k):
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
            
            hop_triples, next_qids_map = self._execute_outgoing_batch(
                list(all_entities_this_hop), property_filter
            )
            
            for entity_qid, triples in hop_triples.items():
                source_qids = entity_to_source.get(entity_qid, set())
                for source_qid in source_qids:
                    qid_triples[source_qid].extend(triples)
            
            if hop < k - 1:
                for source_qid in unique_qids:
                    next_level: Set[str] = set()
                    for entity_qid in current_level[source_qid]:
                        if entity_qid in next_qids_map:
                            next_level.update(next_qids_map[entity_qid])
                    current_level[source_qid] = next_level
        
        # Deduplicate triples (enrichment is done at tool level for performance)
        for source_qid in unique_qids:
            unique_triples = self._deduplicate_triples(qid_triples[source_qid])
            qid_triples[source_qid] = unique_triples
            logger.debug(f"Retrieved {len(unique_triples)} unique outgoing triples for {source_qid} with {k}-hop traversal")
        
        for qid_key, indices in qid_to_indices.items():
            triples = qid_triples.get(qid_key, [])
            for idx in indices:
                results[idx] = triples
        
        return unwrap_single_result(results, is_single)

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

