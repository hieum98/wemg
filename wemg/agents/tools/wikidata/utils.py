"""Common utilities for Wikidata operations - validation, normalization, batch processing."""

import re
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

from typing import TYPE_CHECKING

from wemg.agents.tools.wikidata.models import (
    WikidataEntity,
    WikidataProperty,
)

if TYPE_CHECKING:
    from wemg.agents.tools.wikidata.api_wrapper import CustomWikidataAPIWrapper

T = TypeVar('T')


def normalize_and_validate_qid(qid: str) -> Optional[str]:
    """Normalize and validate a Wikidata QID.
    
    Args:
        qid: QID string (e.g., 'Q42', 'q42', ' Q42 ')
        
    Returns:
        Normalized QID (e.g., 'Q42') or None if invalid
    """
    if not qid or not isinstance(qid, str):
        return None
    normalized = qid.strip().upper()
    if re.fullmatch(r"Q\d+", normalized):
        return normalized
    return None


def normalize_and_validate_pid(pid: str) -> Optional[str]:
    """Normalize and validate a Wikidata PID.
    
    Args:
        pid: PID string (e.g., 'P31', 'p31', ' P31 ')
        
    Returns:
        Normalized PID (e.g., 'P31') or None if invalid
    """
    if not pid or not isinstance(pid, str):
        return None
    normalized = pid.strip().upper()
    if re.fullmatch(r"P\d+", normalized):
        return normalized
    return None


def extract_id_from_uri(uri: str) -> Optional[str]:
    """Extract QID or PID from a Wikidata URI.
    
    Args:
        uri: URI string (e.g., 'http://www.wikidata.org/entity/Q42')
        
    Returns:
        Extracted ID (e.g., 'Q42') or None if not found
    """
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


def normalize_single_or_list(value: Union[T, List[T]]) -> Tuple[bool, List[T]]:
    """Normalize a single value or list to a list, tracking if it was single.
    
    Args:
        value: Either a single value or a list of values
        
    Returns:
        Tuple of (is_single, list_of_values)
    """
    is_single = isinstance(value, str) if isinstance(value, (str, list)) else not isinstance(value, list)
    if is_single:
        return True, [value]
    else:
        return False, list(value)


def map_results_to_indices(
    results: Dict[str, T],
    id_to_indices: Dict[str, List[int]],
    total: int
) -> List[Optional[T]]:
    """Map results dictionary back to original positions using index mapping.
    
    Args:
        results: Dictionary mapping ID to result
        id_to_indices: Dictionary mapping ID to list of indices where it appears
        total: Total number of positions in output
        
    Returns:
        List of results aligned to original positions (None for missing)
    """
    output: List[Optional[T]] = [None] * total
    for id_key, indices in id_to_indices.items():
        result = results.get(id_key)
        for idx in indices:
            output[idx] = result
    return output


def unwrap_single_result(
    result: Union[T, List[T]],
    is_single: bool
) -> Union[T, List[T]]:
    """Unwrap a single result from a list if it was originally single.
    
    Args:
        result: Result that may be a single value or list
        is_single: Whether the original input was a single value
        
    Returns:
        Single value if is_single, otherwise list
    """
    if is_single:
        if isinstance(result, list):
            return result[0] if result else None
        return result
    return result if isinstance(result, list) else [result]


def flatten_and_map_ids(ids_per_query: List[List[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Flatten IDs from multiple queries and create mapping to original positions.
    
    Args:
        ids_per_query: List of ID lists, one per query
        
    Returns:
        Tuple of (flattened_unique_ids, id_to_query_indices)
    """
    all_ids: List[str] = []
    id_to_query_idx: Dict[str, List[int]] = {}
    
    for query_idx, ids in enumerate(ids_per_query):
        for id_val in ids:
            if id_val not in id_to_query_idx:
                id_to_query_idx[id_val] = []
                all_ids.append(id_val)
            id_to_query_idx[id_val].append(query_idx)
    
    return all_ids, id_to_query_idx


def build_result_map(
    results: Union[T, List[T]],
    all_ids: List[str]
) -> Dict[str, T]:
    """Build a dictionary mapping IDs to results.
    
    Args:
        results: Either a single result or list of results aligned with all_ids
        all_ids: List of IDs corresponding to results
        
    Returns:
        Dictionary mapping ID to result
    """
    result_map: Dict[str, T] = {}
    if isinstance(results, list):
        for id_val, result in zip(all_ids, results):
            if result is not None:
                result_map[id_val] = result
    elif results is not None:
        if all_ids:
            result_map[all_ids[0]] = results
    return result_map


def create_minimal_entity(qid: str) -> WikidataEntity:
    """Create a minimal WikidataEntity with just QID and URL.
    
    This is used during triple construction when full entity details
    will be enriched later.
    
    Args:
        qid: Wikidata QID
        
    Returns:
        WikidataEntity with minimal information
    """
    return WikidataEntity(
        qid=qid,
        label="",
        description="",
        aliases=[],
        url=f"https://www.wikidata.org/wiki/{qid}"
    )


def create_minimal_property(
    pid: str,
    api_wrapper: "CustomWikidataAPIWrapper"
) -> WikidataProperty:
    """Create a WikidataProperty with label from cache if available.
    
    Args:
        pid: Property ID
        api_wrapper: API wrapper to access property label cache
        
    Returns:
        WikidataProperty with label from cache or empty
    """
    label = ""
    description = None
    
    if pid in api_wrapper.wikidata_props_with_labels:
        prop_info = api_wrapper.wikidata_props_with_labels[pid]
        label = prop_info.get("label", "") or ""
        description = prop_info.get("description")
    
    return WikidataProperty(
        pid=pid,
        label=label,
        description=description
    )


def build_property_filter(properties: List[str]) -> str:
    """Build a SPARQL FILTER clause for properties.
    
    Args:
        properties: List of property IDs (e.g., ['P31', 'P27'])
        
    Returns:
        SPARQL FILTER clause string
    """
    if not properties:
        return "FILTER(STRSTARTS(STR(?prop), 'http://www.wikidata.org/prop/direct/'))"
    
    prop_filters = " || ".join([f'?prop = wdt:{prop}' for prop in properties])
    return f"FILTER({prop_filters})"


def build_property_values_clause(properties: List[str]) -> Tuple[str, Set[str]]:
    """Build a SPARQL VALUES clause for properties and return property set.
    
    Args:
        properties: List of property IDs
        
    Returns:
        Tuple of (VALUES clause string, set of property IDs)
    """
    if not properties:
        return "", set()
    
    prop_values = " ".join([f"<http://www.wikidata.org/prop/direct/{p}>" for p in properties])
    prop_values_clause = f"VALUES ?relation {{ {prop_values} }}"
    return prop_values_clause, set(properties)


def validate_and_normalize_ids(
    ids: List[str],
    id_type: str = "item"
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Validate and normalize a list of IDs, tracking their positions.
    
    Args:
        ids: List of ID strings (may be mixed case, have whitespace)
        id_type: Either "item" (QIDs) or "property" (PIDs)
        
    Returns:
        Tuple of (valid_normalized_ids, id_to_indices)
    """
    valid_ids: List[str] = []
    id_to_indices: Dict[str, List[int]] = {}
    
    validator = normalize_and_validate_qid if id_type != "property" else normalize_and_validate_pid
    
    for idx, id_val in enumerate(ids):
        if id_val and isinstance(id_val, str):
            normalized = validator(id_val)
            if normalized:
                valid_ids.append(normalized)
                if normalized not in id_to_indices:
                    id_to_indices[normalized] = []
                id_to_indices[normalized].append(idx)
    
    return valid_ids, id_to_indices

