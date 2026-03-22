"""Entity linking: text entities -> Wikidata QIDs."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from wemg.llm.roles import Entity as NEREntity
from wemg.retrieval.reranker import Reranker
from wemg.retrieval.wikidata import WikidataEntity

logger = logging.getLogger(__name__)


def _choose_wikidata_for_mention(
    ner_entity: NEREntity,
    results: List[WikidataEntity],
    reranker: Optional[Reranker],
) -> Optional[WikidataEntity]:
    """Pick one Wikidata entity for a NER mention; optionally rerank by NER description."""
    if not results:
        return None
    first = results[0]
    if not isinstance(first, WikidataEntity) or not first.qid:
        return None
    if len(results) == 1 or reranker is None:
        return first
    desc = (ner_entity.description or "").strip()
    if not desc:
        return first
    documents = [str(e) for e in results]
    idx = reranker.best_index(desc, documents)
    if idx < 0 or idx >= len(results):
        idx = 0
    return results[idx]


async def link_entities_llm(
    client,  # LLMClient
    text: str,
    wikidata_client,  # WikidataClient
    top_k_entities: int = 1,
    interaction_memory=None,
    known_entities: Optional[List[Any]] = None,
    reranker: Optional[Reranker] = None,
) -> Tuple[Dict, Dict, Dict]:
    """Link entities in text to Wikidata using LLM NER.

    1. Run NER role on text to extract Entity objects
    2. For each entity with unknown QID, search Wikidata by entity.name (up to top_k_entities hits)
    3. If multiple hits and ``reranker`` is set and the NER entity has a non-empty description,
       rerank candidates so the best match aligns with that description; otherwise use the first hit.
    4. Return:
     - wikidata_entities: List[WikidataEntity], one resolved entity per unknown mention
     - entity_dict: Dict[str, str] {name: qid}, all entities in the text with their Wikidata QIDs
     - log_data: Dict: {NER: [(input_str, raw_output), ...]} for logging the NER role calls
    """
    from wemg.llm.roles import NER, NERInput, execute_role

    ner_input = NERInput(text=text, known_entities=known_entities)
    responses, log = await execute_role(
        client=client,
        role=NER,
        input_data=ner_input,
        interaction_memory=interaction_memory,
        n=1,
    )

    ner_output = responses[0] if responses else None
    if not ner_output or not hasattr(ner_output, 'entities') or not ner_output.entities:
        return [], {}, log

    entities_to_link = [e for e in ner_output.entities if e.id is None]
    entity_names = [e.name for e in entities_to_link]
    retrieval_results: Union[List[WikidataEntity], List[List[WikidataEntity]]] = await wikidata_client.asearch_entities(
        entity_names, num_results=top_k_entities, get_details=True
    )

    entity_dict: Dict[str, str] = {e.name: e.id for e in ner_output.entities if e.id is not None and e.id.startswith('Q')}
    wikidata_entities: List[WikidataEntity] = []

    if retrieval_results and isinstance(retrieval_results[0], list):
        for ner_entity, results in zip(entities_to_link, retrieval_results):
            chosen = _choose_wikidata_for_mention(ner_entity, results, reranker)
            if chosen:
                entity_dict[ner_entity.name] = chosen.qid
                wikidata_entities.append(chosen)

    unique = list(set(wikidata_entities))  # deduplicate
    return unique, entity_dict, log


async def link_entities_azure(
    text: str,
    wikidata_client,
    endpoint: Optional[str] = None,
    key: Optional[str] = None,
) -> Tuple[List[WikidataEntity], Dict, Dict]:
    """Link entities using Azure Text Analytics.

    1. Call Azure recognize_linked_entities
    2. Resolve Wikipedia URLs to Wikidata QIDs
    3. Fetch entity details from Wikidata by QID
    4. Return (wikidata_entities, entity_dict, log_data)
    """
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        logger.warning("Azure Text Analytics SDK not installed")
        return [], {}, {}

    import os
    import requests

    endpoint = endpoint or os.getenv("AZURE_TEXTANALYTICS_ENDPOINT")
    key = key or os.getenv("AZURE_TEXTANALYTICS_KEY")

    if not endpoint or not key:
        logger.warning("Azure endpoint/key not configured")
        return [], {}, {}

    client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    try:
        result = client.recognize_linked_entities([text])
        doc = result[0]
        if doc.is_error:
            logger.error(f"Azure entity linking error: {doc.error}")
            return [], {}, {}
    except Exception as e:
        logger.error(f"Azure entity linking failed: {e}")
        return [], {}, {}

    name_qid_pairs = []
    for entity in doc.entities:
        if entity.data_source == "Wikipedia":
            qid = _wikipedia_url_to_qid(entity.url)
            if qid:
                name_qid_pairs.append((entity.name, qid))

    if not name_qid_pairs:
        return [], {}, {}

    qids = [qid for _, qid in name_qid_pairs]
    results = await wikidata_client.asearch_entities(qids, num_results=1, is_qids=True, get_details=True)

    from wemg.llm.roles import Entity as RoleEntity
    entity_dict = {}
    all_entities = []

    if isinstance(results, list):
        flat = []
        if results and isinstance(results[0], list):
            flat = [e for sub in results for e in sub]
        else:
            flat = results

        for name, qid in name_qid_pairs:
            for e in flat:
                if hasattr(e, 'qid') and e.qid == qid:
                    entity_dict[name] = e.qid
                    break
        all_entities: List[WikidataEntity] = [e for e in flat]
    all_entities = list(set(all_entities))  # deduplicate
    return all_entities, entity_dict, {}


def _wikipedia_url_to_qid(url: str) -> Optional[str]:
    """Resolve Wikipedia URL to Wikidata QID via Wikipedia info page."""
    if not url:
        return None
    try:
        import requests
        title = url.split("/wiki/")[-1]
        api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageprops&format=json"
        resp = requests.get(api_url, timeout=5)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            qid = page_data.get("pageprops", {}).get("wikibase_item")
            if qid:
                return qid
    except Exception as e:
        logger.warning(f"Failed to resolve Wikipedia URL to QID: {url}: {e}")
    return None
