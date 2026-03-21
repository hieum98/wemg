"""Tests for entity linking (link_entities_llm) using real LLM and Wikidata."""

import pytest

from conftest import get_llm_env, requires_llm
from wemg.retrieval.entity_linking import link_entities_llm
from wemg.retrieval.wikidata import WikidataClient


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_link_entities_llm_real():
    """Real API: link_entities_llm returns entities and entity_dict for text with entities."""
    requires_llm()
    api_key, url, model = get_llm_env()
    if not url:
        pytest.skip("LLM_URL not set")
    from wemg.llm.client import LLMClient
    client = LLMClient(
        model_name=model,
        url=url,
        api_key=api_key,
        max_retries=1,
        max_tokens=32768,
    )
    wikidata_client = WikidataClient()
    try:
        entities, entity_dict, log = await link_entities_llm(
            client=client,
            text="Berlin is the capital of Germany.",
            wikidata_client=wikidata_client,
            top_k_entities=1,
        )
        assert isinstance(entities, list)
        assert isinstance(entity_dict, dict)
        assert isinstance(log, dict)
        # May or may not find entities depending on NER and Wikidata
        if entities:
            assert all(hasattr(e, "qid") for e in entities)
    finally:
        client.close()
