"""Entity linking with real LLM and Wikidata."""

import pytest

pytest.importorskip("SPARQLWrapper")

from tests.conftest import requires_llm_credentials
from wemg.llm.client import LLMClient
from wemg.retrieval.entity_linking import link_entities_llm
from wemg.retrieval.wikidata import WikidataClient, WikidataEntity


@pytest.mark.requires_llm
@pytest.mark.requires_wikidata
@pytest.mark.asyncio
async def test_link_entities_llm_real(wemg_config):
    requires_llm_credentials(wemg_config)
    client = LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        max_retries=1,
        max_tokens=min(wemg_config.llm.generation.max_tokens, 8192),
        cache_config={"enabled": False},
    )
    wikidata_client = WikidataClient()
    try:
        entities, entity_dict, log = await link_entities_llm(
            client=client,
            text="Berlin is the capital of Germany.",
            wikidata_client=wikidata_client,
            top_k_entities=1,
        )
        assert isinstance(entities, (list, dict))
        assert isinstance(entity_dict, dict)
        assert isinstance(log, dict)
        if isinstance(entities, list) and entities:
            assert all(hasattr(e, "qid") for e in entities)
    finally:
        client.close()


@pytest.mark.requires_reranker
@pytest.mark.asyncio
async def test_link_entities_llm_reranks_multi_candidate(wemg_config, live_reranker, monkeypatch):
    """When Wikidata returns multiple hits, real reranker scores candidates vs NER description."""
    requires_llm_credentials(wemg_config)
    from wemg.llm.roles import Entity, NEROutput

    ner_out = NEROutput(
        entities=[
            Entity(
                id=None,
                name="Springfield",
                description="city in Illinois, USA",
            ),
        ]
    )

    async def fake_execute_role(*args, **kwargs):
        return [ner_out], {}

    monkeypatch.setattr("wemg.llm.roles.execute_role", fake_execute_role)

    class FakeWikidata:
        async def asearch_entities(self, queries, num_results=1, get_details=True):
            return [
                [
                    WikidataEntity(
                        qid="Q1",
                        label="Springfield",
                        description="city in Massachusetts",
                    ),
                    WikidataEntity(
                        qid="Q2",
                        label="Springfield",
                        description="city in Illinois",
                    ),
                ]
            ]

    entities, entity_dict, log = await link_entities_llm(
        client=None,
        text="ignored",
        wikidata_client=FakeWikidata(),
        top_k_entities=2,
        reranker=live_reranker,
    )
    assert entity_dict.get("Springfield") == "Q2"
    assert len(entities) == 1
    assert entities[0].qid == "Q2"
    assert isinstance(log, dict)
