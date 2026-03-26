"""Integration tests for batched Wikipedia content fetching."""

import pytest

pytest.importorskip("SPARQLWrapper")
pytest.importorskip("requests")

import wemg.retrieval.wikidata as wikidata_module
from wemg.retrieval.wikidata import WikidataClient, WikidataEntity

pytestmark = [pytest.mark.requires_wikidata, pytest.mark.integration]


def test_fetch_wikipedia_contents_batches_multiple_titles_into_one_real_api_call(monkeypatch):
    client = WikidataClient()
    monkeypatch.setattr(
        client,
        "_lookup_wikipedia_dump_contents",
        lambda urls: {url: None for url in urls},
    )

    calls = []
    original_get = wikidata_module.requests.Session.get

    def spy_get(self, url, **kwargs):
        if url == wikidata_module.WIKIPEDIA_API_URL:
            calls.append(
                {
                    "url": url,
                    "params": kwargs.get("params"),
                    "timeout": kwargs.get("timeout"),
                }
            )
        return original_get(self, url, **kwargs)

    monkeypatch.setattr(wikidata_module.requests.Session, "get", spy_get)

    entities = [
        WikidataEntity(
            qid="Q64",
            wikipedia_url="https://en.wikipedia.org/wiki/Berlin",
        ),
        WikidataEntity(
            qid="Q937",
            wikipedia_url="https://en.wikipedia.org/wiki/Albert_Einstein",
        ),
    ]

    client._fetch_wikipedia_contents_concurrent(entities, max_workers=1)

    assert len(calls) == 1
    assert calls[0]["url"] == wikidata_module.WIKIPEDIA_API_URL
    assert calls[0]["params"]["prop"] == "revisions"
    assert calls[0]["params"]["titles"] == "Berlin|Albert_Einstein"
    assert isinstance(entities[0].wikipedia_content, str)
    assert isinstance(entities[1].wikipedia_content, str)
    assert len(entities[0].wikipedia_content) > 10_000
    assert len(entities[1].wikipedia_content) > 10_000
    assert "Berlin" in entities[0].wikipedia_content
    assert "Einstein" in entities[1].wikipedia_content
