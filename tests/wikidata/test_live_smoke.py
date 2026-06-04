"""Live-API smoke tests against Wikidata (local QEndpoint SPARQL when configured).

Default SPARQL: ``LANGGRAPH_TEST_SPARQL_URL`` or tunneled
``http://127.0.0.1:30162/api/endpoint/sparql``. Omit / unreachable → public
``query.wikidata.org``.

Run::

    ssh -fN -L 30162:n0162:1234 t2
    pytest tests/wikidata/test_live_smoke.py -m requires_wikidata -v
"""

from __future__ import annotations

import os

import httpx
import pytest

SPARQL_URL = os.environ.get(
    "LANGGRAPH_TEST_SPARQL_URL",
    "http://127.0.0.1:30162/api/endpoint/sparql",
)


def _sparql_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                url,
                params={"query": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"},
                headers={"Accept": "application/sparql-results+json"},
            )
            return resp.status_code == 200
    except Exception:
        return False


_USE_LOCAL_SPARQL = _sparql_alive(SPARQL_URL)

pytestmark = [pytest.mark.requires_wikidata, pytest.mark.integration]


@pytest.fixture
def live_client():
    """WikidataClient: local QEndpoint SPARQL when reachable, else public endpoint."""
    from langgraph_coe.tools.wikidata_client import WikidataClient

    if _USE_LOCAL_SPARQL:
        return WikidataClient(
            sparql_endpoint=SPARQL_URL,
            max_sparql_rps=10.0,
            max_wikipedia_rps=10.0,
        )
    return WikidataClient()


async def test_live_link_berlin_top_1_is_q64(live_client):
    result = await live_client.link_entities("Berlin", top_k=1)
    assert result, "Berlin must resolve to something"
    assert result[0].qid == "Q64"


async def test_live_enrich_q64_has_wikipedia_content(live_client):
    result = await live_client.enrich_entities(["Q64"], get_details=True)
    assert len(result) == 1
    assert result[0].qid == "Q64"
    assert result[0].label is not None
    if result[0].wikipedia_content is not None:
        assert "Berlin" in result[0].wikipedia_content


async def test_live_khop_q64_k1_forward_nonempty(live_client):
    triples = await live_client.get_k_hop_triples(
        "Q64", k=1, bidirectional=False, enrich=False
    )
    assert isinstance(triples, list)
    assert len(triples) > 0


async def test_live_property_p36_is_capital(live_client):
    result = await live_client.search_properties("P36", top_k=1)
    assert result and result[0].pid == "P36"
    if result[0].label:
        assert "capital" in result[0].label.lower()
