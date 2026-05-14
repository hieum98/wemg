"""Live-API smoke tests against the real Wikidata/Wikipedia endpoints.

Skipped by default; run with::

    pytest tests/wikidata/test_live_smoke.py -m requires_wikidata -v
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.requires_wikidata, pytest.mark.integration]


@pytest.fixture
def live_client():
    """Production WikidataClient configured for live use.

    Skips if the production backend / dependencies are not in place.
    """
    pytest.importorskip("SPARQLWrapper")
    try:
        from langgraph_coe.tools.wikidata_client import WikidataClient
        return WikidataClient()
    except TypeError as e:
        pytest.skip(
            f"WikidataClient requires explicit backend; production factory not yet "
            f"available ({e})"
        )
    except Exception as e:
        pytest.skip(f"Cannot construct WikidataClient for live use: {e}")


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
