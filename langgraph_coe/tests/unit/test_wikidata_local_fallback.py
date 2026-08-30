"""Local-SPARQL fallbacks for the two throttled public Wikidata API calls.

``wikidata.sparql_endpoint`` points at a local QLever instance, so k-hop triple
fetching is local and unthrottled. Entity **search** (``wbsearchentities``) and entity
**details** (``wbgetentities``) have no local equivalent in that configuration and still
go to the public API, where they are rate-limited hard.

Measured cost of that gap, from the ``lab_*`` evaluation runs: **3,805 name lookups lost
to HTTP 429** (2,160 distinct) across two runs, about 8 per question. Each loss means no
QID, so a plan intent cannot close and its resolved referent never becomes available to
ground a later retrieval query — which is the dominant reason unresolved intents were
measured to have retrieved facts but no binding.
"""

from __future__ import annotations

import asyncio

import pytest

from langgraph_coe.tools.wikidata_backend import (
    HTTPWikidataBackend,
    WikidataRateLimitError,
)


class _FakeBackend:
    """Public calls always throttle; local calls are recorded and answered."""

    def __init__(self, local_search=None, local_details=None):
        self.search_calls = 0
        self.details_calls = 0
        self.local_search_calls = 0
        self.local_details_calls = 0
        self._local_search = local_search or []
        self._local_details = local_details or {}

    async def search_entities_text(self, query, *, limit):
        self.search_calls += 1
        raise WikidataRateLimitError(retry_after=30.0)

    async def get_entity_details(self, qids):
        self.details_calls += 1
        raise WikidataRateLimitError(retry_after=30.0)

    async def search_entities_local(self, query, *, limit=1):
        self.local_search_calls += 1
        return list(self._local_search)

    async def get_entity_details_local(self, qids):
        self.local_details_calls += 1
        return dict(self._local_details)


def _client(backend):
    from langgraph_coe.tools.wikidata_client import WikidataClient

    c = WikidataClient(sparql_endpoint="http://localhost:9/unused")
    c._backend = backend
    return c


def test_a_throttled_entity_search_falls_back_to_the_local_endpoint():
    b = _FakeBackend(local_search=["Q180453"])
    c = _client(b)
    qids = asyncio.run(c._link_one("Dolly Parton", top_k=1))
    assert qids == ["Q180453"]
    assert b.search_calls >= 1, "the public API must still be tried first"
    assert b.local_search_calls == 1


def test_the_public_api_is_tried_before_the_local_endpoint():
    """Ordering is load-bearing, not incidental.

    The API ranks by relevance; the fallback ranks by statement count. They agree only
    where the label is unambiguous — measured 67% top-1 agreement over 30 real names, and
    the disagreements were all short ambiguous labels ("ABC", "State"). Local-first would
    therefore be *wrong* on roughly 10% of names, and a wrong QID is worse than no QID
    because it manufactures a false binding.
    """
    order = []

    class _Ordered(_FakeBackend):
        async def search_entities_text(self, query, *, limit):
            order.append("public")
            raise WikidataRateLimitError(retry_after=1.0)

        async def search_entities_local(self, query, *, limit=1):
            order.append("local")
            return ["Q1"]

    c = _client(_Ordered())
    asyncio.run(c._link_one("Some Name", top_k=1))
    assert order[0] == "public" and "local" in order


def test_throttled_entity_details_fall_back_so_labels_survive():
    """Without a label, ``entity_dict`` is a bare ``{"qid": ...}``.

    That silently disables ``_known_entity_labels`` — so the KG fan-out gate stops firing
    on entities already linked — and label-based binding resolution.
    """
    b = _FakeBackend(
        local_details={"Q180453": {"qid": "Q180453", "label": "Dolly Parton"}}
    )
    c = _client(b)
    asyncio.run(c._fetch_entities(["Q180453"]))
    assert b.details_calls >= 1
    assert b.local_details_calls == 1
    assert c._entities["Q180453"].get("label") == "Dolly Parton"


def test_a_backend_without_the_fallback_still_raises():
    """The fallback is opt-in by capability; a plain backend must not change behaviour."""

    class _NoFallback:
        async def search_entities_text(self, query, *, limit):
            raise WikidataRateLimitError(retry_after=1.0)

    c = _client(_NoFallback())
    with pytest.raises(Exception):
        asyncio.run(c._link_one("Anything", top_k=1))


# ── The dominance gate ────────────────────────────────────────────────────────
#
# Statement count is a prominence proxy. It reproduces ``wbsearchentities``' ranking on
# unambiguous labels and carries no signal on ambiguous ones, so the resolver declines
# rather than guesses. Measured over 30 real names, comparing top-1 against the API:
#
#   threshold   accepted   correct   wrong   precision   coverage
#          1x         23        20       3         87%        77%
#          4x         18        17       1         94%        60%
#        > 8x         15        15       0        100%        50%


def _backend_returning(rows):
    b = HTTPWikidataBackend(sparql_endpoint="http://localhost:9/unused")

    async def _fake(query):
        return rows

    b._sparql_query = _fake  # type: ignore[assignment]
    return b


def _row(qid, n):
    return {
        "e": {"value": f"http://www.wikidata.org/entity/{qid}"},
        "n": {"value": str(n)},
    }


def test_a_dominant_label_match_is_accepted():
    # "Dolly Parton": 462 statements against 18 for a Warhol painting of her.
    b = _backend_returning([_row("Q180453", 462), _row("Q104025729", 18)])
    assert asyncio.run(b.search_entities_local("Dolly Parton")) == ["Q180453"]


def test_a_sole_label_match_is_accepted():
    b = _backend_returning([_row("Q130796", 83)])
    assert asyncio.run(b.search_entities_local("Molotov-Ribbentrop Pact")) == ["Q130796"]


@pytest.mark.parametrize(
    "n1,n2,name",
    [
        (74, 58, "ABC"),             # 1.3x — API says Q169889, this ranks Q287078
        (21, 21, "State"),           # 1.0x — a coin flip
        (167, 34, "The Book Thief"),  # 4.9x — still below the gate, still wrong
    ],
)
def test_an_ambiguous_label_is_declined_rather_than_guessed(n1, n2, name):
    b = _backend_returning([_row("Q1", n1), _row("Q2", n2)])
    assert asyncio.run(b.search_entities_local(name)) == [], (
        "every measured disagreement with the public API sat below the gate; "
        "a wrong QID manufactures a false binding"
    )


def test_the_gate_threshold_is_the_measured_one():
    from langgraph_coe.tools.wikidata_backend import HTTPWikidataBackend as B

    assert B._LOCAL_DOMINANCE == 8.0, (
        "8x was the lowest threshold with zero errors on the 30-name sample; "
        "15/15 is consistent with a true precision near 80%, so do not lower it "
        "without re-measuring"
    )


def test_a_dead_local_endpoint_returns_empty_rather_than_raising():
    """A fallback that fails must not take the hop down with it."""
    b = HTTPWikidataBackend(sparql_endpoint="http://localhost:9/unused")

    async def _boom(query):
        raise RuntimeError("endpoint down")

    b._sparql_query = _boom  # type: ignore[assignment]
    assert asyncio.run(b.search_entities_local("Anything")) == []
    assert asyncio.run(b.get_entity_details_local(["Q1"])) == {}


def test_details_lookup_needs_no_gate_and_skips_malformed_qids():
    rows = [
        {
            "e": {"value": "http://www.wikidata.org/entity/Q180453"},
            "l": {"value": "Dolly Parton"},
            "d": {"value": "American singer, songwriter and actress"},
        }
    ]
    b = _backend_returning(rows)
    out = asyncio.run(b.get_entity_details_local(["Q180453", "not-a-qid", ""]))
    assert set(out) == {"Q180453"}
    assert out["Q180453"]["label"] == "Dolly Parton"
    assert out["Q180453"]["description"].startswith("American singer")
    # A truthy dump carries no aliases or Wikipedia titles — partial by construction.
    assert out["Q180453"]["aliases"] == []
