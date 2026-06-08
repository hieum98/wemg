"""Fixtures for wikidata tool tests."""

from __future__ import annotations

import pytest

from ._fixtures import VirtualClock, build_mini_graph
from .fake_backend import FakeWikidataBackend


def _import_client():
    """Import ``WikidataClient`` lazily so collection works even if the import
    chain pulls in heavy deps."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    return WikidataClient


@pytest.fixture
def fake_backend() -> FakeWikidataBackend:
    """Empty FakeWikidataBackend."""
    return FakeWikidataBackend()


@pytest.fixture
def mini_graph(fake_backend: FakeWikidataBackend) -> FakeWikidataBackend:
    """Pre-populated mini-graph with Berlin/Germany/Paris/France/etc."""
    return build_mini_graph(fake_backend)


@pytest.fixture
def client(mini_graph: FakeWikidataBackend):
    """WikidataClient wired to the mini-graph with effectively-disabled rate limits."""
    WikidataClient = _import_client()
    return WikidataClient(
        backend=mini_graph,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
    )


@pytest.fixture
def empty_client(fake_backend: FakeWikidataBackend):
    """WikidataClient with empty backend (no preloaded data)."""
    WikidataClient = _import_client()
    return WikidataClient(
        backend=fake_backend,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
    )


@pytest.fixture
def slow_client(mini_graph: FakeWikidataBackend):
    """WikidataClient at production-default rate limits (2/10 RPS)."""
    WikidataClient = _import_client()
    return WikidataClient(
        backend=mini_graph,
        max_sparql_rps=2.0,
        max_wikipedia_rps=10.0,
    )


@pytest.fixture
def mock_monotonic(monkeypatch):
    """Patch ``time.monotonic`` + ``asyncio.sleep`` on the wikidata_client module
    so rate-limit tests run in virtual time."""
    from langgraph_coe.tools import wikidata_client as _mod
    clock = VirtualClock()
    monkeypatch.setattr(_mod.time, "monotonic", clock.now, raising=False)
    monkeypatch.setattr(_mod.asyncio, "sleep", clock.async_sleep, raising=False)
    return clock


@pytest.fixture
def fake_redis():
    """In-process Redis emulator (decode_responses=False to match production)."""
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=False)


@pytest.fixture
def client_with_redis(mini_graph: FakeWikidataBackend, fake_redis):
    WikidataClient = _import_client()
    return WikidataClient(
        backend=mini_graph,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
        redis=fake_redis,
    )
