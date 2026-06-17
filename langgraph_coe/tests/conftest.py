"""Shared fixtures for langgraph_coe tests.

Selectively reuses the FakeWikidataBackend / mini-graph infrastructure already
in ``coe/tests/wikidata/`` to avoid duplication. New, langgraph-specific fixtures
live here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Make ``coe.tests.wikidata.*`` importable when this conftest is loaded as part
# of ``langgraph_coe/tests``. pyproject testpaths includes both trees, but the
# ``coe`` package only appears on sys.path once pytest discovers it; reading
# from it directly inside this conftest needs the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config) -> None:
    """Register the custom markers these suites use.

    Mirrors the ``coe/tests/conftest.py`` so the markers are also known when the
    suite is run on its own (``pytest langgraph_coe/tests``) — that sibling
    conftest is not loaded in that case, which otherwise raises
    ``PytestUnknownMarkWarning``.
    """
    markers = [
        ("requires_wikidata", "needs live Wikidata/Wikipedia APIs"),
        ("integration", "real external services"),
        ("slow_integration", "long-running real-system flows"),
    ]
    for name, desc in markers:
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture
def config():
    """Loaded LangGraphCoeConfig (defaults + config.yaml merge)."""
    from langgraph_coe.config import LangGraphCoeConfig

    return LangGraphCoeConfig.from_yaml()


@pytest.fixture
def fake_redis():
    """Synchronous in-process Redis emulator. Matches ``coe/tests/wikidata/conftest.py``."""
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=False)


@pytest.fixture
def fake_redis_factory():
    """Build named ``FakeStrictRedis`` instances (used to verify separate db indices)."""
    fakeredis = pytest.importorskip("fakeredis")

    def _build(**kwargs: Any):
        return fakeredis.FakeStrictRedis(decode_responses=False, **kwargs)

    return _build


@pytest.fixture
def mini_backend():
    """Pre-populated ``FakeWikidataBackend`` (Berlin/Germany/Paris/France/...)."""
    from coe.tests.wikidata._fixtures import build_mini_graph
    from coe.tests.wikidata.fake_backend import FakeWikidataBackend

    return build_mini_graph(FakeWikidataBackend())


@pytest.fixture
def empty_backend():
    from coe.tests.wikidata.fake_backend import FakeWikidataBackend

    return FakeWikidataBackend()


@pytest.fixture(autouse=True)
def _reset_sessions():
    """Reset per-question ContextVars between tests so cross-test bleed cannot occur.

    ``reset_wikidata_session`` exists today. ``reset_web_research_session`` is a
    addition — import lazily and skip if absent so collection
    still works pre-implementation.
    """
    from langgraph_coe.tools.wikidata import reset_wikidata_session

    reset_wikidata_session()
    try:
        from langgraph_coe.tools.web import reset_web_research_session
    except ImportError:
        reset_web_research_session = None # type: ignore[assignment]
    if reset_web_research_session is not None:
        reset_web_research_session()
    yield


@pytest.fixture
def init_wikidata_tools(config, mini_backend, monkeypatch):
    """Wire the ``langgraph_coe.tools.wikidata`` module to a fake backend.

    Bypasses ``init_wikidata`` to avoid building a real ``WikidataClient`` and
    instead injects a wrapper exposing the methods the @tool functions need.
    Yields a tuple ``(client_mock, backend)`` so tests can drive both.
    """
    from langgraph_coe.tools import wikidata as wd_mod
    from langgraph_coe.tools.wikidata_client import WikidataClient

    client = WikidataClient(
        backend=mini_backend,
        max_sparql_rps=1000,
        max_wikipedia_rps=1000,
    )
    monkeypatch.setattr(wd_mod, "_wikidata_client", client, raising=False)
    monkeypatch.setattr(wd_mod, "_wikidata_config", config.wikidata, raising=False)
    wd_mod.entity_cache.clear()
    yield client, mini_backend
