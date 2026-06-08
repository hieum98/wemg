"""Test-side WikidataBackend protocol and TypedDicts.

Re-exports the production types so that test fakes and production code share
the same ``WikidataRateLimitError`` class (otherwise the client's ``except
WikidataRateLimitError`` would never catch errors raised by ``FakeWikidataBackend``).
"""

from __future__ import annotations

from langgraph_coe.tools.wikidata_backend import (
    EntityRecord,
    PropertyRecord,
    WikidataBackend,
    WikidataRateLimitError,
)

__all__ = [
    "EntityRecord",
    "PropertyRecord",
    "WikidataBackend",
    "WikidataRateLimitError",
]
