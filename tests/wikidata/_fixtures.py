"""Canonical mini-graph + VirtualClock helper for wikidata tests."""

from __future__ import annotations

import asyncio

from .fake_backend import FakeWikidataBackend

# Capture the real asyncio.sleep at import time so VirtualClock can yield to
# the event loop even when ``asyncio.sleep`` itself is monkey-patched in tests.
_REAL_ASYNCIO_SLEEP = asyncio.sleep


class VirtualClock:
    """Replaces ``time.monotonic`` + ``asyncio.sleep`` for deterministic timing.

    Tests inject this via the ``mock_monotonic`` fixture. ``asyncio.sleep``
    becomes an instantaneous time-advance, so the suite runs in milliseconds
    while still exercising real async scheduling.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._t = start
        self.sleep_log: list[float] = []

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep_log.append(seconds)
        if seconds > 0:
            self._t += seconds
        # Yield to the loop using the REAL asyncio.sleep, not the patched one.
        await _REAL_ASYNCIO_SLEEP(0)


# Stable QIDs / PIDs used across the suite.
QID_BERLIN = "Q64"
QID_GERMANY = "Q183"
QID_EUROPE = "Q46"
QID_BRANDENBURG = "Q1208"
QID_PARIS = "Q90"
QID_FRANCE = "Q142"
QID_MERKEL = "Q567"
QID_HAMBURG = "Q1055"
QID_CAPITAL_CITY_CLASS = "Q5119"

PID_CAPITAL = "P36"
PID_COUNTRY = "P17"
PID_CONTINENT = "P30"
PID_INSTANCE_OF = "P31"
PID_CAPITAL_OF = "P1376"
PID_HEAD_OF_GOV = "P6"
PID_LOCATED_IN_ADMIN = "P131"


def build_mini_graph(backend: FakeWikidataBackend) -> FakeWikidataBackend:
    """Populate a small, predictable Wikidata-like graph for tests."""

    # Entities
    backend.add_entity(
        QID_BERLIN, label="Berlin", description="capital of Germany",
        aliases=("Berlin City",), wikipedia_title="Berlin",
        search_terms=("Berlin",),
    )
    backend.add_entity(
        QID_GERMANY, label="Germany", description="country in Central Europe",
        aliases=("Federal Republic of Germany", "Deutschland"),
        wikipedia_title="Germany", search_terms=("Germany",),
    )
    backend.add_entity(
        QID_EUROPE, label="Europe", description="continent",
        wikipedia_title="Europe", search_terms=("Europe",),
    )
    backend.add_entity(
        QID_BRANDENBURG, label="Brandenburg",
        description="state of Germany",
        wikipedia_title="Brandenburg", search_terms=("Brandenburg",),
    )
    backend.add_entity(
        QID_PARIS, label="Paris", description="capital of France",
        wikipedia_title="Paris", search_terms=("Paris",),
    )
    backend.add_entity(
        QID_FRANCE, label="France", description="country in Western Europe",
        wikipedia_title="France", search_terms=("France",),
    )
    backend.add_entity(
        QID_MERKEL, label="Angela Merkel",
        description="Chancellor of Germany 2005-2021",
        wikipedia_title="Angela Merkel",
        search_terms=("Merkel", "Angela Merkel"),
    )
    backend.add_entity(
        QID_HAMBURG, label="Hamburg", description="city in Germany",
        wikipedia_title="Hamburg", search_terms=("Hamburg",),
    )

    # Properties
    backend.add_property(
        PID_CAPITAL, label="capital",
        description="seat of government",
        search_terms=("capital",),
    )
    backend.add_property(
        PID_COUNTRY, label="country",
        description="sovereign state",
        search_terms=("country",),
    )
    backend.add_property(
        PID_CONTINENT, label="continent",
        description="continent of which the subject is a part",
        search_terms=("continent",),
    )
    backend.add_property(
        PID_INSTANCE_OF, label="instance of",
        description="that class of which this subject is a particular example",
        search_terms=("instance of",),
    )
    backend.add_property(
        PID_CAPITAL_OF, label="capital of",
        description="country that has this city as capital",
        search_terms=("capital of",),
    )
    backend.add_property(
        PID_HEAD_OF_GOV, label="head of government",
        description="head of the executive power",
        search_terms=("head of government",),
    )
    backend.add_property(
        PID_LOCATED_IN_ADMIN,
        label="located in the administrative territorial entity",
        description="the item is located in",
        search_terms=("located in",),
    )

    # Triples (outgoing direction; backend auto-populates incoming)
    backend.add_triple(QID_BERLIN, PID_CAPITAL_OF, QID_GERMANY)
    backend.add_triple(QID_BERLIN, PID_COUNTRY, QID_GERMANY)
    backend.add_triple(QID_BERLIN, PID_INSTANCE_OF, QID_CAPITAL_CITY_CLASS)
    backend.add_triple(QID_GERMANY, PID_CAPITAL, QID_BERLIN)
    backend.add_triple(QID_GERMANY, PID_CONTINENT, QID_EUROPE)
    backend.add_triple(QID_GERMANY, PID_HEAD_OF_GOV, QID_MERKEL)
    backend.add_triple(QID_BRANDENBURG, PID_COUNTRY, QID_GERMANY)
    backend.add_triple(QID_HAMBURG, PID_COUNTRY, QID_GERMANY)
    backend.add_triple(QID_PARIS, PID_CAPITAL_OF, QID_FRANCE)
    backend.add_triple(QID_FRANCE, PID_CAPITAL, QID_PARIS)
    backend.add_triple(QID_FRANCE, PID_CONTINENT, QID_EUROPE)

    # Wikipedia content
    backend.add_wikipedia(
        "Berlin",
        "Berlin is the capital and largest city of Germany, both by area and by population.",
    )
    backend.add_wikipedia(
        "Germany",
        "Germany, officially the Federal Republic of Germany, is a country in Central Europe.",
    )
    backend.add_wikipedia("Europe", "Europe is a continent located entirely in the Northern Hemisphere.")
    backend.add_wikipedia("Brandenburg", "Brandenburg is a state in northeastern Germany.")
    backend.add_wikipedia("Paris", "Paris is the capital and most populous city of France.")
    backend.add_wikipedia("France", "France is a country in Western Europe.")
    backend.add_wikipedia("Angela Merkel", "Angela Dorothea Merkel is a German former politician.")
    backend.add_wikipedia("Hamburg", "Hamburg is the second-largest city in Germany.")

    return backend
