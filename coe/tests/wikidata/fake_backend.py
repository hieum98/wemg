"""In-memory backend with a controllable mini-graph + test instrumentation.

Three knobs for tests:
  - data:   ``.add_entity`` / ``.add_property`` / ``.add_triple`` / ``.add_wikipedia``
  - faults: ``.inject_error(method, exc, times=1)``
  - timing: ``.inject_delay(method, seconds, times=1)``

Three observables:
  - ``.call_log``         : full list of ``BackendCall`` (method, args, kwargs, ts)
  - ``.calls(method)`` / ``.call_count(method)`` : convenience filters
  - ``.max_in_flight[method]``  : peak observed concurrency per method
  - return values served from the configured mini-graph
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import EntityRecord, PropertyRecord


@dataclass
class BackendCall:
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)


class FakeWikidataBackend:
    """In-memory backend implementing ``WikidataBackend``."""

    def __init__(self) -> None:
        # Data
        self._entities: dict[str, EntityRecord] = {}
        self._properties: dict[str, PropertyRecord] = {}
        self._outgoing: dict[str, list[tuple[str, str]]] = {}
        self._incoming: dict[str, list[tuple[str, str]]] = {}
        self._wikipedia: dict[str, str] = {}
        self._entity_search_index: dict[str, list[str]] = {}
        self._property_search_index: dict[str, list[str]] = {}

        # Instrumentation
        self.call_log: list[BackendCall] = []
        self._fail_queue: dict[str, list[BaseException]] = {}
        self._delay_queue: dict[str, list[float]] = {}
        self._in_flight: dict[str, int] = {}
        self.max_in_flight: dict[str, int] = {}

    # ---------------- population helpers ----------------

    def add_entity(
        self,
        qid: str,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        aliases: tuple[str, ...] = (),
        wikipedia_title: Optional[str] = None,
        search_terms: tuple[str, ...] = (),
    ) -> None:
        rec: EntityRecord = {
            "qid": qid,
            "label": label,
            "description": description,
            "aliases": list(aliases),
            "wikipedia_title": wikipedia_title,
            "wikipedia_url": (
                f"https://en.wikipedia.org/wiki/{wikipedia_title.replace(' ', '_')}"
                if wikipedia_title else None
            ),
        }
        self._entities[qid] = rec
        terms: list[str] = []
        if label:
            terms.append(label)
        terms.extend(search_terms)
        for term in terms:
            self._entity_search_index.setdefault(term.lower(), []).append(qid)

    def add_property(
        self,
        pid: str,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        search_terms: tuple[str, ...] = (),
    ) -> None:
        rec: PropertyRecord = {
            "pid": pid,
            "label": label,
            "description": description,
        }
        self._properties[pid] = rec
        terms: list[str] = []
        if label:
            terms.append(label)
        terms.extend(search_terms)
        for term in terms:
            self._property_search_index.setdefault(term.lower(), []).append(pid)

    def add_triple(self, subj: str, rel: str, obj: str) -> None:
        self._outgoing.setdefault(subj, []).append((rel, obj))
        self._incoming.setdefault(obj, []).append((rel, subj))

    def add_wikipedia(self, title: str, content: str) -> None:
        self._wikipedia[title] = content

    # ---------------- fault / delay injection ----------------

    def inject_error(self, method: str, exc: BaseException, *, times: int = 1) -> None:
        self._fail_queue.setdefault(method, []).extend([exc] * times)

    def inject_delay(self, method: str, seconds: float, *, times: int = 1) -> None:
        self._delay_queue.setdefault(method, []).extend([seconds] * times)

    # ---------------- query helpers ----------------

    def calls(self, method: str) -> list[BackendCall]:
        return [c for c in self.call_log if c.method == method]

    def call_count(self, method: str) -> int:
        return len(self.calls(method))

    # ---------------- internals ----------------

    def _log(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.call_log.append(BackendCall(method=method, args=args, kwargs=kwargs))

    def _enter(self, method: str) -> None:
        n = self._in_flight.get(method, 0) + 1
        self._in_flight[method] = n
        if n > self.max_in_flight.get(method, 0):
            self.max_in_flight[method] = n

    def _exit(self, method: str) -> None:
        self._in_flight[method] = max(0, self._in_flight.get(method, 0) - 1)

    async def _maybe_fail(self, method: str) -> None:
        q = self._fail_queue.get(method)
        if q:
            exc = q.pop(0)
            raise exc

    async def _maybe_delay(self, method: str) -> None:
        q = self._delay_queue.get(method)
        if q:
            secs = q.pop(0)
            if secs > 0:
                await asyncio.sleep(secs)

    # ---------------- backend protocol ----------------

    async def search_entities_text(self, query: str, *, limit: int) -> list[str]:
        self._enter("search_entities_text")
        self._log("search_entities_text", (query,), {"limit": limit})
        try:
            await self._maybe_fail("search_entities_text")
            await self._maybe_delay("search_entities_text")
            matches = list(self._entity_search_index.get(query.lower(), []))
            return matches[:limit]
        finally:
            self._exit("search_entities_text")

    async def get_entity_details(self, qids: list[str]) -> dict[str, EntityRecord]:
        self._enter("get_entity_details")
        self._log("get_entity_details", (list(qids),), {})
        try:
            await self._maybe_fail("get_entity_details")
            await self._maybe_delay("get_entity_details")
            return {q: self._entities[q] for q in qids if q in self._entities}
        finally:
            self._exit("get_entity_details")

    async def search_properties_text(self, query: str, *, limit: int) -> list[str]:
        self._enter("search_properties_text")
        self._log("search_properties_text", (query,), {"limit": limit})
        try:
            await self._maybe_fail("search_properties_text")
            await self._maybe_delay("search_properties_text")
            matches = list(self._property_search_index.get(query.lower(), []))
            return matches[:limit]
        finally:
            self._exit("search_properties_text")

    async def get_property_details(self, pids: list[str]) -> dict[str, PropertyRecord]:
        self._enter("get_property_details")
        self._log("get_property_details", (list(pids),), {})
        try:
            await self._maybe_fail("get_property_details")
            await self._maybe_delay("get_property_details")
            return {p: self._properties[p] for p in pids if p in self._properties}
        finally:
            self._exit("get_property_details")

    async def fetch_outgoing(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]:
        self._enter("fetch_outgoing")
        self._log("fetch_outgoing", (list(qids),), {})
        try:
            await self._maybe_fail("fetch_outgoing")
            await self._maybe_delay("fetch_outgoing")
            return {q: list(self._outgoing.get(q, [])) for q in qids}
        finally:
            self._exit("fetch_outgoing")

    async def fetch_incoming(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]:
        self._enter("fetch_incoming")
        self._log("fetch_incoming", (list(qids),), {})
        try:
            await self._maybe_fail("fetch_incoming")
            await self._maybe_delay("fetch_incoming")
            return {q: list(self._incoming.get(q, [])) for q in qids}
        finally:
            self._exit("fetch_incoming")

    async def get_wikipedia_contents(self, titles: list[str]) -> dict[str, Optional[str]]:
        self._enter("get_wikipedia_contents")
        self._log("get_wikipedia_contents", (list(titles),), {})
        try:
            await self._maybe_fail("get_wikipedia_contents")
            await self._maybe_delay("get_wikipedia_contents")
            return {t: self._wikipedia.get(t) for t in titles}
        finally:
            self._exit("get_wikipedia_contents")
