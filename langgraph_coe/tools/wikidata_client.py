"""Async Wikidata client.

Owns L1 LRU + optional L2 Redis caches, per-endpoint rate limiters
(SPARQL vs. MediaWiki), a concurrency-bounding semaphore, single-flight
coalescing per cache key, and k-hop traversal. The wire layer is plugged
in via a ``WikidataBackend`` (see ``wikidata_backend``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

import pydantic

from .wikidata_backend import (
    EntityRecord,
    HTTPWikidataBackend,
    PropertyRecord,
    WikidataBackend,
    WikidataRateLimitError,
)
from .wikidata_properties import DEFAULT_PROPERTIES, PROPERTY_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants and validators
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
MAX_ENTITIES_PER_HOP = 500

_QID_RE = re.compile(r"^Q[A-Za-z0-9_]+$")
_PID_RE = re.compile(r"^P[A-Za-z0-9_]+$")


def _is_valid_qid(s: Any) -> bool:
    return isinstance(s, str) and bool(_QID_RE.match(s.strip().upper()))


def _is_valid_pid(s: Any) -> bool:
    return isinstance(s, str) and bool(_PID_RE.match(s.strip().upper()))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class WikidataEntity(pydantic.BaseModel):
    qid: str
    label: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = pydantic.Field(default_factory=list)
    url: Optional[str] = None
    wikipedia_url: Optional[str] = None
    wikipedia_content: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.qid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WikidataEntity) and self.qid == other.qid

    def __str__(self) -> str:
        base = self.label or self.qid
        return f"{base} ({self.description})" if self.description else base

    def to_context(self, include_wiki_page: bool = False) -> str:
        parts = [self.label or self.qid]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.aliases:
            parts.append(f"Also known as: {', '.join(self.aliases)}")
        if include_wiki_page and self.wikipedia_content:
            parts.append(f"Wikipedia: {self.wikipedia_content[:500]}")
        return "\n".join(parts)


class WikidataProperty(pydantic.BaseModel):
    pid: str
    label: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.pid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WikidataProperty) and self.pid == other.pid

    def __str__(self) -> str:
        return (self.label or self.description or self.pid).strip() or self.pid


class WikiTriple(pydantic.BaseModel):
    subject: Any
    relation: Any
    object: Any

    @pydantic.model_validator(mode="before")
    @classmethod
    def _reconstruct_nested(cls, data: Any) -> Any:
        # Allow Redis-deserialized dicts to round-trip back into pydantic models.
        if not isinstance(data, dict):
            return data
        if isinstance(data.get("subject"), dict):
            data["subject"] = WikidataEntity.model_validate(data["subject"])
        if isinstance(data.get("relation"), dict):
            data["relation"] = WikidataProperty.model_validate(data["relation"])
        if isinstance(data.get("object"), dict):
            try:
                data["object"] = WikidataEntity.model_validate(data["object"])
            except Exception:
                pass
        return data

    def _sig(self) -> Tuple[str, str, str]:
        s = self.subject.qid if hasattr(self.subject, "qid") else str(self.subject)
        r = self.relation.pid if hasattr(self.relation, "pid") else str(self.relation)
        o = self.object.qid if hasattr(self.object, "qid") else str(self.object)
        return (s, r, o)

    def __hash__(self) -> int:
        return hash(self._sig())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WikiTriple) and self._sig() == other._sig()

    def __str__(self) -> str:
        return f"Subject: {self.subject} - Relation: {self.relation} - Object: {self.object}"


# ---------------------------------------------------------------------------
# LRU + async rate limiter
# ---------------------------------------------------------------------------


class _LRU:
    """Minimal LRU. Single-threaded asyncio → no lock needed."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._d: "OrderedDict[Any, Any]" = OrderedDict()

    def __contains__(self, key: Any) -> bool:
        return key in self._d

    def __getitem__(self, key: Any) -> Any:
        self._d.move_to_end(key)
        return self._d[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self._capacity:
            self._d.popitem(last=False)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return default

    def __len__(self) -> int:
        return len(self._d)


class _AsyncRateLimiter:
    """Min-interval limiter: one acquirer at a time, spaced by ``1/rps``."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / max(0.0001, rps)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            wait = self._last + self._interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# ---------------------------------------------------------------------------
# WikidataClient
# ---------------------------------------------------------------------------


class WikidataClient:
    """Async Wikidata orchestrator.

    Public methods:
      - ``link_entities(names, top_k=1)``
      - ``enrich_entities(qids, get_details=False)``
      - ``search_properties(query, top_k=1)``
      - ``get_k_hop_triples(qids, k=1, bidirectional=True, enrich=True)``
    """

    def __init__(
        self,
        *,
        backend: Optional[WikidataBackend] = None,
        max_sparql_rps: float = 2.0,
        max_wikipedia_rps: float = 10.0,
        lru_capacity: int = 5000,
        redis: Optional[Any] = None,
        redis_ttl_seconds: int = 86400,
        concurrency_limit: int = 10,
    ) -> None:
        self._backend: WikidataBackend = backend if backend is not None else HTTPWikidataBackend()
        self._sparql_limiter = _AsyncRateLimiter(max_sparql_rps)
        self._wiki_limiter = _AsyncRateLimiter(max_wikipedia_rps)
        self._semaphore = asyncio.Semaphore(max(1, concurrency_limit))

        self._entities: _LRU = _LRU(lru_capacity)          # qid -> EntityRecord
        self._properties: _LRU = _LRU(lru_capacity)        # pid -> PropertyRecord
        self._outgoing: _LRU = _LRU(lru_capacity)          # qid -> list[(pid, obj)]
        self._incoming: _LRU = _LRU(lru_capacity)          # qid -> list[(pid, subj)]
        self._wiki: _LRU = _LRU(lru_capacity)              # title -> str | None
        self._entity_search: _LRU = _LRU(lru_capacity)     # (text, top_k) -> list[qid]
        self._property_search: _LRU = _LRU(lru_capacity)   # (text, top_k) -> list[pid]

        self._redis = redis
        self._redis_ttl = redis_ttl_seconds
        self._inflight: Dict[str, asyncio.Future] = {}

    # ==========================================================
    # Public API
    # ==========================================================

    async def link_entities(
        self,
        names: Union[str, List[str]],
        *,
        top_k: int = 1,
    ) -> Union[List[WikidataEntity], List[List[WikidataEntity]]]:
        single = isinstance(names, str)
        name_list = [names] if single else list(names)
        if not name_list:
            return []

        per_name = await asyncio.gather(*[self._link_one(n, top_k=top_k) for n in name_list])
        all_qids = list(dict.fromkeys(q for cs in per_name for q in cs))
        if all_qids:
            await self._fetch_entities(all_qids)

        out = [[self._make_entity(q) for q in cs] for cs in per_name]
        return out[0] if single else out

    async def enrich_entities(
        self,
        qids: Union[str, List[str]],
        *,
        get_details: bool = False,
    ) -> List[WikidataEntity]:
        qid_list = [qids] if isinstance(qids, str) else list(qids)
        qid_list = [q for q in qid_list if q]
        if not qid_list:
            return []

        await self._fetch_entities(qid_list)

        if get_details:
            titles: List[str] = []
            seen: Set[str] = set()
            for q in qid_list:
                rec = self._entities.get(q)
                title = rec.get("wikipedia_title") if rec else None
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
            if titles:
                await self._fetch_wikipedia(titles)

        return [self._make_entity(q, with_wiki=get_details) for q in qid_list]

    async def search_properties(
        self,
        query: Union[str, List[str]],
        *,
        top_k: int = 1,
    ) -> Union[List[WikidataProperty], List[List[WikidataProperty]]]:
        single = isinstance(query, str)
        q_list = [query] if single else list(query)
        if not q_list:
            return []

        per_q = await asyncio.gather(*[self._search_one_property(q, top_k=top_k) for q in q_list])
        all_pids = list(dict.fromkeys(p for ps in per_q for p in ps))
        if all_pids:
            await self._fetch_properties(all_pids)

        out: List[List[WikidataProperty]] = []
        for ps in per_q:
            props = [self._make_property(p) for p in ps]
            # Drop uninformative entries where label is missing or equals the PID.
            out.append([p for p in props if p.label and p.label != p.pid])
        return out[0] if single else out

    async def get_k_hop_triples(
        self,
        qids: Union[str, List[str]],
        *,
        k: int = 1,
        bidirectional: bool = True,
        enrich: bool = True,
    ) -> Union[List[WikiTriple], List[List[WikiTriple]]]:
        single = isinstance(qids, str)
        seeds = [qids] if single else list(qids)
        if not seeds:
            return []

        results: Dict[str, List[Tuple[str, str, str]]] = {s: [] for s in seeds}
        visited: Dict[str, Set[str]] = {
            s: ({s.strip().upper()} if _is_valid_qid(s) else set()) for s in seeds
        }
        frontier: Dict[str, Set[str]] = {s: set(visited[s]) for s in seeds}

        for _ in range(max(0, k)):
            needed = self._aggregate_frontier(seeds, frontier)
            if not needed:
                break
            outgoing = await self._fetch_outgoing(needed)
            incoming = await self._fetch_incoming(needed) if bidirectional else {}
            frontier = self._expand_frontier(
                seeds, frontier, visited, results, outgoing, incoming, bidirectional
            )

        if enrich:
            await self._enrich_triple_endpoints(seeds, results)

        triples = {s: [self._make_triple(sig, enrich) for sig in results[s]] for s in seeds}
        return triples[seeds[0]] if single else [triples[s] for s in seeds]

    # ==========================================================
    # Link / property-search per-input
    # ==========================================================

    async def _link_one(self, name: str, *, top_k: int) -> List[str]:
        if not isinstance(name, str) or not name.strip():
            return []
        n = name.strip()
        if _is_valid_qid(n):
            return [n.upper()]

        key = ("e", n.lower(), top_k)
        cached = self._entity_search.get(key)
        if cached is not None:
            return list(cached)

        qids = await self._coalesce(
            f"search_e:{n.lower()}:{top_k}",
            lambda: self._call_with_retry(
                self._wiki_limiter, self._backend.search_entities_text, n, limit=top_k
            ),
        )
        self._entity_search[key] = list(qids)
        return list(qids)

    async def _search_one_property(self, query: str, *, top_k: int) -> List[str]:
        if not isinstance(query, str) or not query.strip():
            return []
        q = query.strip()
        if _is_valid_pid(q):
            return [q.upper()]

        key = ("p", q.lower(), top_k)
        cached = self._property_search.get(key)
        if cached is not None:
            return list(cached)

        pids = await self._coalesce(
            f"search_p:{q.lower()}:{top_k}",
            lambda: self._call_with_retry(
                self._wiki_limiter, self._backend.search_properties_text, q, limit=top_k
            ),
        )
        self._property_search[key] = list(pids)
        return list(pids)

    # ==========================================================
    # K-hop helpers
    # ==========================================================

    @staticmethod
    def _aggregate_frontier(
        seeds: List[str], frontier: Dict[str, Set[str]]
    ) -> List[str]:
        """Union frontiers across seeds, dedup, cap at MAX_ENTITIES_PER_HOP."""
        out: List[str] = []
        seen: Set[str] = set()
        for s in seeds:
            for q in frontier[s]:
                if q in seen:
                    continue
                seen.add(q)
                out.append(q)
                if len(out) >= MAX_ENTITIES_PER_HOP:
                    return out
        return out

    @staticmethod
    def _expand_frontier(
        seeds: List[str],
        frontier: Dict[str, Set[str]],
        visited: Dict[str, Set[str]],
        results: Dict[str, List[Tuple[str, str, str]]],
        outgoing: Dict[str, List[Tuple[str, str]]],
        incoming: Dict[str, List[Tuple[str, str]]],
        bidirectional: bool,
    ) -> Dict[str, Set[str]]:
        """Append new triples to ``results`` and return the next-hop frontier."""
        new_frontier: Dict[str, Set[str]] = {s: set() for s in seeds}
        for s in seeds:
            seen = set(results[s])
            for src in frontier[s]:
                for (rel, obj) in outgoing.get(src, []):
                    sig = (src, rel, obj)
                    if sig in seen:
                        continue
                    results[s].append(sig)
                    seen.add(sig)
                    if _is_valid_qid(obj) and obj not in visited[s]:
                        visited[s].add(obj)
                        new_frontier[s].add(obj)
                if bidirectional:
                    for (rel, subj) in incoming.get(src, []):
                        sig = (subj, rel, src)
                        if sig in seen:
                            continue
                        results[s].append(sig)
                        seen.add(sig)
                        if _is_valid_qid(subj) and subj not in visited[s]:
                            visited[s].add(subj)
                            new_frontier[s].add(subj)
        return new_frontier

    async def _enrich_triple_endpoints(
        self,
        seeds: List[str],
        results: Dict[str, List[Tuple[str, str, str]]],
    ) -> None:
        qids: Set[str] = set()
        pids: Set[str] = set()
        for s in seeds:
            for (subj, rel, obj) in results[s]:
                if _is_valid_qid(subj):
                    qids.add(subj)
                if _is_valid_qid(obj):
                    qids.add(obj)
                if _is_valid_pid(rel):
                    pids.add(rel)
        if qids:
            await self._fetch_entities(list(qids))
        if pids:
            await self._fetch_properties(list(pids))

    def _make_triple(self, sig: Tuple[str, str, str], enrich: bool) -> WikiTriple:
        subj, rel, obj = sig
        subj_ent = (
            self._make_entity(subj) if enrich and _is_valid_qid(subj)
            else WikidataEntity(qid=subj)
        )
        rel_prop = (
            self._make_property(rel) if enrich and _is_valid_pid(rel)
            else WikidataProperty(pid=rel)
        )
        if _is_valid_qid(obj):
            obj_val: Any = self._make_entity(obj) if enrich else WikidataEntity(qid=obj)
        else:
            obj_val = obj
        return WikiTriple(subject=subj_ent, relation=rel_prop, object=obj_val)

    # ==========================================================
    # Cached batched fetches (the only L1/L2/single-flight path)
    # ==========================================================

    async def _fetch_cached(
        self,
        items: List[str],
        *,
        lru: _LRU,
        redis_prefix: str,
        coalesce_prefix: str,
        backend_fn: Callable[..., Awaitable[Dict[str, Any]]],
        limiter: _AsyncRateLimiter,
        on_miss: Callable[[str], Any],
        redis_normalize: Optional[Callable[[Any], Any]] = None,
    ) -> Dict[str, Any]:
        """L1 LRU → L2 Redis → single-flight + rate-limited backend fetch."""
        result: Dict[str, Any] = {}
        missing: List[str] = []
        for x in items:
            if x in lru:
                result[x] = lru[x]
                continue
            cached = self._redis_get(f"{redis_prefix}{x}")
            if cached is not None:
                if redis_normalize is not None:
                    cached = redis_normalize(cached)
                lru[x] = cached
                result[x] = cached
                continue
            missing.append(x)
        if not missing:
            return result

        unique = sorted(set(missing))
        fetched = await self._coalesce(
            f"{coalesce_prefix}:" + ",".join(unique),
            lambda: self._call_with_retry(limiter, backend_fn, unique),
        )
        for x in missing:
            val = fetched.get(x)
            if val is None:
                val = on_miss(x)
            lru[x] = val
            if val is not None:
                self._redis_set(f"{redis_prefix}{x}", val)
            result[x] = val
        return result

    async def _fetch_entities(self, qids: List[str]) -> None:
        await self._fetch_cached(
            qids,
            lru=self._entities,
            redis_prefix="wiki:ent:",
            coalesce_prefix="ent",
            backend_fn=self._backend.get_entity_details,
            limiter=self._wiki_limiter,
            on_miss=lambda q: {"qid": q},
        )

    async def _fetch_properties(self, pids: List[str]) -> None:
        # O(1) short-circuit for well-known PIDs; the rest go through the backend.
        to_fetch: List[str] = []
        for p in pids:
            if p in self._properties:
                continue
            builtin = PROPERTY_LABELS.get(p)
            if builtin is not None:
                self._properties[p] = {"pid": p, **builtin}
                continue
            to_fetch.append(p)
        if to_fetch:
            await self._fetch_cached(
                to_fetch,
                lru=self._properties,
                redis_prefix="wiki:prop:",
                coalesce_prefix="prop",
                backend_fn=self._backend.get_property_details,
                limiter=self._wiki_limiter,
                on_miss=lambda p: {"pid": p},
            )

    async def _fetch_outgoing(self, qids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        return await self._fetch_cached(
            qids,
            lru=self._outgoing,
            redis_prefix="wiki:out:",
            coalesce_prefix="out",
            backend_fn=self._backend.fetch_outgoing,
            limiter=self._sparql_limiter,
            on_miss=lambda _q: [],
            redis_normalize=lambda lst: [tuple(e) for e in lst],
        )

    async def _fetch_incoming(self, qids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        return await self._fetch_cached(
            qids,
            lru=self._incoming,
            redis_prefix="wiki:in:",
            coalesce_prefix="in",
            backend_fn=self._backend.fetch_incoming,
            limiter=self._sparql_limiter,
            on_miss=lambda _q: [],
            redis_normalize=lambda lst: [tuple(e) for e in lst],
        )

    async def _fetch_wikipedia(self, titles: List[str]) -> None:
        await self._fetch_cached(
            titles,
            lru=self._wiki,
            redis_prefix="wiki:wp:",
            coalesce_prefix="wp",
            backend_fn=self._backend.get_wikipedia_contents,
            limiter=self._wiki_limiter,
            on_miss=lambda _t: None,
        )

    # ==========================================================
    # Retry + concurrency + single-flight
    # ==========================================================

    async def _call_with_retry(
        self,
        limiter: _AsyncRateLimiter,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Acquire semaphore + rate-limit, call, retry on 429.

        The semaphore is released BEFORE the retry sleep, otherwise retries
        could deadlock with ``concurrency_limit=1``.
        """
        for attempt in range(MAX_RETRIES):
            async with self._semaphore:
                await limiter.acquire()
                try:
                    return await fn(*args, **kwargs)
                except WikidataRateLimitError as e:
                    if attempt >= MAX_RETRIES - 1:
                        raise
                    retry_after = e.retry_after if e.retry_after is not None else (2 ** attempt)
            await asyncio.sleep(retry_after)
        raise RuntimeError("unreachable")  # for type checkers

    async def _coalesce(
        self,
        key: str,
        fetch_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Coalesce concurrent identical fetches via a per-key Future.

        Followers ``await`` the leader's future. If the leader is cancelled,
        a follower retries as the new leader; the follower's own cancellation
        propagates immediately.
        """
        while True:
            fut = self._inflight.get(key)
            if fut is None or fut.done():
                break
            try:
                return await fut
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and getattr(task, "cancelling", lambda: 0)():
                    raise
                # Leader was cancelled; loop and try to become the new leader.

        new_fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._inflight[key] = new_fut
        try:
            result = await fetch_fn()
            new_fut.set_result(result)
            return result
        except asyncio.CancelledError:
            new_fut.cancel()
            raise
        except BaseException as e:
            new_fut.set_exception(e)
            raise
        finally:
            if self._inflight.get(key) is new_fut:
                del self._inflight[key]

    # ==========================================================
    # Model construction from cached records
    # ==========================================================

    def _make_entity(self, qid: str, with_wiki: bool = False) -> WikidataEntity:
        rec = self._entities.get(qid)
        if rec is None:
            return WikidataEntity(qid=qid)
        content: Optional[str] = None
        title = rec.get("wikipedia_title")
        if with_wiki and title:
            content = self._wiki.get(title)
        return WikidataEntity(
            qid=qid,
            label=rec.get("label"),
            description=rec.get("description"),
            aliases=list(rec.get("aliases") or []),
            url=f"https://www.wikidata.org/wiki/{qid}",
            wikipedia_url=rec.get("wikipedia_url"),
            wikipedia_content=content,
        )

    def _make_property(self, pid: str) -> WikidataProperty:
        rec = self._properties.get(pid)
        if rec is None:
            return WikidataProperty(pid=pid)
        return WikidataProperty(
            pid=pid, label=rec.get("label"), description=rec.get("description")
        )

    # ==========================================================
    # Redis L2 (best-effort; never raises)
    # ==========================================================

    def _redis_get(self, key: str) -> Any:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
        except Exception:
            return None
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return raw if isinstance(raw, str) else None

    def _redis_set(self, key: str, value: Any) -> None:
        if self._redis is None:
            return
        try:
            payload = json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):
            return
        try:
            self._redis.setex(key, self._redis_ttl, payload)
        except Exception:
            pass


__all__ = [
    "WikidataClient",
    "WikidataEntity",
    "WikidataProperty",
    "WikiTriple",
    "DEFAULT_PROPERTIES",
    "PROPERTY_LABELS",
]
