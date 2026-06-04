"""Production WikidataBackend Protocol + httpx-based implementation.

This is the wire layer the WikidataClient talks to. Everything is async
(httpx.AsyncClient) so the client never has to fall back to asyncio.to_thread.

Endpoints used:
  - https://query.wikidata.org/sparql                (SPARQL k-hop)
  - https://www.wikidata.org/w/api.php               (wbsearchentities, wbgetentities)
  - https://en.wikipedia.org/w/api.php               (page extracts)
"""

from __future__ import annotations

import logging
import random
import string
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Records (TypedDicts the backend returns)
# ---------------------------------------------------------------------------


class EntityRecord(TypedDict, total=False):
    qid: str
    label: Optional[str]
    description: Optional[str]
    aliases: list[str]
    wikipedia_title: Optional[str]
    wikipedia_url: Optional[str]


class PropertyRecord(TypedDict, total=False):
    pid: str
    label: Optional[str]
    description: Optional[str]


# ---------------------------------------------------------------------------
# Rate-limit signal
# ---------------------------------------------------------------------------


class WikidataRateLimitError(Exception):
    """Backend signals HTTP 429 (or equivalent). Carries Retry-After if available."""

    def __init__(self, retry_after: Optional[float] = None) -> None:
        super().__init__(f"Wikidata rate limit hit (retry_after={retry_after}s)")
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WikidataBackend(Protocol):
    """Pure fetch boundary. The client owns cache/rate-limit/batching/traversal."""

    async def search_entities_text(self, query: str, *, limit: int) -> list[str]: ...
    async def get_entity_details(self, qids: list[str]) -> dict[str, EntityRecord]: ...
    async def search_properties_text(self, query: str, *, limit: int) -> list[str]: ...
    async def get_property_details(self, pids: list[str]) -> dict[str, PropertyRecord]: ...
    async def fetch_outgoing(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]: ...
    async def fetch_incoming(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]: ...
    async def get_wikipedia_contents(self, titles: list[str]) -> dict[str, Optional[str]]: ...


# ---------------------------------------------------------------------------
# HTTP implementation
# ---------------------------------------------------------------------------


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

_ENTITY_URI_PREFIX = "http://www.wikidata.org/entity/"
_PROP_DIRECT_URI_PREFIX = "http://www.wikidata.org/prop/direct/"

# QEndpoint and some local SPARQL engines require explicit PREFIX declarations for wd:.
_SPARQL_PREFIXES = (
    "PREFIX wd: <http://www.wikidata.org/entity/> "
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
)


def _random_id(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


DEFAULT_USER_AGENT = (
    f"WEMG/0.2.0 (langgraph_coe; bot/{_random_id()}) python-httpx"
)


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value


def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class HTTPWikidataBackend:
    """httpx.AsyncClient-based implementation of WikidataBackend."""

    def __init__(
        self,
        *,
        sparql_endpoint: str = SPARQL_ENDPOINT,
        wikidata_api: str = WIKIDATA_API,
        wikipedia_api: str = WIKIPEDIA_API,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 60.0,
    ) -> None:
        self._sparql_endpoint = sparql_endpoint
        self._wikidata_api = wikidata_api
        self._wikipedia_api = wikipedia_api
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"User-Agent": user_agent, "Accept-Language": "en"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------- internal helpers ----------------

    @staticmethod
    def _raise_on_rate_limit(resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise WikidataRateLimitError(retry_after=_retry_after_seconds(resp))

    async def _wikidata_api_call(self, params: dict[str, Any]) -> dict:
        resp = await self._client.get(self._wikidata_api, params=params)
        self._raise_on_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    async def _sparql_query(self, query: str) -> list[dict]:
        resp = await self._client.get(
            self._sparql_endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        self._raise_on_rate_limit(resp)
        resp.raise_for_status()
        return resp.json().get("results", {}).get("bindings", [])

    # ---------------- entity search ----------------

    async def search_entities_text(self, query: str, *, limit: int) -> list[str]:
        data = await self._wikidata_api_call({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "type": "item",
            "format": "json",
            "limit": max(1, min(limit, 50)),
        })
        return [r["id"] for r in data.get("search", []) if isinstance(r, dict) and "id" in r]

    async def get_entity_details(self, qids: list[str]) -> dict[str, EntityRecord]:
        if not qids:
            return {}
        result: dict[str, EntityRecord] = {}
        # wbgetentities supports up to 50 ids per request
        for i in range(0, len(qids), 50):
            chunk = qids[i: i + 50]
            data = await self._wikidata_api_call({
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels|descriptions|aliases|sitelinks/urls",
                "languages": "en",
                "sitefilter": "enwiki",
                "format": "json",
            })
            entities = data.get("entities") or {}
            for qid, ent in entities.items():
                if "missing" in ent:
                    continue
                label = (ent.get("labels", {}).get("en") or {}).get("value")
                desc = (ent.get("descriptions", {}).get("en") or {}).get("value")
                aliases_raw = ent.get("aliases", {}).get("en") or []
                aliases = [a["value"] for a in aliases_raw if isinstance(a, dict) and "value" in a]
                sitelink = ent.get("sitelinks", {}).get("enwiki") or {}
                result[qid] = {
                    "qid": qid,
                    "label": label,
                    "description": desc,
                    "aliases": aliases,
                    "wikipedia_title": sitelink.get("title"),
                    "wikipedia_url": sitelink.get("url"),
                }
        return result

    # ---------------- property search ----------------

    async def search_properties_text(self, query: str, *, limit: int) -> list[str]:
        data = await self._wikidata_api_call({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "type": "property",
            "format": "json",
            "limit": max(1, min(limit, 50)),
        })
        return [r["id"] for r in data.get("search", []) if isinstance(r, dict) and "id" in r]

    async def get_property_details(self, pids: list[str]) -> dict[str, PropertyRecord]:
        if not pids:
            return {}
        result: dict[str, PropertyRecord] = {}
        for i in range(0, len(pids), 50):
            chunk = pids[i: i + 50]
            data = await self._wikidata_api_call({
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels|descriptions",
                "languages": "en",
                "format": "json",
            })
            entities = data.get("entities") or {}
            for pid, ent in entities.items():
                if "missing" in ent:
                    continue
                label = (ent.get("labels", {}).get("en") or {}).get("value")
                desc = (ent.get("descriptions", {}).get("en") or {}).get("value")
                result[pid] = {"pid": pid, "label": label, "description": desc}
        return result

    # ---------------- SPARQL k-hop ----------------

    async def fetch_outgoing(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]:
        if not qids:
            return {}
        values = " ".join(f"wd:{q}" for q in qids)
        query = (
            f"{_SPARQL_PREFIXES}"
            "SELECT ?seed ?p ?o WHERE { "
            f"VALUES ?seed {{ {values} }} "
            "?seed ?p ?o . "
            "FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/')) "
            "} LIMIT 10000"
        )
        rows = await self._sparql_query(query)
        result: dict[str, list[tuple[str, str]]] = {q: [] for q in qids}
        for row in rows:
            try:
                seed = _strip_prefix(row["seed"]["value"], _ENTITY_URI_PREFIX)
                pid = _strip_prefix(row["p"]["value"], _PROP_DIRECT_URI_PREFIX)
                obj_node = row["o"]
                if obj_node.get("type") == "uri":
                    obj = _strip_prefix(obj_node["value"], _ENTITY_URI_PREFIX)
                else:
                    obj = obj_node.get("value", "")
                if seed in result:
                    result[seed].append((pid, obj))
            except (KeyError, TypeError):
                continue
        return result

    async def fetch_incoming(self, qids: list[str]) -> dict[str, list[tuple[str, str]]]:
        if not qids:
            return {}
        values = " ".join(f"wd:{q}" for q in qids)
        query = (
            f"{_SPARQL_PREFIXES}"
            "SELECT ?seed ?p ?s WHERE { "
            f"VALUES ?seed {{ {values} }} "
            "?s ?p ?seed . "
            "FILTER(STRSTARTS(STR(?p), 'http://www.wikidata.org/prop/direct/')) "
            "} LIMIT 10000"
        )
        rows = await self._sparql_query(query)
        result: dict[str, list[tuple[str, str]]] = {q: [] for q in qids}
        for row in rows:
            try:
                seed = _strip_prefix(row["seed"]["value"], _ENTITY_URI_PREFIX)
                pid = _strip_prefix(row["p"]["value"], _PROP_DIRECT_URI_PREFIX)
                subj_node = row["s"]
                if subj_node.get("type") == "uri":
                    subj = _strip_prefix(subj_node["value"], _ENTITY_URI_PREFIX)
                else:
                    continue  # incoming edges must originate from an entity URI
                if seed in result:
                    result[seed].append((pid, subj))
            except (KeyError, TypeError):
                continue
        return result

    # ---------------- Wikipedia ----------------

    async def get_wikipedia_contents(self, titles: list[str]) -> dict[str, Optional[str]]:
        if not titles:
            return {}
        result: dict[str, Optional[str]] = {t: None for t in titles}
        # Wikipedia API supports up to 50 titles per call
        for i in range(0, len(titles), 20):
            chunk = titles[i: i + 20]
            resp = await self._client.get(self._wikipedia_api, params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": "|".join(chunk),
                "format": "json",
                "redirects": 1,
            })
            self._raise_on_rate_limit(resp)
            resp.raise_for_status()
            data = resp.json()
            query = data.get("query") or {}
            normalized = {n["from"]: n["to"] for n in query.get("normalized", []) if "from" in n and "to" in n}
            redirects = {r["from"]: r["to"] for r in query.get("redirects", []) if "from" in r and "to" in r}
            pages = query.get("pages") or {}
            # Map MediaWiki's canonical title back to each originally-requested title
            extract_by_title: dict[str, Optional[str]] = {}
            for page in pages.values():
                if "missing" in page:
                    continue
                title = page.get("title")
                if title:
                    extract_by_title[title] = page.get("extract")
            for orig in chunk:
                canonical = normalized.get(orig, orig)
                canonical = redirects.get(canonical, canonical)
                if canonical in extract_by_title:
                    result[orig] = extract_by_title[canonical]
        return result
