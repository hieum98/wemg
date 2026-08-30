"""Production WikidataBackend Protocol + httpx-based implementation.

This is the wire layer the WikidataClient talks to. Everything is async
(httpx.AsyncClient) so the client never has to fall back to asyncio.to_thread.

Endpoints used:
  - https://query.wikidata.org/sparql                (SPARQL k-hop)
  - https://www.wikidata.org/w/api.php               (wbsearchentities, wbgetentities)
  - https://en.wikipedia.org/w/api.php               (page extracts)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
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
    async def get_property_details(
        self, pids: list[str]
    ) -> dict[str, PropertyRecord]: ...
    async def fetch_outgoing(
        self, qids: list[str]
    ) -> dict[str, list[tuple[str, str]]]: ...
    async def fetch_incoming(
        self, qids: list[str]
    ) -> dict[str, list[tuple[str, str]]]: ...
    async def get_wikipedia_contents(
        self, titles: list[str]
    ) -> dict[str, Optional[str]]: ...


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


def default_user_agent() -> str:
    """Wikimedia-compliant UA (https://w.wiki/4wJS): project id + contact, not ``bot/``."""
    override = os.environ.get("WIKIDATA_USER_AGENT") or os.environ.get("COE_USER_AGENT")
    if override:
        return override.strip()
    contact = os.environ.get("WIKIDATA_CONTACT", "contact/hieum@uoregon.edu").strip()
    return f"COE/0.2.0.{_random_id()} (langgraph_coe; {contact}) python-httpx"


DEFAULT_USER_AGENT = default_user_agent()


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


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
            # Force IPv4. www.wikidata.org / en.wikipedia.org resolve to an IPv6
            # address that this (IPv4-only-egress) compute node cannot route, and
            # the async client otherwise stalls on the dead IPv6 path until the
            # connect timeout (the ConnectTimeouts seen in eval). Binding the
            # source to the IPv4 wildcard pins the socket family to AF_INET, so
            # only IPv4 destinations are attempted — matching the working curl -4.
            transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
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
        data = await self._wikidata_api_call(
            {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "type": "item",
                "format": "json",
                "limit": max(1, min(limit, 50)),
            }
        )
        return [
            r["id"] for r in data.get("search", []) if isinstance(r, dict) and "id" in r
        ]

    #: Minimum ratio between the best and second-best candidate's statement count for a
    #: SPARQL label match to be trusted. Statement count is a prominence proxy, and it
    #: ranks the *unambiguous* cases the same way ``wbsearchentities`` does while giving
    #: no useful signal on genuinely ambiguous labels.
    #:
    #: Measured on 30 entity names taken from real runs (every one of which had failed
    #: with HTTP 429), comparing this resolver's top-1 against the API's top-1:
    #:
    #:   threshold   accepted   correct   wrong   precision   coverage
    #:          1x         23        20       3         87%        77%
    #:          4x         18        17       1         94%        60%
    #:        > 8x         15        15       0        100%        50%
    #:
    #: The errors at low thresholds are exactly the short ambiguous labels — "ABC"
    #: (1.3x), "State" (1.0x), "The Book Thief" (4.9x) — while the correct ones separate
    #: sharply ("Dolly Parton" 25.7x, and most have no runner-up at all). 8x is chosen
    #: conservatively: 15/15 is consistent with a true precision near 80%, and a wrong
    #: QID is worse than no QID because it manufactures a false binding.
    _LOCAL_DOMINANCE = 8.0

    async def search_entities_local(self, query: str, *, limit: int = 1) -> list[str]:
        """Resolve an English label to QIDs over the configured SPARQL endpoint.

        A fallback for when ``wbsearchentities`` is unavailable, **not** a replacement:
        the API ranks by relevance, this ranks by statement count, and the two agree only
        where the label is unambiguous. Returns ``[]`` rather than guessing when the top
        candidate does not dominate (see ``_LOCAL_DOMINANCE``).

        Motivation, measured: 3,805 name lookups across two evaluation runs were lost to
        HTTP 429 on the public API — about 8 per question. Each loss means no QID, so an
        intent cannot close and its referent never becomes available to ground a later
        retrieval query, which is the dominant reason unresolved intents had facts but no
        binding.
        """
        name = (query or "").strip()
        if not name:
            return []
        # JSON string escaping is a valid subset of SPARQL string escaping for the
        # quote/backslash/control characters that appear in entity labels.
        literal = json.dumps(name)
        sparql = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            f"SELECT ?e (COUNT(?p) AS ?n) WHERE {{ ?e rdfs:label {literal}@en . "
            "?e ?p ?o }\n"
            "GROUP BY ?e ORDER BY DESC(?n) LIMIT 2"
        )
        try:
            rows = await self._sparql_query(sparql)
        except Exception as exc:  # noqa: BLE001 — a dead fallback must not fail the hop
            logger.debug("[wikidata] local label lookup failed for %r: %s", name, exc)
            return []
        if not rows:
            return []

        def _count(row: dict) -> int:
            try:
                return int(row.get("n", {}).get("value", 0))
            except (TypeError, ValueError):
                return 0

        def _qid(row: dict) -> str:
            return str(row.get("e", {}).get("value", "")).rsplit("/", 1)[-1]

        best, best_n = _qid(rows[0]), _count(rows[0])
        if not re.fullmatch(r"Q[1-9][0-9]*", best):
            return []
        runner_n = _count(rows[1]) if len(rows) > 1 else 0
        if runner_n and best_n < runner_n * self._LOCAL_DOMINANCE:
            logger.debug(
                "[wikidata] local label %r is ambiguous (%d vs %d statements); "
                "declining to guess",
                name,
                best_n,
                runner_n,
            )
            return []
        logger.info(
            "[wikidata] resolved %r -> %s locally (%d statements, runner-up %d)",
            name,
            best,
            best_n,
            runner_n,
        )
        return [best][: max(1, limit)]

    async def get_entity_details_local(
        self, qids: list[str]
    ) -> dict[str, EntityRecord]:
        """Labels and English descriptions for *qids* over the SPARQL endpoint.

        The other half of the public-API gap. ``wbgetentities`` is throttled exactly like
        ``wbsearchentities``, and when it fails ``entity_dict`` has no labels — which
        silently disables ``_known_entity_labels`` (so the KG fan-out gate stops firing on
        known entities) and label-based binding resolution.

        Unlike :meth:`search_entities_local` this needs no disambiguation gate: the lookup
        is keyed on an exact QID, so there is nothing to guess. Aliases and Wikipedia
        titles are not recovered — a truthy dump carries neither — so this is a partial
        record by construction and the caller keeps whatever the API already gave it.
        """
        wanted = [q for q in qids if isinstance(q, str) and re.fullmatch(r"Q[1-9][0-9]*", q)]
        if not wanted:
            return {}
        out: dict[str, EntityRecord] = {}
        # Batched VALUES rather than one query per QID: 35ms for a batch against ~14ms
        # per single lookup, and the k-hop path routinely resolves dozens at once.
        for i in range(0, len(wanted), 100):
            chunk = wanted[i : i + 100]
            values = " ".join(f"<{_ENTITY_URI_PREFIX}{q}>" for q in chunk)
            sparql = (
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                "PREFIX schema: <http://schema.org/>\n"
                f"SELECT ?e ?l ?d WHERE {{ VALUES ?e {{ {values} }} "
                '?e rdfs:label ?l . FILTER(lang(?l)="en") '
                'OPTIONAL {{ ?e schema:description ?d FILTER(lang(?d)="en") }} }}'
            ).replace("{{", "{").replace("}}", "}")
            try:
                rows = await self._sparql_query(sparql)
            except Exception as exc:  # noqa: BLE001 — a dead fallback must not fail the hop
                logger.debug("[wikidata] local entity details failed: %s", exc)
                continue
            for row in rows:
                qid = str(row.get("e", {}).get("value", "")).rsplit("/", 1)[-1]
                if not qid:
                    continue
                out[qid] = {
                    "qid": qid,
                    "label": row.get("l", {}).get("value") or None,
                    "description": row.get("d", {}).get("value") or None,
                    "aliases": [],
                }
        if out:
            logger.info(
                "[wikidata] resolved %d/%d entity label(s) locally",
                len(out),
                len(wanted),
            )
        return out

    async def get_property_details_local(
        self, pids: list[str]
    ) -> dict[str, PropertyRecord]:
        """Property labels over the SPARQL endpoint. Same rationale as entity details.

        ``PROPERTY_LABELS`` already short-circuits the common PIDs, so this only covers
        the tail — but a throttled fetch there used to raise and take the whole triple
        labelling with it, leaving readable triples as bare P-numbers.
        """
        wanted = [
            p for p in pids if isinstance(p, str) and re.fullmatch(r"P[1-9][0-9]*", p)
        ]
        if not wanted:
            return {}
        out: dict[str, PropertyRecord] = {}
        for i in range(0, len(wanted), 100):
            chunk = wanted[i : i + 100]
            values = " ".join(f"<{_ENTITY_URI_PREFIX}{p}>" for p in chunk)
            sparql = (
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                f"SELECT ?p ?l WHERE {{ VALUES ?p {{ {values} }} "
                '?p rdfs:label ?l FILTER(lang(?l)="en") }'
            )
            try:
                rows = await self._sparql_query(sparql)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[wikidata] local property labels failed: %s", exc)
                continue
            for row in rows:
                pid = str(row.get("p", {}).get("value", "")).rsplit("/", 1)[-1]
                if pid:
                    out[pid] = {"pid": pid, "label": row.get("l", {}).get("value")}
        if out:
            logger.info(
                "[wikidata] resolved %d/%d property label(s) locally",
                len(out),
                len(wanted),
            )
        return out

    async def get_entity_details(self, qids: list[str]) -> dict[str, EntityRecord]:
        if not qids:
            return {}
        result: dict[str, EntityRecord] = {}
        # wbgetentities supports up to 50 ids per request
        for i in range(0, len(qids), 50):
            chunk = qids[i : i + 50]
            data = await self._wikidata_api_call(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels|descriptions|aliases|sitelinks/urls",
                    "languages": "en",
                    "sitefilter": "enwiki",
                    "format": "json",
                }
            )
            entities = data.get("entities") or {}
            for qid, ent in entities.items():
                if "missing" in ent:
                    continue
                label = (ent.get("labels", {}).get("en") or {}).get("value")
                desc = (ent.get("descriptions", {}).get("en") or {}).get("value")
                aliases_raw = ent.get("aliases", {}).get("en") or []
                aliases = [
                    a["value"]
                    for a in aliases_raw
                    if isinstance(a, dict) and "value" in a
                ]
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
        data = await self._wikidata_api_call(
            {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "type": "property",
                "format": "json",
                "limit": max(1, min(limit, 50)),
            }
        )
        return [
            r["id"] for r in data.get("search", []) if isinstance(r, dict) and "id" in r
        ]

    async def get_property_details(self, pids: list[str]) -> dict[str, PropertyRecord]:
        if not pids:
            return {}
        result: dict[str, PropertyRecord] = {}
        for i in range(0, len(pids), 50):
            chunk = pids[i : i + 50]
            data = await self._wikidata_api_call(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels|descriptions",
                    "languages": "en",
                    "format": "json",
                }
            )
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

    async def get_wikipedia_contents(
        self, titles: list[str]
    ) -> dict[str, Optional[str]]:
        if not titles:
            return {}
        result: dict[str, Optional[str]] = {t: None for t in titles}
        # Wikipedia API supports up to 50 titles per call
        for i in range(0, len(titles), 20):
            chunk = titles[i : i + 20]
            resp = await self._client.get(
                self._wikipedia_api,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "titles": "|".join(chunk),
                    "format": "json",
                    "redirects": 1,
                },
            )
            self._raise_on_rate_limit(resp)
            resp.raise_for_status()
            data = resp.json()
            query = data.get("query") or {}
            normalized = {
                n["from"]: n["to"]
                for n in query.get("normalized", [])
                if "from" in n and "to" in n
            }
            redirects = {
                r["from"]: r["to"]
                for r in query.get("redirects", [])
                if "from" in r and "to" in r
            }
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
