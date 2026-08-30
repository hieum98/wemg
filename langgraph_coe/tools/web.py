from __future__ import annotations

import asyncio
import logging
import os
import re
from contextvars import ContextVar
from typing import Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool

from ..config import WebSearchConfig

logger = logging.getLogger(__name__)

_web_search_instance = None
_web_config = None
_web_cache = None

# ──────────────────────────────────────────────────────────────────────────────
# Provider chain
# ──────────────────────────────────────────────────────────────────────────────
#
# History, because the ordering here is the whole design and looks arbitrary
# without it:
#
#   1. ``ddgs`` (which rotates DuckDuckGo / Yandex / Brave internally) was the free
#      primary. Under an evaluation sweep it throttled hard — 201 HTTP 429s across
#      two concurrent runs in under an hour, which is what made a 120-question depth
#      study a ~10-hour job rather than a ~2-hour one.
#   2. Tavily was added as the billed escape hatch for exactly those 429s.
#   3. Then ``ddgs`` stopped being merely throttled and became *unreachable* from
#      this host: ``duckduckgo.com`` resolves to an intercepted address and the TCP
#      connect times out. With the free primary gone, every query fell through to
#      Tavily, which burned its 1000-request free tier and started returning
#      HTTP 432 ("exceeds your plan's set usage limit"). Both tiers dead, and
#      because a failed provider returns ``[]`` rather than raising, the whole web
#      surface degraded to empty results *silently*.
#
# So the chain is now explicit and ordered by how exhaustible each provider is,
# and it must not be "improved" into a race or a round-robin:
#
#   unmetered  →  metered  →  billed
#
# A provider backed by a finite bucket is a fallback, not a peer: a race would spend
# a Serper credit on every query that SearXNG could have answered for free.
#
# The ordering is safe only because the head of the chain is nearly as good as the
# metered providers *under production settings*. Measured over 8 subquestion-shaped
# queries, gold answer present anywhere in the top-3 documents:
#
#   provider   snippets only   after crawl   mean full_text chars/query
#   searxng          2/8           7/8              69,808
#   builtin          8/8           8/8             101,452
#   brave            8/8           8/8             129,747
#   wikipedia        8/8           8/8               4,005
#
# Two things to take from that table. First, SearXNG's snippets are genuinely bad
# (2/8) and only ``crawl_full_text`` rescues it to 7/8. Second, ``wikipedia`` matches
# the web providers' recall on this query shape using ~17-32x less text, which
# matters because the binding constraint on accuracy here is the memory budget
# (~7.7 surviving items per question), not corpus coverage.
#
# So ``wikipedia`` leads the chain, and the honest description of that is not "extra
# coverage" but *near-exclusivity*: MediaWiki almost always returns something, the
# chain stops at the first non-empty provider, so everything below it mostly never
# runs. Three things follow.
#
# First, it is chosen on gold-source grounds, not cost grounds. Both leading
# providers are unmetered, so ordering them cannot save anything; the reason is that
# the eval set is MuSiQue, whose gold paragraphs are drawn from Wikipedia. Retrieval
# is 78% of failures here, so this targets the dominant term.
#
# Second, it is accepted with a known failure mode rather than because the failure
# was disproved: MediaWiki full-text search collapses on compositional queries —
# "who founded the company that made the Model T" returns A. T. Cross and The Boring
# Company, no Ford — and a wrong Wikipedia hit now shadows *every* other provider
# rather than falling through. The bet is that the agent issues subquestion-shaped
# queries (the 8/8 row), not whole multi-hop questions. What would refute it: recall
# dropping against the searxng-first order, with ``read_provider_usage()`` showing
# everything below ``wikipedia`` near zero.
#
# Third, ordering is the wrong instrument for the two unmetered providers and this is
# a stopgap. Since neither costs anything, the shadowing is pure loss: the principled
# fix is to query ``wikipedia`` and ``searxng`` together and merge, or to make
# "answered" a relevance test rather than ``len(results) > 0``. Both remove the
# either/or this ordering is forced to resolve.
#
# Below the free tier, ``brave`` precedes ``builtin``: Brave's 2k refills monthly,
# Serper's ~2.5k is one-time and never replenished, so under the exhaustibility
# ordering above Serper is the later fallback.
_SEARXNG_DEFAULT_URL = "http://localhost:8080"

_tavily_key: str | None = None
_brave_key: str | None = None
_searxng_url: str = _SEARXNG_DEFAULT_URL
_providers: List[str] = ["builtin", "tavily"]

_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Wikimedia blocks generic user agents outright, so this has to identify the client.
_WIKI_UA = "wemg-research/0.1 (langgraph_coe eval; https://github.com/hieuman/wemg)"

# Per-provider call counts, so spend against the finite buckets is reportable rather
# than a surprise. Only requests that actually reached the provider are counted.
_provider_calls: Dict[str, int] = {}


def _count(provider: str) -> None:
    _provider_calls[provider] = _provider_calls.get(provider, 0) + 1


def read_provider_usage() -> Dict[str, int]:
    """Successful requests per provider so far in this process."""
    return dict(_provider_calls)


def read_tavily_usage() -> int:
    """Billed Tavily requests so far in this process."""
    return _provider_calls.get("tavily", 0)


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    """Drop the term-highlight markup some providers put in their snippets."""
    return _TAG_RE.sub("", text)


class _WebSession:
    __slots__ = ("visited",)

    def __init__(self) -> None:
        self.visited: set[str] = set()


_cv_web_session: ContextVar[_WebSession | None] = ContextVar(
    "web_research_session", default=None
)


def _get_web_session() -> _WebSession:
    s = _cv_web_session.get(None)
    if s is None:
        s = _WebSession()
        _cv_web_session.set(s)
    return s


def reset_web_research_session() -> None:
    _cv_web_session.set(_WebSession())


def init_web_search(config: WebSearchConfig):
    global _web_search_instance, _web_config, _tavily_key, _brave_key
    global _searxng_url, _providers
    _web_config = config
    api_key = config.api_key or os.environ.get("SERPER_API_KEY")

    if api_key:
        _web_search_instance = GoogleSerperAPIWrapper(serper_api_key=api_key)
    else:
        _web_search_instance = DuckDuckGoSearchAPIWrapper(max_results=config.top_k)

    # Env vars take precedence: they keep the secrets out of the committed YAML.
    _tavily_key = os.environ.get("TAVILY_API_KEY") or config.tavily_api_key
    _brave_key = os.environ.get("BRAVE_API_KEY") or config.brave_api_key
    _searxng_url = (
        os.environ.get("SEARXNG_URL") or config.searxng_url or _SEARXNG_DEFAULT_URL
    ).rstrip("/")

    # An unknown name in the chain is a config typo, and the failure mode without
    # this check is the worst kind: the provider is silently skipped and the run
    # completes with a quietly shallower document surface.
    unknown = [p for p in config.providers if p not in _PROVIDERS]
    if unknown:
        raise ValueError(
            f"web_search.providers contains unknown provider(s) {unknown}; "
            f"known providers are {sorted(_PROVIDERS)}"
        )
    _providers = list(config.providers)
    _provider_calls.clear()

    # A provider named in the chain but missing its key never fires. Say so at
    # init rather than letting the chain look longer than it really is.
    disabled = []
    if "tavily" in _providers and not _tavily_key:
        disabled.append("tavily(no key)")
    if "brave" in _providers and not _brave_key:
        disabled.append("brave(no key)")
    logger.info(
        "[web_search] chain=%s | builtin=%s | searxng=%s%s",
        " → ".join(_providers) or "(none)",
        "serper" if api_key else "ddgs(free)",
        _searxng_url if "searxng" in _providers else "off",
        f" | inert: {', '.join(disabled)}" if disabled else "",
    )


async def _crawl_page(url: str, client: httpx.AsyncClient) -> str:
    try:
        response = await client.get(
            url,
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        def _parse(html: str) -> str:
            soup = BeautifulSoup(html, "lxml")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return re.sub(r"\n{3,}", "\n\n", text)

        return await asyncio.to_thread(_parse, response.text)
    except Exception:
        return ""


async def _search_builtin(query: str, top_k: int) -> List[Dict[str, str]]:
    """Serper (if keyed) or the ``ddgs`` rotation. May raise, may return [].

    Named ``builtin`` rather than ``free`` because it is only free in the unkeyed
    ``ddgs`` case; with a Serper key it spends from a finite credit bucket, which is
    why the chain no longer treats this slot as the always-first provider.
    """
    out: List[Dict[str, str]] = []
    if isinstance(_web_search_instance, GoogleSerperAPIWrapper):
        res = await asyncio.to_thread(_web_search_instance.results, query)
        _count("builtin")
        for item in res.get("organic", [])[:top_k]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "full_text": "",
                }
            )
        return out

    res = await asyncio.to_thread(_web_search_instance.results, query, top_k)
    _count("builtin")
    if isinstance(res, list):
        for item in res:
            out.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "full_text": "",
                }
            )
    elif res:
        # ``ddgs`` degrades to a single string on some backends. Keep it: a snippet
        # with no URL is still evidence, it just cannot be crawled.
        out.append({"title": "", "snippet": str(res), "url": "", "full_text": ""})
    return out


async def _search_searxng(query: str, top_k: int) -> List[Dict[str, str]]:
    """Local SearXNG metasearch. Unmetered, but now behind ``wikipedia``.

    Requires the container from ``setup/searxng_up.sh``; when it is not running
    this returns ``[]`` and the chain moves on, which is the intended degradation.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{_searxng_url}/search",
                params={"q": query, "format": "json", "language": "en"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001 — a dead provider must not fail the hop
        logger.info(
            "[web_search] searxng unavailable (%s: %s); falling through",
            type(e).__name__,
            str(e)[:120],
        )
        return []
    _count("searxng")

    out: List[Dict[str, str]] = []
    for item in (data.get("results") or [])[:top_k]:
        out.append(
            {
                "title": str(item.get("title") or ""),
                "snippet": str(item.get("content") or ""),
                "url": str(item.get("url") or ""),
                # SearXNG returns engine snippets only, never page bodies, so the
                # crawl in ``web_search`` still has to run for these.
                "full_text": "",
            }
        )
    # The wikipedia/wikidata engines answer into ``infoboxes`` rather than
    # ``results``, so an entity query can look empty while carrying the single best
    # piece of evidence in the response. Backfill from there.
    if len(out) < top_k:
        for box in (data.get("infoboxes") or [])[: top_k - len(out)]:
            urls = box.get("urls") or []
            out.append(
                {
                    "title": str(box.get("infobox") or ""),
                    "snippet": str(box.get("content") or ""),
                    "url": str(box.get("id") or (urls[0].get("url") if urls else "")),
                    "full_text": "",
                }
            )
    if data.get("unresponsive_engines"):
        logger.debug(
            "[web_search] searxng unresponsive engines: %s",
            data["unresponsive_engines"],
        )
    return out


async def _search_wikipedia(query: str, top_k: int) -> List[Dict[str, str]]:
    """MediaWiki search. Unmetered, keyless, and the head of the chain.

    ``generator=search`` + ``prop=extracts`` returns the matched pages *and* their
    text in a single round trip, so a hit skips the separate crawl the way Tavily's
    ``raw_content`` does. ``exintro`` keeps that text to the lead section: measured
    500–1700 chars, which is evidence-dense rather than a whole article dumped into
    the context window.
    """
    lang = (_web_config.wikipedia_lang if _web_config else "en") or "en"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrsearch": query,
                    "gsrlimit": max(1, int(top_k)),
                    "prop": "extracts|info",
                    "inprop": "url",
                    "explaintext": 1,
                    "exintro": 1,
                    # Without ``exlimit=max`` only the first page comes back with an
                    # extract and the rest are silently text-free.
                    "exlimit": "max",
                },
                headers={"User-Agent": _WIKI_UA},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[web_search] wikipedia provider failed: %s: %s", type(e).__name__, e
        )
        return []
    _count("wikipedia")

    pages = (data.get("query") or {}).get("pages") or {}
    # ``pages`` is a dict keyed by pageid; ``index`` carries the search rank, and
    # dict order does not.
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 1 << 30))
    out: List[Dict[str, str]] = []
    for page in ordered[:top_k]:
        extract = str(page.get("extract") or "")
        title = str(page.get("title") or "")
        out.append(
            {
                "title": title,
                # No separate snippet field is available alongside an extract, so
                # the lead sentence stands in for one.
                "snippet": extract[:300],
                "url": str(
                    page.get("fullurl")
                    or f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                ),
                "full_text": extract,
            }
        )
    return out


async def _search_brave(query: str, top_k: int) -> List[Dict[str, str]]:
    """Brave Search API. Keyed and quota'd (2k/month free), hence behind SearXNG."""
    if not _brave_key:
        return []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                _BRAVE_URL,
                params={"q": query, "count": max(1, int(top_k))},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": _brave_key,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("[web_search] brave failed: %s: %s", type(e).__name__, e)
        return []
    _count("brave")

    out: List[Dict[str, str]] = []
    for item in ((data.get("web") or {}).get("results") or [])[:top_k]:
        out.append(
            {
                "title": str(item.get("title") or ""),
                # Brave marks up query terms in ``description`` with <strong>.
                "snippet": _strip_tags(str(item.get("description") or "")),
                "url": str(item.get("url") or ""),
                "full_text": "",
            }
        )
    return out


async def _search_tavily(query: str, top_k: int) -> List[Dict[str, str]]:
    """Billed Tavily search. Last in the chain; only reached when all else is empty.

    ``include_raw_content`` is requested so a hit can skip the separate crawl:
    Tavily already fetched the page, and re-fetching it would spend the latency the
    fallback exists to avoid.
    """
    if not _tavily_key:
        return []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _TAVILY_URL,
                json={
                    "query": query,
                    "max_results": top_k,
                    "search_depth": "basic",
                    "include_raw_content": True,
                },
                headers={"Authorization": f"Bearer {_tavily_key}"},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001 — a dead fallback must not fail the hop
        logger.warning("[web_search] tavily fallback failed: %s: %s", type(e).__name__, e)
        return []
    # Counted only after the response is known good. Counting before the request
    # (as this did) reported quota-rejected calls as spend, which is how an
    # exhausted key looked like 1000 useful searches instead of a dead provider.
    _count("tavily")
    out: List[Dict[str, str]] = []
    for item in (data.get("results") or [])[:top_k]:
        out.append(
            {
                "title": str(item.get("title") or ""),
                "snippet": str(item.get("content") or ""),
                "url": str(item.get("url") or ""),
                "full_text": str(item.get("raw_content") or ""),
            }
        )
    logger.info(
        "[web_search] tavily fallback returned %d results (billed call #%d) for %r",
        len(out),
        _provider_calls.get("tavily", 0),
        query[:60],
    )
    return out


# Registry consulted by ``init_web_search`` for validation and by ``web_search`` for
# dispatch. Adding a provider means adding it here and nowhere else.
_PROVIDERS: Dict[str, Callable[[str, int], Awaitable[List[Dict[str, str]]]]] = {
    "builtin": _search_builtin,
    "searxng": _search_searxng,
    "wikipedia": _search_wikipedia,
    "brave": _search_brave,
    "tavily": _search_tavily,
}


# ──────────────────────────────────────────────────────────────────────────────
# Plan-conditioned retrieval depth
# ──────────────────────────────────────────────────────────────────────────────
#
# ``retriever.enabled`` is false in this configuration, so the entire document
# surface is web search at ``top_k: 3``. Measured: 2.55 queries per hop, so ~7.6
# documents per hop and ~24 per question, from which a mean 7.7 memory items survive.
# That budget — not corpus coverage — is what puts the gold answer outside memory on
# 78% of failures.
#
# Reallocating that budget by hop width was **tried and rejected**: giving a one-query
# hop 8 results instead of 3 eroded the measured call saving (Wilcoxon p = 0.51, median
# +1.0 calls) without moving accuracy (+2, against +5 without it). The depth override
# below is kept only as the single place ``top_k`` is read; nothing sets it. The plan's
# retrieval contribution is *query quality*, not query depth — pooled over 461 pairs the
# gold answer reached memory on 41.4% of plan questions against 37.5% without
# (theta = 0.590, p = 0.0886).
#
# Superseded reasoning, kept because it is what the numbers refuted: concentration is
# unsafe without it: fewer queries with no plan is simply less coverage.
_cv_query_depth: ContextVar[Optional[int]] = ContextVar("web_query_depth", default=None)


def set_query_depth(top_k: Optional[int]) -> None:
    """Override ``top_k`` for web searches issued on this async Task."""
    _cv_query_depth.set(int(top_k) if top_k and top_k > 0 else None)


def clear_query_depth() -> None:
    _cv_query_depth.set(None)


def _effective_top_k() -> int:
    override = _cv_query_depth.get(None)
    if override:
        return override
    return int(_web_config.top_k) if _web_config else 3


@tool
async def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web for up-to-date information on a specific query."""
    if not _web_search_instance or not _web_config:
        raise RuntimeError("Web search not initialized. Call init_web_search first.")

    top_k = _effective_top_k()
    normalized_query = query.strip()
    # ``top_k`` is in the cache key already, so a concentrated fetch does not collide
    # with a base-depth fetch of the same query.
    cache_key = f"web:{normalized_query.lower()}:{top_k}"
    if _web_cache is not None:
        try:
            cached = _web_cache.get(cache_key)
        except Exception:
            cached = None
        if isinstance(cached, list):
            return [r for r in cached if isinstance(r, dict)]

    # Walk the chain in configured order and stop at the first provider that yields
    # anything. Unmetered providers come first (see the ordering note above), so a
    # query that SearXNG can answer never spends a Serper credit or a Tavily request.
    search_results: List[Dict[str, str]] = []
    for name in _providers:
        provider = _PROVIDERS.get(name)
        if provider is None:  # pragma: no cover — init_web_search rejects these
            continue
        try:
            search_results = await provider(query, top_k)
        except Exception as e:  # noqa: BLE001 — one dead provider must not fail the hop
            logger.info(
                "[web_search] provider %r failed (%s: %s); trying next",
                name,
                type(e).__name__,
                str(e)[:120],
            )
            continue
        if search_results:
            break
    if not search_results:
        # Previously this returned empty in silence, so a fully dead chain was
        # indistinguishable from a query with no hits. It is the single most
        # expensive failure this module has (every hop retrieves nothing), so it
        # warns rather than debugs.
        logger.warning(
            "[web_search] no results from any provider in chain %s for %r",
            " → ".join(_providers) or "(empty)",
            query[:60],
        )

    if _web_config.crawl_full_text:
        concurrency = max(1, int(_web_config.max_crawl_requests_per_second))
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_crawl(result: Dict[str, str], client: httpx.AsyncClient):
            # Tavily and Wikipedia return page text inline, so re-fetching it would
            # spend exactly the latency those providers were used to avoid.
            if result.get("full_text"):
                return result["full_text"]
            url = result.get("url") or ""
            if not url:
                return ""
            async with semaphore:
                return await _crawl_page(url, client)

        async with httpx.AsyncClient() as client:
            tasks = [bounded_crawl(r, client) for r in search_results]
            full_texts = await asyncio.gather(*tasks)

        for r, text in zip(search_results, full_texts):
            r["full_text"] = text

    session = _get_web_session()
    deduped: List[Dict[str, str]] = []
    for r in search_results:
        url = str(r.get("url", "")).strip()
        if url:
            if url in session.visited:
                continue
            session.visited.add(url)
        deduped.append(
            {
                "title": str(r.get("title", "")),
                "url": url,
                "snippet": str(r.get("snippet", "")),
                "full_text": str(r.get("full_text", "")),
            }
        )

    if _web_cache is not None:
        try:
            _web_cache.set(cache_key, deduped)
        except Exception:
            pass

    return deduped
