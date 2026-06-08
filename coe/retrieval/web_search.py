"""Web search via Serper API with DuckDuckGo fallback and page crawling."""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pydantic
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Max requests per second when crawling pages; stay under this to avoid destination rate limits.
DEFAULT_MAX_CRAWL_REQUESTS_PER_SECOND = 2.0


class _InMemoryCache:
    """Thread-safe in-memory cache with optional TTL."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}  # key -> (value, expire_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if expire_at is not None and time.monotonic() >= expire_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int]) -> None:
        expire_at = (time.monotonic() + ttl) if ttl is not None else None
        with self._lock:
            self._store[key] = (value, expire_at)

    def __len__(self) -> int:
        return len(self._store)


class WebSearchResult(pydantic.BaseModel):
    title: str = ""
    link: str = ""
    snippet: str = ""
    full_text: str = ""


class KGEntity(pydantic.BaseModel):
    title: str = ""
    description: str = ""
    url: str = ""
    attributes: Dict[str, str] = pydantic.Field(default_factory=dict)


class WebSearchOutput(pydantic.BaseModel):
    query: str
    results: List[WebSearchResult] = pydantic.Field(default_factory=list)
    is_success: bool = False


def _serper_search(query: str, api_key: str) -> tuple[List[WebSearchResult], Optional[KGEntity]]:
    """Search using Serper API."""
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    response = requests.post("https://google.serper.dev/search", headers=headers, data=payload, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = [
        WebSearchResult(
            title=item.get("title", ""),
            link=item.get("link", ""),
            snippet=item.get("snippet", ""),
            full_text=item.get("snippet", ""),
        )
        for item in data.get("organic", [])
    ]

    kg = data.get("knowledgeGraph")
    kg_entity = KGEntity(
        title=kg.get("title", ""),
        description=kg.get("description", ""),
        url=kg.get("url", ""),
        attributes=kg.get("attributes", {}),
    ) if kg else None

    if not results:
        raise ValueError("No search results returned")
    return results, kg_entity


def _ddgs_search(query: str) -> tuple[List[WebSearchResult], None]:
    """Fallback search using DuckDuckGo."""
    from ddgs import DDGS
    ddgs = DDGS()
    response = ddgs.text(query, max_results=10)
    results = [
        WebSearchResult(
            title=item.get("title", ""),
            link=item.get("href", ""),
            snippet=item.get("body", ""),
            full_text=item.get("body", ""),
        )
        for item in response
    ]
    if not results:
        raise ValueError("No search results returned")
    return results, None


def crawl_page(url: str, timeout: int = 10) -> str:
    """Crawl a single web page and extract text content."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    except Exception as e:
        logger.warning("Failed to crawl %s; falling back to empty text: %s", url, e)
        return ""


_crawl_rate_lock = threading.Lock()
_crawl_last_request_time: List[float] = [0.0]


def _crawl_page_rate_limited(
    url: str,
    timeout: int,
    max_requests_per_second: float,
) -> str:
    """Crawl a single page after waiting to respect the max requests-per-second limit."""
    with _crawl_rate_lock:
        now = time.monotonic()
        min_interval = 1.0 / max_requests_per_second
        elapsed = now - _crawl_last_request_time[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _crawl_last_request_time[0] = time.monotonic()
    return crawl_page(url, timeout=timeout)


def crawl_pages(
    urls: List[str],
    max_workers: int = 8,
    max_requests_per_second: Optional[float] = None,
) -> List[str]:
    """Crawl multiple web pages with rate limiting to avoid destination rate limits.

    Args:
        urls: URLs to crawl.
        max_workers: Max concurrent crawl workers.
        max_requests_per_second: Max crawl rate (requests per second). If None, uses
            DEFAULT_MAX_CRAWL_REQUESTS_PER_SECOND. If the effective rate from
            max_workers would exceed this, crawling is throttled to this limit.
    """
    rate = max_requests_per_second if max_requests_per_second is not None else DEFAULT_MAX_CRAWL_REQUESTS_PER_SECOND
    rate = max(0.1, min(rate, 10.0))
    workers = max(1, min(max_workers, max(1, int(rate))))
    if max_workers > workers:
        logger.debug(
            "Crawl max_workers reduced from %s to %s to respect max_requests_per_second=%.1f",
            max_workers, workers, rate,
        )

    def _crawl_one(url: str) -> str:
        return _crawl_page_rate_limited(url, timeout=10, max_requests_per_second=rate)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_crawl_one, urls))


class WebSearchTool:
    """Web search with Serper API, DuckDuckGo fallback, and page crawling.

    Args:
        api_key: Serper API key (falls back to SERPER_API_KEY env var).
        max_crawl_requests_per_second: Rate limit for page crawling.
        query_cache_ttl: Seconds to cache search results per query. ``None`` disables
            query caching; ``0`` caches forever.
        url_cache_ttl: Seconds to cache crawled page content per URL. ``None`` disables
            URL caching; ``0`` caches forever.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_crawl_requests_per_second: Optional[float] = None,
        query_cache_ttl: Optional[int] = 3600,
        url_cache_ttl: Optional[int] = 3600,
    ):
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        self.max_crawl_requests_per_second = (
            max_crawl_requests_per_second
            if max_crawl_requests_per_second is not None
            else DEFAULT_MAX_CRAWL_REQUESTS_PER_SECOND
        )
        # None means caching disabled; convert 0 -> None for _InMemoryCache (no expiry).
        self._query_cache: Optional[_InMemoryCache] = _InMemoryCache() if query_cache_ttl is not None else None
        self._query_ttl: Optional[int] = query_cache_ttl if query_cache_ttl != 0 else None
        self._url_cache: Optional[_InMemoryCache] = _InMemoryCache() if url_cache_ttl is not None else None
        self._url_ttl: Optional[int] = url_cache_ttl if url_cache_ttl != 0 else None

    def _cached_search(self, query: str) -> Optional[List[WebSearchResult]]:
        if self._query_cache is None:
            return None
        raw = self._query_cache.get(query)
        if raw is None:
            return None
        logger.debug("Query cache hit: %r", query)
        return [WebSearchResult(**r) for r in raw]

    def _cache_search(self, query: str, results: List[WebSearchResult]) -> None:
        if self._query_cache is not None:
            self._query_cache.set(query, [r.model_dump() for r in results], self._query_ttl)

    def _cached_crawl(self, url: str) -> Optional[str]:
        if self._url_cache is None:
            return None
        cached = self._url_cache.get(url)
        if cached is None:
            return None
        logger.debug("URL cache hit: %s", url)
        return cached

    def _cache_crawl(self, url: str, content: str) -> None:
        if self._url_cache is not None:
            self._url_cache.set(url, content, self._url_ttl)

    def search(self, query: str, top_k: int = 5, crawl_full_text: bool = True) -> WebSearchOutput:
        """Synchronous search."""
        cached_results = self._cached_search(query)
        if cached_results is not None:
            top_results = cached_results[:top_k]
        else:
            try:
                results, kg = _serper_search(query, self.api_key)
            except Exception as e:
                logger.warning(f"Serper API failed: {e}. Falling back to DDGS.")
                try:
                    results, kg = _ddgs_search(query)
                except Exception as e2:
                    logger.error(f"DDGS also failed: {e2}. Returning empty results.")
                    return WebSearchOutput(query=query)
            self._cache_search(query, results)
            top_results = results[:top_k]

        if crawl_full_text:
            uncached_indices = []
            uncached_urls = []
            for i, result in enumerate(top_results):
                cached_text = self._cached_crawl(result.link)
                if cached_text is not None:
                    result.full_text = cached_text
                else:
                    uncached_indices.append(i)
                    uncached_urls.append(result.link)

            if uncached_urls:
                full_texts = crawl_pages(
                    uncached_urls,
                    max_requests_per_second=self.max_crawl_requests_per_second,
                )
                for idx, url, text in zip(uncached_indices, uncached_urls, full_texts):
                    if text:
                        top_results[idx].full_text = text
                        self._cache_crawl(url, text)

        return WebSearchOutput(query=query, results=top_results, is_success=True)

    async def asearch(self, query: str, top_k: int = 5, crawl_full_text: bool = True) -> WebSearchOutput:
        """Async search (runs sync search in thread to avoid blocking)."""
        import asyncio
        return await asyncio.to_thread(self.search, query, top_k, crawl_full_text)
