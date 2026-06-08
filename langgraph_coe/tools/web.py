from __future__ import annotations

import asyncio
import os
import re
from contextvars import ContextVar
from typing import Dict, List

import httpx
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool

from ..config import WebSearchConfig

_web_search_instance = None
_web_config = None
_web_cache = None


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
    global _web_search_instance, _web_config
    _web_config = config
    api_key = config.api_key or os.environ.get("SERPER_API_KEY")

    if api_key:
        _web_search_instance = GoogleSerperAPIWrapper(serper_api_key=api_key)
    else:
        _web_search_instance = DuckDuckGoSearchAPIWrapper(max_results=config.top_k)


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


@tool
async def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web for up-to-date information on a specific query."""
    if not _web_search_instance or not _web_config:
        raise RuntimeError("Web search not initialized. Call init_web_search first.")

    normalized_query = query.strip()
    cache_key = f"web:{normalized_query.lower()}:{_web_config.top_k}"
    if _web_cache is not None:
        try:
            cached = _web_cache.get(cache_key)
        except Exception:
            cached = None
        if isinstance(cached, list):
            return [r for r in cached if isinstance(r, dict)]

    search_results: List[Dict[str, str]] = []

    if isinstance(_web_search_instance, GoogleSerperAPIWrapper):
        res = await asyncio.to_thread(_web_search_instance.results, query)
        organic = res.get("organic", [])[: _web_config.top_k]
        for item in organic:
            search_results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "full_text": "",
                }
            )
    else:
        res = await asyncio.to_thread(
            _web_search_instance.results, query, _web_config.top_k
        )
        if isinstance(res, list):
            for item in res:
                search_results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        "full_text": "",
                    }
                )
        else:
            search_results.append(
                {"title": "", "snippet": str(res), "url": "", "full_text": ""}
            )

    if _web_config.crawl_full_text:
        concurrency = max(1, int(_web_config.max_crawl_requests_per_second))
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_crawl(url: str, client: httpx.AsyncClient):
            if not url:
                return ""
            async with semaphore:
                return await _crawl_page(url, client)

        async with httpx.AsyncClient() as client:
            tasks = [bounded_crawl(r["url"], client) for r in search_results]
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
