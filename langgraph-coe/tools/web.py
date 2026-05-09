import os
import re
import httpx
import asyncio
from typing import List, Dict
from bs4 import BeautifulSoup

from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

from ..config import WebSearchConfig

_web_search_instance = None
_web_config = None

def init_web_search(config: WebSearchConfig):
    global _web_search_instance, _web_config
    _web_config = config
    api_key = config.api_key or os.environ.get("SERPER_API_KEY")
    
    if api_key:
        _web_search_instance = GoogleSerperAPIWrapper(serper_api_key=api_key)
    else:
        # Fallback to DuckDuckGo if no API key is provided
        _web_search_instance = DuckDuckGoSearchAPIWrapper(max_results=config.top_k)

async def _crawl_page(url: str, client: httpx.AsyncClient) -> str:
    """Crawl a single web page and extract text content."""
    try:
        response = await client.get(url, timeout=10.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        def _parse(html):
            soup = BeautifulSoup(html, "lxml")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return re.sub(r'\n{3,}', '\n\n', text)
            
        return await asyncio.to_thread(_parse, response.text)
    except Exception as e:
        return ""

@tool
async def web_search(query: str) -> List[str]:
    """Search the web for up-to-date information on a specific query."""
    if not _web_search_instance or not _web_config:
        raise RuntimeError("Web search not initialized. Call init_web_search first.")
        
    search_results: List[Dict[str, str]] = []
    
    if isinstance(_web_search_instance, GoogleSerperAPIWrapper):
        # We need async execution for the sync results method
        res = await asyncio.to_thread(_web_search_instance.results, query)
        organic = res.get("organic", [])[:_web_config.top_k]
        for item in organic:
            search_results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
    else:
        # DuckDuckGo
        res = await asyncio.to_thread(_web_search_instance.results, query, _web_config.top_k)
        if isinstance(res, list):
            for item in res:
                search_results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
        else:
            search_results.append({"title": "", "snippet": str(res), "link": ""})
            
    if _web_config.crawl_full_text:
        # Concurrency limit roughly approximating max requests/sec. 
        concurrency = max(1, int(_web_config.max_crawl_requests_per_second))
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_crawl(url: str, client: httpx.AsyncClient):
            if not url:
                return ""
            async with semaphore:
                return await _crawl_page(url, client)
                
        async with httpx.AsyncClient() as client:
            tasks = [bounded_crawl(r["link"], client) for r in search_results]
            full_texts = await asyncio.gather(*tasks)
            
        for r, text in zip(search_results, full_texts):
            r["full_text"] = text
            
    final_docs = []
    for r in search_results:
        parts = [p for p in [r.get("title"), r.get("snippet"), r.get("full_text")] if p]
        final_docs.append("\n".join(parts))
        
    return final_docs
