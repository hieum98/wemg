import json
import logging
import os
import re
from typing import List, Optional, Dict, Type, Union
import pydantic
import nest_asyncio
import requests
from langchain_core.tools import BaseTool
from langchain_community.document_loaders import WebBaseLoader

nest_asyncio.apply()
logger = logging.getLogger(__name__)


class WebSearchResults(pydantic.BaseModel):
    title: str = pydantic.Field(..., description="The title of the search result.")
    link: str = pydantic.Field(..., description="The URL of the search result.")
    snippet: str = pydantic.Field(..., description="A brief snippet from the search result.")
    full_text: str = pydantic.Field(..., description="The full text content of the search result.")

class KGEntity(pydantic.BaseModel):
    title: str = pydantic.Field(..., description="The title of the knowledge graph entity.")
    description: str = pydantic.Field(..., description="A brief description of the entity.")
    url: str = pydantic.Field(..., description="The URL to more information about the entity.")
    attributes: Dict[str, str] = pydantic.Field(..., description="Additional attributes of the entity.")

class WebSearchOutput(pydantic.BaseModel):
    query: str = pydantic.Field(..., description="The search query used.")
    results: list[WebSearchResults] = pydantic.Field(..., description="A list of search results.")
    is_success: bool = pydantic.Field(..., description="Indicates if the search was successful.")


class SerperAPIWrapper:
    url = "https://google.serper.dev/search"
    api_key: str = "your_serper_api_key"

    def run(self, query: str):
        try:
            payload = json.dumps({
            "q": query
            })
            headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", self.url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            response_json = response.json()
            all_web_results = []
            for item in response_json.get("organic", []):
                web_result = WebSearchResults(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    full_text=item.get("snippet", "")  # Assuming full_text is same as snippet for this example
                )
                all_web_results.append(web_result)
            
            entity = response_json.get("knowledgeGraph", {})
            if entity:
                kg_entity = KGEntity(
                    title=entity.get("title", ""),
                    description=entity.get("description", ""),
                    url=entity.get("url", ""),
                    attributes=entity.get("attributes", {})
                )
            else:
                kg_entity = None
            
            if len(all_web_results) == 0:
                logger.warning(f"No results found for query: {query}")
                raise ValueError("No search results returned")
            
            logger.info(f"Web search completed for query: {query} with {len(all_web_results)} results.")
            return all_web_results, kg_entity
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to access Serper API URL: {e}")
            raise
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Serper API response: {e}")
            raise
    

class DDGSAPIWrapper:
    def run(self, query: str):
        try:
            from ddgs import DDGS

            ddgs = DDGS()
            response = ddgs.text(query, max_results=10)
            all_web_results = []
            for item in response:
                web_result = WebSearchResults(
                    title=item.get("title", ""),
                    link=item.get("href", ""),
                    snippet=item.get("body", ""),
                    full_text=item.get("body", "")  # Assuming full_text is same as body for this example
                )
                all_web_results.append(web_result)
            
            if len(all_web_results) == 0:
                logger.warning(f"No results found for query: {query}")
                raise ValueError("No search results returned")
            
            logger.info(f"Web search completed for query: {query} with {len(all_web_results)} results.")
            return all_web_results, None
        except Exception as e:
            logger.error(f"Failed to access DDGS API: {e}")
            raise


class WebSearchTool(BaseTool):
    """Tool that performs a web search and returns structured results."""

    name: str = "web_search_tool"
    description: str = (
        "A tool that performs web searches to retrieve current information. "
        "Input should be a search query string."
    )
    args_schema: Type[pydantic.BaseModel] = pydantic.create_model(
        "WebSearchInput",
        query=(str, pydantic.Field(..., description="The search query string.")),
    )
    serper_api_key: str = os.getenv("SERPER_API_KEY", "your-api-key")

    @staticmethod
    def crawl_web_pages(urls: Union[str, List[str]]) -> List[str]:
        """Crawl web pages to extract their full text content."""
        if isinstance(urls, str):
            urls = [urls]
        loader = WebBaseLoader(urls, requests_per_second=16, continue_on_failure=True)
        documents = loader.load()
        page_contents = [doc.page_content for doc in documents]
        # simple cleanup
        page_contents = [re.sub(r'\n+', '\n', content) for content in page_contents]
        if len(urls) == 1:
            return page_contents[0]
        return page_contents

    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[object] = None,
    ) -> WebSearchOutput:
        """Perform the web search and return structured results."""
        logger.info(f"Performing web search for query: {query}")
        # Try using Serper API first and fallback to DDGS if it fails, if all fails, return warning message with empty results
        try:
            serper_wrapper = SerperAPIWrapper()
            serper_wrapper.api_key = self.serper_api_key
            all_web_results, kg_entity = serper_wrapper.run(query)
        except Exception as e:
            logger.warning(f"Serper API failed with error: {e}. Falling back to DDGS.")
            try:
                ddgs_wrapper = DDGSAPIWrapper()
                all_web_results, kg_entity = ddgs_wrapper.run(query)
            except Exception as e2:
                logger.error(f"DDGS also failed with error: {e2}. Returning empty results.")
                return WebSearchOutput(
                    query=query,
                    results=[],
                    is_success=False
                )
        # Crawl full text for top_k results
        top_results = all_web_results[:top_k]
        urls_to_crawl = [result.link for result in top_results]
        full_texts = self.crawl_web_pages(urls_to_crawl)
        for i, result in enumerate(top_results):
            result.full_text = full_texts[i]
        return WebSearchOutput(
            query=query,
            results=top_results,
            is_success=True
        )
    
    @staticmethod
    async def _acrawl_web_pages(urls: Union[str, List[str]]) -> List[str]:
        """Async crawl web pages to extract their full text content."""
        if isinstance(urls, str):
            urls = [urls]
        loader = WebBaseLoader(urls, requests_per_second=8, continue_on_failure=True)
        documents = [doc async for doc in loader.alazy_load()]
        page_contents = [doc.page_content for doc in documents]
        # simple cleanup
        page_contents = [re.sub(r'\n+', '\n', content) for content in page_contents]
        return page_contents

    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[object] = None,
    ) -> WebSearchOutput:
        """Async perform the web search and return structured results."""
        logger.info(f"Performing async web search for query: {query}")
        # Try using Serper API first and fallback to DDGS if it fails
        try:
            serper_wrapper = SerperAPIWrapper()
            serper_wrapper.api_key = self.serper_api_key
            all_web_results, kg_entity = serper_wrapper.run(query)
        except Exception as e:
            logger.warning(f"Serper API failed with error: {e}. Falling back to DDGS.")
            try:
                ddgs_wrapper = DDGSAPIWrapper()
                all_web_results, kg_entity = ddgs_wrapper.run(query)
            except Exception as e2:
                logger.error(f"DDGS also failed with error: {e2}. Returning empty results.")
                return WebSearchOutput(
                    query=query,
                    results=[],
                    is_success=False
                )
        # Crawl full text for top_k results asynchronously
        top_results = all_web_results[:top_k]
        urls_to_crawl = [result.link for result in top_results]
        full_texts = await self._acrawl_web_pages(urls_to_crawl)
        for i, result in enumerate(top_results):
            result.full_text = full_texts[i]
        return WebSearchOutput(
            query=query,
            results=top_results,
            is_success=True
        )
        
        
if __name__=='__main__':
    serper_api_key = "your_serper_api_key"

    web_search = WebSearchTool(serper_api_key=serper_api_key)
    result = web_search.invoke({"query": "What is the capital of France?"})
    breakpoint()
    print(f"Query: {result.query}")
    print(f"Success: {result.is_success}")
    print(f"Number of results: {len(result.results)}")
    breakpoint()
    for i, r in enumerate(result.results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {r.title}")
        print(f"Link: {r.link}")
        print(f"Snippet: {r.snippet[:200]}..." if len(r.snippet) > 200 else f"Snippet: {r.snippet}")
        print(f"Full text length: {len(r.full_text)} chars")
            
