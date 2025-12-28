"""
Comprehensive tests for the agents/tools module (WebSearch).

These are real integration tests that call actual web search APIs.
Note: Wikidata tests have been moved to test_wikidata_refactored.py
"""
import os
import pytest
import asyncio
from typing import List

from wemg.agents.tools.web_search import (
    WebSearchTool, 
    WebSearchOutput,
    DDGSAPIWrapper,
    SerperAPIWrapper
)
# Wikidata tests have been moved to test_wikidata_refactored.py
# This file now focuses on WebSearch tool tests
from wemg.agents.base_llm_agent import BaseLLMAgent


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your-serper-api-key")

class TestWebSearchTool:
    """Test suite for WebSearchTool functionality."""
    
    @pytest.fixture
    def ddgs_search_tool(self):
        """Create a WebSearchTool that will use DuckDuckGo backend.
        
        Since WebSearchTool tries Serper first and falls back to DDGS,
        we use an invalid serper_api_key to force it to use DDGS.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
        )
    
    @pytest.mark.slow
    def test_ddgs_basic_search(self, ddgs_search_tool):
        """Test basic web search with DuckDuckGo."""
        query = "Python programming language"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        assert output.query == query
        assert len(output.results) > 0
        
        print(f"✓ DDGS Basic Search")
        print(f"  Query: {query}")
        print(f"  Results: {len(output.results)}")
        for i, result in enumerate(output.results[:3]):
            print(f"  {i+1}. {result.title[:60]}...")
    
    @pytest.mark.slow
    def test_ddgs_search_with_full_text(self, ddgs_search_tool):
        """Test web search with full text extraction."""
        query = "machine learning applications"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 3})
        )
        
        assert output.is_success is True
        
        # Check that results have content
        for result in output.results:
            assert result.title is not None
            assert result.link is not None
            assert result.snippet is not None
        
        print(f"✓ DDGS Search with full text")
        print(f"  Retrieved {len(output.results)} results with full text")
    
    @pytest.mark.slow
    def test_ddgs_technical_query(self, ddgs_search_tool):
        """Test search with technical query."""
        query = "transformer architecture attention mechanism"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        assert len(output.results) > 0
        
        print(f"✓ DDGS Technical Query")
        print(f"  Query: {query}")
        print(f"  Found {len(output.results)} results")
    
    @pytest.mark.slow
    def test_ddgs_factual_query(self, ddgs_search_tool):
        """Test search for factual information."""
        query = "population of Tokyo Japan 2024"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        
        print(f"✓ DDGS Factual Query")
        print(f"  Query: {query}")
        for result in output.results[:2]:
            print(f"    - {result.snippet[:100]}...")


class TestWebSearchIntegration:
    """Integration tests combining web search with LLM."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool that will use DuckDuckGo backend.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
        )
    
    @pytest.mark.slow
    def test_search_and_extract(self, llm_agent, web_search_tool):
        """Test searching and extracting information."""
        from wemg.agents import roles
        from wemg.runners.procedures.base_role_execution import execute_role
        
        # Search for information
        query = "When was the first smartphone invented"
        search_output: WebSearchOutput = asyncio.run(
            web_search_tool.ainvoke({"query": query, "top_k": 3})
        )
        
        assert search_output.is_success
        
        # Extract from search results
        if search_output.results:
            raw_data = "\n\n".join([
                f"{r.title}\n{r.snippet}\n{r.full_text}" 
                for r in search_output.results[:2]
            ])
            
            extract_input = roles.extractor.ExtractionInput(
                question=query,
                raw_data=raw_data
            )
            
            results, _ = asyncio.run(execute_role(
                llm_agent=llm_agent,
                role=roles.extractor.Extractor(),
                input_data=extract_input,
                n=1
            ))
            
            output: roles.extractor.ExtractionOutput = results[0]
            
            print(f"✓ Search and Extract Integration")
            print(f"  Query: {query}")
            print(f"  Search results: {len(search_output.results)}")
            print(f"  Relevant information: {output.relevant_information}")


class TestToolErrorHandling:
    """Test error handling in tools."""
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool that will use DuckDuckGo backend.
        
        Since WebSearchTool tries Serper first and falls back to DDGS,
        we use an invalid serper_api_key to force it to use DDGS.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
        )
    
    @pytest.mark.slow
    def test_search_empty_query(self, web_search_tool):
        """Test search with empty query."""
        try:
            output: WebSearchOutput = asyncio.run(
                web_search_tool.ainvoke({"query": "", "top_k": 5})
            )
            # May succeed with empty results or fail
            print(f"✓ Empty query handled: success={output.is_success}")
        except Exception as e:
            print(f"✓ Empty query raised expected error: {type(e).__name__}")
    
    # Wikidata error handling tests have been moved to test_wikidata_refactored.py


# Run tests with: pytest test_tools.py -v -s --tb=short
# Run slow tests: pytest test_tools.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
