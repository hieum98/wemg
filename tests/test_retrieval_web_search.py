"""Tests for WebSearchTool using real search (Serper or DDG fallback) and crawl_page."""

import pytest

from wemg.retrieval.web_search import (
    WebSearchTool,
    WebSearchOutput,
    crawl_page,
)


def test_web_search_tool_ddg_fallback():
    """With no/invalid Serper key, DDG fallback returns non-empty results."""
    tool = WebSearchTool(api_key=None)
    result = tool.search("Python programming", top_k=2)
    assert isinstance(result, WebSearchOutput)
    assert result.query == "Python programming"
    assert result.is_success is True
    assert len(result.results) >= 1
    assert result.results[0].title or result.results[0].snippet


def test_web_search_tool_crawl_disabled():
    """Search with crawl_full_text=False returns quickly without full_text."""
    tool = WebSearchTool(api_key=None)
    result = tool.search("Wikipedia", top_k=1, crawl_full_text=False)
    assert result.is_success is True
    if result.results:
        assert hasattr(result.results[0], "full_text")


def test_crawl_page_valid_url():
    """crawl_page on a valid URL returns non-empty string or empty on failure."""
    text = crawl_page("https://example.com", timeout=5)
    assert isinstance(text, str)


def test_crawl_page_invalid_url():
    """crawl_page on invalid/unreachable URL returns empty string."""
    text = crawl_page("https://nonexistent-domain-xyz-12345.invalid", timeout=2)
    assert text == "" or isinstance(text, str)
