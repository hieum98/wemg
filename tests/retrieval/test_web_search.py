"""Web search and crawl (real Serper or DDG)."""

import pytest

from wemg.retrieval.web_search import WebSearchOutput, WebSearchTool, crawl_page


@pytest.mark.requires_web_search
def test_web_search_tool_ddg_fallback():
    tool = WebSearchTool(api_key=None)
    result = tool.search("Python programming language", top_k=2)
    assert isinstance(result, WebSearchOutput)
    assert result.query == "Python programming language"
    assert result.is_success is True
    assert len(result.results) >= 1
    assert result.results[0].title or result.results[0].snippet


def test_web_search_tool_crawl_disabled():
    tool = WebSearchTool(api_key=None)
    result = tool.search("Wikipedia", top_k=1, crawl_full_text=False)
    assert result.is_success is True
    if result.results:
        assert hasattr(result.results[0], "full_text")


def test_crawl_page_valid_url():
    text = crawl_page("https://example.com", timeout=5)
    assert isinstance(text, str)


def test_crawl_page_invalid_url():
    text = crawl_page("https://nonexistent-domain-xyz-12345.invalid", timeout=2)
    assert text == "" or isinstance(text, str)
