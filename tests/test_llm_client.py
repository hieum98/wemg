"""Tests for LLMClient using real API (no mocks). Skip when API_KEY not set."""

import os
import pytest

from conftest import get_llm_env, requires_llm
from wemg.llm.client import LLMClient


@pytest.mark.requires_llm
def test_llm_client_generate_real():
    """Real API: generate returns one choice with content."""
    requires_llm()
    api_key, url, model = get_llm_env()
    if not url:
        pytest.skip("LLM_URL or OPENAI_BASE_URL not set")
    client = LLMClient(
        model_name=model,
        url=url,
        api_key=api_key,
        concurrency=1,
        max_retries=1,
        max_tokens=10,
        temperature=0,
    )
    try:
        idx, choices = client.generate(
            0,
            [{"role": "user", "content": "Say exactly: ok"}],
            max_tokens=10,
        )
        assert idx == 0
        assert isinstance(choices, list)
        assert len(choices) >= 1
        assert "output" in choices[0]
        assert choices[0]["is_valid"] is True
    finally:
        client.close()


@pytest.mark.requires_llm
def test_llm_client_get_embeddings_real():
    """Real API: get_embeddings returns list of floats (requires embedding model URL)."""
    requires_llm()
    api_key, url, _ = get_llm_env()
    embed_model = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"
    embed_url = url
    client = LLMClient(
        model_name=embed_model,
        url=embed_url,
        api_key=api_key,
        is_embedding=True,
    )
    try:
        result = client.get_embeddings(["hello"])
        if result is None:
            pytest.skip("Embedding API not available or model not found")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(x, (int, float)) for x in result[0])
    finally:
        client.close()


def test_llm_client_init_no_cache():
    """Client initializes without cache when cache_config disabled."""
    client = LLMClient(
        model_name="test",
        url="http://localhost/v1",
        api_key="test-key",
        cache_config={"enabled": False},
    )
    assert client._cache is None
    assert client._use_cache is False
    client.close()
