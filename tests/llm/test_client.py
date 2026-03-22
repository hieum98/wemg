"""Tests for LLMClient using real API (no mocks)."""

import pytest

from tests.conftest import requires_llm_credentials
from wemg.llm.client import LLMClient


@pytest.mark.requires_llm
def test_llm_client_generate_real(wemg_config):
    requires_llm_credentials(wemg_config)
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        concurrency=1,
        max_retries=1,
        max_tokens=64,
        temperature=0,
        cache_config={"enabled": False},
        timeout=min(gen.timeout, 120),
    )
    try:
        idx, choices = client.generate(
            0,
            [{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=32,
        )
        assert idx == 0
        assert isinstance(choices, list)
        assert len(choices) >= 1
        assert "output" in choices[0]
        assert choices[0]["is_valid"] is True
    finally:
        client.close()


@pytest.mark.requires_llm
def test_llm_client_get_embeddings_real(wemg_config, live_embedding_client):
    requires_llm_credentials(wemg_config)
    result = live_embedding_client.get_embeddings(["hello"])
    if result is None:
        pytest.skip("Embedding API not available or model not found")
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, (int, float)) for x in result[0])


def test_llm_client_init_no_cache():
    client = LLMClient(
        model_name="test",
        url="http://localhost/v1",
        api_key="test-key",
        cache_config={"enabled": False},
    )
    assert client._cache is None
    assert client._use_cache is False
    client.close()


@pytest.mark.requires_llm
def test_llm_client_batch_generate_dedup(wemg_config):
    requires_llm_credentials(wemg_config)
    client = LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        concurrency=2,
        max_retries=1,
        max_tokens=32,
        temperature=0,
        cache_config={"enabled": False},
    )
    msg = [{"role": "user", "content": "Say: hi"}]
    try:
        out = client.batch_generate([msg, msg], max_tokens=16)
        assert len(out) == 2
        assert out[0] == out[1]
    finally:
        client.close()
