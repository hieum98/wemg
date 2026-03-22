"""Optional Redis-backed LLM cache (real Redis)."""

import pytest

from tests.conftest import requires_llm_credentials
from wemg.llm.client import LLMClient


@pytest.mark.requires_llm
@pytest.mark.requires_redis
def test_llm_client_cache_roundtrip_when_redis_available(wemg_config):
    requires_llm_credentials(wemg_config)
    c = wemg_config.cache
    cache_config = {
        "enabled": True,
        "host": c.host,
        "port": c.port,
        "db": c.db,
        "password": c.password,
        "prefix": f"{c.prefix}_pytest",
        "ttl": c.ttl,
    }
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        concurrency=1,
        max_retries=1,
        max_tokens=32,
        temperature=0.0,
        cache_config=cache_config,
        timeout=min(gen.timeout, 120),
    )
    if not client._use_cache or client._cache is None:
        pytest.skip("Redis cache not active (unreachable or disabled after connect attempt)")
    try:
        messages = [{"role": "user", "content": "Reply with exactly: cache_test_ok"}]
        _, first = client.generate(0, messages, max_tokens=16, use_cache=True)
        _, second = client.generate(0, messages, max_tokens=16, use_cache=True)
        assert first and first[0].get("is_valid")
        assert second and second[0].get("is_valid")
        assert first[0]["output"] == second[0]["output"]
    finally:
        client.close()
