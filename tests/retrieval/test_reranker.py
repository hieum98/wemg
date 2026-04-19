"""Reranker against real /rerank endpoint (from config)."""

import pytest

from tests.conftest import requires_llm_credentials, requires_reranker_config
from wemg.llm.client import LLMClient
from wemg.retrieval.reranker import Reranker


@pytest.mark.requires_llm
@pytest.mark.requires_reranker
def test_reranker_empty_documents(wemg_config):
    requires_reranker_config(wemg_config)
    requires_llm_credentials(wemg_config)
    r = wemg_config.reranker
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=r.model_name,
        url=r.url,
        api_key=r.api_key,
        concurrency=r.concurrency,
        max_retries=wemg_config.llm.max_retries,
        timeout=gen.timeout,
        temperature=gen.temperature,
        n=gen.n,
        top_p=gen.top_p,
        min_p=gen.min_p,
        max_tokens=gen.max_tokens,
        max_input_tokens=gen.max_input_tokens,
        top_k=gen.top_k,
        presence_penalty=gen.presence_penalty,
        repetition_penalty=gen.repetition_penalty,
        enable_thinking=gen.enable_thinking,
        random_seed=gen.random_seed,
        cache_config={"enabled": False},
    )
    try:
        rr = Reranker(client=client, top_k=r.top_k)
        assert rr.rerank("What is 2+2?", []) == []
    finally:
        client.close()


@pytest.mark.requires_llm
@pytest.mark.requires_reranker
def test_reranker_orders_by_relevance(wemg_config):
    requires_reranker_config(wemg_config)
    requires_llm_credentials(wemg_config)
    r = wemg_config.reranker
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=r.model_name,
        url=r.url,
        api_key=r.api_key,
        concurrency=r.concurrency,
        max_retries=wemg_config.llm.max_retries,
        timeout=gen.timeout,
        temperature=gen.temperature,
        n=gen.n,
        top_p=gen.top_p,
        min_p=gen.min_p,
        max_tokens=gen.max_tokens,
        max_input_tokens=gen.max_input_tokens,
        top_k=gen.top_k,
        presence_penalty=gen.presence_penalty,
        repetition_penalty=gen.repetition_penalty,
        enable_thinking=gen.enable_thinking,
        random_seed=gen.random_seed,
        cache_config={"enabled": False},
    )
    try:
        rr = Reranker(client=client, top_k=2)
        docs = [
            "The Eiffel Tower is in Paris, France.",
            "Penguins live in Antarctica.",
            "Paris is the capital of France.",
        ]
        out = rr.rerank("Capital of France", docs, top_k=2)
        assert len(out) == 2
        assert all(isinstance(s, str) for s in out)
        joined = " ".join(out).lower()
        assert "france" in joined or "paris" in joined
    finally:
        client.close()


@pytest.mark.requires_llm
@pytest.mark.requires_reranker
def test_reranker_top_k_override(wemg_config):
    requires_reranker_config(wemg_config)
    requires_llm_credentials(wemg_config)
    r = wemg_config.reranker
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=r.model_name,
        url=r.url,
        api_key=r.api_key,
        concurrency=r.concurrency,
        max_retries=wemg_config.llm.max_retries,
        timeout=gen.timeout,
        temperature=gen.temperature,
        n=gen.n,
        top_p=gen.top_p,
        min_p=gen.min_p,
        max_tokens=gen.max_tokens,
        max_input_tokens=gen.max_input_tokens,
        top_k=gen.top_k,
        presence_penalty=gen.presence_penalty,
        repetition_penalty=gen.repetition_penalty,
        enable_thinking=gen.enable_thinking,
        random_seed=gen.random_seed,
        cache_config={"enabled": False},
    )
    try:
        rr = Reranker(client=client, top_k=10)
        docs = ["a", "b", "c"]
        out = rr.rerank("query", docs, top_k=1)
        assert len(out) == 1
    finally:
        client.close()
