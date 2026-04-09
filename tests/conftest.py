"""Shared fixtures and markers for WEMG real-system tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.helpers.bootstrap import corpus_paths_from_config_or_env, default_config_yaml, load_test_env

load_test_env()


def pytest_configure(config):
    markers = [
        ("requires_llm", "needs LLM URL and API key (from config.yaml + .env)"),
        ("requires_redis", "needs reachable Redis for LLM cache"),
        ("requires_corpus", "needs corpus_path + FAISS index on disk"),
        ("requires_wikidata", "needs live Wikidata/Wikipedia APIs"),
        ("requires_freebase", "needs live Freebase SPARQL endpoint"),
        ("requires_web_search", "needs web search (Serper or DDG)"),
        ("requires_reranker", "needs rerank HTTP endpoint from config"),
        ("slow_integration", "long-running real-system flows"),
        ("integration", "real external services"),
    ]
    for name, desc in markers:
        config.addinivalue_line("markers", f"{name}: {desc}")


# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root_path() -> Path:
    return default_config_yaml().parent.parent


@pytest.fixture(scope="session")
def wemg_config_path() -> Path:
    p = default_config_yaml()
    if not p.is_file():
        pytest.skip(f"Missing config YAML: {p}")
    return p


@pytest.fixture(scope="session")
def wemg_config(wemg_config_path: Path):
    from wemg.config import WEMGConfig

    return WEMGConfig.from_yaml(wemg_config_path)


@pytest.fixture
def wemg_config_overrides() -> List[str]:
    """Override in parametrized tests; default empty."""
    return []


@pytest.fixture
def wemg_config_with_overrides(wemg_config_path: Path, wemg_config_overrides: List[str]):
    from wemg.config import WEMGConfig

    return WEMGConfig.from_yaml(wemg_config_path, wemg_config_overrides or None)


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def get_llm_env():
    """(api_key, url, model) — prefers loaded WEMG config when possible."""
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("LLM_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    return api_key, url, model


def requires_llm_url():
    _, url, _ = get_llm_env()
    if not url:
        pytest.skip("LLM_URL or OPENAI_BASE_URL not set (after .env load)")


def requires_llm_credentials(wemg_config=None):
    """Skip if LLM cannot be called (need URL and api_key)."""
    if wemg_config is not None:
        if not wemg_config.llm.url:
            pytest.skip("config.llm.url missing")
        if not wemg_config.llm.api_key:
            pytest.skip("config.llm.api_key missing (set API_KEY or YAML)")
        return
    api_key, url, _ = get_llm_env()
    if not url:
        pytest.skip("LLM_URL or OPENAI_BASE_URL not set")
    if not api_key:
        pytest.skip("API_KEY or OPENAI_API_KEY not set")


def requires_corpus_paths(wemg_config):
    cpath, ipath = corpus_paths_from_config_or_env(wemg_config)
    if not cpath or not ipath:
        pytest.skip("corpus_path or index_path missing (config or CORPUS_PATH/INDEX_PATH)")
    from pathlib import Path as P

    if not P(ipath).is_file():
        pytest.skip(f"index file not found: {ipath}")


def requires_reranker_config(wemg_config):
    if not wemg_config.reranker.enabled:
        pytest.skip("reranker.enabled is false in config")
    if not wemg_config.reranker.url:
        pytest.skip("reranker.url missing")


# ---------------------------------------------------------------------------
# Legacy fixtures (used by minimal dict-based tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "chroma_db"


@pytest.fixture
def minimal_config_dict(wemg_config):
    """Small valid config; paths/embedder taken from session wemg_config when possible."""
    c = wemg_config.retriever.corpus
    return {
        "llm": {
            "model_name": wemg_config.llm.model_name,
            "url": wemg_config.llm.url,
            "api_key": wemg_config.llm.api_key,
            "concurrency": 2,
            "max_retries": 1,
            "generation": {
                "timeout": 120,
                "temperature": 0.0,
                "n": 1,
                "max_tokens": 512,
            },
        },
        "cache": {"enabled": False},
        "reranker": {"enabled": False},
        "retriever": {
            "type": "corpus",
            "corpus": {
                "embedder": {
                    "model_name": c.embedder.model_name,
                    "url": c.embedder.url,
                    "api_key": c.embedder.api_key,
                },
                "corpus_path": c.corpus_path,
                "index_path": c.index_path,
            },
        },
        "search": {"strategy": "mcts", "mcts": {"num_iterations": 1, "max_tree_depth": 2}},
        "node_generation": {
            "n": 1,
            "n_subquestions": 2,
            "top_k_websearch": 2,
            "top_k_entities": 1,
            "entity_linking_method": "llm",
        },
        "memory": {
            "working_memory": {"max_textual_memory_tokens": 4096},
            "interaction_memory": {"enabled": False},
        },
        "logging": {"level": "INFO", "format": "%(message)s"},
        "output": {"include_reasoning": False, "show_search_tree": False},
    }


@pytest.fixture
def minimal_config_web_search(minimal_config_dict):
    cfg = {**minimal_config_dict}
    key = os.environ.get("SERPER_API_KEY") or "dummy-for-ddgs-fallback"
    cfg["retriever"] = {
        "type": "web_search",
        "web_search": {
            "api_key": key,
            "top_k": 2,
            "crawl_full_text": False,
            "max_crawl_requests_per_second": 1.0,
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Live clients (real system)
# ---------------------------------------------------------------------------


@pytest.fixture
def live_llm_client(wemg_config):
    from wemg.llm.client import LLMClient

    requires_llm_credentials(wemg_config)
    gen = wemg_config.llm.generation
    client = LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        concurrency=min(4, wemg_config.llm.concurrency),
        max_retries=wemg_config.llm.max_retries,
        cache_config={"enabled": False},
        timeout=gen.timeout,
        temperature=gen.temperature,
        n=gen.n,
        top_p=gen.top_p,
        max_tokens=min(gen.max_tokens, 4096),
        max_input_tokens=min(gen.max_input_tokens, 32768),
        top_k=gen.top_k,
        enable_thinking=gen.enable_thinking,
        random_seed=gen.random_seed,
    )
    yield client
    client.close()


@pytest.fixture
def live_embedding_client(wemg_config):
    from wemg.llm.client import LLMClient

    requires_llm_credentials(wemg_config)
    c = wemg_config.retriever.corpus.embedder
    client = LLMClient(
        model_name=c.model_name,
        url=c.url,
        api_key=c.api_key,
        is_embedding=True,
        cache_config={"enabled": False},
    )
    yield client
    client.close()


@pytest.fixture
def live_wikidata_client(wemg_config):
    pytest.importorskip("SPARQLWrapper")
    from wemg.retrieval.wikidata import WikidataClient

    max_rps = None
    if wemg_config.retriever.type == "web_search":
        max_rps = wemg_config.retriever.web_search.max_crawl_requests_per_second
    return WikidataClient(max_wikipedia_requests_per_second=max_rps)


@pytest.fixture
def live_corpus_retriever(wemg_config):
    """Corpus retriever; skips if config is not corpus or paths/index missing."""
    if wemg_config.retriever.type != "corpus":
        pytest.skip("retriever.type is not corpus")
    requires_corpus_paths(wemg_config)
    from wemg.retrieval.corpus import CorpusRetriever

    c = wemg_config.retriever.corpus
    cpath, ipath = corpus_paths_from_config_or_env(wemg_config)
    return CorpusRetriever(
        embedder_config={
            "model_name": c.embedder.model_name,
            "url": c.embedder.url,
            "api_key": c.embedder.api_key,
        },
        corpus_path=cpath,
        index_path=ipath,
        embedder_type=c.embedder.embedder_type,
    )


@pytest.fixture
def live_web_search_tool(wemg_config):
    from wemg.retrieval.web_search import WebSearchTool

    ws = wemg_config.retriever.web_search
    return WebSearchTool(
        api_key=ws.api_key,
        max_crawl_requests_per_second=ws.max_crawl_requests_per_second,
    )


@pytest.fixture
def live_retriever(wemg_config):
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)
        from wemg.retrieval.corpus import CorpusRetriever

        c = wemg_config.retriever.corpus
        cpath, ipath = corpus_paths_from_config_or_env(wemg_config)
        return CorpusRetriever(
            embedder_config={
                "model_name": c.embedder.model_name,
                "url": c.embedder.url,
                "api_key": c.embedder.api_key,
            },
            corpus_path=cpath,
            index_path=ipath,
            embedder_type=c.embedder.embedder_type,
        )
    if wemg_config.retriever.type == "web_search":
        ws = wemg_config.retriever.web_search
        from wemg.retrieval.web_search import WebSearchTool

        return WebSearchTool(
            api_key=ws.api_key,
            max_crawl_requests_per_second=ws.max_crawl_requests_per_second,
        )
    pytest.skip(f"Unknown retriever type: {wemg_config.retriever.type}")


@pytest.fixture
def live_reranker(wemg_config):
    requires_reranker_config(wemg_config)
    from wemg.llm.client import LLMClient
    from wemg.retrieval.reranker import Reranker

    r = wemg_config.reranker
    client = LLMClient(
        model_name=r.model_name,
        url=r.url,
        api_key=r.api_key,
        concurrency=r.concurrency,
        temperature=0.0,
        max_tokens=1,
        cache_config={"enabled": False},
    )
    reranker = Reranker(client=client, top_k=r.top_k, instruction=r.instruction)
    yield reranker
    reranker.close()


@pytest.fixture
def optional_live_reranker(wemg_config):
    """Reranker client when enabled in config; otherwise None (does not skip)."""
    if not wemg_config.reranker.enabled:
        yield None
        return
    if not wemg_config.reranker.url:
        yield None
        return
    from wemg.llm.client import LLMClient
    from wemg.retrieval.reranker import Reranker

    r = wemg_config.reranker
    client = LLMClient(
        model_name=r.model_name,
        url=r.url,
        api_key=r.api_key,
        concurrency=r.concurrency,
        temperature=0.0,
        max_tokens=1,
        cache_config={"enabled": False},
    )
    reranker = Reranker(client=client, top_k=r.top_k, instruction=r.instruction)
    yield reranker
    reranker.close()


@pytest.fixture
def live_working_memory(wemg_config, live_wikidata_client):
    from wemg.reasoning.memory import WorkingMemory

    return WorkingMemory(
        max_textual_memory_tokens=wemg_config.memory.working_memory.max_textual_memory_tokens,
        wikidata_client=live_wikidata_client,
    )


@pytest.fixture
def node_generator_kwargs(wemg_config) -> Dict[str, Any]:
    ng = wemg_config.node_generation
    ws = wemg_config.retriever.web_search
    return {
        "n": ng.n,
        "n_subquestions": ng.n_subquestions,
        "top_k_websearch": ng.top_k_websearch,
        "top_k_entities": ng.top_k_entities,
        "n_hops": ng.n_hops,
        "entity_linking_method": ng.entity_linking_method,
        "rerank_kb_documents": ng.rerank_kb_documents,
        "azure_endpoint": ng.azure_endpoint,
        "azure_key": ng.azure_key,
        "max_crawl_requests_per_second": ws.max_crawl_requests_per_second,
    }


@pytest.fixture
def live_node_generator(
    wemg_config,
    live_llm_client,
    live_retriever,
    live_wikidata_client,
    live_working_memory,
    node_generator_kwargs,
    optional_live_reranker,
):
    from wemg.reasoning.generator import NodeGenerator

    reranker = optional_live_reranker
    return NodeGenerator(
        client=live_llm_client,
        retriever=live_retriever,
        wikidata_client=live_wikidata_client,
        working_memory=live_working_memory,
        interaction_memory=None,
        reranker=reranker,
        **node_generator_kwargs,
    )


@pytest.fixture
def smoke_system_overrides() -> List[str]:
    """Aggressive limits for end-to-end smoke tests."""
    return [
        "search.mcts.num_iterations=1",
        "search.mcts.max_tree_depth=2",
        "search.mcts.early_termination.enabled=false",
        "search.cot.max_depth=2",
        "node_generation.n_subquestions=2",
        "node_generation.top_k_websearch=2",
        "node_generation.top_k_entities=2",
        "memory.interaction_memory.enabled=false",
        "output.show_search_tree=false",
        "output.include_reasoning=false",
    ]
