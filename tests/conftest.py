"""Shared fixtures and markers for WEMG real-system tests."""

import os
import pytest
from pathlib import Path


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API (API_KEY and optionally LLM_URL)"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis for cache"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration (real external services)"
    )


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary directory for ChromaDB or other file-based storage."""
    return tmp_path / "chroma_db"


@pytest.fixture
def minimal_config_dict():
    """Minimal config dict that passes validation (corpus retriever with path)."""
    return {
        "llm": {
            "model_name": "Qwen3.5-9B",
            "url": "http://n0378:4000/v1",
            "api_key": None,
            "concurrency": 2,
            "max_retries": 1,
            "generation": {
                "timeout": 60,
                "temperature": 0.0,
                "n": 1,
                "max_tokens": 256,
            },
        },
        "cache": {"enabled": False},
        "retriever": {
            "type": "corpus",
            "corpus": {
                "corpus_path": "Hieuman/wiki23-processed",
                "index_path": "/home/hieum/uonlp/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss",
            },
        },
        "search": {"strategy": "mcts", "mcts": {"num_iterations": 2, "max_tree_depth": 2}},
        "node_generation": {
            "n": 1,
            "n_subquestions": 2,
            "top_k_websearch": 2,
            "top_k_entities": 1,
            "top_k_properties": 1,
            "entity_linking_method": "llm",
        },
        "memory": {
            "working_memory": {"max_textual_memory_tokens": 4096},
            "interaction_memory": {
                "enabled": False,
            },
        },
        "logging": {"level": "INFO", "format": "%(message)s"},
        "output": {"include_reasoning": False, "show_search_tree": False},
    }


@pytest.fixture
def minimal_config_web_search(minimal_config_dict):
    """Minimal config with web_search retriever (needs SERPER_API_KEY or validation override)."""
    cfg = dict(minimal_config_dict)
    cfg["retriever"] = {
        "type": "web_search",
        "web_search": {
            "api_key": os.environ.get("SERPER_API_KEY") or "dummy-for-ddgs-fallback",
            "top_k": 2,
            "crawl_full_text": False,
            "max_crawl_requests_per_second": 1.0,
        },
    }
    return cfg


def get_llm_env():
    """Return (api_key, url, model_name) from env for real LLM tests."""
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("LLM_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("LLM_MODEL") or "gpt-3.5-turbo"
    return api_key, url, model


def requires_llm():
    """Skip if LLM_URL (or OPENAI_BASE_URL) is not set."""
    _, url, _ = get_llm_env()
    if not url:
        pytest.skip("LLM_URL or OPENAI_BASE_URL not set; skipping LLM test")
