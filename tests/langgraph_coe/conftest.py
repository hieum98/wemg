"""Fixtures loading ``langgraph_coe/config.yaml`` (not ``wemg/config.yaml``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.bootstrap import load_test_env

try:
    import langgraph_coe.tools.wikidata as _wikidata_tools_mod
except ImportError:
    _wikidata_tools_mod = None


load_test_env()


@pytest.fixture(scope="session")
def langgraph_coe_config_path():
    from langgraph_coe.config import LangGraphCoeConfig

    p = LangGraphCoeConfig.default_yaml_path()
    if not p.is_file():
        pytest.skip(f"Missing CoE config YAML: {p}")
    return p


@pytest.fixture(scope="session")
def langgraph_coe_config(langgraph_coe_config_path):
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml(langgraph_coe_config_path)
    idx_override = os.environ.get("LANGGRAPH_CORPUS_INDEX_PATH", "").strip()
    if idx_override:
        cfg = cfg.model_copy(deep=True)
        cfg.retriever.corpus.index_path = idx_override
    return cfg


@pytest.fixture(scope="session")
def langgraph_coe_wikidata_initialized(langgraph_coe_config):
    if _wikidata_tools_mod is None:
        pytest.skip(
            "Wikidata tools unavailable (missing SPARQLWrapper or dependency import error)"
        )
    _wikidata_tools_mod.init_wikidata(langgraph_coe_config.wikidata)
    return True


def requires_coe_llm(langgraph_coe_config):
    """Skip unless YAML/env provides credentials and LiteLLM endpoints for role tiers."""
    if not getattr(langgraph_coe_config.llm, "api_key", None):
        pytest.skip("langgraph_coe llm.api_key missing (YAML or API_KEY / OPENAI_API_KEY)")
    role_tiers = getattr(langgraph_coe_config.llm, "role_tiers", {}) or {}
    tier_name = role_tiers.get("triple_pruner", "medium")
    tier_cfg = langgraph_coe_config.llm.tiers.get(tier_name)
    if not tier_cfg or not tier_cfg.api_base:
        pytest.skip(f"Tier {tier_name!r}: missing api_base in langgraph_coe config")


def requires_coe_corpus_index(langgraph_coe_config):
    idx = (langgraph_coe_config.retriever.corpus.index_path or "").strip()
    if not idx:
        pytest.skip(
            "retriever.corpus.index_path empty; set LANGGRAPH_CORPUS_INDEX_PATH or YAML"
        )
    p = Path(idx)
    if not p.name.endswith(".faiss"):
        pytest.skip(
            "Expected basename ending in .faiss (LangChain expects index_name.faiss)"
        )
    if not p.is_file():
        pytest.skip(f"No FAISS index file at corpus path ({p}); skip real corpus_search")
    if not p.parent.is_dir():
        pytest.skip(f"Corpus folder missing for index: {p.parent}")


@pytest.fixture
def langgraph_coe_web_ready(langgraph_coe_config):
    """Initialise DuckDuckGo/Serper web_search; teardown globals for other tests."""

    import langgraph_coe.tools.web as wmod

    wmod.init_web_search(langgraph_coe_config.web_search)
    yield langgraph_coe_config
    wmod._web_search_instance = None
    wmod._web_config = None


@pytest.fixture(autouse=True)
def _wikidata_tool_isolation_each_test():
    """Clear hop counters and entity cache before and after tests (no-op when Wikidata deps missing)."""

    def _cleanup() -> None:
        if _wikidata_tools_mod is None:
            return
        _wikidata_tools_mod.reset_wikidata_session()
        _wikidata_tools_mod.entity_cache.clear()

    _cleanup()
    yield
    _cleanup()
