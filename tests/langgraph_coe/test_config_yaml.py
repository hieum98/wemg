"""Load ``langgraph_coe/config.yaml`` (no mocks for file I/O or parsing)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from langgraph_coe.config import LangGraphCoeConfig


def test_default_yaml_path_points_to_package_file():
    p = LangGraphCoeConfig.default_yaml_path()
    assert p.name == "config.yaml"
    assert p.parent.name == "langgraph_coe"


def test_from_yaml_reads_committed_defaults(langgraph_coe_config_path: Path):
    cfg = LangGraphCoeConfig.from_yaml(langgraph_coe_config_path, merge_api_key_env=False)
    assert cfg.wikidata.max_hops == 3
    assert cfg.wikidata.reranker_url is None
    assert cfg.llm.api_key is None or isinstance(cfg.llm.api_key, str)
    assert cfg.web_search.crawl_full_text is False
    assert cfg.reranker.enabled is False
    assert cfg.retriever.corpus.index_path.endswith(".faiss")


def test_from_yaml_api_key_from_env(monkeypatch, tmp_path: Path):
    y = tmp_path / "mini.yaml"
    y.write_text(yaml.safe_dump({"llm": {"api_key": None}, "wikidata": {"max_hops": 2}}), encoding="utf-8")
    monkeypatch.setenv("API_KEY", "from-env")
    cfg = LangGraphCoeConfig.from_yaml(y, merge_api_key_env=True)
    assert cfg.llm.api_key == "from-env"
    assert cfg.retriever.corpus.embedder.api_key == "from-env"
    assert cfg.reranker.api_key == "from-env"
    assert cfg.wikidata.max_hops == 2


def test_missing_yaml_file_results_in_constructed_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing = tmp_path / "does-not-exist.yaml"
    cfg = LangGraphCoeConfig.from_yaml(missing, merge_api_key_env=True)
    assert isinstance(cfg.llm.role_tiers, dict)
    assert cfg.wikidata.max_sparql_rps > 0
