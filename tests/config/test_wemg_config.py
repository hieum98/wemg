"""Real config loading: YAML, overrides, env merge, validation (no mocks)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from wemg.config import WEMGConfig, get_default_config_path


def test_default_config_path_points_to_package_yaml():
    p = get_default_config_path()
    assert p.name == "config.yaml"
    assert p.parent.name == "wemg"


def test_from_yaml_loads_existing_file(wemg_config_path: Path):
    cfg = WEMGConfig.from_yaml(wemg_config_path)
    assert cfg.llm.model_name
    assert cfg.llm.url
    assert cfg.search.strategy in ("mcts", "cot")


def test_from_yaml_missing_file_uses_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = WEMGConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert cfg.llm.model_name  # model default from pydantic


def test_from_yaml_dotted_overrides(tmp_path: Path, wemg_config_path: Path):
    cfg = WEMGConfig.from_yaml(
        wemg_config_path,
        overrides=["search.strategy=cot", "logging.level=DEBUG"],
    )
    assert cfg.search.strategy == "cot"
    assert cfg.logging.level == "DEBUG"


def test_parse_override_value_types(tmp_path: Path, wemg_config_path: Path):
    cfg = WEMGConfig.from_yaml(
        wemg_config_path,
        overrides=[
            "cache.enabled=false",
            "reranker.enabled=true",
            "llm.api_key=null",
        ],
    )
    assert cfg.cache.enabled is False
    assert cfg.reranker.enabled is True
    assert cfg.llm.api_key is None


def test_env_api_key_fills_llm_when_yaml_null(monkeypatch, tmp_path: Path):
    """After load, API_KEY in environment populates nested api_key fields."""
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://localhost/v1", "api_key": None},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "x", "index_path": "y"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("API_KEY", "secret-from-env")
    cfg = WEMGConfig.from_yaml(yaml_path)
    assert cfg.llm.api_key == "secret-from-env"
    assert cfg.retriever.corpus.embedder.api_key == "secret-from-env"


def test_from_yaml_loads_api_key_from_dotenv(monkeypatch, tmp_path: Path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://localhost/v1", "api_key": None},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "x", "index_path": "y"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("API_KEY=secret-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = WEMGConfig.from_yaml(yaml_path)

    assert cfg.llm.api_key == "secret-from-dotenv"
    assert cfg.retriever.corpus.embedder.api_key == "secret-from-dotenv"


def test_validate_config_missing_llm_key(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://x/v1", "api_key": None},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "p", "index_path": "i"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = WEMGConfig.from_yaml(yaml_path)
    errs = cfg.validate_config()
    assert any("llm.api_key" in e for e in errs)


def test_validate_config_web_search_requires_key(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://x/v1", "api_key": "k"},
                "retriever": {
                    "type": "web_search",
                    "web_search": {"api_key": None},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = WEMGConfig.from_yaml(yaml_path)
    errs = cfg.validate_config()
    assert any("web_search.api_key" in e for e in errs)


def test_validate_config_corpus_requires_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("API_KEY", "k")
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://x/v1", "api_key": "k"},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "", "index_path": "/tmp/i"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = WEMGConfig.from_yaml(yaml_path)
    errs = cfg.validate_config()
    assert any("corpus_path" in e for e in errs)


def test_validate_config_azure_entity_linking(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("API_KEY", "k")
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://x/v1", "api_key": "k"},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "p", "index_path": "i"}},
                "node_generation": {
                    "entity_linking_method": "azure",
                    "azure_endpoint": None,
                    "azure_key": None,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = WEMGConfig.from_yaml(yaml_path)
    errs = cfg.validate_config()
    assert any("azure" in e.lower() for e in errs)


def test_node_generation_defaults_include_kb_source(wemg_config_path: Path):
    """kb_source defaults to 'wikidata' and freebase_subgraph_cache to None."""
    cfg = WEMGConfig.from_yaml(wemg_config_path)
    assert cfg.node_generation.kb_source == "wikidata"
    assert cfg.node_generation.freebase_subgraph_cache is None


def test_node_generation_kb_source_override(wemg_config_path: Path):
    cfg = WEMGConfig.from_yaml(
        wemg_config_path,
        overrides=["node_generation.kb_source=freebase_subgraph"],
    )
    assert cfg.node_generation.kb_source == "freebase_subgraph"


def test_node_generation_freebase_subgraph_cache_override(wemg_config_path: Path):
    cfg = WEMGConfig.from_yaml(
        wemg_config_path,
        overrides=["node_generation.freebase_subgraph_cache=./cache/test.jsonl"],
    )
    assert cfg.node_generation.freebase_subgraph_cache == "./cache/test.jsonl"


def test_node_generation_kb_source_invalid_value_rejected(wemg_config_path: Path):
    """Unknown kb_source values must raise a validation error."""
    with pytest.raises(Exception):
        WEMGConfig.from_yaml(
            wemg_config_path,
            overrides=["node_generation.kb_source=unknown_kb"],
        )


def test_wemg_system_rejects_invalid_config(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "llm": {"model_name": "m", "url": "http://x/v1", "api_key": None},
                "retriever": {"type": "corpus", "corpus": {"corpus_path": "p", "index_path": "i"}},
            }
        ),
        encoding="utf-8",
    )
    from wemg.system import WEMGSystem

    with pytest.raises(ValueError, match="Config validation failed"):
        WEMGSystem(config_path=yaml_path)
