"""Load `.env` and normalize common env aliases before tests read `wemg/config.yaml`."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False


def repo_root() -> Path:
    """Repository root (parent of `tests/`)."""
    return Path(__file__).resolve().parent.parent.parent


def default_config_yaml() -> Path:
    return repo_root() / "wemg" / "config.yaml"


def load_test_env() -> None:
    """Load `.env` from repo root if present; do not override existing os.environ."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    root = repo_root()
    env_file = root / ".env"
    if load_dotenv is not None and env_file.is_file():
        load_dotenv(env_file, override=False)

    if not os.environ.get("API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"]
    if not os.environ.get("LLM_URL") and os.environ.get("OPENAI_BASE_URL"):
        os.environ["LLM_URL"] = os.environ["OPENAI_BASE_URL"]


def corpus_paths_from_config_or_env(cfg) -> tuple[str | None, str | None]:
    """Corpus path and FAISS index: env CORPUS_PATH / INDEX_PATH override config."""
    cpath = os.environ.get("CORPUS_PATH")
    ipath = os.environ.get("INDEX_PATH")
    if cfg.retriever.type == "corpus":
        cpath = cpath or cfg.retriever.corpus.corpus_path
        ipath = ipath or cfg.retriever.corpus.index_path
    return cpath, ipath
