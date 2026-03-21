"""Tests for CorpusRetriever. Requires CORPUS_PATH and INDEX_PATH for real run."""

import os
import pytest

from wemg.retrieval.corpus import CorpusRetriever


def test_corpus_retriever_init():
    """CorpusRetriever initializes with embedder_config and paths."""
    r = CorpusRetriever(
        embedder_config={"model_name": "Qwen3-Embedding-4B", "url": "http://n0378:4000/v1", "api_key": "sk-your-very-secure-master-key-here"},
        corpus_path="Hieuman/wiki23-processed",
        index_path="/home/hieum/uonlp/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss",
    )
    assert r.corpus_path == "Hieuman/wiki23-processed"
    assert r.index_path == "/home/hieum/uonlp/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss"
    assert r.embedder_type == "openai"


def test_corpus_retriever_retrieve_skip_no_corpus():
    """Without real corpus path and index, retrieve is skipped."""
    corpus_path = os.environ.get("CORPUS_PATH")
    index_path = os.environ.get("INDEX_PATH")
    if not corpus_path or not index_path:
        pytest.skip("CORPUS_PATH and INDEX_PATH not set; skipping real corpus retrieval")
    embedder_config = {
        "model_name": os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
        "url": os.environ.get("LLM_URL", "http://n0378:4000/v1"),
        "api_key": os.environ.get("API_KEY", "sk-your-very-secure-master-key-here"),
    }
    retriever = CorpusRetriever(
        embedder_config=embedder_config,
        corpus_path=corpus_path,
        index_path=index_path,
    )
    contents, scores = retriever.retrieve("test query", top_k=2)

    assert isinstance(contents, list)
    assert isinstance(scores, list)
    assert len(contents) <= 2
    assert len(contents) == len(scores)
