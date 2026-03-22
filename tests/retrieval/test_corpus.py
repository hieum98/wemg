"""CorpusRetriever against real embedder + on-disk FAISS (from config or env)."""

import pytest

from tests.conftest import requires_corpus_paths, requires_llm_credentials
from tests.helpers.bootstrap import corpus_paths_from_config_or_env
from wemg.retrieval.corpus import CorpusRetriever


def test_corpus_retriever_paths_match_config(wemg_config):
    if wemg_config.retriever.type != "corpus":
        pytest.skip("retriever.type is not corpus")
    c = wemg_config.retriever.corpus
    cpath, ipath = corpus_paths_from_config_or_env(wemg_config)
    r = CorpusRetriever(
        embedder_config={
            "model_name": c.embedder.model_name,
            "url": c.embedder.url,
            "api_key": c.embedder.api_key,
        },
        corpus_path=cpath,
        index_path=ipath,
        embedder_type=c.embedder.embedder_type,
    )
    assert r.corpus_path == cpath
    assert r.index_path == ipath


@pytest.mark.requires_llm
@pytest.mark.requires_corpus
def test_corpus_retriever_retrieve_real(wemg_config):
    requires_llm_credentials(wemg_config)
    requires_corpus_paths(wemg_config)
    c = wemg_config.retriever.corpus
    cpath, ipath = corpus_paths_from_config_or_env(wemg_config)
    retriever = CorpusRetriever(
        embedder_config={
            "model_name": c.embedder.model_name,
            "url": c.embedder.url,
            "api_key": c.embedder.api_key,
        },
        corpus_path=cpath,
        index_path=ipath,
        embedder_type=c.embedder.embedder_type,
    )
    contents, scores = retriever.retrieve("capital city France", top_k=2)
    assert isinstance(contents, list)
    assert isinstance(scores, list)
    assert len(contents) <= 2
    assert len(contents) == len(scores)
