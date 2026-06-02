"""Phase 0 integration smoke tests — live corpus retrieval and SGLang reranker.

Exercises ``corpus_search`` and ``call_sglang_reranker`` against real deployed
endpoints (embedder + optional reranker) and a local FAISS corpus bundle or
raw-index + HF dataset layout. Skipped when prerequisites are unreachable.

Run manually (from repo root, with tunnels / env set)::

    pytest langgraph_coe/tests/phase0/test_retrieval_integration.py -v -s

Environment (all optional; defaults match Phase 1 tunnel conventions)::

    LANGGRAPH_TEST_EMBED_URL       embedder OpenAI-compatible base (default localhost:30164/v1)
    LANGGRAPH_TEST_EMBED_MODEL     embedder model id (default Qwen/Qwen3-Embedding-4B)
    LANGGRAPH_TEST_RERANK_URL      reranker base incl. /v1 (default localhost:30002/v1)
    LANGGRAPH_TEST_RERANK_MODEL    reranker model id (default Qwen3-Reranker-4B)
    LANGGRAPH_CORPUS_INDEX_PATH    override retriever.corpus.index_path
    LANGGRAPH_CORPUS_DATASET       raw-index layout: HF dataset or load_from_disk path
    API_KEY / OPENAI_API_KEY       forwarded to embedder/reranker when set in .env

Example tunnels::

    ssh -fN -L 30164:n0385:4000 -L 30002:n0999:30002 <host>
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.tools.retrieval import (
    call_sglang_reranker,
    corpus_search,
    init_retrieval_pipeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

EMBED_URL = os.environ.get("LANGGRAPH_TEST_EMBED_URL", "http://localhost:30164/v1")
EMBED_MODEL = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
RERANK_URL = os.environ.get("LANGGRAPH_TEST_RERANK_URL", "http://localhost:30002/v1")
RERANK_MODEL = os.environ.get("LANGGRAPH_TEST_RERANK_MODEL", "Qwen3-Reranker-4B")

_QUERY_FRANCE = "What is the capital of France?"
_QUERY_GERMANY = "Who wrote Faust?"


def _endpoint_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(f"{url.rstrip('/')}/models")
            return resp.status_code == 200
    except Exception:
        return False


def _reranker_alive() -> bool:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{RERANK_URL.rstrip('/')}/rerank",
                json={
                    "model": RERANK_MODEL,
                    "query": "ping",
                    "documents": ["alpha", "beta"],
                },
                headers={"Authorization": "Bearer EMPTY"},
            )
            return resp.status_code == 200 and bool(resp.json().get("results"))
    except Exception:
        return False


def _corpus_index_usable(cfg: LangGraphCoeConfig) -> bool:
    index_path = cfg.retriever.corpus.index_path
    if os.path.isfile(index_path):
        return True
    folder = os.path.dirname(index_path) or "."
    index_name = os.path.basename(index_path).replace(".faiss", "")
    return os.path.isfile(os.path.join(folder, f"{index_name}.pkl"))


def _corpus_stack_ready() -> bool:
    if not _endpoint_alive(EMBED_URL):
        return False
    cfg = _build_retrieval_config(reranker_enabled=False)
    if _corpus_index_usable(cfg):
        return True
    # Raw layout: need both bare .faiss (or build path) and corpus_dataset.
    return bool(cfg.retriever.corpus.corpus_dataset)


def _build_retrieval_config(*, reranker_enabled: bool) -> LangGraphCoeConfig:
    cfg = LangGraphCoeConfig.from_yaml()
    emb = cfg.retriever.corpus.embedder
    emb.url = EMBED_URL
    emb.model_name = EMBED_MODEL
    emb.api_key = emb.api_key or "EMPTY"

    cfg.reranker.url = RERANK_URL
    cfg.reranker.model_name = RERANK_MODEL
    cfg.reranker.api_key = cfg.reranker.api_key or "EMPTY"
    cfg.reranker.enabled = reranker_enabled

    return cfg


@pytest.fixture(autouse=True)
def _reset_retrieval_module_state():
    """Avoid sharing one global retriever/reranker config across tests."""
    from langgraph_coe.tools import retrieval as rmod

    rmod._retriever_instance = None
    rmod._reranker_config = None
    yield
    rmod._retriever_instance = None
    rmod._reranker_config = None


requires_embedder = pytest.mark.skipif(
    not _endpoint_alive(EMBED_URL),
    reason=(
        f"Embedder unreachable at {EMBED_URL!r}. "
        "Open a tunnel, e.g. ssh -fN -L 30164:n0385:4000 <host>"
    ),
)

requires_reranker = pytest.mark.skipif(
    not _reranker_alive(),
    reason=(
        f"Reranker unreachable at {RERANK_URL!r}/rerank. "
        "Open a tunnel, e.g. ssh -fN -L 30002:n0999:30002 <host>"
    ),
)

requires_corpus = pytest.mark.skipif(
    not _corpus_stack_ready(),
    reason=(
        "Corpus stack not ready: need live embedder plus either "
        "LANGGRAPH_CORPUS_INDEX_PATH (or config.yaml index_path) with a "
        ".faiss or .pkl sidecar, or LANGGRAPH_CORPUS_DATASET for raw-index layout."
    ),
)


@requires_reranker
async def test_call_sglang_reranker_orders_by_relevance():
    """Reranker should put the clearly relevant passage first."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    query = "What is the capital of France?"
    docs = [
        "The Braille system is a tactile writing system.",
        "Paris is the capital and largest city of France.",
        "Plackett–Burman designs are used in fractional factorial experiments.",
    ]
    ranked = await call_sglang_reranker(query, docs, cfg.reranker, timeout=60.0)

    assert len(ranked) == len(docs)
    assert ranked[0][0] == 1, f"expected Paris passage first, got indices {[i for i, _ in ranked]}"
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@requires_reranker
async def test_call_sglang_reranker_respects_top_k_slice_via_corpus_helper():
    """``_rerank_documents`` path is exercised indirectly when reranker top_k < candidate count."""
    cfg = _build_retrieval_config(reranker_enabled=True)
    cfg.reranker.top_k = 2
    query = "capital of France"
    docs = [
        "Unrelated passage about cricket rules.",
        "Paris is the capital of France.",
        "Another unrelated passage about ocean currents.",
    ]
    from langchain_core.documents import Document

    from langgraph_coe.tools.retrieval import _rerank_documents

    out = await _rerank_documents(
        [Document(page_content=t) for t in docs],
        query,
        cfg.reranker,
    )
    assert len(out) == 2
    assert "Paris" in out[0].page_content


@requires_corpus
async def test_corpus_search_returns_nonempty_passages():
    """FAISS + live Qwen3 query embeddings return scored passages."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)

    passages = await corpus_search.ainvoke({"query": _QUERY_FRANCE})

    assert isinstance(passages, list)
    assert len(passages) > 0
    assert all(isinstance(p, str) and p.strip() for p in passages)
    joined = " ".join(passages).lower()
    assert "paris" in joined or "france" in joined, (
        "top hits should mention France/Paris; if every query returns the same hub "
        "passages, check embedder raw-text path and query_instruction"
    )


@requires_corpus
async def test_corpus_search_queries_are_not_collapsed():
    """Different queries should not return identical hit lists (embedding sanity check)."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)

    france_hits = await corpus_search.ainvoke({"query": _QUERY_FRANCE, "top_k": 5})
    germany_hits = await corpus_search.ainvoke({"query": _QUERY_GERMANY, "top_k": 5})

    assert france_hits != germany_hits, (
        "identical top-5 lists for unrelated queries usually means broken query embeddings"
    )


@requires_corpus
async def test_corpus_search_top_k_cap():
    cfg = _build_retrieval_config(reranker_enabled=False)
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)

    passages = await corpus_search.ainvoke({"query": _QUERY_FRANCE, "top_k": 3})

    assert len(passages) <= 3


@requires_corpus
@requires_reranker
async def test_corpus_search_with_live_reranker():
    """End-to-end: FAISS candidates then SGLang rerank; output length bounded by reranker.top_k."""
    cfg = _build_retrieval_config(reranker_enabled=True)
    cfg.reranker.top_k = 3
    cfg.retriever.corpus.search_k = 8
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)

    passages = await corpus_search.ainvoke({"query": _QUERY_FRANCE})

    assert 0 < len(passages) <= cfg.reranker.top_k
    assert all(isinstance(p, str) and p.strip() for p in passages)
