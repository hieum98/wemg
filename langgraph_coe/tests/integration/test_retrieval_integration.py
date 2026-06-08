"""Phase 0 integration smoke tests — live corpus retrieval and SGLang reranker.

Exercises ``corpus_search`` and ``call_sglang_reranker`` against real deployed
endpoints (embedder + optional reranker) and a local FAISS corpus bundle or
raw-index + HF dataset layout. Skipped when prerequisites are unreachable.

Run manually (from repo root, with tunnels / env set)::

    # Direct HTTP probe (prints request/response JSON per example):
    uv run python langgraph_coe/scripts/probe_reranker_server.py

    uv run pytest langgraph_coe/tests/phase0/test_retrieval_integration.py -v -s

Defaults come from ``langgraph_coe/config.yaml`` (``retriever.corpus``, ``reranker``).
Optional env overrides for CI or SSH tunnels::

    LANGGRAPH_TEST_EMBED_URL / LANGGRAPH_TEST_EMBED_MODEL
    LANGGRAPH_TEST_RERANK_URL / LANGGRAPH_TEST_RERANK_MODEL
    LANGGRAPH_CORPUS_INDEX_PATH / LANGGRAPH_CORPUS_PATH (or LANGGRAPH_CORPUS_DATASET)
    API_KEY / OPENAI_API_KEY       forwarded to embedder/reranker when set in .env
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import httpx
import pytest

from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.tools.retrieval import (
    call_sglang_reranker,
    corpus_search,
    init_retrieval_pipeline,
)

from .._fixtures import log_config_override
from .._servers import endpoint_alive as _endpoint_alive

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def _integration_cfg() -> LangGraphCoeConfig:
    """Loaded from langgraph_coe/config.yaml; env vars override when set."""
    return LangGraphCoeConfig.from_yaml()


def _embed_url() -> str:
    return _integration_cfg().retriever.corpus.embedder.url


def _embed_model() -> str:
    return _integration_cfg().retriever.corpus.embedder.model_name


def _rerank_url() -> str:
    return _integration_cfg().reranker.url


def _rerank_model() -> str:
    return _integration_cfg().reranker.model_name


_QUERY_FRANCE = "What is the capital of France?"
_QUERY_GERMANY = "Who wrote Faust?"

# Minimum score gap between #1 and #2 when the reranker is actually discriminating.
_MIN_TOP2_SCORE_GAP = 1e-6


@dataclass(frozen=True)
class RerankExample:
    """Explicit query + passages where exactly one document is clearly on-topic."""

    case_id: str
    query: str
    documents: Tuple[str, ...]
    relevant_index: int
    relevant_must_contain: str


# Curated contrasts: one passage answers the query; others are off-topic but fluent.
EXPLICIT_RERANK_CASES: Tuple[RerankExample, ...] = (
    RerankExample(
        case_id="france_capital_vs_programming_and_cricket",
        query="What is the capital of France?",
        documents=(
            "Paris is the capital and largest city of France.",
            "The Python programming language was created by Guido van Rossum in 1991.",
            "Cricket is a bat-and-ball game played between two teams of eleven players.",
        ),
        relevant_index=0,
        relevant_must_contain="Paris",
    ),
    RerankExample(
        case_id="faust_author_vs_france_and_photosynthesis",
        query="Who wrote the play Faust?",
        documents=(
            "Johann Wolfgang von Goethe wrote the tragic play Faust.",
            "Berlin is the capital and largest city of Germany.",
            "Photosynthesis converts light energy into chemical energy in chloroplasts.",
        ),
        relevant_index=0,
        relevant_must_contain="Goethe",
    ),
)


def _reranker_alive() -> bool:
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{_rerank_url().rstrip('/')}/rerank",
                json={
                    "model": _rerank_model(),
                    "query": "ping",
                    "documents": ["alpha", "beta"],
                },
                headers={"Authorization": "Bearer EMPTY"},
            )
            body = resp.json()
            if isinstance(body, list):
                return resp.status_code == 200 and len(body) > 0
            return resp.status_code == 200 and bool(body.get("results"))
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
    if not _endpoint_alive(_embed_url()):
        return False
    cfg = _build_retrieval_config(reranker_enabled=False)
    if _corpus_index_usable(cfg):
        return True
    return bool(cfg.retriever.corpus.resolved_corpus_source())


def _build_retrieval_config(*, reranker_enabled: bool) -> LangGraphCoeConfig:
    cfg = _integration_cfg()
    emb = cfg.retriever.corpus.embedder
    emb.api_key = emb.api_key or "EMPTY"
    cfg.reranker.api_key = cfg.reranker.api_key or "EMPTY"
    cfg.reranker.enabled = reranker_enabled
    if cfg.reranker.instruction is None:
        cfg.reranker.instruction = "Given a web search query, retrieve relevant passages that answer the query."
    return cfg


def _format_ranking(
    ranked: Sequence[Tuple[int, float]], documents: Sequence[str]
) -> str:
    lines = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        snippet = documents[idx].replace("\n", " ")[:120]
        lines.append(f"  #{rank} index={idx} score={score:.6f} text={snippet!r}")
    return "\n".join(lines)


def _assert_reranker_discriminates(
    ranked: Sequence[Tuple[int, float]],
    documents: Sequence[str],
    example: RerankExample,
) -> None:
    """Fail with a diagnostic table when the server returns flat or wrong ordering."""
    assert len(ranked) == len(documents), (
        f"{example.case_id}: expected {len(documents)} scores, got {len(ranked)}"
    )

    scores = [score for _, score in ranked]
    rounded = {round(s, 8) for s in scores}
    if len(rounded) == 1:
        pytest.fail(
            f"{example.case_id}: reranker returned identical scores for every document "
            f"({scores[0]:.8f}). The endpoint at {_rerank_url()!r} is reachable but not "
            "discriminating — check that Qwen3-Reranker is loaded and the SGLang chat "
            "template (see wemg/utils/sglang-qwen3-reranker.jinja) is applied.\n"
            f"Query: {example.query!r}\n"
            f"Ranking:\n{_format_ranking(ranked, documents)}"
        )

    top_idx, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else top_score
    if top_score - second_score < _MIN_TOP2_SCORE_GAP:
        pytest.fail(
            f"{example.case_id}: top score {top_score:.8f} is not above second "
            f"{second_score:.8f} by at least {_MIN_TOP2_SCORE_GAP}.\n"
            f"Query: {example.query!r}\n"
            f"Ranking:\n{_format_ranking(ranked, documents)}"
        )

    if top_idx != example.relevant_index:
        pytest.fail(
            f"{example.case_id}: expected document index {example.relevant_index} "
            f"({example.relevant_must_contain!r}) first, got index {top_idx}.\n"
            f"Query: {example.query!r}\n"
            f"Ranking:\n{_format_ranking(ranked, documents)}"
        )

    top_text = documents[top_idx]
    assert example.relevant_must_contain.lower() in top_text.lower(), (
        f"{example.case_id}: top passage should contain {example.relevant_must_contain!r}, "
        f"got: {top_text[:200]!r}"
    )

    scores_sorted = [s for _, s in ranked]
    assert scores_sorted == sorted(scores_sorted, reverse=True), (
        f"{example.case_id}: scores not returned in descending order: {scores_sorted}"
    )


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
    not _endpoint_alive(_embed_url()),
    reason=(
        f"Embedder unreachable at {_embed_url()!r}. "
        "Set retriever.corpus.embedder.url in langgraph_coe/config.yaml or "
        "LANGGRAPH_TEST_EMBED_URL."
    ),
)

requires_reranker = pytest.mark.skipif(
    not _reranker_alive(),
    reason=(
        f"Reranker unreachable at {_rerank_url()!r}/rerank. "
        "Set reranker.url in langgraph_coe/config.yaml or LANGGRAPH_TEST_RERANK_URL."
    ),
)

requires_corpus = pytest.mark.skipif(
    not _corpus_stack_ready(),
    reason=(
        "Corpus stack not ready: need live embedder (config.yaml embedder.url), "
        "retriever.corpus.index_path with .faiss or .pkl, and retriever.corpus.corpus_path "
        "for raw-index layout."
    ),
)


@requires_reranker
@pytest.mark.parametrize("example", EXPLICIT_RERANK_CASES, ids=lambda e: e.case_id)
async def test_reranker_explicit_contrast_puts_relevant_passage_first(
    example: RerankExample,
):
    """Each case has one obviously relevant passage; it must rank #1 with separable scores."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    ranked = await call_sglang_reranker(
        example.query,
        list(example.documents),
        cfg.reranker,
        timeout=60.0,
    )
    _assert_reranker_discriminates(ranked, example.documents, example)


@requires_reranker
async def test_reranker_order_invariant_when_passages_are_shuffled():
    """Relevant passage must win regardless of its position in the candidate list."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    query = "What is the capital of France?"
    relevant = "Paris is the capital and largest city of France."
    distractor = "Mount Everest is the highest mountain above sea level on Earth."

    for relevant_index, documents in (
        (0, (relevant, distractor)),
        (1, (distractor, relevant)),
    ):
        ranked = await call_sglang_reranker(
            query, list(documents), cfg.reranker, timeout=60.0
        )
        example = RerankExample(
            case_id=f"shuffle_relevant_at_{relevant_index}",
            query=query,
            documents=documents,
            relevant_index=relevant_index,
            relevant_must_contain="Paris",
        )
        _assert_reranker_discriminates(ranked, documents, example)


@requires_reranker
async def test_reranker_opposite_queries_pick_opposite_winners():
    """France query → Paris doc; Faust query → Goethe doc (cross-check, not one static bias)."""
    cfg = _build_retrieval_config(reranker_enabled=False)
    paris_doc = "Paris is the capital and largest city of France."
    goethe_doc = "Johann Wolfgang von Goethe wrote the tragic play Faust."
    documents = (paris_doc, goethe_doc)

    france_ranked = await call_sglang_reranker(
        "What is the capital of France?",
        list(documents),
        cfg.reranker,
        timeout=60.0,
    )
    faust_ranked = await call_sglang_reranker(
        "Who wrote the play Faust?",
        list(documents),
        cfg.reranker,
        timeout=60.0,
    )

    _assert_reranker_discriminates(
        france_ranked,
        documents,
        RerankExample(
            case_id="cross_query_france",
            query="What is the capital of France?",
            documents=documents,
            relevant_index=0,
            relevant_must_contain="Paris",
        ),
    )
    _assert_reranker_discriminates(
        faust_ranked,
        documents,
        RerankExample(
            case_id="cross_query_faust",
            query="Who wrote the play Faust?",
            documents=documents,
            relevant_index=1,
            relevant_must_contain="Goethe",
        ),
    )


@requires_reranker
async def test_reranker_documents_helper_respects_top_k():
    """``_rerank_documents`` returns top_k strings with the relevant passage first."""
    from langchain_core.documents import Document

    from langgraph_coe.tools.retrieval import _rerank_documents

    cfg = _build_retrieval_config(reranker_enabled=True)
    # This case has 3 documents; top_k must drop below that to prove the cap
    # truncates (config.yaml reranker.top_k is larger than the candidate count).
    cfg.reranker.top_k = log_config_override(
        "reranker.top_k",
        cfg.reranker.top_k,
        2,
        reason="3-doc case needs top_k<3 to verify the cap truncates",
    )
    example = EXPLICIT_RERANK_CASES[0]
    out = await _rerank_documents(
        [Document(page_content=t) for t in example.documents],
        example.query,
        cfg.reranker,
    )
    assert len(out) == cfg.reranker.top_k
    assert example.relevant_must_contain.lower() in out[0].page_content.lower()


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
    # reranker.top_k and corpus.search_k both come from config.yaml; the assertion
    # is written against config.reranker.top_k so it tracks whatever is configured.
    cfg = _build_retrieval_config(reranker_enabled=True)
    init_retrieval_pipeline(cfg.retriever, cfg.reranker)

    passages = await corpus_search.ainvoke({"query": _QUERY_FRANCE})

    assert 0 < len(passages) <= cfg.reranker.top_k
    assert all(isinstance(p, str) and p.strip() for p in passages)
