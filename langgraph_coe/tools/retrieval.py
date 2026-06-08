"""Corpus retrieval + reranker tools.

FAISS corpus loading follows ``langgraph_sar/tools/retrieval.py``: LangChain
bundle (``.faiss`` + ``.pkl``) or raw HF-datasets index with a side corpus
dataset. Queries use ``_Qwen3QueryEmbeddings`` (raw text + query instruction).

The optional reranker speaks SGLang's Cohere-compatible ``/v1/rerank`` API.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from langchain_community.docstore.base import Docstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from ..config import EmbedderConfig, RerankerConfig, RetrieverConfig, CorpusConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SGLang reranker HTTP client
# ──────────────────────────────────────────────────────────────────────────────


def _parse_rerank_results(data: Any) -> List[Dict[str, Any]]:
    """Normalize SGLang /v1/rerank JSON (top-level list or Cohere-style dict)."""
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [row for row in results if isinstance(row, dict)]
    return []


def _coerce_score(row: Dict[str, Any]) -> float:
    """Extract the relevance score from a rerank result row."""
    for key in ("relevance_score", "score"):
        if key in row:
            return float(row[key])
    raise KeyError(
        f"rerank result row missing both 'relevance_score' and 'score': {row!r}"
    )


async def call_sglang_reranker(
    query: str,
    texts: Sequence[str],
    cfg: RerankerConfig,
    *,
    timeout: float = 30.0,
) -> List[Tuple[int, float]]:
    """POST to the SGLang ``/rerank`` endpoint and return ``[(index, score), ...]``."""
    if not texts:
        return []
    if not cfg.url:
        raise RuntimeError("reranker URL is not configured; cannot rerank")

    payload: Dict[str, Any] = {
        "model": cfg.model_name,
        "query": query,
        "documents": list(texts),
    }
    if cfg.instruction:
        payload["instruct"] = cfg.instruction
    headers = {"Authorization": f"Bearer {cfg.api_key or 'EMPTY'}"}
    endpoint = f"{cfg.url.rstrip('/')}/rerank"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)

    if resp.status_code >= 400:
        body_preview = resp.text[:2000]
        logger.error(
            "[reranker] HTTP %d from %s\n"
            "----- BEGIN BODY -----\n%s\n----- END BODY -----",
            resp.status_code,
            endpoint,
            body_preview,
        )
        resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        body_preview = resp.text[:2000]
        logger.error(
            "[reranker] non-JSON response from %s:\n"
            "----- BEGIN BODY -----\n%s\n----- END BODY -----",
            endpoint,
            body_preview,
        )
        raise

    results = _parse_rerank_results(data)
    if not results:
        logger.error("[reranker] no 'results' in response from %s: %r", endpoint, data)
        raise RuntimeError(
            f"SGLang reranker returned no results for {len(texts)} candidates"
        )

    ranked: List[Tuple[int, float]] = []
    for row in results:
        if not isinstance(row, dict):
            logger.error("[reranker] unexpected row shape: %r", row)
            raise TypeError(f"rerank row is not a dict: {type(row).__name__}")
        idx = int(row["index"])
        score = _coerce_score(row)
        if not (0 <= idx < len(texts)):
            raise IndexError(
                f"rerank row index {idx} out of range for {len(texts)} candidates"
            )
        ranked.append((idx, score))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


async def _rerank_documents(
    documents: Sequence[Document], query: str, cfg: RerankerConfig
) -> Sequence[Document]:
    """Rerank ``Document`` objects via SGLang and return top-``cfg.top_k``."""
    if not documents:
        return []
    texts = [doc.page_content for doc in documents]
    ranked = await call_sglang_reranker(query, texts, cfg)
    top = ranked[: cfg.top_k]
    return [documents[idx] for idx, _ in top]


# ──────────────────────────────────────────────────────────────────────────────
# FAISS retriever wiring (langgraph_sar pattern)
# ──────────────────────────────────────────────────────────────────────────────


class _Qwen3QueryEmbeddings(Embeddings):
    """Query-side embeddings for the asymmetric Qwen3-Embedding retriever.

    Two corrections over a bare ``OpenAIEmbeddings``, both required to match how the
    corpus was indexed by ``state_aware_rag/servers/build_index.py``:

    * **Send raw text, not token ids.** ``OpenAIEmbeddings`` defaults to
      ``check_embedding_ctx_length=True``, which tiktoken-encodes the input and posts
      *integer token ids* (OpenAI cl100k vocab). SGLang re-reads those ids in Qwen's
      own vocab, so the query vector is meaningless and the search returns the same hub
      passages for every query. We force the raw-text path.
    * **Apply the query instruction.** Qwen3-Embedding expects queries (not passages) to
      carry an ``Instruct: …\\nQuery: …`` prefix. Passages were indexed raw, so only the
      query path wraps; ``embed_documents`` is left untouched for symmetry/tests.
    """

    def __init__(self, emb_cfg: EmbedderConfig) -> None:
        self._inner = OpenAIEmbeddings(
            model=emb_cfg.model_name,
            base_url=emb_cfg.url,
            api_key=emb_cfg.api_key or "EMPTY",
            check_embedding_ctx_length=False,
        )
        self._instruction = emb_cfg.query_instruction or ""

    def _wrap(self, text: str) -> str:
        if self._instruction and "{query}" in self._instruction:
            return self._instruction.format(query=text)
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(self._wrap(text))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._inner.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self._inner.aembed_query(self._wrap(text))


class _LazyTextDocstore(Docstore):
    """Read-only docstore that builds a ``Document`` on demand from a text list."""

    def __init__(self, texts: Any) -> None:
        self._texts = texts

    def search(self, search: str) -> Any:
        try:
            i = int(search)
            text = self._texts[i]
        except (ValueError, TypeError, IndexError, KeyError):
            return f"ID {search} not found."
        return Document(
            page_content="" if text is None else str(text), metadata={"row": i}
        )


class _IdentityStrMapping(Mapping):
    """``{i: str(i)}`` for ``i in range(n)`` without materializing the dict."""

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __getitem__(self, i: int) -> str:
        try:
            idx = int(i)
        except (TypeError, ValueError):
            raise KeyError(i)
        if 0 <= idx < self._n:
            return str(idx)
        raise KeyError(i)

    def __iter__(self):
        return iter(range(self._n))

    def __len__(self) -> int:
        return self._n


def index_name_for(index_path: str) -> str:
    """Expected ``.pkl`` sidecar filename for a given ``.faiss`` index path."""
    return os.path.basename(index_path).replace(".faiss", "") + ".pkl"


def _load_raw_faiss_corpus(
    index_path: str, embeddings: Any, corpus_cfg: CorpusConfig
) -> FAISS:
    """Wrap a bare HF-datasets FAISS index in a LangChain store via a side corpus."""
    import faiss  # local import: only needed for the raw-index layout
    import datasets

    dataset_ref = corpus_cfg.resolved_corpus_source()
    if not dataset_ref:
        raise FileNotFoundError(
            f"No docstore sidecar {index_name_for(index_path)!r} next to {index_path!r}, "
            "so the index is a raw HF-datasets index."
        )
    if os.path.exists(dataset_ref):
        ds = datasets.load_from_disk(dataset_ref)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[corpus_cfg.corpus_split]
    else:
        ds = datasets.load_dataset(dataset_ref, split=corpus_cfg.corpus_split)

    if not os.path.exists(index_path):
        logger.info("Index not found at %r, building index...", index_path)
        texts = ds[corpus_cfg.text_column]
        vector_store = FAISS.from_texts(texts, embeddings=embeddings)
        vector_store.save_local(
            folder_path=os.path.dirname(index_path),
            index_name=os.path.basename(index_path).replace(".faiss", ""),
        )
        return vector_store

    index = faiss.read_index(index_path)

    if index.ntotal != len(ds):
        raise ValueError(
            f"Index/corpus size mismatch: FAISS index {index_path!r} has "
            f"{index.ntotal} vectors but corpus {dataset_ref!r} "
            f"(split={corpus_cfg.corpus_split!r}) has {len(ds)} rows. The corpus "
            "must be the exact dataset the index was built from, in the same order."
        )

    text_col = corpus_cfg.text_column
    if text_col not in ds.column_names:
        fallback = next(
            (
                c
                for c in ("contents", "text", "passage", "content")
                if c in ds.column_names
            ),
            None,
        )
        if fallback is None:
            raise ValueError(
                f"text_column {text_col!r} not found in corpus {dataset_ref!r}. "
                f"Available columns: {ds.column_names}. Set retriever.corpus.text_column."
            )
        text_col = fallback

    texts = ds[text_col]

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=_LazyTextDocstore(texts),
        index_to_docstore_id=_IdentityStrMapping(len(texts)),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return vector_store


def get_corpus_retriever(retriever_cfg: RetrieverConfig):
    """Build the FAISS retriever, embedding queries via the deployed endpoint.

    Two index layouts are supported (see ``CorpusConfig``):

    * **LangChain bundle** — ``<name>.faiss`` + ``<name>.pkl`` docstore sidecar;
      loaded via :meth:`FAISS.load_local`.
    * **Raw HF-datasets index** — a bare ``<name>.faiss`` from ``build_index.py``,
      whose passage text lives in a separate HF dataset (``corpus.corpus_path``).
    """
    corpus = retriever_cfg.corpus
    emb_cfg = corpus.embedder

    embeddings = _Qwen3QueryEmbeddings(emb_cfg)

    index_path = corpus.index_path
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path!r}")

    folder = os.path.dirname(index_path) or "."
    index_name = os.path.basename(index_path).replace(".faiss", "")
    pkl_path = os.path.join(folder, index_name + ".pkl")

    if os.path.exists(pkl_path):
        vector_store = FAISS.load_local(
            folder_path=folder,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
    else:
        vector_store = _load_raw_faiss_corpus(index_path, embeddings, corpus)

    return vector_store.as_retriever(search_kwargs={"k": corpus.search_k})


_retriever_instance = None
_reranker_config: Optional[RerankerConfig] = None


def init_retrieval_pipeline(
    retriever_cfg: RetrieverConfig, reranker_cfg: RerankerConfig
) -> None:
    """One-shot init; build the FAISS retriever before first ``corpus_search`` use."""
    global _retriever_instance, _reranker_config
    _retriever_instance = get_corpus_retriever(retriever_cfg)
    _reranker_config = reranker_cfg


def set_retriever(retriever) -> None:
    """Inject a retriever instance directly (tests / custom backends)."""
    global _retriever_instance
    _retriever_instance = retriever


def _doc_to_text(doc: Any) -> str:
    """Normalize retriever outputs from LangChain ``Document`` or simple stubs."""
    if isinstance(doc, Document):
        return doc.page_content
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, dict) and "page_content" in doc:
        return str(doc["page_content"])
    return str(doc)


@tool
async def corpus_search(query: str, top_k: Optional[int] = None) -> List[str]:
    """Retrieve relevant passages from the knowledge corpus for a query.

    Returns score-ordered passage contents. When a reranker is configured and
    enabled, candidates are reranked before return. ``top_k`` optionally caps
    the result count after retrieval (and reranking).
    """
    if _retriever_instance is None:
        raise RuntimeError(
            "Retriever pipeline not initialized. Call init_retrieval_pipeline first."
        )

    docs = await _retriever_instance.ainvoke(query)

    if _reranker_config and _reranker_config.enabled:
        docs = await _rerank_documents(docs, query, _reranker_config)

    texts = [_doc_to_text(doc) for doc in docs]
    if top_k is not None:
        texts = texts[:top_k]
    return texts
