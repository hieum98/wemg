"""Corpus retrieval using FAISS index and embeddings."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CorpusRetriever:
    """FAISS-based corpus retriever with embedding support."""

    def __init__(
        self,
        embedder_config: dict,
        corpus_path: Optional[Union[str, Path]] = None,
        index_path: Optional[Union[str, Path]] = None,
        embedder_type: str = "openai",
    ):
        self.embedder_config = embedder_config
        self.corpus_path = str(corpus_path) if corpus_path else None
        self.index_path = str(index_path) if index_path else None
        self.embedder_type = embedder_type

        self._embedder = None
        self._corpus = None

        model_name = embedder_config.get("model_name", "").lower()
        if "e5" in model_name:
            self._query_prefix = "query: "
            self._candidate_prefix = "passage: "
        elif "gte" in model_name or "qwen" in model_name:
            self._query_prefix = "Instruct: Retrieve relevant passages\nQuery: "
            self._candidate_prefix = ""
        else:
            self._query_prefix = ""
            self._candidate_prefix = ""

    def warmup(self) -> None:
        """Eagerly load embedder and corpus/index (e.g. before multi-threaded batch use)."""
        _ = self.embedder
        _ = self.corpus

    @property
    def embedder(self):
        if self._embedder is None:
            from coe.llm.client import LLMClient
            self._embedder = LLMClient(
                model_name=self.embedder_config["model_name"],
                url=self.embedder_config.get("url", ""),
                api_key=self.embedder_config.get("api_key"),
                is_embedding=True,
            )
        return self._embedder

    @property
    def corpus(self):
        if self._corpus is None:
            self._corpus = self._load_corpus()
        return self._corpus

    def _load_corpus(self):
        """Load corpus dataset and FAISS index."""
        from datasets import load_dataset, load_from_disk, Dataset
        import os

        if self.corpus_path is None:
            raise ValueError("corpus_path is required")

        if os.path.isdir(self.corpus_path):
            corpus = load_from_disk(self.corpus_path)
        else:
            corpus = load_dataset(self.corpus_path, split="train")

        if self.index_path and Path(self.index_path).exists():
            corpus.load_faiss_index("embeddings", self.index_path)
        else:
            corpus = self._build_index(corpus)

        return corpus

    def _build_index(self, corpus):
        """Build FAISS index for corpus."""
        logger.info("Building FAISS index...")

        def get_embeddings(batch):
            texts = batch["contents"]
            texts = [self._candidate_prefix + t for t in texts]
            embeddings = self.embedder.get_embeddings(texts)
            return {"embeddings": embeddings}

        corpus = corpus.map(get_embeddings, batched=True, batch_size=32)
        corpus.add_faiss_index(column="embeddings")

        if self.index_path:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            corpus.save_faiss_index("embeddings", self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")

        return corpus

    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: int = 5,
    ) -> Tuple[Union[List[str], List[List[str]]], Union[List[float], List[List[float]]]]:
        """Retrieve relevant passages for queries."""
        single = isinstance(queries, str)
        if single:
            queries = [queries]

        queries = [self._query_prefix + q for q in queries]
        query_embeddings = self.embedder.get_embeddings(queries)

        import numpy as np
        if not isinstance(query_embeddings[0], list):
            query_embeddings = [query_embeddings]

        query_array = np.array(query_embeddings, dtype=np.float32)
        scores, results = self.corpus.get_nearest_examples_batch("embeddings", query_array, k=top_k)
        all_contents = []
        all_scores = []
        for i in range(len(queries)):
            contents = results[i]["contents"]
            batch_scores = scores[i].tolist() if hasattr(scores[i], 'tolist') else list(scores[i])
            all_contents.append(contents)
            all_scores.append(batch_scores)

        if single:
            return all_contents[0], all_scores[0]
        return all_contents, all_scores



