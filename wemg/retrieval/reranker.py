"""Reranker: question + documents -> top-k using dedicated /rerank endpoint client."""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wemg.llm.client import LLMClient


class Reranker:
    """Reranks documents by relevance using a dedicated reranker server client."""

    def __init__(
        self,
        client: "LLMClient",
        top_k: int = 10,
        instruction: Optional[str] = None,
    ):
        self._client = client
        self.top_k = top_k
        self.instruction = instruction

    def rerank(
        self,
        question: str,
        documents: List[str],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[str]:
        """Return top-k documents reranked by relevance to the question."""
        if not documents:
            return []
        k = top_k if top_k is not None else self.top_k
        inst = instruction if instruction is not None else self.instruction
        return self._client.rerank_documents(question, documents, top_k=k, instruction=inst)

    def best_index(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> int:
        """Index of the highest-scoring document for the query (0 if documents empty)."""
        if not documents or not query:
            return 0
        inst = instruction if instruction is not None else self.instruction
        sorted_indices, _, _ = self._client.get_reranking_scores(query, documents, inst)
        return sorted_indices[0] if sorted_indices else 0

    def close(self) -> None:
        """Close the underlying LLM client."""
        if self._client:
            self._client.close()
