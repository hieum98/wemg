import os
import httpx
from typing import List, Sequence, Optional

from pydantic import Field
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ..config import RetrieverConfig, RerankerConfig, EmbedderConfig

async def _rerank_documents(
    documents: Sequence[Document], query: str, cfg: RerankerConfig
) -> Sequence[Document]:
    """Custom reranker using Qwen Reranker API endpoint."""
    if not documents:
        return []
        
    texts = [doc.page_content for doc in documents]
    
    async with httpx.AsyncClient() as client:
        # Assumes vLLM/TEI standard /v1/rerank or similar OpenAI-compatible endpoint
        payload = {
            "model": cfg.model_name,
            "query": query,
            "texts": texts,
        }
        headers = {"Authorization": f"Bearer {cfg.api_key or 'EMPTY'}"}
        
        try:
            response = await client.post(f"{cfg.url.rstrip('/')}/rerank", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Assume results list has dicts with 'index' and 'score'
            results = data.get("results", [])
            if not results:
                return []
                
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            top_results = sorted_results[:cfg.top_k]
            
            return [documents[r["index"]] for r in top_results]
        except Exception as e:
            # Fallback: if reranking fails, just return top_k un-reranked documents
            return list(documents[:cfg.top_k])


def get_corpus_retriever(retriever_cfg: RetrieverConfig):
    """Factory to build the FAISS retriever."""
    emb_cfg = retriever_cfg.corpus.embedder
    
    embeddings = OpenAIEmbeddings(
        model=emb_cfg.model_name,
        base_url=emb_cfg.url,
        api_key=emb_cfg.api_key or "EMPTY",
    )
    
    # Load FAISS vector store
    vector_store = FAISS.load_local(
        folder_path=os.path.dirname(retriever_cfg.corpus.index_path),
        embeddings=embeddings,
        index_name=os.path.basename(retriever_cfg.corpus.index_path).replace(".faiss", ""),
        allow_dangerous_deserialization=True  # Required for local FAISS loads
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 10})

_retriever_instance = None
_reranker_config = None

def init_retrieval_pipeline(retriever_cfg: RetrieverConfig, reranker_cfg: RerankerConfig):
    global _retriever_instance, _reranker_config
    _retriever_instance = get_corpus_retriever(retriever_cfg)
    _reranker_config = reranker_cfg


@tool
async def corpus_search(query: str) -> List[str]:
    """Retrieve highly relevant context from the local knowledge corpus for a given query."""
    if not _retriever_instance:
        raise RuntimeError("Retriever pipeline not initialized. Call init_retrieval_pipeline first.")
    
    docs = await _retriever_instance.ainvoke(query)
    
    if _reranker_config and _reranker_config.enabled:
        docs = await _rerank_documents(docs, query, _reranker_config)
        
    return [doc.page_content for doc in docs]
