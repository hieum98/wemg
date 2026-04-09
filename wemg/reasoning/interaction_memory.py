"""Interaction memory: ChromaDB-backed storage for few-shot examples.

This module contains:
- Async/thread-safe read-write locks
- OpenAI-compatible embedding wrapper for local embedding APIs
- `InteractionMemory` for logging and retrieving interaction examples
- Helper `log_to_interaction_memory`
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from openai import OpenAI

from wemg.utils.text import approximate_token_count

logger = logging.getLogger(__name__)


# =============================================================================
# Concurrency primitives
# =============================================================================


class AsyncReadWriteLock:
    """Async-compatible Read-Write Lock.

    Allows multiple concurrent readers OR a single exclusive writer.
    Writers have priority to prevent starvation.
    """

    def __init__(self):
        self._read_ready = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._pending_writers = 0

    async def acquire_read(self):
        async with self._read_ready:
            while self._writer or self._pending_writers > 0:
                await self._read_ready.wait()
            self._readers += 1

    async def release_read(self):
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    async def acquire_write(self):
        async with self._read_ready:
            self._pending_writers += 1
            try:
                while self._readers > 0 or self._writer:
                    await self._read_ready.wait()
                self._writer = True
            finally:
                self._pending_writers -= 1

    async def release_write(self):
        async with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()

    def read_lock(self):
        return _AsyncReadLockCtx(self)

    def write_lock(self):
        return _AsyncWriteLockCtx(self)


class _AsyncReadLockCtx:
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_read()
        return False


class _AsyncWriteLockCtx:
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_write()
        return False


class ThreadSafeReadWriteLock:
    """Thread-safe Read-Write Lock using threading primitives.

    Allows multiple concurrent readers OR a single exclusive writer.
    Writers have priority to prevent starvation.
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer = False
        self._pending_writers = 0

    def acquire_read(self):
        with self._read_ready:
            while self._writer or self._pending_writers > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        with self._read_ready:
            self._pending_writers += 1
            try:
                while self._readers > 0 or self._writer:
                    self._read_ready.wait()
                self._writer = True
            finally:
                self._pending_writers -= 1

    def release_write(self):
        with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()

    def read_lock(self):
        return _ThreadReadLockCtx(self)

    def write_lock(self):
        return _ThreadWriteLockCtx(self)


class _ThreadReadLockCtx:
    def __init__(self, lock: ThreadSafeReadWriteLock):
        self._lock = lock

    def __enter__(self):
        self._lock.acquire_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release_read()
        return False


class _ThreadWriteLockCtx:
    def __init__(self, lock: ThreadSafeReadWriteLock):
        self._lock = lock

    def __enter__(self):
        self._lock.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release_write()
        return False


# =============================================================================
# OpenAI-compatible embedding function for ChromaDB
# =============================================================================


class LocalCompatibleEmbedding(EmbeddingFunction):
    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY"):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, input: List[str]):
        response = self.client.embeddings.create(input=input, model=self.model_name)
        return [item.embedding for item in response.data]

    def close(self):
        try:
            self.client.close()
            logger.info("Successfully closed OpenAI client.")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")


# =============================================================================
# InteractionMemory
# =============================================================================


class InteractionMemory:
    """ChromaDB-backed interaction memory for few-shot example retrieval.

    Supports similarity and MMR retrieval strategies, thread-safe concurrent
    access via read-write locks, embedding caching, and role count caching.
    """

    def __init__(
        self,
        db_client: Optional[chromadb.Client] = None,
        db_path: Optional[str] = None,
        collection_name: str = "interaction_memory",
        token_budget: int = 8192,
        is_local_embedding_api: bool = False,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_base_url: str = "http://localhost:8000/v1",
        embedding_api_key: str = "EMPTY",
        enable_embedding_cache: bool = True,
    ):
        if is_local_embedding_api:
            self.embedding_function = LocalCompatibleEmbedding(
                base_url=embedding_base_url,
                model_name=embedding_model_name,
                api_key=embedding_api_key,
            )
        else:
            import torch

            logger.warning(
                "No embedding function provided. Using default Qwen/Qwen3-Embedding-0.6B "
                "model for embeddings. Consider running on a gpu for better performance."
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                device=device,
            )

        self.token_budget = token_budget
        self.collection_name = collection_name

        if db_client:
            logger.info("Using provided database client")
            self.db_client = db_client
        elif db_path:
            logger.info(f"Using persistent database at {db_path}")
            self.db_client = chromadb.PersistentClient(path=db_path)
        else:
            logger.info("Using in-memory database")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.db_client = chromadb.EphemeralClient()
                    break
                except (ValueError, AttributeError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"ChromaDB EphemeralClient initialization failed "
                            f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                        )
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(
                            f"ChromaDB EphemeralClient initialization failed "
                            f"after {max_retries} attempts"
                        )
                        raise

        self.collection = self.db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self._thread_rw_lock = ThreadSafeReadWriteLock()
        self._rw_lock = AsyncReadWriteLock()

        self._embedding_cache: Optional[Dict] = {} if enable_embedding_cache else None
        self._cache_max_size = 10000
        self._role_count_cache: Dict[str, int] = {}

    # -----------------------------------------------------------------
    # Write operations
    # -----------------------------------------------------------------

    def log_turn(
        self,
        role: str,
        user_input: Union[str, List[str]],
        assistant_output: Union[str, List[str]],
        batch_size: int = 32,
    ):
        """Log interaction turn(s) with thread-safe write lock and batched writes."""
        with self._thread_rw_lock.write_lock():
            if isinstance(user_input, str):
                user_input = [user_input]
            if isinstance(assistant_output, str):
                assistant_output = [assistant_output]

            if len(user_input) != len(assistant_output):
                logger.error(
                    f"Mismatched lengths: user_input={len(user_input)}, "
                    f"assistant_output={len(assistant_output)}"
                )
                return

            documents = []
            metadatas = []
            ids = []

            for u_in, a_out in zip(user_input, assistant_output):
                u_str = str(u_in) if not isinstance(u_in, str) else u_in
                a_str = str(a_out) if not isinstance(a_out, str) else a_out

                token_count = approximate_token_count(u_str) + approximate_token_count(a_str)
                if token_count > self.token_budget * 2:
                    logger.warning(
                        f"Skipping entry with {token_count} tokens "
                        f"(budget: {self.token_budget * 2})"
                    )
                    continue

                documents.append(u_str)
                metadatas.append({"role": role, "assistant_output": a_str})
                ids.append(str(uuid.uuid4()))

            if not documents:
                return

            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                self.collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

            self._role_count_cache[role] = (
                self._role_count_cache.get(role, 0) + len(documents)
            )

    async def log_turn_async(
        self,
        role: str,
        user_input: Union[str, List[str]],
        assistant_output: Union[str, List[str]],
        batch_size: int = 100,
    ):
        """Async wrapper – delegates to sync log_turn."""
        self.log_turn(role, user_input, assistant_output, batch_size=batch_size)

    # -----------------------------------------------------------------
    # Read operations
    # -----------------------------------------------------------------

    def _get_role_count(self, role: str) -> int:
        if role in self._role_count_cache:
            return self._role_count_cache[role]
        try:
            results = self.collection.get(where={"role": role}, limit=1)
            count = len(results["ids"]) if results["ids"] else 0
            if count == 0:
                self._role_count_cache[role] = 0
            return count
        except Exception as e:
            logger.warning("Role count lookup failed for %s; falling back to count=1: %s", role, e)
            return 1

    def get_examples(
        self,
        role: str,
        query: str,
        k: int = 3,
        strategy: str = "mmr",
    ) -> List[List[Dict[str, str]]]:
        """Retrieve few-shot examples with thread-safe read lock.

        strategy: 'similarity' (standard KNN) or 'mmr' (Maximal Marginal Relevance).
        """
        with self._thread_rw_lock.read_lock():
            if self.collection.count() == 0:
                return []

            if self._get_role_count(role) == 0:
                return []

            query_token_count = approximate_token_count(query)
            if query_token_count > self.token_budget * 2:
                logger.warning(
                    f"Query token count ({query_token_count}) exceeds "
                    f"token budget ({self.token_budget * 2})"
                )
                return []

            if strategy == "similarity":
                messages = self._fetch_similarity(role, query, k)
            elif strategy == "mmr":
                messages = self._fetch_mmr(role, query, k)
            else:
                raise ValueError("Unknown strategy. Use 'similarity' or 'mmr'")

            if messages:
                all_messages = [msg for pair in messages for msg in pair]
                total_tokens = approximate_token_count(all_messages)
                while total_tokens > self.token_budget and messages:
                    removed_pair = messages.pop(0)
                    removed_tokens = approximate_token_count(
                        [msg for msg in removed_pair]
                    )
                    total_tokens -= removed_tokens

            return messages

    async def get_examples_async(
        self,
        role: str,
        query: str,
        k: int = 3,
        strategy: str = "mmr",
    ) -> List[List[Dict[str, str]]]:
        """Async wrapper – delegates to sync get_examples."""
        return self.get_examples(role, query, k, strategy)

    # -----------------------------------------------------------------
    # Retrieval strategies
    # -----------------------------------------------------------------

    def _fetch_similarity(self, role: str, query: str, k: int):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"role": role},
        )
        return self._format_results(results)

    def _fetch_mmr(
        self,
        role: str,
        query: str,
        k: int,
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
    ):
        if self.collection.count() == 0:
            return []

        cache_key = f"{role}:{query}" if self._embedding_cache is not None else None
        if cache_key and cache_key in self._embedding_cache:
            query_embedding = self._embedding_cache[cache_key]
        else:
            query_embedding = self.collection._embedding_function([query])[0]
            if cache_key and self._embedding_cache is not None:
                if len(self._embedding_cache) >= self._cache_max_size:
                    self._embedding_cache.pop(next(iter(self._embedding_cache)))
                self._embedding_cache[cache_key] = query_embedding

        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where={"role": role},
            include=["embeddings", "metadatas", "documents"],
        )

        if not results["embeddings"] or len(results["embeddings"][0]) == 0:
            return []

        candidates_embeddings = results["embeddings"][0]
        candidates_docs = results["documents"][0]
        candidates_metas = results["metadatas"][0]

        num_candidates = len(candidates_embeddings)
        if num_candidates == 0:
            return []
        k = min(k, num_candidates)

        query_vec = np.array(query_embedding, dtype=np.float32)
        cand_vecs = np.array(candidates_embeddings, dtype=np.float32)

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        cand_norms = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)

        sim_to_query_all = np.dot(cand_norms, query_norm)
        cand_cand_similarities = np.dot(cand_norms, cand_norms.T)

        selected_indices: List[int] = []
        remaining_indices = set(range(num_candidates))

        for _ in range(k):
            if not remaining_indices:
                break

            remaining_list = list(remaining_indices)
            sim_to_query = sim_to_query_all[remaining_list]

            if selected_indices:
                sim_to_selected = cand_cand_similarities[
                    np.ix_(remaining_list, selected_indices)
                ]
                max_sim_to_selected = np.max(sim_to_selected, axis=1)
            else:
                max_sim_to_selected = np.zeros(len(remaining_list))

            mmr_scores = (lambda_mult * sim_to_query) - (
                (1 - lambda_mult) * max_sim_to_selected
            )

            best_local_idx = int(np.argmax(mmr_scores))
            best_real_idx = remaining_list[best_local_idx]

            selected_indices.append(best_real_idx)
            remaining_indices.remove(best_real_idx)

        final_examples = []
        for idx in selected_indices:
            final_examples.append(
                [
                    {"role": "user", "content": candidates_docs[idx]},
                    {"role": "assistant", "content": candidates_metas[idx]["assistant_output"]},
                ]
            )
        return final_examples

    @staticmethod
    def _format_results(results) -> List[List[Dict[str, str]]]:
        examples = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                examples.append(
                    [
                        {"role": "user", "content": results["documents"][0][i]},
                        {"role": "assistant", "content": results["metadatas"][0][i]["assistant_output"]},
                    ]
                )
        return examples

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def release(self, should_delete_db: bool = False):
        if hasattr(self, "embedding_function") and hasattr(self.embedding_function, "close"):
            self.embedding_function.close()
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        self._role_count_cache.clear()
        if should_delete_db and hasattr(self, "db_client"):
            try:
                self.db_client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")


# =============================================================================
# Interaction memory helpers
# =============================================================================


def log_to_interaction_memory(
    interaction_memory: Optional[InteractionMemory],
    log_data: Dict[str, List[Tuple[str, str]]],
    batch_size: int = 32,
) -> None:
    """Log role interactions from log_data dict to interaction memory."""
    if not interaction_memory or not log_data:
        return
    for role, entries in log_data.items():
        if entries:
            inputs, outputs = zip(*entries)
            interaction_memory.log_turn(
                role=role,
                user_input=list(inputs),
                assistant_output=list(outputs),
                batch_size=batch_size,
            )


__all__ = [
    "AsyncReadWriteLock",
    "ThreadSafeReadWriteLock",
    "LocalCompatibleEmbedding",
    "InteractionMemory",
    "log_to_interaction_memory",
]

