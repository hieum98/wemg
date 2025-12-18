import asyncio
from datetime import datetime
import gc
from typing import List, Union
import logging
import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from wemg.utils.preprocessing import approximate_token_count


class AsyncReadWriteLock:
    """
    An async-compatible Read-Write Lock.
    
    Allows multiple concurrent readers OR a single exclusive writer.
    Writers have priority to prevent starvation.
    """
    def __init__(self):
        self._read_ready = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._pending_writers = 0
    
    async def acquire_read(self):
        """Acquire a read lock. Multiple readers can hold this simultaneously."""
        async with self._read_ready:
            # Wait while there's an active writer or pending writers (writer priority)
            while self._writer or self._pending_writers > 0:
                await self._read_ready.wait()
            self._readers += 1
    
    async def release_read(self):
        """Release a read lock."""
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
    
    async def acquire_write(self):
        """Acquire an exclusive write lock."""
        async with self._read_ready:
            self._pending_writers += 1
            try:
                # Wait until no readers and no active writer
                while self._readers > 0 or self._writer:
                    await self._read_ready.wait()
                self._writer = True
            finally:
                self._pending_writers -= 1
    
    async def release_write(self):
        """Release the write lock."""
        async with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()
    
    def read_lock(self):
        """Context manager for read lock."""
        return _ReadLockContext(self)
    
    def write_lock(self):
        """Context manager for write lock."""
        return _WriteLockContext(self)


class _ReadLockContext:
    """Async context manager for read lock."""
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock
    
    async def __aenter__(self):
        await self._lock.acquire_read()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_read()
        return False


class _WriteLockContext:
    """Async context manager for write lock."""
    def __init__(self, lock: AsyncReadWriteLock):
        self._lock = lock
    
    async def __aenter__(self):
        await self._lock.acquire_write()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_write()
        return False


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class LocalCompatibleEmbedding(EmbeddingFunction):
    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def __call__(self, input: List[str]):
        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def close(self):
        try:
            self.client.close()
            logger.info("Successfully closed OpenAI client.")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")


class InteractionMemory:
    def __init__(
            self, 
            db_path="./memory_db", 
            collection_name="interaction_memory",
            token_budget: int = 8192,
            is_local_embedding_api: bool = False,
            embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
            embedding_base_url: str = "http://localhost:8000/v1",
            embedding_api_key: str = "EMPTY",
            ):
        if is_local_embedding_api:
            self.embedding_function = LocalCompatibleEmbedding(
                base_url=embedding_base_url,
                model_name=embedding_model_name,
                api_key=embedding_api_key,
            )
        else:
            logger.warning("No embedding function provided. Using default Qwen/Qwen3-Embedding-0.6B model for embeddings. Consider running on a gpu for better performance.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="Qwen/Qwen3-Embedding-0.6B", 
                device=device
                )
        
        self.token_budget = token_budget
        self.collection_name = collection_name
        
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        
        # Read-Write lock for concurrent access
        # Allows multiple concurrent reads (get_examples) but exclusive writes (log_turn)
        self._rw_lock = AsyncReadWriteLock()

    def release(self, should_delete_db: bool = False):
        if isinstance(self.embedding_function, LocalCompatibleEmbedding):
            self.embedding_function.close()
        
        # 1. Optionally delete the database files
        if should_delete_db:
            self.db_client.delete_collection(name=self.collection_name)
        
        # 2. Break references to ChromaDB objects
        # This signals to Python that these heavy objects can be destroyed
        self.collection = None
        self.db_client = None
        self.embedding_function = None

        # 3. Force Garbage Collection
        # This is critical for Vector DBs to actually free the RAM immediately
        gc.collect()
    
    def log_turn(self, role: str, user_input: Union[str, List[str]], assistant_output: Union[str, List[str]]):
        """Synchronous version - use log_turn_async for concurrent access."""
        if isinstance(user_input, str):
            user_input = [user_input]
        if isinstance(assistant_output, str):
            assistant_output = [assistant_output]
        assert len(user_input) == len(assistant_output), "user_input and assistant_output must have the same length"

        turn_ids = []
        metadatas = []
        documents = []
        for u_input, a_output in zip(user_input, assistant_output):
            turn_id = str(uuid.uuid4())
            metadata = {
                    "role": role,
                    "assistant_output": a_output,
                    "timestamp": datetime.now().isoformat()
                }
            turn_ids.append(turn_id)
            metadatas.append(metadata)
            documents.append(u_input)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=turn_ids
        )
    
    async def log_turn_async(self, role: str, user_input: Union[str, List[str]], assistant_output: Union[str, List[str]]):
        """Async version with write lock - exclusive access during writes."""
        async with self._rw_lock.write_lock():
            self.log_turn(role, user_input, assistant_output)
    
    def get_examples(self, role: str, query: str, k: int = 3, strategy: str = "mmr"):
        """
        Synchronous version - use get_examples_async for concurrent access.
        strategy: 'similarity' (standard) or 'mmr' (diverse)
        """
        if strategy == "similarity":
            messages = self._fetch_similarity(role, query, k)
        elif strategy == "mmr":
            messages = self._fetch_mmr(role, query, k)
        else:
            raise ValueError("Unknown strategy. Use 'similarity' or 'mmr'")

        # check for token budget, trim if necessary
        total_tokens = approximate_token_count([msg for pair in messages for msg in pair])
        while total_tokens > self.token_budget and messages:
            messages.pop(0) # remove the oldest example
            total_tokens = approximate_token_count([msg for pair in messages for msg in pair])
        return messages
    
    async def get_examples_async(self, role: str, query: str, k: int = 3, strategy: str = "mmr"):
        """
        Async version with read lock - allows multiple concurrent reads.
        strategy: 'similarity' (standard) or 'mmr' (diverse)
        """
        async with self._rw_lock.read_lock():
            return self.get_examples(role, query, k, strategy)

    def _fetch_similarity(self, role: str, query: str, k: int):
        """Standard KNN search provided by Chroma."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"role": role}
        )
        return self._format_results(results)

    def _fetch_mmr(self, role: str, query: str, k: int, fetch_k=20, lambda_mult=0.6):
        """
        Maximal Marginal Relevance (MMR) Selection.
        
        fetch_k: How many candidates to fetch initially (pool size).
        lambda_mult: 0.0 = Pure Diversity, 1.0 = Pure Relevance. 
        """
        query_embedding = self.collection._embedding_function([query])[0]
        
        # 2. Fetch a larger pool of candidates + their embeddings
        results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where={"role": role},
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not results['embeddings'] or len(results['embeddings'][0]) == 0:
            return []

        # Extract lists for easier handling
        candidates_embeddings = results['embeddings'][0] # List of vectors
        candidates_docs = results['documents'][0]
        candidates_metas = results['metadatas'][0]
        
        # 3. MMR Logic
        selected_indices = []
        candidate_indices = list(range(len(candidates_embeddings)))

        # Convert to numpy for sklearn processing
        query_vec = np.array([query_embedding])
        cand_vecs = np.array(candidates_embeddings)

        for _ in range(min(k, len(candidate_indices))):
            # Calculate similarities
            sim_to_query = cosine_similarity(query_vec, cand_vecs[candidate_indices])[0]
            
            # Sim(Candidates, Selected)
            if selected_indices:
                sim_to_selected = cosine_similarity(
                    cand_vecs[candidate_indices], 
                    cand_vecs[selected_indices]
                )
                # Max similarity to ANY already selected example
                max_sim_to_selected = np.max(sim_to_selected, axis=1)
            else:
                max_sim_to_selected = np.zeros(len(candidate_indices))

            # MMR Score = lambda * Rel - (1-lambda) * Redundancy
            mmr_score = (lambda_mult * sim_to_query) - ((1 - lambda_mult) * max_sim_to_selected)
            
            # Pick the candidate with the highest MMR score
            best_idx_in_subset = np.argmax(mmr_score)
            best_real_idx = candidate_indices[best_idx_in_subset]
            
            selected_indices.append(best_real_idx)
            candidate_indices.pop(best_idx_in_subset)

        # 4. Reconstruct the format based on selected indices
        final_examples = []
        for idx in selected_indices:
            final_examples.append([
                {"role": "user", "content": candidates_docs[idx]},
                {"role": "assistant", "content": candidates_metas[idx]['assistant_output']}
            ])
            
        return final_examples

    def _format_results(self, results):
        """Helper to format Chroma results into list of messages."""
        examples = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                examples.append([
                    {"role": "user", "content": results['documents'][0][i]},
                    {"role": "assistant", "content": results['metadatas'][0][i]['assistant_output']}
                ])
        return examples

        
if __name__ == "__main__":
    memory = InteractionMemory()

    # Let's log similar inputs to see if MMR can filter them
    role = "SQL_Expert"

    # Three very similar examples
    memory.log_turn(role, "How to select all users?", "SELECT * FROM users;")
    memory.log_turn(role, "Select all columns from users table", "SELECT * FROM users;") 
    memory.log_turn(role, "Get every record from users", "SELECT * FROM users;")

    # One distinct example
    memory.log_turn(role, "How to delete a table?", "DROP TABLE users;")

    # User Query
    query = "Show me how to get data from users"

    print("\n--- Strategy: Standard Similarity ---")
    # Likely returns the 2 in 3 "Select" examples because they are semantically closest
    sim_ex = memory.get_examples(role, query, k=2, strategy="similarity")
    for i, ex in enumerate(sim_ex):
        print(f"{i+1}. {ex[0]['content']}")

    print("\n--- Strategy: MMR (Diversity) ---")
    # Should return 1 "Select" example and 1 "Delete" example (to ensure diversity)
    mmr_ex = memory.get_examples(role, query, k=2, strategy="mmr")
    for i, ex in enumerate(mmr_ex):
        print(f"{i+1}. {ex[0]['content']}")

            