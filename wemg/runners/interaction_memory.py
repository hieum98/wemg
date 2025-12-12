from datetime import datetime
import gc
from typing import List
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
    
    def log_turn(self, role: str, user_input: str, assistant_output: str):
        turn_id = str(uuid.uuid4())
        metadata = {
                "role": role,
                "assistant_output": assistant_output,
                "timestamp": datetime.now().isoformat()
            }
        self.collection.add(
            documents=[user_input],
            metadatas=[metadata],
            ids=[turn_id]
        )
    
    def get_examples(self, role: str, query: str, k: int = 3, strategy: str = "mmr"):
        """
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

            