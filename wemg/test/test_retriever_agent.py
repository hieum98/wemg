import os
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

import datasets

from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.base_llm_agent import BaseLLMAgent


# Test configuration
TEST_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

# Wiki corpus configuration
WIKI_CORPUS_HF = "Hieuman/wiki23-processed"
WIKI_INDEX_PATH = Path("retriever_corpora/Qwen3-4B-Emb-index.faiss")


class TestRetrieverAgentWikiCorpus:
    """Test suite for RetrieverAgent with actual wiki corpus from HuggingFace."""
    
    @pytest.fixture
    def wiki_embedder_config(self):
        """Create embedder configuration for wiki corpus."""
        return {
            'model_name': TEST_MODEL,
            'url': TEST_API_BASE,
            'api_key': TEST_API_KEY,
            'is_embedding': True,
            'timeout': 60,
        }
    
    @pytest.fixture
    def wiki_retriever_agent(self, wiki_embedder_config):
        """Create a RetrieverAgent instance with wiki corpus and pre-indexed FAISS."""
        # Check if index exists
        if not WIKI_INDEX_PATH.exists():
            pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}. Please ensure the index is available.")
        
        # Try to load corpus from local directory first (if it was previously saved)
        # Otherwise, load from HuggingFace
        local_corpus_path = WIKI_INDEX_PATH.parent / "without_embeddings"
        if local_corpus_path.exists():
            corpus_path = local_corpus_path
        else:
            # Load from HuggingFace - will be saved locally after first load
            corpus_path = Path(WIKI_CORPUS_HF)
        
        agent = RetrieverAgent(
            embedder_config=wiki_embedder_config,
            corpus_path=corpus_path,
            index_path=WIKI_INDEX_PATH,
            embedder_type='openai'
        )
        
        return agent
    
    @pytest.mark.slow
    def test_wiki_retriever_initialization(self, wiki_retriever_agent):
        """Test that wiki retriever initializes correctly with pre-indexed corpus."""
        assert wiki_retriever_agent is not None
        assert wiki_retriever_agent.indexed_corpus is not None
        assert len(wiki_retriever_agent.indexed_corpus) > 0
        
        # Check that corpus has required fields
        assert 'contents' in wiki_retriever_agent.indexed_corpus.column_names
        # assert 'embedding' in wiki_retriever_agent.indexed_corpus.column_names
        
        print(f"✓ Wiki retriever initialized successfully")
        print(f"  Corpus size: {len(wiki_retriever_agent.indexed_corpus)} documents")
        print(f"  Model: {wiki_retriever_agent.model_name}")
        print(f"  Index path: {wiki_retriever_agent.index_path}")
    
    @pytest.mark.slow
    def test_wiki_retrieve_single_query(self, wiki_retriever_agent):
        """Test retrieval from wiki corpus with a single query."""
        query = "What is the capital of France?"
        top_k = 5
        
        contents, scores = wiki_retriever_agent.retrieve(query, top_k=top_k)
        
        # Verify output structure
        assert isinstance(contents, list)
        assert len(contents) == top_k
        assert len(scores) == top_k
        
        # Verify content types
        for content in contents:
            assert isinstance(content, str)
            assert len(content) > 0
        
        # Scores should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores should be in descending order"
        
        print(f"✓ Single query retrieval from wiki corpus")
        print(f"  Query: {query}")
        print(f"  Retrieved {len(contents)} documents")
        print(f"  Top result: {contents[0][:150]}...")
        print(f"  Top score: {scores[0]:.4f}")
        print(f"  Score range: {scores[-1]:.4f} - {scores[0]:.4f}")
    
    @pytest.mark.slow
    def test_wiki_retrieve_batch_queries(self, wiki_retriever_agent):
        """Test batch retrieval from wiki corpus."""
        queries = [
            "Who was the first person to walk on the moon?",
            "What is machine learning?",
            "When was World War II?",
        ]
        top_k = 3
        
        all_contents, all_scores = wiki_retriever_agent.retrieve(queries, top_k=top_k)
        
        # Verify output structure
        assert isinstance(all_contents, list)
        assert len(all_contents) == len(queries)
        assert len(all_scores) == len(queries)
        
        # Verify each query result
        for i, (contents, scores, query) in enumerate(zip(all_contents, all_scores, queries)):
            assert isinstance(contents, list)
            assert len(contents) == top_k
            assert len(scores) == top_k
            
            # Verify content
            for content in contents:
                assert isinstance(content, str)
                assert len(content) > 0
            
            # Verify scores are in descending order
            for j in range(len(scores) - 1):
                assert scores[j] >= scores[j + 1]
            
            print(f"\n✓ Batch query {i+1}: {query}")
            print(f"  Top result: {contents[0][:120]}...")
            print(f"  Top score: {scores[0]:.4f}")
    
    @pytest.mark.slow
    def test_wiki_retrieve_relevance(self, wiki_retriever_agent):
        """Test that wiki retrieval returns relevant documents."""
        test_cases = [
            ("capital of France", ["paris", "france"]),
            ("Einstein theory of relativity", ["einstein", "relativity"]),
            ("World War II dates", ["world war", "1945", "1939"]),
            ("machine learning algorithms", ["machine learning", "algorithm"]),
            ("Shakespeare plays", ["shakespeare", "play"]),
        ]
        
        for query, expected_keywords in test_cases:
            contents, scores = wiki_retriever_agent.retrieve(query, top_k=3)
            
            assert len(contents) == 3
            assert len(scores) == 3
            
            # Check if top result contains at least one expected keyword
            top_result = contents[0].lower()
            found = any(keyword.lower() in top_result for keyword in expected_keywords)
            
            print(f"✓ Query: '{query}'")
            print(f"  Top result: {contents[0][:120]}...")
            print(f"  Score: {scores[0]:.4f}")
            print(f"  Relevant: {found}")
            
            # At least one of the top 3 should be relevant
            found_in_top3 = any(
                any(kw.lower() in content.lower() for kw in expected_keywords)
                for content in contents[:3]
            )
            assert found_in_top3, f"Expected relevant content for query: {query}"
    
    @pytest.mark.slow
    def test_wiki_retrieve_different_top_k(self, wiki_retriever_agent):
        """Test retrieval with different top_k values."""
        query = "What is artificial intelligence?"
        
        for top_k in [1, 3, 5, 10]:
            contents, scores = wiki_retriever_agent.retrieve(query, top_k=top_k)
            
            assert len(contents) == top_k
            assert len(scores) == top_k
            
            # Verify scores are in descending order
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]
            
            print(f"✓ top_k={top_k}: Retrieved {len(contents)} documents, score range: {scores[-1]:.4f} - {scores[0]:.4f}")
    
    @pytest.mark.slow
    def test_wiki_retrieve_empty_query(self, wiki_retriever_agent):
        """Test retrieval with empty or very short query."""
        query = "test"
        contents, scores = wiki_retriever_agent.retrieve(query, top_k=5)
        
        assert len(contents) == 5
        assert len(scores) == 5
        
        print(f"✓ Empty/short query handled")
        print(f"  Query: '{query}'")
        print(f"  Retrieved {len(contents)} documents")
    
    @pytest.mark.slow
    def test_wiki_retrieve_long_query(self, wiki_retriever_agent):
        """Test retrieval with a long, complex query."""
        query = "What are the main causes and consequences of climate change, and what are the potential solutions?"
        contents, scores = wiki_retriever_agent.retrieve(query, top_k=5)
        
        assert len(contents) == 5
        assert len(scores) == 5
        
        # Check if results are relevant to climate change
        found_climate = any(
            "climate" in content.lower() or "warming" in content.lower() or "environment" in content.lower()
            for content in contents[:3]
        )
        
        print(f"✓ Long query handled")
        print(f"  Query length: {len(query)} characters")
        print(f"  Top result: {contents[0][:120]}...")
        print(f"  Relevant to climate: {found_climate}")
    
    @pytest.mark.slow
    def test_wiki_retrieve_entity_queries(self, wiki_retriever_agent):
        """Test retrieval with entity-focused queries."""
        entity_queries = [
            "Barack Obama",
            "Eiffel Tower",
            "Python programming language",
            "Mount Everest",
        ]
        
        for query in entity_queries:
            contents, scores = wiki_retriever_agent.retrieve(query, top_k=3)
            
            assert len(contents) == 3
            assert len(scores) == 3
            
            # Top result should mention the entity
            top_result = contents[0].lower()
            entity_keywords = query.lower().split()
            found_entity = any(keyword in top_result for keyword in entity_keywords)
            
            print(f"✓ Entity query: '{query}'")
            print(f"  Top result mentions entity: {found_entity}")
            print(f"  Top score: {scores[0]:.4f}")
            
            assert found_entity, f"Top result should mention entity from query: {query}"


if __name__ == "__main__":
    # Run the tests with verbose output and logging
    # Options:
    # -v: verbose output
    # -s: show print statements
    # --tb=short: short traceback format
    # --log-cli-level=DEBUG: show all log messages (DEBUG, INFO, WARNING, ERROR)
    # --log-cli-level=INFO: show INFO and above (default)
    pytest.main([__file__, "-v", "-s", "--tb=short", "--log-cli-level=DEBUG"])
