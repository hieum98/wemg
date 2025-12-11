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
TEST_API_BASE = "http://n0372:4000/v1"
TEST_API_KEY = "sk-your-very-secure-master-key-here"
TEST_MODEL = "Qwen3-Embedding-4B"


class TestRetrieverAgent:
    """Test suite for RetrieverAgent functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        # Create temp directory in current path instead of /tmp/
        current_path = Path.cwd()
        temp_dir = tempfile.mkdtemp(prefix="test_retriever_", dir=current_path)
        yield Path(temp_dir)
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_corpus_jsonl(self, temp_dir):
        """Create a sample corpus in JSONL format."""
        corpus_path = temp_dir / "test_corpus.jsonl"
        
        # Sample documents about various topics
        documents = [
            {"contents": "Python is a high-level programming language known for its simplicity and readability."},
            {"contents": "Machine learning is a subset of artificial intelligence that focuses on learning from data."},
            {"contents": "Natural language processing enables computers to understand and generate human language."},
            {"contents": "Deep learning uses neural networks with multiple layers to learn complex patterns."},
            {"contents": "Computer vision allows machines to interpret and understand visual information from images."},
            {"contents": "The quick brown fox jumps over the lazy dog is a common typing test sentence."},
            {"contents": "Climate change refers to long-term shifts in temperatures and weather patterns."},
            {"contents": "Renewable energy sources include solar, wind, and hydroelectric power."},
            {"contents": "Quantum computing uses quantum mechanics to process information in fundamentally new ways."},
            {"contents": "Blockchain technology provides a decentralized and secure way to record transactions."},
        ]
        
        with open(corpus_path, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        
        return corpus_path
    
    @pytest.fixture
    def embedder_config(self):
        """Create embedder configuration for testing."""
        return {
            'model_name': TEST_MODEL,
            'url': TEST_API_BASE,
            'api_key': TEST_API_KEY,
            'is_embedding': True,
            'timeout': 60,
        }
    
    @pytest.fixture
    def retriever_agent(self, embedder_config, sample_corpus_jsonl, temp_dir):
        """Create a RetrieverAgent instance for testing."""
        index_path = temp_dir / "test_index.faiss"
        
        agent = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=sample_corpus_jsonl,
            index_path=index_path,
            embedder_type='openai'
        )
        
        return agent
    
    def test_get_index_from_scratch(self, embedder_config, sample_corpus_jsonl, temp_dir):
        """Test get_index method when building index from scratch."""
        # Create a new agent without pre-existing index
        index_path = temp_dir / "new_index.faiss"
        
        agent = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=sample_corpus_jsonl,
            index_path=index_path,
            embedder_type='openai'
        )
        
        # get_index is called in __init__, so indexed_corpus should be ready
        assert agent.indexed_corpus is not None
        assert isinstance(agent.indexed_corpus, datasets.Dataset)
        
        # Check that corpus has the expected fields
        assert 'contents' in agent.indexed_corpus.column_names
        assert 'embedding' in agent.indexed_corpus.column_names
        
        # Check that we have all documents
        assert len(agent.indexed_corpus) == 10
        
        # Check that index file was created
        assert index_path.exists()
        
        print(f"✓ Index created from scratch with {len(agent.indexed_corpus)} documents")
        print(f"✓ Index saved at: {index_path}")
    
    def test_get_index_from_existing(self, embedder_config, sample_corpus_jsonl, temp_dir):
        """Test get_index method when loading from existing index."""
        index_path = temp_dir / "existing_index.faiss"
        
        # First, create an agent to build the index
        agent1 = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=sample_corpus_jsonl,
            index_path=index_path,
            embedder_type='openai'
        )
        
        assert index_path.exists()
        first_indexed_corpus_size = len(agent1.indexed_corpus)
        
        # Now create a second agent that should load the existing index
        agent2 = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=sample_corpus_jsonl,
            index_path=index_path,
            embedder_type='openai'
        )
        
        assert agent2.indexed_corpus is not None
        assert isinstance(agent2.indexed_corpus, datasets.Dataset)
        assert len(agent2.indexed_corpus) == first_indexed_corpus_size
        
        print(f"✓ Index loaded from existing file")
        print(f"✓ Loaded {len(agent2.indexed_corpus)} documents from index")
    
    def test_retrieve_single_query(self, retriever_agent):
        """Test retrieve method with a single query."""
        query = "What is machine learning?"
        top_k = 3
        
        contents, scores = retriever_agent.retrieve(query, top_k=top_k)
        
        # Verify output structure
        assert isinstance(contents, list)
        assert len(contents) == top_k
        assert len(scores) == top_k
        
        # Verify content types
        for content in contents:
            assert isinstance(content, str)
            assert len(content) > 0
        
        # Scores should be in descending order (higher score = more relevant)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores should be in descending order"
        
        print(f"✓ Single query retrieved {len(contents)} documents")
        print(f"✓ Query: {query}")
        print(f"✓ Top result: {contents[0][:80]}...")
        print(f"✓ Top score: {scores[0]:.4f}")
    
    def test_retrieve_batch_queries(self, retriever_agent):
        """Test retrieve method with multiple queries."""
        queries = [
            "What is Python programming?",
            "Tell me about climate change",
            "What is quantum computing?",
        ]
        top_k = 2
        
        all_contents, all_scores = retriever_agent.retrieve(queries, top_k=top_k)
        
        # Verify output structure
        assert isinstance(all_contents, list)
        assert len(all_contents) == len(queries)
        assert len(all_scores) == len(queries)
        
        # Verify each query result
        for i, (contents, scores) in enumerate(zip(all_contents, all_scores)):
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
            
            print(f"✓ Query {i+1}: {queries[i]}")
            print(f"  Top result: {contents[0][:60]}...")
            print(f"  Score: {scores[0]:.4f}")
    
    def test_retrieve_relevance(self, retriever_agent):
        """Test that retrieve returns relevant documents."""
        # Query about programming
        query = "programming language Python"
        contents, scores = retriever_agent.retrieve(query, top_k=3)
        
        # The top result should contain relevant keywords
        top_result = contents[0].lower()
        assert any(keyword in top_result for keyword in ['python', 'programming', 'language'])
        
        print(f"✓ Query: {query}")
        print(f"✓ Top result is relevant: {contents[0][:80]}...")
        
        # Query about climate
        query2 = "climate and renewable energy"
        contents2, scores2 = retriever_agent.retrieve(query2, top_k=3)
        
        # Top results should be about climate or energy
        found_relevant = False
        for content in contents2[:2]:  # Check top 2 results
            if any(keyword in content.lower() for keyword in ['climate', 'energy', 'renewable']):
                found_relevant = True
                break
        
        assert found_relevant, "Should retrieve relevant documents for climate query"
        print(f"✓ Query: {query2}")
        print(f"✓ Retrieved relevant climate/energy documents")

    def test_retrieve_from_loaded_index(self, retriever_agent):
        """Test retrieval from a previously loaded index."""
        # First retrieve some documents
        query = "natural language processing"
        contents1, scores1 = retriever_agent.retrieve(query, top_k=2)
        
        # Now create a new agent that loads the existing index
        new_agent = RetrieverAgent(
            embedder_config=retriever_agent.embedder_config,
            corpus_path=retriever_agent.corpus_path / "without_embeddings",
            index_path=retriever_agent.index_path,
            embedder_type='openai'
        )
        
        contents2, scores2 = new_agent.retrieve(query, top_k=2)
        breakpoint()
        # The results should be the same
        assert contents1 == contents2
        
        print(f"✓ Retrieval from loaded index matches original retrieval")
        print(f"✓ Query: {query}")
        print(f"✓ Top result: {contents1[0][:80]}...")
        print(f"✓ Score: {scores1[0]:.4f}")


class TestRetrieverAgentIntegration:
    """Integration tests for RetrieverAgent with real use cases."""
    
    @pytest.fixture
    def qa_corpus(self, tmp_path):
        """Create a Q&A corpus for testing."""
        corpus_path = tmp_path / "qa_corpus.jsonl"
        
        qa_pairs = [
            {"contents": "Q: What is the capital of France? A: Paris is the capital of France."},
            {"contents": "Q: What is the speed of light? A: The speed of light is approximately 299,792 kilometers per second."},
            {"contents": "Q: Who wrote Romeo and Juliet? A: William Shakespeare wrote Romeo and Juliet."},
            {"contents": "Q: What is photosynthesis? A: Photosynthesis is the process by which plants convert light energy into chemical energy."},
            {"contents": "Q: What is the largest planet? A: Jupiter is the largest planet in our solar system."},
            {"contents": "Q: What is the human genome? A: The human genome is the complete set of genetic information in human DNA."},
            {"contents": "Q: What is blockchain? A: Blockchain is a distributed ledger technology that records transactions across multiple computers."},
            {"contents": "Q: What is machine learning? A: Machine learning is a subset of AI that enables systems to learn from data."},
        ]
        
        with open(corpus_path, 'w') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + '\n')
        
        return corpus_path
    
    @pytest.fixture
    def qa_retriever(self, qa_corpus, tmp_path):
        """Create a retriever for Q&A corpus."""
        embedder_config = {
            'model_name': TEST_MODEL,
            'url': TEST_API_BASE,
            'api_key': TEST_API_KEY,
            'is_embedding': True,
            'timeout': 60,
        }
        
        index_path = tmp_path / "qa_index.faiss"
        
        agent = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=qa_corpus,
            index_path=index_path,
            embedder_type='openai'
        )
        
        return agent
    
    def test_qa_retrieval(self, qa_retriever):
        """Test retrieval on Q&A corpus."""
        # Test various queries
        test_cases = [
            ("capital of France", ["paris", "france"]),
            ("speed of light", ["light", "speed"]),
            ("Shakespeare plays", ["shakespeare"]),
            ("largest planet", ["planet", "jupiter"]),
            ("machine learning AI", ["machine learning", "ai"]),
        ]
        
        for query, expected_keywords in test_cases:
            contents, scores = qa_retriever.retrieve(query, top_k=1)
            
            assert len(contents) == 1
            top_result = contents[0].lower()
            
            # Check if at least one expected keyword is in the result
            found = any(keyword.lower() in top_result for keyword in expected_keywords)
            
            print(f"✓ Query: '{query}'")
            print(f"  Result: {contents[0][:80]}...")
            print(f"  Score: {scores[0]:.4f}")
            print(f"  Relevant: {found}")
    
    def test_batch_qa_retrieval(self, qa_retriever):
        """Test batch retrieval on Q&A corpus."""
        queries = [
            "What is the capital of France?",
            "Tell me about the largest planet",
            "Explain machine learning",
        ]
        
        all_contents, all_scores = qa_retriever.retrieve(queries, top_k=2)
        
        assert len(all_contents) == 3
        assert len(all_scores) == 3
        
        for i, (contents, scores, query) in enumerate(zip(all_contents, all_scores, queries)):
            assert len(contents) == 2
            assert len(scores) == 2
            
            print(f"\n✓ Batch query {i+1}: {query}")
            print(f"  Top result: {contents[0][:80]}...")
            print(f"  Score: {scores[0]:.4f}")


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
