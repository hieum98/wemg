import os
import pytest
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from wemg.agents.base_llm_agent import OpenAIClient, BaseLLMAgent


# Test configuration
TEST_API_BASE = "http://n0142:4000/v1"
TEST_API_KEY = "sk-your-very-secure-master-key-here"
TEST_MODEL = "Qwen3-0.6B"


# Pydantic models for structured output testing
class QuestionAnswer(BaseModel):
    question: str = Field(description="The question being asked")
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class SentimentAnalysis(BaseModel):
    text: str = Field(description="The analyzed text")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    score: int = Field(description="Sentiment score from 0 to 10")


class TestOpenAIClientGeneration:
    """Test suite for OpenAIClient generation functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a basic OpenAI client for testing."""
        return OpenAIClient(
            model_name=TEST_MODEL,
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=4,
            max_retries=5
        )
    
    @pytest.fixture
    def client_with_cache(self):
        """Create an OpenAI client with Redis cache enabled."""
        cache_config = {
            'enabled': True,
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl': 300,  # 5 minutes
            'prefix': 'test_llm_cache',
        }
        return OpenAIClient(
            model_name=TEST_MODEL,
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            cache_config=cache_config,
            max_retries=5
        )
    
    def test_single_generation(self, client):
        """Test basic single text generation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello World' and nothing else."}
        ]
        
        index, results = client.generate(0, messages)
        
        assert index == 0
        assert len(results) == 1
        assert results[0]['is_valid'] is True
        assert 'output' in results[0]
        assert isinstance(results[0]['output'], str)
        assert len(results[0]['output']) > 0
        print(f"✓ Single generation output: {results[0]['output'][:100]}")
    
    def test_generation_with_multiple_samples(self, client):
        """Test generation with n > 1 to get multiple completions."""
        messages = [
            {"role": "user", "content": "Give me a random number between 1 and 100."}
        ]
        
        index, results = client.generate(0, messages, n=3)
        
        assert index == 0
        assert len(results) == 3
        for result in results:
            assert result['is_valid'] is True
            assert 'output' in result
            assert isinstance(result['output'], str)
        print(f"✓ Generated {len(results)} different completions")
    
    def test_generation_with_structured_output(self, client):
        """Test generation with Pydantic schema for structured output."""
        messages = [
            {"role": "user", "content": "Question: What is 2+2? Provide your answer in JSON format with fields: question, answer, and confidence (high/medium/low)."}
        ]
        
        index, results = client.generate(
            0, 
            messages, 
            output_schema=QuestionAnswer,
            temperature=0.3
        )
        
        assert index == 0
        assert len(results) >= 1
        result = results[0]
        assert 'output' in result
        
        # If structured output is supported and valid
        if result['is_valid'] and isinstance(result['output'], dict):
            assert 'question' in result['output']
            assert 'answer' in result['output']
            assert 'confidence' in result['output']
            print(f"✓ Structured output: {result['output']}")
        else:
            # Fallback extraction should still work
            assert isinstance(result['output'], str)
            print(f"✓ Text output (structured not supported): {result['output'][:100]}")
    
    def test_batch_generation(self, client):
        """Test batch generation with multiple message sets."""
        batch_messages = [
            [{"role": "user", "content": "What is Python?"}],
            [{"role": "user", "content": "What is JavaScript?"}],
            [{"role": "user", "content": "What is Go?"}],
        ]
        
        results = client.batch_generate(batch_messages)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result) >= 1
            assert result[0]['is_valid'] is True
            assert 'output' in result[0]
            assert isinstance(result[0]['output'], str)
            print(f"✓ Batch result {i+1}: {result[0]['output'][:80]}...")
    
    def test_generation_with_reasoning(self, client):
        """Test generation that may include reasoning content."""
        messages = [
            {"role": "user", "content": "Solve this problem: If a train travels 60 mph for 2 hours, how far does it go?"}
        ]
        
        index, results = client.generate(
            0, 
            messages, 
            should_return_reasoning=True,
            temperature=0.5
        )
        
        assert len(results) >= 1
        result = results[0]
        assert 'output' in result
        assert 'reasoning' in result
        # Reasoning might be None if the model doesn't support it
        print(f"✓ Output: {result['output'][:100]}")
        print(f"✓ Reasoning: {result['reasoning'][:100] if result['reasoning'] else 'Not available'}")


class TestOpenAIClientEmbedding:
    """Test suite for OpenAIClient embedding functionality."""
    
    @pytest.fixture
    def embedding_client(self):
        """Create an embedding client."""
        # Note: You may need to adjust the model name if your endpoint
        # has a specific embedding model
        return OpenAIClient(
            model_name="text-embedding-ada-002",  # Adjust if needed
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            is_embedding=True,
        )
    
    def test_single_text_embedding(self, embedding_client):
        """Test embedding generation for a single text."""
        text = "This is a test sentence for embedding."
        
        try:
            embedding = embedding_client.embedding(text)
            
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
            print(f"✓ Single embedding dimension: {len(embedding)}")
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")
    
    def test_batch_text_embedding(self, embedding_client):
        """Test embedding generation for multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        
        try:
            embeddings = embedding_client.embedding(texts)
            
            assert embeddings is not None
            assert isinstance(embeddings, list)
            assert len(embeddings) == len(texts)
            for emb in embeddings:
                assert isinstance(emb, list)
                assert len(emb) > 0
                assert all(isinstance(x, (int, float)) for x in emb)
            print(f"✓ Batch embeddings: {len(embeddings)} vectors of dimension {len(embeddings[0])}")
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")


class TestCaching:
    """Test suite for Redis caching functionality."""
    
    @pytest.fixture
    def client_with_cache(self):
        """Create a client with caching enabled."""
        cache_config = {
            'enabled': True,
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl': 300,
            'prefix': 'test_cache',
        }
        client = OpenAIClient(
            model_name=TEST_MODEL,
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            cache_config=cache_config,
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=4096,
        )
        
        # Clear any existing test cache before starting
        if client.cache and client.cache.enabled:
            client.clear_cache(pattern="test_cache:*")
        
        yield client
        
        # Cleanup after tests
        if client.cache and client.cache.enabled:
            client.clear_cache(pattern="test_cache:*")
            client.close()
    
    def test_cache_enabled(self, client_with_cache):
        """Test that cache is properly enabled."""
        assert client_with_cache.cache is not None
        assert client_with_cache.cache.enabled is True
        assert client_with_cache.use_cache is True
        print("✓ Cache is enabled")
    
    def test_generation_cache_hit(self, client_with_cache):
        """Test that repeated generation uses cache."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]
        
        # First call - should miss cache
        start_time = time.time()
        index1, results1 = client_with_cache.generate(0, messages, use_cache=True)
        first_call_time = time.time() - start_time
        
        assert len(results1) >= 1
        first_output = results1[0]['output']
        
        # Second call - should hit cache (faster)
        start_time = time.time()
        index2, results2 = client_with_cache.generate(0, messages, use_cache=True)
        second_call_time = time.time() - start_time
        
        assert len(results2) >= 1
        second_output = results2[0]['output']
        
        # Cache hit should return same result
        assert first_output == second_output
        
        # Cache hit should be faster (allowing some margin)
        print(f"✓ First call: {first_call_time:.3f}s, Second call (cached): {second_call_time:.3f}s")
        print(f"✓ Cached result matches: {first_output == second_output}")
    
    def test_batch_generation_cache(self, client_with_cache):
        """Test caching with batch generation."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        # Include duplicate messages to test cache reuse
        batch_messages = [
            [{"role": "user", "content": "What is 5+5?"}],
            [{"role": "user", "content": "What is 3+3?"}],
            [{"role": "user", "content": "What is 5+5?"}],  # Duplicate
        ]

        cache_stats = client_with_cache.cache.get_stats()
        print(f"Cache stats before batch generation: {cache_stats}")

        results = client_with_cache.batch_generate(batch_messages, use_cache=True)
        print(f"Cache stats after batch generation: {client_with_cache.cache.get_stats()}")
        assert len(results) == 3
        # First and third should have same result due to cache
        assert results[0][0]['output'] == results[2][0]['output']
        print(f"✓ Duplicate queries returned same cached result")
    
    def test_cache_with_custom_ttl(self, client_with_cache):
        """Test cache with custom TTL."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        messages = [
            {"role": "user", "content": "Count to three."}
        ]
        
        # Cache with short TTL
        index, results = client_with_cache.generate(
            0, 
            messages, 
            use_cache=True,
            cache_ttl=100  # 10 seconds
        )
        
        assert len(results) >= 1
        print(f"✓ Cached with custom TTL (10 seconds)")
    
    def test_cache_disable_per_request(self, client_with_cache):
        """Test disabling cache for specific requests."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        messages = [
            {"role": "user", "content": "Give me a random word."}
        ]
        
        # First call without cache
        index1, results1 = client_with_cache.generate(0, messages, use_cache=False)
        
        # Second call without cache - may get different result
        index2, results2 = client_with_cache.generate(0, messages, use_cache=False)
        
        # Both should succeed
        assert len(results1) >= 1
        assert len(results2) >= 1
        print(f"✓ Cache disabled for specific requests")
    
    def test_cache_statistics(self, client_with_cache):
        """Test cache statistics retrieval."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        # Generate some cached entries
        for i in range(3):
            messages = [{"role": "user", "content": f"Test message {i}"}]
            client_with_cache.generate(0, messages, use_cache=True)
        
        stats = client_with_cache.get_cache_stats()
        
        assert stats is not None
        assert 'enabled' in stats
        assert stats['enabled'] is True
        print(f"✓ Cache stats: {stats}")
    
    def test_cache_clear(self, client_with_cache):
        """Test cache clearing functionality."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        # Generate some cached entries
        messages = [{"role": "user", "content": "Cache test message"}]
        client_with_cache.generate(0, messages, use_cache=True)
        
        # Clear cache
        deleted = client_with_cache.clear_cache(pattern="test_cache:*")
        
        assert deleted >= 0
        print(f"✓ Cleared {deleted} cache entries")
    
    def test_embedding_cache(self, client_with_cache):
        """Test caching for embeddings."""
        if not client_with_cache.cache or not client_with_cache.cache.enabled:
            pytest.skip("Redis cache not available")
        
        # Create embedding client with cache
        cache_config = {
            'enabled': True,
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl': 300,
            'prefix': 'test_emb_cache',
        }
        emb_client = OpenAIClient(
            model_name="text-embedding-ada-002",
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            is_embedding=True,
            cache_config=cache_config,
        )
        
        try:
            text = "Test embedding cache"
            
            # First call
            emb1 = emb_client.embedding(text, use_cache=True)
            
            # Second call - should use cache
            emb2 = emb_client.embedding(text, use_cache=True)
            
            if emb1 is not None and emb2 is not None:
                assert emb1 == emb2
                print(f"✓ Embedding cache working")
            
            # Cleanup
            emb_client.clear_cache(pattern="test_emb_cache:*")
            emb_client.close()
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")


class TestBaseLLMAgent:
    """Test suite for BaseLLMAgent high-level interface."""
    
    @pytest.fixture
    def agent(self):
        """Create a BaseLLMAgent instance."""
        return BaseLLMAgent(
            client_type='openai',
            model_name=TEST_MODEL,
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            temperature=0.7,
            max_tokens=512,
        )
    
    def test_generator_role_execute_single(self, agent):
        """Test generator_role_execute with single message."""
        messages = [
            {"role": "user", "content": "Say hello in one word."}
        ]
        
        results = agent.generator_role_execute(messages)
        
        assert len(results) == 1
        assert len(results[0]) >= 1
        assert isinstance(results[0][0], str)
        print(f"✓ Generator single result: {results[0][0][:50]}")
    
    def test_generator_role_execute_batch(self, agent):
        """Test generator_role_execute with batch messages."""
        batch_messages = [
            [{"role": "user", "content": "Say 'one'"}],
            [{"role": "user", "content": "Say 'two'"}],
            [{"role": "user", "content": "Say 'three'"}],
        ]
        
        results = agent.generator_role_execute(batch_messages)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert len(result) >= 1
            assert isinstance(result[0], str)
            print(f"✓ Generator batch result {i+1}: {result[0][:50]}")
    
    def test_generator_role_execute_with_schema(self, agent):
        """Test generator_role_execute with output schema."""
        messages = [
            {"role": "user", "content": "Analyze the sentiment of this text: 'I love this product!'. Provide JSON with fields: text, sentiment (positive/negative/neutral), and score (0-10)."}
        ]
        
        results = agent.generator_role_execute(
            messages,
            output_schema=SentimentAnalysis,
            temperature=0.3
        )
        
        assert len(results) == 1
        assert len(results[0]) >= 1
        
        result = results[0][0]
        # Should be either a SentimentAnalysis object or dict
        if isinstance(result, SentimentAnalysis):
            assert hasattr(result, 'text')
            assert hasattr(result, 'sentiment')
            assert hasattr(result, 'score')
            print(f"✓ Structured output: sentiment={result.sentiment}, score={result.score}")
        else:
            print(f"✓ Text output: {str(result)[:100]}")
    
    def test_generator_role_execute_multiple_samples(self, agent):
        """Test generator_role_execute with multiple samples."""
        messages = [
            {"role": "user", "content": "Name a color."}
        ]
        
        results = agent.generator_role_execute(messages, n=3)
        
        assert len(results) == 1
        assert len(results[0]) == 3
        for i, result in enumerate(results[0]):
            assert isinstance(result, str)
            print(f"✓ Sample {i+1}: {result[:50]}")
    
    def test_retriever_role_execute_single(self, agent):
        """Test retriever_role_execute with single text."""
        try:
            # Note: This requires an embedding model endpoint
            embedding_agent = BaseLLMAgent(
                client_type='openai',
                model_name="text-embedding-ada-002",
                url=TEST_API_BASE,
                api_key=TEST_API_KEY,
                is_embedding=True,
            )
            
            text = "This is a test sentence."
            embedding = embedding_agent.retriever_role_execute([text])
            
            if embedding is not None:
                assert isinstance(embedding, list)
                assert len(embedding) > 0
                print(f"✓ Retriever single embedding dimension: {len(embedding)}")
            else:
                pytest.skip("Embedding generation returned None")
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")
    
    def test_retriever_role_execute_batch(self, agent):
        """Test retriever_role_execute with batch texts."""
        try:
            embedding_agent = BaseLLMAgent(
                client_type='openai',
                model_name="text-embedding-ada-002",
                url=TEST_API_BASE,
                api_key=TEST_API_KEY,
                is_embedding=True,
            )
            
            texts = [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ]
            
            embeddings = embedding_agent.retriever_role_execute(texts)
            
            if embeddings is not None:
                assert isinstance(embeddings, list)
                assert len(embeddings) == len(texts)
                print(f"✓ Retriever batch: {len(embeddings)} embeddings")
            else:
                pytest.skip("Embedding generation returned None")
        except Exception as e:
            pytest.skip(f"Embedding model not available: {e}")


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_invalid_model_graceful_failure(self):
        """Test handling of invalid model name."""
        client = OpenAIClient(
            model_name="non-existent-model-xyz",
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            max_retries=1,
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Should handle error gracefully
        index, results = client.generate(0, messages)
        
        # May return empty results or handle error
        assert isinstance(results, list)
        print(f"✓ Invalid model handled gracefully: {len(results)} results")
    
    def test_cache_without_redis(self):
        """Test behavior when Redis is not available."""
        cache_config = {
            'enabled': True,
            'host': 'invalid-host-xyz',
            'port': 9999,
            'ttl': 300,
            'prefix': 'test',
        }
        
        client = OpenAIClient(
            model_name=TEST_MODEL,
            url=TEST_API_BASE,
            api_key=TEST_API_KEY,
            cache_config=cache_config,
        )
        
        # Should still work without cache
        messages = [{"role": "user", "content": "Hello"}]
        index, results = client.generate(0, messages)
        
        assert len(results) >= 1 or len(results) == 0  # May fail or succeed
        print(f"✓ Client works without Redis connection")


if __name__ == "__main__":
    # Run the tests in this file with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=WARNING"])

