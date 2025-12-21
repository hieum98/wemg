"""Test script to verify interaction memory optimizations."""
import asyncio
import time
import os
from wemg.runners.interaction_memory import InteractionMemory
from wemg.agents import roles


async def test_optimization():
    """Test the optimized interaction memory."""
    print("="*80)
    print("TESTING INTERACTION MEMORY OPTIMIZATION")
    print("="*80)
    
    # Create memory instance
    memory = InteractionMemory(enable_embedding_cache=True)
    
    # Add some test data
    role = "test_role"
    print("\n1. Adding test data...")
    for i in range(10):
        memory.log_turn(
            role, 
            f"Question {i}: What is the capital of country {i}?",
            f"Answer {i}: The capital is City {i}."
        )
    print(f"   Added 10 examples")
    
    # Test query
    query = "What is the capital of France?"
    
    # Test MMR with timing
    print("\n2. Testing MMR search (optimized):")
    start = time.time()
    results = memory.get_examples(role, query, k=3, strategy="mmr")
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Results: {len(results)} examples")
    
    # Test similarity with timing
    print("\n3. Testing similarity search:")
    start = time.time()
    results = memory.get_examples(role, query, k=3, strategy="similarity")
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Results: {len(results)} examples")
    
    # Test async version
    print("\n4. Testing async MMR search:")
    start = time.time()
    results = await memory.get_examples_async(role, query, k=3, strategy="mmr")
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s")
    print(f"   Results: {len(results)} examples")
    
    # Test cache hit (same query)
    print("\n5. Testing cache hit (same query):")
    start = time.time()
    results = memory.get_examples(role, query, k=3, strategy="mmr")
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s (should be faster due to cache)")
    print(f"   Results: {len(results)} examples")
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_optimization())

