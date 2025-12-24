"""Profile generate_answer function to identify performance bottlenecks."""
import asyncio
import cProfile
import pstats
import io
from pstats import SortKey
import time
import os
from pathlib import Path

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool
from wemg.runners.procedures.node_generator import NodeGenerator
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory


def profile_generate_answer():
    """Profile generate_answer with a test case."""
    # Setup - use environment variables or defaults
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
        max_tokens=65536,
        concurrency=2,
        max_retries=3
    )
    
    # Use WebSearchTool for retrieval
    serper_api_key = os.getenv("SERPER_API_KEY", "your-serper-api-key")
    retriever_agent = WebSearchTool(serper_api_key=serper_api_key)
    
    working_memory = WorkingMemory()
    interaction_memory = InteractionMemory()
    
    generator = NodeGenerator(
        llm_agent=llm_agent,
        retriever_agent=retriever_agent,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        top_k_websearch=5,
        top_k_entities=1,
        top_k_properties=1,
        n_hops=1,
        n=1
    )
    
    question = "Which magazine was started first Arthur's Magazine or First for Women?"
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the async function
    print("="*80)
    print("PROFILING generate_answer")
    print("="*80)
    print(f"Question: {question}")
    print("\nProfiling...")
    start_time = time.time()
    
    profiler.enable()
    result = asyncio.run(generator.generate_answer(question))
    profiler.disable()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✓ Execution completed in {elapsed_time:.2f} seconds")
    print(f"✓ Generated {len(result.answers)} answer(s)")
    if result.answers:
        print(f"✓ Answer preview: {result.answers[0].answer[:200]}...")
    print(f"✓ Retrieved {len(result.retrieved_triples)} triples")
    print(f"✓ Found {len(result.entity_dict)} entities")
    print(f"✓ Found {len(result.property_dict)} properties")
    
    # Create stats sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("="*80)
    print(s.getvalue())
    
    # Also print by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(20)  # Top 20 functions
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by total time)")
    print("="*80)
    print(s2.getvalue())
    
    # Save detailed profile to file
    profile_file = 'generate_answer_profile.prof'
    profiler.dump_stats(profile_file)
    print(f"\n✓ Detailed profile saved to '{profile_file}'")
    print(f"  View with: python -m pstats {profile_file}")


if __name__ == "__main__":
    profile_generate_answer()

