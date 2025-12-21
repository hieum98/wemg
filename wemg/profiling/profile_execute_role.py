"""Profile execute_role function to identify performance bottlenecks."""
import asyncio
import cProfile
import pstats
import io
from pstats import SortKey
import time
import os

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.interaction_memory import InteractionMemory


def profile_execute_role():
    """Profile execute_role with a simple test case."""
    # Setup - use environment variables or defaults
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
    )
    interaction_memory = InteractionMemory()
    
    input_data = roles.generator.SubquestionGenerationInput(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Paris is the capital and largest city of France."
    )
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the async function
    print("Profiling execute_role...")
    start_time = time.time()
    
    profiler.enable()
    results, log = asyncio.run(execute_role(
        llm_agent=llm_agent,
        role=roles.generator.SubquestionGenerator(),
        input_data=input_data,
        interaction_memory=interaction_memory,
        n=1
    ))
    profiler.disable()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✓ Execution completed in {elapsed_time:.2f} seconds")
    print(f"✓ Results: {len(results)} output(s)")
    
    # Create stats
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
    profiler.dump_stats('execute_role_profile.prof')
    print("\n✓ Detailed profile saved to 'execute_role_profile.prof'")
    print("  View with: python -m pstats execute_role_profile.prof")


if __name__ == "__main__":
    profile_execute_role()

