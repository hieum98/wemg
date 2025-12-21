"""Detailed profiling of execute_role function with timing breakdown."""
import asyncio
import time
import os
from contextlib import contextmanager

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.interaction_memory import InteractionMemory


@contextmanager
def timer(name):
    """Context manager to time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")


async def profile_execute_role_detailed():
    """Profile execute_role with detailed timing breakdown."""
    # Setup
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
    )
    interaction_memory = InteractionMemory()
    
    input_data = roles.generator.SubquestionGenerationInput(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Paris is the capital and largest city of France."
    )
    
    print("="*80)
    print("DETAILED PROFILING OF execute_role")
    print("="*80)
    
    total_start = time.time()
    
    # Break down execute_role manually to profile each part
    print("\n1. Input preparation:")
    with timer("  Input validation and conversion"):
        is_single = False
        if isinstance(input_data, type(input_data).__bases__[0] if hasattr(type(input_data), '__bases__') else pydantic.BaseModel):
            input_data_list = [input_data]
            is_single = True
        else:
            input_data_list = input_data
            is_single = False
    
    print("\n2. Message formatting:")
    all_messages = []
    role = roles.generator.SubquestionGenerator()
    
    for item in input_data_list:
        with timer(f"  format_messages_async for {type(item).__name__}"):
            messages = await role.format_messages_async(item, interaction_memory=interaction_memory)
            all_messages.append(messages)
    
    print("\n3. LLM generation:")
    kwargs = {
        'n': 1,
        'output_schema': role.output_model,
    }
    with timer("  generator_role_execute"):
        response, raw_response = llm_agent.generator_role_execute(all_messages, **kwargs)
    
    print("\n4. Response processing:")
    with timer("  String conversion of input_data"):
        input_data_str = [str(item) for item in input_data_list]
    
    with timer("  Log data preparation"):
        to_log_data = {
            role.role_name: [tuple(pair) for pair in zip(input_data_str, raw_response) if pair[1]]
        }
    
    print("\n5. Response parsing:")
    parsed_response = []
    for idx, res in enumerate(response):
        with timer(f"  Parsing response batch {idx+1} ({len(res)} items)"):
            r = []
            for item in res:
                parsed_item = role.parse_response(item)
                if parsed_item:
                    r.append(parsed_item)
            parsed_response.append(r)
    
    print("\n6. Final result preparation:")
    with timer("  Result formatting"):
        if is_single:
            final_result = parsed_response[0], to_log_data
        else:
            final_result = parsed_response, to_log_data
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_elapsed:.4f}s")
    print(f"Number of responses: {len(response)}")
    print(f"Number of parsed outputs: {sum(len(r) for r in parsed_response)}")
    print(f"Results: {len(final_result[0]) if isinstance(final_result[0], list) else 1} output(s)")
    
    return final_result


if __name__ == "__main__":
    import pydantic
    results, log = asyncio.run(profile_execute_role_detailed())
    print(f"\n✓ Profiling completed. Results: {len(results)} output(s)")

