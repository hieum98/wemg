"""Detailed profiling of the generate method to identify bottlenecks."""
import time
import os
from contextlib import contextmanager
import json

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles
from wemg.runners.interaction_memory import InteractionMemory
import asyncio


@contextmanager
def timer(name):
    """Context manager to time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")


def profile_generate_method():
    """Profile the generate method with detailed timing breakdown."""
    # Setup
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
    )
    
    # Prepare messages
    input_data = roles.generator.SubquestionGenerationInput(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Paris is the capital and largest city of France."
    )
    role = roles.generator.SubquestionGenerator()
    interaction_memory = InteractionMemory()
    
    messages = asyncio.run(role.format_messages_async(input_data, interaction_memory=interaction_memory))
    
    kwargs = {
        'n': 1,
        'output_schema': role.output_model,
        'should_return_reasoning': True,
    }
    
    print("="*80)
    print("DETAILED PROFILING OF generate() METHOD")
    print("="*80)
    
    total_start = time.time()
    client = llm_agent.get_client()
    
    # Step 1: Extract parameters
    print("\n1. Parameter extraction:")
    with timer("  Extract should_return_reasoning, output_schema, use_cache"):
        should_return_reasoning = kwargs.get('should_return_reasoning', True)
        output_schema = kwargs.get('output_schema', None)
        use_cache = kwargs.get('use_cache', client.use_cache)
        cache_ttl = kwargs.get('cache_ttl', None)
    
    # Step 2: Prepare model kwargs
    print("\n2. Model kwargs preparation:")
    with timer("  prepare_model_kwargs"):
        model_kwargs = client.prepare_model_kwargs(**kwargs)
        n = model_kwargs.get('n', 1)
        print(f"    n={n}, temperature={model_kwargs.get('temperature')}, max_tokens={model_kwargs.get('max_tokens')}")
    
    # Step 3: Cache key creation
    print("\n3. Cache operations:")
    cache_key_data = None
    if use_cache and client.cache:
        with timer("  Create cache key"):
            cache_key_data = {
                'model': client.model_name,
                'messages': messages,
                'kwargs': {k: v for k, v in model_kwargs.items() if k not in ['timeout']},
            }
        
        with timer("  Cache lookup"):
            cached_result = client.cache.get(cache_key_data)
            if cached_result is not None:
                print("    ✓ Cache hit!")
                return 0, cached_result
        print("    ✗ Cache miss")
    else:
        print("    Cache disabled")
    
    # Step 4: Generation loop
    print("\n4. Generation loop:")
    valid_choices = []
    attempts = 0
    max_retries = client.max_retries
    
    while len(valid_choices) < n and attempts < max_retries:
        attempts += 1
        print(f"\n  Attempt {attempts}/{max_retries}:")
        
        # Step 4a: Trim messages
        with timer("    Trim messages"):
            import litellm
            trimmed_messages = litellm.utils.trim_messages(messages, max_tokens=client.max_inputs_tokens)
            print(f"      Original tokens: ~{sum(len(m.get('content', '').split()) for m in messages)}")
            print(f"      Trimmed tokens: ~{sum(len(m.get('content', '').split()) for m in trimmed_messages)}")
        
        # Step 4b: API call
        with timer("    completion() API call"):
            try:
                response = client.completion(trimmed_messages, **model_kwargs)
                print(f"      ✓ API call successful")
                print(f"      Response type: {type(response)}")
                if hasattr(response, 'choices'):
                    print(f"      Number of choices: {len(response.choices)}")
            except Exception as e:
                print(f"      ✗ API call failed: {e}")
                response = None
                with timer("      Sleep before retry"):
                    time.sleep(2 * attempts)
        
        # Step 4c: Validate response
        if response is None or not hasattr(response, 'choices'):
            print("    ✗ Invalid response, adjusting parameters and retrying...")
            model_kwargs['temperature'] = min(1.0, model_kwargs.get('temperature', 0.7) + 0.1 * attempts)
            model_kwargs['top_p'] = min(1.0, model_kwargs.get('top_p', 0.8) + 0.1 * attempts)
            continue
        
        # Step 4d: Process choices
        print(f"    Processing {len(response.choices)} choice(s):")
        with timer("    Process all choices"):
            for choice_idx, choice in enumerate(response.choices):
                print(f"\n      Choice {choice_idx+1}:")
                
                with timer("        Check message"):
                    if not choice.message:
                        print("        ✗ No message, skipping")
                        continue
                
                with timer("        Extract content"):
                    content = choice.message.content
                    used_reasoning_content_as_content = False
                    if not content:
                        try:
                            content = choice.message.reasoning_content
                            used_reasoning_content_as_content = True
                            print("        Using reasoning_content as content")
                        except Exception:
                            pass
                    
                    if not content:
                        print("        ✗ No content, skipping")
                        continue
                
                with timer("        Extract reasoning"):
                    reasoning = None
                    if should_return_reasoning and not used_reasoning_content_as_content:
                        try:
                            reasoning = choice.message.reasoning_content 
                        except Exception:
                            reasoning = None
                
                # Step 4e: Parse output
                if output_schema and client.structure_output_supported:
                    with timer("        Parse structured output"):
                        try:
                            parsed_output = output_schema.model_validate_json(content)
                            output = {
                                'output': parsed_output.model_dump() if hasattr(parsed_output, 'model_dump') else parsed_output,
                                'raw_output': content,
                                'reasoning': reasoning,
                                'is_valid': True
                            }
                            print("        ✓ Successfully parsed structured output")
                            valid_choices.append(output)
                        except Exception as e:
                            print(f"        ✗ Failed to parse: {e}")
                            if attempts == max_retries - 1:
                                output = {
                                    'output': content,
                                    'raw_output': content,
                                    'reasoning': reasoning,
                                    'is_valid': False
                                }
                                valid_choices.append(output)
                else:
                    with timer("        Create simple output"):
                        output = {
                            'output': content,
                            'raw_output': content,
                            'reasoning': reasoning,
                            'is_valid': True
                        }
                        valid_choices.append(output)
        
        # Break if we have enough valid choices
        if len(valid_choices) >= n:
            print(f"    ✓ Got {len(valid_choices)} valid choice(s), breaking loop")
            break
    
    # Step 5: Final processing
    print("\n5. Final processing:")
    if len(valid_choices) == 0:
        print("  ✗ No valid completions")
        return 0, []
    
    with timer("  Trim to n choices"):
        valid_choices = valid_choices[:n]
    
    # Step 6: Cache result
    if use_cache and client.cache and all([vc['is_valid'] for vc in valid_choices]):
        with timer("  Cache result"):
            client.cache.set(cache_key_data, valid_choices, ttl=cache_ttl)
            print("    ✓ Result cached")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_elapsed:.4f}s")
    print(f"Number of attempts: {attempts}")
    print(f"Valid choices: {len(valid_choices)}")
    print(f"Cache used: {use_cache and client.cache is not None}")
    
    return 0, valid_choices


if __name__ == "__main__":
    index, results = profile_generate_method()
    print(f"\n✓ Profiling completed.")
    print(f"  Index: {index}")
    print(f"  Results: {len(results)} choice(s)")

