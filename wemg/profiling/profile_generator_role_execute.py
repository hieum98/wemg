"""Detailed profiling of generator_role_execute to identify bottlenecks."""
import asyncio
import time
import os
from contextlib import contextmanager
import json

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles
from wemg.runners.interaction_memory import InteractionMemory


@contextmanager
def timer(name):
    """Context manager to time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")


def profile_generator_role_execute():
    """Profile generator_role_execute with detailed timing breakdown."""
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
    all_messages = [messages]
    
    kwargs = {
        'n': 1,
        'output_schema': role.output_model,
    }
    
    print("="*80)
    print("DETAILED PROFILING OF generator_role_execute")
    print("="*80)
    
    total_start = time.time()
    
    # Step 1: Get output_schema and validate
    print("\n1. Schema validation:")
    with timer("  Get and validate output_schema"):
        output_schema = kwargs.get('output_schema', None)
        if output_schema is not None:
            assert issubclass(output_schema, type(roles.generator.SubquestionGenerationOutput)), \
                "Output schema must be a subclass of pydantic.BaseModel"
    
    # Step 2: Get client
    print("\n2. Client initialization:")
    with timer("  get_client()"):
        client = llm_agent.get_client()
    
    # Step 3: Determine batch vs single
    print("\n3. Message processing setup:")
    with timer("  Check if batch or single"):
        is_batch = isinstance(all_messages[0], list)
        print(f"    Is batch: {is_batch}")
    
    # Step 4: Call generate or batch_generate
    print("\n4. LLM API call:")
    if is_batch:
        with timer("  batch_generate"):
            results = client.batch_generate(all_messages, **kwargs)
    else:
        with timer("  generate (single)"):
            _, results = client.generate(0, all_messages[0], **kwargs)
            results = [results]
    
    print(f"    Number of result batches: {len(results)}")
    print(f"    Items in first batch: {len(results[0]) if results else 0}")
    
    # Step 5: Process results
    print("\n5. Result processing:")
    all_outputs = [None] * len(results)
    all_raw_outputs = [None] * len(results)
    
    for idx, res in enumerate(results):
        print(f"\n  Processing batch {idx+1}:")
        with timer(f"    Total batch {idx+1} processing"):
            outputs = []
            raw_output = None
            
            for item_idx, item in enumerate(res):
                with timer(f"      Item {item_idx+1} processing"):
                    # Get raw output
                    if raw_output is None and item.get('is_valid'):
                        raw_output = item.get('raw_output')
                    
                    # Process output based on schema
                    if output_schema and client.structure_output_supported:
                        with timer(f"        Parse structured output for item {item_idx+1}"):
                            if item.get('is_valid'):
                                try:
                                    parsed_output = output_schema(**item['output'])
                                    outputs.append(parsed_output)
                                except Exception as e:
                                    print(f"        Warning: Failed to parse: {e}")
                            else:
                                # Fallback parsing logic
                                with timer(f"        Fallback parsing for item {item_idx+1}"):
                                    keys = output_schema.model_fields.keys()
                                    value_types = [field.annotation.__name__ for field in output_schema.model_fields.values()]
                                    if isinstance(item['output'], str):
                                        data = item['output']
                                    elif isinstance(item['output'], dict):
                                        data = json.dumps(item['output'])
                                    else:
                                        data = str(item['output'])
                                    
                                    from wemg.utils.parsing import extract_info_from_text
                                    extracted = extract_info_from_text(data, keys, value_types)
                                    try:
                                        parsed_output = output_schema(**extracted)
                                        outputs.append(parsed_output)
                                    except Exception as e2:
                                        print(f"        Error: Failed even after extraction: {e2}")
                    else:
                        outputs.append(item['output'])
            
            all_outputs[idx] = outputs
            all_raw_outputs[idx] = raw_output
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_elapsed:.4f}s")
    print(f"Number of batches: {len(results)}")
    print(f"Total outputs: {sum(len(o) for o in all_outputs)}")
    print(f"Valid outputs: {sum(len([x for x in o if hasattr(x, '__dict__') or isinstance(x, dict)]) for o in all_outputs)}")
    
    return all_outputs, all_raw_outputs


if __name__ == "__main__":
    outputs, raw_outputs = profile_generator_role_execute()
    print(f"\n✓ Profiling completed.")
    print(f"  Outputs: {len(outputs)} batch(es)")
    print(f"  Raw outputs: {len(raw_outputs)} batch(es)")

