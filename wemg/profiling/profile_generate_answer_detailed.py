"""Detailed profiling of generate_answer with timing breakdown for each step."""
import asyncio
import time
import os
from contextlib import contextmanager
from pathlib import Path

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool
from wemg.runners.procedures.node_generator import NodeGenerator
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.procedures.retrieval import explore
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.agents import roles
from wemg.utils.preprocessing import format_context


@contextmanager
def timer(name):
    """Context manager to time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")


async def profile_generate_answer_detailed():
    """Profile generate_answer with detailed timing breakdown."""
    # Setup
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
        max_tokens=65536,
        concurrency=32,
        max_retries=3
    )
    
    serper_api_key = os.getenv("SERPER_API_KEY", "your-serper-api-key")
    retriever_agent = WebSearchTool(serper_api_key=serper_api_key)
    
    working_memory = WorkingMemory()
    interaction_memory = InteractionMemory()
    
    question = "Which magazine was started first Arthur's Magazine or First for Women?"
    
    # Generation kwargs
    generation_kwargs = {
        'top_k_websearch': 5,
        'top_k_entities': 1,
        'top_k_properties': 1,
        'n_hops': 1,
        'use_question_for_graph_retrieval': True,
        'n': 1
    }
    
    print("="*80)
    print("DETAILED PROFILING OF generate_answer() METHOD")
    print("="*80)
    print(f"Question: {question}\n")
    
    total_start = time.time()
    
    # Step 1: Prepare in-memory entities and relations
    print("1. Prepare in-memory entities and relations:")
    with timer("  Extract entities and relations from working memory"):
        from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty
        in_memory_entities = list[WikidataEntity](set(working_memory.entity_dict.values()))
        in_memory_relations = list[WikidataProperty](set(working_memory.property_dict.values()))
        print(f"    Found {len(in_memory_entities)} entities, {len(in_memory_relations)} relations")
    
    # Step 2: Explore external resources
    print("\n2. Explore external resources (retrieval):")
    with timer("  explore() - web search and KB retrieval"):
        retrieved_documents, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=question,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=generation_kwargs.get('top_k_websearch', 5),
            top_k_entities=generation_kwargs.get('top_k_entities', 1),
            top_k_properties=generation_kwargs.get('top_k_properties', 1),
            n_hops=generation_kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=generation_kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=interaction_memory
        )
        print(f"    Retrieved {len(retrieved_documents)} documents, {len(retrieved_triples)} triples")
        print(f"    Found {len(entity_dict)} entities, {len(property_dict)} properties")
    
    # Step 3: Extract information from web search results
    print("\n3. Extract information from web search results:")
    with timer("  Prepare extractor inputs"):
        extractor_inputs = [
            roles.extractor.ExtractionInput(question=question, raw_data=data) 
            for data in retrieved_documents
        ]
        print(f"    Created {len(extractor_inputs)} extractor inputs")
    
    with timer("  execute_role() - Extractor"):
        extracted_results, extractor_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_inputs,
            interaction_memory=interaction_memory,
            n=1
        )
    
    with timer("  Process extraction results"):
        all_extractions: list = sum(extracted_results, [])
        info_from_websearch = []
        for item in all_extractions:
            if item.relevant_information:
                info_from_websearch.extend(item.relevant_information)
        print(f"    Extracted {len(info_from_websearch)} relevant information items")
    
    # Step 4: Build context
    print("\n4. Build context:")
    with timer("  Format KB triples"):
        info_from_kb = [str(t) for t in retrieved_triples]
    
    with timer("  Combine retrieval info"):
        all_retrieved_info = info_from_websearch + info_from_kb
    
    with timer("  Format working memory"):
        memory = working_memory.format_textual_memory()
    
    with timer("  format_context()"):
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)
        print(f"    Context length: {len(context)} characters")
    
    # Step 5: Generate answer
    print("\n5. Generate answer:")
    with timer("  Prepare QA input"):
        qa_input = roles.generator.AnswerGenerationInput(question=question, context=context)
    
    with timer("  execute_role() - AnswerGenerator"):
        answers, qa_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=qa_input,
            interaction_memory=interaction_memory,
            n=generation_kwargs.get('n', 1)
        )
        print(f"    Generated {len(answers)} answer(s)")
        if answers:
            print(f"    Answer preview: {answers[0].answer[:200]}...")
    
    # Step 6: Merge logs
    print("\n6. Merge logs:")
    with timer("  merge_logs()"):
        from wemg.utils.common import merge_logs
        log_data = merge_logs(exploration_log, extractor_log, qa_log)
    
    total_end = time.time()
    total_elapsed = total_end - total_start
    
    print("\n" + "="*80)
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print("="*80)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Total execution time: {total_elapsed:.4f}s")
    print(f"  Retrieved documents: {len(retrieved_documents)}")
    print(f"  Retrieved triples: {len(retrieved_triples)}")
    print(f"  Extracted information items: {len(info_from_websearch)}")
    print(f"  Generated answers: {len(answers)}")
    if answers:
        print(f"  Answer confidence: {answers[0].confidence_level}")


def main():
    """Main entry point."""
    asyncio.run(profile_generate_answer_detailed())


if __name__ == "__main__":
    main()

