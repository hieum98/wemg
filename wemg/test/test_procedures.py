"""
Comprehensive tests for the runners/procedures module.

These are real integration tests for retrieval and role execution procedures.
"""
import os
import pytest
import asyncio
from typing import List, Dict

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool, DDGSAPIWrapper
from wemg.agents.tools.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikidataEntityRetrievalTool,
    WikidataPropertyRetrievalTool
)
from wemg.agents import roles
from wemg.runners.procedures.retrieval import (
    retrieve_from_web,
    retrieve_entities_from_kb,
    explore
)
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.working_memory import WorkingMemory


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0142:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-32B")


class TestRetrievalProcedures:
    """Test suite for retrieval procedures."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_websearch_tool(self, web_search_tool):
        """Test retrieve_from_web with WebSearchTool."""
        query = "What is machine learning?"
        
        results = asyncio.run(
            retrieve_from_web(
                query=query,
                retriever_agent=web_search_tool,
                top_k=5
            )
        )
        
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)
        
        print(f"✓ retrieve_from_web with WebSearchTool")
        print(f"  Query: {query}")
        print(f"  Retrieved {len(results)} documents")
        for i, doc in enumerate(results[:2]):
            print(f"  Doc {i+1}: {doc[:100]}...")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb(self, llm_agent, interaction_memory):
        """Test retrieve_entities_from_kb with query."""
        # Create Wikidata retrievers
        entity_retriever = WikidataEntityRetrievalTool()
        property_retriever = WikidataPropertyRetrievalTool()
        
        query = "What is the capital of France?"
        
        entity_dict, property_dict, triples, log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=query,
                entities=[],
                relations=[],
                top_k_entities=2,
                top_k_properties=2,
                interaction_memory=interaction_memory
            )
        )
        
        print(f"✓ retrieve_entities_from_kb")
        print(f"  Query: {query}")
        print(f"  Entities found: {len(entity_dict)}")
        print(f"  Properties found: {len(property_dict)}")
        print(f"  Triples retrieved: {len(triples)}")
        
        for entity_name, entity in list(entity_dict.items())[:3]:
            print(f"    Entity: {entity_name} -> {entity.label if hasattr(entity, 'label') else entity}")
    
    @pytest.mark.slow
    def test_explore_full_pipeline(self, llm_agent, web_search_tool, interaction_memory):
        """Test the full explore pipeline."""
        question = "When was the Eiffel Tower built?"
        
        documents, triples, entity_dict, property_dict, log = asyncio.run(
            explore(
                llm_agent=llm_agent,
                retriever_agent=web_search_tool,
                question=question,
                entities=[],
                relations=[],
                top_k_websearch=3,
                top_k_entities=2,
                top_k_properties=2,
                n_hops=1,
                use_question_for_graph_retrieval=True,
                interaction_memory=interaction_memory
            )
        )
        
        assert documents is not None
        
        print(f"✓ explore full pipeline")
        print(f"  Question: {question}")
        print(f"  Documents: {len(documents)}")
        print(f"  Triples: {len(triples)}")
        print(f"  Entities: {len(entity_dict)}")
        print(f"  Properties: {len(property_dict)}")
    
    @pytest.mark.slow
    def test_explore_with_existing_entities(self, llm_agent, web_search_tool, interaction_memory):
        """Test explore with pre-existing entities."""
        # First get some entities
        entity_retriever = WikidataEntityRetrievalTool()
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "top_k_results": 1})
        )
        
        existing_entities = []
        if entity_results and entity_results[0]:
            existing_entities = [entity_results[0][0]]
        
        question = "What is the population of Paris?"
        
        documents, triples, entity_dict, property_dict, log = asyncio.run(
            explore(
                llm_agent=llm_agent,
                retriever_agent=web_search_tool,
                question=question,
                entities=existing_entities,
                relations=[],
                top_k_websearch=3,
                top_k_entities=2,
                top_k_properties=2,
                n_hops=1,
                use_question_for_graph_retrieval=False,
                interaction_memory=interaction_memory
            )
        )
        
        print(f"✓ explore with existing entities")
        print(f"  Question: {question}")
        print(f"  Pre-existing entities: {len(existing_entities)}")
        print(f"  Documents: {len(documents)}")
        print(f"  Triples: {len(triples)}")


# WorkingMemory tests have been moved to test_memory.py


class TestConcurrentExecution:
    """Test concurrent execution of procedures."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent with higher concurrency."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=8,  # Higher concurrency for parallel tests
            max_retries=3
        )
    
    @pytest.mark.slow
    def test_concurrent_role_execution(self, llm_agent):
        """Test concurrent execution of multiple roles."""
        import time
        
        inputs = [
            roles.extractor.ExtractionInput(
                question=f"Question {i}",
                raw_data=f"This is sample data {i} containing information about topic {i}."
            )
            for i in range(5)
        ]
        
        start_time = time.time()
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=inputs,
            n=1
        ))
        
        elapsed = time.time() - start_time
        
        assert len(results) == 5
        
        print(f"✓ Concurrent role execution")
        print(f"  Inputs: 5")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Avg per input: {elapsed/5:.2f}s")


# Run tests with: pytest test_procedures.py -v -s --tb=short
# Run slow tests: pytest test_procedures.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
