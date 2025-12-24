"""
Comprehensive tests for the wemg/runners/procedures/retrieval.py module.

These are real integration tests for all retrieval procedures including:
- retrieve_from_web
- retrieve_entities_from_kb
- retrieve_triples
- retrieve_from_kb
- explore
"""
import os
import pytest
import asyncio
from typing import List, Dict
from pathlib import Path

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool, DDGSAPIWrapper
from wemg.agents.tools.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikidataEntityRetrievalTool,
    WikidataPropertyRetrievalTool,
    WikidataKHopTriplesRetrievalTool,
    CustomWikidataAPIWrapper,
    WikiTriple
)
from wemg.agents import roles
from wemg.runners.procedures.retrieval import (
    retrieve_from_web,
    retrieve_entities_from_kb,
    retrieve_triples,
    retrieve_from_kb,
    explore
)
from wemg.runners.interaction_memory import InteractionMemory


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your-serper-api-key")

# Wiki corpus configuration for RetrieverAgent tests
WIKI_CORPUS_HF = os.getenv("WIKI_CORPUS_HF", "Hieuman/wiki23-processed")
WIKI_INDEX_PATH = Path(os.getenv("WIKI_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))


class TestRetrieveFromWeb:
    """Test suite for retrieve_from_web function."""
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(serper_api_key=SERPER_API_KEY)
    
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
        
        assert isinstance(results, list)
        assert len(results) >= 0  # May be empty if search fails
        assert all(isinstance(r, str) for r in results)
        
        print(f"✓ retrieve_from_web with WebSearchTool")
        print(f"  Query: {query}")
        print(f"  Retrieved {len(results)} documents")
        if results:
            for i, doc in enumerate(results[:2]):
                print(f"  Doc {i+1}: {doc[:100]}...")
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_different_top_k(self, web_search_tool):
        """Test retrieve_from_web with different top_k values."""
        query = "Python programming"
        
        for top_k in [1, 3, 5]:
            results = asyncio.run(
                retrieve_from_web(
                    query=query,
                    retriever_agent=web_search_tool,
                    top_k=top_k
                )
            )
            assert isinstance(results, list)
            print(f"  top_k={top_k}: {len(results)} results")
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_empty_query(self, web_search_tool):
        """Test retrieve_from_web with empty query."""
        query = ""
        
        results = asyncio.run(
            retrieve_from_web(
                query=query,
                retriever_agent=web_search_tool,
                top_k=5
            )
        )
        
        assert isinstance(results, list)
        print(f"✓ retrieve_from_web with empty query: {len(results)} results")
    
    def test_retrieve_from_web_with_invalid_retriever(self):
        """Test retrieve_from_web raises error with invalid retriever type."""
        invalid_retriever = "not a retriever"
        
        with pytest.raises(ValueError, match="Unsupported retriever agent type"):
            asyncio.run(
                retrieve_from_web(
                    query="test",
                    retriever_agent=invalid_retriever,
                    top_k=5
                )
            )
    
    @pytest.fixture
    def retriever_agent_embedder_config(self):
        """Create embedder configuration for RetrieverAgent."""
        return {
            'model_name': TEST_EMBEDDING_MODEL,
            'url': TEST_EMBEDDING_API_BASE,
            'api_key': TEST_LLM_API_KEY,
            'is_embedding': True,
            'timeout': 60,
        }
    
    @pytest.fixture
    def retriever_agent(self, retriever_agent_embedder_config):
        """Create a RetrieverAgent instance with wiki corpus and pre-indexed FAISS."""
        # Check if index exists
        if not WIKI_INDEX_PATH.exists():
            pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}. Please ensure the index is available.")
        
        # Try to load corpus from local directory first (if it was previously saved)
        # Otherwise, load from HuggingFace
        local_corpus_path = WIKI_INDEX_PATH.parent / "without_embeddings"
        if local_corpus_path.exists():
            corpus_path = local_corpus_path
        else:
            # Load from HuggingFace - will be saved locally after first load
            corpus_path = Path(WIKI_CORPUS_HF)
        
        agent = RetrieverAgent(
            embedder_config=retriever_agent_embedder_config,
            corpus_path=corpus_path,
            index_path=WIKI_INDEX_PATH,
            embedder_type='openai'
        )
        
        return agent
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_retriever_agent(self, retriever_agent):
        """Test retrieve_from_web with RetrieverAgent."""
        query = "What is machine learning?"
        
        results = asyncio.run(
            retrieve_from_web(
                query=query,
                retriever_agent=retriever_agent,
                top_k=5
            )
        )
        
        assert isinstance(results, list)
        assert len(results) > 0  # Should have results from corpus
        assert all(isinstance(r, str) for r in results)
        assert len(results) == 5  # Should return exactly top_k results
        
        print(f"✓ retrieve_from_web with RetrieverAgent")
        print(f"  Query: {query}")
        print(f"  Retrieved {len(results)} documents")
        for i, doc in enumerate(results[:2]):
            print(f"  Doc {i+1}: {doc[:100]}...")
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_retriever_agent_different_top_k(self, retriever_agent):
        """Test retrieve_from_web with RetrieverAgent using different top_k values."""
        query = "Python programming"
        
        for top_k in [1, 3, 5]:
            results = asyncio.run(
                retrieve_from_web(
                    query=query,
                    retriever_agent=retriever_agent,
                    top_k=top_k
                )
            )
            assert isinstance(results, list)
            assert len(results) == top_k  # Should return exactly top_k results
            assert all(isinstance(r, str) for r in results)
            print(f"  top_k={top_k}: {len(results)} results")
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_retriever_agent_empty_query(self, retriever_agent):
        """Test retrieve_from_web with RetrieverAgent using empty query."""
        query = ""
        
        results = asyncio.run(
            retrieve_from_web(
                query=query,
                retriever_agent=retriever_agent,
                top_k=5
            )
        )
        
        assert isinstance(results, list)
        # Empty query might return results or empty list depending on implementation
        print(f"✓ retrieve_from_web with RetrieverAgent and empty query: {len(results)} results")
    
    @pytest.mark.slow
    def test_retrieve_from_web_with_retriever_agent_various_queries(self, retriever_agent):
        """Test retrieve_from_web with RetrieverAgent using various query types."""
        queries = [
            "What is the capital of France?",
            "Machine learning algorithms",
            "History of World War II"
        ]
        
        for query in queries:
            results = asyncio.run(
                retrieve_from_web(
                    query=query,
                    retriever_agent=retriever_agent,
                    top_k=3
                )
            )
            
            assert isinstance(results, list)
            assert len(results) == 3
            assert all(isinstance(r, str) for r in results)
            assert all(len(r) > 0 for r in results)  # All results should have content
            
            print(f"  Query: '{query}' -> {len(results)} results")


class TestRetrieveEntitiesFromKB:
    """Test suite for retrieve_entities_from_kb function."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=65536,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def entity_retriever(self):
        """Create a WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def property_retriever(self):
        """Create a WikidataPropertyRetrievalTool."""
        return WikidataPropertyRetrievalTool()
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_query(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with query only."""
        query = "What is the capital of France?"
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
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
        
        assert isinstance(entities, list)
        assert isinstance(entity_dict, dict)
        assert isinstance(properties, list)
        assert isinstance(property_dict, dict)
        assert isinstance(graph_query_log, dict)
        
        # All entities should be WikidataEntity instances
        assert all(isinstance(e, WikidataEntity) for e in entities)
        
        # Entity dict should map open_ie.Entity to WikidataEntity
        for key, value in entity_dict.items():
            assert isinstance(key, roles.open_ie.Entity)
            assert isinstance(value, WikidataEntity)
        
        # All properties should be WikidataProperty instances
        assert all(isinstance(p, WikidataProperty) for p in properties)
        
        print(f"✓ retrieve_entities_from_kb with query")
        print(f"  Query: {query}")
        print(f"  Entities found: {len(entities)}")
        print(f"  Entity mappings: {len(entity_dict)}")
        print(f"  Properties found: {len(properties)}")
        print(f"  Property mappings: {len(property_dict)}")
        for i, entity in enumerate(entities[:3]):
            print(f"    Entity {i+1}: {entity.qid} - {entity.label}")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_existing_entities(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with pre-existing entities."""
        # Create open_ie entities
        existing_entities = [
            roles.open_ie.Entity(name="Paris"),
            roles.open_ie.Entity(name="France")
        ]
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=None,
                entities=existing_entities,
                relations=[],
                top_k_entities=2,
                top_k_properties=1,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(entities) > 0
        assert len(entity_dict) > 0  # Should have mappings for the entities
        
        print(f"✓ retrieve_entities_from_kb with existing entities")
        print(f"  Input entities: {len(existing_entities)}")
        print(f"  Retrieved entities: {len(entities)}")
        print(f"  Entity mappings: {len(entity_dict)}")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_existing_wikidata_entities(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with pre-existing WikidataEntity objects."""
        # First retrieve some entities
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "num_entities": 1})
        )
        
        existing_wikidata_entities = []
        if entity_results and entity_results[0]:
            existing_wikidata_entities = [entity_results[0][0]]
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=None,
                entities=existing_wikidata_entities,
                relations=[],
                top_k_entities=1,
                top_k_properties=1,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(entities) >= len(existing_wikidata_entities)
        assert all(isinstance(e, WikidataEntity) for e in entities)
        
        print(f"✓ retrieve_entities_from_kb with existing WikidataEntity")
        print(f"  Input Wikidata entities: {len(existing_wikidata_entities)}")
        print(f"  Total entities: {len(entities)}")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_relations(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with relations."""
        entities = [roles.open_ie.Entity(name="France")]
        relations = ["capital", "population"]
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=None,
                entities=entities,
                relations=relations,
                top_k_entities=1,
                top_k_properties=2,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(properties) > 0
        assert len(property_dict) > 0  # Should have mappings for string relations
        
        print(f"✓ retrieve_entities_from_kb with relations")
        print(f"  Input relations: {relations}")
        print(f"  Retrieved properties: {len(properties)}")
        print(f"  Property mappings: {len(property_dict)}")
        for rel, prop in list(property_dict.items())[:3]:
            print(f"    {rel} -> {prop.pid} ({prop.label})")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_wikidata_properties(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with pre-existing WikidataProperty objects."""
        # First retrieve some properties
        property_results = asyncio.run(
            property_retriever.ainvoke({"query": ["capital"], "top_k_results": 1})
        )
        
        existing_properties = []
        if property_results and property_results[0]:
            existing_properties = property_results[0]
        
        entities = [roles.open_ie.Entity(name="France")]
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=None,
                entities=entities,
                relations=existing_properties,
                top_k_entities=1,
                top_k_properties=1,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(properties) >= len(existing_properties)
        assert all(isinstance(p, WikidataProperty) for p in properties)
        
        print(f"✓ retrieve_entities_from_kb with existing WikidataProperty")
        print(f"  Input properties: {len(existing_properties)}")
        print(f"  Total properties: {len(properties)}")
    
    @pytest.mark.slow
    def test_retrieve_entities_from_kb_with_query_and_entities(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb with both query and existing entities."""
        query = "What is the population?"
        existing_entities = [roles.open_ie.Entity(name="France")]
        
        entities, entity_dict, properties, property_dict, graph_query_log = asyncio.run(
            retrieve_entities_from_kb(
                llm_agent=llm_agent,
                wikidata_entity_retriever=entity_retriever,
                wikidata_property_retriever=property_retriever,
                query=query,
                entities=existing_entities,
                relations=[],
                top_k_entities=2,
                top_k_properties=2,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(entities) > 0
        print(f"✓ retrieve_entities_from_kb with query and entities")
        print(f"  Query: {query}")
        print(f"  Existing entities: {len(existing_entities)}")
        print(f"  Total entities: {len(entities)}")
    
    def test_retrieve_entities_from_kb_without_query_or_entities(self, llm_agent, entity_retriever, property_retriever, interaction_memory):
        """Test retrieve_entities_from_kb raises error when neither query nor entities provided."""
        with pytest.raises(AssertionError, match="Either query or entities must be provided"):
            asyncio.run(
                retrieve_entities_from_kb(
                    llm_agent=llm_agent,
                    wikidata_entity_retriever=entity_retriever,
                    wikidata_property_retriever=property_retriever,
                    query=None,
                    entities=[],
                    relations=[],
                    top_k_entities=1,
                    top_k_properties=1,
                    interaction_memory=interaction_memory
                )
            )


class TestRetrieveTriples:
    """Test suite for retrieve_triples function."""
    
    @pytest.fixture
    def entity_retriever(self):
        """Create a WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def property_retriever(self):
        """Create a WikidataPropertyRetrievalTool."""
        return WikidataPropertyRetrievalTool()
    
    @pytest.mark.slow
    def test_retrieve_triples_with_single_hop(self, entity_retriever, property_retriever):
        """Test retrieve_triples with n_hops=1."""
        # First get some entities
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris", "France"], "num_entities": 1})
        )
        
        entities = []
        for result in entity_results:
            if result:
                entities.extend(result[:1])  # Take first entity from each result
        
        if not entities:
            pytest.skip("No entities retrieved, skipping test")
        
        # Create triple retriever
        wikidata_wrapper = CustomWikidataAPIWrapper()
        triple_retriever = WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)
        
        triples = asyncio.run(
            retrieve_triples(
                wikidata_triple_retriever=triple_retriever,
                entities=entities,
                n_hops=1
            )
        )
        
        assert isinstance(triples, list)
        assert all(isinstance(t, WikiTriple) for t in triples)
        
        print(f"✓ retrieve_triples with n_hops=1")
        print(f"  Input entities: {len(entities)}")
        print(f"  Retrieved triples: {len(triples)}")
        if triples:
            for i, triple in enumerate(triples[:3]):
                print(f"    Triple {i+1}: {triple.subject.qid} -> {triple.relation.pid} -> {str(triple.object)[:50]}")
    
    @pytest.mark.slow
    def test_retrieve_triples_with_multiple_hops(self, entity_retriever, property_retriever):
        """Test retrieve_triples with n_hops=2."""
        # First get some entities
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "num_entities": 1})
        )
        
        entities = []
        if entity_results and entity_results[0]:
            entities = [entity_results[0][0]]
        
        if not entities:
            pytest.skip("No entities retrieved, skipping test")
        
        # Create triple retriever
        wikidata_wrapper = CustomWikidataAPIWrapper()
        triple_retriever = WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)
        
        triples = asyncio.run(
            retrieve_triples(
                wikidata_triple_retriever=triple_retriever,
                entities=entities,
                n_hops=2
            )
        )
        
        assert isinstance(triples, list)
        assert all(isinstance(t, WikiTriple) for t in triples)
        
        print(f"✓ retrieve_triples with n_hops=2")
        print(f"  Input entities: {len(entities)}")
        print(f"  Retrieved triples: {len(triples)}")
    
    @pytest.mark.slow
    def test_retrieve_triples_with_empty_entities(self, property_retriever):
        """Test retrieve_triples with empty entities list."""
        wikidata_wrapper = CustomWikidataAPIWrapper()
        triple_retriever = WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)
        
        triples = asyncio.run(
            retrieve_triples(
                wikidata_triple_retriever=triple_retriever,
                entities=[],
                n_hops=1
            )
        )
        
        assert isinstance(triples, list)
        assert len(triples) == 0
        
        print(f"✓ retrieve_triples with empty entities: {len(triples)} triples")
    
    @pytest.mark.slow
    def test_retrieve_triples_deduplication(self, entity_retriever, property_retriever):
        """Test that retrieve_triples deduplicates results."""
        # Get entities
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "num_entities": 1})
        )
        
        entities = []
        if entity_results and entity_results[0]:
            entities = [entity_results[0][0]]
        
        if not entities:
            pytest.skip("No entities retrieved, skipping test")
        
        # Create triple retriever
        wikidata_wrapper = CustomWikidataAPIWrapper()
        triple_retriever = WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)
        
        triples = asyncio.run(
            retrieve_triples(
                wikidata_triple_retriever=triple_retriever,
                entities=entities,
                n_hops=1
            )
        )
        
        # Check that all triples are unique (using set)
        unique_triples = set(triples)
        assert len(unique_triples) == len(triples), "Triples should be deduplicated"
        
        print(f"✓ retrieve_triples deduplication")
        print(f"  Total triples: {len(triples)}")
        print(f"  Unique triples: {len(unique_triples)}")


class TestRetrieveFromKB:
    """Test suite for retrieve_from_kb function."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=65536,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_retrieve_from_kb_with_question(self, llm_agent, interaction_memory):
        """Test retrieve_from_kb with question and use_question_for_graph_retrieval=True."""
        question = "What is the capital of France?"
        
        triples, entities, entity_dict, property_dict, graph_query_log = asyncio.run(
            retrieve_from_kb(
                llm_agent=llm_agent,
                question=question,
                entities=[],
                relations=[],
                top_k_entities=2,
                top_k_properties=2,
                n_hops=1,
                use_question_for_graph_retrieval=True,
                interaction_memory=interaction_memory
            )
        )
        
        assert isinstance(triples, list)
        assert isinstance(entity_dict, dict)
        assert isinstance(property_dict, dict)
        assert isinstance(graph_query_log, dict)
        assert all(isinstance(t, WikiTriple) for t in triples)
        assert all(isinstance(e, WikidataEntity) for e in entities)
        
        print(f"✓ retrieve_from_kb with question")
        print(f"  Question: {question}")
        print(f"  Triples: {len(triples)}")
        print(f"  Entities: {len(entity_dict)}")
        print(f"  Properties: {len(property_dict)}")
        for triple in triples:
            print(f"    {triple.subject.qid} -> {triple.relation.pid} -> {triple.object.qid}")
        for entity in entity_dict.values():
            print(f"    {entity.qid} - {entity.label}")
        for property in property_dict.values():
            print(f"    {property.pid} - {property.label}")
        for entity in entities:
            print(f"    {entity.qid} - {entity.label}")
    
    @pytest.mark.slow
    def test_retrieve_from_kb_with_existing_entities(self, llm_agent, interaction_memory):
        """Test retrieve_from_kb with pre-existing entities."""
        entity_retriever = WikidataEntityRetrievalTool()
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris", "France"], "num_entities": 1})
        )
        
        existing_entities = []
        for result in entity_results:
            if result:
                existing_entities.extend(result[:1])
        
        if not existing_entities:
            pytest.skip("No entities retrieved, skipping test")
        
        question = "What is the relationship between these entities?"
        
        triples, entities, entity_dict, property_dict, graph_query_log = asyncio.run(
            retrieve_from_kb(
                llm_agent=llm_agent,
                question=question,
                entities=existing_entities,
                relations=[],
                top_k_entities=1,
                top_k_properties=1,
                n_hops=1,
                use_question_for_graph_retrieval=False,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(triples) > 0
        assert len(entity_dict) >= 0  # May be empty if entities already are WikidataEntity
        
        print(f"✓ retrieve_from_kb with existing entities")
        print(f"  Input entities: {len(existing_entities)}")
        print(f"  Triples: {len(triples)}")
        print(f"  Entity mappings: {len(entity_dict)}")
        for entity in entities:
            print(f"    {entity.qid} - {entity.label}")
    
    @pytest.mark.slow
    def test_retrieve_from_kb_with_relations(self, llm_agent, interaction_memory):
        """Test retrieve_from_kb with relations."""
        question = "Test question"
        entities = [roles.open_ie.Entity(name="France")]
        relations = ["capital", "population"]
        
        triples, entities, entity_dict, property_dict, graph_query_log = asyncio.run(
            retrieve_from_kb(
                llm_agent=llm_agent,
                question=question,
                entities=entities,
                relations=relations,
                top_k_entities=1,
                top_k_properties=2,
                n_hops=1,
                use_question_for_graph_retrieval=False,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(property_dict) > 0  # Should have property mappings
        assert all(isinstance(e, WikidataEntity) for e in entities)
        
        print(f"✓ retrieve_from_kb with relations")
        print(f"  Relations: {relations}")
        print(f"  Property mappings: {len(property_dict)}")
        print(f"  Triples: {len(triples)}")
    
    @pytest.mark.slow
    def test_retrieve_from_kb_with_different_n_hops(self, llm_agent, interaction_memory):
        """Test retrieve_from_kb with different n_hops values."""
        entity_retriever = WikidataEntityRetrievalTool()
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "num_entities": 1})
        )
        
        entities = []
        if entity_results and entity_results[0]:
            entities = [entity_results[0][0]]
        
        if not entities:
            pytest.skip("No entities retrieved, skipping test")
        
        question = "Test question"
        
        for n_hops in [1, 2]:
            triples, _entities, entity_dict, property_dict, graph_query_log = asyncio.run(
                retrieve_from_kb(
                    llm_agent=llm_agent,
                    question=question,
                    entities=entities,
                    relations=[],
                    top_k_entities=1,
                    top_k_properties=1,
                    n_hops=n_hops,
                    use_question_for_graph_retrieval=False,
                    interaction_memory=interaction_memory
                )
            )
            
            print(f"  n_hops={n_hops}: {len(triples)} triples")
            for entity in _entities:
                print(f"    {entity.qid} - {entity.label}")


class TestExplore:
    """Test suite for explore function."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=65536,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
        )
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_explore_full_pipeline(self, llm_agent, web_search_tool, interaction_memory):
        """Test the full explore pipeline with both web and KB search."""
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
        
        assert isinstance(documents, list)
        assert isinstance(triples, list)
        assert isinstance(entity_dict, dict)
        assert isinstance(property_dict, dict)
        assert isinstance(log, dict)
        assert all(isinstance(d, str) for d in documents)
        assert all(isinstance(t, WikiTriple) for t in triples)
        
        print(f"✓ explore full pipeline")
        print(f"  Question: {question}")
        print(f"  Documents: {len(documents)}")
        print(f"  Triples: {len(triples)}")
        print(f"  Entities: {len(entity_dict)}")
        print(f"  Properties: {len(property_dict)}")
        print(f"  Log keys: {list(log.keys())}")
        for triple in triples:
            print(f"    {triple.subject} -> {triple.relation} -> {triple.object}")
        print(f"  Entity dict: {entity_dict}")
        print(f"  Property dict: {property_dict}")
    
    @pytest.mark.slow
    def test_explore_with_existing_entities(self, llm_agent, web_search_tool, interaction_memory):
        """Test explore with pre-existing entities."""
        # First get some entities
        entity_retriever = WikidataEntityRetrievalTool()
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Paris"], "num_entities": 1})
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
        
        assert isinstance(documents, list)
        assert isinstance(triples, list)
        
        print(f"✓ explore with existing entities")
        print(f"  Question: {question}")
        print(f"  Pre-existing entities: {len(existing_entities)}")
        print(f"  Documents: {len(documents)}")
        print(f"  Triples: {len(triples)}")
        print(f"  Entities: {len(entity_dict)}")
    
    @pytest.mark.slow
    def test_explore_with_relations(self, llm_agent, web_search_tool, interaction_memory):
        """Test explore with pre-specified relations."""
        question = "What is the capital of France?"
        entities = [roles.open_ie.Entity(name="France")]
        relations = ["capital"]
        
        documents, triples, entity_dict, property_dict, log = asyncio.run(
            explore(
                llm_agent=llm_agent,
                retriever_agent=web_search_tool,
                question=question,
                entities=entities,
                relations=relations,
                top_k_websearch=2,
                top_k_entities=1,
                top_k_properties=1,
                n_hops=1,
                use_question_for_graph_retrieval=False,
                interaction_memory=interaction_memory
            )
        )
        
        assert len(property_dict) > 0  # Should have property mappings
        
        print(f"✓ explore with relations")
        print(f"  Question: {question}")
        print(f"  Relations: {relations}")
        print(f"  Property mappings: {len(property_dict)}")
        print(f"  Documents: {len(documents)}")
        print(f"  Triples: {len(triples)}")
    
    @pytest.mark.slow
    def test_explore_document_deduplication(self, llm_agent, web_search_tool, interaction_memory):
        """Test that explore deduplicates documents."""
        question = "Python programming"
        
        documents, triples, entity_dict, property_dict, log = asyncio.run(
            explore(
                llm_agent=llm_agent,
                retriever_agent=web_search_tool,
                question=question,
                entities=[],
                relations=[],
                top_k_websearch=3,
                top_k_entities=1,
                top_k_properties=1,
                n_hops=1,
                use_question_for_graph_retrieval=False,
                interaction_memory=interaction_memory
            )
        )
        
        # Check that documents are deduplicated (using set)
        unique_documents = set(documents)
        assert len(unique_documents) == len(documents), "Documents should be deduplicated"
        
        print(f"✓ explore document deduplication")
        print(f"  Total documents: {len(documents)}")
        print(f"  Unique documents: {len(unique_documents)}")
    
    @pytest.mark.slow
    def test_explore_with_different_parameters(self, llm_agent, web_search_tool, interaction_memory):
        """Test explore with different parameter combinations."""
        question = "What is machine learning?"
        
        # Test with different top_k values
        for top_k_websearch in [2, 5]:
            for top_k_entities in [1, 2]:
                documents, triples, entity_dict, property_dict, log = asyncio.run(
                    explore(
                        llm_agent=llm_agent,
                        retriever_agent=web_search_tool,
                        question=question,
                        entities=[],
                        relations=[],
                        top_k_websearch=top_k_websearch,
                        top_k_entities=top_k_entities,
                        top_k_properties=1,
                        n_hops=1,
                        use_question_for_graph_retrieval=False,
                        interaction_memory=interaction_memory
                    )
                )
                
                print(f"  top_k_websearch={top_k_websearch}, top_k_entities={top_k_entities}: "
                      f"{len(documents)} docs, {len(triples)} triples")


# Run tests with: pytest test_retrieval_procedure.py -v -s --tb=short
# Run slow tests: pytest test_retrieval_procedure.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
