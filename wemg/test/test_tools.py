"""
Comprehensive tests for the agents/tools module (WebSearch, Wikidata).

These are real integration tests that call actual web search APIs and Wikidata.
"""
import os
import pytest
import asyncio
from typing import List

from wemg.agents.tools.web_search import (
    WebSearchTool, 
    WebSearchOutput,
    DDGSAPIWrapper,
    SerperAPIWrapper
)
from wemg.agents.tools.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    WikidataEntityRetrievalTool,
    WikidataPropertyRetrievalTool,
    WikidataKHopTriplesRetrievalTool
)
from wemg.agents.base_llm_agent import BaseLLMAgent


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0142:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-32B")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0372:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")


class TestWebSearchTool:
    """Test suite for WebSearchTool functionality."""
    
    @pytest.fixture
    def ddgs_search_tool(self):
        """Create a WebSearchTool with DuckDuckGo backend."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.mark.slow
    def test_ddgs_basic_search(self, ddgs_search_tool):
        """Test basic web search with DuckDuckGo."""
        query = "Python programming language"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        assert output.query == query
        assert len(output.results) > 0
        
        print(f"✓ DDGS Basic Search")
        print(f"  Query: {query}")
        print(f"  Results: {len(output.results)}")
        for i, result in enumerate(output.results[:3]):
            print(f"  {i+1}. {result.title[:60]}...")
    
    @pytest.mark.slow
    def test_ddgs_search_with_full_text(self, ddgs_search_tool):
        """Test web search with full text extraction."""
        query = "machine learning applications"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 3})
        )
        
        assert output.is_success is True
        
        # Check that results have content
        for result in output.results:
            assert result.title is not None
            assert result.link is not None
            assert result.snippet is not None
        
        print(f"✓ DDGS Search with full text")
        print(f"  Retrieved {len(output.results)} results with full text")
    
    @pytest.mark.slow
    def test_ddgs_technical_query(self, ddgs_search_tool):
        """Test search with technical query."""
        query = "transformer architecture attention mechanism"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        assert len(output.results) > 0
        
        print(f"✓ DDGS Technical Query")
        print(f"  Query: {query}")
        print(f"  Found {len(output.results)} results")
    
    @pytest.mark.slow
    def test_ddgs_factual_query(self, ddgs_search_tool):
        """Test search for factual information."""
        query = "population of Tokyo Japan 2024"
        
        output: WebSearchOutput = asyncio.run(
            ddgs_search_tool.ainvoke({"query": query, "top_k": 5})
        )
        
        assert output.is_success is True
        
        print(f"✓ DDGS Factual Query")
        print(f"  Query: {query}")
        for result in output.results[:2]:
            print(f"    - {result.snippet[:100]}...")


class TestWikidataEntityRetrieval:
    """Test suite for Wikidata entity retrieval."""
    
    @pytest.fixture
    def entity_retriever(self):
        """Create a WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.mark.slow
    def test_retrieve_single_entity(self, entity_retriever):
        """Test retrieving a single entity by name."""
        query = ["Albert Einstein"]
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": query, "top_k_results": 3})
        )
        
        assert len(results) == 1
        assert len(results[0]) > 0
        
        # Check first result
        entity: WikidataEntity = results[0][0]
        assert entity.label is not None
        assert entity.qid is not None
        assert "einstein" in entity.label.lower() or "albert" in entity.label.lower()
        
        print(f"✓ Single Entity Retrieval")
        print(f"  Query: Albert Einstein")
        print(f"  Found: {entity.label} ({entity.qid})")
        print(f"  Description: {entity.description[:100] if entity.description else 'N/A'}")
    
    @pytest.mark.slow
    def test_retrieve_multiple_entities(self, entity_retriever):
        """Test retrieving multiple entities at once."""
        queries = ["Paris", "Tokyo", "New York"]
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": queries, "top_k_results": 2})
        )
        
        assert len(results) == 3
        
        print(f"✓ Multiple Entity Retrieval")
        for i, (query, result_list) in enumerate(zip(queries, results)):
            if result_list:
                entity = result_list[0]
                print(f"  {query}: {entity.label} ({entity.qid})")
            else:
                print(f"  {query}: No results")
    
    @pytest.mark.slow
    def test_retrieve_entity_with_disambiguation(self, entity_retriever):
        """Test entity retrieval with ambiguous query."""
        query = ["Apple"]  # Could be fruit or company
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": query, "top_k_results": 5})
        )
        
        assert len(results) == 1
        assert len(results[0]) > 1  # Should return multiple candidates
        
        print(f"✓ Entity with Disambiguation")
        print(f"  Query: Apple")
        print(f"  Found {len(results[0])} candidates:")
        for entity in results[0][:3]:
            print(f"    - {entity.label}: {entity.description[:50] if entity.description else 'N/A'}...")


class TestWikidataPropertyRetrieval:
    """Test suite for Wikidata property retrieval."""
    
    @pytest.fixture
    def property_retriever(self):
        """Create a WikidataPropertyRetrievalTool."""
        return WikidataPropertyRetrievalTool()
    
    @pytest.mark.slow
    def test_retrieve_single_property(self, property_retriever):
        """Test retrieving a single property."""
        query = ["capital"]
        
        results = asyncio.run(
            property_retriever.ainvoke({"query": query, "top_k_results": 3})
        )
        
        assert len(results) == 1
        assert len(results[0]) > 0
        
        prop: WikidataProperty = results[0][0]
        assert prop.label is not None
        assert prop.pid is not None
        
        print(f"✓ Single Property Retrieval")
        print(f"  Query: capital")
        print(f"  Found: {prop.label} ({prop.pid})")
    
    @pytest.mark.slow
    def test_retrieve_multiple_properties(self, property_retriever):
        """Test retrieving multiple properties."""
        queries = ["population", "area", "country"]
        
        results = asyncio.run(
            property_retriever.ainvoke({"query": queries, "top_k_results": 2})
        )
        
        assert len(results) == 3
        
        print(f"✓ Multiple Property Retrieval")
        for query, result_list in zip(queries, results):
            if result_list:
                prop = result_list[0]
                print(f"  {query}: {prop.label} ({prop.pid})")
            else:
                print(f"  {query}: No results")


class TestWikidataKHopTriples:
    """Test suite for Wikidata k-hop triples retrieval."""
    
    @pytest.fixture
    def khop_retriever(self):
        """Create a WikidataKHopTriplesRetrievalTool."""
        return WikidataKHopTriplesRetrievalTool()
    
    @pytest.fixture
    def entity_retriever(self):
        """Create a WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.mark.slow
    def test_get_triples_by_query(self, khop_retriever):
        """Test getting k-hop triples by query string."""
        # Query directly by name
        triples = asyncio.run(
            khop_retriever.ainvoke({
                "query": "France",
                "k": 1,
                "num_entities": 1
            })
        )
        
        assert len(triples) > 0
        assert isinstance(triples[0], WikiTriple)
        
        print(f"✓ K-Hop Triples by Query")
        print(f"  Query: France")
        print(f"  Found {len(triples)} triples:")
        for triple in triples[:5]:
            print(f"    - {triple.subject.label} --[{triple.relation.label}]--> {triple.object.label if hasattr(triple.object, 'label') else triple.object}")
    
    @pytest.mark.slow
    def test_get_triples_by_qid(self, khop_retriever):
        """Test getting k-hop triples by QID directly."""
        # Query by QID (Q142 is France)
        triples = asyncio.run(
            khop_retriever.ainvoke({
                "query": "Q142",
                "is_qids": True,
                "k": 1,
                "num_entities": 1
            })
        )
        
        assert len(triples) > 0
        
        print(f"✓ K-Hop Triples by QID")
        print(f"  QID: Q142 (France)")
        print(f"  Found {len(triples)} triples")
    
    @pytest.mark.slow
    def test_get_triples_batch(self, khop_retriever):
        """Test getting k-hop triples for multiple queries."""
        queries = ["France", "Germany", "Japan"]
        
        results = asyncio.run(
            khop_retriever.ainvoke({
                "query": queries,
                "k": 1,
                "num_entities": 1
            })
        )
        
        assert len(results) == 3
        
        print(f"✓ K-Hop Triples Batch")
        for query, triples in zip(queries, results):
            print(f"  {query}: {len(triples)} triples")
    
    @pytest.mark.slow
    def test_get_triples_with_details(self, khop_retriever):
        """Test getting k-hop triples with full entity details."""
        triples = asyncio.run(
            khop_retriever.ainvoke({
                "query": "Albert Einstein",
                "k": 1,
                "num_entities": 1,
                "update_with_details": True
            })
        )
        
        if triples:
            # Check that entities have full details
            first_triple = triples[0]
            assert first_triple.subject.label is not None
            
            print(f"✓ K-Hop Triples with Details")
            print(f"  Query: Albert Einstein")
            print(f"  Found {len(triples)} triples with full details")
            for triple in triples[:3]:
                obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                print(f"    - {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")


class TestWebSearchIntegration:
    """Integration tests combining web search with LLM."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent."""
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
        """Create a WebSearchTool."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.mark.slow
    def test_search_and_extract(self, llm_agent, web_search_tool):
        """Test searching and extracting information."""
        from wemg.agents import roles
        from wemg.runners.procedures.base_role_execution import execute_role
        
        # Search for information
        query = "When was the first smartphone invented"
        search_output: WebSearchOutput = asyncio.run(
            web_search_tool.ainvoke({"query": query, "top_k": 3})
        )
        
        assert search_output.is_success
        
        # Extract from search results
        if search_output.results:
            raw_data = "\n\n".join([
                f"{r.title}\n{r.snippet}\n{r.full_text}" 
                for r in search_output.results[:2]
            ])
            
            extract_input = roles.extractor.ExtractionInput(
                question=query,
                raw_data=raw_data
            )
            
            results, _ = asyncio.run(execute_role(
                llm_agent=llm_agent,
                role=roles.extractor.Extractor(),
                input_data=extract_input,
                n=1
            ))
            
            output: roles.extractor.ExtractionOutput = results[0]
            
            print(f"✓ Search and Extract Integration")
            print(f"  Query: {query}")
            print(f"  Search results: {len(search_output.results)}")
            print(f"  Extraction decision: {output.decision}")
            print(f"  Extracted info: {len(output.information)} items")


class TestWikidataIntegration:
    """Integration tests for Wikidata tools with knowledge graph building."""
    
    @pytest.fixture
    def entity_retriever(self):
        """Create WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def khop_retriever(self):
        """Create WikidataKHopTriplesRetrievalTool."""
        return WikidataKHopTriplesRetrievalTool()
    
    @pytest.mark.slow
    def test_build_mini_knowledge_graph(self, entity_retriever, khop_retriever):
        """Test building a mini knowledge graph from Wikidata."""
        import networkx as nx
        
        # Start with an entity
        entity_results = asyncio.run(
            entity_retriever.ainvoke({"query": ["Barack Obama"], "top_k_results": 1})
        )
        
        assert len(entity_results[0]) > 0
        entity: WikidataEntity = entity_results[0][0]
        
        # Get k-hop triples for the entity
        triples: List[WikiTriple] = asyncio.run(
            khop_retriever.ainvoke({
                "query": entity.qid,
                "is_qids": True,
                "k": 1,
                "num_entities": 1,
                "update_with_details": True
            })
        )
        
        # Build graph
        G = nx.DiGraph()
        for triple in triples:
            subj_label = triple.subject.label if triple.subject.label else triple.subject.qid
            obj_label = triple.object.label if hasattr(triple.object, 'label') and triple.object.label else str(triple.object)
            rel_label = triple.relation.label if triple.relation.label else triple.relation.pid
            G.add_edge(subj_label, obj_label, relation=rel_label)
        
        print(f"✓ Mini Knowledge Graph Built")
        print(f"  Root entity: {entity.label}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Sample edges:")
        for u, v, data in list(G.edges(data=True))[:5]:
            print(f"    {u[:30]} --[{data.get('relation', 'N/A')[:20]}]--> {v[:30]}")


class TestToolErrorHandling:
    """Test error handling in tools."""
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.fixture
    def entity_retriever(self):
        """Create WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.mark.slow
    def test_search_empty_query(self, web_search_tool):
        """Test search with empty query."""
        try:
            output: WebSearchOutput = asyncio.run(
                web_search_tool.ainvoke({"query": "", "top_k": 5})
            )
            # May succeed with empty results or fail
            print(f"✓ Empty query handled: success={output.is_success}")
        except Exception as e:
            print(f"✓ Empty query raised expected error: {type(e).__name__}")
    
    @pytest.mark.slow
    def test_entity_nonexistent(self, entity_retriever):
        """Test entity retrieval with nonexistent entity."""
        query = ["xyznonexistententity123"]
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": query, "top_k_results": 3})
        )
        
        # Should return empty results, not error
        assert len(results) == 1
        assert len(results[0]) == 0 or results[0] is None or results[0] == []
        
        print(f"✓ Nonexistent entity handled gracefully")


# Run tests with: pytest test_tools.py -v -s --tb=short
# Run slow tests: pytest test_tools.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
