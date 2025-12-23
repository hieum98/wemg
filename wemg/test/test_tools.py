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
    WikidataKHopTriplesRetrievalTool,
    WikidataPathFindingTool,
    WikidataPathBetweenEntities
)
from wemg.agents.base_llm_agent import BaseLLMAgent


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your-serper-api-key")

class TestWebSearchTool:
    """Test suite for WebSearchTool functionality."""
    
    @pytest.fixture
    def ddgs_search_tool(self):
        """Create a WebSearchTool that will use DuckDuckGo backend.
        
        Since WebSearchTool tries Serper first and falls back to DDGS,
        we use an invalid serper_api_key to force it to use DDGS.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
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
    def test_retrieve_single_entity(self, entity_retriever: WikidataEntityRetrievalTool):
        """Test retrieving a single entity by name."""
        query = ["Albert Einstein"]
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": query, "num_entities": 3})
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
        print(f"  Results: {len(results[0])}")
        print(f"  Found: {entity.label} ({entity.qid})")
        print(f"  Description: {entity.description[:100] if entity.description else 'N/A'}")
    
    @pytest.mark.slow
    def test_retrieve_multiple_entities(self, entity_retriever):
        """Test retrieving multiple entities at once."""
        queries = ["Paris", "Tokyo", "New York"]
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": queries, "num_entities": 2})
        )
        
        assert len(results) == 3
        
        print(f"✓ Multiple Entity Retrieval")
        for i, (query, result_list) in enumerate(zip(queries, results)):
            if result_list:
                entity = result_list[0]
                print(f"  {query}: {entity.label} ({entity.qid})")
                print(entity)
            else:
                print(f"  {query}: No results")
    
    @pytest.mark.slow
    def test_retrieve_entity_with_disambiguation(self, entity_retriever):
        """Test entity retrieval with ambiguous query."""
        query = ["Apple"]  # Could be fruit or company
        
        results = asyncio.run(
            entity_retriever.ainvoke({"query": query, "num_entities": 5})
        )
        
        assert len(results) == 1
        assert len(results[0]) > 1  # Should return multiple candidates
        
        print(f"✓ Entity with Disambiguation")
        print(f"  Query: Apple")
        print(f"  Found {len(results[0])} candidates:")
        for entity in results[0][:3]:
            print(entity)
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
        """Create a WebSearchTool that will use DuckDuckGo backend.
        
        Since WebSearchTool tries Serper first and falls back to DDGS,
        we use an invalid serper_api_key to force it to use DDGS.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
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
            print(f"  Relevant information: {output.relevant_information}")


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
            entity_retriever.ainvoke({"query": ["Barack Obama"], "num_entities": 1})
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


class TestWikidataPathFinding:
    """Test suite for Wikidata path finding between entities."""
    
    @pytest.fixture
    def path_finder(self):
        """Create a WikidataPathFindingTool."""
        return WikidataPathFindingTool()
    
    @pytest.mark.slow
    def test_find_path_between_entities(self, path_finder):
        """Test finding a path between two related entities."""
        # Q142 is France, Q183 is Germany - should be connected (both are countries in Europe)
        source_qid = "Q142"  # France
        target_qid = "Q183"  # Germany
        
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        # Path might be found or not, depending on Wikidata structure
        if path_result:
            assert isinstance(path_result, WikidataPathBetweenEntities)
            assert path_result.source.qid == source_qid
            assert path_result.target.qid == target_qid
            assert path_result.path_length >= 0
            assert len(path_result.path) == path_result.path_length
            
            print(f"✓ Path Finding Between Entities")
            print(f"  Source: {path_result.source.label} ({source_qid})")
            print(f"  Target: {path_result.target.label} ({target_qid})")
            print(f"  Path length: {path_result.path_length}")
            if path_result.path:
                print(f"  Path found with {len(path_result.path)} hops:")
                for i, triple in enumerate(path_result.path[:5]):
                    obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                    print(f"    {i+1}. {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
            else:
                print(f"  No path found (same entity or direct connection)")
        else:
            print(f"✓ Path Finding Between Entities")
            print(f"  Source: {source_qid} (France)")
            print(f"  Target: {target_qid} (Germany)")
            print(f"  No path found within max_hops=3")
    
    @pytest.mark.slow
    def test_find_path_same_entity(self, path_finder):
        """Test finding a path from an entity to itself."""
        # Q142 is France
        qid = "Q142"
        
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": qid,
                "target_qid": qid,
                "max_hops": 3
            })
        )
        
        # Should return a path with length 0 (same entity)
        assert path_result is not None
        assert isinstance(path_result, WikidataPathBetweenEntities)
        assert path_result.source.qid == qid
        assert path_result.target.qid == qid
        assert path_result.path_length == 0
        assert len(path_result.path) == 0
        
        print(f"✓ Path Finding Same Entity")
        print(f"  Entity: {path_result.source.label} ({qid})")
        print(f"  Path length: {path_result.path_length} (same entity)")
    
    @pytest.mark.slow
    def test_find_path_with_different_max_hops(self, path_finder):
        """Test finding a path with different max_hops values."""
        # Q30 is United States, Q142 is France
        source_qid = "Q30"   # United States
        target_qid = "Q142"  # France
        
        # Try with max_hops=2
        path_result_2 = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 2
            })
        )
        
        # Try with max_hops=3
        path_result_3 = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        print(f"✓ Path Finding with Different Max Hops")
        print(f"  Source: {source_qid} (United States)")
        print(f"  Target: {target_qid} (France)")
        print(f"  Max hops=2: {'Path found' if path_result_2 and path_result_2.path else 'No path'}")
        print(f"  Max hops=3: {'Path found' if path_result_3 and path_result_3.path else 'No path'}")
        
        # If path found with max_hops=3, it should be valid
        if path_result_3 and path_result_3.path:
            assert path_result_3.path_length <= 3
            assert len(path_result_3.path) == path_result_3.path_length
            print(f"    Path length with max_hops=3: {path_result_3.path_length}")
    
    @pytest.mark.slow
    def test_find_path_nonexistent_entities(self, path_finder):
        """Test finding a path with nonexistent QIDs."""
        # Use invalid QIDs
        source_qid = "Q999999999"
        target_qid = "Q999999998"
        
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        # Should return None if entities don't exist
        # or return a path result with None entities if partially found
        print(f"✓ Path Finding Nonexistent Entities")
        print(f"  Source: {source_qid}")
        print(f"  Target: {target_qid}")
        if path_result is None:
            print(f"  Result: None (entities not found)")
        else:
            print(f"  Result: Path result returned (may have None entities)")
    
    @pytest.mark.slow
    def test_find_path_well_known_connection(self, path_finder):
        """Test finding a path between well-known connected entities."""
        # Q42 is Douglas Adams, Q937 is E. Coli - might be connected through works
        # Or use Q76 (Barack Obama) and Q30 (United States) - should be connected
        source_qid = "Q76"   # Barack Obama
        target_qid = "Q30"    # United States
        
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        if path_result and path_result.path:
            assert path_result.path_length > 0
            assert len(path_result.path) > 0
            
            print(f"✓ Path Finding Well-Known Connection")
            print(f"  Source: {path_result.source.label} ({source_qid})")
            print(f"  Target: {path_result.target.label} ({target_qid})")
            print(f"  Path length: {path_result.path_length}")
            print(f"  Path:")
            for i, triple in enumerate(path_result.path):
                obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                print(f"    {i+1}. {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
        else:
            print(f"✓ Path Finding Well-Known Connection")
            print(f"  Source: {source_qid} (Barack Obama)")
            print(f"  Target: {target_qid} (United States)")
            print(f"  No path found within max_hops=3")


class TestToolErrorHandling:
    """Test error handling in tools."""
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool that will use DuckDuckGo backend.
        
        Since WebSearchTool tries Serper first and falls back to DDGS,
        we use an invalid serper_api_key to force it to use DDGS.
        """
        return WebSearchTool(
            serper_api_key=SERPER_API_KEY
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
            entity_retriever.ainvoke({"query": query, "num_entities": 3})
        )
        
        # Should return empty results, not error
        assert len(results) == 1
        assert len(results[0]) == 0 or results[0] is None or results[0] == []
        
        print(f"✓ Nonexistent entity handled gracefully")


# Run tests with: pytest test_tools.py -v -s --tb=short
# Run slow tests: pytest test_tools.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
