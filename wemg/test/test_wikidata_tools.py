"""Test suite for Wikidata retrieval tools.

Tests WikidataEntityRetrievalTool and WikidataKHopTriplesRetrievalTool
using langchain's invoke and ainvoke interfaces.
"""
import asyncio
import pytest
from typing import List

from wemg.agents.tools.wikidata import (
    WikidataEntityRetrievalTool,
    WikidataKHopTriplesRetrievalTool,
    WikidataEntity,
    WikiTriple,
    CustomWikidataAPIWrapper,
)

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class TestWikidataEntityRetrievalTool:
    """Test suite for WikidataEntityRetrievalTool."""
    
    @pytest.fixture
    def entity_tool(self):
        """Create a WikidataEntityRetrievalTool instance for testing."""
        return WikidataEntityRetrievalTool(
            wikidata_wrapper=CustomWikidataAPIWrapper(lang="en", top_k_results=3)
        )
    
    def test_invoke_basic(self, entity_tool):
        """Test basic entity retrieval using invoke (sync)."""
        # Query for a well-known entity
        result = entity_tool.invoke({"query": "Albert Einstein"})
        
        # Should return a list of WikidataEntity objects
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verify first entity has expected attributes
        first_entity = result[0]
        assert isinstance(first_entity, WikidataEntity)
        assert first_entity.qid is not None
        assert first_entity.qid.startswith("Q")
        assert first_entity.label is not None
        assert first_entity.url is not None
        assert "wikidata.org" in first_entity.url
        
        print(f"✓ Retrieved {len(result)} entities for 'Albert Einstein'")
        print(f"  - QID: {first_entity.qid}")
        print(f"  - Label: {first_entity.label}")
        print(f"  - Description: {first_entity.description}")
    
    def test_invoke_with_num_entities(self, entity_tool):
        """Test entity retrieval with custom num_entities parameter."""
        # Request fewer entities
        result = entity_tool.invoke({"query": "Python programming", "num_entities": 2})
        
        assert isinstance(result, list)
        # Should return at most 2 entities
        assert len(result) <= 2
        
        if result:
            for entity in result:
                assert isinstance(entity, WikidataEntity)
                assert entity.qid is not None
        
        print(f"✓ Retrieved {len(result)} entities with num_entities=2")
    
    def test_invoke_empty_query(self, entity_tool):
        """Test entity retrieval with query that returns no results."""
        # Use a very specific/nonsense query
        result = entity_tool.invoke({"query": "xyzzy123nonexistent456entity"})
        
        # Should return empty list for non-existent entities
        assert isinstance(result, list)
        assert len(result) == 0
        
        print("✓ Empty query returned empty list as expected")
    
    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, entity_tool):
        """Test basic entity retrieval using ainvoke (async)."""
        # Query for a well-known entity
        result = await entity_tool.ainvoke({"query": "Marie Curie"})
        
        # Should return a list of WikidataEntity objects
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verify first entity
        first_entity = result[0]
        assert isinstance(first_entity, WikidataEntity)
        assert first_entity.qid is not None
        assert first_entity.label is not None
        
        print(f"✓ [ASYNC] Retrieved {len(result)} entities for 'Marie Curie'")
        print(f"  - QID: {first_entity.qid}")
        print(f"  - Label: {first_entity.label}")
    
    @pytest.mark.asyncio
    async def test_ainvoke_with_num_entities(self, entity_tool):
        """Test async entity retrieval with custom num_entities parameter."""
        result = await entity_tool.ainvoke({"query": "New York City", "num_entities": 1})
        
        assert isinstance(result, list)
        assert len(result) <= 1
        
        if result:
            assert isinstance(result[0], WikidataEntity)
            assert result[0].qid is not None
        
        print(f"✓ [ASYNC] Retrieved {len(result)} entity with num_entities=1")
    
    def test_entity_content_populated(self, entity_tool):
        """Test that entity content fields are properly populated."""
        result = entity_tool.invoke({"query": "Paris France", "num_entities": 1})
        
        assert len(result) > 0
        entity = result[0]
        
        # Check that wikidata_content is populated
        assert entity.wikidata_content is not None
        assert len(entity.wikidata_content) > 0
        
        print(f"✓ Entity '{entity.label}' has wikidata_content populated")
        print(f"  Content preview: {entity.wikidata_content[:200]}...")


class TestWikidataKHopTriplesRetrievalTool:
    """Test suite for WikidataKHopTriplesRetrievalTool."""
    
    @pytest.fixture
    def khop_tool(self):
        """Create a WikidataKHopTriplesRetrievalTool instance for testing."""
        return WikidataKHopTriplesRetrievalTool(
            wikidata_wrapper=CustomWikidataAPIWrapper(lang="en", top_k_results=3)
        )
    
    def test_invoke_basic_1hop(self, khop_tool):
        """Test basic 1-hop triple retrieval using invoke (sync)."""
        result = khop_tool.invoke({
            "query": "Barack Obama",
            "k": 1,
            "num_entities": 1,
            "update_with_details": False  # Skip detailed fetch for speed
        })
        
        # Should return a list of WikiTriple objects
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verify first triple structure
        first_triple = result[0]
        assert isinstance(first_triple, WikiTriple)
        assert first_triple.subject is not None
        assert first_triple.relation is not None
        assert first_triple.object is not None
        
        # Subject should be a WikidataEntity
        assert isinstance(first_triple.subject, WikidataEntity)
        assert first_triple.subject.qid is not None
        
        # Relation should have pid
        assert first_triple.relation.pid is not None
        assert first_triple.relation.pid.startswith("P")
        
        print(f"✓ Retrieved {len(result)} triples for 'Barack Obama' (1-hop)")
        print(f"  - Subject: {first_triple.subject.label} ({first_triple.subject.qid})")
        print(f"  - Relation: {first_triple.relation.label} ({first_triple.relation.pid})")
        if hasattr(first_triple.object, 'label'):
            print(f"  - Object: {first_triple.object.label}")
        else:
            print(f"  - Object: {first_triple.object}")
    
    def test_invoke_outgoing_triples(self, khop_tool):
        """Test outgoing triple retrieval (bidirectional=False)."""
        result = khop_tool.invoke({
            "query": "Tokyo Japan",
            "k": 1,
            "num_entities": 1,
            "bidirectional": False,
            "update_with_details": False
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # All triples should have the queried entity as subject
        for triple in result:
            assert isinstance(triple, WikiTriple)
            assert triple.subject.qid is not None
        
        print(f"✓ Retrieved {len(result)} outgoing triples for 'Tokyo Japan'")
    
    def test_invoke_bidirectional_triples(self, khop_tool):
        """Test bidirectional triple retrieval."""
        result = khop_tool.invoke({
            "query": "Leonardo da Vinci",
            "k": 1,
            "num_entities": 1,
            "bidirectional": True,
            "update_with_details": False
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # All triples should be valid WikiTriple objects
        for triple in result:
            assert isinstance(triple, WikiTriple)
        
        print(f"✓ Retrieved {len(result)} bidirectional triples for 'Leonardo da Vinci'")
    
    def test_invoke_with_details(self, khop_tool):
        """Test triple retrieval with full entity details."""
        result = khop_tool.invoke({
            "query": "Apple Inc",
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        assert isinstance(result, list)
        
        if result:
            # Check that entities have detailed content
            first_triple = result[0]
            subject = first_triple.subject
            
            # With update_with_details=True, entities should have content
            assert subject.wikidata_content is not None
            
            print(f"✓ Retrieved triples with entity details")
            print(f"  Subject content preview: {subject.wikidata_content[:150]}...")
    
    def test_invoke_empty_query(self, khop_tool):
        """Test triple retrieval with query that returns no results."""
        result = khop_tool.invoke({
            "query": "xyzzy123nonexistent456entity789",
            "k": 1,
            "num_entities": 1
        })
        
        # Should return None or empty list for non-existent entities
        assert result is None or (isinstance(result, list) and len(result) == 0)
        
        print("✓ Non-existent query handled correctly")
    
    @pytest.mark.asyncio
    async def test_ainvoke_basic(self, khop_tool):
        """Test basic triple retrieval using ainvoke (async)."""
        result = await khop_tool.ainvoke({
            "query": "Elon Musk",
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        first_triple = result[0]
        assert isinstance(first_triple, WikiTriple)
        assert first_triple.subject.qid is not None
        assert first_triple.relation.pid is not None
        
        print(f"✓ [ASYNC] Retrieved {len(result)} triples for 'Elon Musk'")
    
    @pytest.mark.asyncio
    async def test_ainvoke_bidirectional(self, khop_tool):
        """Test async bidirectional triple retrieval."""
        result = await khop_tool.ainvoke({
            "query": "Python programming language",
            "k": 1,
            "num_entities": 1,
            "bidirectional": True,
            "update_with_details": False
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        for triple in result:
            assert isinstance(triple, WikiTriple)
        
        print(f"✓ [ASYNC] Retrieved {len(result)} bidirectional triples")
    
    @pytest.mark.asyncio
    async def test_ainvoke_with_details(self, khop_tool):
        """Test async triple retrieval with entity details."""
        result = await khop_tool.ainvoke({
            "query": "Google",
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        assert isinstance(result, list)
        
        if result:
            first_triple = result[0]
            # With details, entities should have content populated
            assert first_triple.subject.wikidata_content is not None
        
        print(f"✓ [ASYNC] Retrieved triples with entity details")
    
    def test_dedup_triples(self, khop_tool):
        """Test that duplicate triples are properly deduplicated."""
        result = khop_tool.invoke({
            "query": "United States",
            "k": 1,
            "num_entities": 2,  # Multiple entities may have overlapping triples
            "update_with_details": False
        })
        
        assert isinstance(result, list)
        
        # Check for uniqueness - no duplicate (subject, relation, object) combinations
        seen = set()
        for triple in result:
            if hasattr(triple.object, 'qid'):
                triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
            else:
                triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
            
            assert triple_id not in seen, f"Found duplicate triple: {triple_id}"
            seen.add(triple_id)
        
        print(f"✓ All {len(result)} triples are unique (deduplication working)")


class TestWikidataToolsIntegration:
    """Integration tests combining both tools."""
    
    @pytest.fixture
    def entity_tool(self):
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def khop_tool(self):
        return WikidataKHopTriplesRetrievalTool()
    
    def test_entity_to_khop_workflow(self, entity_tool, khop_tool):
        """Test workflow: retrieve entity, then get its triples."""
        # First, retrieve an entity
        entities = entity_tool.invoke({"query": "Isaac Newton", "num_entities": 1})
        
        assert len(entities) > 0
        entity = entities[0]
        
        # Then, get triples for that entity using its label
        triples = khop_tool.invoke({
            "query": entity.label,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        assert isinstance(triples, list)
        assert len(triples) > 0
        
        print(f"✓ Entity-to-KHop workflow completed")
        print(f"  Entity: {entity.label} ({entity.qid})")
        print(f"  Retrieved {len(triples)} triples")
    
    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self, entity_tool, khop_tool):
        """Test multiple concurrent async calls."""
        queries = ["Albert Einstein", "Marie Curie", "Nikola Tesla"]
        
        # Execute multiple entity retrievals concurrently
        tasks = [
            entity_tool.ainvoke({"query": q, "num_entities": 1})
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        assert len(successful_results) > 0
        
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                print(f"  - {query}: Error - {result}")
            else:
                assert isinstance(result, list)
                if result:
                    print(f"  - {query}: Retrieved {len(result)} entities")
                else:
                    print(f"  - {query}: No entities found")
        
        print(f"✓ [ASYNC] Concurrent calls completed: {len(successful_results)}/{len(queries)} successful")


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
