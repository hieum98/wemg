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
    WikidataPathFindingTool,
    WikidataEntity,
    WikiTriple,
    WikidataPathBetweenEntities,
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
    
    # ==================== Single Query Tests (returns List[WikidataEntity]) ====================
    
    def test_invoke_single_query_basic(self, entity_tool):
        """Test single query returns List[WikidataEntity] (not nested)."""
        # Query for a well-known entity
        result = entity_tool.invoke({"query": "Albert Einstein"})
        
        # Single query should return List[WikidataEntity]
        assert isinstance(result, list)
        assert len(result) > 0
        # Should NOT be nested list
        assert not isinstance(result[0], list)
        
        # Verify first entity has expected attributes
        first_entity = result[0]
        assert isinstance(first_entity, WikidataEntity)
        assert first_entity.qid is not None
        assert first_entity.qid.startswith("Q")
        assert first_entity.label is not None
        assert first_entity.url is not None
        assert "wikidata.org" in first_entity.url
        
        print(f"✓ Single query retrieved {len(result)} entities for 'Albert Einstein'")
        print(f"  - QID: {first_entity.qid}")
        print(f"  - Label: {first_entity.label}")
        print(f"  - Description: {first_entity.description}")
    
    def test_invoke_single_query_with_num_entities(self, entity_tool):
        """Test single query with custom num_entities parameter."""
        result = entity_tool.invoke({"query": "Python programming", "num_entities": 2})
        
        # Single query returns List[WikidataEntity]
        assert isinstance(result, list)
        assert len(result) <= 2
        assert not isinstance(result[0], list) if result else True
        
        for entity in result:
            assert isinstance(entity, WikidataEntity)
            assert entity.qid is not None
        
        print(f"✓ Single query retrieved {len(result)} entities with num_entities=2")
    
    def test_invoke_single_query_empty(self, entity_tool):
        """Test single query with no results returns empty List."""
        result = entity_tool.invoke({"query": "xyzzy123nonexistent456entity"})
        
        # Should return empty list for non-existent entities
        assert isinstance(result, list)
        assert len(result) == 0
        
        print("✓ Single empty query returned empty list as expected")
    
    @pytest.mark.asyncio
    async def test_ainvoke_single_query_basic(self, entity_tool):
        """Test async single query returns List[WikidataEntity]."""
        result = await entity_tool.ainvoke({"query": "Marie Curie"})
        
        # Single query should return List[WikidataEntity]
        assert isinstance(result, list)
        assert len(result) > 0
        assert not isinstance(result[0], list)
        
        first_entity = result[0]
        assert isinstance(first_entity, WikidataEntity)
        assert first_entity.qid is not None
        assert first_entity.label is not None
        
        print(f"✓ [ASYNC] Single query retrieved {len(result)} entities for 'Marie Curie'")
    
    # ==================== Batch Query Tests (returns List[List[WikidataEntity]]) ====================
    
    def test_invoke_batch_query_basic(self, entity_tool):
        """Test batch query returns List[List[WikidataEntity]]."""
        queries = ["Albert Einstein", "Marie Curie"]
        result = entity_tool.invoke({"query": queries, "num_entities": 2})
        
        # Batch query should return List[List[WikidataEntity]]
        assert isinstance(result, list)
        assert len(result) == len(queries)
        
        for i, query_result in enumerate(result):
            # Each element should be a list
            assert isinstance(query_result, list), f"Result {i} should be a list"
            # Inner list contains WikidataEntity objects
            for entity in query_result:
                assert isinstance(entity, WikidataEntity)
                assert entity.qid is not None
        
        print(f"✓ Batch query returned {len(result)} result lists for {len(queries)} queries")
        for i, (q, r) in enumerate(zip(queries, result)):
            print(f"  - '{q}': {len(r)} entities")
    
    def test_invoke_batch_query_preserves_order(self, entity_tool):
        """Test that batch query results align with input order."""
        queries = ["Paris France", "Tokyo Japan", "New York City"]
        result = entity_tool.invoke({"query": queries, "num_entities": 1})
        
        # Should have same length as input
        assert len(result) == len(queries)
        
        # Each result should correspond to its query
        for i, (query, query_result) in enumerate(zip(queries, result)):
            assert isinstance(query_result, list)
            print(f"  Query '{query}' -> {len(query_result)} results")
        
        print(f"✓ Batch query preserved order for {len(queries)} queries")
    
    def test_invoke_batch_query_with_empty_results(self, entity_tool):
        """Test batch query handles mixed results (some empty)."""
        queries = ["Albert Einstein", "xyzzy123nonexistent", "Marie Curie"]
        result = entity_tool.invoke({"query": queries, "num_entities": 1})
        
        # Should always return List[List[...]] for batch
        assert len(result) == len(queries)
        assert isinstance(result[0], list)  # Einstein results
        assert isinstance(result[1], list)  # Empty list for nonexistent
        assert isinstance(result[2], list)  # Curie results
        
        # The nonexistent query should return empty list
        assert len(result[1]) == 0
        
        print(f"✓ Batch query handled mixed results correctly")
        print(f"  - '{queries[0]}': {len(result[0])} entities")
        print(f"  - '{queries[1]}': {len(result[1])} entities (expected 0)")
        print(f"  - '{queries[2]}': {len(result[2])} entities")
    
    def test_invoke_batch_query_with_qids(self, entity_tool):
        """Test batch query with QIDs directly."""
        qids = ["Q937", "Q7186"]  # Albert Einstein, Marie Curie
        result = entity_tool.invoke({"query": qids, "is_qids": True})
        
        # Should return List[List[WikidataEntity]]
        assert len(result) == len(qids)
        
        for i, (qid, query_result) in enumerate(zip(qids, result)):
            assert isinstance(query_result, list)
            if query_result:
                assert query_result[0].qid == qid
        
        print(f"✓ Batch QID query returned aligned results")
    
    @pytest.mark.asyncio
    async def test_ainvoke_batch_query_basic(self, entity_tool):
        """Test async batch query returns List[List[WikidataEntity]]."""
        queries = ["Albert Einstein", "Marie Curie", "Isaac Newton"]
        result = await entity_tool.ainvoke({"query": queries, "num_entities": 1})
        
        # Batch query should return List[List[WikidataEntity]]
        assert isinstance(result, list)
        assert len(result) == len(queries)
        
        for query_result in result:
            assert isinstance(query_result, list)
            for entity in query_result:
                assert isinstance(entity, WikidataEntity)
        
        print(f"✓ [ASYNC] Batch query returned {len(result)} result lists")
    
    @pytest.mark.asyncio
    async def test_ainvoke_batch_query_with_empty(self, entity_tool):
        """Test async batch query with some empty results."""
        queries = ["Marie Curie", "nonexistent12345xyz"]
        result = await entity_tool.ainvoke({"query": queries, "num_entities": 1})
        
        assert len(result) == len(queries)
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        assert len(result[1]) == 0  # Should be empty
        
        print(f"✓ [ASYNC] Batch query handled empty results correctly")
    
    # ==================== Content and Details Tests ====================
    
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


class TestWikidataIdSearch:
    """Test suite for CustomWikidataAPIWrapper._get_id method using real Wikidata server."""
    
    @pytest.fixture
    def wrapper(self):
        """Create a CustomWikidataAPIWrapper instance for testing."""
        return CustomWikidataAPIWrapper(lang="en", top_k_results=3, wikidata_props=[])
    
    def test_get_id_returns_qids_for_entity_search(self, wrapper):
        """Test that _get_id returns QIDs for entity search."""
        ids = wrapper._get_id("Albert Einstein")
        
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(x.startswith("Q") for x in ids)
        # Q937 is Albert Einstein's QID
        assert "Q937" in ids
        
        print(f"✓ _get_id returned {len(ids)} QIDs for 'Albert Einstein'")
        print(f"  IDs: {ids}")
    
    def test_get_id_returns_pids_for_property_search(self, wrapper):
        """Test that _get_id returns PIDs when id_type='property'."""
        ids = wrapper._get_id("instance of", id_type="property")
        
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(x.startswith("P") for x in ids)
        # P31 is "instance of" property
        assert "P31" in ids
        
        print(f"✓ _get_id returned {len(ids)} PIDs for 'instance of'")
        print(f"  IDs: {ids}")
    
    def test_get_id_direct_qid_passthrough(self, wrapper):
        """Test that direct QID input is passed through."""
        # Direct QID should be returned as-is
        ids = wrapper._get_id("Q42")
        assert ids == ["Q42"]
        
        # Direct QID with wrong id_type should return empty
        ids = wrapper._get_id("Q42", id_type="property")
        assert ids == []
        
        print("✓ Direct QID passthrough working correctly")
    
    def test_get_id_direct_pid_passthrough(self, wrapper):
        """Test that direct PID input is passed through."""
        # Direct PID (lowercase) should be normalized and returned
        ids = wrapper._get_id("p31", id_type="property")
        assert ids == ["P31"]
        
        # Direct PID with wrong id_type should return empty
        ids = wrapper._get_id("p31")  # default is entity
        assert ids == []
        
        print("✓ Direct PID passthrough working correctly")
    
    def test_get_id_empty_query(self, wrapper):
        """Test that empty query returns empty list."""
        ids = wrapper._get_id("")
        assert ids == []
        
        print("✓ Empty query returned empty list")
    
    def test_get_id_nonexistent_query(self, wrapper):
        """Test that nonexistent query returns empty list."""
        ids = wrapper._get_id("xyzzy123nonexistent456entity789")
        assert isinstance(ids, list)
        assert len(ids) == 0
        
        print("✓ Nonexistent query returned empty list")
    
    def test_get_id_batch_queries(self, wrapper):
        """Test batch _get_id with multiple queries."""
        queries = ["Albert Einstein", "Marie Curie", "Q42"]
        ids = wrapper._get_id(queries)
        
        # Batch query should return List[List[str]]
        assert isinstance(ids, list)
        assert len(ids) == len(queries)
        
        for i, query_ids in enumerate(ids):
            assert isinstance(query_ids, list)
        
        # First two are searches, third is direct QID
        assert "Q937" in ids[0]  # Albert Einstein
        assert "Q7186" in ids[1]  # Marie Curie
        assert ids[2] == ["Q42"]  # Direct QID passthrough
        
        print(f"✓ Batch _get_id returned aligned results")
        for q, r in zip(queries, ids):
            print(f"  - '{q}': {r}")
    
    def test_get_id_batch_with_empty(self, wrapper):
        """Test batch _get_id handles empty queries."""
        queries = ["Albert Einstein", "", "Q99"]
        ids = wrapper._get_id(queries)
        
        assert len(ids) == len(queries)
        assert len(ids[0]) > 0  # Einstein results
        assert ids[1] == []  # Empty query
        assert ids[2] == ["Q99"]  # Direct QID
        
        print(f"✓ Batch _get_id handled empty queries correctly")


class TestWikidataKHopTriplesRetrievalTool:
    """Test suite for WikidataKHopTriplesRetrievalTool."""
    
    @pytest.fixture
    def khop_tool(self):
        """Create a WikidataKHopTriplesRetrievalTool instance for testing."""
        return WikidataKHopTriplesRetrievalTool(
            wikidata_wrapper=CustomWikidataAPIWrapper(lang="en", top_k_results=3)
        )
    
    # ==================== Single Query Tests (returns List[WikiTriple]) ====================
    
    def test_invoke_single_query_basic_1hop(self, khop_tool):
        """Test single query returns List[WikiTriple] (not nested)."""
        result = khop_tool.invoke({
            "query": "Barack Obama",
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        # Single query should return List[WikiTriple]
        assert isinstance(result, list)
        assert len(result) > 0
        # Should NOT be nested list
        assert not isinstance(result[0], list)
        
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
        
        print(f"✓ Single query retrieved {len(result)} triples for 'Barack Obama' (1-hop)")
        print(f"  - Subject: {first_triple.subject.label} ({first_triple.subject.qid})")
        print(f"  - Relation: {first_triple.relation.label} ({first_triple.relation.pid})")
        if hasattr(first_triple.object, 'label'):
            print(f"  - Object: {first_triple.object.label}")
        else:
            print(f"  - Object: {first_triple.object}")
    
    def test_invoke_single_query_outgoing(self, khop_tool):
        """Test single query outgoing triple retrieval (bidirectional=False)."""
        result = khop_tool.invoke({
            "query": "Tokyo Japan",
            "k": 1,
            "num_entities": 1,
            "bidirectional": False,
            "update_with_details": False
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        assert len(result) > 0
        assert not isinstance(result[0], list)
        
        for triple in result:
            assert isinstance(triple, WikiTriple)
            assert triple.subject.qid is not None
        
        print(f"✓ Single query retrieved {len(result)} outgoing triples for 'Tokyo Japan'")
    
    def test_invoke_single_query_bidirectional(self, khop_tool):
        """Test single query bidirectional triple retrieval."""
        result = khop_tool.invoke({
            "query": "Leonardo da Vinci",
            "k": 1,
            "num_entities": 1,
            "bidirectional": True,
            "update_with_details": False
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        assert len(result) > 0
        assert not isinstance(result[0], list)
        
        for triple in result:
            assert isinstance(triple, WikiTriple)
        
        print(f"✓ Single query retrieved {len(result)} bidirectional triples for 'Leonardo da Vinci'")
    
    def test_invoke_single_query_with_details(self, khop_tool):
        """Test single query with full entity details."""
        result = khop_tool.invoke({
            "query": "Apple Inc",
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        
        if result:
            assert not isinstance(result[0], list)
            first_triple = result[0]
            subject = first_triple.subject
            
            # With update_with_details=True, entities should have content
            assert subject.wikidata_content is not None
            
            print(f"✓ Single query retrieved triples with entity details")
            print(f"  Subject content preview: {subject.wikidata_content[:150]}...")
    
    def test_invoke_single_query_empty(self, khop_tool):
        """Test single query with no results returns empty List."""
        result = khop_tool.invoke({
            "query": "xyzzy123nonexistent456entity789",
            "k": 1,
            "num_entities": 1
        })
        
        # Should return empty list for non-existent entities
        assert isinstance(result, list)
        assert len(result) == 0
        
        print("✓ Single empty query returned empty list as expected")
    
    @pytest.mark.asyncio
    async def test_ainvoke_single_query_basic(self, khop_tool):
        """Test async single query returns List[WikiTriple]."""
        result = await khop_tool.ainvoke({
            "query": "Elon Musk",
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        assert len(result) > 0
        assert not isinstance(result[0], list)
        
        first_triple = result[0]
        assert isinstance(first_triple, WikiTriple)
        assert first_triple.subject.qid is not None
        assert first_triple.relation.pid is not None
        
        print(f"✓ [ASYNC] Single query retrieved {len(result)} triples for 'Elon Musk'")
    
    @pytest.mark.asyncio
    async def test_ainvoke_single_query_bidirectional(self, khop_tool):
        """Test async single query bidirectional triple retrieval."""
        result = await khop_tool.ainvoke({
            "query": "Python programming language",
            "k": 1,
            "num_entities": 1,
            "bidirectional": True,
            "update_with_details": False
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        assert len(result) > 0
        assert not isinstance(result[0], list)
        
        for triple in result:
            assert isinstance(triple, WikiTriple)
        
        print(f"✓ [ASYNC] Single query retrieved {len(result)} bidirectional triples")
    
    @pytest.mark.asyncio
    async def test_ainvoke_single_query_with_details(self, khop_tool):
        """Test async single query with entity details."""
        result = await khop_tool.ainvoke({
            "query": "Google",
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        # Single query returns List[WikiTriple]
        assert isinstance(result, list)
        
        if result:
            assert not isinstance(result[0], list)
            first_triple = result[0]
            assert first_triple.subject.wikidata_content is not None
        
        print(f"✓ [ASYNC] Single query retrieved triples with entity details")
    
    # ==================== Batch Query Tests (returns List[List[WikiTriple]]) ====================
    
    def test_invoke_batch_query_basic(self, khop_tool):
        """Test batch query returns List[List[WikiTriple]]."""
        queries = ["Albert Einstein", "Marie Curie"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        # Batch query should return List[List[WikiTriple]]
        assert isinstance(result, list)
        assert len(result) == len(queries)
        
        for i, query_result in enumerate(result):
            # Each element should be a list
            assert isinstance(query_result, list), f"Result {i} should be a list"
            # Inner list contains WikiTriple objects
            for triple in query_result:
                assert isinstance(triple, WikiTriple)
        
        print(f"✓ Batch query returned {len(result)} result lists for {len(queries)} queries")
        for i, (q, r) in enumerate(zip(queries, result)):
            print(f"  - '{q}': {len(r)} triples")
    
    def test_invoke_batch_query_preserves_order(self, khop_tool):
        """Test that batch query results align with input order."""
        queries = ["Paris France", "Tokyo Japan", "New York City"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "bidirectional": False,
            "update_with_details": False
        })
        
        # Should have same length as input
        assert len(result) == len(queries)
        
        # Each result should correspond to its query
        for i, (query, query_result) in enumerate(zip(queries, result)):
            assert isinstance(query_result, list)
            print(f"  Query '{query}' -> {len(query_result)} triples")
        
        print(f"✓ Batch query preserved order for {len(queries)} queries")
    
    def test_invoke_batch_query_with_empty_results(self, khop_tool):
        """Test batch query handles mixed results (some empty)."""
        queries = ["Albert Einstein", "xyzzy123nonexistent", "Marie Curie"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        # Should always return List[List[...]] for batch
        assert len(result) == len(queries)
        assert isinstance(result[0], list)  # Einstein results
        assert isinstance(result[1], list)  # Empty list for nonexistent
        assert isinstance(result[2], list)  # Curie results
        
        # The nonexistent query should return empty list
        assert len(result[1]) == 0
        
        print(f"✓ Batch query handled mixed results correctly")
        print(f"  - '{queries[0]}': {len(result[0])} triples")
        print(f"  - '{queries[1]}': {len(result[1])} triples (expected 0)")
        print(f"  - '{queries[2]}': {len(result[2])} triples")
    
    def test_invoke_batch_query_with_qids(self, khop_tool):
        """Test batch query with QIDs directly."""
        qids = ["Q937", "Q7186"]  # Albert Einstein, Marie Curie
        result = khop_tool.invoke({
            "query": qids,
            "is_qids": True,
            "k": 1,
            "update_with_details": False
        })
        
        # Should return List[List[WikiTriple]]
        assert len(result) == len(qids)
        
        for i, (qid, query_result) in enumerate(zip(qids, result)):
            assert isinstance(query_result, list)
            # Triples should be from the corresponding entity
            for triple in query_result:
                assert triple.subject.qid == qid
        
        print(f"✓ Batch QID query returned aligned results")
        for qid, r in zip(qids, result):
            print(f"  - {qid}: {len(r)} triples")
    
    def test_invoke_batch_query_bidirectional(self, khop_tool):
        """Test batch query with bidirectional traversal."""
        queries = ["Barack Obama", "Elon Musk"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "bidirectional": True,
            "update_with_details": False
        })
        
        # Should return List[List[WikiTriple]]
        assert len(result) == len(queries)
        
        for i, query_result in enumerate(result):
            assert isinstance(query_result, list)
        
        print(f"✓ Batch bidirectional query returned {len(result)} result lists")
    
    def test_invoke_batch_query_with_details(self, khop_tool):
        """Test batch query with update_with_details=True."""
        queries = ["Albert Einstein", "Marie Curie"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        # Should return List[List[WikiTriple]]
        assert len(result) == len(queries)
        
        for query_result in result:
            assert isinstance(query_result, list)
            for triple in query_result:
                # With details, entities should have content
                if triple.subject.wikidata_content:
                    assert len(triple.subject.wikidata_content) > 0
        
        print(f"✓ Batch query with details returned {len(result)} result lists")
    
    @pytest.mark.asyncio
    async def test_ainvoke_batch_query_basic(self, khop_tool):
        """Test async batch query returns List[List[WikiTriple]]."""
        queries = ["Albert Einstein", "Marie Curie", "Isaac Newton"]
        result = await khop_tool.ainvoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        # Batch query should return List[List[WikiTriple]]
        assert isinstance(result, list)
        assert len(result) == len(queries)
        
        for query_result in result:
            assert isinstance(query_result, list)
            for triple in query_result:
                assert isinstance(triple, WikiTriple)
        
        print(f"✓ [ASYNC] Batch query returned {len(result)} result lists")
    
    @pytest.mark.asyncio
    async def test_ainvoke_batch_query_with_empty(self, khop_tool):
        """Test async batch query with some empty results."""
        queries = ["Marie Curie", "nonexistent12345xyz"]
        result = await khop_tool.ainvoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        assert len(result) == len(queries)
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        assert len(result[1]) == 0  # Should be empty
        
        print(f"✓ [ASYNC] Batch query handled empty results correctly")
    
    @pytest.mark.asyncio
    async def test_ainvoke_batch_query_with_details(self, khop_tool):
        """Test async batch query with update_with_details=True."""
        queries = ["Google", "Apple Inc"]
        result = await khop_tool.ainvoke({
            "query": queries,
            "k": 1,
            "num_entities": 1,
            "update_with_details": True
        })
        
        assert len(result) == len(queries)
        
        for query_result in result:
            assert isinstance(query_result, list)
            for triple in query_result:
                if triple.subject.wikidata_content:
                    assert len(triple.subject.wikidata_content) > 0
        
        print(f"✓ [ASYNC] Batch query with details returned {len(result)} result lists")
    
    # ==================== Deduplication Tests ====================
    
    def test_dedup_triples_single_query(self, khop_tool):
        """Test that duplicate triples are properly deduplicated for single query."""
        result = khop_tool.invoke({
            "query": "United States",
            "k": 1,
            "num_entities": 2,  # Multiple entities may have overlapping triples
            "update_with_details": False
        })
        
        # Single query returns List[WikiTriple]
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
    
    def test_dedup_triples_batch_query(self, khop_tool):
        """Test that each batch result has deduplicated triples."""
        queries = ["United States", "Germany"]
        result = khop_tool.invoke({
            "query": queries,
            "k": 1,
            "num_entities": 2,
            "update_with_details": False
        })
        
        # Batch query returns List[List[WikiTriple]]
        assert len(result) == len(queries)
        
        for i, query_result in enumerate(result):
            seen = set()
            for triple in query_result:
                if hasattr(triple.object, 'qid'):
                    triple_id = (triple.subject.qid, triple.relation.pid, triple.object.qid)
                else:
                    triple_id = (triple.subject.qid, triple.relation.pid, str(triple.object))
                
                assert triple_id not in seen, f"Query {i}: Found duplicate triple: {triple_id}"
                seen.add(triple_id)
        
        print(f"✓ Batch results all have unique triples (deduplication working)")


class TestWikidataPathFindingTool:
    """Test suite for WikidataPathFindingTool using real Wikidata calls."""

    @pytest.fixture
    def path_tool(self):
        """Create a WikidataPathFindingTool instance for testing."""
        return WikidataPathFindingTool(
            wikidata_wrapper=CustomWikidataAPIWrapper(lang="en", top_k_results=3)
        )

    def test_invoke_basic_path_found(self, path_tool):
        """Test path finding using invoke (sync)."""
        result = path_tool.invoke({
            "source_qid": "Q937", # Albert Einstein
            "target_qid": "Q30277523", # Abraham Einstein (his grandfather)
            "max_hops": 2,
        })
        assert isinstance(result, WikidataPathBetweenEntities)
        assert result.source.qid == "Q937"
        assert result.target.qid == "Q191583"
        assert result.path_length == len(result.path)
        assert result.path_length > 0

        # Basic structural checks for triples in the path
        for triple in result.path:
            assert isinstance(triple, WikiTriple)
            assert triple.subject is not None
            assert triple.subject.qid is not None
            assert triple.relation is not None
            assert triple.relation.pid is not None

        print(f"✓ Path found (sync): length={result.path_length}")

    @pytest.mark.asyncio
    async def test_ainvoke_basic_path_found(self, path_tool):
        """Test path finding using ainvoke (async)."""
        result = await path_tool.ainvoke({
            "source_qid": "Q937",
            "target_qid": "Q183",
            "max_hops": 2,
        })

        assert isinstance(result, WikidataPathBetweenEntities)
        assert result.source.qid == "Q937"
        assert result.target.qid == "Q183"
        assert result.path_length == len(result.path)
        assert result.path_length > 0

        print(f"✓ Path found (async): length={result.path_length}")


class TestWikidataToolsIntegration:
    """Integration tests combining both tools."""
    
    @pytest.fixture
    def entity_tool(self):
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def khop_tool(self):
        return WikidataKHopTriplesRetrievalTool()
    
    def test_entity_to_khop_workflow_single(self, entity_tool, khop_tool):
        """Test workflow: retrieve single entity, then get its triples."""
        # Single query returns List[WikidataEntity]
        entities = entity_tool.invoke({"query": "Isaac Newton", "num_entities": 1})
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert not isinstance(entities[0], list)  # Not nested
        entity = entities[0]
        
        # Single query returns List[WikiTriple]
        triples = khop_tool.invoke({
            "query": entity.label,
            "k": 1,
            "num_entities": 1,
            "update_with_details": False
        })
        
        assert isinstance(triples, list)
        assert len(triples) > 0
        assert not isinstance(triples[0], list)  # Not nested
        
        print(f"✓ Single entity-to-KHop workflow completed")
        print(f"  Entity: {entity.label} ({entity.qid})")
        print(f"  Retrieved {len(triples)} triples")
    
    def test_batch_entity_to_khop_workflow(self, entity_tool, khop_tool):
        """Test batch workflow: retrieve batch entities, then get triples."""
        queries = ["Albert Einstein", "Marie Curie"]
        
        # Batch query returns List[List[WikidataEntity]]
        entity_results = entity_tool.invoke({"query": queries, "num_entities": 1})
        
        assert len(entity_results) == len(queries)
        for er in entity_results:
            assert isinstance(er, list)
        
        # Extract QIDs from all results
        all_qids = []
        for er in entity_results:
            if er:
                all_qids.append(er[0].qid)
        
        # Batch k-hop query with QIDs returns List[List[WikiTriple]]
        triple_results = khop_tool.invoke({
            "query": all_qids,
            "is_qids": True,
            "k": 1,
            "update_with_details": False
        })
        
        assert len(triple_results) == len(all_qids)
        for tr in triple_results:
            assert isinstance(tr, list)
        
        print(f"✓ Batch entity-to-KHop workflow completed")
        for i, (q, er, tr) in enumerate(zip(queries, entity_results, triple_results)):
            print(f"  - '{q}': {len(er)} entities, {len(tr)} triples")
    
    @pytest.mark.asyncio
    async def test_concurrent_async_single_calls(self, entity_tool, khop_tool):
        """Test multiple concurrent single async calls."""
        queries = ["Albert Einstein", "Marie Curie", "Nikola Tesla"]
        
        # Execute multiple single entity retrievals concurrently
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
                # Single query returns List[WikidataEntity] (not nested)
                assert isinstance(result, list)
                if result:
                    assert not isinstance(result[0], list)
                    print(f"  - {query}: Retrieved {len(result)} entities")
                else:
                    print(f"  - {query}: No entities found")
        
        print(f"✓ [ASYNC] Concurrent single calls completed: {len(successful_results)}/{len(queries)} successful")
    
    @pytest.mark.asyncio
    async def test_concurrent_async_batch_call(self, entity_tool, khop_tool):
        """Test single batch async call vs concurrent single calls."""
        queries = ["Albert Einstein", "Marie Curie", "Nikola Tesla"]
        
        # Single batch call returns List[List[WikidataEntity]]
        batch_result = await entity_tool.ainvoke({"query": queries, "num_entities": 1})
        
        assert len(batch_result) == len(queries)
        for i, (q, r) in enumerate(zip(queries, batch_result)):
            assert isinstance(r, list)
            print(f"  - Batch '{q}': {len(r)} entities")
        
        print(f"✓ [ASYNC] Batch call completed with {len(queries)} queries")
    
    def test_debatching_entities(self, entity_tool):
        """Test that batch results can be easily debatched."""
        queries = ["Paris France", "Tokyo Japan", "New York City"]
        
        # Batch query returns List[List[WikidataEntity]]
        batch_result = entity_tool.invoke({"query": queries, "num_entities": 2})
        
        # Easy debatching - each index corresponds to original query
        for i, query in enumerate(queries):
            query_entities = batch_result[i]
            print(f"  - Query '{query}' -> {len(query_entities)} entities")
            
            # Can process each query's results independently
            for entity in query_entities:
                assert isinstance(entity, WikidataEntity)
        
        print(f"✓ Debatching entities working correctly")
    
    def test_debatching_triples(self, khop_tool):
        """Test that batch triple results can be easily debatched."""
        qids = ["Q937", "Q7186", "Q9047"]  # Einstein, Curie, Newton
        
        # Batch query returns List[List[WikiTriple]]
        batch_result = khop_tool.invoke({
            "query": qids,
            "is_qids": True,
            "k": 1,
            "update_with_details": False
        })
        
        # Easy debatching - each index corresponds to original QID
        for i, qid in enumerate(qids):
            query_triples = batch_result[i]
            print(f"  - QID '{qid}' -> {len(query_triples)} triples")
            
            # All triples should be from this QID's graph
            for triple in query_triples:
                assert triple.subject.qid == qid
        
        print(f"✓ Debatching triples working correctly")


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
