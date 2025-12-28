"""
Comprehensive tests for the refactored Wikidata module.

Tests cover:
- Utilities (utils.py): validation, normalization, batch processing
- Models (models.py): entity, property, triple models
- Enrichment (enrichment.py): EnrichmentCollector
- API Wrapper (api_wrapper.py): core methods and k-hop traversal
- Tools (tools.py): BaseTool classes
- Integration: end-to-end workflows

These tests include both unit tests (fast, no API calls) and integration tests (slow, real API calls).
"""
import os
import pytest
import asyncio
from typing import List, Dict, Set

from wemg.agents.tools.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    WikidataPathBetweenEntities,
    CustomWikidataAPIWrapper,
    WikidataEntityRetrievalTool,
    WikidataPropertyRetrievalTool,
    WikidataKHopTriplesRetrievalTool,
    WikidataPathFindingTool,
    DEFAULT_PROPERTIES,
    PROPERTY_LABELS,
)
from wemg.agents.tools.wikidata.utils import (
    normalize_and_validate_qid,
    normalize_and_validate_pid,
    extract_id_from_uri,
    normalize_single_or_list,
    map_results_to_indices,
    unwrap_single_result,
    flatten_and_map_ids,
    build_result_map,
    create_minimal_entity,
    create_minimal_property,
    build_property_filter,
    build_property_values_clause,
    validate_and_normalize_ids,
)
from wemg.agents.tools.wikidata.enrichment import EnrichmentCollector


# ============================================================================
# Unit Tests - Utilities
# ============================================================================

class TestWikidataUtils:
    """Unit tests for Wikidata utilities."""
    
    def test_normalize_and_validate_qid(self):
        """Test QID normalization and validation."""
        # Valid QIDs
        assert normalize_and_validate_qid("Q42") == "Q42"
        assert normalize_and_validate_qid("q42") == "Q42"
        assert normalize_and_validate_qid(" Q42 ") == "Q42"
        assert normalize_and_validate_qid("Q123456") == "Q123456"
        
        # Invalid QIDs
        assert normalize_and_validate_qid("P42") is None  # Wrong prefix
        assert normalize_and_validate_qid("42") is None  # Missing prefix
        assert normalize_and_validate_qid("Q") is None  # No number
        assert normalize_and_validate_qid("") is None  # Empty
        assert normalize_and_validate_qid(None) is None  # None
        assert normalize_and_validate_qid("Q42abc") is None  # Invalid chars
    
    def test_normalize_and_validate_pid(self):
        """Test PID normalization and validation."""
        # Valid PIDs
        assert normalize_and_validate_pid("P31") == "P31"
        assert normalize_and_validate_pid("p31") == "P31"
        assert normalize_and_validate_pid(" P31 ") == "P31"
        assert normalize_and_validate_pid("P123456") == "P123456"
        
        # Invalid PIDs
        assert normalize_and_validate_pid("Q31") is None  # Wrong prefix
        assert normalize_and_validate_pid("31") is None  # Missing prefix
        assert normalize_and_validate_pid("P") is None  # No number
        assert normalize_and_validate_pid("") is None  # Empty
        assert normalize_and_validate_pid(None) is None  # None
    
    def test_extract_id_from_uri(self):
        """Test ID extraction from URIs."""
        # Valid URIs
        assert extract_id_from_uri("http://www.wikidata.org/entity/Q42") == "Q42"
        assert extract_id_from_uri("https://www.wikidata.org/entity/Q42") == "Q42"
        assert extract_id_from_uri("http://www.wikidata.org/entity/P31") == "P31"
        assert extract_id_from_uri("Q42") == "Q42"  # Already an ID
        assert extract_id_from_uri("P31") == "P31"  # Already an ID
        
        # Invalid URIs
        assert extract_id_from_uri("http://example.com") is None
        assert extract_id_from_uri("") is None
        assert extract_id_from_uri(None) is None
    
    def test_normalize_single_or_list(self):
        """Test single/list normalization."""
        # Single value
        is_single, values = normalize_single_or_list("Q42")
        assert is_single is True
        assert values == ["Q42"]
        
        # List
        is_single, values = normalize_single_or_list(["Q42", "Q43"])
        assert is_single is False
        assert values == ["Q42", "Q43"]
        
        # Empty list
        is_single, values = normalize_single_or_list([])
        assert is_single is False
        assert values == []
    
    def test_map_results_to_indices(self):
        """Test result mapping to indices."""
        results = {"Q42": WikidataEntity(qid="Q42", label="Test"), "Q43": WikidataEntity(qid="Q43", label="Test2")}
        id_to_indices = {"Q42": [0, 2], "Q43": [1]}
        total = 3
        
        output = map_results_to_indices(results, id_to_indices, total)
        
        assert len(output) == 3
        assert output[0].qid == "Q42"
        assert output[1].qid == "Q43"
        assert output[2].qid == "Q42"
        assert output[0] == output[2]  # Same object
    
    def test_unwrap_single_result(self):
        """Test unwrapping single results."""
        # Single result
        result = unwrap_single_result(["Q42"], True)
        assert result == "Q42"
        
        # List result
        result = unwrap_single_result(["Q42", "Q43"], False)
        assert result == ["Q42", "Q43"]
        
        # Empty list, single
        result = unwrap_single_result([], True)
        assert result is None
    
    def test_flatten_and_map_ids(self):
        """Test ID flattening and mapping."""
        ids_per_query = [["Q42", "Q43"], ["Q42", "Q44"], ["Q45"]]
        
        all_ids, id_to_query_idx = flatten_and_map_ids(ids_per_query)
        
        assert set(all_ids) == {"Q42", "Q43", "Q44", "Q45"}
        assert id_to_query_idx["Q42"] == [0, 1]  # Appears in queries 0 and 1
        assert id_to_query_idx["Q43"] == [0]
        assert id_to_query_idx["Q44"] == [1]
        assert id_to_query_idx["Q45"] == [2]
    
    def test_build_result_map(self):
        """Test result map building."""
        results = [
            WikidataEntity(qid="Q42", label="Test1"),
            WikidataEntity(qid="Q43", label="Test2"),
        ]
        all_ids = ["Q42", "Q43"]
        
        result_map = build_result_map(results, all_ids)
        
        assert len(result_map) == 2
        assert result_map["Q42"].qid == "Q42"
        assert result_map["Q43"].qid == "Q43"
        
        # Single result
        single_result = WikidataEntity(qid="Q42", label="Test")
        result_map = build_result_map(single_result, ["Q42"])
        assert result_map["Q42"].qid == "Q42"
    
    def test_create_minimal_entity(self):
        """Test minimal entity creation."""
        entity = create_minimal_entity("Q42")
        
        assert entity.qid == "Q42"
        assert entity.label == ""
        assert entity.description == ""
        assert entity.url == "https://www.wikidata.org/wiki/Q42"
    
    def test_create_minimal_property(self):
        """Test minimal property creation."""
        wrapper = CustomWikidataAPIWrapper(lang="en")
        wrapper.wikidata_props_with_labels["P31"] = {
            "label": "instance of",
            "description": "that class of which this subject is a particular example"
        }
        
        prop = create_minimal_property("P31", wrapper)
        
        assert prop.pid == "P31"
        assert prop.label == "instance of"
        assert prop.description == "that class of which this subject is a particular example"
        
        # Property not in cache
        prop2 = create_minimal_property("P999", wrapper)
        assert prop2.pid == "P999"
        assert prop2.label == ""
    
    def test_build_property_filter(self):
        """Test property filter building."""
        # With properties
        properties = ["P31", "P27"]
        filter_clause = build_property_filter(properties)
        assert "P31" in filter_clause
        assert "P27" in filter_clause
        assert "FILTER" in filter_clause
        
        # Empty properties
        filter_clause = build_property_filter([])
        assert "STRSTARTS" in filter_clause
    
    def test_build_property_values_clause(self):
        """Test property VALUES clause building."""
        properties = ["P31", "P27"]
        clause, props_set = build_property_values_clause(properties)
        
        assert "P31" in clause
        assert "P27" in clause
        assert "VALUES" in clause
        assert props_set == {"P31", "P27"}
        
        # Empty
        clause, props_set = build_property_values_clause([])
        assert clause == ""
        assert props_set == set()
    
    def test_validate_and_normalize_ids(self):
        """Test ID validation and normalization."""
        ids = ["Q42", "q43", " Q44 ", "invalid", "Q45"]
        valid_ids, id_to_indices = validate_and_normalize_ids(ids, id_type="item")
        
        assert set(valid_ids) == {"Q42", "Q43", "Q44", "Q45"}
        assert id_to_indices["Q42"] == [0]
        assert id_to_indices["Q43"] == [1]
        assert id_to_indices["Q44"] == [2]
        assert id_to_indices["Q45"] == [4]
        
        # Properties
        pids = ["P31", "p27", " P28 ", "invalid"]
        valid_pids, pid_to_indices = validate_and_normalize_ids(pids, id_type="property")
        assert set(valid_pids) == {"P31", "P27", "P28"}
        assert pid_to_indices["P31"] == [0]
        assert pid_to_indices["P27"] == [1]
        assert pid_to_indices["P28"] == [2]


# ============================================================================
# Unit Tests - Models
# ============================================================================

class TestWikidataModels:
    """Unit tests for Wikidata models."""
    
    def test_wikidata_entity(self):
        """Test WikidataEntity model."""
        entity = WikidataEntity(
            qid="Q42",
            label="Douglas Adams",
            description="English writer",
            aliases=["Douglas Noël Adams"],
            url="https://www.wikidata.org/wiki/Q42"
        )
        
        assert entity.qid == "Q42"
        assert entity.label == "Douglas Adams"
        assert entity.description == "English writer"
        assert len(entity.aliases) == 1
        assert entity.url == "https://www.wikidata.org/wiki/Q42"
        
        # Test with None url (should be allowed)
        entity2 = WikidataEntity(qid="Q42", label="Test", url=None)
        assert entity2.url is None
    
    def test_wikidata_entity_to_context(self):
        """Test entity to_context method."""
        entity = WikidataEntity(
            qid="Q42",
            label="Douglas Adams",
            description="English writer"
        )
        
        context = entity.to_context(include_wiki_page=False)
        assert "Douglas Adams" in context
        assert "English writer" in context
        
        # With Wikipedia content
        entity.wikipedia_content = "Douglas Adams was an English writer..."
        context = entity.to_context(include_wiki_page=True)
        assert "Wikipedia Content" in context
        assert "Douglas Adams was an English writer" in context
    
    def test_wikidata_entity_str(self):
        """Test entity __str__ method."""
        entity = WikidataEntity(qid="Q42", label="Douglas Adams", description="English writer")
        assert str(entity) == "Douglas Adams - English writer"
        
        # No description
        entity2 = WikidataEntity(qid="Q42", label="Douglas Adams")
        assert str(entity2) == "Douglas Adams"
        
        # No label
        entity3 = WikidataEntity(qid="Q42")
        assert str(entity3) == "Q42"
    
    def test_wikidata_entity_hash(self):
        """Test entity hashing and equality (compares by QID only)."""
        entity1 = WikidataEntity(qid="Q42", label="Test")
        entity2 = WikidataEntity(qid="Q42", label="Different")
        
        # Hash is based on QID
        assert hash(entity1) == hash(entity2)  # Same QID
        
        # Entities compare by QID only (custom __eq__ method)
        assert entity1 == entity2  # Same QID, so equal
        
        # Entities with different QIDs are not equal
        entity3 = WikidataEntity(qid="Q43", label="Test")
        assert entity1 != entity3  # Different QID
    
    def test_wikidata_property(self):
        """Test WikidataProperty model."""
        prop = WikidataProperty(
            pid="P31",
            label="instance of",
            description="that class of which this subject is a particular example"
        )
        
        assert prop.pid == "P31"
        assert prop.label == "instance of"
        assert prop.description == "that class of which this subject is a particular example"
    
    def test_wikidata_property_str(self):
        """Test property __str__ method."""
        prop = WikidataProperty(pid="P31", label="instance of", description="test")
        assert str(prop) == "instance of: test"
        
        # No description
        prop2 = WikidataProperty(pid="P31", label="instance of")
        assert str(prop2) == "instance of"
        
        # No label (should use PID)
        prop3 = WikidataProperty(pid="P31")
        assert str(prop3) == "P31"
    
    def test_wikidata_property_hash(self):
        """Test property hashing."""
        prop1 = WikidataProperty(pid="P31", label="Test")
        prop2 = WikidataProperty(pid="P31", label="Different")
        
        assert hash(prop1) == hash(prop2)  # Same PID
    
    def test_wiki_triple(self):
        """Test WikiTriple model."""
        subject = WikidataEntity(qid="Q42", label="Douglas Adams")
        relation = WikidataProperty(pid="P31", label="instance of")
        object_entity = WikidataEntity(qid="Q5", label="human")
        
        triple = WikiTriple(
            subject=subject,
            relation=relation,
            object=object_entity
        )
        
        assert triple.subject.qid == "Q42"
        assert triple.relation.pid == "P31"
        assert triple.object.qid == "Q5"
        
        # With literal object
        triple2 = WikiTriple(
            subject=subject,
            relation=relation,
            object="some literal value"
        )
        assert triple2.object == "some literal value"
    
    def test_wiki_triple_str(self):
        """Test triple __str__ method."""
        subject = WikidataEntity(qid="Q42", label="Douglas Adams")
        relation = WikidataProperty(pid="P31", label="instance of")
        object_entity = WikidataEntity(qid="Q5", label="human")
        
        triple = WikiTriple(subject=subject, relation=relation, object=object_entity)
        triple_str = str(triple)
        
        assert "Douglas Adams" in triple_str
        assert "instance of" in triple_str
        assert "human" in triple_str


# ============================================================================
# Unit Tests - Enrichment
# ============================================================================

class TestEnrichmentCollector:
    """Unit tests for EnrichmentCollector."""
    
    @pytest.fixture
    def api_wrapper(self):
        """Create a CustomWikidataAPIWrapper for testing."""
        return CustomWikidataAPIWrapper(lang="en", top_k_results=3)
    
    @pytest.fixture
    def collector(self, api_wrapper):
        """Create an EnrichmentCollector."""
        return EnrichmentCollector(api_wrapper)
    
    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert len(collector.entity_qids_to_enrich) == 0
        assert len(collector.property_pids_to_enrich) == 0
        assert len(collector.enriched_entities) == 0
        assert len(collector.enriched_properties) == 0
    
    def test_add_entity_qid(self, collector):
        """Test adding entity QIDs."""
        collector.add_entity_qid("Q42")
        assert "Q42" in collector.entity_qids_to_enrich
        
        # Adding same QID twice should not duplicate
        collector.add_entity_qid("Q42")
        assert len(collector.entity_qids_to_enrich) == 1
        
        # Invalid QID
        collector.add_entity_qid("")
        assert "" not in collector.entity_qids_to_enrich
    
    def test_add_property_pid(self, collector, api_wrapper):
        """Test adding property PIDs."""
        collector.add_property_pid("P31")
        assert "P31" in collector.property_pids_to_enrich
        
        # If already in cache, should not add
        api_wrapper.wikidata_props_with_labels["P31"] = {
            "label": "instance of",
            "description": "test"
        }
        collector2 = EnrichmentCollector(api_wrapper)
        collector2.add_property_pid("P31")
        assert "P31" not in collector2.property_pids_to_enrich
    
    def test_collect_from_triples(self, collector):
        """Test collecting from triples."""
        subject = WikidataEntity(qid="Q42", label="")
        relation = WikidataProperty(pid="P31", label="")
        object_entity = WikidataEntity(qid="Q5", label="")
        
        triple = WikiTriple(subject=subject, relation=relation, object=object_entity)
        
        collector.collect_from_triples([triple])
        
        print(f"\n✓ test_collect_from_triples")
        print(f"  Collected entity QIDs: {sorted(collector.entity_qids_to_enrich)}")
        print(f"  Collected property PIDs: {sorted(collector.property_pids_to_enrich)}")
        
        assert "Q42" in collector.entity_qids_to_enrich
        assert "Q5" in collector.entity_qids_to_enrich
        assert "P31" in collector.property_pids_to_enrich
        
        # With literal object
        triple2 = WikiTriple(
            subject=subject,
            relation=relation,
            object="literal value"
        )
        collector2 = EnrichmentCollector(collector.api_wrapper)
        collector2.collect_from_triples([triple2])
        print(f"  With literal object - QIDs: {sorted(collector2.entity_qids_to_enrich)}, PIDs: {sorted(collector2.property_pids_to_enrich)}")
        assert "Q42" in collector2.entity_qids_to_enrich
        assert "P31" in collector2.property_pids_to_enrich
    
    def test_collect_from_entities(self, collector):
        """Test collecting from entities."""
        entities = [
            WikidataEntity(qid="Q42", label=""),
            WikidataEntity(qid="Q43", label=""),
        ]
        
        collector.collect_from_entities(entities)
        
        print(f"\n✓ test_collect_from_entities")
        print(f"  Input entities: {len(entities)}")
        print(f"  Collected QIDs: {sorted(collector.entity_qids_to_enrich)}")
        
        assert "Q42" in collector.entity_qids_to_enrich
        assert "Q43" in collector.entity_qids_to_enrich


# ============================================================================
# Integration Tests - API Wrapper
# ============================================================================

class TestCustomWikidataAPIWrapper:
    """Integration tests for CustomWikidataAPIWrapper."""
    
    @pytest.fixture
    def wrapper(self):
        """Create a CustomWikidataAPIWrapper."""
        return CustomWikidataAPIWrapper(lang="en", top_k_results=3)
    
    @pytest.mark.slow
    def test_get_id_single(self, wrapper):
        """Test getting ID for single query."""
        result = wrapper._get_id("Albert Einstein")
        print(f"\n✓ test_get_id_single")
        print(f"  Query: 'Albert Einstein'")
        print(f"  Found {len(result)} IDs:")
        for r in result[:5]:
            print(f"    - {r}")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].startswith("Q")
    
    @pytest.mark.slow
    def test_get_id_multiple(self, wrapper):
        """Test getting IDs for multiple queries."""
        results = wrapper._get_id(["Albert Einstein", "Paris"])
        print(f"\n✓ test_get_id_multiple")
        print(f"  Query: ['Albert Einstein', 'Paris']")
        print(f"  Results: {len(results)} query results")
        for i, r in enumerate(results):
            print(f"    Query {i+1}: {len(r)} IDs found - {r[:3]}")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) > 0 for r in results)
    
    @pytest.mark.slow
    def test_get_id_direct_qid(self, wrapper):
        """Test getting ID with direct QID input."""
        result = wrapper._get_id("Q42")
        print(f"\n✓ test_get_id_direct_qid")
        print(f"  Query: 'Q42'")
        print(f"  Result: {result}")
        assert isinstance(result, list)
        assert result[0] == "Q42"
    
    @pytest.mark.slow
    def test_get_item_single(self, wrapper: CustomWikidataAPIWrapper):
        """Test getting single item."""
        entity = wrapper._get_item("Q42")
        print(f"\n✓ test_get_item_single")
        print(f"  QID: Q42")
        print(f"  Entity: {entity.label} ({entity.qid})")
        print(f"  Description: {entity.description}")
        assert entity is not None
        assert isinstance(entity, WikidataEntity)
        assert entity.qid == "Q42"
        assert entity.label is not None
    
    @pytest.mark.slow
    def test_get_item_multiple(self, wrapper):
        """Test getting multiple items."""
        entities = wrapper._get_item(["Q42", "Q142"])
        print(f"\n✓ test_get_item_multiple")
        print(f"  QIDs: ['Q42', 'Q142']")
        print(f"  Results: {len(entities)} entities")
        for e in entities:
            if e:
                print(f"    - {e.label} ({e.qid}): {e.description}")
        assert isinstance(entities, list)
        assert len(entities) == 2
        assert all(isinstance(e, WikidataEntity) for e in entities if e is not None)
    
    @pytest.mark.slow
    def test_get_property_single(self, wrapper):
        """Test getting single property."""
        prop = wrapper._get_property("P31")
        print(f"\n✓ test_get_property_single")
        print(f"  PID: P31")
        print(f"  Property: {prop.label} ({prop.pid})")
        print(f"  Description: {prop.description}")
        assert prop is not None
        assert isinstance(prop, WikidataProperty)
        assert prop.pid == "P31"
        assert prop.label is not None
    
    @pytest.mark.slow
    def test_get_property_multiple(self, wrapper):
        """Test getting multiple properties."""
        props = wrapper._get_property(["P31", "P27"])
        print(f"\n✓ test_get_property_multiple")
        print(f"  PIDs: ['P31', 'P27']")
        print(f"  Results: {len(props)} properties")
        for p in props:
            if p:
                print(f"    - {p.label} ({p.pid}): {p.description}")
        assert isinstance(props, list)
        assert len(props) == 2
        assert all(isinstance(p, WikidataProperty) for p in props if p is not None)
    
    @pytest.mark.slow
    def test_deduplicate_triples(self, wrapper):
        """Test triple deduplication."""
        subject = WikidataEntity(qid="Q42", label="Test")
        relation = WikidataProperty(pid="P31", label="instance of")
        object_entity = WikidataEntity(qid="Q5", label="human")
        
        triple = WikiTriple(subject=subject, relation=relation, object=object_entity)
        triples = [triple, triple, triple]  # Duplicates
        
        unique = wrapper._deduplicate_triples(triples)
        assert len(unique) == 1
    
    @pytest.mark.slow
    def test_get_k_hop_outgoing(self, wrapper):
        """Test k-hop outgoing traversal."""
        triples = wrapper._get_k_hop_outgoing("Q142", k=3)  # France
        print(f"\n✓ test_get_k_hop_outgoing")
        print(f"  Entity: Q142 (France), k=1")
        print(f"  Found {len(triples)} triples")
        for t in triples[:5]:
            obj_str = t.object.label if isinstance(t.object, WikidataEntity) else str(t.object)
            print(f"    {t.subject.label} --[{t.relation.label}]--> {obj_str}")
        assert isinstance(triples, list)
        assert len(triples) > 0
        assert all(isinstance(t, WikiTriple) for t in triples)
    
    @pytest.mark.slow
    def test_get_k_hop_bidirectional(self, wrapper):
        """Test k-hop bidirectional traversal."""
        triples = wrapper._get_k_hop_bidirectional("Q142", k=2)  # France
        print(f"\n✓ test_get_k_hop_bidirectional")
        print(f"  Entity: Q142 (France), k=2, bidirectional=True")
        print(f"  Found {len(triples)} triples")
        if triples:
            for t in triples[:5]:
                obj_str = t.object.label if isinstance(t.object, WikidataEntity) else str(t.object)
                print(f"    {t.subject.label} --[{t.relation.label}]--> {obj_str}")
        assert isinstance(triples, list)
        # May be empty or have results
        if triples:
            assert all(isinstance(t, WikiTriple) for t in triples)


# ============================================================================
# Integration Tests - Tools
# ============================================================================

class TestWikidataToolsRefactored:
    """Integration tests for refactored Wikidata tools."""
    
    @pytest.fixture
    def entity_tool(self):
        """Create WikidataEntityRetrievalTool."""
        return WikidataEntityRetrievalTool()
    
    @pytest.fixture
    def property_tool(self):
        """Create WikidataPropertyRetrievalTool."""
        return WikidataPropertyRetrievalTool()
    
    @pytest.fixture
    def khop_tool(self):
        """Create WikidataKHopTriplesRetrievalTool."""
        return WikidataKHopTriplesRetrievalTool()
    
    @pytest.mark.slow
    def test_entity_tool_single_query(self, entity_tool):
        """Test entity tool with single query."""
        results = asyncio.run(
            entity_tool.ainvoke({"query": "Albert Einstein", "num_entities": 3})
        )
        print(f"\n✓ test_entity_tool_single_query")
        print(f"  Query: 'Albert Einstein', num_entities=3")
        print(f"  Found {len(results)} entities:")
        for e in results:
            print(f"    - {e.label} ({e.qid}): {e.description}")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(e, WikidataEntity) for e in results)
    
    @pytest.mark.slow
    def test_entity_tool_multiple_queries(self, entity_tool):
        """Test entity tool with multiple queries."""
        results = asyncio.run(
            entity_tool.ainvoke({"query": ["Paris", "Tokyo"], "num_entities": 2})
        )
        print(f"\n✓ test_entity_tool_multiple_queries")
        print(f"  Queries: ['Paris', 'Tokyo'], num_entities=2")
        print(f"  Found {len(results)} query results:")
        for i, r in enumerate(results):
            print(f"    Query {i+1}: {len(r)} entities")
            for e in r[:2]:
                print(f"      - {e.label} ({e.qid})")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
    
    @pytest.mark.slow
    def test_entity_tool_with_qids(self, entity_tool):
        """Test entity tool with QIDs."""
        results = asyncio.run(
            entity_tool.ainvoke({"query": "Q42", "is_qids": True, "num_entities": 1})
        )
        print(f"\n✓ test_entity_tool_with_qids")
        print(f"  Query: 'Q42', is_qids=True")
        print(f"  Found {len(results)} entities:")
        for e in results:
            print(f"    - {e.label} ({e.qid}): {e.description}")
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0].qid == "Q42"
    
    @pytest.mark.slow
    def test_entity_tool_with_qids_multiple(self, entity_tool):
        """Test entity tool with multiple QIDs."""
        results = asyncio.run(
            entity_tool.ainvoke({"query": ["Q42", "Q142"], "is_qids": True, "num_entities": 1})
        )
        print(f"\n✓ test_entity_tool_with_qids_multiple")
        print(f"  Queries: ['Q42', 'Q142'], is_qids=True, num_entities=1")
        print(f"  Found {len(results)} query results:")
        for i, r in enumerate(results):
            print(f"    Query {i+1}: {len(r)} entities")
            for e in r[:2]:
                print(f"      - {e.label} ({e.qid})")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.slow
    def test_property_tool_single_query(self, property_tool):
        """Test property tool with single query."""
        results = asyncio.run(
            property_tool.ainvoke({"query": "capital", "top_k_results": 3})
        )
        print(f"\n✓ test_property_tool_single_query")
        print(f"  Query: 'capital', top_k_results=3")
        print(f"  Found {len(results)} properties:")
        for p in results:
            print(f"    - {p.label} ({p.pid}): {p.description}")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(p, WikidataProperty) for p in results)
    
    def test_property_tool_with_multiple_queries(self, property_tool):
        """Test property tool with multiple queries."""
        results = asyncio.run(
            property_tool.ainvoke({"query": ["capital", "population"], "top_k_results": 3})
        )
        print(f"\n✓ test_property_tool_with_multiple_queries")
        print(f"  Queries: ['capital', 'population'], top_k_results=3")
        print(f"  Found {len(results)} query results:")
        for i, r in enumerate(results):
            print(f"    Query {i+1}: {len(r)} properties")
            for p in r[:2]:
                print(f"      - {p.label} ({p.pid}): {p.description}")
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
    
    @pytest.mark.slow
    def test_khop_tool_with_enrichment(self, khop_tool: WikidataKHopTriplesRetrievalTool):
        """Test k-hop tool with enrichment (tests EnrichmentCollector integration)."""
        triples = asyncio.run(
            khop_tool.ainvoke({
                "query": "France",
                "k": 1,
                "num_entities": 1,
                "enrich": True,
                "get_details": True
            })
        )
        print(f"\n✓ test_khop_tool_with_enrichment")
        print(f"  Query: 'France', k=1, enrich=True, get_details=True")
        print(f"  Found {len(triples)} triples (enriched):")
        for t in triples[:5]:
            obj_str = t.object.label if isinstance(t.object, WikidataEntity) else str(t.object)
            print(f"    {t.subject.label} --[{t.relation.label}]--> {obj_str}")
        assert isinstance(triples, list)
        if triples:
            # Check that entities are enriched
            first_triple = triples[0]
            assert first_triple.subject.label is not None
            assert first_triple.relation.label is not None
    
    @pytest.mark.slow
    def test_khop_tool_bidirectional(self, khop_tool):
        """Test k-hop tool with bidirectional traversal."""
        triples = khop_tool.invoke({
                "query": "Q142",  # France
                "is_qids": True,
                "k": 1,
                "bidirectional": True,
                "num_entities": 1
            })
        print(f"\n✓ test_khop_tool_bidirectional")
        print(f"  Query: 'Q142', k=1, bidirectional=True")
        print(f"  Found {len(triples)} triples:")
        for t in triples[:5]:
            obj_str = t.object.label if isinstance(t.object, WikidataEntity) else str(t.object)
            print(f"    {t.subject.label} --[{t.relation.label}]--> {obj_str}")
        assert isinstance(triples, list)
        # May be empty or have results
    
    @pytest.mark.slow
    def test_property_inference_and_triple_search(self, property_tool, khop_tool):
        """Test inferring property from text (not in default) and using it for triple search."""
        # Step 1: Search for a property that's likely NOT in DEFAULT_PROPERTIES
        # Using "spouse" (P26) or "child" (P40) which are common but may not be in defaults
        property_query = "spouse"
        
        property_results = asyncio.run(
            property_tool.ainvoke({"query": property_query, "top_k_results": 3})
        )
        
        assert isinstance(property_results, list)
        assert len(property_results) > 0
        assert all(isinstance(p, WikidataProperty) for p in property_results)
        
        # Get the first property (should be P26 for spouse)
        inferred_property = property_results[0]
        assert inferred_property.pid is not None
        assert inferred_property.pid.startswith("P")
        
        print(f"✓ Property Inference")
        print(f"  Query: '{property_query}'")
        print(f"  Inferred PID: {inferred_property.pid}")
        print(f"  Label: {inferred_property.label}")
        
        # Step 2: Verify it's not in default properties (or at least test the workflow)
        # Note: This property might actually be in defaults, but the workflow should work regardless
        entity_tool = WikidataEntityRetrievalTool()
        
        # Step 3: Find an entity to search triples for
        entity_results = asyncio.run(
            entity_tool.ainvoke({"query": "Barack Obama", "num_entities": 1})
        )
        
        assert len(entity_results) > 0
        entity = entity_results[0]
        
        # Step 4: Use the inferred property to filter triples
        triples = asyncio.run(
            khop_tool.ainvoke({
                "query": entity.qid,
                "is_qids": True,
                "k": 1,
                "num_entities": 1,
                "prop": inferred_property.pid  # Use inferred property
            })
        )
        
        assert isinstance(triples, list)
        print(f"✓ Triple Search with Inferred Property")
        print(f"  Entity: {entity.label} ({entity.qid})")
        print(f"  Property: {inferred_property.label} ({inferred_property.pid})")
        print(f"  Found {len(triples)} triples filtered by property")
        
        # Verify that all triples use the inferred property
        if triples:
            for triple in triples[:5]:
                assert triple.relation.pid == inferred_property.pid
                print(f"    - {triple.subject.label} --[{triple.relation.label}]--> "
                      f"{triple.object.label if hasattr(triple.object, 'label') else triple.object}")
    
    @pytest.mark.slow
    def test_property_not_in_defaults_workflow(self, property_tool, khop_tool, entity_tool):
        """Test complete workflow: infer property not in defaults -> search triples."""
        # Properties that are commonly NOT in DEFAULT_PROPERTIES
        # Try multiple to ensure at least one works
        property_queries = ["sibling", "parent", "award", "member of"]
        
        inferred_property = None
        property_query_used = None
        
        for prop_query in property_queries:
            property_results = asyncio.run(
                property_tool.ainvoke({"query": prop_query, "top_k_results": 1})
            )
            
            if property_results and len(property_results) > 0:
                prop = property_results[0]
                # Check if property is NOT in default properties
                wrapper = khop_tool.wikidata_wrapper
                if prop.pid not in wrapper.wikidata_props:
                    inferred_property = prop
                    property_query_used = prop_query
                    break
        
        # If we found a property not in defaults, test the workflow
        if inferred_property:
            print(f"✓ Found property not in defaults: {inferred_property.pid} ({inferred_property.label})")
            
            # Find an entity
            entity_results = asyncio.run(
                entity_tool.ainvoke({"query": "Albert Einstein", "num_entities": 1})
            )
            
            if entity_results and len(entity_results) > 0:
                entity = entity_results[0]
                
                # Search triples with the inferred property
                triples = asyncio.run(
                    khop_tool.ainvoke({
                        "query": entity.qid,
                        "is_qids": True,
                        "k": 1,
                        "num_entities": 1,
                        "prop": inferred_property.pid
                    })
                )
                
                assert isinstance(triples, list)
                print(f"✓ Complete Workflow Test")
                print(f"  Property query: '{property_query_used}'")
                print(f"  Inferred PID: {inferred_property.pid}")
                print(f"  Entity: {entity.label} ({entity.qid})")
                print(f"  Triples found: {len(triples)}")
                
                # Verify property filtering worked
                if triples:
                    for triple in triples[:3]:
                        assert triple.relation.pid == inferred_property.pid
        else:
            # If all properties are in defaults, still test the workflow
            print(f"✓ All tested properties are in defaults, testing workflow anyway")
            property_results = asyncio.run(
                property_tool.ainvoke({"query": "spouse", "top_k_results": 1})
            )
            if property_results:
                inferred_property = property_results[0]
                entity_results = asyncio.run(
                    entity_tool.ainvoke({"query": "Barack Obama", "num_entities": 1})
                )
                if entity_results:
                    entity = entity_results[0]
                    triples = asyncio.run(
                        khop_tool.ainvoke({
                            "query": entity.qid,
                            "is_qids": True,
                            "k": 1,
                            "num_entities": 1,
                            "prop": inferred_property.pid
                        })
                    )
                    assert isinstance(triples, list)
    
    @pytest.mark.slow
    def test_property_inference_multiple_candidates(self, property_tool, khop_tool):
        """Test property inference with multiple candidates and selecting the right one."""
        # Query that might return multiple property candidates
        property_query = "birth"
        
        property_results = asyncio.run(
            property_tool.ainvoke({"query": property_query, "top_k_results": 5})
        )
        
        assert isinstance(property_results, list)
        assert len(property_results) > 0
        
        # Should have multiple candidates (e.g., "date of birth", "place of birth")
        print(f"✓ Property Inference with Multiple Candidates")
        print(f"  Query: '{property_query}'")
        print(f"  Found {len(property_results)} candidates:")
        for i, prop in enumerate(property_results[:3]):
            print(f"    {i+1}. {prop.label} ({prop.pid})")
        
        # Use the first one (most relevant) for triple search
        selected_property = property_results[0]
        
        entity_tool = WikidataEntityRetrievalTool()
        entity_results = asyncio.run(
            entity_tool.ainvoke({"query": "Marie Curie", "num_entities": 1})
        )
        
        if entity_results and len(entity_results) > 0:
            entity = entity_results[0]
            
            # Search triples with selected property
            triples = asyncio.run(
                khop_tool.ainvoke({
                    "query": entity.qid,
                    "is_qids": True,
                    "k": 1,
                    "num_entities": 1,
                    "prop": selected_property.pid
                })
            )
            
            assert isinstance(triples, list)
            print(f"  Selected: {selected_property.label} ({selected_property.pid})")
            print(f"  Entity: {entity.label} ({entity.qid})")
            print(f"  Triples found: {len(triples)}")
            
            # Verify all triples use the selected property
            if triples:
                for triple in triples[:3]:
                    assert triple.relation.pid == selected_property.pid
                    obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                    print(f"    - {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
    
    @pytest.mark.slow
    def test_property_inference_and_bidirectional_triples(self, property_tool, khop_tool):
        """Test property inference and bidirectional triple search."""
        # Infer a property
        property_results = asyncio.run(
            property_tool.ainvoke({"query": "occupation", "top_k_results": 1})
        )
        
        if property_results and len(property_results) > 0:
            inferred_property = property_results[0]
            
            entity_tool = WikidataEntityRetrievalTool()
            entity_results = asyncio.run(
                entity_tool.ainvoke({"query": "Leonardo da Vinci", "num_entities": 1})
            )
            
            if entity_results and len(entity_results) > 0:
                entity = entity_results[0]
                
                # Search bidirectional triples with inferred property
                triples = asyncio.run(
                    khop_tool.ainvoke({
                        "query": entity.qid,
                        "is_qids": True,
                        "k": 1,
                        "num_entities": 1,
                        "bidirectional": True,
                        "prop": inferred_property.pid
                    })
                )
                
                assert isinstance(triples, list)
                print(f"✓ Property Inference + Bidirectional Triples")
                print(f"  Property: {inferred_property.label} ({inferred_property.pid})")
                print(f"  Entity: {entity.label} ({entity.qid})")
                print(f"  Bidirectional triples found: {len(triples)}")
                
                if triples:
                    # Verify property filtering
                    for triple in triples[:3]:
                        assert triple.relation.pid == inferred_property.pid
    
    @pytest.mark.slow
    def test_property_not_in_defaults_explicit(self, property_tool, khop_tool, entity_tool):
        """Test explicitly using a property that's NOT in DEFAULT_PROPERTIES."""
        # Get the default properties list
        from wemg.agents.tools.wikidata import DEFAULT_PROPERTIES
        
        # Try to find a property that's definitely not in defaults
        # Common properties that might not be in defaults: P26 (spouse), P40 (child), 
        # P22 (father), P25 (mother), P451 (unmarried partner), P1038 (relative)
        property_queries = [
            ("spouse", "P26"),
            ("child", "P40"), 
            ("father", "P22"),
            ("mother", "P25"),
            ("sibling", "P3373"),
            ("award", "P166"),
        ]
        
        inferred_property = None
        property_query_used = None
        is_in_defaults = False
        
        for prop_query, expected_pid in property_queries:
            property_results = asyncio.run(
                property_tool.ainvoke({"query": prop_query, "top_k_results": 3})
            )
            
            if property_results and len(property_results) > 0:
                # Find the property matching expected PID or use first result
                for prop in property_results:
                    if prop.pid == expected_pid or prop.pid not in DEFAULT_PROPERTIES:
                        inferred_property = prop
                        property_query_used = prop_query
                        is_in_defaults = prop.pid in DEFAULT_PROPERTIES
                        break
                
                if inferred_property:
                    break
        
        if inferred_property:
            print(f"✓ Property Not in Defaults Test")
            print(f"  Property query: '{property_query_used}'")
            print(f"  Inferred PID: {inferred_property.pid}")
            print(f"  Label: {inferred_property.label}")
            print(f"  In DEFAULT_PROPERTIES: {is_in_defaults}")
            
            # Find an entity
            entity_results = asyncio.run(
                entity_tool.ainvoke({"query": "Marie Curie", "num_entities": 1})
            )
            
            if entity_results and len(entity_results) > 0:
                entity = entity_results[0]
                
                # Test that we can use this property even if not in defaults
                # The API wrapper should handle properties passed via 'prop' parameter
                triples = asyncio.run(
                    khop_tool.ainvoke({
                        "query": entity.qid,
                        "is_qids": True,
                        "k": 1,
                        "num_entities": 1,
                        "prop": inferred_property.pid
                    })
                )
                
                assert isinstance(triples, list)
                print(f"  Entity: {entity.label} ({entity.qid})")
                print(f"  Triples found: {len(triples)}")
                
                # Verify all triples use the inferred property
                if triples:
                    for triple in triples[:3]:
                        assert triple.relation.pid == inferred_property.pid
                        obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                        print(f"    - {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
                    print(f"  ✓ Successfully filtered triples using property not in defaults")
                else:
                    print(f"  Note: No triples found (entity may not have this property)")
    
    @pytest.mark.slow
    def test_complete_property_inference_workflow(self, property_tool, entity_tool, khop_tool):
        """Test complete workflow: text query -> property inference -> entity search -> filtered triples."""
        # Step 1: Infer property from text query
        property_text = "educated at"
        property_results = asyncio.run(
            property_tool.ainvoke({"query": property_text, "top_k_results": 3})
        )
        
        assert property_results and len(property_results) > 0
        inferred_property = property_results[0]
        assert inferred_property.pid is not None
        assert inferred_property.pid.startswith("P")
        
        print(f"✓ Complete Property Inference Workflow")
        print(f"  Step 1: Property inference")
        print(f"    Text query: '{property_text}'")
        print(f"    Inferred: {inferred_property.label} ({inferred_property.pid})")
        
        # Step 2: Add property to wrapper's wikidata_props if not already there
        # This simulates the workflow where we dynamically add inferred properties
        wrapper = khop_tool.wikidata_wrapper
        if inferred_property.pid not in wrapper.wikidata_props:
            wrapper.wikidata_props.append(inferred_property.pid)
            # Also add to cache
            wrapper.wikidata_props_with_labels[inferred_property.pid] = {
                "label": inferred_property.label,
                "description": inferred_property.description
            }
            print(f"    Added {inferred_property.pid} to wrapper's properties list")
        
        # Step 3: Find entity
        entity_text = "Albert Einstein"
        entity_results = asyncio.run(
            entity_tool.ainvoke({"query": entity_text, "num_entities": 1})
        )
        
        assert entity_results and len(entity_results) > 0
        entity = entity_results[0]
        
        print(f"  Step 2: Entity search")
        print(f"    Text query: '{entity_text}'")
        print(f"    Found: {entity.label} ({entity.qid})")
        
        # Step 4: Search triples filtered by inferred property
        triples = asyncio.run(
            khop_tool.ainvoke({
                "query": entity.qid,
                "is_qids": True,
                "k": 1,
                "num_entities": 1,
                "prop": inferred_property.pid
            })
        )
        
        assert isinstance(triples, list)
        print(f"  Step 3: Filtered triple search")
        print(f"    Property filter: {inferred_property.pid}")
        print(f"    Triples found: {len(triples)}")
        
        # Step 5: Verify results
        if triples:
            print(f"  Step 4: Verification")
            for i, triple in enumerate(triples[:5]):
                assert triple.relation.pid == inferred_property.pid
                obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                print(f"    {i+1}. {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
            print(f"  ✓ All triples correctly filtered by inferred property")
        else:
            print(f"  Note: No triples found (entity may not have this property)")
    
    @pytest.mark.slow
    def test_property_inference_dynamic_addition(self, property_tool, entity_tool, khop_tool):
        """Test inferring property and dynamically adding it to wrapper for triple search."""
        # Step 1: Infer a property that's likely not in defaults
        property_queries = ["spouse", "child", "award received", "member of political party"]
        
        inferred_property = None
        for prop_query in property_queries:
            property_results = asyncio.run(
                property_tool.ainvoke({"query": prop_query, "top_k_results": 1})
            )
            
            if property_results and len(property_results) > 0:
                inferred_property = property_results[0]
                break
        
        assert inferred_property is not None
        print(f"✓ Property Inference with Dynamic Addition")
        print(f"  Inferred property: {inferred_property.label} ({inferred_property.pid})")
        
        # Step 2: Check if in defaults and add if needed
        wrapper = khop_tool.wikidata_wrapper
        was_in_defaults = inferred_property.pid in wrapper.wikidata_props
        
        if not was_in_defaults:
            # Dynamically add the property
            wrapper.wikidata_props.append(inferred_property.pid)
            wrapper.wikidata_props_with_labels[inferred_property.pid] = {
                "label": inferred_property.label,
                "description": inferred_property.description
            }
            print(f"  Property NOT in defaults - added dynamically")
        else:
            print(f"  Property already in defaults")
        
        # Step 3: Find entity and search triples
        entity_results = asyncio.run(
            entity_tool.ainvoke({"query": "Marie Curie", "num_entities": 1})
        )
        
        if entity_results and len(entity_results) > 0:
            entity = entity_results[0]
            
            # Now the property should work
            triples = asyncio.run(
                khop_tool.ainvoke({
                    "query": entity.qid,
                    "is_qids": True,
                    "k": 1,
                    "num_entities": 1,
                    "prop": inferred_property.pid
                })
            )
            
            assert isinstance(triples, list)
            print(f"  Entity: {entity.label} ({entity.qid})")
            print(f"  Triples found: {len(triples)}")
            
            if triples:
                # Verify all use the inferred property
                for triple in triples[:3]:
                    assert triple.relation.pid == inferred_property.pid
                    obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                    print(f"    - {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
                print(f"  ✓ Successfully used dynamically added property for filtering")


# ============================================================================
# Integration Tests - Path Finding Tool
# ============================================================================

class TestWikidataPathFindingTool:
    """Integration tests for WikidataPathFindingTool."""
    
    @pytest.fixture
    def path_finder(self):
        """Create a WikidataPathFindingTool."""
        return WikidataPathFindingTool()
    
    @pytest.fixture
    def entity_tool(self):
        """Create a WikidataEntityRetrievalTool for finding entities."""
        return WikidataEntityRetrievalTool()
    
    @pytest.mark.slow
    def test_find_path_same_entity(self, path_finder):
        """Test finding path from an entity to itself."""
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
        print(f"  Path: {path_result}")
    
    @pytest.mark.slow
    def test_find_path_between_related_entities(self, path_finder):
        """Test finding path between two related entities."""
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
            
            print(f"✓ Path Finding Between Related Entities")
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
            print(f"✓ Path Finding Between Related Entities")
            print(f"  Source: {source_qid} (France)")
            print(f"  Target: {target_qid} (Germany)")
            print(f"  No path found within max_hops=3")
    
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
            print(f"    Path: {path_result_3}")
    
    @pytest.mark.slow
    def test_find_path_well_known_connection(self, path_finder):
        """Test finding a path between well-known connected entities."""
        # Q76 (Barack Obama) and Q30 (United States) - should be connected
        source_qid = "Q76"   # Barack Obama
        target_qid = "Q30"   # United States
        
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
    def test_find_path_with_entity_search(self, path_finder, entity_tool):
        """Test finding path using entity search (text to QID conversion)."""
        # Step 1: Find entities by text
        source_results = asyncio.run(
            entity_tool.ainvoke({"query": "Paris", "num_entities": 1})
        )
        target_results = asyncio.run(
            entity_tool.ainvoke({"query": "London", "num_entities": 1})
        )
        
        if source_results and len(source_results) > 0 and target_results and len(target_results) > 0:
            source_entity = source_results[0]
            target_entity = target_results[0]
            
            # Step 2: Find path between them
            path_result = asyncio.run(
                path_finder.ainvoke({
                    "source_qid": source_entity.qid,
                    "target_qid": target_entity.qid,
                    "max_hops": 3
                })
            )
            
            assert path_result is not None
            assert isinstance(path_result, WikidataPathBetweenEntities)
            
            print(f"✓ Path Finding with Entity Search")
            print(f"  Source: {path_result.source.label} ({source_entity.qid})")
            print(f"  Target: {path_result.target.label} ({target_entity.qid})")
            print(f"  Path length: {path_result.path_length}")
            if path_result.path:
                print(f"  Path found with {len(path_result.path)} hops")
    
    @pytest.mark.slow
    def test_find_path_direct_connection(self, path_finder):
        """Test finding path between directly connected entities."""
        # Q142 (France) and Q142's capital (should be directly connected)
        # Let's use Q142 and Q90 (Paris, which is capital of France)
        source_qid = "Q142"  # France
        target_qid = "Q90"    # Paris
        
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 2
            })
        )
        
        if path_result:
            print(f"✓ Path Finding Direct Connection")
            print(f"  Source: {path_result.source.label} ({source_qid})")
            print(f"  Target: {path_result.target.label} ({target_qid})")
            print(f"  Path length: {path_result.path_length}")
            if path_result.path:
                # Should be short path (1-2 hops)
                assert path_result.path_length <= 2
                print(f"  Direct connection found with {len(path_result.path)} hop(s)")
                for i, triple in enumerate(path_result.path):
                    obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                    print(f"    {i+1}. {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")
    
    @pytest.mark.slow
    def test_find_path_async_version(self, path_finder):
        """Test async version of path finding."""
        source_qid = "Q142"  # France
        target_qid = "Q183"  # Germany
        
        # Test async version
        path_result = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        # Should work the same as sync version
        if path_result:
            assert isinstance(path_result, WikidataPathBetweenEntities)
            print(f"✓ Path Finding Async Version")
            print(f"  Source: {path_result.source.label} ({source_qid})")
            print(f"  Target: {path_result.target.label} ({target_qid})")
            print(f"  Path length: {path_result.path_length}")
    
    @pytest.mark.slow
    def test_find_path_max_hops_limit(self, path_finder):
        """Test that max_hops limit is respected."""
        # Use entities that might be far apart
        source_qid = "Q30"   # United States
        target_qid = "Q142"  # France
        
        # Try with very small max_hops
        path_result_1 = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 1
            })
        )
        
        # Try with larger max_hops
        path_result_3 = asyncio.run(
            path_finder.ainvoke({
                "source_qid": source_qid,
                "target_qid": target_qid,
                "max_hops": 3
            })
        )
        
        print(f"✓ Path Finding Max Hops Limit")
        print(f"  Source: {source_qid} (United States)")
        print(f"  Target: {target_qid} (France)")
        print(f"  Max hops=1: {'Path found' if path_result_1 and path_result_1.path else 'No path'}")
        print(f"  Max hops=3: {'Path found' if path_result_3 and path_result_3.path else 'No path'}")
        
        # If path found with max_hops=1, verify it's actually 1 hop
        if path_result_1 and path_result_1.path:
            assert path_result_1.path_length <= 1
            assert len(path_result_1.path) <= 1
        
        # If path found with max_hops=3, verify it's within limit
        if path_result_3 and path_result_3.path:
            assert path_result_3.path_length <= 3
            assert len(path_result_3.path) <= 3
    
    @pytest.mark.slow
    def test_find_path_person_to_location(self, path_finder, entity_tool):
        """Test finding path from a person to a location."""
        # Find a person and a location
        person_results = asyncio.run(
            entity_tool.ainvoke({"query": "Albert Einstein", "num_entities": 1})
        )
        location_results = asyncio.run(
            entity_tool.ainvoke({"query": "Princeton", "num_entities": 1})
        )
        
        if person_results and len(person_results) > 0 and location_results and len(location_results) > 0:
            person = person_results[0]
            location = location_results[0]
            
            path_result = asyncio.run(
                path_finder.ainvoke({
                    "source_qid": person.qid,
                    "target_qid": location.qid,
                    "max_hops": 3
                })
            )
            
            if path_result:
                print(f"✓ Path Finding Person to Location")
                print(f"  Person: {path_result.source.label} ({person.qid})")
                print(f"  Location: {path_result.target.label} ({location.qid})")
                print(f"  Path length: {path_result.path_length}")
                if path_result.path:
                    print(f"  Path found:")
                    for i, triple in enumerate(path_result.path[:3]):
                        obj_label = triple.object.label if hasattr(triple.object, 'label') else str(triple.object)
                        print(f"    {i+1}. {triple.subject.label} --[{triple.relation.label}]--> {obj_label}")


# ============================================================================
# Integration Tests - End-to-End
# ============================================================================

class TestWikidataEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_full_workflow_with_enrichment(self):
        """Test full workflow: search -> retrieve -> k-hop -> enrichment."""
        entity_tool = WikidataEntityRetrievalTool()
        khop_tool = WikidataKHopTriplesRetrievalTool()
        
        # Step 1: Find entity
        entities = asyncio.run(
            entity_tool.ainvoke({"query": "France", "num_entities": 1})
        )
        assert len(entities) > 0
        entity = entities[0]
        
        print(f"\n✓ test_full_workflow_with_enrichment")
        print(f"  Step 1 - Entity found: {entity.label} ({entity.qid})")
        
        # Step 2: Get k-hop triples with enrichment
        triples = asyncio.run(
            khop_tool.ainvoke({
                "query": entity.qid,
                "is_qids": True,
                "k": 1,
                "num_entities": 1,
                "enrich": True,
                "get_details": True
            })
        )
        
        print(f"  Step 2 - Found {len(triples)} triples (enriched):")
        for t in triples[:5]:
            obj_str = t.object.label if isinstance(t.object, WikidataEntity) else str(t.object)
            print(f"    {t.subject.label} --[{t.relation.label}]--> {obj_str}")
        
        assert isinstance(triples, list)
        if triples:
            # Verify enrichment worked
            assert all(t.subject.label for t in triples)
            assert all(t.relation.label for t in triples)
    
    @pytest.mark.slow
    def test_batch_processing(self):
        """Test batch processing with multiple entities."""
        entity_tool = WikidataEntityRetrievalTool()
        
        # Batch retrieval
        results = asyncio.run(
            entity_tool.ainvoke({
                "query": ["Paris", "Tokyo", "New York"],
                "num_entities": 2
            })
        )
        
        print(f"\n✓ test_batch_processing")
        print(f"  Queries: ['Paris', 'Tokyo', 'New York'], num_entities=2")
        print(f"  Found {len(results)} query results:")
        for i, r in enumerate(results):
            print(f"    Query {i+1}: {len(r)} entities")
            for e in r:
                print(f"      - {e.label} ({e.qid}): {e.description}")
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) > 0 for r in results)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run unit tests: pytest test_wikidata_refactored.py -v -k "TestWikidataUtils or TestWikidataModels or TestEnrichmentCollector"
    # Run integration tests: pytest test_wikidata_refactored.py -v -m slow
    # Run all: pytest test_wikidata_refactored.py -v
    pytest.main([__file__, "-v", "-s", "--tb=short"])

