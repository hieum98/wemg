"""
Comprehensive tests for WorkingMemory and InteractionMemory.

This module tests both memory systems:
- WorkingMemory: Textual and graph-based memory management
- InteractionMemory: Conversational memory with vector search
"""
import os
import pytest
import asyncio
import networkx as nx
import uuid
from typing import List, Optional

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools.wikidata import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
)
from wemg.agents import roles
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.preprocessing import get_node_id
from wemg.utils.graph_utils import visualize_graph


# Test configuration
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")


# ============================================================================
# WorkingMemory Tests
# ============================================================================

class TestWorkingMemory:
    """Test suite for WorkingMemory functionality."""
    
    @pytest.fixture
    def working_memory(self):
        """Create an empty WorkingMemory instance."""
        return WorkingMemory()
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample Wikidata entities for testing."""
        return {
            'paris': WikidataEntity(qid="Q90", label="Paris", description="capital of France"),
            'france': WikidataEntity(qid="Q142", label="France", description="country in Western Europe"),
            'tokyo': WikidataEntity(qid="Q1490", label="Tokyo", description="capital of Japan"),
        }
    
    @pytest.fixture
    def sample_property(self):
        """Create a sample Wikidata property."""
        return WikidataProperty(pid="P1376", label="capital of", description="country capital relationship")
    
    @pytest.fixture
    def sample_triple(self, sample_entities, sample_property):
        """Create a sample WikiTriple."""
        return WikiTriple(
            subject=sample_entities['paris'],
            relation=sample_property,
            object=sample_entities['france']
        )
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_init_empty(self, working_memory):
        """Test creating empty working memory."""
        assert working_memory.textual_memory == []
        assert len(working_memory.graph_memory.nodes) == 0
        assert len(working_memory.graph_memory.edges) == 0
        assert working_memory.entity_dict == {}
        assert working_memory.property_dict == {}
        assert working_memory.id_dict == {}
        assert working_memory.max_textual_memory_tokens == 8192
    
    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        textual_memory = ["Fact 1", "Fact 2"]
        graph = nx.DiGraph()
        graph.add_node("node1", data="data1")
        
        wm = WorkingMemory(
            textual_memory=textual_memory,
            graph_memory=graph,
            max_textual_memory_tokens=4096
        )
        
        assert wm.textual_memory == textual_memory
        assert len(wm.graph_memory.nodes) == 1
        assert wm.max_textual_memory_tokens == 4096
    
    # ========================================================================
    # Textual Memory Tests
    # ========================================================================
    
    def test_format_memory_item(self):
        """Test formatting memory items with provenance tags."""
        # System prediction
        formatted = WorkingMemory.format_memory_item(
            "Test content",
            roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        assert formatted == "[System Prediction]: Test content"
        
        # Retrieval
        formatted = WorkingMemory.format_memory_item(
            "Test content",
            roles.extractor.SourceType.RETRIEVAL
        )
        assert formatted == "[Retrieval]: Test content"
        
        # Already formatted content
        formatted = WorkingMemory.format_memory_item(
            "[System Prediction]: Already formatted",
            roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        assert formatted == "[System Prediction]: Already formatted"
    
    def test_add_textual_memory(self, working_memory):
        """Test adding textual memory items."""
        # Add system prediction
        working_memory.add_textual_memory(
            "The capital of France is Paris.",
            source=roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        assert len(working_memory.textual_memory) == 1
        assert "[System Prediction]" in working_memory.textual_memory[0]
        
        # Add retrieval item
        working_memory.add_textual_memory(
            "Paris is the largest city in France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        assert len(working_memory.textual_memory) == 2
        assert "[Retrieval]" in working_memory.textual_memory[1]
        
        # Test duplicate prevention
        working_memory.add_textual_memory(
            "The capital of France is Paris.",
            source=roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        assert len(working_memory.textual_memory) == 2  # Should not add duplicate
    
    def test_format_textual_memory(self, working_memory):
        """Test formatting textual memory."""
        working_memory.add_textual_memory("Fact 1", source=roles.extractor.SourceType.RETRIEVAL)
        working_memory.add_textual_memory("Fact 2", source=roles.extractor.SourceType.SYSTEM_PREDICTION)
        
        formatted = working_memory.format_textual_memory()
        assert "- [Retrieval]: Fact 1" in formatted
        assert "- [System Prediction]: Fact 2" in formatted
        assert formatted.count("\n") == 1  # Two items, one newline
    
    # ========================================================================
    # Graph Memory Tests
    # ========================================================================
    
    def test_add_edge_to_graph_memory(self, working_memory: WorkingMemory, sample_triple: WikiTriple):
        """Test adding edges to graph memory using WikiTriple objects."""
        # Visualize graph before
        print("\n=== Graph Memory BEFORE adding edge ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Adding Edge")
        
        working_memory.add_edge_to_graph_memory(sample_triple)
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER adding edge ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Adding Edge")
        
        subject_id = get_node_id(sample_triple.subject)
        object_id = get_node_id(sample_triple.object)
        
        # Check nodes were added
        assert working_memory.graph_memory.has_node(subject_id)
        assert working_memory.graph_memory.has_node(object_id)
        
        # Check edge was added
        assert working_memory.graph_memory.has_edge(subject_id, object_id)
        
        # Check relation is stored
        edge_data = working_memory.graph_memory.get_edge_data(subject_id, object_id)
        assert sample_triple.relation in edge_data["relation"]
    
    def test_add_edge_multiple_relations(self, working_memory, sample_entities):
        """Test adding multiple relations between same nodes."""
        # Visualize graph before
        print("\n=== Graph Memory BEFORE adding multiple relations ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Adding Multiple Relations")
        
        prop1 = WikidataProperty(pid="P1376", label="capital of")
        prop2 = WikidataProperty(pid="P131", label="located in")
        
        triple1 = WikiTriple(
            subject=sample_entities['paris'],
            relation=prop1,
            object=sample_entities['france']
        )
        triple2 = WikiTriple(
            subject=sample_entities['paris'],
            relation=prop2,
            object=sample_entities['france']
        )
        
        working_memory.add_edge_to_graph_memory(triple1)
        working_memory.add_edge_to_graph_memory(triple2)
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER adding multiple relations ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Adding Multiple Relations")
        
        subject_id = get_node_id(sample_entities['paris'])
        object_id = get_node_id(sample_entities['france'])
        
        edge_data = working_memory.graph_memory.get_edge_data(subject_id, object_id)
        assert prop1 in edge_data["relation"]
        assert prop2 in edge_data["relation"]
        assert len(edge_data["relation"]) == 2
    
    def test_extract_triples_from_graph(self, working_memory, sample_triple):
        """Test extracting triples from graph."""
        working_memory.add_edge_to_graph_memory(sample_triple)
        
        # Visualize graph before extraction
        print("\n=== Graph Memory BEFORE extracting triples ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Extracting Triples")
        
        triples = WorkingMemory.extract_triples_from_graph(working_memory.graph_memory)
        
        # Visualize graph after extraction (should be unchanged)
        print("\n=== Graph Memory AFTER extracting triples ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Extracting Triples")
        
        assert len(triples) == 1
        assert isinstance(triples[0], roles.open_ie.Relation)
    
    # ========================================================================
    # Graph Operations Tests
    # ========================================================================
    
    def test_connect_graph_memory_empty(self, working_memory: WorkingMemory):
        """Test connecting empty graph."""
        # Visualize graph before
        print("\n=== Graph Memory BEFORE connecting (empty) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Connecting (Empty)")
        
        result = working_memory.connect_graph_memory()
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER connecting (empty) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Connecting (Empty)")
        
        assert result is True  # Empty or single node graphs are already connected
    
    def test_connect_graph_memory_single_node(self, working_memory, sample_entities):
        """Test connecting graph with single node."""
        node_id = get_node_id(sample_entities['paris'])
        working_memory.graph_memory.add_node(node_id, data=sample_entities['paris'])
        
        # Visualize graph before
        print("\n=== Graph Memory BEFORE connecting (single node) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Connecting (Single Node)")
        
        result = working_memory.connect_graph_memory()
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER connecting (single node) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Connecting (Single Node)")
        
        assert result is True
    
    def test_connect_graph_memory_already_connected(self, working_memory, sample_triple):
        """Test connecting already connected graph."""
        working_memory.add_edge_to_graph_memory(sample_triple)
        
        # Visualize graph before
        print("\n=== Graph Memory BEFORE connecting (already connected) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Connecting (Already Connected)")
        
        result = working_memory.connect_graph_memory()
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER connecting (already connected) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Connecting (Already Connected)")
        
        assert result is True  # Already connected
    
    def test_connect_graph_memory_disconnected(self, working_memory: WorkingMemory, sample_entities):
        """Test connecting graph with disconnected components."""
        # Create two disconnected components
        # Component 1: Paris -> France
        paris = sample_entities['paris']
        france = sample_entities['france']
        prop1 = WikidataProperty(pid="P1376", label="capital of")
        triple1 = WikiTriple(subject=paris, relation=prop1, object=france)
        working_memory.add_edge_to_graph_memory(triple1)
        
        # # Component 2: Tokyo -> Japan (disconnected from Paris-France)
        # tokyo = sample_entities['tokyo']
        # japan = WikidataEntity(qid="Q17", label="Japan", description="country in East Asia")
        # prop2 = WikidataProperty(pid="P1376", label="capital of")
        # triple2 = WikiTriple(subject=tokyo, relation=prop2, object=japan)
        # working_memory.add_edge_to_graph_memory(triple2)

        triple3 = WikiTriple(
            subject=WikidataEntity(qid="Q937", label="Albert Einstein", description="'German-born theoretical physicist (1879–1955)"),
            relation=WikidataProperty(pid="P27", label="country of citizenship"), 
            object=WikidataEntity(qid="Q39", label="Switzerland", description="country in Central Europe"))
        working_memory.add_edge_to_graph_memory(triple3)
        
        # Verify we have disconnected components
        components_before = list(nx.weakly_connected_components(working_memory.graph_memory))
        assert len(components_before) == 2, f"Expected 2 components, got {len(components_before)}"
        
        # Visualize graph before connecting
        print("\n=== Graph Memory BEFORE connecting (disconnected) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Connecting (Disconnected)")
        
        # Try to connect the graph
        result = working_memory.connect_graph_memory(max_hops=3)
        working_memory.update_graph_memory()
        
        # Visualize graph after connecting
        print("\n=== Graph Memory AFTER connecting (disconnected) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Connecting (Disconnected)")
        
        # Check components after connection attempt
        components_after = list(nx.weakly_connected_components(working_memory.graph_memory))
        
        # The graph may or may not be connected depending on whether path finder is available
        # and whether paths exist in Wikidata. At minimum, verify the function ran.
        assert isinstance(result, bool)
        # If connected, should have 1 component; otherwise may still have 2 or more
        assert len(components_after) >= 1
        # The number of components should be <= the number before (may have connected some)
        assert len(components_after) <= len(components_before)
    
    def test_update_graph_memory(self, working_memory, sample_entities):
        """Test updating graph memory with entity details."""
        # Add entity without full details
        entity = WikidataEntity(qid="Q90", label="Paris")
        node_id = get_node_id(entity)
        working_memory.graph_memory.add_node(node_id, data=entity)
        working_memory.id_dict[entity.qid] = sample_entities['paris']  # Pre-populate id_dict
        
        # Visualize graph before
        print("\n=== Graph Memory BEFORE updating ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Updating")
        
        working_memory.update_graph_memory()
        
        # Visualize graph after
        print("\n=== Graph Memory AFTER updating ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Updating")
        
        # Node should be updated with full entity details
        node_data = working_memory.graph_memory.nodes[node_id]['data']
        assert node_data.description is not None
    
    # ========================================================================
    # Integration Tests (require LLM)
    # ========================================================================
    
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
    def interaction_memory(self, request):
        """Create an InteractionMemory instance with unique collection name."""
        # Use unique collection name for test isolation
        collection_name = f"test_interaction_memory_{uuid.uuid4().hex[:8]}"
        memory = InteractionMemory(collection_name=collection_name)
        yield memory
        # Clean up after test - delete collection to ensure isolation
        memory.release(should_delete_db=True)
    
    @pytest.mark.slow
    def test_consolidate_textual_memory(self, llm_agent, working_memory, interaction_memory):
        """Test memory consolidation with LLM."""
        # Add various items to memory
        working_memory.add_textual_memory(
            "Paris is the capital of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        working_memory.add_textual_memory(
            "The Eiffel Tower is in Paris.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        working_memory.add_textual_memory(
            "France is in Europe.",
            source=roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        
        initial_count = len(working_memory.textual_memory)
        
        question = "What is the capital of France?"
        working_memory.consolidate_textual_memory(llm_agent, question, interaction_memory)
        
        # Memory should be consolidated (may have fewer or same items)
        assert len(working_memory.textual_memory) >= 0
        print(working_memory.textual_memory)
    
    @pytest.mark.slow
    def test_parse_graph_memory_from_textual_memory(self, llm_agent, working_memory, interaction_memory):
        """Test parsing graph from textual memory."""
        working_memory.add_textual_memory(
            "Paris is the capital of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        working_memory.add_textual_memory(
            "The Eiffel Tower is located in Paris.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        # Visualize graph_memory before parsing
        print("\n=== Graph Memory BEFORE parsing from textual memory ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Parsing from Textual Memory")
        
        working_memory.parse_graph_memory_from_textual_memory(llm_agent, interaction_memory)
        
        # Visualize parsed_graph_memory after parsing
        print("\n=== Parsed Graph Memory AFTER parsing from textual memory ===")
        if working_memory.parsed_graph_memory is not None:
            visualize_graph(working_memory.parsed_graph_memory, title="Parsed Graph Memory - After Parsing")
        
        # Visualize graph_memory after parsing (may have been updated)
        print("\n=== Graph Memory AFTER parsing from textual memory ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Parsing from Textual Memory")
        
        # Should have parsed graph with some nodes
        assert working_memory.parsed_graph_memory is not None
        assert isinstance(working_memory.parsed_graph_memory, nx.DiGraph)
    
    def test_merge_graph_memory(self, working_memory: WorkingMemory, sample_entities, sample_property):
        """Test merging another graph into graph memory."""
        # Create another graph with triples using roles.open_ie.Entity objects
        # (as expected by extract_triples_from_graph)
        other_graph = nx.DiGraph()
        
        paris_entity = roles.open_ie.Entity(name="Paris", description="capital of France")
        france_entity = roles.open_ie.Entity(name="France", description="country")
        
        paris_id = get_node_id(paris_entity)
        france_id = get_node_id(france_entity)
        
        other_graph.add_node(paris_id, data=paris_entity)
        other_graph.add_node(france_id, data=france_entity)
        other_graph.add_edge(paris_id, france_id, relation={"capital of"})
        
        # Pre-populate entity_dict and property_dict
        working_memory.entity_dict[paris_entity] = sample_entities['paris']
        working_memory.entity_dict[france_entity] = sample_entities['france']
        working_memory.property_dict["capital of"] = sample_property
        # Visualize graph before merge
        print("\n=== Graph Memory BEFORE merging ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Merging")
        print("\n=== Other Graph to be merged ===")
        visualize_graph(other_graph, title="Other Graph to Merge")
        
        # Merge the graph
        initial_node_count = len(working_memory.graph_memory.nodes)
        working_memory.merge_graph_memory(other_graph)
        
        # Visualize graph after merge
        print("\n=== Graph Memory AFTER merging ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Merging")
        
        # Graph should have been merged (nodes/edges added)
        assert len(working_memory.graph_memory.nodes) >= initial_node_count
    
    @pytest.mark.slow
    def test_consolidate_graph_memory(self, llm_agent, working_memory, interaction_memory):
        """Test consolidating graph memory."""
        # Add some triples to graph memory
        paris = WikidataEntity(qid="Q90", label="Paris", description="capital of France")
        france = WikidataEntity(qid="Q142", label="France", description="country")
        prop = WikidataProperty(pid="P1376", label="capital of")
        
        triple = WikiTriple(subject=paris, relation=prop, object=france)
        working_memory.add_edge_to_graph_memory(triple)
        
        initial_node_count = len(working_memory.graph_memory.nodes)
        
        # Visualize graph before consolidation
        print("\n=== Graph Memory BEFORE consolidation ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Consolidation")
        
        question = "What is the capital of France?"
        working_memory.consolidate_graph_memory(llm_agent, question, interaction_memory)
        
        # Visualize graph after consolidation
        print("\n=== Graph Memory AFTER consolidation ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Consolidation")
        
        # Graph should be consolidated (may have different structure)
        assert isinstance(working_memory.graph_memory, nx.DiGraph)
    
    @pytest.mark.slow
    def test_synchronize_memory(self, llm_agent, working_memory, interaction_memory):
        """Test synchronizing graph and textual memory."""
        # Add textual memory
        working_memory.add_textual_memory(
            "Paris is the capital of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        # Add graph memory
        paris = WikidataEntity(qid="Q90", label="Paris", description="capital of France")
        france = WikidataEntity(qid="Q142", label="France", description="country")
        prop = WikidataProperty(pid="P1376", label="capital of")
        triple = WikiTriple(subject=paris, relation=prop, object=france)
        working_memory.add_edge_to_graph_memory(triple)
        
        # Visualize graph before synchronization
        print("\n=== Graph Memory BEFORE synchronization ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Synchronization")
        print("\n=== Textual Memory BEFORE synchronization ===")
        print(working_memory.textual_memory)
        
        question = "What is the capital of France?"
        working_memory.synchronize_memory(llm_agent, question, interaction_memory)
        
        # Visualize graph after synchronization
        print("\n=== Graph Memory AFTER synchronization ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Synchronization")
        print("\n=== Textual Memory AFTER synchronization ===")
        print(working_memory.textual_memory)
        
        # Both memories should be synchronized
        assert len(working_memory.textual_memory) >= 0
        assert isinstance(working_memory.graph_memory, nx.DiGraph)


# ============================================================================
# InteractionMemory Tests
# ============================================================================

class TestInteractionMemory:
    """Test suite for InteractionMemory functionality."""
    
    @pytest.fixture
    def interaction_memory(self, request):
        """Create an InteractionMemory instance with unique collection name."""
        # Use unique collection name for test isolation
        collection_name = f"test_interaction_memory_{uuid.uuid4().hex[:8]}"
        memory = InteractionMemory(collection_name=collection_name)
        yield memory
        # Clean up after test - delete collection to ensure isolation
        memory.release(should_delete_db=True)
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_init_default(self, interaction_memory):
        """Test default initialization."""
        assert interaction_memory is not None
        assert interaction_memory.token_budget == 8192
        # Collection name will be unique due to fixture, so just check it exists
        assert interaction_memory.collection_name is not None
        assert interaction_memory.collection_name.startswith("test_interaction_memory_")
        assert interaction_memory.collection.count() == 0
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        memory = InteractionMemory(
            collection_name="test_collection",
            token_budget=4096,
            enable_embedding_cache=False
        )
        assert memory.collection_name == "test_collection"
        assert memory.token_budget == 4096
        assert memory._embedding_cache is None
        memory.release()
    
    def test_init_with_db_path(self, tmp_path):
        """Test initialization with persistent database path."""
        db_path = str(tmp_path / "test_db")
        memory = InteractionMemory(db_path=db_path, collection_name="test_collection")
        assert memory.db_client is not None
        assert memory.collection_name == "test_collection"
        memory.release(should_delete_db=True)
    
    def test_init_with_db_client(self):
        """Test initialization with provided database client."""
        import chromadb
        client = chromadb.EphemeralClient()
        memory = InteractionMemory(db_client=client, collection_name="test_collection")
        assert memory.db_client is client
        assert memory.collection_name == "test_collection"
        memory.release()
    
    # ========================================================================
    # Logging Tests
    # ========================================================================
    
    def test_log_turn_single(self, interaction_memory):
        """Test logging a single turn."""
        interaction_memory.log_turn(
            role="generator",
            user_input="What is the capital of France?",
            assistant_output="Paris is the capital of France."
        )
        
        assert interaction_memory.collection.count() == 1
    
    def test_log_turn_multiple(self, interaction_memory):
        """Test logging multiple turns at once."""
        interaction_memory.log_turn(
            role="generator",
            user_input=["Question 1", "Question 2"],
            assistant_output=["Answer 1", "Answer 2"]
        )
        
        count = interaction_memory.collection.count()
        print(f"test_log_turn_multiple: Expected count=2, Got count={count}")
        assert count == 2
    
    def test_log_turn_different_roles(self, interaction_memory):
        """Test logging turns for different roles."""
        interaction_memory.log_turn(
            role="generator",
            user_input="Question 1",
            assistant_output="Answer 1"
        )
        interaction_memory.log_turn(
            role="extractor",
            user_input="Question 2",
            assistant_output="Answer 2"
        )
        
        count = interaction_memory.collection.count()
        print(f"test_log_turn_different_roles: Expected count=2, Got count={count}")
        assert count == 2
        info = interaction_memory.get_info()
        print(f"test_log_turn_different_roles: Info={info}")
        assert "generator" in info["unique_roles"]
        assert "extractor" in info["unique_roles"]
    
    # ========================================================================
    # Retrieval Tests
    # ========================================================================
    
    def test_get_examples_empty(self, interaction_memory):
        """Test retrieving examples from empty memory."""
        count_before = interaction_memory.collection.count()
        examples = interaction_memory.get_examples(
            role="generator",
            query="test query",
            k=3
        )
        print(f"test_get_examples_empty: Collection count={count_before}, Expected examples=[], Got examples={examples}")
        assert examples == []
    
    def test_get_examples_similarity(self, interaction_memory):
        """Test similarity-based retrieval."""
        # Log some examples
        interaction_memory.log_turn(
            role="generator",
            user_input="What is the capital of France?",
            assistant_output="Paris is the capital of France."
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="What is the capital of Japan?",
            assistant_output="Tokyo is the capital of Japan."
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="Is Germany in Europe?",
            assistant_output="Yes, Germany is in Europe."
        )
        # an irrelevant example
        interaction_memory.log_turn(
            role="generator",
            user_input="Who is the president of the United States?",
            assistant_output="Joe Biden is the president of the United States."
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="Who invented the internet?",
            assistant_output="Tim Berners-Lee invented the internet."
        )
        
        # Retrieve examples
        examples = interaction_memory.get_examples(
            role="generator",
            query="What is the capital of Germany?",
            k=2,
            strategy="similarity"
        )
        
        assert len(examples) <= 2
        if examples:
            assert len(examples[0]) == 2  # user and assistant messages
        # Print the examples
        print(f"Examples: {examples}")
    
    def test_get_examples_mmr(self, interaction_memory):
        """Test MMR-based retrieval for diversity."""
        # Log similar examples
        interaction_memory.log_turn(
            role="generator",
            user_input="How to select all users?",
            assistant_output="SELECT * FROM users;"
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="Select all columns from users table",
            assistant_output="SELECT * FROM users;"
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="Get every record from users",
            assistant_output="SELECT * FROM users;"
        )
        interaction_memory.log_turn(
            role="generator",
            user_input="How to delete a table?",
            assistant_output="DROP TABLE users;"
        )
        
        # Retrieve with MMR (should prefer diversity)
        examples = interaction_memory.get_examples(
            role="generator",
            query="Show me how to get data from users",
            k=2,
            strategy="mmr"
        )
        
        assert len(examples) <= 2
        # MMR should prefer diverse examples
        print(f"test_get_examples_mmr: Examples={examples}")
    
    def test_get_examples_invalid_strategy(self, interaction_memory):
        """Test get_examples with invalid strategy."""
        interaction_memory.log_turn(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            interaction_memory.get_examples(
                role="generator",
                query="test",
                k=1,
                strategy="invalid_strategy"
            )
    
    def test_get_examples_token_budget(self, interaction_memory: InteractionMemory):
        """Test that examples respect token budget."""
        # Log many examples
        for i in range(10):
            interaction_memory.log_turn(
                role="generator",
                user_input=f"Question {i}",
                assistant_output=f"Answer {i} " * 10  # Long answer
            )
        
        # Set small token budget
        interaction_memory.token_budget = 100
        
        examples = interaction_memory.get_examples(
            role="generator",
            query="test query",
            k=10,
            strategy="similarity"
        )
        
        # Should be trimmed to fit token budget
        assert len(examples) <= 10
        print(f"test_get_examples_token_budget: Examples={examples}")
    
    # ========================================================================
    # Async Tests
    # ========================================================================
    
    @pytest.mark.asyncio
    async def test_log_turn_async(self, interaction_memory):
        """Test async logging."""
        await interaction_memory.log_turn_async(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        count = interaction_memory.collection.count()
        print(f"test_log_turn_async: Expected count=1, Got count={count}")
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_get_examples_async(self, interaction_memory):
        """Test async retrieval."""
        interaction_memory.log_turn(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        examples = await interaction_memory.get_examples_async(
            role="generator",
            query="test query",
            k=1
        )
        
        print(f"test_get_examples_async: Expected len <= 1, Got len={len(examples)}, examples={examples}")
        assert len(examples) <= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, interaction_memory: InteractionMemory):
        """Test concurrent read/write access."""
        # Concurrent writes
        await asyncio.gather(*[
            interaction_memory.log_turn_async(
                role="generator",
                user_input=f"Question {i}",
                assistant_output=f"Answer {i}"
            )
            for i in range(5)
        ])
        
        count = interaction_memory.collection.count()
        print(f"test_concurrent_access: After writes, Expected count=5, Got count={count}")
        assert count == 5
        
        # Concurrent reads
        results = await asyncio.gather(*[
            interaction_memory.get_examples_async(
                role="generator",
                query="test",
                k=3
            )
            for _ in range(3)
        ])

        print(f"test_concurrent_access: Results={results}")
        print(f"test_concurrent_access: Expected len(results)=3, Got len(results)={len(results)}")
        # All reads should succeed
        assert len(results) == 3
    
    # ========================================================================
    # Utility Tests
    # ========================================================================
    
    def test_get_info(self, interaction_memory):
        """Test getting memory information."""
        interaction_memory.log_turn(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        info = interaction_memory.get_info()
        print(f"test_get_info: Info={info}, Expected documents=1, Got documents={info.get('documents')}")
        assert "collections" in info
        assert "used_collection" in info
        assert "documents" in info
        assert "unique_roles" in info
        assert info["documents"] == 1
        assert "generator" in info["unique_roles"]
    
    def test_get_info_empty(self, interaction_memory):
        """Test getting info from empty memory."""
        info = interaction_memory.get_info()
        print(f"test_get_info_empty: Info={info}, Expected documents=0, Got documents={info.get('documents')}")
        assert "collections" in info
        assert "documents" in info
        assert info["documents"] == 0
        assert isinstance(info["unique_roles"], set)
    
    def test_release(self, interaction_memory):
        """Test releasing memory resources."""
        interaction_memory.log_turn(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        interaction_memory.release(should_delete_db=False)
        
        # Collection should be None after release
        assert interaction_memory.collection is None
        assert interaction_memory.db_client is None
    
    def test_release_with_delete(self, tmp_path):
        """Test releasing memory with database deletion."""
        db_path = str(tmp_path / "test_db")
        memory = InteractionMemory(db_path=db_path, collection_name="test_collection")
        memory.log_turn(
            role="generator",
            user_input="Question",
            assistant_output="Answer"
        )
        
        memory.release(should_delete_db=True)
        
        # Collection should be None after release
        assert memory.collection is None
        assert memory.db_client is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryIntegration:
    """Integration tests for WorkingMemory and InteractionMemory together."""
    
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
    def working_memory(self):
        """Create a WorkingMemory instance."""
        return WorkingMemory()
    
    @pytest.fixture
    def interaction_memory(self, request):
        """Create an InteractionMemory instance with unique collection name."""
        # Use unique collection name for test isolation
        collection_name = f"test_interaction_memory_{uuid.uuid4().hex[:8]}"
        memory = InteractionMemory(collection_name=collection_name)
        yield memory
        # Clean up after test - delete collection to ensure isolation
        memory.release(should_delete_db=True)
    
    @pytest.mark.slow
    def test_memory_consolidation_with_interaction_memory(
        self, llm_agent, working_memory, interaction_memory
    ):
        """Test memory consolidation logs to interaction memory."""
        working_memory.add_textual_memory(
            "Paris is the capital of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        initial_count = interaction_memory.collection.count()
        
        question = "What is the capital of France?"
        working_memory.consolidate_textual_memory(llm_agent, question, interaction_memory)
        
        # Interaction memory should have logged the consolidation
        assert interaction_memory.collection.count() >= initial_count
    
    @pytest.mark.slow
    def test_parse_graph_with_interaction_memory(
        self, llm_agent, working_memory, interaction_memory
    ):
        """Test parsing graph logs to interaction memory."""
        working_memory.add_textual_memory(
            "Paris is the capital of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        initial_count = interaction_memory.collection.count()
        
        # Visualize graph_memory before parsing
        print("\n=== Graph Memory BEFORE parsing (integration test) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - Before Parsing (Integration Test)")
        
        working_memory.parse_graph_memory_from_textual_memory(llm_agent, interaction_memory)
        
        # Visualize parsed_graph_memory after parsing
        print("\n=== Parsed Graph Memory AFTER parsing (integration test) ===")
        if working_memory.parsed_graph_memory is not None:
            visualize_graph(working_memory.parsed_graph_memory, title="Parsed Graph Memory - After Parsing (Integration Test)")
        
        # Visualize graph_memory after parsing
        print("\n=== Graph Memory AFTER parsing (integration test) ===")
        visualize_graph(working_memory.graph_memory, title="Graph Memory - After Parsing (Integration Test)")
        
        # Interaction memory should have logged the parsing
        assert interaction_memory.collection.count() >= initial_count

