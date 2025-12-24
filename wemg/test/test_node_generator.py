"""
Comprehensive tests for wemg/runners/procedures/node_generator.py.

These are real integration tests that call actual LLM servers and retrieval APIs.
Tests cover:
- GenerationResult dataclass
- NodeGenerator initialization
- generate_answer method
- generate_subquestion method
- generate_rephrase method
- generate_self_correction method
- generate_synthesis method
- update_working_memory method
"""
import os
import pytest
import asyncio
from pathlib import Path
from typing import List, Dict

from wemg.runners.procedures.node_generator import NodeGenerator, GenerationResult
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool, DDGSAPIWrapper
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.agents import roles


# ============================================================================
# Test Configuration
# ============================================================================

TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your-serper-api-key")

# Wiki corpus configuration for RetrieverAgent tests
WIKI_CORPUS_HF = os.getenv("WIKI_CORPUS_HF", "Hieuman/wiki23-processed")
WIKI_INDEX_PATH = Path(os.getenv("WIKI_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def llm_agent():
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
def web_search_tool():
    """Create a WebSearchTool for testing."""
    return WebSearchTool(serper_api_key=SERPER_API_KEY)


@pytest.fixture
def retriever_agent_embedder_config():
    """Create embedder configuration for RetrieverAgent."""
    return {
        'model_name': TEST_EMBEDDING_MODEL,
        'url': TEST_EMBEDDING_API_BASE,
        'api_key': TEST_LLM_API_KEY,
        'is_embedding': True,
        'timeout': 60,
    }


@pytest.fixture
def retriever_agent(retriever_agent_embedder_config):
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


@pytest.fixture
def working_memory():
    """Create a WorkingMemory instance."""
    return WorkingMemory()


@pytest.fixture
def interaction_memory():
    """Create an InteractionMemory instance."""
    return InteractionMemory()


@pytest.fixture
def node_generator_with_websearch(llm_agent, web_search_tool, working_memory, interaction_memory):
    """Create a NodeGenerator instance with WebSearchTool."""
    return NodeGenerator(
        llm_agent=llm_agent,
        retriever_agent=web_search_tool,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        top_k_websearch=3,
        top_k_entities=1,
        top_k_properties=1,
        n_hops=1,
        n=1
    )


@pytest.fixture
def node_generator_with_retriever(llm_agent, retriever_agent, working_memory, interaction_memory):
    """Create a NodeGenerator instance with RetrieverAgent."""
    if not WIKI_INDEX_PATH.exists():
        pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}.")
    
    return NodeGenerator(
        llm_agent=llm_agent,
        retriever_agent=retriever_agent,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        top_k_websearch=3,
        top_k_entities=1,
        top_k_properties=1,
        n_hops=1,
        n=1
    )


# ============================================================================
# GenerationResult Tests
# ============================================================================

class TestGenerationResult:
    """Test suite for GenerationResult dataclass."""
    
    def test_init_with_defaults(self):
        """Test GenerationResult initialization with default values."""
        result = GenerationResult()
        assert result.answers == []
        assert result.retrieved_triples == []
        assert result.entity_dict == {}
        assert result.property_dict == {}
        assert result.log_data == {}
    
    def test_init_with_values(self):
        """Test GenerationResult initialization with provided values."""
        from wemg.agents import roles
        
        answer = roles.generator.AnswerGenerationOutput(
            answer="Test answer",
            concise_answer="Test",
            reasoning="Test reasoning",
            confidence_level="high"
        )
        
        result = GenerationResult(
            answers=[answer],
            retrieved_triples=[],
            entity_dict={"e1": "entity1"},
            property_dict={"p1": "property1"},
            log_data={"role1": [("input", "output")]}
        )
        
        assert len(result.answers) == 1
        assert result.answers[0].answer == "Test answer"
        assert result.entity_dict == {"e1": "entity1"}
        assert result.property_dict == {"p1": "property1"}
        assert result.log_data == {"role1": [("input", "output")]}
    
    def test_init_with_none_values(self):
        """Test GenerationResult post_init handles None values."""
        result = GenerationResult(
            answers=None,
            retrieved_triples=None,
            entity_dict=None,
            property_dict=None,
            log_data=None
        )
        
        assert result.answers == []
        assert result.retrieved_triples == []
        assert result.entity_dict == {}
        assert result.property_dict == {}
        assert result.log_data == {}


# ============================================================================
# NodeGenerator Initialization Tests
# ============================================================================

class TestNodeGeneratorInit:
    """Test suite for NodeGenerator initialization."""
    
    def test_init_with_all_params(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test NodeGenerator initialization with all parameters."""
        generator = NodeGenerator(
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            top_k_websearch=10,
            n=3
        )
        
        assert generator.llm_agent == llm_agent
        assert generator.retriever_agent == web_search_tool
        assert generator.working_memory == working_memory
        assert generator.interaction_memory == interaction_memory
        assert generator.kwargs['top_k_websearch'] == 10
        assert generator.kwargs['n'] == 3
    
    def test_init_without_interaction_memory(self, llm_agent, web_search_tool, working_memory):
        """Test NodeGenerator initialization without interaction memory."""
        generator = NodeGenerator(
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory
        )
        
        assert generator.interaction_memory is None


# ============================================================================
# generate_answer Tests
# ============================================================================

class TestGenerateAnswer:
    """Test suite for generate_answer method."""
    
    @pytest.mark.slow
    def test_generate_answer_with_websearch(self, node_generator_with_websearch: NodeGenerator):
        """Test answer generation with WebSearchTool."""
        question = "Which magazine was started first Arthur's Magazine or First for Women?"
        
        result = asyncio.run(node_generator_with_websearch.generate_answer(question))
        
        assert isinstance(result, GenerationResult)
        assert len(result.answers) > 0
        assert result.answers[0].answer is not None
        assert len(result.answers[0].answer) > 0
        assert result.answers[0].confidence_level in ["high", "medium", "low"]
        
        print(f"✓ generate_answer with WebSearchTool")
        print(f"  Question: {question}")
        print(f"  Answer: {result.answers[0].answer[:200]}")
        print(f"  Confidence: {result.answers[0].confidence_level}")
        print(f"  Retrieved triples: {len(result.retrieved_triples)}")
        print(f"  Entities: {len(result.entity_dict)}")
        print(f"  Properties: {len(result.property_dict)}")
    
    @pytest.mark.slow
    def test_generate_answer_with_retriever(self, node_generator_with_retriever):
        """Test answer generation with RetrieverAgent."""
        question = "Which magazine was started first Arthur's Magazine or First for Women?"
        
        result = asyncio.run(node_generator_with_retriever.generate_answer(question))
        
        assert isinstance(result, GenerationResult)
        assert len(result.answers) > 0
        assert result.answers[0].answer is not None
        
        print(f"✓ generate_answer with RetrieverAgent")
        print(f"  Question: {question}")
        print(f"  Answer: {result.answers[0].answer[:200]}")
        print(f"  Retrieved documents: {len(result.retrieved_triples)}")
    
    @pytest.mark.slow
    def test_generate_answer_with_memory_entities(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test answer generation with entities in working memory."""
        from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty
        from wemg.agents import roles
        
        # Add some entities to working memory
        entity = WikidataEntity(qid="Q142", label="France", description="Country in Western Europe")
        prop = WikidataProperty(pid="P36", label="capital", description="Capital city")
        
        open_ie_entity = roles.open_ie.Entity(name="France")
        working_memory.entity_dict[open_ie_entity] = entity
        working_memory.property_dict["capital"] = prop
        
        generator = NodeGenerator(
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            top_k_websearch=3,
            top_k_entities=3,
            top_k_properties=10,
            n_hops=1,
            n=1
        )
        
        question = "What is the capital of France?"
        result = asyncio.run(generator.generate_answer(question))
        
        assert isinstance(result, GenerationResult)
        assert len(result.answers) > 0
        
        print(f"✓ generate_answer with memory entities")
        print(f"  Memory entities: {len(working_memory.entity_dict)}")
        print(f"  Result entities: {len(result.entity_dict)}")
    
    @pytest.mark.slow
    def test_generate_answer_multiple_samples(self, node_generator_with_websearch):
        """Test answer generation with multiple samples (n>1)."""
        # Create generator with n=3
        generator = NodeGenerator(
            llm_agent=node_generator_with_websearch.llm_agent,
            retriever_agent=node_generator_with_websearch.retriever_agent,
            working_memory=node_generator_with_websearch.working_memory,
            interaction_memory=node_generator_with_websearch.interaction_memory,
            top_k_websearch=3,
            n=3
        )
        
        question = "What are the main causes of climate change?"
        result = asyncio.run(generator.generate_answer(question))
        
        assert len(result.answers) == 3
        for i, answer in enumerate(result.answers):
            assert answer.answer is not None
            print(f"  Sample {i+1}: {answer.concise_answer[:100]}")
        
        print(f"✓ generate_answer with n=3")


# ============================================================================
# generate_subquestion Tests
# ============================================================================

class TestGenerateSubquestion:
    """Test suite for generate_subquestion method."""
    
    @pytest.mark.slow
    def test_generate_subquestion_answerable(self, node_generator_with_websearch):
        """Test subquestion generation when question is answerable."""
        question = "What is the capital of France?"
        
        # Add context to working memory to make it answerable
        node_generator_with_websearch.working_memory.add_textual_memory(
            "France is a country in Western Europe. Paris is the capital and largest city of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        subquestion, should_direct_answer, log_data = asyncio.run(
            node_generator_with_websearch.generate_subquestion(question)
        )
        
        # Should be answerable with context
        assert should_direct_answer is True or subquestion is None
        
        print(f"✓ generate_subquestion (answerable case)")
        print(f"  Should direct answer: {should_direct_answer}")
        print(f"  Subquestion: {subquestion}")
    
    @pytest.mark.slow
    def test_generate_subquestion_not_answerable(self, node_generator_with_websearch):
        """Test subquestion generation when question is not answerable."""
        question = "Who was the president of France when the Eiffel Tower was built?"
        
        # Minimal context that doesn't answer the question
        node_generator_with_websearch.working_memory.add_textual_memory(
            "The Eiffel Tower is a famous landmark in Paris.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        subquestion, should_direct_answer, log_data = asyncio.run(
            node_generator_with_websearch.generate_subquestion(question)
        )
        
        # Should generate a subquestion since context is insufficient
        if not should_direct_answer:
            assert subquestion is not None
            assert len(subquestion) > 0
        
        print(f"✓ generate_subquestion (not answerable case)")
        print(f"  Should direct answer: {should_direct_answer}")
        print(f"  Subquestion: {subquestion}")
    
    @pytest.mark.slow
    def test_generate_subquestion_empty_memory(self, node_generator_with_websearch):
        """Test subquestion generation with empty working memory."""
        question = "What is the capital of France?"
        
        subquestion, should_direct_answer, log_data = asyncio.run(
            node_generator_with_websearch.generate_subquestion(question)
        )
        
        # Without context, should generate subquestion or indicate not answerable
        assert isinstance(should_direct_answer, bool)
        
        print(f"✓ generate_subquestion (empty memory)")
        print(f"  Should direct answer: {should_direct_answer}")
        print(f"  Subquestion: {subquestion}")


# ============================================================================
# generate_rephrase Tests
# ============================================================================

class TestGenerateRephrase:
    """Test suite for generate_rephrase method."""
    
    @pytest.mark.slow
    def test_generate_rephrase_success(self, node_generator_with_websearch):
        """Test successful question rephrasing."""
        question = "What is the capital of France?"
        
        rephrased_questions, log_data = asyncio.run(
            node_generator_with_websearch.generate_rephrase(question)
        )
        
        assert isinstance(rephrased_questions, list)
        assert len(rephrased_questions) > 0
        assert all(isinstance(q, str) and len(q) > 0 for q in rephrased_questions)
        
        print(f"✓ generate_rephrase")
        print(f"  Original: {question}")
        for i, rephrased in enumerate(rephrased_questions[:3]):
            print(f"  Rephrase {i+1}: {rephrased}")
    
    @pytest.mark.slow
    def test_generate_rephrase_multiple_samples(self, node_generator_with_websearch):
        """Test rephrase generation with multiple samples."""
        # Create generator with n=3
        generator = NodeGenerator(
            llm_agent=node_generator_with_websearch.llm_agent,
            retriever_agent=node_generator_with_websearch.retriever_agent,
            working_memory=node_generator_with_websearch.working_memory,
            interaction_memory=node_generator_with_websearch.interaction_memory,
            n=3
        )
        
        question = "What is machine learning?"
        rephrased_questions, log_data = asyncio.run(generator.generate_rephrase(question))
        
        assert len(rephrased_questions) >= 1  # At least one rephrase
        print(f"✓ generate_rephrase with n=3")
        print(f"  Generated {len(rephrased_questions)} rephrases")


# ============================================================================
# generate_self_correction Tests
# ============================================================================

class TestGenerateSelfCorrection:
    """Test suite for generate_self_correction method."""
    
    @pytest.mark.slow
    def test_generate_self_correction_success(self, node_generator_with_websearch):
        """Test successful self-correction."""
        sub_question = "What is the capital of France?"
        sub_answer = "Lyon"  # Incorrect answer
        
        result = asyncio.run(
            node_generator_with_websearch.generate_self_correction(sub_question, sub_answer)
        )
        
        assert isinstance(result, GenerationResult)
        assert len(result.answers) > 0
        assert result.answers[0].status in ["correct", "partial", "incorrect", "unsupported"]
        assert result.answers[0].refined_answer is not None
        assert result.answers[0].confidence_level in ["high", "medium", "low"]
        
        print(f"✓ generate_self_correction")
        print(f"  Question: {sub_question}")
        print(f"  Proposed answer: {sub_answer}")
        print(f"  Status: {result.answers[0].status}")
        print(f"  Refined answer: {result.answers[0].refined_answer[:200]}")
        print(f"  Confidence: {result.answers[0].confidence_level}")
    
    @pytest.mark.slow
    def test_generate_self_correction_correct_answer(self, node_generator_with_websearch):
        """Test self-correction with a correct answer."""
        sub_question = "What is the capital of France?"
        sub_answer = "Paris"  # Correct answer
        
        result = asyncio.run(
            node_generator_with_websearch.generate_self_correction(sub_question, sub_answer)
        )
        
        assert len(result.answers) > 0
        # Should verify as correct or partial
        assert result.answers[0].status in ["correct", "partial"]
        
        print(f"✓ generate_self_correction (correct answer)")
        print(f"  Status: {result.answers[0].status}")


# ============================================================================
# generate_synthesis Tests
# ============================================================================

class TestGenerateSynthesis:
    """Test suite for generate_synthesis method."""
    
    @pytest.mark.slow
    def test_generate_synthesis_success(self, node_generator_with_websearch):
        """Test successful reasoning synthesis."""
        question = "What is the capital of France?"
        
        # Add some context to working memory
        node_generator_with_websearch.working_memory.add_textual_memory(
            "France is a country in Western Europe.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        node_generator_with_websearch.working_memory.add_textual_memory(
            "Paris is the capital and largest city of France.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        outputs, log_data = asyncio.run(
            node_generator_with_websearch.generate_synthesis(question)
        )
        
        assert isinstance(outputs, list)
        assert len(outputs) > 0
        assert outputs[0].is_answerable is not None
        assert outputs[0].step_conclusion is not None
        assert outputs[0].confidence_level in ["high", "medium", "low"]
        
        print(f"✓ generate_synthesis")
        print(f"  Question: {question}")
        print(f"  Is answerable: {outputs[0].is_answerable}")
        print(f"  Conclusion: {outputs[0].step_conclusion[:200]}")
        print(f"  Confidence: {outputs[0].confidence_level}")
    
    @pytest.mark.slow
    def test_generate_synthesis_not_answerable(self, node_generator_with_websearch):
        """Test synthesis when question is not answerable."""
        question = "Who was the president of France in 1800?"
        
        # Minimal context that doesn't answer the question
        node_generator_with_websearch.working_memory.add_textual_memory(
            "France is a country in Western Europe.",
            source=roles.extractor.SourceType.RETRIEVAL
        )
        
        outputs, log_data = asyncio.run(
            node_generator_with_websearch.generate_synthesis(question)
        )
        
        assert len(outputs) > 0
        # Should indicate not answerable or low confidence
        assert outputs[0].is_answerable is False or outputs[0].confidence_level == "low"
        
        print(f"✓ generate_synthesis (not answerable)")
        print(f"  Is answerable: {outputs[0].is_answerable}")


# ============================================================================
# update_working_memory Tests
# ============================================================================

class TestUpdateWorkingMemory:
    """Test suite for update_working_memory method."""
    
    def test_update_working_memory_with_entities_and_properties(self, node_generator_with_websearch):
        """Test updating working memory with entities, properties, and triples."""
        from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty, WikiTriple
        from wemg.agents import roles
        
        # Create test entities and properties
        entity1 = WikidataEntity(qid="Q142", label="France", description="Country")
        entity2 = WikidataEntity(qid="Q90", label="Paris", description="Capital city")
        prop = WikidataProperty(pid="P36", label="capital", description="Capital city")
        
        open_ie_entity = roles.open_ie.Entity(name="France")
        
        # Create a triple
        triple = WikiTriple(
            subject=entity1,
            relation=prop,
            object=entity2
        )
        
        result = GenerationResult(
            entity_dict={open_ie_entity: entity1},
            property_dict={"capital": prop},
            retrieved_triples=[triple]
        )
        
        initial_entity_count = len(node_generator_with_websearch.working_memory.entity_dict)
        initial_property_count = len(node_generator_with_websearch.working_memory.property_dict)
        
        node_generator_with_websearch.update_working_memory(result)
        
        # Verify entity_dict was updated
        assert len(node_generator_with_websearch.working_memory.entity_dict) > initial_entity_count
        assert open_ie_entity in node_generator_with_websearch.working_memory.entity_dict
        
        # Verify property_dict was updated
        assert len(node_generator_with_websearch.working_memory.property_dict) > initial_property_count
        assert "capital" in node_generator_with_websearch.working_memory.property_dict
        
        # Verify triple was added to graph
        assert node_generator_with_websearch.working_memory.graph_memory.number_of_edges() > 0
        
        print(f"✓ update_working_memory")
        print(f"  Entities: {len(node_generator_with_websearch.working_memory.entity_dict)}")
        print(f"  Properties: {len(node_generator_with_websearch.working_memory.property_dict)}")
        print(f"  Graph edges: {node_generator_with_websearch.working_memory.graph_memory.number_of_edges()}")
    
    def test_update_working_memory_empty_result(self, node_generator_with_websearch):
        """Test updating working memory with empty result."""
        result = GenerationResult()
        
        initial_entity_count = len(node_generator_with_websearch.working_memory.entity_dict)
        initial_property_count = len(node_generator_with_websearch.working_memory.property_dict)
        
        node_generator_with_websearch.update_working_memory(result)
        
        # Should not change counts
        assert len(node_generator_with_websearch.working_memory.entity_dict) == initial_entity_count
        assert len(node_generator_with_websearch.working_memory.property_dict) == initial_property_count
        
        print(f"✓ update_working_memory (empty result)")
    
    def test_update_working_memory_multiple_triples(self, node_generator_with_websearch):
        """Test updating working memory with multiple triples."""
        from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty, WikiTriple
        
        entity1 = WikidataEntity(qid="Q142", label="France", description="Country")
        entity2 = WikidataEntity(qid="Q90", label="Paris", description="Capital")
        prop1 = WikidataProperty(pid="P36", label="capital", description="Capital")
        prop2 = WikidataProperty(pid="P1082", label="population", description="Population")
        
        triple1 = WikiTriple(subject=entity1, relation=prop1, object=entity2)
        triple2 = WikiTriple(subject=entity2, relation=prop2, object="2.1 million")
        
        result = GenerationResult(retrieved_triples=[triple1, triple2])
        
        initial_edges = node_generator_with_websearch.working_memory.graph_memory.number_of_edges()
        
        node_generator_with_websearch.update_working_memory(result)
        
        # Should add edges for both triples
        new_edges = node_generator_with_websearch.working_memory.graph_memory.number_of_edges()
        assert new_edges >= initial_edges + 2
        
        print(f"✓ update_working_memory (multiple triples)")
        print(f"  Added {new_edges - initial_edges} edges")


# ============================================================================
# Integration Tests
# ============================================================================

class TestNodeGeneratorIntegration:
    """Integration tests for NodeGenerator workflows."""
    
    @pytest.mark.slow
    def test_full_answer_generation_workflow(self, node_generator_with_websearch):
        """Test complete workflow: generate answer and update memory."""
        question = "What is the capital of France?"
        
        # Generate answer
        result = asyncio.run(node_generator_with_websearch.generate_answer(question))
        
        assert isinstance(result, GenerationResult)
        assert len(result.answers) > 0
        
        # Update working memory
        initial_entity_count = len(node_generator_with_websearch.working_memory.entity_dict)
        node_generator_with_websearch.update_working_memory(result)
        
        # Verify memory was updated
        assert len(node_generator_with_websearch.working_memory.entity_dict) >= initial_entity_count
        
        print(f"✓ Full workflow test")
        print(f"  Question: {question}")
        print(f"  Answer: {result.answers[0].answer[:200]}")
        print(f"  Memory entities: {len(node_generator_with_websearch.working_memory.entity_dict)}")
    
    @pytest.mark.slow
    def test_subquestion_to_answer_workflow(self, node_generator_with_websearch):
        """Test workflow: generate subquestion, then answer it."""
        main_question = "Who was the president of France when the Eiffel Tower was built?"
        
        # Generate subquestion
        subquestion, should_direct, _ = asyncio.run(
            node_generator_with_websearch.generate_subquestion(main_question)
        )
        
        print(f"✓ Subquestion workflow")
        print(f"  Main question: {main_question}")
        print(f"  Should direct answer: {should_direct}")
        print(f"  Subquestion: {subquestion}")
        
        # If a subquestion was generated, we could answer it
        if subquestion:
            # This would be done in a real workflow
            print(f"  Would answer subquestion: {subquestion}")
    
    @pytest.mark.slow
    def test_rephrase_to_answer_workflow(self, node_generator_with_websearch):
        """Test workflow: rephrase question, then answer rephrased version."""
        original_question = "What is the capital of France?"
        
        # Generate rephrases
        rephrased_questions, _ = asyncio.run(
            node_generator_with_websearch.generate_rephrase(original_question)
        )
        
        assert len(rephrased_questions) > 0
        
        # Answer one of the rephrased questions
        if rephrased_questions:
            rephrased_result = asyncio.run(
                node_generator_with_websearch.generate_answer(rephrased_questions[0])
            )
            
            assert len(rephrased_result.answers) > 0
            print(f"✓ Rephrase to answer workflow")
            print(f"  Original: {original_question}")
            print(f"  Rephrased: {rephrased_questions[0]}")
            print(f"  Answer: {rephrased_result.answers[0].answer[:200]}")
    
    @pytest.mark.slow
    def test_self_correction_workflow(self, node_generator_with_websearch):
        """Test workflow: generate answer, then self-correct it."""
        question = "What is the capital of France?"
        
        # Generate initial answer
        initial_result = asyncio.run(
            node_generator_with_websearch.generate_answer(question)
        )
        
        assert len(initial_result.answers) > 0
        initial_answer = initial_result.answers[0].answer
        
        # Self-correct the answer
        correction_result = asyncio.run(
            node_generator_with_websearch.generate_self_correction(question, initial_answer)
        )
        
        assert len(correction_result.answers) > 0
        
        print(f"✓ Self-correction workflow")
        print(f"  Question: {question}")
        print(f"  Initial answer: {initial_answer[:200]}")
        print(f"  Corrected status: {correction_result.answers[0].status}")
        print(f"  Refined answer: {correction_result.answers[0].refined_answer[:200]}")
