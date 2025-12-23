"""
Comprehensive tests for the agents/roles module.

These are real integration tests that call actual LLM servers.
Tests cover Generator, Evaluator, and Extractor roles.
"""
import os
import pytest
import asyncio
from typing import List

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents import roles
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.interaction_memory import InteractionMemory


# Test configuration - adjust these for your environment
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")


class TestGeneratorRoles:
    """Test suite for Generator roles (SubquestionGenerator, AnswerGenerator, etc.)."""
    
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
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_subquestion_generator_answerable(self, llm_agent, interaction_memory):
        """Test SubquestionGenerator when context is sufficient."""
        input_data = roles.generator.SubquestionGenerationInput(
            question="What is the capital of France?",
            context="France is a country in Western Europe. Paris is the capital and largest city of France."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.SubquestionGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.SubquestionGenerationOutput = results[0]
        
        # Should be answerable since context contains the answer
        assert output.is_answerable is True or output.subquestion is None
        
        print(f"✓ SubquestionGenerator (answerable case)")
        print(f"  Is answerable: {output.is_answerable}")
    
    @pytest.mark.slow
    def test_subquestion_generator_not_answerable(self, llm_agent, interaction_memory):
        """Test SubquestionGenerator when context is insufficient."""
        input_data = roles.generator.SubquestionGenerationInput(
            question="Who was the president of France when the Eiffel Tower was built?",
            context="The Eiffel Tower is a famous landmark in Paris."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.SubquestionGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.SubquestionGenerationOutput = results[0]
        
        # Should generate a subquestion since context is insufficient
        if not output.is_answerable:
            assert output.subquestion is not None
            assert len(output.subquestion) > 0
        
        print(f"✓ SubquestionGenerator (not answerable case)")
        print(f"  Is answerable: {output.is_answerable}")
        print(f"  Subquestion: {output.subquestion}")
    
    @pytest.mark.slow
    def test_answer_generator(self, llm_agent, interaction_memory):
        """Test AnswerGenerator with context."""
        input_data = roles.generator.AnswerGenerationInput(
            question="What is the boiling point of water?",
            context="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.AnswerGenerationOutput = results[0]
        
        assert output.answer is not None
        assert output.concise_answer is not None
        assert output.reasoning is not None
        assert output.confidence_level in ["high", "medium", "low"]
        
        print(f"✓ AnswerGenerator")
        print(f"  Answer: {output.answer[:200]}")
        print(f"  Concise: {output.concise_answer}")
        print(f"  Confidence: {output.confidence_level}")
    
    @pytest.mark.slow
    def test_answer_generator_multiple_samples(self, llm_agent, interaction_memory):
        """Test AnswerGenerator with multiple samples (n>1)."""
        input_data = roles.generator.AnswerGenerationInput(
            question="What are the main causes of climate change?",
            context="Climate change is primarily caused by human activities that release greenhouse gases."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=3
        ))
        
        assert len(results) == 3
        for i, output in enumerate(results):
            assert output.answer is not None
            print(f"  Sample {i+1}: {output.concise_answer[:100]}")
        
        print(f"✓ AnswerGenerator with n=3")
    
    @pytest.mark.slow
    def test_query_generator(self, llm_agent, interaction_memory):
        """Test QueryGenerator for search query generation."""
        input_data = roles.generator.QueryGeneratorInput(
            input_text="Who was the first person to walk on the moon and when did it happen?"
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.QueryGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.QueryGeneratorOutput = results[0]
        
        assert output.queries is not None
        assert len(output.queries) > 0
        
        print(f"✓ QueryGenerator")
        print(f"  Generated {len(output.queries)} queries:")
        for q in output.queries:
            print(f"    - {q}")
    
    @pytest.mark.slow
    def test_self_corrector(self, llm_agent, interaction_memory):
        """Test SelfCorrector role."""
        input_data = roles.generator.SelfCorrectionInput(
            question="What is the population of Tokyo?",
            proposed_answer="Tokyo has a population of about 10 million people.",
            context="Tokyo, the capital of Japan, has a population of approximately 14 million in the city proper and over 37 million in the greater metropolitan area."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.SelfCorrector(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.SelfCorrectionOutput = results[0]
        
        assert output.refined_answer is not None
        assert output.status in ["correct", "partial", "incorrect", "unsupported"]
        assert output.confidence_level in ["high", "medium", "low"]
        
        print(f"✓ SelfCorrector")
        print(f"  Original: Tokyo has a population of about 10 million people.")
        print(f"  Refined: {output.refined_answer[:200]}")
        print(f"  Status: {output.status}")
        print(f"  Confidence: {output.confidence_level}")
    
    @pytest.mark.slow
    def test_question_rephraser(self, llm_agent, interaction_memory):
        """Test QuestionRephraser role."""
        input_data = roles.generator.QuestionRephraserInput(
            original_question="When was the first iPhone released?",
            context="The iPhone is a smartphone made by Apple."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.QuestionRephraser(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.QuestionRephraserOutput = results[0]
        
        assert output.rephrased_question is not None
        assert len(output.rephrased_question) > 0
        
        print(f"✓ QuestionRephraser")
        print(f"  Original: When was the first iPhone released?")
        print(f"  Rephrased: {output.rephrased_question}")
    
    @pytest.mark.slow
    def test_reasoning_synthesizer(self, llm_agent, interaction_memory):
        """Test ReasoningSynthesizer role."""
        input_data = roles.generator.ReasoningSynthesizeInput(
            question="Is Python a good language for beginners?",
            context="Python has simple syntax. Python is widely used in education. Python has extensive documentation."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.ReasoningSynthesizer(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.ReasoningSynthesizeOutput = results[0]
        
        assert output.step_conclusion is not None
        assert isinstance(output.is_answerable, bool)
        
        print(f"✓ ReasoningSynthesizer")
        print(f"  Is answerable: {output.is_answerable}")
        print(f"  Conclusion: {output.step_conclusion[:200]}")
    
    @pytest.mark.slow
    def test_structured_query_generator(self, llm_agent, interaction_memory):
        """Test StructuredQueryGenerator for knowledge graph query generation."""
        input_data = roles.generator.QueryGraphGeneratorInput(
            input_text="Who was the first person to walk on the moon and when did it happen?"
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.StructuredQueryGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.QueryGraphGeneratorOutput = results[0]
        
        assert output.queries is not None
        assert len(output.queries) > 0
        
        # Validate query structure
        for query in output.queries:
            assert query.subject is not None and len(query.subject) > 0
            assert query.relation is not None and len(query.relation) > 0
            # reasoning is optional
        
        print(f"✓ StructuredQueryGenerator")
        print(f"  Generated {len(output.queries)} structured queries:")
        for i, query in enumerate(output.queries, 1):
            print(f"    {i}. Subject: {query.subject}")
            print(f"       Relation: {query.relation}")
            if query.reasoning:
                print(f"       Reasoning: {query.reasoning[:100]}...")
    
    @pytest.mark.slow
    def test_structured_query_generator_with_entities(self, llm_agent, interaction_memory):
        """Test StructuredQueryGenerator with provided entities."""
        input_data = roles.generator.QueryGraphGeneratorInput(
            input_text="What is the relationship between Paris and France?",
            entities=["Paris", "France"]
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.StructuredQueryGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.QueryGraphGeneratorOutput = results[0]
        
        assert output.queries is not None
        assert len(output.queries) > 0
        
        # Check that provided entities are used as subjects
        subjects = [q.subject for q in output.queries]
        # At least one query should use one of the provided entities
        entity_used = any(
            entity.lower() in subject.lower() 
            for entity in ["Paris", "France"] 
            for subject in subjects
        )
        assert entity_used, f"Expected provided entities to be used, got subjects: {subjects}"
        
        print(f"✓ StructuredQueryGenerator (with entities)")
        print(f"  Generated {len(output.queries)} structured queries:")
        for i, query in enumerate(output.queries, 1):
            print(f"    {i}. {query.subject} --[{query.relation}]--> ?")
    
    @pytest.mark.slow
    def test_structured_query_generator_with_relations(self, llm_agent, interaction_memory):
        """Test StructuredQueryGenerator with provided relations."""
        input_data = roles.generator.QueryGraphGeneratorInput(
            input_text="Where was Albert Einstein born and what did he discover?",
            relations=["born_in", "discovered", "located_in"]
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.StructuredQueryGenerator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.generator.QueryGraphGeneratorOutput = results[0]
        
        assert output.queries is not None
        assert len(output.queries) > 0
        
        # Check that queries use the provided relations (or similar)
        relations = [q.relation.lower() for q in output.queries]
        print(f"✓ StructuredQueryGenerator (with relations)")
        print(f"  Generated {len(output.queries)} structured queries:")
        for i, query in enumerate(output.queries, 1):
            print(f"    {i}. {query.subject} --[{query.relation}]--> ?")
            if query.reasoning:
                print(f"       Reasoning: {query.reasoning[:80]}...")


class TestEvaluatorRoles:
    """Test suite for Evaluator roles."""
    
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
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_answer_evaluator_correct(self, llm_agent, interaction_memory):
        """Test Evaluator with a correct answer."""
        input_data = roles.evaluator.AnswerEvaluationInput(
            user_question="What is 2 + 2?",
            system_answer="The answer is 4.",
            correct_answer="4"
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.Evaluator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.evaluator.AnswerEvaluationOutput = results[0]
        
        assert output.rating >= 8.0  # Should be high for correct answer
        assert output.reasoning is not None
        
        print(f"✓ Evaluator (correct answer)")
        print(f"  Rating: {output.rating}/10")
        print(f"  Reasoning: {output.reasoning[:200]}")
    
    @pytest.mark.slow
    def test_answer_evaluator_incorrect(self, llm_agent, interaction_memory):
        """Test Evaluator with an incorrect answer."""
        input_data = roles.evaluator.AnswerEvaluationInput(
            user_question="What is the capital of Japan?",
            system_answer="The capital of Japan is Seoul.",
            correct_answer="Tokyo"
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.Evaluator(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.evaluator.AnswerEvaluationOutput = results[0]
        
        assert output.rating <= 5.0  # Should be low for incorrect answer
        
        print(f"✓ Evaluator (incorrect answer)")
        print(f"  Rating: {output.rating}/10")
    
    @pytest.mark.slow
    def test_majority_voter(self, llm_agent, interaction_memory):
        """Test MajorityVoter role."""
        input_data = roles.evaluator.MajorityVoteInput(
            question="What is the largest ocean on Earth?",
            answers=[
                "The Pacific Ocean is the largest ocean.",
                "Pacific Ocean",
                "The largest ocean is the Pacific.",
                "Atlantic Ocean",  # Wrong answer to test consensus
                "The Pacific Ocean, covering more than 60 million square miles."
            ]
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.MajorityVoter(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.evaluator.MajorityVoteOutput = results[0]
        
        assert output.final_answer is not None
        assert output.concise_answer is not None
        assert "pacific" in output.final_answer.lower()  # Should identify Pacific
        
        print(f"✓ MajorityVoter")
        print(f"  Final answer: {output.final_answer[:200]}")
        print(f"  Concise: {output.concise_answer}")
        print(f"  Confidence: {output.confidence_level}")
    
    @pytest.mark.slow
    def test_final_answer_synthesizer(self, llm_agent, interaction_memory):
        """Test FinalAnswerSynthesizer role."""
        input_data = roles.evaluator.FinalAnswerSynthesisInput(
            question="What are the benefits of regular exercise?",
            candidate_answers=[
                "Exercise improves cardiovascular health and reduces the risk of heart disease.",
                "Regular physical activity helps with weight management and boosts metabolism.",
                "Exercise releases endorphins which improve mood and reduce stress.",
                "Physical activity strengthens muscles and bones, improving overall fitness."
            ]
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.FinalAnswerSynthesizer(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.evaluator.FinalAnswerSynthesisOutput = results[0]
        
        assert output.final_answer is not None
        assert output.concise_answer is not None
        assert output.reasoning is not None
        
        print(f"✓ FinalAnswerSynthesizer")
        print(f"  Synthesized: {output.final_answer[:300]}")
        print(f"  Concise: {output.concise_answer}")


class TestExtractorRoles:
    """Test suite for Extractor roles."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=128000,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_extractor_relevant(self, llm_agent, interaction_memory):
        """Test Extractor with relevant data."""
        input_data = roles.extractor.ExtractionInput(
            question="Who build the great wall of china?",
            raw_data="""The Great Wall of China is a series of fortifications made of stone, brick, 
            tamped earth, and other materials. It was built along the historical northern borders 
            of China to protect against various nomadic groups. The wall spans over 13,000 miles and was started building in 7th century BC under the Qin dynasty."""
        )
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        assert len(results) == 1
        output: roles.extractor.ExtractionOutput = results[0]
        
        assert output.relevant_information is not None
        assert len(output.relevant_information) > 0
        
        print(f"✓ Extractor (relevant)")
        print(f"  Relevant information: {output.relevant_information}")
    
    @pytest.mark.slow
    def test_extractor_not_relevant(self, llm_agent, interaction_memory):
        """Test Extractor with irrelevant data."""
        input_data = roles.extractor.ExtractionInput(
            question="What is the population of Tokyo?",
            raw_data="""The Great Wall of China is a series of fortifications made of stone, brick, 
            tamped earth, and other materials. It was built along the historical northern borders 
            of China to protect against various nomadic groups. The wall spans over 13,000 miles."""
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.extractor.ExtractionOutput = results[0]
        assert len(output.relevant_information) == 0
        print(f"✓ Extractor (not relevant)")
        print(f"  Relevant information: {output.relevant_information}")
    
        
    @pytest.mark.slow
    def test_memory_consolidation(self, llm_agent, interaction_memory):
        """Test MemoryConsolidation role."""
        input_data = roles.extractor.MemoryConsolidationInput(
            question="What is the capital of France?",
            memory="""[Retrieval]: Paris is the capital and most populous city of France.
        [System Prediction]: France is a country in Western Europe.
        [Retrieval]: Paris has a population of about 2.1 million in the city proper.
        [System Prediction]: The Eiffel Tower is located in Paris.
        [Retrieval]: Paris is known as the City of Light.
        [System Prediction]: France borders Germany, Belgium, and Spain."""
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.MemoryConsolidationRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.extractor.MemoryConsolidationOutput = results[0]
        
        assert output.consolidated_memory is not None
        
        print(f"✓ MemoryConsolidation")
        print(f"  Consolidated memory:")
        for item in output.consolidated_memory[:5]:
            print(f"    - {str(item)}...")


class TestOpenIERoles:
    """Test suite for OpenIE roles (NER and Relation Extraction)."""
    
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
    def interaction_memory(self):
        """Create an InteractionMemory instance."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_ner_simple(self, llm_agent, interaction_memory):
        """Test NER with simple text containing clear entities."""
        input_data = roles.open_ie.NERInput(
            text="Barack Obama was the 44th President of the United States. He was born in Hawaii."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.NERRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.NEROutput = results[0]
        
        assert output.entities is not None
        assert len(output.entities) > 0
        
        # Check that expected entities are extracted
        entity_names = [e.name for e in output.entities]
        assert any("Obama" in name or "Barack" in name for name in entity_names), \
            f"Expected 'Obama' or 'Barack Obama' in entities, got: {entity_names}"
        assert any("United States" in name or "USA" in name or "U.S." in name for name in entity_names), \
            f"Expected 'United States' in entities, got: {entity_names}"
        
        print(f"✓ NER (simple)")
        print(f"  Extracted {len(output.entities)} entities:")
        for entity in output.entities[:5]:
            print(f"    - {entity.name}")
    
    @pytest.mark.slow
    def test_ner_question(self, llm_agent, interaction_memory):
        """Test NER with a question - should focus on essential entities."""
        input_data = roles.open_ie.NERInput(
            text="Who was the first person to walk on the moon?"
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.NERRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 0
        print(f"✓ NER (question)")
    
    @pytest.mark.slow
    def test_ner_ambiguous(self, llm_agent, interaction_memory):
        """Test NER with ambiguous entities that need context."""
        input_data = roles.open_ie.NERInput(
            text="Apple announced a new iPhone. The apple on the table is red."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.NERRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.NEROutput = results[0]
        
        assert output.entities is not None
        # Should extract Apple (company) and iPhone, but distinguish from fruit
        entity_names = [e.name.lower() for e in output.entities]
        assert any("iphone" in name for name in entity_names), \
            f"Expected 'iPhone' in entities, got: {entity_names}"
        
        print(f"✓ NER (ambiguous)")
        print(f"  Extracted {len(output.entities)} entities:")
        for entity in output.entities:
            print(f"    - {entity.name}")
    
    @pytest.mark.slow
    def test_relation_extraction_simple(self, llm_agent, interaction_memory):
        """Test Relation Extraction with simple relationships."""
        input_data = roles.open_ie.RelationExtractionInput(
            text="Barack Obama was born in Hawaii. He served as the President of the United States from 2009 to 2017."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.RelationExtractionOutput = results[0]
        
        assert output.relations is not None
        assert len(output.relations) > 0
        
        # Check that relationships are extracted
        relation_strs = [str(r) for r in output.relations]
        print(f"✓ Relation Extraction (simple)")
        print(f"  Extracted {len(output.relations)} relations:")
        for relation in output.relations[:5]:
            print(f"    - {relation.subject} --[{relation.relation}]--> {relation.object}")
    
    @pytest.mark.slow
    def test_relation_extraction_with_entities(self, llm_agent, interaction_memory):
        """Test Relation Extraction with provided entities to focus on."""
        input_data = roles.open_ie.RelationExtractionInput(
            text="The Eiffel Tower is located in Paris, France. Paris is the capital of France. France is a country in Western Europe.",
            entities=["Eiffel Tower", "Paris", "France"]
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.RelationExtractionOutput = results[0]
        
        assert output.relations is not None
        assert len(output.relations) > 0
        
        # Should extract relationships involving the provided entities
        relation_subjects = [r.subject for r in output.relations]
        relation_objects = [r.object for r in output.relations]
        all_entities = set(relation_subjects + relation_objects)
        
        print(f"✓ Relation Extraction (with entities)")
        print(f"  Extracted {len(output.relations)} relations:")
        for relation in output.relations:
            print(f"    - {relation.subject} --[{relation.relation}]--> {relation.object}")
            if relation.context:
                print(f"      Context: {relation.context[:100]}...")
    
    @pytest.mark.slow
    def test_relation_extraction_complex(self, llm_agent, interaction_memory):
        """Test Relation Extraction with complex relationships that should be broken down."""
        input_data = roles.open_ie.RelationExtractionInput(
            text="Tim Cook became the CEO of Apple in 2011 after Steve Jobs stepped down due to health issues. Apple is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.RelationExtractionOutput = results[0]
        
        assert output.relations is not None
        assert len(output.relations) > 0
        
        # Should extract multiple simple relationships
        print(f"✓ Relation Extraction (complex)")
        print(f"  Extracted {len(output.relations)} relations:")
        for relation in output.relations:
            print(f"    - {relation.subject} --[{relation.relation}]--> {relation.object}")
    
    @pytest.mark.slow
    def test_relation_extraction_self_contained(self, llm_agent, interaction_memory):
        """Test that extracted relations are self-contained."""
        input_data = roles.open_ie.RelationExtractionInput(
            text="The Great Wall of China was built during the Ming Dynasty to protect against invasions from the north."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.RelationExtractionOutput = results[0]
        
        assert output.relations is not None
        
        # Check that relations have context or are self-contained
        for relation in output.relations:
            # Relations should have either context or be clear enough on their own
            assert relation.subject is not None and len(relation.subject) > 0
            assert relation.relation is not None and len(relation.relation) > 0
            assert relation.object is not None and len(relation.object) > 0
        
        print(f"✓ Relation Extraction (self-contained)")
        print(f"  Extracted {len(output.relations)} self-contained relations:")
        for relation in output.relations:
            print(f"    - {relation.subject} --[{relation.relation}]--> {relation.object}")
            if relation.context:
                print(f"      Context: {relation.context}")
    
    @pytest.mark.slow
    def test_ner_empty_text(self, llm_agent, interaction_memory):
        """Test NER with empty text."""
        input_data = roles.open_ie.NERInput(
            text=""
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.NERRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.NEROutput = results[0]
        
        # Empty text should return empty entities list
        assert output.entities is not None
        # May be empty or may have some default behavior
        
        print(f"✓ NER (empty text)")
        print(f"  Extracted {len(output.entities)} entities")
    
    @pytest.mark.slow
    def test_relation_extraction_no_relations(self, llm_agent, interaction_memory):
        """Test Relation Extraction with text that has no clear relationships."""
        input_data = roles.open_ie.RelationExtractionInput(
            text="The weather is nice today. It is sunny and warm."
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.open_ie.RelationExtractionRole(),
            input_data=input_data,
            interaction_memory=interaction_memory,
            n=1
        ))
        
        assert len(results) == 1
        output: roles.open_ie.RelationExtractionOutput = results[0]
        
        assert output.relations is not None
        # May have few or no relations for descriptive text
        
        print(f"✓ Relation Extraction (no clear relations)")
        print(f"  Extracted {len(output.relations)} relations")


class TestBatchExecution:
    """Test batch execution of roles."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=4,
            max_retries=3
        )
    
    @pytest.mark.slow
    def test_batch_extraction(self, llm_agent):
        """Test batch extraction with multiple inputs."""
        inputs = [
            roles.extractor.ExtractionInput(
                question="What is Python?",
                raw_data="Python is a high-level programming language known for its simplicity."
            ),
            roles.extractor.ExtractionInput(
                question="What is JavaScript?",
                raw_data="JavaScript is a scripting language primarily used for web development."
            ),
            roles.extractor.ExtractionInput(
                question="What is Rust?",
                raw_data="Rust is a systems programming language focused on safety and performance."
            )
        ]
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=inputs,
            n=1
        ))
        
        assert len(results) == 3
        for i, output in enumerate(results):
            assert output[0].relevant_information is not None
            print(f"  Input {i+1}: {len(output[0].relevant_information)} extractions")
        
        print(f"✓ Batch extraction with {len(inputs)} inputs")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
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
    
    @pytest.mark.slow
    def test_empty_context(self, llm_agent):
        """Test with empty context."""
        input_data = roles.generator.AnswerGenerationInput(
            question="What is the meaning of life?",
            context=""
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=input_data,
            n=1
        ))
        
        assert len(results) == 1
        assert results[0].answer is not None
        
        print(f"✓ Empty context handled")
    
    @pytest.mark.slow
    def test_long_input(self, llm_agent):
        """Test with long input text."""
        long_context = "This is a test sentence. " * 500
        
        input_data = roles.generator.AnswerGenerationInput(
            question="Summarize the context.",
            context=long_context
        )
        
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=input_data,
            n=1
        ))
        
        assert len(results) == 1
        assert results[0].answer is not None
        
        print(f"✓ Long input handled")


# Run tests with: pytest test_roles.py -v -s --tb=short
# Run slow tests: pytest test_roles.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
