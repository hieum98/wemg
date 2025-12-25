"""Shared node generation logic for reasoning nodes.
"""
import logging
import os
from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty


import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.procedures.retrieval import explore
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.preprocessing import format_context
from wemg.utils.common import merge_logs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


@dataclass
class GenerationResult:
    answers: List[roles.generator.AnswerGenerationOutput] = None
    retrieved_triples: List = None
    entity_dict: Dict = None
    property_dict: Dict = None
    log_data: Dict = None
    
    def __post_init__(self):
        self.answers = self.answers or []
        self.retrieved_triples = self.retrieved_triples or []
        self.entity_dict = self.entity_dict or {}
        self.property_dict = self.property_dict or {}
        self.log_data = self.log_data or {}


class NodeGenerator:
    """Shared generator for creating reasoning node content.
    
    This class encapsulates the common logic for generating answers,
    subquestions, and other node types used by both MCTS and CoT.
    """
    
    def __init__(
        self,
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory] = None,
        **generation_kwargs
        ) -> None:
        self.llm_agent = llm_agent
        self.retriever_agent = retriever_agent
        self.working_memory = working_memory
        self.interaction_memory = interaction_memory
        self.kwargs = generation_kwargs
    
    async def generate_answer(self, question: str) -> GenerationResult:
        """Generate an answer for a given question using retrieval and LLM.
        
        This is the core answer generation pipeline used by multiple node types.
        """
        in_memory_entities = list[WikidataEntity](set(self.working_memory.entity_dict.values()))
        in_memory_relations = list[WikidataProperty](set(self.working_memory.property_dict.values()))
        
        # Explore external resources
        retrieved_documents, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=self.llm_agent,
            retriever_agent=self.retriever_agent,
            question=question,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=self.kwargs.get('top_k_websearch', 5),
            top_k_entities=self.kwargs.get('top_k_entities', 1),
            top_k_properties=self.kwargs.get('top_k_properties', 1),
            n_hops=self.kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=self.kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=self.interaction_memory
        )

        # Extract information from web search results
        extractor_inputs = [
            roles.extractor.ExtractionInput(question=question, raw_data=data) 
            for data in retrieved_documents
        ]
        extracted_results, extractor_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_inputs,
            interaction_memory=self.interaction_memory,
            n=1
        )
        
        # Flatten and filter relevant information
        all_extractions: List[roles.extractor.ExtractionOutput] = sum(extracted_results, [])
        info_from_websearch = []
        for item in all_extractions:
            if item.relevant_information:
                info_from_websearch.extend(item.relevant_information)

        # Build context
        info_from_kb = [str(t) for t in retrieved_triples]
        all_retrieved_info = info_from_websearch + info_from_kb
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)

        # Generate answer
        qa_input = roles.generator.AnswerGenerationInput(question=question, context=context)
        answers, qa_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=qa_input,
            interaction_memory=self.interaction_memory,
            n=self.kwargs.get('n', 1)
        )
        
        log_data = merge_logs(exploration_log, extractor_log, qa_log)
        
        return GenerationResult(
            answers=answers,
            retrieved_triples=retrieved_triples,
            entity_dict=entity_dict,
            property_dict=property_dict,
            log_data=log_data
        )
    
    async def generate_subquestion(self, user_question: str) -> Tuple[Optional[str], bool, Dict]:
        """Generate a subquestion to advance reasoning.
        
        Returns:
            Tuple of (subquestion, should_direct_answer, log_data)
        """
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        
        subq_input = roles.generator.SubquestionGenerationInput(
            question=user_question, 
            context=context
        )
        subquestions, subq_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.generator.SubquestionGenerator(),
            input_data=subq_input,
            interaction_memory=self.interaction_memory,
            n=self.kwargs.get('n', 1)
        )
        
        # Select an unanswerable subquestion
        unanswerable = [sq for sq in subquestions if not sq.is_answerable]
        subquestion = random.choice(unanswerable).subquestion if unanswerable else None
        
        # Calculate if we should directly answer
        answerable_count = sum(1 for sq in subquestions if sq and sq.is_answerable)
        should_direct_answer = (answerable_count / len(subquestions)) > 0.5 if subquestions else False
        
        return subquestion, should_direct_answer, subq_log
    
    async def generate_rephrase(self, question: str) -> Tuple[List[str], Dict]:
        """Generate rephrased versions of a question.
        
        Returns:
            Tuple of (rephrased_questions, log_data)
        """
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        
        rephrase_input = roles.generator.QuestionRephraserInput(
            context=context, 
            original_question=question
        )
        rephrased, rephrase_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.generator.QuestionRephraser(),
            input_data=rephrase_input,
            interaction_memory=self.interaction_memory,
            n=self.kwargs.get('n', 1)
        )
        rephrased: List[roles.generator.QuestionRephraserOutput]
        rephrased_questions = [r.rephrased_question for r in rephrased]
        return rephrased_questions, rephrase_log
    
    async def generate_self_correction(self, sub_question: str, sub_answer: str) -> GenerationResult:
        """Generate a self-corrected answer for a sub-question.
        
        Returns result with corrected answers.
        """
        step_objective = (
            f"Verify and correct the answer for the sub-question: {sub_question}.\n"
            f"Proposed answer: {sub_answer}"
        )
        
        in_memory_entities = list[WikidataEntity](set(self.working_memory.entity_dict.values()))
        in_memory_relations = list[WikidataProperty](set(self.working_memory.property_dict.values()))
        
        # Explore for verification
        retrieved_docs, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=self.llm_agent,
            retriever_agent=self.retriever_agent,
            question=step_objective,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=self.kwargs.get('top_k_websearch', 5),
            top_k_entities=self.kwargs.get('top_k_entities', 1),
            top_k_properties=self.kwargs.get('top_k_properties', 1),
            n_hops=self.kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=self.kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=self.interaction_memory
        )

        # Extract relevant info
        extractor_inputs = [
            roles.extractor.ExtractionInput(question=sub_question, raw_data=data) 
            for data in retrieved_docs
        ]
        extracted_results, extractor_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_inputs,
            interaction_memory=self.interaction_memory,
            n=1
        )
        
        all_extractions = sum(extracted_results, [])
        info_from_websearch = []
        for item in all_extractions:
            if item.decision == "relevant":
                info_from_websearch.extend(item.information)
        
        # Build context and correct
        info_from_kb = [str(t) for t in retrieved_triples]
        all_retrieved_info = info_from_websearch + info_from_kb
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)

        correction_input = roles.generator.SelfCorrectionInput(
            question=sub_question, 
            proposed_answer=sub_answer, 
            context=context
        )
        corrections, qa_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.generator.SelfCorrector(),
            input_data=correction_input,
            interaction_memory=self.interaction_memory,
            n=self.kwargs.get('n', 1)
        )
        
        log_data = merge_logs(exploration_log, extractor_log, qa_log)
        
        return GenerationResult(
            answers=corrections,
            retrieved_triples=retrieved_triples,
            entity_dict=entity_dict,
            property_dict=property_dict,
            log_data=log_data
        )
    
    async def generate_synthesis(self, user_question: str) -> Tuple[List, Dict]:
        """Generate synthesized reasoning from current context.
        
        Returns:
            Tuple of (synthesis_outputs, log_data)
        """
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        
        reasoner_input = roles.generator.ReasoningSynthesizeInput(
            question=user_question, 
            context=context
        )
        outputs, reasoning_log = await execute_role(
            llm_agent=self.llm_agent,
            role=roles.generator.ReasoningSynthesizer(),
            input_data=reasoner_input,
            interaction_memory=self.interaction_memory,
            n=self.kwargs.get('n', 1)
        )
        
        return outputs, reasoning_log
    
    def update_working_memory(self, result: GenerationResult) -> None:
        """Update working memory with generation results."""
        self.working_memory.entity_dict.update(result.entity_dict)
        self.working_memory.property_dict.update(result.property_dict)
        
        for triple in result.retrieved_triples:
            self.working_memory.add_edge_to_graph_memory(triple)

