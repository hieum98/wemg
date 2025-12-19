"""Chain-of-Thought (CoT) reasoning implementation.

Unlike MCTS which explores multiple branches, CoT follows a single reasoning chain
by sequentially generating subqa nodes until reaching a final answer.
"""
import asyncio
import random
from typing import Dict, List, Optional, Tuple
from pyparsing import Union

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
from wemg.runners.procerduces.base_role_excercution import execute_role
from wemg.runners.procerduces.retrieval import explore
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.preprocessing import format_context


class CoTReasoningNode(BaseReasoningNode):
    """A node in the Chain-of-Thought reasoning chain.
    
    Simpler than MCTSReasoningNode - no UCT scores or backpropagation needed
    since CoT follows a single path.
    """
    
    def __init__(
            self,
            node_state: NodeState,
            parent: Optional['CoTReasoningNode'] = None,
            children: Optional[list['CoTReasoningNode']] = None,
            max_depth: int = 10,
    ):
        super().__init__(node_state=node_state, parent=parent, children=children, max_depth=max_depth)
        self.children: List["CoTReasoningNode"]
        self.parent: Optional["CoTReasoningNode"]
        self.node_state = node_state

    def generate_next_step(
            self,
            llm_agent: BaseLLMAgent, 
            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
            working_memory: WorkingMemory, 
            interaction_memory: Optional[InteractionMemory] = None,
            **node_generation_kwargs
            ) -> Optional["CoTReasoningNode"]:
        """Generate the next reasoning step (subqa node) and update working memory.
        
        This method generates a single subqa node to continue the reasoning chain.
        If the question is answerable, it generates a final answer node instead.
        
        Args:
            llm_agent: The language model agent
            retriever_agent: The retriever agent for external knowledge
            working_memory: The working memory to update (modified in place)
            interaction_memory: Optional interaction memory for logging
            **node_generation_kwargs: Additional arguments for node generation
            
        Returns:
            The next reasoning node, or None if no more steps needed
        """
        if self.depth > self.max_depth:
            # Force final answer generation
            node = asyncio.run(self._generate_final_answer(
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                **node_generation_kwargs
            ))
            return node
        
        # Generate next subqa step
        node = asyncio.run(self._generate_subqa(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **node_generation_kwargs
        ))
        
        return node
    
    async def _generate_answer(self, 
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            question: str,
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ):
        """Generate answer given question and memory."""
        in_memory_entities = list(set(working_memory.entity_dict.values()))
        in_memory_relations = list(set(working_memory.property_dict.values()))
        
        # Explore external resources
        retrieved_documents, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=question,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=node_generation_kwargs.get('top_k_websearch', 5),
            top_k_entities=node_generation_kwargs.get('top_k_entities', 1),
            top_k_properties=node_generation_kwargs.get('top_k_properties', 1),
            n_hops=node_generation_kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=node_generation_kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=interaction_memory
        )

        # Extract information from web search results
        extractor_input = [roles.extractor.ExtractionInput(question=question, raw_data=data) for data in retrieved_documents]
        extracted_info_from_websearch, extractor_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_input,
            interaction_memory=interaction_memory,
            n=1
        )
        all_extraction: List[roles.extractor.ExtractionOutput] = sum([], extracted_info_from_websearch)
        info_from_websearch = []
        for item in all_extraction:
            if item.decision == "relevant":
                info_from_websearch.extend(item.information)

        # Build context
        info_from_kb = [str(t) for t in retrieved_triples]
        all_retrieved_info = info_from_websearch + info_from_kb
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)

        qa_input = roles.generator.AnswerGenerationInput(question=question, context=context)
        answers, qa_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=qa_input,
            interaction_memory=interaction_memory,
            n=node_generation_kwargs.get('n', 1)
        )
        answers: List[roles.generator.AnswerGenerationOutput]

        # Update working memory
        working_memory.entity_dict.update(entity_dict)
        working_memory.property_dict.update(property_dict)
        for triple in retrieved_triples:
            working_memory.add_edge_to_graph_memory(triple)

        # Process log
        all_log_keys = set(list(qa_log.keys()) + list(extractor_log.keys()) + list(exploration_log.keys()))
        to_log_data = {key: qa_log.get(key, []) + extractor_log.get(key, []) + exploration_log.get(key, []) \
                       for key in all_log_keys}
        
        # Log to interaction memory
        if interaction_memory and to_log_data:
            for key, value in to_log_data.items():
                model_input, model_output = zip(*value)
                interaction_memory.log_turn(
                    role=key,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )
        
        return answers[0] if answers else None
    
    async def _generate_final_answer(self, 
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ) -> Optional["CoTReasoningNode"]:
        """Generate final answer node."""
        answer = await self._generate_answer(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=self.user_question,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **node_generation_kwargs
        )
        
        if not answer:
            return None
            
        node_state = NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={
                "user_question": self.user_question,
                "final_answer": answer.answer,
                "concise_answer": answer.concise_answer,
                "reasoning": answer.reasoning,
            }
        )
        node = CoTReasoningNode(node_state=node_state, max_depth=self.max_depth)
        node.parent = self
        return node

    async def _generate_subqa(self,
                        llm_agent: BaseLLMAgent, 
                        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                        working_memory: WorkingMemory, 
                        interaction_memory: Optional[InteractionMemory] = None,
                        **node_generation_kwargs
                        ) -> Optional["CoTReasoningNode"]:
        """Generate sub-qa pair to advance the reasoning process."""
        # Generate subquestion
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory)
        subq_gen_input = roles.generator.SubquestionGenerationInput(question=self.user_question, context=context)
        subquestions, subq_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.SubquestionGenerator(),
            input_data=subq_gen_input,
            interaction_memory=interaction_memory,
            n=node_generation_kwargs.get('n', 1)
        )
        subquestions: List[roles.generator.SubquestionGenerationOutput]
        
        # Log subquestion generation
        if interaction_memory and subq_log:
            for key, value in subq_log.items():
                model_input, model_output = zip(*value)
                interaction_memory.log_turn(
                    role=key,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )
        
        # Check if we should generate final answer
        unanswerable_subquestions = [sq for sq in subquestions if not sq.is_answerable]
        subquestion = None
        if unanswerable_subquestions:
            subquestion = random.choice(unanswerable_subquestions).subquestion
        
        should_direct_answer = [sq.is_answerable for sq in subquestions if sq]
        should_direct_answer = sum(should_direct_answer) / len(should_direct_answer) if should_direct_answer else 0

        if should_direct_answer > 0.5 or not subquestion:
            # Generate final answer
            return await self._generate_final_answer(
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                **node_generation_kwargs
            )
        
        # Generate answer for subquestion
        answer = await self._generate_answer(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=subquestion,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **node_generation_kwargs
        )
        
        if not answer:
            return None
        
        # Create subqa node
        node_state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={
                'user_question': self.user_question,
                'sub_question': subquestion,
                'sub_answer': answer.answer,
            }
        )
        node = CoTReasoningNode(node_state=node_state, max_depth=self.max_depth)
        node.parent = self
        
        # Add to textual memory
        working_memory.add_textual_memory(
            f"Q: {subquestion}; A: {answer.answer}",
            source=roles.extractor.SourceType.SYSTEM_PREDICTION
        )
        
        return node


def cot_search(
        question: str,
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory] = None,
        max_depth: int = 10,
        **node_generation_kwargs
        ):
    """Perform Chain-of-Thought (CoT) reasoning by sequentially generating subqa nodes.
    
    Unlike MCTS which explores multiple branches, CoT follows a single reasoning chain
    by only generating subqa nodes until reaching a final answer. This is simpler and
    more deterministic than MCTS.
    
    Args:
        question: The user's question to answer
        llm_agent: The language model agent
        retriever_agent: The retriever agent for external knowledge
        working_memory: The working memory (updated in place)
        interaction_memory: Optional interaction memory for logging
        max_depth: Maximum reasoning depth before forcing a final answer
        **node_generation_kwargs: Additional arguments for node generation
        
    Returns:
        Tuple of (terminal node content, reasoning path)
    """
    # Create root node from question
    root_node_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question})
    root = CoTReasoningNode(node_state=root_node_state, max_depth=max_depth)
    
    current_node = root
    reasoning_path: List[CoTReasoningNode] = [root]
    
    while not current_node.is_terminal():
        # Generate next reasoning step
        next_node = current_node.generate_next_step(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **node_generation_kwargs
        )
        
        # Sync the working memory after each iteration
        working_memory.synchronize_memory(
            llm_agent=llm_agent,
            question=question,
            interaction_memory=interaction_memory
        )

        if next_node is None:
            # Failed to generate next step
            break
        
        current_node = next_node
        reasoning_path.append(current_node)
    
    terminal_node = current_node if current_node.is_terminal() else None
    return terminal_node.node_state.content if terminal_node else None, reasoning_path


def cot_get_answer(
        terminal_content: Optional[Dict],
        reasoning_path: List[CoTReasoningNode]
        ) -> Tuple[str, str]:
    """Extract the final answer from CoT reasoning result.
    
    Args:
        terminal_content: The terminal node content from cot_search
        reasoning_path: The reasoning path from cot_search
        
    Returns:
        Tuple of (full_answer, concise_answer)
    """
    if terminal_content is None:
        path = []
        for node in reasoning_path:
            if node.node_state.node_type == NodeType.SUB_QA_NODE:
                sub_q = node.node_state.content.get('sub_question', 'N/A')
                sub_a = node.node_state.content.get('sub_answer', 'N/A')
                path.append(f"Q: {sub_q}\nA: {sub_a}")
            elif node.node_state.node_type == NodeType.FINAL_ANSWER:
                final_a = node.node_state.content.get('final_answer', 'N/A')
                path.append(f"Final Answer: {final_a}")
        path = [f"{i}. {step}" for i, step in enumerate(path, 1)]
        full_answer = '\n'.join(path)
        concise_answer = full_answer
        return full_answer, concise_answer
    
    full_answer = terminal_content.get('final_answer', 'No answer')
    concise_answer = terminal_content.get('concise_answer', full_answer)
    
    return full_answer, concise_answer
