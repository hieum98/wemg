"""Chain-of-Thought (CoT) reasoning implementation.

Unlike MCTS which explores multiple branches, CoT follows a single reasoning chain
by sequentially generating subqa nodes until reaching a final answer.
"""
import asyncio
from typing import Dict, List, Optional, Tuple, Union

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
from wemg.runners.procedures.node_generator import NodeGenerator, GenerationResult
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.common import merge_logs, log_to_interaction_memory


class CoTReasoningNode(BaseReasoningNode):
    """A node in the Chain-of-Thought reasoning chain.
    
    Simpler than MCTSReasoningNode - no UCT scores or backpropagation needed
    since CoT follows a single path.
    """
    
    def __init__(
        self,
        node_state: NodeState,
        parent: Optional['CoTReasoningNode'] = None,
        children: Optional[Tuple['CoTReasoningNode']] = None,
        max_depth: int = 10,
    ):
        super().__init__(node_state=node_state, parent=parent, children=children, max_depth=max_depth)
        self.children: Tuple["CoTReasoningNode"]
        self.parent: Optional["CoTReasoningNode"]

    def generate_children(self) -> List["CoTReasoningNode"]:
        """Not used directly in CoT - use generate_next_step instead."""
        raise NotImplementedError("CoT uses generate_next_step() instead")

    def generate_next_step(
        self,
        llm_agent: BaseLLMAgent, 
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory, 
        interaction_memory: Optional[InteractionMemory] = None,
        **kwargs
    ) -> Optional["CoTReasoningNode"]:
        """Generate the next reasoning step.
        
        Returns:
            The next reasoning node, or None if no more steps needed.
        """
        generator = NodeGenerator(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **kwargs
        )
        
        if self.depth > self.max_depth:
            node, result = asyncio.run(self._generate_final_answer(generator))
            if node:
                # Update working memory after generation
                generator.update_working_memory(result)
                node.parent = self
                self._add_child_to_memory(node, working_memory)
                log_to_interaction_memory(interaction_memory, result.log_data)
            return node
        
        node, result = asyncio.run(self._generate_subqa(generator))
        if node:
            # Update working memory after generation
            generator.update_working_memory(result)
            node.parent = self
            self._add_child_to_memory(node, working_memory)
            log_to_interaction_memory(interaction_memory, result.log_data)
        return node
    
    async def _generate_final_answer(
        self, 
        generator: NodeGenerator
    ) -> Tuple[Optional["CoTReasoningNode"], GenerationResult]:
        """Generate final answer node."""
        if self.depth < 2:
            # Explore external resources for the first two levels
            should_explore = True
        else:
            # Use only what's saved in working memory
            should_explore = False
        
        result: GenerationResult = await generator.generate_answer(self.user_question, should_explore=should_explore)
        
        if not result.answers:
            return None, result
        
        answer = result.answers[0]
        state = NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={
                "user_question": self.user_question,
                "final_answer": answer.answer,
                "concise_answer": answer.concise_answer,
                "reasoning": answer.reasoning,
            }
        )
        
        node = CoTReasoningNode(node_state=state, max_depth=self.max_depth)
        return node, result

    async def _generate_subqa(
        self, 
        generator: NodeGenerator
    ) -> Tuple[Optional["CoTReasoningNode"], GenerationResult]:
        """Generate sub-question/answer node.
        
        Returns:
            Tuple of (node, GenerationResult). Node is None if no answer generated.
        """
        # Generate subquestion
        subquestion, should_direct_answer, subq_log = await generator.generate_subquestion(
            self.user_question
        )
        
        # Check if we should generate final answer instead
        if should_direct_answer or not subquestion:
            node, result = await self._generate_final_answer(generator)
            # Merge subquestion log with answer log
            merged_log = merge_logs(subq_log, result.log_data)
            result.log_data = merged_log
            return node, result
        
        # Generate answer for subquestion
        result: GenerationResult = await generator.generate_answer(subquestion, should_explore=True)
        
        if not result.answers:
            return None, result
        
        answer = result.answers[0]
        # Create subqa node
        state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={
                'user_question': self.user_question,
                'sub_question': subquestion,
                'sub_answer': answer.answer,
                'reasoning': answer.reasoning,
            }
        )
        
        node = CoTReasoningNode(node_state=state, max_depth=self.max_depth)
        
        # Merge subquestion log with answer log
        merged_log = merge_logs(subq_log, result.log_data)
        result.log_data = merged_log
        
        return node, result
    
    def _add_child_to_memory(self, child: "CoTReasoningNode", working_memory: WorkingMemory) -> None:
        """Add child node content to working memory."""
        content = child.node_state.content
        source = roles.extractor.SourceType.SYSTEM_PREDICTION
        
        if child.node_type == NodeType.SUB_QA_NODE:
            sub_q, sub_a = content.get('sub_question', ''), content.get('sub_answer', '')
            if sub_q and sub_a:
                working_memory.add_textual_memory(f"Q: {sub_q}; A: {sub_a}", source=source)
        elif child.node_type == NodeType.SYNTHESIS_NODE:
            synthesis = content.get('synthesized_reasoning', '')
            if synthesis:
                working_memory.add_textual_memory(synthesis, source=source)


def cot_search(
    question: str,
    llm_agent: BaseLLMAgent,
    retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
    working_memory: WorkingMemory,
    interaction_memory: Optional[InteractionMemory] = None,
    max_depth: int = 10,
    **kwargs
) -> Tuple[Optional[Dict], List[CoTReasoningNode]]:
    """Perform Chain-of-Thought reasoning."""
    # Create root node
    root_state = NodeState(
        node_type=NodeType.USER_QUESTION, 
        content={'user_question': question}
    )
    root = CoTReasoningNode(node_state=root_state, max_depth=max_depth)
    
    current = root
    reasoning_path: List[CoTReasoningNode] = [root]
    
    while not current.is_terminal():
        next_node = current.generate_next_step(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **kwargs
        )
        
        # Sync working memory
        working_memory.synchronize_memory(llm_agent, question, interaction_memory)
        
        if next_node is None:
            break
        
        current = next_node
        reasoning_path.append(current)
    
    terminal_content = current.node_state.content if current.is_terminal() else None
    return terminal_content, reasoning_path


def cot_get_answer(
    terminal_content: Optional[Dict],
    reasoning_path: List[CoTReasoningNode]
) -> Tuple[str, str]:
    """Extract the final answer from CoT reasoning result."""
    if terminal_content is None:
        # Build answer from reasoning path
        steps = []
        for node in reasoning_path:
            content = node.node_state.content
            if node.node_type == NodeType.SUB_QA_NODE:
                sub_q = content.get('sub_question', 'N/A')
                sub_a = content.get('sub_answer', 'N/A')
                steps.append(f"Q: {sub_q}\nA: {sub_a}")
            elif node.node_type == NodeType.FINAL_ANSWER:
                steps.append(f"Final Answer: {content.get('final_answer', 'N/A')}")
        
        full_answer = '\n'.join(f"{i}. {step}" for i, step in enumerate(steps, 1))
        return full_answer, full_answer
    
    full_answer = terminal_content.get('final_answer', 'No answer')
    concise_answer = terminal_content.get('concise_answer', full_answer)
    reasoning = terminal_content.get('reasoning', '')
    return f"Final Answer: {full_answer}\nReasoning: {reasoning}", concise_answer
