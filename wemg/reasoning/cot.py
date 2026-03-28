"""Chain-of-Thought reasoning."""

import logging
from typing import Dict, List, Optional, Tuple, Union

from wemg.llm.roles import SourceType
from wemg.reasoning.nodes import (
    CoTNode, NodeType, NodeState,
    check_correctness, make_final_answer_state, make_subqa_state,
)
from wemg.reasoning.generator import NodeGenerator, GenerationResult, merge_logs
from wemg.reasoning.memory import WorkingMemory, InteractionMemory, log_to_interaction_memory

logger = logging.getLogger(__name__)


def _add_child_to_memory(child: CoTNode, working_memory: WorkingMemory):
    content = str(child.node_state)
    source = SourceType.SYSTEM_PREDICTION
    if child.node_type in [NodeType.SUB_QA_NODE, NodeType.FINAL_ANSWER]:
        working_memory.add_textual_memory(content, source=source)


async def generate_next_step(
    current: CoTNode,
    generator: NodeGenerator,
    working_memory: WorkingMemory,
    interaction_memory=None,
) -> Optional[CoTNode]:
    """Generate the next CoT reasoning step."""
    if current.depth > current.max_depth:
        node, result = await _generate_final_answer(current, generator)
    else:
        node, result = await _generate_subqa(current, generator)
    
    if node and result:
        generator.update_working_memory(result)
        _add_child_to_memory(node, working_memory)
        log_to_interaction_memory(interaction_memory, result.log_data)
    
    return node


async def _generate_final_answer(current: CoTNode, gen: NodeGenerator) -> Tuple[Optional[CoTNode], GenerationResult]:
    should_explore = current.depth < 2
    result = await gen.generate_answer(current.user_question, should_explore=should_explore)
    if not result.answers:
        return None, result
    node = CoTNode(
        node_state=make_final_answer_state(current.user_question, result.answers[0]),
        parent=current,
        max_depth=current.max_depth,
    )
    return node, result


async def _generate_subqa(current: CoTNode, gen: NodeGenerator) -> Tuple[Optional[CoTNode], GenerationResult]:
    subquestions, should_direct, subq_log = await gen.generate_subquestion(current.user_question)
    
    if should_direct or not subquestions:
        node, result = await _generate_final_answer(current, gen)
        if result:
            result.log_data = merge_logs(subq_log, result.log_data)
        return node, result
    
    subquestion = subquestions[0]
    result = await gen.generate_answer(subquestion, should_explore=True)
    
    if not result.answers:
        return None, result
    
    node = CoTNode(
        node_state=make_subqa_state(current.user_question, subquestion, result.answers[0]),
        parent=current,
        max_depth=current.max_depth,
    )
    result.log_data = merge_logs(subq_log, result.log_data)
    return node, result


async def cot_search(
    question: str,
    client,
    retriever,
    wikidata_client,
    reranker=None,
    working_memory: WorkingMemory = None,
    interaction_memory=None,
    max_depth: int = 10,
    correct_answers: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Tuple[Optional[Dict], List[CoTNode], Optional[int]]:
    """Run Chain-of-Thought reasoning. Returns (terminal_content, reasoning_path, pass_at_k)."""
    root = CoTNode(
        node_state=NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question}),
        max_depth=max_depth,
    )
    generator = NodeGenerator(client=client, retriever=retriever, wikidata_client=wikidata_client,
                                   reranker=reranker, working_memory=working_memory,
                                   interaction_memory=interaction_memory, **kwargs)
    
    current = root
    reasoning_path = [root]
    pass_at_k = None

    while not current.is_terminal():        
        next_node = await generate_next_step(current, generator, working_memory, interaction_memory)
        if next_node is None:
            break
        await working_memory.asynchronize_memory(client, question, interaction_memory, reranker=reranker, **kwargs)
        
        current = next_node
        reasoning_path.append(current)
        if pass_at_k is None and correct_answers and current.is_terminal():
            answer = current.node_state.content.get('concise_answer') or current.node_state.content.get('final_answer', '')
            if answer and check_correctness(answer, correct_answers):
                pass_at_k = len(reasoning_path) - 1
    if working_memory is not None:
        await working_memory.afinalize(client, question, interaction_memory, **kwargs)
    terminal_content = current.node_state.content if current.is_terminal() else None
    return terminal_content, reasoning_path, pass_at_k


def cot_get_answer(terminal_content: Optional[Dict], reasoning_path: List[CoTNode]) -> Tuple[str, str]:
    """Extract final answer from CoT result."""
    if terminal_content is None:
        steps = []
        for node in reasoning_path:
            c = node.node_state.content
            if node.node_type == NodeType.SUB_QA_NODE:
                steps.append(f"Q: {c.get('sub_question', 'N/A')}\nA: {c.get('sub_answer', 'N/A')}")
            elif node.node_type == NodeType.FINAL_ANSWER:
                steps.append(f"Final Answer: {c.get('final_answer', 'N/A')}")
        full = '\n'.join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return full, full
    
    full = terminal_content.get('final_answer', 'No answer')
    concise = terminal_content.get('concise_answer', full)
    reasoning = terminal_content.get('reasoning', '')
    return f"Final Answer: {full}\nReasoning: {reasoning}", concise
