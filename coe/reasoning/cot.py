"""Chain-of-Thought reasoning."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

from coe.reasoning.nodes import (
    CoTNode, NodeType, NodeState,
    make_final_answer_state, make_subqa_state, make_subqa_batch_state,
    add_node_content_to_memory,
)
from coe.reasoning.generator import NodeGenerator, GenerationResult, merge_logs
from coe.evaluation.metrics import compute_acc
from coe.reasoning.memory import WorkingMemory, InteractionMemory, log_to_interaction_memory

logger = logging.getLogger(__name__)



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
        node, result = await _generate_subqa_batch(current, generator)
        if node is None:
            if result:
                log_to_interaction_memory(interaction_memory, result.log_data)
            node, result = await _generate_final_answer(current, generator)

    if node and result:
        generator.update_working_memory(result, hop_depth=current.depth + 1)
        add_node_content_to_memory(node, working_memory)
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


async def _generate_subqa_batch(
    current: CoTNode, gen: NodeGenerator
) -> Tuple[Optional[CoTNode], GenerationResult]:
    """Answer all subquestions from one generate_subquestion call in parallel.

    Returns (batch_node, merged_result) on success, or (None, result) as a
    fallback signal — caller will invoke _generate_final_answer instead.
    """
    if current.node_type == NodeType.SUB_QA_BATCH_NODE:
        sub_answers = current.node_state.content.get('sub_answers', [])
        intermediate = '\n'.join(sub_answers) if sub_answers else None
    elif current.node_type == NodeType.SUB_QA_NODE:
        intermediate = current.node_state.content.get('sub_answer')
    else:
        intermediate = None

    subquestions, should_direct, subq_log = await gen.generate_subquestion(
        current.user_question, intermediate_answer=intermediate
    )

    if should_direct or not subquestions:
        return None, GenerationResult(log_data=subq_log)

    # Answer all subquestions in parallel
    results: List[GenerationResult] = await asyncio.gather(
        *[gen.generate_answer(sq, should_explore=True) for sq in subquestions]
    )

    # Filter out failures
    valid_pairs = [(sq, r) for sq, r in zip(subquestions, results) if r.answers]
    if not valid_pairs:
        return None, GenerationResult(log_data=subq_log)

    valid_subqs = [sq for sq, _ in valid_pairs]
    valid_results = [r for _, r in valid_pairs]

    node = CoTNode(
        node_state=make_subqa_batch_state(
            current.user_question, valid_subqs, [r.answers[0] for r in valid_results]
        ),
        parent=current,
        max_depth=current.max_depth,
    )

    merged = GenerationResult(
        answers=[r.answers[0] for r in valid_results],
        retrieved_triples=[t for r in valid_results for t in r.retrieved_triples],
        retrieved_entities=[e for r in valid_results for e in r.retrieved_entities],
        information_items=[item for r in valid_results for item in r.information_items],
        log_data=merge_logs(subq_log, *[r.log_data for r in valid_results]),
    )
    return node, merged


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
    """Run Chain-of-Thought reasoning.

    ``pass_at_k`` is the first step where evaluator Acc is strictly > 0.8
    (equivalent to evaluator rating > 8.0).
    """
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
        next_node = await generate_next_step(current, generator, generator.working_memory, interaction_memory)
        if next_node is None:
            break
        await generator.working_memory.synchronize_memory(client, question, interaction_memory, reranker=reranker, **kwargs)
        
        current = next_node
        reasoning_path.append(current)
        if pass_at_k is None and correct_answers and current.is_terminal():
            answer = current.node_state.content.get('concise_answer') or current.node_state.content.get('final_answer', '')
            if answer:
                acc = await compute_acc(question, answer, correct_answers, client, interaction_memory=interaction_memory)
                if acc > 0.8:
                    pass_at_k = len(reasoning_path) - 1
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
            elif node.node_type == NodeType.SUB_QA_BATCH_NODE:
                pairs = '\n'.join(
                    f"Q: {q}\nA: {a}"
                    for q, a in zip(c.get('sub_questions', []), c.get('sub_answers', []))
                )
                steps.append(pairs)
            elif node.node_type == NodeType.FINAL_ANSWER:
                steps.append(f"Final Answer: {c.get('final_answer', 'N/A')}")
        full = '\n'.join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return full, full
    
    full = terminal_content.get('final_answer', 'No answer')
    concise = terminal_content.get('concise_answer', full)
    return f"Final Answer: {full}", concise
