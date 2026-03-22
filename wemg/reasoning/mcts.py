"""Monte Carlo Tree Search reasoning."""

import asyncio
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union

from wemg.llm.roles import (
    execute_role, SourceType,
    ANSWER_GENERATOR, EVALUATOR, CONSENSUS_EVALUATOR,
    FINAL_ANSWER_SYNTHESIZER, MAJORITY_VOTER,
    AnswerGenerationInput, AnswerEvaluationInput,
    ConsensusEvaluationInput, FinalAnswerSynthesisInput, MajorityVoteInput,
)
from wemg.reasoning.nodes import MCTSNode, NodeType, NodeState
from wemg.reasoning.generator import NodeGenerator, GenerationResult, merge_logs
from wemg.reasoning.memory import WorkingMemory, InteractionMemory, log_to_interaction_memory
from wemg.utils.text import format_context

logger = logging.getLogger(__name__)


def select(root: MCTSNode, exploration_weight: float = 2.0) -> List[MCTSNode]:
    """Select path from root to leaf using UCT."""
    path = []
    node = root
    while True:
        if not node.children or node.is_terminal():
            path.append(node)
            return path
        path.append(node)
        uct_scores = [c.upper_confidence_bound(exploration_weight) for c in node.children]
        node = node.children[uct_scores.index(max(uct_scores))]


def expand(node: MCTSNode, generator: NodeGenerator, is_cot_simulation: bool = False) -> Tuple[List[MCTSNode], bool]:
    """Expand node by generating children. Returns (children, has_semantic_signal)."""
    if node.is_terminal():
        return [], False
    
    if node.depth > node.max_depth:
        nodes, result, _ = asyncio.run(_generate_final_answer_nodes(node, generator))
        generator.update_working_memory(result)
        log_to_interaction_memory(generator.interaction_memory, result.log_data)
        return nodes, False
    
    if is_cot_simulation:
        nodes, result, has_signal = asyncio.run(_generate_subqa_nodes(node, generator))
        generator.update_working_memory(result)
        log_to_interaction_memory(generator.interaction_memory, result.log_data)
        return nodes, has_signal
    
    strategies = {
        NodeType.USER_QUESTION: [_generate_final_answer_nodes, _generate_subqa_nodes],
        NodeType.SUB_QA_NODE: [_generate_subqa_nodes, _self_correct_nodes, _rephrase_nodes, _strengthen_nodes],
        NodeType.REPHRASED_QUESTION_NODE: [_generate_subqa_nodes],
        NodeType.SELF_CORRECTED_NODE: [_generate_subqa_nodes, _strengthen_nodes],
        NodeType.SYNTHESIS_NODE: [_generate_subqa_nodes],
    }
    
    gens = strategies.get(node.node_type, [])
    if not gens:
        return [], False
    
    results = asyncio.run(_gather_expand_generators(node, generator, gens))
    
    all_nodes = []
    has_semantic_signal = False
    all_logs = []
    for nodes, result, has_signal in results:
        all_nodes.extend(nodes)
        generator.update_working_memory(result)
        all_logs.append(result.log_data)
        if has_signal: # If any child node has a semantic signal (answerable or not), consider the question as answered
            has_semantic_signal = True
    
    log_to_interaction_memory(generator.interaction_memory, merge_logs(*all_logs))
    return all_nodes, has_semantic_signal


async def _gather_expand_generators(
    node: MCTSNode, generator: NodeGenerator, gens: List
) -> List[Tuple[List[MCTSNode], GenerationResult, bool]]:
    return list(await asyncio.gather(*[gen(node, generator) for gen in gens]))


async def _generate_final_answer_nodes(node: MCTSNode, gen: NodeGenerator) -> Tuple[List[MCTSNode], GenerationResult, bool]:
    should_explore = node.depth < 2
    result = await gen.generate_answer(node.user_question, should_explore=should_explore)
    nodes = []
    for answer in result.answers:
        state = NodeState(node_type=NodeType.FINAL_ANSWER, content={
            "user_question": node.user_question,
            "final_answer": answer.answer,
            "concise_answer": answer.concise_answer,
            "reasoning": answer.reasoning,
        })
        nodes.append(MCTSNode(node_state=state, parent=node, max_depth=node.max_depth))
    return nodes, result, False

async def _generate_subqa_nodes(node: MCTSNode, gen: NodeGenerator) -> Tuple[List[MCTSNode], GenerationResult, bool]:
    if node.node_type == NodeType.REPHRASED_QUESTION_NODE:
        subquestions = [node.node_state.content['sub_question']]
        subq_log = {}
        should_direct = False
    else:
        subquestions, should_direct, subq_log = await gen.generate_subquestion(node.user_question)
        if should_direct or not subquestions:
            nodes, result, _ = await _generate_final_answer_nodes(node, gen)
            result.log_data = merge_logs(subq_log, result.log_data)
            return nodes, result, True
    
    results = await asyncio.gather(*[gen.generate_answer(sq, should_explore=True) for sq in subquestions])
    nodes = []
    all_logs = [subq_log]
    last_result = None
    for sq, result in zip(subquestions, results):
        last_result = result
        all_logs.append(result.log_data)
        for answer in result.answers:
            state = NodeState(node_type=NodeType.SUB_QA_NODE, content={
                'user_question': node.user_question, 'sub_question': sq,
                'sub_answer': answer.answer, 'reasoning': answer.reasoning,
            })
            nodes.append(MCTSNode(node_state=state, parent=node, max_depth=node.max_depth))
    
    if last_result is None:
        last_result = GenerationResult()
    last_result.log_data = merge_logs(*all_logs)
    return nodes, last_result, False

async def _rephrase_nodes(node: MCTSNode, gen: NodeGenerator) -> Tuple[List[MCTSNode], GenerationResult, bool]:
    question = node.node_state.content.get('sub_question', node.user_question)
    rephrased, log = await gen.generate_rephrase(question)
    nodes = [MCTSNode(
        node_state=NodeState(node_type=NodeType.REPHRASED_QUESTION_NODE, content={'user_question': node.user_question, 'sub_question': rq}),
        parent=node, max_depth=node.max_depth
    ) for rq in rephrased]
    return nodes, GenerationResult(log_data=log), False

async def _self_correct_nodes(node: MCTSNode, gen: NodeGenerator) -> Tuple[List[MCTSNode], GenerationResult, bool]:
    sub_q = node.node_state.content.get('sub_question')
    sub_a = node.node_state.content.get('sub_answer')
    if not sub_q or not sub_a:
        return [], GenerationResult(), False
    result = await gen.generate_self_correction(sub_q, sub_a)
    nodes = [MCTSNode(
        node_state=NodeState(node_type=NodeType.SELF_CORRECTED_NODE, content={
            'user_question': node.user_question, 'sub_question': sub_q, 'sub_answer': c.refined_answer
        }),
        parent=node, max_depth=node.max_depth
    ) for c in result.answers]
    return nodes, result, False

async def _strengthen_nodes(node: MCTSNode, gen: NodeGenerator) -> Tuple[List[MCTSNode], GenerationResult, bool]:
    result = await gen.generate_synthesis(node.user_question)
    nodes = []
    has_signal = False
    for output in result.answers:
        if output.is_answerable:
            state = NodeState(node_type=NodeType.FINAL_ANSWER, content={"user_question": node.user_question, "final_answer": output.step_conclusion})
            has_signal = True
        else:
            state = NodeState(node_type=NodeType.SYNTHESIS_NODE, content={'user_question': node.user_question, 'synthesized_reasoning': output.step_conclusion})
        nodes.append(MCTSNode(node_state=state, parent=node, max_depth=node.max_depth))
    return nodes, result, has_signal


def _add_child_to_memory(child: MCTSNode, working_memory: WorkingMemory):
    """Add child node content to working memory."""
    source = SourceType.SYSTEM_PREDICTION
    if child.node_type in (NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE, NodeType.FINAL_ANSWER, NodeType.SYNTHESIS_NODE):
        content = str(child.node_state)
        working_memory.add_textual_memory(content, source=source)


def simulate(node: MCTSNode, generator: NodeGenerator, max_simulation_depth: int = 5) -> Tuple[MCTSNode, bool]:
    """Simulate rollout from node to terminal."""
    current = node
    any_signal = False
    for _ in range(max_simulation_depth):
        if current.is_terminal():
            break
        children, has_signal = expand(current, generator, is_cot_simulation=True)
        any_signal = any_signal or has_signal
        if not children:
            break
        current = random.choice(children)
    return current, any_signal


def evaluate(node: MCTSNode, client, working_memory: WorkingMemory, golden_answer: Optional[str] = None) -> float:
    """Evaluate terminal node. Returns reward in [-1, 1].
    
    Reward = average of (answer_evaluation_reward, consensus_reward)
    """
    if node.node_type != NodeType.FINAL_ANSWER:
        return -1.0
    
    question = node.user_question
    textual_memory = working_memory.format_textual_memory()
    graph_memory = working_memory.format_graph_memory()
    
    textual_qa = AnswerGenerationInput(question=question, context=format_context(memory=textual_memory))
    graph_qa = AnswerGenerationInput(question=question, context=format_context(memory=graph_memory))
    
    try:
        answers, _ = asyncio.run(execute_role(client=client, role=ANSWER_GENERATOR, input_data=[textual_qa, graph_qa], n=1))
        textual_answer = f"Answer: {answers[0][0].answer}\nReasoning: {answers[0][0].reasoning}"
        graph_answer = f"Answer: {answers[1][0].answer}\nReasoning: {answers[1][0].reasoning}"
        consensus_input = ConsensusEvaluationInput(question=question, candidate_answers=[textual_answer, graph_answer])
        consensus_results, _ = asyncio.run(execute_role(client=client, role=CONSENSUS_EVALUATOR, input_data=consensus_input, n=1))
        consensus_reward = (consensus_results[0].rating - 5) / 5.0
    except Exception:
        consensus_reward = 0.0
    
    eval_input = AnswerEvaluationInput(
        user_question=question,
        system_answer=node.node_state.content.get('final_answer', ''),
        correct_answer=golden_answer or node.golden_answer or "Not available"
    )
    try:
        eval_results, _ = asyncio.run(execute_role(client=client, role=EVALUATOR, input_data=eval_input, n=1))
        answer_reward = (eval_results[0].rating - 5) / 5.0
    except Exception:
        answer_reward = 0.0
    
    return (answer_reward + consensus_reward) / 2.0


def mcts_search(
    question: str,
    client,
    retriever,
    wikidata_client,
    reranker=None,
    working_memory: WorkingMemory = None,
    interaction_memory=None,
    num_iterations: int = 10,
    exploration_weight: float = 2.0,
    max_tree_depth: int = 10,
    max_simulation_depth: int = 5,
    golden_answer: Optional[str] = None,
    early_termination_enabled: bool = True,
    min_iterations: int = 3,
    high_confidence_threshold: float = 0.9,
    convergence_patience: int = 3,
    semantic_sufficiency_count: int = 2,
    correct_answers: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Tuple[Dict, MCTSNode, Optional[int]]:
    """Run MCTS search. Returns (best_content, root_node, pass_at_k)."""
    root = MCTSNode(
        node_state=NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question}),
        max_depth=max_tree_depth,
    )
    generator = NodeGenerator(client=client, retriever=retriever, wikidata_client=wikidata_client,
                                   reranker=reranker, working_memory=working_memory,
                                   interaction_memory=interaction_memory, **kwargs)
    
    best_node = None
    best_reward = -float('inf')
    semantic_signals = 0
    no_improvement = 0
    termination_reason = None
    pass_at_k = None
    
    def check_correctness(predicted, answers):
        if not predicted or not answers:
            return False
        if isinstance(answers, str):
            answers = [answers]
        predicted_lower = predicted.lower()
        return any(a and a.lower() in predicted_lower for a in answers)
    
    def extract_answer(node):
        if node.node_type == NodeType.FINAL_ANSWER:
            return node.node_state.content.get('concise_answer') or node.node_state.content.get('final_answer', '')
        return None
    
    for iteration in range(num_iterations):    
        path = select(root, exploration_weight)
        selected = path[-1]
        
        has_signal = False
        if not selected.is_terminal():
            children, has_signal = expand(selected, generator, is_cot_simulation=False)
            if children:
                selected = random.choice(children)
                for child in children:
                    _add_child_to_memory(child, working_memory)
                if has_signal:
                    semantic_signals += 1
        
        if not selected.is_terminal():
            terminal, sim_signal = simulate(selected, generator, max_simulation_depth)
            if sim_signal:
                semantic_signals += 1
        else:
            terminal = selected
        
        reward = evaluate(terminal, client, working_memory, golden_answer)
        terminal.backpropagate(reward)
        
        working_memory.synchronize_memory(client, question, interaction_memory, reranker=reranker, **kwargs)
        
        if terminal.is_terminal():
            if pass_at_k is None and correct_answers:
                answer = extract_answer(terminal)
                if answer and check_correctness(answer, correct_answers):
                    pass_at_k = iteration + 1
            if reward > best_reward:
                best_reward = reward
                best_node = terminal
                no_improvement = 0
            else:
                no_improvement += 1
        
        if early_termination_enabled and iteration + 1 >= min_iterations:
            if reward >= high_confidence_threshold and terminal.is_terminal():
                termination_reason = f"High confidence (reward={reward:.3f})"
                break
            if semantic_signals >= semantic_sufficiency_count:
                termination_reason = f"Semantic sufficiency ({semantic_signals} signals)"
                break
            if no_improvement >= convergence_patience:
                termination_reason = f"Convergence (no improvement for {no_improvement} iterations)"
                break
    
    if termination_reason:
        logger.info(f"MCTS terminated at iteration {iteration + 1}: {termination_reason}")
    
    return (best_node.node_state.content if best_node else {}), root, pass_at_k


def get_answer(root: MCTSNode, client, interaction_memory=None) -> Tuple[str, str]:
    """Extract final answer from MCTS tree via synthesis or majority vote."""
    terminals = []
    def collect(node):
        if node.is_terminal():
            terminals.append(node)
        for child in node.children:
            collect(child)
    collect(root)
    
    if not terminals:
        return "No final answer found.", "No answer"
    
    answers = [str(n.node_state) for n in terminals]
    question = root.user_question
    
    try:
        synth_input = FinalAnswerSynthesisInput(question=question, candidate_answers=answers)
        results, log = asyncio.run(execute_role(client=client, role=FINAL_ANSWER_SYNTHESIZER, input_data=synth_input, interaction_memory=interaction_memory, n=1))
        if results:
            log_to_interaction_memory(interaction_memory, log)
            return f"Final Answer: {results[0].final_answer}\nReasoning: {results[0].reasoning}", results[0].concise_answer
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")
    
    vote_input = MajorityVoteInput(question=question, answers=answers)
    results, log = asyncio.run(execute_role(client=client, role=MAJORITY_VOTER, input_data=vote_input, interaction_memory=interaction_memory, n=1))
    if results:
        log_to_interaction_memory(interaction_memory, log)
        return f"Final Answer: {results[0].final_answer}\nReasoning: {results[0].reasoning}", results[0].concise_answer
    
    return "Unable to determine final answer.", "Unable to determine"
