"""Monte Carlo Tree Search reasoning."""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

from wemg.llm.roles import (
    execute_role, SourceType,
    ANSWER_GENERATOR, EVALUATOR, CONSENSUS_EVALUATOR,
    FINAL_ANSWER_SYNTHESIZER, MAJORITY_VOTER,
    AnswerGenerationInput, AnswerEvaluationInput,
    ConsensusEvaluationInput, FinalAnswerSynthesisInput, MajorityVoteInput,
)
from wemg.reasoning.nodes import (
    MCTSNode, NodeType, NodeState,
    check_correctness, make_final_answer_state, make_subqa_state,
)
from wemg.reasoning.generator import NodeGenerator, GenerationResult, merge_logs
from wemg.reasoning.memory import GlobalKnowledge, WorkingMemory, log_to_interaction_memory
from wemg.utils.text import format_context

logger = logging.getLogger(__name__)

_NODE_TYPE_PRIOR = {
    # Encourage deeper exploration before committing to terminal answers.
    NodeType.SUB_QA_NODE: 0.60,
    NodeType.SELF_CORRECTED_NODE: 0.50,
    NodeType.SYNTHESIS_NODE: 0.45,
    NodeType.FINAL_ANSWER: 0.30,
    NodeType.REPHRASED_QUESTION_NODE: 0.40,
}


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
        max_score = max(uct_scores)
        best = [c for c, s in zip(node.children, uct_scores) if s == max_score]
        node = random.choice(best)


async def expand(node: MCTSNode, generator: NodeGenerator) -> Tuple[List[MCTSNode], bool]:
    """Expand node by generating children. Returns (children, has_semantic_signal)."""
    if node.is_terminal():
        return [], False
    
    if node.depth > node.max_depth:
        nodes, result, _ = await _generate_final_answer_nodes(node, generator)
        generator.update_working_memory(result)
        log_to_interaction_memory(generator.interaction_memory, result.log_data)
        return nodes, False
    
    strategies = {
        NodeType.USER_QUESTION: [_generate_final_answer_nodes, _generate_subqa_nodes],
        NodeType.SUB_QA_NODE: [_generate_subqa_nodes, _self_correct_nodes],
        NodeType.SELF_CORRECTED_NODE: [_generate_subqa_nodes],
    }
    
    gens = strategies.get(node.node_type, [])
    if not gens:
        return [], False
    
    results = await _gather_expand_generators(node, generator, gens)
    
    all_nodes = []
    has_semantic_signal = False
    all_logs = []
    for nodes, result, has_signal in results:
        all_nodes.extend(nodes)
        generator.update_working_memory(result)
        all_logs.append(result.log_data)
        if has_signal:
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
    nodes = [
        MCTSNode(
            node_state=make_final_answer_state(node.user_question, answer),
            parent=node,
            max_depth=node.max_depth,
            prior=_NODE_TYPE_PRIOR.get(NodeType.FINAL_ANSWER, 1.0),
        )
        for answer in result.answers
    ]
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
            nodes.append(MCTSNode(
                node_state=make_subqa_state(node.user_question, sq, answer),
                parent=node,
                max_depth=node.max_depth,
                prior=_NODE_TYPE_PRIOR.get(NodeType.SUB_QA_NODE, 1.0),
            ))
    
    if last_result is None:
        last_result = GenerationResult()
    last_result.log_data = merge_logs(*all_logs)
    return nodes, last_result, False

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
        parent=node,
        max_depth=node.max_depth,
        prior=_NODE_TYPE_PRIOR.get(NodeType.SELF_CORRECTED_NODE, 1.0),
    ) for c in result.answers]
    return nodes, result, False

def _add_child_to_memory(child: MCTSNode, working_memory: WorkingMemory):
    """Add child node content to working memory."""
    source = SourceType.SYSTEM_PREDICTION
    if child.node_type in (NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE, NodeType.FINAL_ANSWER, NodeType.SYNTHESIS_NODE):
        content = str(child.node_state)
        working_memory.add_textual_memory(content, source=source)


async def _force_terminal_node(
    node: MCTSNode, generator: NodeGenerator,
) -> Tuple[Optional[MCTSNode], GenerationResult]:
    """Generate a FINAL_ANSWER from current memory only (no retrieval).

    Used as a lightweight replacement for full simulation rollouts.
    """
    result = await generator.generate_answer(node.user_question, should_explore=False)
    if not result.answers:
        return None, result
    child = MCTSNode(
        node_state=make_final_answer_state(node.user_question, result.answers[0]),
        parent=node,
        max_depth=node.max_depth,
    )
    return child, result


async def evaluate(
    node: MCTSNode,
    client,
    working_memory: WorkingMemory,
    golden_answer: Optional[str] = None,
    min_graph_nodes_for_consensus: int = 3,
    consensus_weight: float = 0.7,
) -> float:
    """Evaluate terminal node. Returns reward in [-1, 1].

    Always runs answer-quality evaluation (1 LLM call).
    Runs consensus evaluation only when graph memory has at least
    ``min_graph_nodes_for_consensus`` nodes; in that case the node's own
    answer is compared against a freshly generated graph-memory answer.

    When consensus is available the final reward is a weighted blend:
      ``(1 - consensus_weight) * answer_reward + consensus_weight * consensus_reward``

    The consensus signal is self-supervised (text vs. graph agreement) and
    does not rely on parametric knowledge, so it is weighted more heavily by
    default (0.7).  When a golden answer is provided the EVALUATOR signal is
    already strong; callers may lower ``consensus_weight`` accordingly.
    """
    if node.node_type != NodeType.FINAL_ANSWER:
        return -1.0

    question = node.user_question
    node_answer = node.node_state.content.get('final_answer', '')

    eval_input = AnswerEvaluationInput(
        user_question=question,
        system_answer=node_answer,
        correct_answer=golden_answer or node.golden_answer or "Not available",
    )
    try:
        eval_results, _ = await execute_role(
            client=client, role=EVALUATOR, input_data=eval_input, n=1,
        )
        answer_reward = (eval_results[0].rating - 5) / 5.0
    except Exception:
        answer_reward = 0.0

    graph_node_count = working_memory.graph_memory.number_of_nodes()
    if graph_node_count < min_graph_nodes_for_consensus:
        return answer_reward

    graph_memory_text = working_memory.format_graph_memory()
    graph_qa = AnswerGenerationInput(
        question=question,
        context=format_context(memory=graph_memory_text),
    )
    try:
        graph_answers, _ = await execute_role(
            client=client, role=ANSWER_GENERATOR, input_data=graph_qa, n=1,
        )
        graph_answer = (
            f"Answer: {graph_answers[0].answer}\n"
            f"Reasoning: {graph_answers[0].reasoning}"
        )
        consensus_input = ConsensusEvaluationInput(
            question=question,
            candidate_answers=[node_answer, graph_answer],
        )
        consensus_results, _ = await execute_role(
            client=client, role=CONSENSUS_EVALUATOR,
            input_data=consensus_input, n=1,
        )
        consensus_reward = (consensus_results[0].rating - 5) / 5.0
    except Exception:
        consensus_reward = 0.0

    answer_weight = 1.0 - consensus_weight
    return answer_weight * answer_reward + consensus_weight * consensus_reward


async def mcts_search(
    question: str,
    client,
    retriever,
    wikidata_client,
    reranker=None,
    working_memory: WorkingMemory = None,
    interaction_memory=None,
    global_knowledge: Optional[GlobalKnowledge] = None,
    num_iterations: int = 10,
    exploration_weight: float = 2.0,
    max_tree_depth: int = 10,
    golden_answer: Optional[str] = None,
    early_termination_enabled: bool = True,
    min_iterations: int = 3,
    high_confidence_threshold: float = 0.9,
    convergence_patience: int = 3,
    semantic_sufficiency_count: int = 2,
    correct_answers: Optional[Union[str, List[str]]] = None,
    absorption_min_reward: float = 0.0,
    absorption_top_k: int = 3,
    global_consolidation_every: int = 3,
    min_graph_nodes_for_consensus: int = 3,
    consensus_weight: float = 0.7,
    **kwargs,
) -> Tuple[Dict, MCTSNode, Optional[int]]:
    """Run MCTS search with branch-isolated working memory.

    Each iteration operates on a snapshot of ``working_memory``.  Discoveries
    are promoted to ``global_knowledge`` only when the branch reward exceeds
    ``absorption_min_reward``.

    Simulation has been replaced by a lightweight force-terminal step that
    generates a FINAL_ANSWER from memory only (no retrieval) when the
    expanded child is not already terminal.
    """
    root = MCTSNode(
        node_state=NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question}),
        max_depth=max_tree_depth,
    )
    generator = NodeGenerator(client=client, retriever=retriever, wikidata_client=wikidata_client,
                                   reranker=reranker, working_memory=working_memory,
                                   interaction_memory=interaction_memory, **kwargs)
    
    best_node = None
    best_reward = -float('inf')
    all_branch_deltas: List[Tuple] = []  # (MemoryDelta, reward) for top-k fallback
    absorbed_any = False
    last_consolidated_iteration = -1
    semantic_signals = 0
    no_improvement = 0
    termination_reason = None
    pass_at_k = None

    def extract_answer(node):
        if node.node_type == NodeType.FINAL_ANSWER:
            return node.node_state.content.get('concise_answer') or node.node_state.content.get('final_answer', '')
        return None
    
    for iteration in range(num_iterations):
        branch_memory = working_memory.snapshot()
        generator.working_memory = branch_memory

        path = select(root, exploration_weight)
        selected = path[-1]
        
        has_signal = False
        if not selected.is_terminal():
            expand_target = selected
        elif selected.visits > 0 and len(path) >= 2:
            # Already-visited terminal: re-expand parent to generate new sibling
            # branches instead of re-evaluating the same node again.
            expand_target = path[-2]
        else:
            expand_target = None  # brand-new terminal (visits == 0), evaluate normally

        if expand_target is not None:
            children, has_signal = await expand(expand_target, generator)
            if children:
                ucb_scores = [c.upper_confidence_bound(exploration_weight) for c in children]
                max_score = max(ucb_scores)
                best = [c for c, s in zip(children, ucb_scores) if s == max_score]
                selected = random.choice(best)
                for child in children:
                    _add_child_to_memory(child, branch_memory)
                if has_signal:
                    semantic_signals += 1

        if not selected.is_terminal():
            terminal, force_result = await _force_terminal_node(selected, generator)
            if terminal is not None:
                generator.update_working_memory(force_result)
                _add_child_to_memory(terminal, branch_memory)
                log_to_interaction_memory(generator.interaction_memory, force_result.log_data)
            else:
                terminal = selected
        else:
            terminal = selected

        await branch_memory.asynchronize_memory(client, question, interaction_memory, reranker=reranker, **kwargs)

        is_new_terminal = terminal.visits == 0
        reward = await evaluate(
            terminal, client, branch_memory, golden_answer,
            min_graph_nodes_for_consensus=min_graph_nodes_for_consensus,
            consensus_weight=consensus_weight,
        )
        terminal.backpropagate(reward)

        if global_knowledge is not None:
            delta = branch_memory.get_delta()
            if global_knowledge.absorb(delta, reward, min_reward=absorption_min_reward):
                absorbed_any = True
            if delta.new_textual_items or delta.new_triples:
                all_branch_deltas.append((delta, reward))
            if (iteration + 1) % global_consolidation_every == 0:
                await global_knowledge.aconsolidate_if_needed(client, question, interaction_memory, **kwargs)
                last_consolidated_iteration = iteration

        generator.working_memory = working_memory

        if terminal.is_terminal():
            if pass_at_k is None and correct_answers:
                answer = extract_answer(terminal)
                if answer and check_correctness(answer, correct_answers):
                    pass_at_k = iteration + 1
            if reward > best_reward:
                best_reward = reward
                best_node = terminal
                no_improvement = 0
            elif is_new_terminal:
                # Only count genuine new explorations as no-improvement;
                # re-evaluating an already-visited terminal is just MCTS sampling noise.
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

    if global_knowledge is not None:
        # Fallback: if absorption_min_reward was too strict and nothing got in,
        # unconditionally absorb the top-k branches by reward so synthesis has context.
        if not absorbed_any and all_branch_deltas:
            top_k = sorted(all_branch_deltas, key=lambda x: x[1], reverse=True)[:absorption_top_k]
            for delta, reward in top_k:
                global_knowledge.absorb(delta, reward, min_reward=-float('inf'))
            last_consolidated_iteration = -1  # force final consolidation below
        # Final consolidation: run if the last iteration didn't already trigger it.
        if last_consolidated_iteration != iteration:
            await global_knowledge.aconsolidate_if_needed(client, question, interaction_memory, **kwargs)
        await global_knowledge.afinalize(client, question, interaction_memory, **kwargs)

    if working_memory is not None:
        await working_memory.afinalize(client, question, interaction_memory, **kwargs)

    return (best_node.node_state.content if best_node else {}), root, pass_at_k


async def get_answer(
    root: MCTSNode,
    client,
    interaction_memory=None,
    working_memory: Optional[WorkingMemory] = None,
    global_knowledge: Optional[GlobalKnowledge] = None,
) -> Tuple[str, str]:
    """Extract final answer from MCTS tree via synthesis or majority vote.

    Uses ``global_knowledge`` as synthesis context when available, since it
    accumulates reward-filtered discoveries across all MCTS iterations.
    Falls back to ``working_memory`` when ``global_knowledge`` is not provided.
    """
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
    scores = [n.value / n.visits if n.visits > 0 else 0.0 for n in terminals]
    question = root.user_question
    memory_source = global_knowledge if global_knowledge is not None else working_memory
    if memory_source:
        textual = memory_source.format_textual_memory()
        graph = memory_source.format_graph_memory()
        context = f"{textual}\n\n{graph}".strip()
    else:
        context = ""

    try:
        synth_input = FinalAnswerSynthesisInput(question=question, candidate_answers=answers, candidate_scores=scores, context=context)
        results, log = await execute_role(client=client, role=FINAL_ANSWER_SYNTHESIZER, input_data=synth_input, interaction_memory=interaction_memory, n=1)
        if results:
            log_to_interaction_memory(interaction_memory, log)
            return f"Final Answer: {results[0].final_answer}\nReasoning: {results[0].reasoning}", results[0].concise_answer
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")

    vote_input = MajorityVoteInput(question=question, answers=answers)
    results, log = await execute_role(client=client, role=MAJORITY_VOTER, input_data=vote_input, interaction_memory=interaction_memory, n=1)
    if results:
        log_to_interaction_memory(interaction_memory, log)
        return f"Final Answer: {results[0].final_answer}\nReasoning: {results[0].reasoning}", results[0].concise_answer

    return "Unable to determine final answer.", "Unable to determine"
