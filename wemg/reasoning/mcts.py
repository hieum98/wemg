"""Monte Carlo Tree Search reasoning."""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

from wemg.llm.roles import (
    execute_role, SourceType,
    FINAL_ANSWER_SYNTHESIZER, MAJORITY_VOTER,
    VERIFIER,
    FinalAnswerSynthesisInput, MajorityVoteInput,
    AnswerVerificationInput,
)
from wemg.reasoning.nodes import (
    MCTSNode, NodeType, NodeState,
    check_correctness, make_final_answer_state, make_subqa_state,
)
from wemg.reasoning.generator import NodeGenerator, GenerationResult, merge_logs
from wemg.reasoning.memory import WorkingMemory, log_to_interaction_memory

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
        generator.update_working_memory(result, hop_depth=node.depth + 1)
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
        intermediate = node.node_state.content.get('sub_answer') if node.node_type == NodeType.SUB_QA_NODE else None
        subquestions, should_direct, subq_log = await gen.generate_subquestion(node.user_question, intermediate_answer=intermediate)
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
        working_memory.add_textual_memory(content, source=source, hop_depth=child.depth)


async def simulate(
    node: MCTSNode,
    generator: NodeGenerator,
    max_simulation_depth: int = 5,
) -> Tuple[MCTSNode, bool]:
    """n-step CoT rollout from *node* with full retrieval.

    Returns (terminal_node, has_semantic_signal).
    """
    current = node
    has_signal = False
    for _ in range(max_simulation_depth):
        if current.is_terminal():
            break
        if current.depth > current.max_depth:
            nodes, result, _ = await _generate_final_answer_nodes(current, generator)
        else:
            nodes, result, sig = await _generate_subqa_nodes(current, generator)
            if sig:
                has_signal = True
        generator.update_working_memory(result, hop_depth=current.depth + 1)
        log_to_interaction_memory(generator.interaction_memory, result.log_data)
        if not nodes:
            break
        for child in nodes:
            _add_child_to_memory(child, generator.working_memory)
        current = random.choice(nodes)
    return current, has_signal


async def evaluate(
    node: MCTSNode,
    client,
    working_memory: WorkingMemory,
    golden_answer: Optional[str] = None,
) -> float:
    """Evaluate terminal node with 3 parallel VERIFIER calls. Returns reward in [-1, 1].

    Score 1: no context (LLM uses own knowledge)
    Score 2: textual memory context
    Score 3: graph memory context

    reward = (mean_rating - 5.0) / 5.0
    """
    if node.node_type != NodeType.FINAL_ANSWER:
        return -1.0

    question = node.user_question
    node_answer = (
        f"Answer: {node.node_state.content.get('final_answer', '')}\n"
        f"Reasoning: {node.node_state.content.get('reasoning', '')}"
    )
    text_ctx = working_memory.format_textual_memory() or "Not provided"
    graph_ctx = working_memory.format_graph_memory() or "Not provided"
    golden_ctx = f"Golden Answer: {golden_answer}" if golden_answer else "Not provided"

    try:
        (r1, _), (r2, _), (r3, _) = await asyncio.gather(
            execute_role(client, VERIFIER, AnswerVerificationInput(question=question, candidate_answer=node_answer, context=golden_ctx), n=1),
            execute_role(client, VERIFIER, AnswerVerificationInput(question=question, candidate_answer=node_answer, context=text_ctx), n=1),
            execute_role(client, VERIFIER, AnswerVerificationInput(question=question, candidate_answer=node_answer, context=graph_ctx), n=1),
        )
        s1 = r1[0].rating if r1 else 5.0
        s2 = r2[0].rating if r2 else 5.0
        s3 = r3[0].rating if r3 else 5.0
        return ((s1 + s2 + s3) / 3.0 - 5.0) / 5.0
    except Exception:
        return 0.0


async def mcts_search(
    question: str,
    client,
    retriever,
    wikidata_client,
    reranker=None,
    working_memory: WorkingMemory = None,
    interaction_memory=None,
    num_iterations: int = 15,
    exploration_weight: float = 2.0,
    max_tree_depth: int = 10,
    golden_answer: Optional[str] = None,
    max_simulation_depth: int = 5,
    early_termination_enabled: bool = True,
    min_iterations: int = 3,
    high_confidence_threshold: float = 0.9,
    convergence_patience: int = 3,
    semantic_sufficiency_count: int = 2,
    correct_answers: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> Tuple[Dict, MCTSNode, Optional[int]]:
    """Run MCTS search with shared working memory.

    All iterations operate on the same ``working_memory``.  After each
    iteration, bidirectional text↔graph sync is performed via
    ``synchronize_memory()``.  Simulation uses a full n-step CoT rollout
    (``simulate()``) rather than a lightweight force-terminal step.
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
    semantic_signals = 0
    no_improvement = 0
    termination_reason = None
    pass_at_k = None

    def extract_answer(node):
        if node.node_type == NodeType.FINAL_ANSWER:
            return node.node_state.content.get('concise_answer') or node.node_state.content.get('final_answer', '')
        return None

    for iteration in range(num_iterations):
        path = select(root, exploration_weight)
        selected = path[-1]

        has_signal = False
        if not selected.is_terminal():
            expand_target = selected
        elif selected.visits > 0 and len(path) >= 2:
            expand_target = path[-2]
        else:
            expand_target = None

        if expand_target is not None:
            children, has_signal = await expand(expand_target, generator)
            if children:
                ucb_scores = [c.upper_confidence_bound(exploration_weight) for c in children]
                max_score = max(ucb_scores)
                best = [c for c, s in zip(children, ucb_scores) if s == max_score]
                selected = random.choice(best)
                for child in children:
                    _add_child_to_memory(child, generator.working_memory)

        # Simulate: CoT rollout
        if not selected.is_terminal():
            terminal, rollout_signal = await simulate(selected, generator, max_simulation_depth)
            if rollout_signal:
                has_signal = True
        else:
            terminal = selected

        if has_signal:
            semantic_signals += 1

        # Evaluate with 3-score VERIFIER
        reward = await evaluate(terminal, client, generator.working_memory, golden_answer)
        terminal.backpropagate(reward)

        # Sync shared memory (bidirectional) after each iteration
        await generator.working_memory.synchronize_memory(client, question, interaction_memory, reranker=reranker, **kwargs)

        is_new_terminal = terminal.visits == 1  # just backpropagated
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


async def get_answer(
    root: MCTSNode,
    client,
    interaction_memory=None,
    working_memory: Optional[WorkingMemory] = None,
) -> Tuple[str, str]:
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
    scores = [n.value / n.visits if n.visits > 0 else 0.0 for n in terminals]
    question = root.user_question
    if working_memory:
        textual = working_memory.format_textual_memory()
        graph = working_memory.format_graph_memory()
        context = f"{textual}\n\n{graph}".strip()
    else:
        context = ""

    try:
        synth_input = FinalAnswerSynthesisInput(question=question, candidate_answers=answers, candidate_scores=scores, context=context)
        results, log = await execute_role(client=client, role=FINAL_ANSWER_SYNTHESIZER, input_data=synth_input, interaction_memory=interaction_memory, n=1)
        if results:
            log_to_interaction_memory(interaction_memory, log)
            return f"Final Answer: {results[0].final_answer}", results[0].concise_answer
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")

    vote_input = MajorityVoteInput(question=question, answers=answers)
    results, log = await execute_role(client=client, role=MAJORITY_VOTER, input_data=vote_input, interaction_memory=interaction_memory, n=1)
    if results:
        log_to_interaction_memory(interaction_memory, log)
        return f"Final Answer: {results[0].final_answer}", results[0].concise_answer

    return "Unable to determine final answer.", "Unable to determine"
