"""Monte Carlo Tree Search (MCTS) reasoning implementation.

This module provides MCTS-based reasoning for question answering,
exploring multiple reasoning paths to find the best answer.
"""
import asyncio
import math
import random
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from typing_extensions import TypedDict

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.procedures.node_generator import NodeGenerator, GenerationResult
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.common import merge_logs, log_to_interaction_memory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class MCTSReasoningNode(BaseReasoningNode):
    """Reasoning node with MCTS-specific functionality (UCT, backpropagation)."""
    
    def __init__(
        self,
        node_state: NodeState,
        parent: Optional['MCTSReasoningNode'] = None,
        children: Optional[Tuple['MCTSReasoningNode']] = None,
        max_depth: int = 10,
    ):
        super().__init__(node_state=node_state, parent=parent, children=children, max_depth=max_depth)
        self.children: Tuple["MCTSReasoningNode"]
        self.parent: Optional["MCTSReasoningNode"]
        self.value: float = 0.0
        self.visits: int = 0

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        """Calculate UCT score for balancing exploration vs exploitation."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return float('inf') # unvisited nodes should be prioritized
        
        average_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term
    
    def backpropagate(self, reward: float) -> None:
        """Update scores up the tree from this node."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def generate_children(
        self,
        llm_agent: BaseLLMAgent, 
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory, 
        interaction_memory: Optional[InteractionMemory] = None,
        is_cot_simulation: bool = False,
        **kwargs
    ) -> Tuple[List["MCTSReasoningNode"], bool]:
        """Generate children nodes and update working memory."""
        generator = NodeGenerator(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **kwargs
        )
        
        # Determine which generation methods to run based on node type and state
        if self.depth > self.max_depth:
            nodes, result, _ = asyncio.run(self._generate_final_answer_nodes(generator))
            generator.update_working_memory(result)
            # Link children and update memory
            for child in nodes:
                child.parent = self
                self._add_child_to_memory(child, working_memory)
            log_to_interaction_memory(interaction_memory, result.log_data)
            return nodes, False  # Depth limit, not semantic sufficiency
        
        if is_cot_simulation:
            nodes, result, has_signal = asyncio.run(self._generate_subqa_nodes(generator))
            generator.update_working_memory(result)
            # Link children and update memory
            for child in nodes:
                child.parent = self
                self._add_child_to_memory(child, working_memory)
            log_to_interaction_memory(interaction_memory, result.log_data)
            return nodes, has_signal
        
        # Map node types to generation strategies
        strategies = {
            NodeType.USER_QUESTION: [self._generate_final_answer_nodes, self._generate_subqa_nodes],
            NodeType.SUB_QA_NODE: [self._generate_subqa_nodes, self._self_correct_nodes, 
                                   self._rephrase_nodes, self._strengthen_nodes],
            NodeType.REPHRASED_QUESTION_NODE: [self._generate_subqa_nodes],
            NodeType.SELF_CORRECTED_NODE: [self._generate_subqa_nodes, self._strengthen_nodes],
            NodeType.SYNTHESIS_NODE: [self._generate_subqa_nodes],
        }
        
        generators = strategies.get(self.node_type, [])
        if not generators:
            raise ValueError(f"Unsupported node type: {self.node_type}")
        results = asyncio.run(asyncio.gather(*[gen(generator) for gen in generators]))
        
        all_nodes = []
        generation_results = []
        all_logs = []
        has_semantic_signal = False
        for nodes, result, has_signal in results:
            all_nodes.extend(nodes)
            generation_results.append(result)
            all_logs.append(result.log_data)
            if has_signal:
                has_semantic_signal = True
        # merge_logs accepts *args, so unpack the list
        all_logs = merge_logs(*all_logs) if all_logs else {}
        for result in generation_results:
            generator.update_working_memory(result)
        
        # Link children and update memory
        for child in all_nodes:
            child.parent = self
            self._add_child_to_memory(child, working_memory)
        
        log_to_interaction_memory(interaction_memory, all_logs)
        return all_nodes, has_semantic_signal

    async def _generate_final_answer_nodes(
        self, 
        generator: NodeGenerator
    ) -> Tuple[List["MCTSReasoningNode"], GenerationResult, bool]:
        """Generate final answer node(s)."""
        if self.depth < 2:
            # Explore the external resources for the first two levels (very first level of the tree)
            should_explore = True 
        else:
            # Do not explore the external resources, only use which saved in the working memory
            should_explore = False
        result: GenerationResult = await generator.generate_answer(self.user_question, should_explore=should_explore)
        
        nodes = []
        for answer in result.answers:
            state = NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    "user_question": self.user_question,
                    "final_answer": answer.answer,
                    "concise_answer": answer.concise_answer,
                    "reasoning": answer.reasoning,
                }
            )
            nodes.append(MCTSReasoningNode(node_state=state, max_depth=self.max_depth))
        
        return nodes, result, False  # Not a semantic sufficiency signal

    async def _generate_subqa_nodes(
        self, 
        generator: NodeGenerator
    ) -> Tuple[List["MCTSReasoningNode"], GenerationResult, bool]:
        """Generate sub-question/answer nodes."""
        # Get question (may be rephrased)
        if self.node_type == NodeType.REPHRASED_QUESTION_NODE:
            subquestion = self.node_state.content['sub_question']
            subq_log = {}
            should_direct_answer = False
        else:
            subquestion, should_direct_answer, subq_log = await generator.generate_subquestion(
                self.user_question
            )
            if should_direct_answer or not subquestion:
                nodes, result, _ = await self._generate_final_answer_nodes(generator)
                # Merge subquestion log with answer log
                merged_log = merge_logs(subq_log, result.log_data)
                result.log_data = merged_log
                return nodes, result, True
        
        # Generate answer for subquestion
        result = await generator.generate_answer(subquestion, should_explore=True)
        
        nodes = []
        for answer in result.answers:
            state = NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': self.user_question,
                    'sub_question': subquestion,
                    'sub_answer': answer.answer,
                    'reasoning': answer.reasoning,
                }
            )
            nodes.append(MCTSReasoningNode(node_state=state, max_depth=self.max_depth))
        
        log = merge_logs(subq_log, result.log_data)
        result.log_data = log
        
        return nodes, result, False  # No semantic sufficiency signal when generating sub-QA

    async def _rephrase_nodes(
        self, 
        generator: NodeGenerator
    ) -> Tuple[List["MCTSReasoningNode"], GenerationResult, bool]:
        """Generate rephrased question nodes."""
        question = (self.node_state.content['sub_question'] 
                   if self.node_type == NodeType.SUB_QA_NODE 
                   else self.user_question)
        
        rephrased_questions, log = await generator.generate_rephrase(question)
        
        nodes = []
        for rq in rephrased_questions:
            state = NodeState(
                node_type=NodeType.REPHRASED_QUESTION_NODE,
                content={'user_question': self.user_question, 'sub_question': rq}
            )
            nodes.append(MCTSReasoningNode(node_state=state, max_depth=self.max_depth))
        
        result = GenerationResult(log_data=log)
        return nodes, result, False  # Not a semantic sufficiency signal

    async def _self_correct_nodes(
        self, 
        generator: NodeGenerator
    ) -> Tuple[List["MCTSReasoningNode"], GenerationResult, bool]:
        """Generate self-corrected answer nodes."""
        sub_q = self.node_state.content.get('sub_question')
        sub_a = self.node_state.content.get('sub_answer')
        
        if not sub_q or not sub_a:
            logger.warning("No sub-question or sub-answer found for self-correction(sub_q, sub_a)")
            return [], GenerationResult(), False
        
        result = await generator.generate_self_correction(sub_q, sub_a)
        
        nodes = []
        for correction in result.answers:
            state = NodeState(
                node_type=NodeType.SELF_CORRECTED_NODE,
                content={
                    'user_question': self.user_question,
                    'sub_question': sub_q,
                    'sub_answer': correction.refined_answer
                }
            )
            nodes.append(MCTSReasoningNode(node_state=state, max_depth=self.max_depth))
        
        return nodes, result, False  # Not a semantic sufficiency signal

    async def _strengthen_nodes(
        self, 
        generator: NodeGenerator
    ) -> Tuple[List["MCTSReasoningNode"], GenerationResult, bool]:
        """Generate reasoning synthesis nodes."""
        result = await generator.generate_synthesis(self.user_question)
        
        nodes = []
        has_semantic_signal = False
        for output in result.answers:
            if output.is_answerable:
                state = NodeState(
                    node_type=NodeType.FINAL_ANSWER,
                    content={
                        "user_question": self.user_question,
                        "final_answer": output.step_conclusion,
                    }
                )
                has_semantic_signal = True  # is_answerable=True indicates semantic sufficiency
            else:
                state = NodeState(
                    node_type=NodeType.SYNTHESIS_NODE,
                    content={
                        'user_question': self.user_question,
                        'synthesized_reasoning': output.step_conclusion
                    }
                )
            nodes.append(MCTSReasoningNode(node_state=state, max_depth=self.max_depth))
        
        return nodes, result, has_semantic_signal

    def _add_child_to_memory(self, child: "MCTSReasoningNode", working_memory: WorkingMemory) -> None:
        """Add child node content to working memory."""
        content = child.node_state.content
        source = roles.extractor.SourceType.SYSTEM_PREDICTION
        
        if child.node_type == NodeType.SUB_QA_NODE:
            sub_q, sub_a = content.get('sub_question', ''), content.get('sub_answer', '')
            if sub_q and sub_a:
                working_memory.add_textual_memory(f"Q: {sub_q}; A: {sub_a}", source=source)
        elif child.node_type == NodeType.SELF_CORRECTED_NODE:
            sub_q, sub_a = content.get('sub_question', ''), content.get('sub_answer', '')
            if sub_q and sub_a:
                working_memory.add_textual_memory(f"Q: {sub_q}; A: {sub_a}", source=source)
        elif child.node_type == NodeType.SYNTHESIS_NODE:
            synthesis = content.get('synthesized_reasoning', '')
            if synthesis:
                working_memory.add_textual_memory(synthesis, source=source)


# =============================================================================
# MCTS Search Tree and Algorithms
# =============================================================================

class MCTSSearchTree(TypedDict):
    root: MCTSReasoningNode


def select(tree: MCTSSearchTree, exploration_weight: float = 1.0) -> List[MCTSReasoningNode]:
    """Select a path from root to leaf using UCT algorithm.
    
    Pure UCT implementation: unvisited nodes automatically get infinity UCT,
    so they are always prioritized. No need for explicit unexplored tracking.
    """
    path = []
    node = tree['root']
    
    while True:
        if not node.children or node.is_terminal():
            path.append(node)
            return path
        
        path.append(node)
        uct_scores = [c.upper_confidence_bound(exploration_weight) for c in node.children]
        node = node.children[uct_scores.index(max(uct_scores))]


def expand(
    tree: MCTSSearchTree, 
    selected_node: MCTSReasoningNode,
    llm_agent: BaseLLMAgent,
    retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
    working_memory: WorkingMemory, 
    interaction_memory: Optional[InteractionMemory] = None,
    is_cot_simulation: bool = False,
    **kwargs
) -> Tuple[List[MCTSReasoningNode], bool]:
    """Expand tree by generating children for selected node."""
    if selected_node.is_terminal():
        return [], False
    
    children, has_semantic_signal = selected_node.generate_children(
        llm_agent=llm_agent,
        retriever_agent=retriever_agent,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        is_cot_simulation=is_cot_simulation,
        **kwargs
    )
    
    return children, has_semantic_signal


def simulate(
    node: MCTSReasoningNode,
    llm_agent: BaseLLMAgent,
    retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
    working_memory: WorkingMemory,
    interaction_memory: Optional[InteractionMemory] = None,
    is_cot_simulation: bool = True,
    max_simulation_depth: int = 5,
    **kwargs
) -> Tuple[MCTSReasoningNode, bool]:
    """Simulate a rollout from node to terminal state."""
    current = node
    
    for _ in range(max_simulation_depth):
        if current.is_terminal():
            break
        
        children, has_semantic_signal = current.generate_children(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            is_cot_simulation=is_cot_simulation,
            **kwargs
        )
        
        if not children:
            break
        
        current = random.choice(children)
    
    return current, has_semantic_signal


async def evaluate(
    node: MCTSReasoningNode,
    llm_agent: BaseLLMAgent,
    interaction_memory: Optional[InteractionMemory] = None,
    golden_answer: Optional[str] = None
) -> float:
    """Evaluate terminal node and return reward score (0-1)."""
    if node.node_type != NodeType.FINAL_ANSWER:
        return 0.1 # not a final answer node, return a low reward
    
    eval_input = roles.evaluator.AnswerEvaluationInput(
        user_question=node.user_question,
        system_answer=node.node_state.content.get('final_answer', ''),
        correct_answer=golden_answer or node.golden_answer or "Not available"
    )
    
    eval_results, _ = await execute_role(
        llm_agent=llm_agent,
        role=roles.evaluator.Evaluator(),
        input_data=eval_input,
        interaction_memory=interaction_memory,
        n=1
    )
    
    if eval_results:
        return eval_results[0].rating / 10.0
    return 0.5 # neutral reward for non-final answer nodes


def mcts_search(
    question: str,
    llm_agent: BaseLLMAgent,
    retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
    working_memory: WorkingMemory,
    interaction_memory: Optional[InteractionMemory] = None,
    num_iterations: int = 10,
    exploration_weight: float = 1.0,
    is_cot_simulation: bool = True,
    max_tree_depth: int = 10,
    max_simulation_depth: int = 5,
    golden_answer: Optional[str] = None,
    early_termination_enabled: bool = True,
    min_iterations: int = 3,
    high_confidence_threshold: float = 0.9,
    convergence_patience: int = 3,
    semantic_sufficiency_count: int = 2,
    **kwargs
) -> Tuple[Dict, MCTSSearchTree]:
    """Run MCTS to find the best reasoning path."""
    # Initialize tree
    root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question})
    root = MCTSReasoningNode(node_state=root_state, max_depth=max_tree_depth)
    tree: MCTSSearchTree = {'root': root}
    
    best_node: Optional[MCTSReasoningNode] = None
    best_reward = -float('inf')
    
    # Early termination tracking
    semantic_sufficiency_signals = 0  # Count of semantic sufficiency signals
    no_improvement_count = 0  # Consecutive iterations without improvement
    termination_reason = None
    
    for iteration in range(num_iterations):
        # Selection
        path = select(tree, exploration_weight)
        selected = path[-1]
        
        # Expansion
        children = []
        has_semantic_signal = False
        if not selected.is_terminal():
            children, has_semantic_signal = expand(
                tree, selected, llm_agent, retriever_agent, working_memory,
                interaction_memory, is_cot_simulation=False, **kwargs
            )
            if children:
                selected = children[0] # More deterministic to select the first child
                
                # Track semantic sufficiency signals: when should_direct_answer=True or is_answerable=True
                if has_semantic_signal:
                    semantic_sufficiency_signals += 1
                    logger.debug(f"Iteration {iteration + 1}: Semantic sufficiency signal detected "
                               f"(should_direct_answer=True or is_answerable=True). Count: {semantic_sufficiency_signals}")
        
        # Simulation
        if not selected.is_terminal():
            terminal, has_semantic_signal = simulate(
                selected, llm_agent, retriever_agent, working_memory,
                interaction_memory, is_cot_simulation, max_simulation_depth, **kwargs
            )
            if has_semantic_signal:
                semantic_sufficiency_signals += 1
                logger.debug(f"Iteration {iteration + 1}: Semantic sufficiency signal detected "
                           f"(should_direct_answer=True or is_answerable=True). Count: {semantic_sufficiency_signals}")
        else:
            terminal = selected
        
        # Evaluation and backpropagation
        reward = asyncio.run(evaluate(terminal, llm_agent, interaction_memory, golden_answer))
        terminal.backpropagate(reward)
        
        # Sync working memory
        working_memory.synchronize_memory(llm_agent, question, interaction_memory)
        
        # Track best and convergence
        if terminal.is_terminal():
            if reward > best_reward:
                best_reward = reward
                best_node = terminal
                no_improvement_count = 0  # Reset counter when we find a better answer
            else:
                no_improvement_count += 1  # Only count iterations where we evaluated terminal nodes
        # Note: We don't increment no_improvement_count for non-terminal nodes
        # since we can't meaningfully compare their rewards
        
        # Early termination checks (only after minimum iterations)
        if early_termination_enabled and iteration + 1 >= min_iterations:
            # Check high confidence threshold
            if reward >= high_confidence_threshold and terminal.is_terminal():
                termination_reason = f"High confidence answer found (reward={reward:.3f} >= {high_confidence_threshold})"
                logger.info(f"MCTS early termination at iteration {iteration + 1}: {termination_reason}")
                break
            
            # Check semantic sufficiency
            if semantic_sufficiency_signals >= semantic_sufficiency_count:
                termination_reason = (f"Semantic sufficiency signals reached threshold "
                                   f"({semantic_sufficiency_signals} >= {semantic_sufficiency_count})")
                logger.info(f"MCTS early termination at iteration {iteration + 1}: {termination_reason}")
                break
            
            # Check convergence
            if no_improvement_count >= convergence_patience:
                termination_reason = (f"No improvement for {no_improvement_count} consecutive iterations "
                                    f"(patience={convergence_patience})")
                logger.info(f"MCTS early termination at iteration {iteration + 1}: {termination_reason}")
                break
    
    if termination_reason:
        logger.info(f"MCTS completed in {iteration + 1} iterations (out of {num_iterations}): {termination_reason}")
    else:
        logger.debug(f"MCTS completed all {num_iterations} iterations")
    
    return (best_node.node_state.content if best_node else {}), tree


def get_answer(
    tree: MCTSSearchTree,
    llm_agent: BaseLLMAgent,
    interaction_memory: Optional[InteractionMemory] = None
) -> Tuple[str, str]:
    """Extract final answer from explored tree."""
    # Collect terminal nodes
    terminals: List[MCTSReasoningNode] = []
    
    def collect_terminals(node: MCTSReasoningNode):
        if node.is_terminal():
            terminals.append(node)
        for child in node.children:
            collect_terminals(child)
    
    collect_terminals(tree['root'])
    
    if not terminals:
        return "No final answer found.", "No answer"
    
    # Gather all answers
    answers: List[str] = []
    concise_answers: List[str] = []
    for node in terminals:
        answers.append(str(node.node_state))
        concise_answers.append(node.node_state.content.get('final_answer', ''))
    
    if not answers:
        return "No final answer found.", "No answer"
    
    # Try synthesis, fallback to majority vote
    try:
        synth_input = roles.evaluator.FinalAnswerSynthesisInput(
            question=tree['root'].user_question,
            candidate_answers=answers
        )
        results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.FinalAnswerSynthesizer(),
            input_data=synth_input,
            interaction_memory=interaction_memory,
            n=1
        ))
        if results:
            result: roles.evaluator.FinalAnswerSynthesisOutput = results[0]
            log_to_interaction_memory(interaction_memory, log)
            return f"Final Answer: {result.final_answer}\nReasoning: {result.reasoning}", result.concise_answer
    except Exception as e:
        logger.warning(f"Failed to synthesize final answer: {e}")
    
    # Fallback: majority vote
    vote_input = roles.evaluator.MajorityVoteInput(
        question=tree['root'].user_question,
        answers=answers
    )
    results, log = asyncio.run(execute_role(
        llm_agent=llm_agent,
        role=roles.evaluator.MajorityVoter(),
        input_data=vote_input,
        interaction_memory=interaction_memory,
        n=1
    ))
    
    if results:
        log_to_interaction_memory(interaction_memory, log)
        result: roles.evaluator.MajorityVoteOutput = results[0]
        return f"Final Answer: {result.final_answer}\nReasoning: {result.reasoning}", result.concise_answer
    
    return "Unable to determine final answer.", "Unable to determine"
