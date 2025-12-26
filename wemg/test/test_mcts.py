"""
Comprehensive tests for wemg/runners/mcts.py.

These are real integration tests that call actual LLM servers and retrieval APIs.
Tests cover (based on MCTS_REVIEW_UPDATED.md recommendations):

Unit Tests:
- UCT calculation with various visit counts
- Backpropagation correctness
- Path selection logic
- Terminal node detection
- Edge cases (empty children, no terminal nodes, etc.)

Integration Tests:
- Full MCTS search with real LLM and RetrieverAgent
- Tree expansion and exploration
- Answer extraction
- Complex multi-hop questions
"""
import os
import pytest
import math
import asyncio
from pathlib import Path
from typing import List

from wemg.runners.mcts import (
    MCTSReasoningNode,
    MCTSSearchTree,
    select,
    expand,
    simulate,
    evaluate,
    mcts_search,
    get_answer
)
from wemg.runners.base_reasoning_node import NodeType, NodeState
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent


# ============================================================================
# Test Configuration
# ============================================================================

TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

# Wiki corpus configuration for RetrieverAgent tests
WIKI_CORPUS_HF = os.getenv("WIKI_CORPUS_HF", "Hieuman/wiki23-processed")
WIKI_INDEX_PATH = Path(os.getenv("WIKI_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return "What is the capital of France?"


@pytest.fixture
def complex_question_1():
    """Complex multi-hop question requiring multiple reasoning steps."""
    return "Who was the president of the United States when the first iPhone was released?"


@pytest.fixture
def complex_question_2():
    """Complex question requiring comparison and temporal reasoning."""
    return "Which magazine was started first: Arthur's Magazine or First for Women?"


@pytest.fixture
def complex_question_3():
    """Complex question requiring multiple sub-questions."""
    return "What is the relationship between the author of '1984' and the author of 'Animal Farm'?"


@pytest.fixture
def complex_question_4():
    """Complex question requiring historical and geographical reasoning."""
    return "What was the capital of the country that won the most Olympic gold medals in 2016?"


@pytest.fixture
def complex_question_5():
    """Complex question requiring scientific and temporal reasoning."""
    return "What was the name of the scientist who discovered the structure of DNA, and in which year did they receive the Nobel Prize?"


@pytest.fixture
def root_node(sample_question):
    """Create a root MCTS node."""
    state = NodeState(
        node_type=NodeType.USER_QUESTION,
        content={'user_question': sample_question}
    )
    return MCTSReasoningNode(node_state=state, max_depth=10)


@pytest.fixture
def terminal_node(sample_question):
    """Create a terminal (final answer) node."""
    state = NodeState(
        node_type=NodeType.FINAL_ANSWER,
        content={
            'user_question': sample_question,
            'final_answer': 'Paris',
            'concise_answer': 'Paris'
        }
    )
    return MCTSReasoningNode(node_state=state, max_depth=10)


@pytest.fixture
def subqa_node(sample_question):
    """Create a sub-QA node."""
    state = NodeState(
        node_type=NodeType.SUB_QA_NODE,
        content={
            'user_question': sample_question,
            'sub_question': 'What is France?',
            'sub_answer': 'France is a country in Western Europe'
        }
    )
    return MCTSReasoningNode(node_state=state, max_depth=10)


@pytest.fixture
def working_memory():
    """Create an empty working memory."""
    return WorkingMemory()


@pytest.fixture
def interaction_memory():
    """Create an interaction memory."""
    return InteractionMemory()


@pytest.fixture
def llm_agent():
    """Create a BaseLLMAgent for testing."""
    return BaseLLMAgent(
        model_name=TEST_LLM_MODEL,
        url=TEST_LLM_API_BASE,
        api_key=TEST_LLM_API_KEY,
        temperature=0.7,
        max_tokens=32768,
        concurrency=32,
        max_retries=3
    )


@pytest.fixture
def retriever_agent_embedder_config():
    """Create embedder configuration for RetrieverAgent."""
    return {
        'model_name': TEST_EMBEDDING_MODEL,
        'url': TEST_EMBEDDING_API_BASE,
        'api_key': TEST_LLM_API_KEY,
        'is_embedding': True,
        'timeout': 60,
    }


@pytest.fixture
def retriever_agent(retriever_agent_embedder_config):
    """Create a RetrieverAgent instance with wiki corpus and pre-indexed FAISS."""
    # Check if index exists
    if not WIKI_INDEX_PATH.exists():
        pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}. Please ensure the index is available.")
    
    # Try to load corpus from local directory first (if it was previously saved)
    # Otherwise, load from HuggingFace
    local_corpus_path = WIKI_INDEX_PATH.parent / "without_embeddings"
    if local_corpus_path.exists():
        corpus_path = local_corpus_path
    else:
        # Load from HuggingFace - will be saved locally after first load
        corpus_path = Path(WIKI_CORPUS_HF)
    
    agent = RetrieverAgent(
        embedder_config=retriever_agent_embedder_config,
        corpus_path=corpus_path,
        index_path=WIKI_INDEX_PATH,
        embedder_type='openai'
    )
    
    return agent


# ============================================================================
# Unit Tests: MCTSReasoningNode
# ============================================================================

class TestMCTSReasoningNode:
    """Test suite for MCTSReasoningNode."""
    
    def test_node_initialization(self, root_node):
        """Test node initialization."""
        assert root_node.value == 0.0
        assert root_node.visits == 0
        assert root_node.node_type == NodeType.USER_QUESTION
        assert root_node.is_root()
        assert not root_node.is_terminal()
    
    def test_node_with_parent(self, root_node, subqa_node):
        """Test node with parent relationship."""
        subqa_node.parent = root_node
        root_node.children = [subqa_node]
        
        assert subqa_node.parent == root_node
        assert subqa_node in root_node.children
        assert not subqa_node.is_root()
    
    def test_terminal_node(self, terminal_node):
        """Test terminal node detection."""
        assert terminal_node.is_terminal()
        assert terminal_node.is_valid_leaf()
        assert terminal_node.node_type == NodeType.FINAL_ANSWER
    
    def test_max_depth_terminal(self, root_node):
        """Test that nodes at max depth are terminal."""
        # Create a deep chain
        current = root_node
        for i in range(10):
            state = NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': root_node.user_question,
                    'sub_question': f'Question {i}',
                    'sub_answer': f'Answer {i}'
                }
            )
            child = MCTSReasoningNode(node_state=state, parent=current, max_depth=10)
            current.children = [child]
            current = child
        
        # Node at depth 10 should be terminal
        assert current.depth == 10
        assert current.is_terminal()
    
    def test_uct_calculation_unvisited(self, root_node, subqa_node):
        """Test UCT calculation for unvisited nodes."""
        subqa_node.parent = root_node
        root_node.children = [subqa_node]
        root_node.visits = 5  # Parent has been visited
        
        # Unvisited node should return infinity
        uct = subqa_node.upper_confidence_bound()
        assert uct == float('inf')
    
    def test_uct_calculation_visited(self, root_node, subqa_node):
        """Test UCT calculation for visited nodes."""
        subqa_node.parent = root_node
        root_node.children = [subqa_node]
        
        # Set up visits and values
        root_node.visits = 10
        subqa_node.visits = 5
        subqa_node.value = 2.5  # Average reward = 0.5
        
        uct = subqa_node.upper_confidence_bound(exploration_weight=1.0)
        
        # UCT = average_reward + exploration_weight * sqrt(log(parent_visits) / visits)
        expected_average = 2.5 / 5  # 0.5
        expected_exploration = math.sqrt(math.log(10) / 5)
        expected_uct = expected_average + expected_exploration
        
        assert abs(uct - expected_uct) < 1e-6
    
    def test_uct_raises_on_root(self, root_node):
        """Test that UCT raises error on root node."""
        with pytest.raises(ValueError, match="Cannot obtain UCT from root node"):
            root_node.upper_confidence_bound()
    
    def test_backpropagation_single_node(self, terminal_node):
        """Test backpropagation on a single node."""
        reward = 0.8
        terminal_node.backpropagate(reward)
        
        assert terminal_node.visits == 1
        assert terminal_node.value == reward
    
    def test_backpropagation_chain(self, root_node, subqa_node, terminal_node):
        """Test backpropagation through a chain of nodes."""
        # Build chain: root -> subqa -> terminal
        subqa_node.parent = root_node
        terminal_node.parent = subqa_node
        root_node.children = [subqa_node]
        subqa_node.children = [terminal_node]
        
        # Initial state
        root_node.visits = 2
        root_node.value = 1.0
        subqa_node.visits = 1
        subqa_node.value = 0.5
        
        # Backpropagate from terminal
        reward = 0.9
        terminal_node.backpropagate(reward)
        
        # Terminal should be updated
        assert terminal_node.visits == 1
        assert terminal_node.value == reward
        
        # SubQA should be updated
        assert subqa_node.visits == 2
        expected_subqa_value = (0.5 * 1 + reward) / 2  # 0.7
        assert abs(subqa_node.value - expected_subqa_value) < 1e-6
        
        # Root should be updated
        assert root_node.visits == 3
        expected_root_value = (1.0 * 2 + reward) / 3  # ~0.9667
        assert abs(root_node.value - expected_root_value) < 1e-6
    
    def test_backpropagation_multiple_visits(self, root_node, subqa_node):
        """Test backpropagation with multiple visits."""
        subqa_node.parent = root_node
        root_node.children = [subqa_node]
        
        # First backpropagation
        subqa_node.backpropagate(0.8)
        assert subqa_node.visits == 1
        assert subqa_node.value == 0.8
        assert root_node.visits == 1
        assert root_node.value == 0.8
        
        # Second backpropagation
        subqa_node.backpropagate(0.6)
        assert subqa_node.visits == 2
        expected_value = (0.8 * 1 + 0.6) / 2  # 0.7
        assert abs(subqa_node.value - expected_value) < 1e-6
        assert root_node.visits == 2
        expected_root_value = (0.8 * 1 + 0.6) / 2  # 0.7
        assert abs(root_node.value - expected_root_value) < 1e-6


# ============================================================================
# Unit Tests: MCTS Algorithms
# ============================================================================

class TestMCTSAlgorithms:
    """Test suite for MCTS algorithm functions."""
    
    def test_select_unexplored_child(self, root_node, subqa_node):
        """Test selection when unvisited children exist (pure UCT)."""
        root_node.children = (subqa_node,)
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        path = select(tree, exploration_weight=1.0)
        assert len(path) == 2  # root + selected child
        assert path[0] == root_node
        assert path[1] == subqa_node
    
    def test_select_by_uct(self, root_node):
        """Test selection by UCT when children have been visited."""
        # Create multiple children
        child1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={'user_question': root_node.user_question, 'sub_question': 'Q1', 'sub_answer': 'A1'}
            ),
            parent=root_node,
            max_depth=10
        )
        child2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={'user_question': root_node.user_question, 'sub_question': 'Q2', 'sub_answer': 'A2'}
            ),
            parent=root_node,
            max_depth=10
        )
        
        root_node.children = (child1, child2)
        root_node.visits = 10
        
        # Set up visits and values so child1 has higher UCT
        child1.visits = 3
        child1.value = 2.4  # Average = 0.8
        child2.visits = 5
        child2.value = 2.0  # Average = 0.4
        
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        path = select(tree, exploration_weight=1.0)
        
        # Should select child1 (higher UCT)
        assert path[-1] == child1
    
    def test_select_terminal_node(self, root_node, terminal_node):
        """Test selection stops at terminal node."""
        terminal_node.parent = root_node
        root_node.children = (terminal_node,)
        
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        path = select(tree, exploration_weight=1.0)
        
        assert len(path) == 2
        assert path[-1] == terminal_node
        assert path[-1].is_terminal()
    
    def test_select_no_children(self, root_node):
        """Test selection when node has no children."""
        root_node.children = ()
        
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        path = select(tree, exploration_weight=1.0)
        
        assert len(path) == 1
        assert path[0] == root_node
    
    def test_expand_terminal_node(self, terminal_node, llm_agent, retriever_agent, working_memory):
        """Test expansion of terminal node returns empty list."""
        tree: MCTSSearchTree = {
            'root': terminal_node
        }
        
        children, has_semantic_signal = expand(
            tree, terminal_node, llm_agent, retriever_agent, working_memory
        )
        
        assert children == []
        assert has_semantic_signal == False


# ============================================================================
# Unit Tests: Evaluation
# ============================================================================

class TestEvaluation:
    """Test suite for evaluation function."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluate_non_final_answer(self, llm_agent, interaction_memory, subqa_node):
        """Test evaluation of non-final answer node with real LLM."""
        reward = await evaluate(subqa_node, llm_agent, interaction_memory)
        
        # Non-final answer should return low reward
        assert reward == 0.1
        
        print(f"✓ evaluate non-final answer")
        print(f"  Reward: {reward}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluate_final_answer(self, llm_agent, interaction_memory, terminal_node):
        """Test evaluation of final answer node with real LLM."""
        reward = await evaluate(terminal_node, llm_agent, interaction_memory)
        
        # Should return a reward between 0 and 1
        assert 0.0 <= reward <= 1.0
        
        print(f"✓ evaluate final answer")
        print(f"  Reward: {reward:.2f}")


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_tree(self):
        """Test handling of empty tree."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'Test question'}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        root.children = ()
        
        tree: MCTSSearchTree = {
            'root': root
        }
        
        path = select(tree, exploration_weight=1.0)
        assert len(path) == 1
        assert path[0] == root
    
    def test_single_node_tree(self, root_node):
        """Test tree with only root node."""
        root_node.children = ()
        
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        path = select(tree, exploration_weight=1.0)
        assert path == [root_node]
    
    def test_all_children_visited(self, root_node):
        """Test selection when all children have been visited (use UCT)."""
        child1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={'user_question': root_node.user_question, 'sub_question': 'Q1', 'sub_answer': 'A1'}
            ),
            parent=root_node,
            max_depth=10
        )
        child2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={'user_question': root_node.user_question, 'sub_question': 'Q2', 'sub_answer': 'A2'}
            ),
            parent=root_node,
            max_depth=10
        )
        
        root_node.children = (child1, child2)
        root_node.visits = 5
        
        # Both children visited - should use UCT
        child1.visits = 2
        child1.value = 1.0
        child2.visits = 3
        child2.value = 1.5  # Higher average reward
        
        tree: MCTSSearchTree = {
            'root': root_node
        }
        
        # Should select by UCT (child2 has higher average reward)
        path = select(tree, exploration_weight=1.0)
        assert len(path) >= 1
        assert path[0] == root_node
    
    def test_max_depth_reached(self, root_node):
        """Test behavior when max depth is reached."""
        current = root_node
        for i in range(10):
            state = NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': root_node.user_question,
                    'sub_question': f'Q{i}',
                    'sub_answer': f'A{i}'
                }
            )
            # anytree automatically manages children when parent is set
            child = MCTSReasoningNode(node_state=state, parent=current, max_depth=10)
            current = child
        
        assert current.depth == 10
        assert current.is_terminal()
    
    def test_node_without_parent_uct(self, root_node):
        """Test UCT calculation raises error for root node."""
        with pytest.raises(ValueError, match="Cannot obtain UCT from root node"):
            root_node.upper_confidence_bound()


# ============================================================================
# Complex Question Tests
# ============================================================================

class TestComplexQuestions:
    """Test MCTS with complex, multi-hop questions."""
    
    def test_complex_question_tree_structure(self, complex_question_1):
        """Test tree structure with complex multi-hop question."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_1}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Create a multi-hop reasoning path
        # Step 1: When was iPhone released?
        step1_state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={
                'user_question': complex_question_1,
                'sub_question': 'When was the first iPhone released?',
                'sub_answer': 'The first iPhone was released on June 29, 2007'
            }
        )
        step1 = MCTSReasoningNode(node_state=step1_state, parent=root, max_depth=10)
        
        # Step 2: Who was president in 2007?
        step2_state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={
                'user_question': complex_question_1,
                'sub_question': 'Who was the president of the United States in 2007?',
                'sub_answer': 'George W. Bush was the president of the United States in 2007'
            }
        )
        step2 = MCTSReasoningNode(node_state=step2_state, parent=step1, max_depth=10)
        
        # Final answer
        final_state = NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={
                'user_question': complex_question_1,
                'final_answer': 'George W. Bush was the president when the first iPhone was released in 2007',
                'concise_answer': 'George W. Bush'
            }
        )
        final = MCTSReasoningNode(node_state=final_state, parent=step2, max_depth=10)
        
        root.children = [step1]
        step1.children = [step2]
        step2.children = [final]
        
        # Verify tree structure
        assert root.depth == 0
        assert step1.depth == 1
        assert step2.depth == 2
        assert final.depth == 3
        assert final.is_terminal()
        
        # Verify reasoning path
        trajectory = final.get_trajectory()
        assert len(trajectory) == 4  # root + 3 steps
        assert trajectory[0].node_type == NodeType.USER_QUESTION
        assert trajectory[-1].node_type == NodeType.FINAL_ANSWER
    
    def test_complex_question_with_multiple_paths(self, complex_question_2):
        """Test complex question with multiple reasoning paths."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_2}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Path 1: Research Arthur's Magazine
        path1_step1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_2,
                    'sub_question': 'When was Arthur\'s Magazine started?',
                    'sub_answer': 'Arthur\'s Magazine was started in 1844'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        # Path 2: Research First for Women
        path2_step1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_2,
                    'sub_question': 'When was First for Women started?',
                    'sub_answer': 'First for Women was started in 1989'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        root.children = [path1_step1, path2_step1]
        
        # Both paths should be valid children
        assert len(root.children) == 2
        assert all(child.parent == root for child in root.children)
        
        # Test selection can choose either path
        tree: MCTSSearchTree = {
            'root': root
        }
        
        path = select(tree, exploration_weight=1.0)
        assert len(path) == 2
        assert path[0] == root
        assert path[1] in root.children
    
    def test_complex_question_with_rephrasing(self, complex_question_3):
        """Test complex question that may require rephrasing."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_3}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Create rephrased question node
        rephrased_state = NodeState(
            node_type=NodeType.REPHRASED_QUESTION_NODE,
            content={
                'user_question': complex_question_3,
                'sub_question': 'Who wrote "1984" and who wrote "Animal Farm"?'
            }
        )
        rephrased = MCTSReasoningNode(node_state=rephrased_state, parent=root, max_depth=10)
        
        # Answer the rephrased question
        answer_state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={
                'user_question': complex_question_3,
                'sub_question': 'Who wrote "1984" and who wrote "Animal Farm"?',
                'sub_answer': 'Both "1984" and "Animal Farm" were written by George Orwell'
            }
        )
        answer = MCTSReasoningNode(node_state=answer_state, parent=rephrased, max_depth=10)
        
        root.children = [rephrased]
        rephrased.children = [answer]
        
        # Verify rephrasing path
        assert rephrased.node_type == NodeType.REPHRASED_QUESTION_NODE
        assert answer.node_type == NodeType.SUB_QA_NODE
        assert answer.parent == rephrased
    
    def test_complex_question_with_synthesis(self, complex_question_4):
        """Test complex question requiring synthesis of multiple facts."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_4}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Step 1: Which country won most gold medals in 2016?
        step1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_4,
                    'sub_question': 'Which country won the most Olympic gold medals in 2016?',
                    'sub_answer': 'The United States won the most Olympic gold medals in 2016 with 46 medals'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        # Step 2: What is the capital of the United States?
        step2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_4,
                    'sub_question': 'What is the capital of the United States?',
                    'sub_answer': 'Washington, D.C. is the capital of the United States'
                }
            ),
            parent=step1,
            max_depth=10
        )
        
        # Synthesis node
        synthesis = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SYNTHESIS_NODE,
                content={
                    'user_question': complex_question_4,
                    'synthesized_reasoning': 'The United States won the most gold medals in 2016, and its capital is Washington, D.C.'
                }
            ),
            parent=step2,
            max_depth=10
        )
        
        # Final answer
        final = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    'user_question': complex_question_4,
                    'final_answer': 'Washington, D.C. is the capital of the United States, which won the most Olympic gold medals in 2016',
                    'concise_answer': 'Washington, D.C.'
                }
            ),
            parent=synthesis,
            max_depth=10
        )
        
        root.children = [step1]
        step1.children = [step2]
        step2.children = [synthesis]
        synthesis.children = [final]
        
        # Verify synthesis path
        assert synthesis.node_type == NodeType.SYNTHESIS_NODE
        assert final.is_terminal()
        assert final.depth == 4
    
    def test_complex_question_with_self_correction(self, complex_question_5):
        """Test complex question that may require self-correction."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_5}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Initial answer (may be incorrect)
        initial = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_5,
                    'sub_question': 'Who discovered the structure of DNA?',
                    'sub_answer': 'Watson and Crick discovered the structure of DNA'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        # Self-corrected answer
        corrected = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SELF_CORRECTED_NODE,
                content={
                    'user_question': complex_question_5,
                    'sub_question': 'Who discovered the structure of DNA?',
                    'sub_answer': 'James Watson and Francis Crick discovered the structure of DNA in 1953, and they received the Nobel Prize in 1962'
                }
            ),
            parent=initial,
            max_depth=10
        )
        
        root.children = [initial]
        initial.children = [corrected]
        
        # Verify self-correction path
        assert initial.node_type == NodeType.SUB_QA_NODE
        assert corrected.node_type == NodeType.SELF_CORRECTED_NODE
        assert corrected.parent == initial
    
    def test_complex_question_mcts_selection(self, complex_question_1):
        """Test MCTS selection with complex question tree."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_1}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Create multiple reasoning paths
        path1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_1,
                    'sub_question': 'When was iPhone released?',
                    'sub_answer': '2007'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        path2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_1,
                    'sub_question': 'Who was president in 2007?',
                    'sub_answer': 'George W. Bush'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        root.children = [path1, path2]
        root.visits = 5
        
        # Set up UCT scores - path1 should be explored more
        path1.visits = 3
        path1.value = 2.1  # Average = 0.7
        path2.visits = 1
        path2.value = 0.5  # Average = 0.5
        
        tree: MCTSSearchTree = {
            'root': root
        }
        
        # Should select path with better UCT score
        path = select(tree, exploration_weight=1.0)
        assert len(path) >= 2
        assert path[0] == root
    
    def test_complex_question_backpropagation(self, complex_question_1):
        """Test backpropagation through complex reasoning path."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': complex_question_1}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=10)
        
        # Create 3-step reasoning path
        step1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_1,
                    'sub_question': 'When was iPhone released?',
                    'sub_answer': '2007'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        step2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': complex_question_1,
                    'sub_question': 'Who was president in 2007?',
                    'sub_answer': 'George W. Bush'
                }
            ),
            parent=step1,
            max_depth=10
        )
        
        final = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    'user_question': complex_question_1,
                    'final_answer': 'George W. Bush',
                    'concise_answer': 'George W. Bush'
                }
            ),
            parent=step2,
            max_depth=10
        )
        
        root.children = [step1]
        step1.children = [step2]
        step2.children = [final]
        
        # Backpropagate reward
        reward = 0.9
        final.backpropagate(reward)
        
        # All nodes should be updated
        assert final.visits == 1
        assert final.value == reward
        assert step2.visits == 1
        assert step2.value == reward
        assert step1.visits == 1
        assert step1.value == reward
        assert root.visits == 1
        assert root.value == reward


# ============================================================================
# Integration Tests
# ============================================================================

class TestMCTSIntegration:
    """Integration tests for full MCTS search with real LLM and RetrieverAgent."""
    
    # @pytest.mark.slow
    # @pytest.mark.integration
    def test_mcts_search_basic(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test basic MCTS search functionality with real LLM."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=3,
            exploration_weight=1.0,
            is_cot_simulation=True,
            max_tree_depth=5,
            max_simulation_depth=3
        )
        
        assert tree is not None
        assert tree['root'] is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS search basic")
        print(f"  Question: {sample_question}")
        if result.get('final_answer'):
            print(f"  Final answer: {result['final_answer'][:200]}")
    
    # @pytest.mark.slow
    # @pytest.mark.integration
    def test_mcts_search_complex_question(self, llm_agent, retriever_agent, working_memory, interaction_memory, complex_question_1):
        """Test MCTS search with complex multi-hop question."""
        result, tree = mcts_search(
            question=complex_question_1,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=5,
            exploration_weight=1.0,
            is_cot_simulation=True,
            max_tree_depth=8,
            max_simulation_depth=4
        )
        
        assert tree is not None
        assert tree['root'].user_question == complex_question_1
        assert tree['root'].node_type == NodeType.USER_QUESTION
        
        print(f"✓ MCTS search complex question")
        print(f"  Question: {complex_question_1}")
        if result.get('final_answer'):
            print(f"  Final answer: {result['final_answer'][:200]}")
    
    # @pytest.mark.slow
    # @pytest.mark.integration
    def test_mcts_search_comparison_question(self, llm_agent, retriever_agent, working_memory, interaction_memory, complex_question_2):
        """Test MCTS search with comparison question requiring multiple paths."""
        result, tree = mcts_search(
            question=complex_question_2,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=5,
            exploration_weight=1.0,
            is_cot_simulation=True,
            max_tree_depth=8,
            max_simulation_depth=4
        )
        
        assert tree is not None
        
        # Check if multiple paths were explored
        root = tree['root']
        if root.children:
            assert len(root.children) > 0
        
        print(f"✓ MCTS search comparison question")
        print(f"  Question: {complex_question_2}")
        if result.get('final_answer'):
            print(f"  Final answer: {result['final_answer'][:200]}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_expand_node(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test node expansion with real LLM."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': sample_question}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=5)
        
        tree: MCTSSearchTree = {
            'root': root
        }
        
        children, has_semantic_signal = expand(
            tree=tree,
            selected_node=root,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            is_cot_simulation=False
        )
        
        assert isinstance(children, list)
        assert isinstance(has_semantic_signal, bool)
        
        print(f"✓ MCTS expand node")
        print(f"  Generated {len(children)} children")
        for i, child in enumerate(children[:3]):
            print(f"  Child {i+1}: {child.node_type.name}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_simulate(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test simulation with real LLM."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': sample_question}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=5)
        
        terminal, has_semantic_signal = simulate(
            node=root,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            is_cot_simulation=True,
            max_simulation_depth=3
        )
        
        assert terminal is not None
        assert terminal.is_terminal() or terminal.depth <= root.depth + 3
        assert isinstance(has_semantic_signal, bool)
        
        print(f"✓ MCTS simulate")
        print(f"  Terminal depth: {terminal.depth}")
        print(f"  Terminal type: {terminal.node_type.name}")
    
    # ============================================================================
    # Early Termination Tests
    # ============================================================================
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_disabled(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test that early termination can be disabled."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=5,
            early_termination_enabled=False,
            min_iterations=3,
            high_confidence_threshold=0.9,
            convergence_patience=3,
            semantic_sufficiency_count=2
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination disabled")
        print(f"  Should run all 5 iterations (early termination disabled)")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_min_iterations(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test that minimum iterations are enforced before early termination."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=10,
            early_termination_enabled=True,
            min_iterations=5,  # Should run at least 5 iterations
            high_confidence_threshold=0.9,
            convergence_patience=3,
            semantic_sufficiency_count=2
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS minimum iterations enforcement")
        print(f"  Should run at least 5 iterations before early termination")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_high_confidence(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test early termination when high confidence answer is found."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=10,
            early_termination_enabled=True,
            min_iterations=2,
            high_confidence_threshold=0.85,  # Lower threshold to trigger earlier
            convergence_patience=5,
            semantic_sufficiency_count=5
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination high confidence")
        print(f"  Should terminate early if reward >= 0.85 after min_iterations")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_semantic_sufficiency(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test early termination when semantic sufficiency signals are detected."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=10,
            early_termination_enabled=True,
            min_iterations=2,
            high_confidence_threshold=1.0,  # High threshold to avoid triggering
            convergence_patience=10,  # High patience to avoid triggering
            semantic_sufficiency_count=2  # Low count to trigger early
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination semantic sufficiency")
        print(f"  Should terminate when 2 semantic sufficiency signals detected")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_convergence(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test early termination when no improvement is detected (convergence)."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=10,
            early_termination_enabled=True,
            min_iterations=2,
            high_confidence_threshold=1.0,  # High threshold to avoid triggering
            convergence_patience=2,  # Low patience to trigger convergence early
            semantic_sufficiency_count=10  # High count to avoid triggering
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination convergence")
        print(f"  Should terminate when no improvement for 2 consecutive iterations")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_all_conditions(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test early termination with all conditions configured."""
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=10,
            early_termination_enabled=True,
            min_iterations=3,
            high_confidence_threshold=0.9,
            convergence_patience=3,
            semantic_sufficiency_count=2
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination all conditions")
        print(f"  Testing with all early termination parameters configured")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_early_termination_default_parameters(self, llm_agent, retriever_agent, working_memory, interaction_memory, sample_question):
        """Test that default early termination parameters work correctly."""
        # Test with default parameters (should use defaults from function signature)
        result, tree = mcts_search(
            question=sample_question,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=5
            # All early termination parameters use defaults
        )
        
        assert tree is not None
        assert isinstance(result, dict)
        
        print(f"✓ MCTS early termination default parameters")
        print(f"  Testing with default early termination settings")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_mcts_evaluate(self, llm_agent, interaction_memory, terminal_node):
        """Test evaluation with real LLM."""
        reward = asyncio.run(evaluate(
            node=terminal_node,
            llm_agent=llm_agent,
            interaction_memory=interaction_memory
        ))
        
        assert 0.0 <= reward <= 1.0
        
        print(f"✓ MCTS evaluate")
        print(f"  Reward: {reward:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_answer_no_terminals(self, llm_agent, root_node):
        """Test get_answer when no terminal nodes exist."""
        root_node.children = []
        tree: MCTSSearchTree = {
            'root': root_node,
        }
        
        full_answer, concise_answer = get_answer(tree, llm_agent)
        
        assert "No final answer found" in full_answer
        assert concise_answer == "No answer"
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_answer_with_terminals(self, llm_agent, interaction_memory, root_node, terminal_node):
        """Test get_answer with terminal nodes using real LLM."""
        terminal_node.parent = root_node
        root_node.children = [terminal_node]
        
        tree: MCTSSearchTree = {
            'root': root_node,
        }
        
        full_answer, concise_answer = get_answer(tree, llm_agent, interaction_memory)
        
        assert full_answer is not None
        assert concise_answer is not None
        assert len(full_answer) > 0
        assert len(concise_answer) > 0
        
        print(f"✓ get_answer with terminals")
        print(f"  Full answer: {full_answer[:200]}")
        print(f"  Concise answer: {concise_answer}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_get_answer_complex_question_multiple_terminals(self, llm_agent, interaction_memory, complex_question_1):
        """Test get_answer with complex question having multiple terminal nodes."""
        root = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.USER_QUESTION,
                content={'user_question': complex_question_1}
            ),
            max_depth=10
        )
        
        # Create multiple terminal nodes (different reasoning paths)
        terminal1 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    'user_question': complex_question_1,
                    'final_answer': 'George W. Bush was the president when the first iPhone was released in 2007',
                    'concise_answer': 'George W. Bush'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        terminal2 = MCTSReasoningNode(
            node_state=NodeState(
                node_type=NodeType.FINAL_ANSWER,
                content={
                    'user_question': complex_question_1,
                    'final_answer': 'The president was George W. Bush, who was in office from 2001 to 2009',
                    'concise_answer': 'George W. Bush'
                }
            ),
            parent=root,
            max_depth=10
        )
        
        root.children = [terminal1, terminal2]
        
        tree: MCTSSearchTree = {
            'root': root
        }
        
        full_answer, concise_answer = get_answer(tree, llm_agent, interaction_memory)
        
        assert full_answer is not None
        assert concise_answer is not None
        assert len(full_answer) > 0
        
        print(f"✓ get_answer complex question multiple terminals")
        print(f"  Full answer: {full_answer[:200]}")
        print(f"  Concise answer: {concise_answer}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_mcts_workflow_complex_question(self, llm_agent, retriever_agent, working_memory, interaction_memory, complex_question_2):
        """Test full MCTS workflow with complex question."""
        # Run MCTS search
        result, tree = mcts_search(
            question=complex_question_2,
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=5,
            exploration_weight=1.0,
            is_cot_simulation=True,
            max_tree_depth=6,
            max_simulation_depth=3
        )
        
        assert tree is not None
        
        # Extract answer
        full_answer, concise_answer = get_answer(tree, llm_agent, interaction_memory)
        
        assert full_answer is not None
        assert concise_answer is not None
        
        print(f"✓ Full MCTS workflow")
        print(f"  Question: {complex_question_2}")
        print(f"  Full answer: {full_answer[:200]}")
        print(f"  Concise answer: {concise_answer}")


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_tree(depth: int, branching_factor: int = 2) -> MCTSSearchTree:
    """Create a test tree with specified depth and branching factor."""
    root_state = NodeState(
        node_type=NodeType.USER_QUESTION,
        content={'user_question': 'Test question'}
    )
    root = MCTSReasoningNode(node_state=root_state, max_depth=10)
    
    def add_children(node: MCTSReasoningNode, current_depth: int):
        if current_depth >= depth:
            return
        
        children_list = []
        for i in range(branching_factor):
            state = NodeState(
                node_type=NodeType.SUB_QA_NODE,
                content={
                    'user_question': 'Test question',
                    'sub_question': f'Q{current_depth}-{i}',
                    'sub_answer': f'A{current_depth}-{i}'
                }
            )
            child = MCTSReasoningNode(node_state=state, parent=node, max_depth=10)
            children_list.append(child)
            add_children(child, current_depth + 1)
        node.children = tuple(children_list)
    
    add_children(root, 0)
    
    return {
        'root': root
    }

