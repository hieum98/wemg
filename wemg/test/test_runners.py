"""
Comprehensive tests for the runners module (CoT and MCTS reasoning).

These are real integration tests that call actual LLM servers, retrievers, and web search.
"""
import os
import pytest
import asyncio
from typing import List, Dict

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool, DDGSAPIWrapper
from wemg.agents.tools.wikidata import WikidataEntity, WikidataProperty, WikiTriple
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.base_reasoning_node import NodeType, NodeState
from wemg.runners.cot import (
    CoTReasoningNode, 
    cot_search, 
    cot_get_answer
)
from wemg.runners.mcts import (
    MCTSReasoningNode,
    MCTSSearchTree,
    select,
    expand,
    simulate,
    evaluate,
    evaluate_sync,
    mcts_search,
    get_answer
)
from wemg.agents import roles
from wemg.utils.preprocessing import get_node_id


# Test configuration - adjust these for your environment
TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0142:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-32B")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0372:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")


# WorkingMemory and InteractionMemory tests have been moved to test_memory.py


class TestCoTReasoningNode:
    """Test suite for CoTReasoningNode."""
    
    def test_node_creation(self):
        """Test creating a CoT reasoning node."""
        node_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'What is the capital of France?'}
        )
        node = CoTReasoningNode(node_state=node_state, max_depth=5)
        
        assert node.node_state.node_type == NodeType.USER_QUESTION
        assert node.user_question == 'What is the capital of France?'
        assert node.max_depth == 5
        assert node.depth == 0
    
    def test_is_terminal(self):
        """Test terminal node detection."""
        # Non-terminal node
        user_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'Test question'}
        )
        user_node = CoTReasoningNode(node_state=user_state)
        assert not user_node.is_terminal()
        
        # Terminal node
        final_state = NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={'user_question': 'Test', 'final_answer': 'Answer'}
        )
        final_node = CoTReasoningNode(node_state=final_state)
        assert final_node.is_terminal()
    
    def test_parent_child_relationship(self):
        """Test parent-child relationships."""
        parent_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'Main question'}
        )
        child_state = NodeState(
            node_type=NodeType.SUB_QA_NODE,
            content={'user_question': 'Main question', 'sub_question': 'Sub Q', 'sub_answer': 'Sub A'}
        )
        
        parent = CoTReasoningNode(node_state=parent_state)
        child = CoTReasoningNode(node_state=child_state)
        child.parent = parent
        
        assert child.parent == parent
        assert child in parent.children
        assert child.depth == 1


class TestMCTSReasoningNode:
    """Test suite for MCTSReasoningNode."""
    
    def test_node_creation(self):
        """Test creating an MCTS reasoning node."""
        node_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'What is 2+2?'}
        )
        node = MCTSReasoningNode(node_state=node_state, max_depth=10)
        
        assert node.value == 0.0
        assert node.visits == 0
        assert node.max_depth == 10
    
    def test_backpropagate(self):
        """Test backpropagation of rewards."""
        # Create a chain: root -> child
        root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': 'Q'})
        child_state = NodeState(node_type=NodeType.SUB_QA_NODE, 
                               content={'user_question': 'Q', 'sub_question': 'SQ', 'sub_answer': 'SA'})
        
        root = MCTSReasoningNode(node_state=root_state)
        child = MCTSReasoningNode(node_state=child_state)
        child.parent = root
        
        # Backpropagate reward from child
        child.backpropagate(0.8)
        
        assert child.visits == 1
        assert child.value == 0.8
        assert root.visits == 1
        assert root.value == 0.8
    
    def test_upper_confidence_bound(self):
        """Test UCT calculation."""
        root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': 'Q'})
        child_state = NodeState(node_type=NodeType.SUB_QA_NODE,
                               content={'user_question': 'Q', 'sub_question': 'SQ', 'sub_answer': 'SA'})
        
        root = MCTSReasoningNode(node_state=root_state)
        child = MCTSReasoningNode(node_state=child_state)
        child.parent = root
        
        # Set up visits
        root.visits = 10
        child.visits = 3
        child.value = 0.6
        
        uct = child.upper_confidence_bound(exploration_weight=1.0)
        assert uct > 0  # Should be positive
        
        # Unvisited child returns its value directly
        child2_state = NodeState(node_type=NodeType.SUB_QA_NODE,
                                content={'user_question': 'Q', 'sub_question': 'SQ2', 'sub_answer': 'SA2'})
        child2 = MCTSReasoningNode(node_state=child2_state)
        child2.parent = root
        child2.value = 0.5
        
        uct2 = child2.upper_confidence_bound()
        assert uct2 == 0.5  # Returns value when visits=0


class TestMCTSFunctions:
    """Test MCTS helper functions."""
    
    def test_select_empty_tree(self):
        """Test selection on a tree with only root."""
        root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': 'Q'})
        root = MCTSReasoningNode(node_state=root_state)
        
        tree: MCTSSearchTree = {'root': root, 'explored_nodes': set()}
        path = select(tree)
        
        assert len(path) == 1
        assert path[0] == root
    
    def test_select_with_children(self):
        """Test selection with explored children."""
        root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': 'Q'})
        child1_state = NodeState(node_type=NodeType.SUB_QA_NODE,
                                content={'user_question': 'Q', 'sub_question': 'SQ1', 'sub_answer': 'SA1'})
        child2_state = NodeState(node_type=NodeType.SUB_QA_NODE,
                                content={'user_question': 'Q', 'sub_question': 'SQ2', 'sub_answer': 'SA2'})
        
        root = MCTSReasoningNode(node_state=root_state)
        child1 = MCTSReasoningNode(node_state=child1_state)
        child2 = MCTSReasoningNode(node_state=child2_state)
        
        child1.parent = root
        child2.parent = root
        
        # Mark root as explored, children as unexplored
        tree: MCTSSearchTree = {'root': root, 'explored_nodes': {root}}
        
        path = select(tree)
        # Should select one of the unexplored children
        assert len(path) == 1
        assert path[-1] in [child1, child2]


class TestCoTSearchIntegration:
    """Integration tests for CoT search with real LLM calls."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.fixture
    def working_memory(self):
        """Create an empty working memory."""
        return WorkingMemory()
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an interaction memory."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_cot_search_simple_question(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test CoT search with a simple factual question."""
        question = "What is the capital of France?"
        
        terminal_content, reasoning_path = cot_search(
            question=question,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            max_depth=3
        )
        
        # Verify we got a result
        assert terminal_content is not None
        assert 'final_answer' in terminal_content
        assert len(reasoning_path) >= 1
        
        # Check answer mentions Paris
        answer = terminal_content.get('final_answer', '').lower()
        assert 'paris' in answer or len(answer) > 0
        
        print(f"✓ CoT Search completed")
        print(f"  Question: {question}")
        print(f"  Answer: {terminal_content.get('final_answer', 'N/A')[:200]}")
        print(f"  Reasoning steps: {len(reasoning_path)}")
    
    @pytest.mark.slow
    def test_cot_search_multi_hop_question(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test CoT search with a multi-hop question requiring reasoning."""
        question = "Who was the president of the United States when the iPhone was first released?"
        
        terminal_content, reasoning_path = cot_search(
            question=question,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            max_depth=5
        )
        
        assert terminal_content is not None
        assert len(reasoning_path) >= 1
        
        print(f"✓ Multi-hop CoT Search completed")
        print(f"  Question: {question}")
        print(f"  Reasoning steps: {len(reasoning_path)}")
        
        # Print reasoning path summary
        for i, node in enumerate(reasoning_path[:3]):
            node_type = node.node_state.node_type.name
            print(f"  Step {i+1}: {node_type}")
    
    @pytest.mark.slow
    def test_cot_get_answer(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test extracting answer from CoT result."""
        question = "What year did World War 2 end?"
        
        terminal_content, reasoning_path = cot_search(
            question=question,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            max_depth=3
        )
        
        full_answer, concise_answer = cot_get_answer(terminal_content, reasoning_path)
        
        assert full_answer != "No final answer found."
        assert concise_answer != "No answer"
        
        print(f"✓ CoT answer extraction")
        print(f"  Full answer: {full_answer[:200]}")
        print(f"  Concise answer: {concise_answer}")


class TestMCTSSearchIntegration:
    """Integration tests for MCTS search with real LLM calls."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.fixture
    def working_memory(self):
        """Create an empty working memory."""
        return WorkingMemory()
    
    @pytest.fixture
    def interaction_memory(self):
        """Create an interaction memory."""
        return InteractionMemory()
    
    @pytest.mark.slow
    def test_mcts_search_simple(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test MCTS search with a simple question and few iterations."""
        question = "What is the largest planet in our solar system?"
        
        best_content, tree = mcts_search(
            question=question,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=2,
            exploration_weight=1.0,
            is_cot_simulation=True,
            max_tree_depth=3,
            max_simulation_depth=2
        )
        
        assert best_content is not None
        assert tree is not None
        assert tree['root'] is not None
        
        print(f"✓ MCTS Search completed")
        print(f"  Question: {question}")
        print(f"  Best answer: {best_content.get('final_answer', 'N/A')[:200]}")
        print(f"  Explored nodes: {len(tree['explored_nodes'])}")
    
    @pytest.mark.slow
    def test_mcts_evaluate(self, llm_agent):
        """Test evaluation of a terminal node."""
        # Create a final answer node
        final_state = NodeState(
            node_type=NodeType.FINAL_ANSWER,
            content={
                'user_question': 'What is 2+2?',
                'final_answer': '4',
                'concise_answer': '4',
                'reasoning': 'Simple arithmetic'
            }
        )
        final_node = MCTSReasoningNode(node_state=final_state)
        
        # Evaluate with golden answer
        reward = evaluate_sync(
            node=final_node,
            llm_agent=llm_agent,
            golden_answer="4"
        )
        
        assert 0.0 <= reward <= 1.0
        print(f"✓ MCTS Evaluation: reward={reward}")
    
    @pytest.mark.slow
    def test_mcts_get_answer(self, llm_agent, web_search_tool, working_memory, interaction_memory):
        """Test getting synthesized answer from MCTS tree."""
        question = "What is the speed of light?"
        
        best_content, tree = mcts_search(
            question=question,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=2,
            max_tree_depth=3,
            max_simulation_depth=2
        )
        
        # Get synthesized answer from tree
        answer, short_answer = get_answer(tree, llm_agent, interaction_memory)
        
        assert answer is not None
        assert short_answer is not None
        
        print(f"✓ MCTS get_answer")
        print(f"  Answer: {answer[:200] if answer else 'N/A'}")
        print(f"  Short answer: {short_answer}")


class TestExpandAndSimulate:
    """Test expand and simulate functions."""
    
    @pytest.fixture
    def llm_agent(self):
        """Create a BaseLLMAgent for testing."""
        return BaseLLMAgent(
            model_name=TEST_LLM_MODEL,
            url=TEST_LLM_API_BASE,
            api_key=TEST_LLM_API_KEY,
            temperature=0.7,
            max_tokens=4096,
            concurrency=2,
            max_retries=3
        )
    
    @pytest.fixture
    def web_search_tool(self):
        """Create a WebSearchTool for testing."""
        return WebSearchTool(
            api_wrapper=DDGSAPIWrapper(),
            max_tokens=8192
        )
    
    @pytest.fixture
    def working_memory(self):
        """Create an empty working memory."""
        return WorkingMemory()
    
    @pytest.mark.slow
    def test_expand_from_root(self, llm_agent, web_search_tool, working_memory):
        """Test expanding children from root node."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'What is machine learning?'}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=5)
        
        tree: MCTSSearchTree = {'root': root, 'explored_nodes': set()}
        
        children = expand(
            tree=tree,
            selected_node=root,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            is_cot_simulation=False
        )
        
        assert len(children) > 0
        assert root in tree['explored_nodes']
        
        # Check children are properly linked
        for child in children:
            assert child.parent == root
        
        print(f"✓ Expand generated {len(children)} children")
        for i, child in enumerate(children):
            print(f"  Child {i+1}: {child.node_state.node_type.name}")
    
    @pytest.mark.slow
    def test_simulate_cot_mode(self, llm_agent, web_search_tool, working_memory):
        """Test simulation in CoT mode."""
        root_state = NodeState(
            node_type=NodeType.USER_QUESTION,
            content={'user_question': 'Who invented the telephone?'}
        )
        root = MCTSReasoningNode(node_state=root_state, max_depth=5)
        
        terminal_node = simulate(
            node=root,
            llm_agent=llm_agent,
            retriever_agent=web_search_tool,
            working_memory=working_memory,
            is_cot_simulation=True,
            max_simulation_depth=3
        )
        
        assert terminal_node is not None
        print(f"✓ Simulation reached node type: {terminal_node.node_state.node_type.name}")
        if terminal_node.is_terminal():
            print(f"  Final answer: {terminal_node.node_state.content.get('final_answer', 'N/A')[:100]}")


# Run tests with: pytest test_runners.py -v -s --tb=short
# Run slow tests: pytest test_runners.py -v -s --tb=short -m slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
