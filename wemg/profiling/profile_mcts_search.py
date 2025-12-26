"""Profile mcts_search function to identify performance bottlenecks."""
import asyncio
import cProfile
import pstats
import io
from pstats import SortKey
import time
import os
from contextlib import contextmanager
from typing import Dict
from pathlib import Path

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool
from wemg.runners.mcts import mcts_search, select, expand, simulate, evaluate
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory


@contextmanager
def timer(name):
    """Context manager to time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.4f}s")


def profile_mcts_search(
    question: str = None,
    num_iterations: int = 5,
    max_tree_depth: int = 5,
    max_simulation_depth: int = 3,
    use_profiler: bool = True,
    detailed_timing: bool = True
):
    """Profile mcts_search with a test case.
    
    Args:
        question: Question to use for profiling (default: complex multi-hop question)
        num_iterations: Number of MCTS iterations
        max_tree_depth: Maximum tree depth
        max_simulation_depth: Maximum simulation depth
        use_profiler: Whether to use cProfile
        detailed_timing: Whether to show detailed timing breakdown
    """
    # Setup - use environment variables or defaults
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
        max_tokens=65536,
        concurrency=2,
        max_retries=3
    )
    
    # Use RetrieverAgent if available, otherwise WebSearchTool
    retriever_agent = None
    retriever_agent_type = os.getenv("RETRIEVER_AGENT_TYPE", "websearch")
    
    if retriever_agent_type == "retriever":
        # Create embedder config for RetrieverAgent
        embedder_config = {
            'model_name': os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
            'url': os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1"),
            'api_key': os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
            'is_embedding': True,
            'timeout': 60,
        }
        corpus_path = Path(os.getenv("RETRIEVER_CORPUS_PATH", "Hieuman/wiki23-processed"))
        index_path = Path(os.getenv("RETRIEVER_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))
        retriever_agent = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=corpus_path,
            index_path=index_path if index_path.exists() else None
        )
    else:
        serper_api_key = os.getenv("SERPER_API_KEY", "your-serper-api-key")
        retriever_agent = WebSearchTool(serper_api_key=serper_api_key)
    
    working_memory = WorkingMemory()
    interaction_memory = InteractionMemory()
    
    if question is None:
        question = "Which magazine was started first Arthur's Magazine or First for Women?"
    
    print("="*80)
    print("PROFILING mcts_search")
    print("="*80)
    print(f"Question: {question}")
    print(f"Parameters:")
    print(f"  num_iterations: {num_iterations}")
    print(f"  max_tree_depth: {max_tree_depth}")
    print(f"  max_simulation_depth: {max_simulation_depth}")
    print(f"  retriever_agent: {type(retriever_agent).__name__}")
    print()
    
    # Create profiler
    profiler = cProfile.Profile() if use_profiler else None
    
    # Profile the function
    start_time = time.time()
    
    if profiler:
        profiler.enable()
    
    result, tree = mcts_search(
        question=question,
        llm_agent=llm_agent,
        retriever_agent=retriever_agent,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        num_iterations=num_iterations,
        exploration_weight=1.0,
        is_cot_simulation=True,
        max_tree_depth=max_tree_depth,
        max_simulation_depth=max_simulation_depth,
        top_k_websearch=5,
        top_k_entities=1,
        top_k_properties=1,
        n_hops=1,
        n=1
    )
    
    if profiler:
        profiler.disable()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✓ Execution completed in {elapsed_time:.2f} seconds")
    print(f"✓ Result keys: {list(result.keys()) if result else 'None'}")
    if result.get('final_answer'):
        print(f"✓ Final answer preview: {result['final_answer'][:200]}...")
    
    # Count tree statistics
    def count_nodes(node, stats: Dict):
        stats['total'] += 1
        if node.is_terminal():
            stats['terminals'] += 1
        if node.node_type.value == "SUBQUESTION":
            stats['subqa'] += 1
        elif node.node_type.value == "FINAL_ANSWER":
            stats['final_answers'] += 1
        for child in node.children:
            count_nodes(child, stats)
    
    stats = {'total': 0, 'terminals': 0, 'subqa': 0, 'final_answers': 0}
    count_nodes(tree['root'], stats)
    print(f"✓ Tree statistics:")
    print(f"  Total nodes: {stats['total']}")
    print(f"  Terminal nodes: {stats['terminals']}")
    print(f"  Sub-QA nodes: {stats['subqa']}")
    print(f"  Final answer nodes: {stats['final_answers']}")
    print(f"  Average nodes per iteration: {stats['total'] / num_iterations:.2f}")
    
    # Print profiling results
    if profiler:
        # Sort by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(30)  # Top 30 functions
        
        print("\n" + "="*80)
        print("PROFILING RESULTS (sorted by cumulative time)")
        print("="*80)
        print(s.getvalue())
        
        # Also print by total time
        s2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=s2)
        ps2.sort_stats('tottime')
        ps2.print_stats(20)  # Top 20 functions
        
        print("\n" + "="*80)
        print("PROFILING RESULTS (sorted by total time)")
        print("="*80)
        print(s2.getvalue())
        
        # Save detailed profile to file
        profile_file = 'mcts_search_profile.prof'
        profiler.dump_stats(profile_file)
        print(f"\n✓ Detailed profile saved to '{profile_file}'")
        print(f"  View with: python -m pstats {profile_file}")
    
    return result, tree, elapsed_time


def profile_mcts_components(
    question: str = None,
    num_iterations: int = 3,
    max_tree_depth: int = 5,
    max_simulation_depth: int = 3
):
    """Profile individual MCTS components with detailed timing."""
    from wemg.agents.base_llm_agent import BaseLLMAgent
    from wemg.agents.retriever_agent import RetrieverAgent
    from wemg.agents.tools.web_search import WebSearchTool
    from wemg.runners.mcts import MCTSSearchTree, MCTSReasoningNode, NodeState, NodeType
    
    # Setup
    llm_agent = BaseLLMAgent(
        model_name=os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8"),
        url=os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1"),
        api_key=os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
        temperature=0.7,
        max_tokens=65536,
        concurrency=2,
        max_retries=3
    )
    
    retriever_agent_type = os.getenv("RETRIEVER_AGENT_TYPE", "websearch")
    if retriever_agent_type == "retriever":
        # Create embedder config for RetrieverAgent
        embedder_config = {
            'model_name': os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
            'url': os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1"),
            'api_key': os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
            'is_embedding': True,
            'timeout': 60,
        }
        corpus_path = Path(os.getenv("RETRIEVER_CORPUS_PATH", "Hieuman/wiki23-processed"))
        index_path = Path(os.getenv("RETRIEVER_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))
        retriever_agent = RetrieverAgent(
            embedder_config=embedder_config,
            corpus_path=corpus_path,
            index_path=index_path if index_path.exists() else None
        )
    else:
        serper_api_key = os.getenv("SERPER_API_KEY", "your-serper-api-key")
        retriever_agent = WebSearchTool(serper_api_key=serper_api_key)
    
    working_memory = WorkingMemory()
    interaction_memory = InteractionMemory()
    
    if question is None:
        question = "Which magazine was started first Arthur's Magazine or First for Women?"
    
    print("="*80)
    print("DETAILED PROFILING OF MCTS COMPONENTS")
    print("="*80)
    print(f"Question: {question}")
    print(f"num_iterations: {num_iterations}")
    print()
    
    # Initialize tree
    root_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question})
    root = MCTSReasoningNode(node_state=root_state, max_depth=max_tree_depth)
    tree: MCTSSearchTree = {'root': root}
    
    total_start = time.time()
    
    selection_times = []
    expansion_times = []
    simulation_times = []
    evaluation_times = []
    backprop_times = []
    sync_memory_times = []
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Selection
        start = time.time()
        path = select(tree, exploration_weight=1.0)
        selected = path[-1]
        selection_times.append(time.time() - start)
        print(f"  Selection: {selection_times[-1]:.4f}s")
        
        # Expansion
        if not selected.is_terminal():
            start = time.time()
            children = expand(
                tree, selected, llm_agent, retriever_agent, working_memory,
                interaction_memory, is_cot_simulation=False,
                top_k_websearch=5, top_k_entities=1, top_k_properties=1, n_hops=1, n=1
            )
            expansion_times.append(time.time() - start)
            print(f"  Expansion: {expansion_times[-1]:.4f}s")
            if children:
                selected = children[0]
        else:
            expansion_times.append(0.0)
        
        # Simulation
        if not selected.is_terminal():
            start = time.time()
            terminal = simulate(
                selected, llm_agent, retriever_agent, working_memory,
                interaction_memory, is_cot_simulation=True, 
                max_simulation_depth=max_simulation_depth,
                top_k_websearch=5, top_k_entities=1, top_k_properties=1, n_hops=1, n=1
            )
            simulation_times.append(time.time() - start)
            print(f"  Simulation: {simulation_times[-1]:.4f}s")
        else:
            terminal = selected
            simulation_times.append(0.0)
        
        # Evaluation
        start = time.time()
        reward = asyncio.run(evaluate(terminal, llm_agent, interaction_memory))
        evaluation_times.append(time.time() - start)
        print(f"  Evaluation: {evaluation_times[-1]:.4f}s")
        
        # Backpropagation
        start = time.time()
        terminal.backpropagate(reward)
        backprop_times.append(time.time() - start)
        print(f"  Backpropagation: {backprop_times[-1]:.4f}s")
        
        # Sync working memory
        start = time.time()
        working_memory.synchronize_memory(llm_agent, question, interaction_memory)
        sync_memory_times.append(time.time() - start)
        print(f"  Sync working memory: {sync_memory_times[-1]:.4f}s")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"Total time: {total_elapsed:.4f}s")
    print(f"Average time per iteration: {total_elapsed / num_iterations:.4f}s")
    print(f"\nAverage time per component:")
    if selection_times:
        print(f"  Selection: {sum(selection_times) / len(selection_times):.4f}s (total: {sum(selection_times):.4f}s)")
    if expansion_times:
        print(f"  Expansion: {sum(expansion_times) / len(expansion_times):.4f}s (total: {sum(expansion_times):.4f}s)")
    if simulation_times:
        print(f"  Simulation: {sum(simulation_times) / len(simulation_times):.4f}s (total: {sum(simulation_times):.4f}s)")
    if evaluation_times:
        print(f"  Evaluation: {sum(evaluation_times) / len(evaluation_times):.4f}s (total: {sum(evaluation_times):.4f}s)")
    if backprop_times:
        print(f"  Backpropagation: {sum(backprop_times) / len(backprop_times):.4f}s (total: {sum(backprop_times):.4f}s)")
    if sync_memory_times:
        print(f"  Sync memory: {sum(sync_memory_times) / len(sync_memory_times):.4f}s (total: {sum(sync_memory_times):.4f}s)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "components":
        # Profile individual components
        profile_mcts_components(
            num_iterations=int(os.getenv("NUM_ITERATIONS", "3")),
            max_tree_depth=int(os.getenv("MAX_TREE_DEPTH", "5")),
            max_simulation_depth=int(os.getenv("MAX_SIMULATION_DEPTH", "3"))
        )
    else:
        # Full profiling with cProfile
        profile_mcts_search(
            num_iterations=int(os.getenv("NUM_ITERATIONS", "5")),
            max_tree_depth=int(os.getenv("MAX_TREE_DEPTH", "5")),
            max_simulation_depth=int(os.getenv("MAX_SIMULATION_DEPTH", "3")),
            use_profiler=True,
            detailed_timing=True
        )

