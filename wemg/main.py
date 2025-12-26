"""WEMG Main Module - Question Answering with Graph-Enhanced Retrieval.

This module provides the main interface for using WEMG to answer questions.
It supports both MCTS and Chain-of-Thought (CoT) reasoning strategies.

Usage:
    # Basic usage
    from wemg.main import answer_question, WEMGSystem
    
    answer = answer_question("What is the capital of France?")
    
    # Using the WEMGSystem class for more control
    system = WEMGSystem(config_path="path/to/config.yaml")
    result = system.answer("What is the capital of France?")
    
    # With Hydra
    python -m wemg.main question="What is the capital of France?"
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool
from wemg.config import (
    create_config_from_dict,
    get_cache_config,
    get_generation_kwargs,
    get_node_generation_kwargs,
    load_config,
    validate_config,
)
from wemg.runners.cot import cot_search, cot_get_answer
from wemg.runners.mcts import mcts_search, get_answer
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.base_reasoning_node import BaseReasoningNode
from wemg.utils.graph_utils import textualize_graph, visualize_graph
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Result container for question answering."""
    question: str
    answer: str
    concise_answer: str
    reasoning: Optional[str] = None
    search_tree: Optional[Dict, BaseReasoningNode] = None
    metadata: Optional[Dict[str, Any]] = None
    working_memory: Optional[WorkingMemory] = None


class WEMGSystem:
    """WEMG Question Answering System.
    
    This class encapsulates the entire WEMG pipeline for question answering,
    including LLM agent, retriever, and search strategy configuration.
    
    Attributes:
        cfg: Configuration object
        llm_agent: Language model agent
        retriever_agent: Retriever agent (web search or corpus-based)
        
    Example:
        >>> system = WEMGSystem()
        >>> result = system.answer("What is machine learning?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_overrides: Optional[List[str]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the WEMG system.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default.
            config_overrides: List of override strings (e.g., ["llm.temperature=0.5"])
            config_dict: Dictionary configuration to merge with defaults.
        """
        # Load configuration
        if config_dict is not None:
            self.cfg = create_config_from_dict(config_dict)
        else:
            self.cfg = load_config(config_path, config_overrides)
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        validate_config(self.cfg)
        
        # Initialize components
        self.llm_agent: Optional[BaseLLMAgent] = None
        self.retriever_agent: Optional[Union[WebSearchTool, RetrieverAgent]] = None
        
        # Lazy initialization flag
        self._initialized = False
    
    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        log_level_str = OmegaConf.select(self.cfg, "logging.level") or "INFO"
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        log_format = OmegaConf.select(self.cfg, "logging.format") or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=log_level,
            format=log_format,
        )
        # Set environment variable for other modules
        os.environ["LOGGING_LEVEL"] = log_level_str
    
    def _initialize(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return
        
        logger.info("Initializing WEMG system...")
        
        # Initialize LLM agent
        self.llm_agent = self._create_llm_agent()
        
        # Initialize retriever agent
        self.retriever_agent = self._create_retriever_agent()
        
        self._initialized = True
        logger.info("WEMG system initialized successfully.")
    
    def _create_llm_agent(self) -> BaseLLMAgent:
        """Create the LLM agent based on configuration."""
        cache_config = get_cache_config(self.cfg)
        generation_kwargs = get_generation_kwargs(self.cfg)
        
        return BaseLLMAgent(
            client_type=OmegaConf.select(self.cfg, "llm.client_type") or "openai",
            model_name=OmegaConf.select(self.cfg, "llm.model_name") or "gpt-4o-mini",
            url=OmegaConf.select(self.cfg, "llm.url") or "https://api.openai.com/v1",
            api_key=OmegaConf.select(self.cfg, "llm.api_key"),
            concurrency=OmegaConf.select(self.cfg, "llm.concurrency") or 64,
            max_retries=OmegaConf.select(self.cfg, "llm.max_retries") or 3,
            cache_config=cache_config,
            **generation_kwargs,
        )
    
    def _create_retriever_agent(self) -> Union[WebSearchTool, RetrieverAgent]:
        """Create the retriever agent based on configuration."""
        retriever_type = OmegaConf.select(self.cfg, "retriever.type") or "web_search"
        
        if retriever_type == "web_search":
            retriever = WebSearchTool()
            api_key = OmegaConf.select(self.cfg, "retriever.web_search.api_key")
            if api_key:
                retriever.serper_api_key = api_key
            return retriever
        
        elif retriever_type == "corpus":
            embedder_config = {
                "model_name": OmegaConf.select(self.cfg, "retriever.corpus.embedder.model_name") or "text-embedding-3-small",
                "url": OmegaConf.select(self.cfg, "retriever.corpus.embedder.url") or "https://api.openai.com/v1",
                "api_key": OmegaConf.select(self.cfg, "retriever.corpus.embedder.api_key"),
            }
            corpus_path = OmegaConf.select(self.cfg, "retriever.corpus.corpus_path")
            index_path = OmegaConf.select(self.cfg, "retriever.corpus.index_path")
            return RetrieverAgent(
                embedder_config=embedder_config,
                corpus_path=Path(corpus_path) if corpus_path else None,
                index_path=Path(index_path) if index_path else None,
                embedder_type=OmegaConf.select(self.cfg, "retriever.corpus.embedder.embedder_type") or "openai",
            )
        
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def _create_working_memory(self) -> WorkingMemory:
        """Create a fresh working memory instance."""
        return WorkingMemory(
            max_textual_memory_tokens=OmegaConf.select(self.cfg, "memory.working_memory.max_textual_memory_tokens") or 16384,
        )
    
    def _create_interaction_memory(self, collection_name: str = None) -> Optional[InteractionMemory]:
        """Create interaction memory if enabled."""
        enabled = OmegaConf.select(self.cfg, "memory.interaction_memory.enabled") or False
        if not enabled:
            return None
        
        # Get all interaction memory configuration parameters
        db_path = OmegaConf.select(self.cfg, "memory.interaction_memory.db_path")
        if collection_name is None:
            collection_name = f"interaction_memory_{uuid.uuid4().hex[:8]}"
        token_budget = OmegaConf.select(self.cfg, "memory.interaction_memory.token_budget") or 8192
        is_local_embedding_api = OmegaConf.select(self.cfg, "memory.interaction_memory.is_local_embedding_api") or False
        embedding_model_name = OmegaConf.select(self.cfg, "memory.interaction_memory.embedding_model_name") or "Qwen/Qwen3-Embedding-0.6B"
        embedding_base_url = OmegaConf.select(self.cfg, "memory.interaction_memory.embedding_base_url") or "http://localhost:8000/v1"
        embedding_api_key = OmegaConf.select(self.cfg, "memory.interaction_memory.embedding_api_key") or "EMPTY"
        enable_embedding_cache = OmegaConf.select(self.cfg, "memory.interaction_memory.enable_embedding_cache")
        if enable_embedding_cache is None:
            enable_embedding_cache = True
        
        return InteractionMemory(
            db_path=db_path,
            collection_name=collection_name,
            token_budget=token_budget,
            is_local_embedding_api=is_local_embedding_api,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            enable_embedding_cache=enable_embedding_cache,
        )
    
    def answer(
        self,
        question: str,
        question_id: str = None,
        golden_answer: Optional[str] = None,
    ) -> AnswerResult:
        """Answer a question using the configured search strategy.
        
        Args:
            question: The question to answer.
            golden_answer: Optional ground truth answer for evaluation (MCTS only).
        
        Returns:
            AnswerResult containing the answer and metadata.
        
        Example:
            >>> system = WEMGSystem()
            >>> result = system.answer("What is the speed of light?")
            >>> print(result.concise_answer)
        """
        # Lazy initialization
        self._initialize()
        
        # Create fresh memory instances for each question
        working_memory = self._create_working_memory()
        interaction_memory = self._create_interaction_memory(collection_name=question_id)
        
        # Get node generation kwargs
        node_gen_kwargs = get_node_generation_kwargs(self.cfg)
        
        # Choose search strategy
        strategy = OmegaConf.select(self.cfg, "search.strategy") or "cot"
        
        if strategy == "cot":
            return self._answer_with_cot(
                question_id=question_id,
                question=question,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                node_gen_kwargs=node_gen_kwargs,
                golden_answer=golden_answer,
            )
        elif strategy == "mcts":
            return self._answer_with_mcts(
                question_id=question_id,
                question=question,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                node_gen_kwargs=node_gen_kwargs,
                golden_answer=golden_answer,
            )
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def _answer_with_cot(
        self,
        question_id: str,
        question: str,
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory],
        node_gen_kwargs: Dict[str, Any],
        golden_answer: Optional[str] = None,
    ) -> AnswerResult:
        """Answer using Chain-of-Thought reasoning."""
        logger.info(f"Answering with CoT strategy: {question}")
        
        max_depth = OmegaConf.select(self.cfg, "search.cot.max_depth") or 10
        
        # Run CoT search
        terminal_content, reasoning_path = cot_search(
            question=question,
            llm_agent=self.llm_agent,
            retriever_agent=self.retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            max_depth=max_depth,
            **node_gen_kwargs,
        )
        
        show_search_tree = OmegaConf.select(self.cfg, "output.show_search_tree")
        if show_search_tree is None:
            show_search_tree = False
        if show_search_tree:
            root_node = reasoning_path[0]
            print(f"================================================")
            print(f"CoT Search for question {question_id} with golden answer {golden_answer}: ")
            print(f"================================================")
            root_node.print_tree()
            print(f"================================================")
        # Extract answer
        full_answer, concise_answer = cot_get_answer(terminal_content, reasoning_path)
        
        include_reasoning = OmegaConf.select(self.cfg, "output.include_reasoning")
        if include_reasoning is None:
            include_reasoning = True

        return AnswerResult(
            question=question,
            answer=full_answer,
            concise_answer=concise_answer,
            search_tree=reasoning_path[0] if include_reasoning else None, # the root node of the search tree
            metadata={
                "question_id": question_id,
                "strategy": "cot",
                "max_depth": max_depth,
                "num_reasoning_steps": len(reasoning_path),
            },
            working_memory=working_memory,
        )
    
    def _answer_with_mcts(
        self,
        question_id: str,
        question: str,
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory],
        node_gen_kwargs: Dict[str, Any],
        golden_answer: Optional[str] = None,
    ) -> AnswerResult:
        """Answer using Monte Carlo Tree Search."""
        logger.info(f"Answering with MCTS strategy: {question}")
        
        # Run MCTS search
        use_golden_answer_for_reward = OmegaConf.select(self.cfg, "search.mcts.use_golden_answer_for_reward")
        if use_golden_answer_for_reward is None:
            use_golden_answer_for_reward = False
        
        # Early termination configuration
        early_termination_cfg = OmegaConf.select(self.cfg, "search.mcts.early_termination")
        early_termination_enabled = (
            OmegaConf.select(early_termination_cfg, "enabled")
            if early_termination_cfg is not None
            else True
        )
        min_iterations = (
            OmegaConf.select(early_termination_cfg, "min_iterations")
            if early_termination_cfg is not None
            else 3
        )
        high_confidence_threshold = (
            OmegaConf.select(early_termination_cfg, "high_confidence_threshold")
            if early_termination_cfg is not None
            else 0.9
        )
        convergence_patience = (
            OmegaConf.select(early_termination_cfg, "convergence_patience")
            if early_termination_cfg is not None
            else 3
        )
        semantic_sufficiency_count = (
            OmegaConf.select(early_termination_cfg, "semantic_sufficiency_count")
            if early_termination_cfg is not None
            else 4
        )
        
        best_terminal_content, search_tree = mcts_search(
            question=question,
            llm_agent=self.llm_agent,
            retriever_agent=self.retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=OmegaConf.select(self.cfg, "search.mcts.num_iterations") or 10,
            max_tree_depth=OmegaConf.select(self.cfg, "search.mcts.max_tree_depth") or 10,
            max_simulation_depth=OmegaConf.select(self.cfg, "search.mcts.max_simulation_depth") or 5,
            exploration_weight=OmegaConf.select(self.cfg, "search.mcts.exploration_weight") or 1.0,
            is_cot_simulation=OmegaConf.select(self.cfg, "search.mcts.is_cot_simulation")
            if OmegaConf.select(self.cfg, "search.mcts.is_cot_simulation") is not None
            else True,
            golden_answer=golden_answer if use_golden_answer_for_reward else None,
            early_termination_enabled=early_termination_enabled,
            min_iterations=min_iterations,
            high_confidence_threshold=high_confidence_threshold,
            convergence_patience=convergence_patience,
            semantic_sufficiency_count=semantic_sufficiency_count,
            **node_gen_kwargs,
        )
        
        show_search_tree = OmegaConf.select(self.cfg, "output.show_search_tree")
        if show_search_tree is None:
            show_search_tree = False
        if show_search_tree:
            root_node = search_tree['root']
            print(f"================================================")
            print(f"MCTS Search for question {question_id} with golden answer {golden_answer}: ")
            print(f"================================================")
            root_node.print_tree()
            print(f"================================================")

        # Get final answer from tree
        full_answer, concise_answer = get_answer(
            tree=search_tree,
            llm_agent=self.llm_agent,
            interaction_memory=interaction_memory,
        )
        
        include_reasoning = OmegaConf.select(self.cfg, "output.include_reasoning")
        if include_reasoning is None:
            include_reasoning = True
        return AnswerResult(
            question=question,
            answer=full_answer,
            concise_answer=concise_answer,
            search_tree=search_tree if include_reasoning else None,
            metadata={
                "strategy": "mcts",
                "num_iterations": OmegaConf.select(self.cfg, "search.mcts.num_iterations") or 10,
                "max_tree_depth": OmegaConf.select(self.cfg, "search.mcts.max_tree_depth") or 10,
                "exploration_weight": OmegaConf.select(self.cfg, "search.mcts.exploration_weight") or 1.0,
            },
            working_memory=working_memory,
        )
    
    def close(self) -> None:
        """Clean up resources."""
        if self.llm_agent:
            self.llm_agent.close()
        logger.info("WEMG system closed.")


def answer_question(
    question: str,
    config_path: Optional[Union[str, Path]] = None,
    config_overrides: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Convenience function to answer a question.
    
    This is the simplest interface to WEMG. For repeated queries,
    use the WEMGSystem class directly to avoid reinitializing components.
    
    Args:
        question: The question to answer.
        config_path: Optional path to configuration file.
        config_overrides: Optional list of config overrides.
        **kwargs: Additional keyword arguments passed to WEMGSystem.answer()
    
    Returns:
        The answer string.
    
    Example:
        >>> answer = answer_question("What is the capital of France?")
        >>> print(answer)
        Paris
    """
    system = WEMGSystem(config_path=config_path, config_overrides=config_overrides)
    try:
        result = system.answer(question, **kwargs)
        return result.concise_answer if result.concise_answer else result.answer
    finally:
        system.close()


def answer_questions_batch(
    questions: List[str],
    config_path: Optional[Union[str, Path]] = None,
    config_overrides: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    **kwargs,
) -> List[AnswerResult]:
    """Answer multiple questions efficiently with parallel processing.
    
    Uses a single WEMGSystem instance for all questions to avoid
    repeated initialization overhead. Questions are processed in parallel
    using ThreadPoolExecutor for improved performance.
    
    Args:
        questions: List of questions to answer.
        config_path: Optional path to configuration file.
        config_overrides: Optional list of config overrides.
        max_workers: Maximum number of parallel workers. If None, uses
            min(len(questions), llm.concurrency, 8) for optimal performance.
        **kwargs: Additional keyword arguments passed to WEMGSystem.answer()
    
    Returns:
        List of AnswerResult objects in the same order as input questions.
    
    Example:
        >>> questions = ["What is AI?", "What is ML?"]
        >>> results = answer_questions_batch(questions)
        >>> for r in results:
        ...     print(f"Q: {r.question}")
        ...     print(f"A: {r.concise_answer}")
    """
    if not questions:
        return []
    
    system = WEMGSystem(config_path=config_path, config_overrides=config_overrides)
    
    # Determine optimal number of workers
    if max_workers is None:
        # Use LLM concurrency setting if available, otherwise default to 8
        llm_concurrency = OmegaConf.select(system.cfg, "llm.concurrency") or 64
        # Use up to 8 workers by default, but respect question count and LLM concurrency limits
        max_workers = min(len(questions), llm_concurrency, 8)
    max_workers = max(1, min(max_workers, len(questions)))  # Ensure valid range
    
    # Initialize system before parallel processing
    system._initialize()
    
    # Prepare results list to maintain order
    results: List[Optional[AnswerResult]] = [None] * len(questions)
    
    try:
        # Process questions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions
            future_to_index = {
                executor.submit(system.answer, question, **kwargs): idx
                for idx, question in enumerate(questions)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    logger.info(f"Completed question {idx + 1}/{len(questions)}: {result.question[:50]}...")
                except Exception as e:
                    logger.error(f"Error processing question {idx + 1}/{len(questions)}: {questions[idx][:50]}...: {e}")
                    # Create error result to maintain order
                    results[idx] = AnswerResult(
                        question=questions[idx],
                        answer=f"Error: {str(e)}",
                        concise_answer=f"Error: {str(e)}",
                        metadata={"error": str(e), "index": idx}
                    )
        return results
    finally:
        system.close()


# =====================================================================
# Hydra Entry Point
# =====================================================================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration.
    
    Run with:
        python -m wemg.main question="Your question here"
        python -m wemg.main question="Your question" llm.model_name=gpt-4o
        python -m wemg.main question="Your question" search.strategy=mcts
    """
    # Get question from config or command line
    question = OmegaConf.select(cfg, "question")
    
    if question is None:
        print("Usage: python -m wemg.main question=\"Your question here\"")
        print("\nExample:")
        print("  python -m wemg.main question=\"What is the capital of France?\"")
        print("\nOverride configuration:")
        print("  python -m wemg.main question=\"...\" llm.model_name=gpt-4o")
        print("  python -m wemg.main question=\"...\" search.strategy=mcts")
        return
    
    # Remove the 'question' key from config before passing to WEMGSystem
    # since it's not a system configuration parameter
    system_cfg = OmegaConf.to_container(cfg, resolve=True)
    system_cfg.pop("question", None)
    system_cfg.pop("hydra", None)  # Remove hydra internal config
    
    # Create system with the configuration
    system = WEMGSystem(config_dict=system_cfg)
    
    try:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Strategy: {OmegaConf.select(cfg, 'search.strategy') or 'cot'}")
        print(f"Model: {OmegaConf.select(cfg, 'llm.model_name') or 'gpt-4o-mini'}")
        print(f"{'='*60}\n")
        
        result = system.answer(question)
        
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(f"\n{result.answer}\n")
        
        include_concise = OmegaConf.select(cfg, "output.include_concise_answer")
        if include_concise is None:
            include_concise = True
        if include_concise and result.concise_answer:
            print(f"{'='*60}")
            print("CONCISE ANSWER:")
            print(f"{'='*60}")
            print(f"\n{result.concise_answer}\n")
        
        # Log memory (textual and graph)
        if result.working_memory:
            print(f"{'='*60}")
            print("TEXTUAL MEMORY:")
            print(f"{'='*60}")
            if result.working_memory.textual_memory:
                formatted_textual = result.working_memory.format_textual_memory()
                print(f"\n{formatted_textual}\n")
            else:
                print("\n(No textual memory)\n")
            
            print(f"{'='*60}")
            print("GRAPH MEMORY:")
            print(f"{'='*60}")
            if result.working_memory.graph_memory and result.working_memory.graph_memory.number_of_nodes() > 0:
                # Convert graph to textual representation
                graph_text_parts = []
                for comp in nx.weakly_connected_components(result.working_memory.graph_memory):
                    triples, cluster_text = textualize_graph(comp, result.working_memory.graph_memory, method='dfs')
                    if triples:
                        graph_text_parts.append(cluster_text)
                
                if graph_text_parts:
                    graph_text = "\n\n".join(graph_text_parts)
                    print(f"\n{graph_text}\n")
                else:
                    print("\n(No graph triples found)\n")
                
                # Visualize the graph
                print(f"{'='*60}")
                print("GRAPH MEMORY VISUALIZATION:")
                print(f"{'='*60}")
                visualize_graph(
                    result.working_memory.graph_memory,
                    title=f"Graph Memory - Question: {question[:50]}",
                    save_path="./tmp"
                )
            else:
                print("\n(No graph memory)\n")
    finally:
        system.close()


if __name__ == "__main__":
    main()
