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

import hydra
from omegaconf import DictConfig, OmegaConf

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools.web_search import WebSearchTool
from wemg.config import (
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


logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Result container for question answering."""
    question: str
    answer: str
    concise_answer: str
    reasoning_path: Optional[List[Any]] = None
    search_tree: Optional[Dict] = None
    metadata: Optional[Dict[str, Any]] = None


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
            from wemg.config import create_config_from_dict
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
        log_level = getattr(logging, self.cfg.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format=self.cfg.logging.format,
        )
        # Set environment variable for other modules
        os.environ["LOGGING_LEVEL"] = self.cfg.logging.level
    
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
            client_type=self.cfg.llm.client_type,
            model_name=self.cfg.llm.model_name,
            url=self.cfg.llm.url,
            api_key=self.cfg.llm.api_key,
            concurrency=self.cfg.llm.concurrency,
            max_retries=self.cfg.llm.max_retries,
            cache_config=cache_config,
            **generation_kwargs,
        )
    
    def _create_retriever_agent(self) -> Union[WebSearchTool, RetrieverAgent]:
        """Create the retriever agent based on configuration."""
        retriever_type = self.cfg.retriever.type
        
        if retriever_type == "web_search":
            retriever = WebSearchTool()
            if self.cfg.retriever.web_search.api_key:
                retriever.serper_api_key = self.cfg.retriever.web_search.api_key
            return retriever
        
        elif retriever_type == "corpus":
            embedder_config = {
                "model_name": self.cfg.retriever.corpus.embedder.model_name,
                "url": self.cfg.retriever.corpus.embedder.url,
                "api_key": self.cfg.retriever.corpus.embedder.api_key,
            }
            return RetrieverAgent(
                embedder_config=embedder_config,
                corpus_path=Path(self.cfg.retriever.corpus.corpus_path),
                index_path=Path(self.cfg.retriever.corpus.index_path) if self.cfg.retriever.corpus.index_path else None,
                embedder_type=self.cfg.retriever.corpus.embedder.embedder_type,
            )
        
        elif retriever_type == "hybrid":
            # For hybrid, we default to web search but could implement both
            logger.warning("Hybrid retriever not fully implemented, using web search.")
            retriever = WebSearchTool()
            if self.cfg.retriever.web_search.api_key:
                retriever.serper_api_key = self.cfg.retriever.web_search.api_key
            return retriever
        
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def _create_working_memory(self) -> WorkingMemory:
        """Create a fresh working memory instance."""
        return WorkingMemory(
            max_textual_memory_tokens=self.cfg.memory.working_memory.max_textual_memory_tokens,
        )
    
    def _create_interaction_memory(self) -> Optional[InteractionMemory]:
        """Create interaction memory if enabled."""
        if not self.cfg.memory.interaction_memory.enabled:
            return None
        
        return InteractionMemory(
            log_dir=self.cfg.memory.interaction_memory.log_dir,
        )
    
    def answer(
        self,
        question: str,
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
        interaction_memory = self._create_interaction_memory()
        
        # Get node generation kwargs
        node_gen_kwargs = get_node_generation_kwargs(self.cfg)
        
        # Choose search strategy
        strategy = self.cfg.search.strategy
        
        if strategy == "cot":
            return self._answer_with_cot(
                question=question,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                node_gen_kwargs=node_gen_kwargs,
            )
        elif strategy == "mcts":
            return self._answer_with_mcts(
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
        question: str,
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory],
        node_gen_kwargs: Dict[str, Any],
    ) -> AnswerResult:
        """Answer using Chain-of-Thought reasoning."""
        logger.info(f"Answering with CoT strategy: {question}")
        
        max_depth = self.cfg.search.cot.max_depth
        
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
        
        # Extract answer
        full_answer, concise_answer = cot_get_answer(terminal_content, reasoning_path)
        
        return AnswerResult(
            question=question,
            answer=full_answer,
            concise_answer=concise_answer,
            reasoning_path=reasoning_path if self.cfg.output.include_reasoning else None,
            metadata={
                "strategy": "cot",
                "max_depth": max_depth,
                "num_reasoning_steps": len(reasoning_path),
            },
        )
    
    def _answer_with_mcts(
        self,
        question: str,
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory],
        node_gen_kwargs: Dict[str, Any],
        golden_answer: Optional[str] = None,
    ) -> AnswerResult:
        """Answer using Monte Carlo Tree Search."""
        logger.info(f"Answering with MCTS strategy: {question}")
        
        mcts_cfg = self.cfg.search.mcts
        
        # Run MCTS search
        best_terminal_content, search_tree = mcts_search(
            question=question,
            llm_agent=self.llm_agent,
            retriever_agent=self.retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            num_iterations=mcts_cfg.num_iterations,
            max_tree_depth=mcts_cfg.max_tree_depth,
            max_simulation_depth=mcts_cfg.max_simulation_depth,
            exploration_weight=mcts_cfg.exploration_weight,
            is_cot_simulation=mcts_cfg.is_cot_simulation,
            golden_answer=golden_answer,
            **node_gen_kwargs,
        )
        
        # Get final answer from tree
        full_answer, concise_answer = get_answer(
            tree=search_tree,
            llm_agent=self.llm_agent,
            interaction_memory=interaction_memory,
        )
        
        return AnswerResult(
            question=question,
            answer=full_answer,
            concise_answer=concise_answer,
            search_tree=search_tree if self.cfg.output.include_reasoning else None,
            metadata={
                "strategy": "mcts",
                "num_iterations": mcts_cfg.num_iterations,
                "max_tree_depth": mcts_cfg.max_tree_depth,
                "exploration_weight": mcts_cfg.exploration_weight,
            },
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
    **kwargs,
) -> List[AnswerResult]:
    """Answer multiple questions efficiently.
    
    Uses a single WEMGSystem instance for all questions to avoid
    repeated initialization overhead.
    
    Args:
        questions: List of questions to answer.
        config_path: Optional path to configuration file.
        config_overrides: Optional list of config overrides.
        **kwargs: Additional keyword arguments passed to WEMGSystem.answer()
    
    Returns:
        List of AnswerResult objects.
    
    Example:
        >>> questions = ["What is AI?", "What is ML?"]
        >>> results = answer_questions_batch(questions)
        >>> for r in results:
        ...     print(f"Q: {r.question}")
        ...     print(f"A: {r.concise_answer}")
    """
    system = WEMGSystem(config_path=config_path, config_overrides=config_overrides)
    results = []
    try:
        for question in questions:
            logger.info(f"Processing question: {question}")
            result = system.answer(question, **kwargs)
            results.append(result)
    finally:
        system.close()
    
    return results


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
    question = cfg.get("question", None)
    
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
    from wemg.config import create_config_from_dict
    system = WEMGSystem(config_dict=system_cfg)
    
    try:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Strategy: {cfg.search.strategy}")
        print(f"Model: {cfg.llm.model_name}")
        print(f"{'='*60}\n")
        
        result = system.answer(question)
        
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(f"\n{result.answer}\n")
        
        if cfg.output.include_concise_answer and result.concise_answer:
            print(f"{'='*60}")
            print("CONCISE ANSWER:")
            print(f"{'='*60}")
            print(f"\n{result.concise_answer}\n")
        
        if cfg.output.verbose and result.metadata:
            print(f"{'='*60}")
            print("METADATA:")
            print(f"{'='*60}")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
    
    finally:
        system.close()


if __name__ == "__main__":
    main()
