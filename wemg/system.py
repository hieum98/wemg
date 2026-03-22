"""WEMG System - main orchestrator for question answering."""

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from wemg.config import WEMGConfig
from wemg.llm.client import LLMClient
from wemg.retrieval.web_search import WebSearchTool
from wemg.retrieval.corpus import CorpusRetriever
from wemg.retrieval.reranker import Reranker
from wemg.retrieval.wikidata import WikidataClient
from wemg.reasoning.cot import cot_search, cot_get_answer
from wemg.reasoning.mcts import mcts_search, get_answer
from wemg.reasoning.memory import WorkingMemory, InteractionMemory
from wemg.reasoning.nodes import ReasoningNode
from wemg.utils.graph import textualize_graph, visualize_graph

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    question: str
    answer: str
    concise_answer: str
    reasoning: Optional[str] = None
    search_tree: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    working_memory: Optional[WorkingMemory] = None


class WEMGSystem:
    """WEMG Question Answering System."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_overrides: Optional[List[str]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        config: Optional[WEMGConfig] = None,
    ):
        if config is not None:
            self.cfg = config
        elif config_dict is not None:
            self.cfg = WEMGConfig.from_dict(config_dict)
        else:
            self.cfg = WEMGConfig.from_yaml(config_path, config_overrides)
        
        self._setup_logging()
        errors = self.cfg.validate_config()
        if errors:
            raise ValueError("Config validation failed: " + "; ".join(errors))
        
        self.client: Optional[LLMClient] = None
        self.retriever = None
        self.reranker: Optional[Reranker] = None
        self.interaction_memory: Optional[InteractionMemory] = None
        self.wikidata_client: Optional[WikidataClient] = None
        self._initialized = False
    
    def _setup_logging(self):
        log_level = getattr(logging, self.cfg.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=self.cfg.logging.format)
    
    def _initialize(self):
        if self._initialized:
            return
        
        logger.info("Initializing WEMG system...")
        self.client = self._create_client()
        self.retriever = self._create_retriever()
        self.reranker = self._create_reranker()
        self.wikidata_client = self._create_wikidata_client()

        if self.cfg.memory.interaction_memory.scope == "dataset":
            self.interaction_memory = self._create_interaction_memory("dataset_memory")
        
        self._initialized = True
        logger.info("WEMG system initialized.")
    
    def _create_client(self) -> LLMClient:
        cache_config = None
        if self.cfg.cache.enabled:
            cache_config = {
                "enabled": True,
                "host": self.cfg.cache.host,
                "port": self.cfg.cache.port,
                "db": self.cfg.cache.db,
                "password": self.cfg.cache.password,
                "prefix": self.cfg.cache.prefix,
                "ttl": self.cfg.cache.ttl,
            }
        
        gen = self.cfg.llm.generation
        return LLMClient(
            model_name=self.cfg.llm.model_name,
            url=self.cfg.llm.url,
            api_key=self.cfg.llm.api_key,
            concurrency=self.cfg.llm.concurrency,
            max_retries=self.cfg.llm.max_retries,
            cache_config=cache_config,
            timeout=gen.timeout,
            temperature=gen.temperature,
            n=gen.n,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens,
            max_input_tokens=gen.max_input_tokens,
            top_k=gen.top_k,
            enable_thinking=gen.enable_thinking,
            random_seed=gen.random_seed,
        )
    
    def _create_retriever(self):
        if self.cfg.retriever.type == "web_search":
            tool = WebSearchTool(
                api_key=self.cfg.retriever.web_search.api_key,
                max_crawl_requests_per_second=self.cfg.retriever.web_search.max_crawl_requests_per_second,
            )
            return tool
        elif self.cfg.retriever.type == "corpus":
            c = self.cfg.retriever.corpus
            return CorpusRetriever(
                embedder_config={
                    "model_name": c.embedder.model_name,
                    "url": c.embedder.url,
                    "api_key": c.embedder.api_key,
                },
                corpus_path=c.corpus_path,
                index_path=c.index_path,
                embedder_type=c.embedder.embedder_type,
            )
        raise ValueError(f"Unknown retriever type: {self.cfg.retriever.type}")

    def _create_reranker(self) -> Optional[Reranker]:
        """Create reranker from config; returns None if disabled."""
        r = self.cfg.reranker
        if not r.enabled:
            return None
        reranker_client = LLMClient(
            model_name=r.model_name,
            url=r.url,
            api_key=r.api_key,
            concurrency=r.concurrency,
            temperature=0.0,
            max_tokens=1,
        )
        return Reranker(client=reranker_client, top_k=r.top_k, instruction=r.instruction)

    def _create_working_memory(self) -> WorkingMemory:
        return WorkingMemory(
            max_textual_memory_tokens=self.cfg.memory.working_memory.max_textual_memory_tokens,
            wikidata_client=self.wikidata_client,
        )
    
    def _create_interaction_memory(self, collection_name: str = None) -> Optional[InteractionMemory]:
        im = self.cfg.memory.interaction_memory
        if not im.enabled:
            return None
        if collection_name is None:
            collection_name = f"interaction_memory_{uuid.uuid4().hex[:8]}"
        return InteractionMemory(
            db_path=im.db_path,
            collection_name=collection_name,
            token_budget=im.token_budget,
            is_local_embedding_api=im.is_local_embedding_api,
            embedding_model_name=im.embedding_model_name,
            embedding_base_url=im.embedding_base_url,
            embedding_api_key=im.embedding_api_key,
            enable_embedding_cache=im.enable_embedding_cache,
        )
    
    def _get_node_gen_kwargs(self) -> Dict:
        ng = self.cfg.node_generation
        return {
            "n": ng.n,
            "n_subquestions": ng.n_subquestions,
            "top_k_websearch": ng.top_k_websearch,
            "top_k_entities": ng.top_k_entities,
            "n_hops": ng.n_hops,
            "entity_linking_method": ng.entity_linking_method,
            "rerank_kb_documents": ng.rerank_kb_documents,
            "azure_endpoint": ng.azure_endpoint,
            "azure_key": ng.azure_key,
            "max_crawl_requests_per_second": self.cfg.retriever.web_search.max_crawl_requests_per_second,
        }

    def _create_wikidata_client(self) -> WikidataClient:
        """Create a shared WikidataClient instance for the system.

        This client is reused across working memories and node generators so that
        Wikipedia/Wikidata rate limiting and connection pooling are coordinated
        at the process level.
        """
        max_rps: Optional[float] = None
        # When using web search, reuse the same crawl rate for Wikipedia fetches.
        if self.cfg.retriever.type == "web_search":
            max_rps = self.cfg.retriever.web_search.max_crawl_requests_per_second
        return WikidataClient(max_wikipedia_requests_per_second=max_rps)
    
    def answer(self, question: str, question_id: str = None, golden_answer: Optional[str] = None) -> AnswerResult:
        self._initialize()
        
        working_memory = self._create_working_memory()
        
        if self.cfg.memory.interaction_memory.scope == "dataset" and self.interaction_memory:
            interaction_memory = self.interaction_memory
        else:
            interaction_memory = self._create_interaction_memory(collection_name=question_id)
        
        node_gen_kwargs = self._get_node_gen_kwargs()
        strategy = self.cfg.search.strategy
        
        if strategy == "cot":
            return self._answer_with_cot(question, question_id, working_memory, interaction_memory, node_gen_kwargs, golden_answer)
        elif strategy == "mcts":
            return self._answer_with_mcts(question, question_id, working_memory, interaction_memory, node_gen_kwargs, golden_answer)
        raise ValueError(f"Unknown strategy: {strategy}")
    
    def _answer_with_cot(self, question, question_id, working_memory, interaction_memory, kwargs, golden_answer):
        max_depth = self.cfg.search.cot.max_depth
        terminal_content, reasoning_path, pass_at_k = cot_search(
            question=question, client=self.client, retriever=self.retriever,
            wikidata_client=self.wikidata_client, reranker=self.reranker,
            working_memory=working_memory, interaction_memory=interaction_memory,
            max_depth=max_depth, correct_answers=golden_answer, **kwargs,
        )
        
        if self.cfg.output.show_search_tree and reasoning_path:
            reasoning_path[0].print_tree()

        full_answer, concise_answer = cot_get_answer(terminal_content, reasoning_path)
        return AnswerResult(
            question=question, answer=full_answer, concise_answer=concise_answer,
            search_tree=reasoning_path[0] if self.cfg.output.include_reasoning else None,
            metadata={"question_id": question_id, "strategy": "cot", "pass_at_k": pass_at_k, "num_steps": len(reasoning_path)},
            working_memory=working_memory,
        )
    
    def _answer_with_mcts(self, question, question_id, working_memory, interaction_memory, kwargs, golden_answer):
        mcts_cfg = self.cfg.search.mcts
        et = mcts_cfg.early_termination
        
        best_content, root, pass_at_k = mcts_search(
            question=question, client=self.client, retriever=self.retriever,
            wikidata_client=self.wikidata_client, reranker=self.reranker,
            working_memory=working_memory, interaction_memory=interaction_memory,
            num_iterations=mcts_cfg.num_iterations, max_tree_depth=mcts_cfg.max_tree_depth,
            max_simulation_depth=mcts_cfg.max_simulation_depth, exploration_weight=mcts_cfg.exploration_weight,
            golden_answer=golden_answer if mcts_cfg.use_golden_answer_for_reward else None,
            correct_answers=golden_answer,
            early_termination_enabled=et.enabled, min_iterations=et.min_iterations,
            high_confidence_threshold=et.high_confidence_threshold,
            convergence_patience=et.convergence_patience,
            semantic_sufficiency_count=et.semantic_sufficiency_count,
            **kwargs,
        )
        
        if self.cfg.output.show_search_tree:
            root.print_tree()
        
        full_answer, concise_answer = get_answer(root, self.client, interaction_memory)
        return AnswerResult(
            question=question, answer=full_answer, concise_answer=concise_answer,
            search_tree=root if self.cfg.output.include_reasoning else None,
            metadata={"question_id": question_id, "strategy": "mcts", "pass_at_k": pass_at_k},
            working_memory=working_memory,
        )
    
    def close(self):
        if self.client:
            self.client.close()
        if self.reranker:
            self.reranker.close()
            self.reranker = None
        if self.interaction_memory:
            should_delete = self.cfg.memory.interaction_memory.scope != "dataset"
            self.interaction_memory.release(should_delete_db=should_delete)
            self.interaction_memory = None


def answer_question(question: str, config_path=None, config_overrides=None, **kwargs) -> str:
    system = WEMGSystem(config_path=config_path, config_overrides=config_overrides)
    try:
        result = system.answer(question, **kwargs)
        return result.concise_answer or result.answer
    finally:
        system.close()


def answer_questions_batch(questions: List[str], config_path=None, config_overrides=None, max_workers=None, **kwargs) -> List[AnswerResult]:
    if not questions:
        return []
    system = WEMGSystem(config_path=config_path, config_overrides=config_overrides)
    system._initialize()
    
    if max_workers is None:
        max_workers = min(len(questions), system.cfg.llm.concurrency, 8)
    max_workers = max(1, min(max_workers, len(questions)))
    
    results = [None] * len(questions)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(system.answer, q, **kwargs): i for i, q in enumerate(questions)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = AnswerResult(question=questions[idx], answer=f"Error: {e}", concise_answer=f"Error: {e}", metadata={"error": str(e)})
        return results
    finally:
        system.close()
