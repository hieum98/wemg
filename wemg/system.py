"""WEMG System - main orchestrator for question answering."""

import asyncio
import logging
import os
import re
import hashlib
import uuid
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from wemg.config import WEMGConfig
from wemg.llm.client import LLMClient
from wemg.retrieval.web_search import WebSearchTool
from wemg.retrieval.corpus import CorpusRetriever
from wemg.retrieval.reranker import Reranker
from wemg.retrieval.wikidata import WikidataClient
from wemg.reasoning.cot import cot_search, cot_get_answer
from wemg.reasoning.mcts import mcts_search, get_answer
from wemg.reasoning.memory import WorkingMemory, InteractionMemory
logger = logging.getLogger(__name__)


def _sanitize_chroma_collection_name(
    name: Optional[str],
    *,
    fallback_prefix: str = "interaction_memory",
) -> str:
    """Return a Chroma-compatible collection name.

    Chroma names must be 3-512 chars, use [a-zA-Z0-9._-], and start/end
    with an alphanumeric character.
    """
    if name is None:
        return f"{fallback_prefix}_{uuid.uuid4().hex[:8]}"

    raw = str(name).strip()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]

    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)

    candidate = sanitized
    if (
        not candidate
        or len(candidate) < 3
        or not candidate[0].isalnum()
        or not candidate[-1].isalnum()
    ):
        candidate = f"{fallback_prefix}_{sanitized}" if sanitized else f"{fallback_prefix}_{digest}"
        candidate = re.sub(r"[^a-zA-Z0-9._-]+", "_", candidate)
        candidate = re.sub(r"^[^a-zA-Z0-9]+", "", candidate)
        candidate = re.sub(r"[^a-zA-Z0-9]+$", "", candidate)

    if len(candidate) < 3:
        candidate = f"{fallback_prefix}_{digest}"

    if len(candidate) > 512:
        head = candidate[: 512 - 1 - len(digest)]
        candidate = f"{head}_{digest}"
        candidate = re.sub(r"[^a-zA-Z0-9]+$", "", candidate)
        if len(candidate) < 3:
            candidate = f"{fallback_prefix}_{digest}"

    return candidate


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
        log_level = getattr(logging, self.cfg.logging.level.upper(), logging.CRITICAL)
        logging.basicConfig(level=log_level, format=self.cfg.logging.format)
    
    def _initialize(self):
        if self._initialized:
            return
        
        logger.info("Initializing WEMG system...")
        logger.info(f"Config: {self.cfg}")
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
                query_cache_ttl=self.cfg.retriever.web_search.query_cache_ttl,
                url_cache_ttl=self.cfg.retriever.web_search.url_cache_ttl,
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

    def _create_working_memory(self, wikidata_client=None) -> WorkingMemory:
        wm_cfg = self.cfg.memory.working_memory
        return WorkingMemory(
            max_textual_memory_tokens=wm_cfg.max_textual_memory_tokens,
            wikidata_client=wikidata_client or self.wikidata_client,
            annotate_steps=wm_cfg.annotate_steps,
        )

    def _create_embed_client(self):
        """Return an LLMClient configured for embeddings (corpus embedder settings)."""
        from wemg.llm.client import LLMClient
        ec = self.cfg.retriever.corpus.embedder
        return LLMClient(
            model_name=ec.model_name,
            url=ec.url,
            api_key=ec.api_key or self.cfg.llm.api_key,
            is_embedding=True,
        )


    def _create_interaction_memory(self, collection_name: str = None) -> Optional[InteractionMemory]:
        im = self.cfg.memory.interaction_memory
        if not im.enabled:
            return None
        if collection_name is None:
            collection_name = f"interaction_memory_{uuid.uuid4().hex[:8]}"
        collection_name = _sanitize_chroma_collection_name(collection_name)
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
            "triple_pruning_delta": ng.triple_pruning_delta,
            "triple_pruning_top_k": ng.triple_pruning_top_k,
            "azure_endpoint": ng.azure_endpoint,
            "azure_key": ng.azure_key,
            "max_crawl_requests_per_second": self.cfg.retriever.web_search.max_crawl_requests_per_second,
            "freebase_qid_to_mid_map_path": ng.freebase_qid_to_mid_map_path,
            "freebase_qid_to_mid_candidates": ng.freebase_qid_to_mid_candidates,
        }

    def _create_wikidata_client(self) -> WikidataClient:
        """Create a shared WikidataClient instance for the system.

        This client is reused across working memories and node generators so that
        Wikipedia/Wikidata rate limiting and connection pooling are coordinated
        at the process level.
        """
        max_rps: Optional[float] = self.cfg.retriever.web_search.max_crawl_requests_per_second
        redis_client = None
        if self.cfg.cache.enabled:
            try:
                import redis as redis_lib
                redis_client = redis_lib.Redis(
                    host=self.cfg.cache.host,
                    port=self.cfg.cache.port,
                    db=self.cfg.cache.db,
                    password=self.cfg.cache.password,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
                redis_client.ping()
            except Exception as e:
                logger.warning("Redis unavailable, triple cache disabled: %s", e)
                redis_client = None
        return WikidataClient(max_wikipedia_requests_per_second=max_rps, redis_client=redis_client)
    
    def answer(
        self,
        question: str,
        question_id: str = None,
        golden_answer: Optional[str] = None,
        subgraph_cache_entry: Optional[Dict[str, Any]] = None,
    ) -> AnswerResult:
        self._initialize()
        t_answer = time.perf_counter()
        logger.info(
            "PROFSTEP system stage=answer_start question_id=%s question=%s",
            str(question_id),
            question[:120].replace("\n", " "),
        )

        # Optionally override the KB client with a per-question freebase subgraph.
        kb_client = None
        if (
            self.cfg.node_generation.kb_source == "freebase_subgraph"
            and subgraph_cache_entry is not None
        ):
            from wemg.retrieval.graph_retriever_client import GraphRetrieverClient
            kb_client = GraphRetrieverClient(
                reranker=self.reranker,
                embed_client=self._create_embed_client(),
            )
            kb_client.load_from_cache(subgraph_cache_entry)
        elif self.cfg.node_generation.kb_source == "freebase_live":
            from wemg.retrieval.freebase_client import FreebaseClient
            kb_client = FreebaseClient(
                sparql_url=self.cfg.node_generation.freebase_sparql_url,
                wikidata_client=self.wikidata_client,
                qid_to_mid_map_path=self.cfg.node_generation.freebase_qid_to_mid_map_path,
                qid_to_mid_candidates=self.cfg.node_generation.freebase_qid_to_mid_candidates,
            )

        if self.cfg.memory.interaction_memory.scope == "dataset" and self.interaction_memory:
            interaction_memory = self.interaction_memory
        else:
            interaction_memory = self._create_interaction_memory(collection_name=question_id)

        node_gen_kwargs = self._get_node_gen_kwargs()
        strategy = self.cfg.search.strategy

        if strategy == "cot":
            working_memory = self._create_working_memory(wikidata_client=kb_client)
            result = asyncio.run(self._answer_with_cot(question, question_id, working_memory, interaction_memory, node_gen_kwargs, golden_answer, wikidata_client=kb_client))
            logger.info(
                "PROFSTEP system stage=answer_done strategy=%s elapsed_ms=%.1f question_id=%s",
                strategy,
                (time.perf_counter() - t_answer) * 1000.0,
                str(question_id),
            )
            return result
        elif strategy == "mcts":
            working_memory = self._create_working_memory(wikidata_client=kb_client)
            result = asyncio.run(self._answer_with_mcts(question, question_id, working_memory, interaction_memory, node_gen_kwargs, golden_answer, wikidata_client=kb_client))
            logger.info(
                "PROFSTEP system stage=answer_done strategy=%s elapsed_ms=%.1f question_id=%s",
                strategy,
                (time.perf_counter() - t_answer) * 1000.0,
                str(question_id),
            )
            return result
        raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _answer_with_cot(self, question, question_id, working_memory, interaction_memory, kwargs, golden_answer, wikidata_client=None):
        max_depth = self.cfg.search.cot.max_depth
        terminal_content, reasoning_path, pass_at_k = await cot_search(
            question=question, client=self.client, retriever=self.retriever,
            wikidata_client=wikidata_client or self.wikidata_client, reranker=self.reranker,
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
    
    async def _answer_with_mcts(self, question, question_id, working_memory, interaction_memory, kwargs, golden_answer, wikidata_client=None):
        mcts_cfg = self.cfg.search.mcts
        et = mcts_cfg.early_termination
        t_mcts = time.perf_counter()
        logger.info("PROFSTEP system stage=mcts_search_start question_id=%s", str(question_id))
        best_content, root, pass_at_k = await mcts_search(
            question=question, client=self.client, retriever=self.retriever,
            wikidata_client=wikidata_client or self.wikidata_client, reranker=self.reranker,
            working_memory=working_memory, interaction_memory=interaction_memory,
            num_iterations=mcts_cfg.num_iterations, max_tree_depth=mcts_cfg.max_tree_depth,
            exploration_weight=mcts_cfg.exploration_weight,
            golden_answer=golden_answer if mcts_cfg.use_golden_answer_for_reward else None,
            max_simulation_depth=mcts_cfg.max_simulation_depth,
            correct_answers=golden_answer,
            early_termination_enabled=et.enabled, min_iterations=et.min_iterations,
            high_confidence_threshold=et.high_confidence_threshold,
            convergence_patience=et.convergence_patience,
            semantic_sufficiency_count=et.semantic_sufficiency_count,
            **kwargs,
        )
        logger.info(
            "PROFSTEP system stage=mcts_search_done question_id=%s elapsed_ms=%.1f pass_at_k=%s",
            str(question_id),
            (time.perf_counter() - t_mcts) * 1000.0,
            str(pass_at_k),
        )

        if self.cfg.output.show_search_tree:
            root.print_tree()
        t_final = time.perf_counter()
        logger.info("PROFSTEP system stage=get_answer_start question_id=%s", str(question_id))
        full_answer, concise_answer = await get_answer(root, self.client, interaction_memory, working_memory=working_memory)
        logger.info(
            "PROFSTEP system stage=get_answer_done question_id=%s elapsed_ms=%.1f",
            str(question_id),
            (time.perf_counter() - t_final) * 1000.0,
        )
        return AnswerResult(
            question=question, answer=full_answer, concise_answer=concise_answer,
            search_tree=root if self.cfg.output.include_reasoning else None,
            metadata={"question_id": question_id, "strategy": "mcts", "pass_at_k": pass_at_k},
            working_memory=working_memory,
        )
    
    def answer_batch(
        self,
        questions: List[str],
        question_ids: Optional[List[str]] = None,
        golden_answers: Optional[List] = None,
        subgraph_cache_entries: Optional[List[Optional[Dict[str, Any]]]] = None,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[AnswerResult]:
        """Answer many questions concurrently (thread pool), sharing one system instance."""
        if not questions:
            return []
        self._initialize()
        if isinstance(self.retriever, CorpusRetriever):
            logger.info("Warming up corpus retriever before batch workers start.")
            self.retriever.warmup()

        n = len(questions)
        if question_ids is not None and len(question_ids) != n:
            raise ValueError("question_ids length must match questions")
        if golden_answers is not None and len(golden_answers) != n:
            raise ValueError("golden_answers length must match questions")
        if subgraph_cache_entries is not None and len(subgraph_cache_entries) != n:
            raise ValueError("subgraph_cache_entries length must match questions")

        if max_workers is None:
            max_workers = min(n, self.cfg.llm.concurrency, 8)
        max_workers = max(1, min(max_workers, n))
        logger.info(f"Answering {n} questions with {max_workers} workers")

        def _answer_at(idx: int) -> AnswerResult:
            q = questions[idx]
            qid = question_ids[idx] if question_ids is not None else str(idx)
            ga = None
            if golden_answers is not None:
                ga = golden_answers[idx]
                if ga is not None:
                    if isinstance(ga, (list, tuple)):
                        ga = list(ga)
                    elif not isinstance(ga, str):
                        ga = str(ga)
            cache_entry = subgraph_cache_entries[idx] if subgraph_cache_entries is not None else None
            return self.answer(q, question_id=qid, golden_answer=ga, subgraph_cache_entry=cache_entry, **kwargs)

        results: List[Optional[AnswerResult]] = [None] * n
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_answer_at, i): i for i in range(n)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = AnswerResult(
                        question=questions[idx],
                        answer=f"Error: {e}",
                        concise_answer=f"Error: {e}",
                        metadata={"error": str(e)},
                    )
        return results  # type: ignore[return-value]
    
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
    try:
        return system.answer_batch(questions, max_workers=max_workers, **kwargs)
    finally:
        system.close()
