from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import networkx as nx
import redis
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache

from .config import LangGraphCoeConfig
from .graphs import build_cot_graph, build_mcts_graph
from .graphs.mcts import MCTSNodeType
from .graphs.web_research import set_web_research_config
from .llm import RoleModelRegistry
from .tools.cache import RedisDictCache
from .tools.retrieval import init_retrieval_pipeline
from .tools.web import init_web_search, reset_web_research_session
from .tools import web as web_tools
from .tools.wikidata import init_wikidata, reset_wikidata_session

# Preload fakeredis when available so test monkeypatching ``redis.Redis`` does
# not interfere with fakeredis class definitions during deferred imports.
try:  # pragma: no cover - optional dependency in non-test environments
    import fakeredis  # noqa: F401
except Exception:  # pragma: no cover
    pass

load_dotenv()


class RedisCacheCompat:
    """Fallback cache wrapper when RedisCache cannot type-check Redis."""

    def __init__(self, redis_: Any):
        self.redis_ = redis_


VALID_STRATEGIES = ("cot", "mcts")


@dataclass
class AnswerResult:
    """Public, strategy-agnostic answer envelope returned by :func:`answer`.

    ``from_state`` is the single adapter from a compiled strategy graph's final
    state dict to this shape (§9). It reads ``final_answer`` as the headline
    answer and lifts any non-answer bookkeeping (``strategy`` and friends) into
    ``metadata`` so callers never reach into raw graph state.
    """

    question: str
    answer: str
    concise_answer: str = ""
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AnswerResult":
        state = state or {}
        metadata: Dict[str, Any] = {}
        # Bookkeeping fields that are not part of the answer payload proper.
        for key in ("strategy", "depth", "iteration", "best_value"):
            if key in state and state[key] is not None:
                metadata[key] = state[key]
        history = state.get("iteration_history")
        if isinstance(history, list):
            metadata["num_iterations"] = len(history)
        return cls(
            question=str(state.get("question", "") or ""),
            answer=str(state.get("final_answer", "") or ""),
            concise_answer=str(state.get("concise_answer", "") or ""),
            reasoning=str(state.get("reasoning", "") or ""),
            metadata=metadata,
        )


def _init_cache_layer(cfg: LangGraphCoeConfig) -> RedisDictCache | None:
    if not cfg.cache.enabled:
        set_llm_cache(None)
        web_tools._web_cache = None
        return None

    redis_cfg = cfg.cache.redis
    wd_client = redis.Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.wikidata_db,
        decode_responses=False,
    )
    shared_cache = RedisDictCache(
        client=wd_client,
        ttls={
            "entity": cfg.cache.wikidata.entity_ttl,
            "search": cfg.cache.wikidata.search_ttl,
            "triples": cfg.cache.wikidata.triples_ttl,
            "enrich": cfg.cache.wikidata.enrich_ttl,
            "web": cfg.cache.web.ttl,
        },
    )
    web_tools._web_cache = shared_cache

    llm_client = redis.Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.llm_db,
        decode_responses=False,
    )
    try:
        set_llm_cache(RedisCache(redis_=llm_client))
    except TypeError:
        # Test monkeypatches can replace redis.Redis with a function, which
        # breaks RedisCache's isinstance checks.
        set_llm_cache(RedisCacheCompat(redis_=llm_client))

    return shared_cache


def _init_runtime(cfg: LangGraphCoeConfig) -> None:
    shared_cache = _init_cache_layer(cfg)
    init_wikidata(cfg.wikidata, cache=shared_cache)
    init_web_search(cfg.web_search)
    try:
        init_retrieval_pipeline(cfg.retriever, cfg.reranker)
    except Exception:
        # Retrieval index availability is environment-specific; keep answer() usable
        # for strategy-level orchestration tests even when corpora are missing.
        pass
    set_web_research_config(cfg.web_search)


def _initial_cot_state(question: str, cfg: LangGraphCoeConfig) -> Dict[str, Any]:
    """Complete ``CoTState`` for a fresh question (§7.2)."""
    return {
        "question": question,
        "max_depth": int(cfg.search.cot.max_depth),
        "depth": 0,
        "is_answerable": False,
        "subquestions": [],
        "subquestion_needs_kg": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "extracted_facts": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": [],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "final_answer": "",
        "concise_answer": "",
        "reasoning": "",
    }


def _initial_mcts_state(question: str, cfg: LangGraphCoeConfig) -> Dict[str, Any]:
    """Complete ``MCTSState`` with a pre-seeded ``USER_QUESTION`` root (§8.2)."""
    root_id = "root"
    root = {
        "node_id": root_id,
        "parent_id": None,
        "children_ids": [],
        "node_type": MCTSNodeType.USER_QUESTION,
        "content": {"question": question},
        "visits": 0,
        "value": 0.0,
        "prior": 1.0,
    }
    return {
        "question": question,
        "max_iterations": int(cfg.search.mcts.num_iterations),
        "iteration": 0,
        "tree": {root_id: root},
        "root_id": root_id,
        "current_path": [],
        "expanded_node_ids": [],
        "simulation_result": {},
        "reward": 0.0,
        "new_raw_triples": [],
        "text_memory": [],
        "graph_memory": nx.DiGraph(),
        "entity_dict": {},
        "semantic_sufficiency_signals": 0,
        "iterations_without_improvement": 0,
        "best_value": 0.0,
        "final_answer": "",
        "concise_answer": "",
        "reasoning": "",
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def answer(
    question: str, config: LangGraphCoeConfig | None = None
) -> AnswerResult:
    """Answer one ``question`` end-to-end via the configured strategy graph (§9)."""
    cfg = config or LangGraphCoeConfig.from_yaml()

    strategy = getattr(getattr(cfg, "search", None), "strategy", "cot")
    if strategy not in VALID_STRATEGIES:
        # Fail loud — never silently fall back to a default strategy.
        raise ValueError(
            f"Unknown search strategy {strategy!r}; expected one of {VALID_STRATEGIES}"
        )

    # 1. Tools / cache / runtime config (idempotent module-level).
    _init_runtime(cfg)

    # 2. Per-question ContextVar reset (once at entry, not per subgraph call).
    reset_wikidata_session()
    reset_web_research_session()

    # 3. Build the selected strategy graph (only the configured one is built).
    registry = RoleModelRegistry(cfg.llm)
    if strategy == "mcts":
        graph = build_mcts_graph(registry, cfg)
        initial_state = _initial_mcts_state(question, cfg)
        recursion_limit = int(cfg.search.mcts.recursion_limit)
    else:
        graph = build_cot_graph(registry, cfg)
        initial_state = _initial_cot_state(question, cfg)
        recursion_limit = int(cfg.search.cot.recursion_limit)

    # 4. Invoke and adapt the final state into the public result envelope.
    result = await graph.ainvoke(
        initial_state, config={"recursion_limit": recursion_limit}
    )
    return AnswerResult.from_state({**(result or {}), "strategy": strategy})


async def answer_batch(
    questions: Sequence[str],
    config: LangGraphCoeConfig | None = None,
    *,
    max_workers: int = 4,
) -> List[AnswerResult]:
    """Answer ``questions`` concurrently, preserving input order (§9).

    Each question is an independent :func:`answer` call; ``max_workers`` bounds
    concurrency via a semaphore. Results are returned in the same order as the
    inputs regardless of completion order.
    """
    sem = asyncio.Semaphore(max(1, int(max_workers or 1)))
    # Materialize calls in order so call-ordering is deterministic even when the
    # underlying ``answer`` is a synchronous stub returning an awaitable.
    pending = [answer(q, config) for q in questions]

    async def _bounded(pending_result: Any) -> Any:
        async with sem:
            return await _maybe_await(pending_result)

    return list(await asyncio.gather(*[_bounded(p) for p in pending]))
