# langgraph_coe

LangGraph implementation of **CoE** (Co-Evolving Memory and Graph for RAG
reasoning). It runs the same research idea as the legacy [`coe`](../coe) package —
a textual memory and a graph memory that are continuously synchronized during
reasoning — but the orchestration is expressed as compiled
[LangGraph](https://github.com/langchain-ai/langgraph) state machines with tiered
LLM roles, batched/cached retrieval, and reasoning-aware middleware.

For the repository-level overview and the comparison between the two packages,
see the [root README](../README.md).

## Architecture

A question is answered by one **strategy graph** (MCTS or CoT). Each reasoning
step retrieves evidence, updates the shared memory, and feeds the next step.

For the full LangGraph design — state schemas and reducers, the CoT/MCTS graph
topologies, `Send` fan-out, subgraph composition, and the execution model — see
[`docs/design_langgraph_coe.md`](../docs/design_langgraph_coe.md).

```
answer(question)                         # system.py
  └── build_mcts_graph | build_cot_graph # graphs/{mcts,cot}.py
        ├── subquestion / answer roles   # roles.py via RoleModelRegistry (llm.py)
        ├── kg_search subgraph           # graphs/kg_search.py
        │     NER → link_entities → triple-search agent → enrich
        │     (tools/wikidata*.py: client → backend → bundled properties)
        ├── corpus / web retrieval       # tools/retrieval.py, tools/web.py,
        │                                #   graphs/web_research.py
        ├── rerank (SGLang reranker)     # tools/retrieval.py, tools/wikidata.py
        └── memory_update subgraph       # graphs/memory_update.py
              OpenIE extraction → triple_pruner → consolidation
              (text memory ↔ networkx graph memory)
```

### Module map

| Path | Responsibility |
|------|----------------|
| `system.py` | Public `answer()` / `answer_batch()`, runtime init, `AnswerResult`. |
| `graphs/cot.py` | Chain-of-thought strategy graph. |
| `graphs/mcts.py` | Monte Carlo Tree Search strategy graph (CoT rollouts). |
| `graphs/kg_search.py` | NER → entity linking → triple-search agent → enrichment. |
| `graphs/memory_update.py` | Text↔graph memory synchronization. |
| `graphs/web_research.py` | Optional web-search fan-out subgraph. |
| `roles.py` | All LLM `Role`s with explicit Pydantic input/output models. |
| `llm.py` | `RoleModelRegistry`, role execution, structured-output parsing/retry. |
| `config.py` / `config.yaml` | `LangGraphCoeConfig` schema and defaults. |
| `tools/wikidata*.py` | Async Wikidata access: client, wire backend, bundled property metadata. |
| `tools/retrieval.py` | Corpus FAISS retrieval + reranker client. |
| `tools/web.py` | Web search (Serper, DDG fallback). |
| `tools/cache.py` | Redis-backed dict cache for Wikidata/web. |
| `evaluation/` | Dataset evaluation CLI and runner. |

## Public API

The API is async. A single question:

```python
import asyncio
from langgraph_coe.system import answer

result = asyncio.run(answer("Who directed Inception?"))
print(result.answer)          # headline answer
print(result.concise_answer)  # short-form answer
print(result.metadata)        # strategy, iterations, best_value, …
```

Many questions concurrently (order preserved, bounded by `max_workers`):

```python
import asyncio
from langgraph_coe.system import answer_batch

results = asyncio.run(answer_batch(["Q1", "Q2", "Q3"], max_workers=4))
```

Pass a `LangGraphCoeConfig` to either call to override the defaults loaded from
`config.yaml`.

## Configuration

- **Schema:** `config.py` (`LangGraphCoeConfig`); **defaults:** `config.yaml`.
- **Overrides:** dotted `key=value` (used by the evaluation CLI and tests), e.g.
  `search.strategy=cot`, `search.mcts.num_iterations=8`, `cache.enabled=true`.
- **Secrets:** `API_KEY` / `OPENAI_API_KEY` are read from the repo-root `.env`.

### LLM tiers

Roles are mapped to four tiers via `llm.role_tiers` (unknown roles fall back to
`heavy`); edit `llm.tiers.*` to set the model, sampling knobs, and thinking
budget per tier. With the default `config.yaml` routing:

| Tier | Roles | Thinking |
|------|-------|----------|
| `heavy` | answer/final-answer synthesis, subquestion generation, self-correction, memory consolidation, OpenIE, passage extraction, KG triple-search agent | on, large budget |
| `medium` | relation extraction, verification, evaluation, web research | on |
| `light` | KG NER agent | on, small budget |
| `classify` | bounded fixed-shape outputs (triple pruning, NER list, query rephrasing) | on, small budget |

`role_tiers` is the source of truth for routing; the same role names also have
defaults in `config.py`. Generation knobs (`temperature`, `top_p`, `top_k`, `min_p`, the penalties,
`seed`, `max_tokens`, `enable_thinking`, `thinking_budget`) are per-tier; omit a
knob to use the server default. `top_k` / `min_p` / `repetition_penalty` are
non-OpenAI knobs forwarded to SGLang — see
[`docs/setup_generation_params.md`](../docs/setup_generation_params.md) and
[`docs/setup_thinking_budget.md`](../docs/setup_thinking_budget.md).

### Retrieval and search

- `search.strategy` selects `mcts` or `cot`; per-strategy knobs live under
  `search.mcts` / `search.cot`.
- `wikidata.*` configures the SPARQL endpoint, rate limits, and Stage-A triple
  reranking.
- `retriever.corpus.*` configures the FAISS corpus index and embedder;
  `reranker.*` configures the document reranker.
- `web_search.enabled` toggles the web fan-out (off by default for KG+corpus
  paper parity).
- `cache.enabled` toggles the Redis caches for Wikidata and web tools.

## Running

CLI evaluation over a dataset:

```bash
conda activate coe   # or: uv run
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=bamboogle \
    output_path=results/lgc_bamboogle \
    search.strategy=mcts
```

The corpus index is large and the model/SPARQL/reranker endpoints must be
reachable, so run on a node with the RAM and network access. See
[`evaluation/README.md`](evaluation/README.md) for all CLI keys and the artifact
layout.

## Infrastructure setup

The system expects an OpenAI-compatible LLM endpoint, a Wikidata SPARQL endpoint,
a reranker endpoint, and (optionally) Redis. Setup guides:

- [`docs/deploy_local_wikidata-v2.md`](../docs/deploy_local_wikidata-v2.md) — local Wikidata QEndpoint
- [`docs/setup_wikidata_tools.md`](../docs/setup_wikidata_tools.md) — Wikidata tool configuration
- [`docs/deploy_reranker_server.md`](../docs/deploy_reranker_server.md) — reranker server
- [`docs/setup_redis_cache.md`](../docs/setup_redis_cache.md) — Redis cache
- [`docs/setup_generation_params.md`](../docs/setup_generation_params.md) — sampling/generation knobs
- [`docs/setup_thinking_budget.md`](../docs/setup_thinking_budget.md) — reasoning-token budget

## Tests

```bash
pytest langgraph_coe/tests/unit          # fast, no servers
pytest langgraph_coe/tests/integration   # one live dependency, rest stubbed
pytest langgraph_coe/tests/real_servers   # full live stack
```

Integration and real-server suites self-skip when their endpoints are
unreachable. See [`tests/README.md`](tests/README.md) for markers and the
`LANGGRAPH_TEST_*` endpoint variables.
