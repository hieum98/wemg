# WEMG test suite

Tests are system-level and use real components by default (no broad mocking layer).

Configuration for the core WEMG package comes from `wemg/config.yaml` plus environment variables. **`langgraph_coe`** tests read `langgraph_coe/config.yaml` at the repo root (set `LANGGRAPH_CORPUS_INDEX_PATH` when the corpus FAISS bundle lives elsewhere). During test startup, `tests/conftest.py` loads a repo-root `.env` file if present without overriding variables already set in your shell.

## Quick start

```bash
conda run -n wemg pip install -e ".[dev]"
conda run -n wemg pytest tests/ -m "not slow_integration and not integration"
```

## Test layout

| Directory | Scope |
|-----------|-------|
| `helpers/` | test helpers and bootstrap utilities |
| `langgraph_coe/` | ``langgraph_coe`` YAML + tools (Wikidata, web, corpus) |
| `llm/` | LLM client, parsing, roles, cache |
| `retrieval/` | corpus/web search/Wikidata/entity linking/reranker |
| `reasoning/` | generator and working-memory behavior |
| `system/` | end-to-end orchestration and batch APIs |

## Common test commands

From the repository root:

```bash
conda run -n wemg pip install -e ".[dev]"

# Full suite
conda run -n wemg pytest tests/

# Fast local subset
conda run -n wemg pytest tests/langgraph_coe/test_config_yaml.py tests/llm/test_parsing.py -v

# Skip integration and slow integration
conda run -n wemg pytest tests/ -v -m "not slow_integration and not integration"

# LLM-dependent tests
conda run -n wemg pytest tests/ -v -m "requires_llm"

# Corpus retrieval tests
conda run -n wemg pytest tests/retrieval/test_corpus.py -v -m "requires_corpus"

# Redis cache tests
conda run -n wemg pytest tests/llm/test_cache_redis.py -v -m "requires_redis"

# Slow integration tests
conda run -n wemg pytest tests/ -v -m "slow_integration"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `API_KEY` or `OPENAI_API_KEY` | LLM and embedding API key |
| `LLM_URL` or `OPENAI_BASE_URL` | Optional model endpoint override |
| `SERPER_API_KEY` | Required for web-search retrieval tests |
| `REDIS_PASSWORD` | Redis auth for cache tests |
| `CORPUS_PATH`, `INDEX_PATH` | Override corpus/FAISS paths for WEMG corpus retrieval tests |
| `LANGGRAPH_CORPUS_INDEX_PATH` | Overrides `langgraph_coe` corpus FAISS file path (`.faiss`) for `corpus_search` tests |

## Tips

- Keep a local `.env` in the repository root for test credentials and paths.
- Run focused marker/file subsets while iterating, then run `pytest tests/` before merging.
- Prefer updating or adding tests in matching folders for changed modules.
