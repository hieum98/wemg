# WEMG test suite

Real-system tests (no mocks). Configuration and secrets come from [`wemg/config.yaml`](../wemg/config.yaml) plus environment variables. On import, [`tests/conftest.py`](conftest.py) loads a repo-root `.env` if present (via `python-dotenv`) without overriding variables already set in the shell.

## Layout

| Directory | Scope |
|-----------|--------|
| `helpers/` | `.env` bootstrap, path helpers |
| `config/` | `WEMGConfig` loading, overrides, validation, `WEMGSystem` init errors |
| `llm/` | `LLMClient`, parsing, `execute_role`, optional Redis cache |
| `retrieval/` | corpus, web search, Wikidata, entity linking, reranker |
| `reasoning/` | `NodeGenerator`, `WorkingMemory`, `merge_logs` |
| `system/` | `WEMGSystem.answer`, CoT/MCTS smoke, batch API |

## Running tests

From the repository root:

```bash
pip install -e ".[dev]"

# Fast, mostly local (still needs config YAML present)
pytest tests/config tests/llm/test_parsing.py -v

# Everything except slow end-to-end flows
pytest tests/ -v -m "not slow_integration"

# LLM-backed tests only (needs `llm.api_key` + URL in YAML or `API_KEY` / `LLM_URL`)
pytest tests/ -v -m "requires_llm"

# Corpus retrieval (on-disk FAISS + embedder)
pytest tests/retrieval/test_corpus.py -v -m "requires_corpus"

# Redis cache round-trip
pytest tests/llm/test_cache_redis.py -v -m "requires_redis"

# Full integration including MCTS/CoT smoke
pytest tests/ -v -m "slow_integration"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `API_KEY` or `OPENAI_API_KEY` | LLM and embedder keys merged into config |
| `LLM_URL` or `OPENAI_BASE_URL` | Optional alias for base URL (tests also use `config.yaml`) |
| `SERPER_API_KEY` | Web search when `retriever.type` is `web_search` |
| `REDIS_PASSWORD` | Cache password when Redis is enabled |
| `CORPUS_PATH`, `INDEX_PATH` | Override corpus / FAISS paths for retrieval tests |

Place a `.env` file at the repository root (same level as `wemg/`) to set these once; the suite loads it automatically before reading YAML.
