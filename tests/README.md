# WEMG Test Suite

Real-system tests (no mocks). Each component is tested against live services where applicable.

## Running tests

```bash
# All tests (skip LLM/Redis tests if env not set)
pytest tests/ -v

# Only tests that do not require LLM or Redis
pytest tests/ -v -m "not requires_llm and not requires_redis"

# Only integration tests that require LLM
pytest tests/ -v -m "requires_llm"
```

## Environment variables

Set these to run the full suite:

| Variable | Purpose |
|----------|---------|
| `API_KEY` or `OPENAI_API_KEY` | LLM API key (required for `requires_llm` tests) |
| `LLM_URL` or `OPENAI_BASE_URL` | LLM base URL (e.g. `http://localhost:4000/v1`) |
| `LLM_MODEL` | Model name for completion (default: `gpt-3.5-turbo`) |
| `EMBEDDING_MODEL` | Model name for embeddings (default: `text-embedding-3-small`) |
| `SERPER_API_KEY` | Optional; web search uses DuckDuckGo fallback if unset |
| `REDIS_PASSWORD` | Optional; cache tests skip if Redis not used |
| `CORPUS_PATH`, `INDEX_PATH` | Optional; corpus retrieval tests skip if unset |

