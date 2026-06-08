# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project orientation

Two implementations of the same research idea live side by side:

- `wemg/` — original **pure-Python** implementation (MCTS/CoT search, working-memory coordination, custom retrieval). Treated as **legacy reference**.
- `langgraph_coe/` — **active** "naive" port onto the LangChain / LangGraph ecosystem. New work goes here.

When asked to add a feature or fix a bug, default to `langgraph_coe/` and use LangChain/LangGraph primitives (`create_agent`, `StateGraph`, `@tool`, `with_structured_output`). Only touch `wemg/` if the user explicitly says so or if a shared utility lives there.

The research framing (graph + textual memory coordination, dual retrieval, multi-hop reasoning) is described in `README.md`. Read it for vocabulary, not for implementation details — the README still describes the `wemg/` package.

## langgraph_coe architecture

End-to-end orchestration is **not yet wired up** — `langgraph_coe/system.py` only calls `load_dotenv()`. The package today provides the building blocks:

```
langgraph_coe/
  config.py     LangGraphCoeConfig (Pydantic), tier-based LLM config
  config.yaml   default settings (cluster hostnames, knobs)
  roles.py      17 Role objects = (name, system_prompt, input_model, output_model)
  llm.py        RoleModelRegistry + execute_role_lc (structured output + regex fallback)
  parsing.py    extract_info_from_text — fallback parser when with_structured_output returns no `parsed`
  tools/        @tool definitions: wikidata, retrieval (FAISS), web (Serper/DDG)
  graphs/       compiled LangGraph subgraphs (currently: kg_search)
```

### Tier-based LLM routing — `llm.py`

`RoleModelRegistry` maps **role name → tier (`heavy`/`medium`/`light`) → `ChatLiteLLM` instance**, lazily constructed. To remap a role to a different model, edit `LLMConfig.role_tiers` in `config.py`; do not change role definitions or instantiate clients ad-hoc.

`execute_role_lc(registry, role, input, n=1, tier_override=None)`:
- Uses `model.with_structured_output(role.output_model, include_raw=True)`.
- Gathers `n` parallel completions per input via `asyncio.gather`.
- If structured parsing fails, falls back to `parse_fallback(role, raw.content)` (regex over the raw text, driven by the output model's field annotations via `extraction_type_from_annotation`).
- Raises on all-N failure so a caller's `RetryPolicy` (e.g. LangGraph node retry config) can escalate.

### Roles — `roles.py`

Each `Role` is a dataclass: `(name, system_prompt, input_model, output_model)`. Pydantic `__str__` methods on input models produce the user-message text. When adding a role:
1. Define `*Input` / `*Output` Pydantic models.
2. Add the system prompt.
3. Append `MY_ROLE = Role("my_role", PROMPT, MyInput, MyOutput)`.
4. Add `"my_role": "<tier>"` to `LLMConfig.role_tiers`.

### Tools — `langgraph_coe/tools/`

All tools are LangChain `@tool` async functions. Each module exposes a module-level `init_*` function that must run **once at startup** before any tool invocation:

- `init_wikidata(config.wikidata)` — required for `link_entities`, `fetch_and_prune_subgraph`, `enrich_entities`.
- `init_retrieval_pipeline(config.retriever, config.reranker)` — required for `corpus_search`.
- `init_web_search(config.web_search)` — required for `web_search`; picks Serper if `SERPER_API_KEY` is set, falls back to DuckDuckGo.

Wikidata invariants worth knowing before changing `tools/wikidata.py`:

- **Per-question state uses `ContextVar`** (`_cv_visited_qids`, `_cv_hop_count`) so concurrent `asyncio` Tasks (e.g. `agent.batch(...)`) cannot contaminate each other. Call `reset_wikidata_session()` at the start of each new question. The KG subgraph already does this inside `ner_agent_node`.
- **Three-layer loop prevention**: ContextVar visited-QID set → ContextVar hop counter capped by `wikidata.max_hops` → LangGraph `recursion_limit` passed to `agent.ainvoke(config=...)`. Keep all three when adding new graph-traversal tools.
- **Two-stage pruning** in `fetch_and_prune_subgraph`: Stage A reranker scoring (`_stage_a_prune`, skipped when `reranker_url` is null) → optional Stage B LLM pruning via the `TRIPLE_PRUNER` role (`_stage_b_prune`). The plain `@tool fetch_and_prune_subgraph` runs Stage A only; use `create_fetch_and_prune_tool(registry)` inside agents to enable Stage B.

### Subgraphs — `langgraph_coe/graphs/`

`graphs/kg_search.py` is the current pattern to follow. `build_kg_search_graph(registry)` returns a compiled `StateGraph[KGSearchState]` with three nodes:

1. `ner_agent` — `create_agent(model, tools=[link_entities], ...)` resolves entity mentions to QIDs.
2. `triple_search_agent` — `create_agent(model, tools=[create_fetch_and_prune_tool(registry)], ...)` walks the subgraph.
3. `enrich` — plain async node, calls `enrich_entities` directly (no LLM).

Tool outputs are reconstructed from the agent message trace by filtering `ToolMessage` by `.name` (see `_parse_link_entities_tool_payloads`, `_parse_triple_tool_payloads`) — keep that pattern when adding agent nodes so the subgraph remains observable through messages.

## Configuration

- Schema: `LangGraphCoeConfig` in `langgraph_coe/config.py`.
- Defaults: `langgraph_coe/config.yaml`.
- Load: `LangGraphCoeConfig.from_yaml()` (no arg → default YAML; merges `API_KEY` / `OPENAI_API_KEY` env into `llm.api_key`, embedder, reranker).
- `.env` at repo root is auto-loaded via `system.py` / `python-dotenv`.

The default `api_base` (`http://n0142:4000/v1`) and embedder/reranker URLs (`n0385`, `n0999`) are **SLURM cluster hostnames** for an internal LiteLLM gateway. Override them in `config.yaml` or via environment for any non-cluster machine. `LANGGRAPH_CORPUS_INDEX_PATH` is mentioned in the YAML as an override hook for the FAISS index.

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Legacy CLI (wemg/ package; langgraph_coe has no CLI yet)
python -m wemg "What is the capital of France?"

# Tests — pytest config in pyproject.toml; asyncio_mode=auto, log_cli=INFO
pytest                             # testpaths = ["tests"]
pytest tests/path/to/test_x.py::test_name   # single test
```

Notes on tests: the legacy `tests/` tree was deleted on the current `langchain` branch (see `git status`); new tests should land in `langgraph_coe/tests/`. Until `pyproject.toml` is updated, you may need `pytest langgraph_coe/tests`. `pytest-asyncio` is in `[dev]` and `asyncio_mode = "auto"` is set, so async test functions don't need an explicit `@pytest.mark.asyncio`.

Python: pyproject pins `>=3.12`; the README's "3.10+" applies only to the legacy `wemg` path.

## Agent / tooling setup

- `agentsync.toml` regenerates the symlink fan-out and MCP configs for Claude, Copilot, Cursor, Codex, Gemini, OpenCode. Don't hand-edit the symlinked agent files.
- `.mcp.json` (and `.cursor/mcp.json`, `opencode.json`) register one MCP server: **`docs-langchain`** at `https://docs.langchain.com/mcp`. Use it for first-party LangChain / LangGraph API questions instead of guessing.
- Shared skill/command directories `.agents/skills/` and `.agents/commands/` are empty; populate them there (not under `.claude/` etc.) so all agents see them.
