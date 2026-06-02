# Codebase Index

Generated map of the `wemg` repository. Two packages coexist:

| Package | Status | Purpose |
|---|---|---|
| `langgraph_coe/` | **Active**, on `langchain` branch | Naive port of WEMG onto LangChain / LangGraph primitives. New work lives here. |
| `wemg/` | **Legacy reference** | Original pure-Python MCTS / CoT implementation. Do not modify unless explicitly asked. |

Orchestration in `langgraph_coe/` is **not yet wired end-to-end** — `system.py` only loads `.env`. The package currently ships building blocks: config, roles, LLM registry, tools, one compiled subgraph (`kg_search`).

---

## 1. `langgraph_coe/` — active package

### Configuration & entry

| File | Lines | What lives here |
|---|---:|---|
| `__init__.py` | 3 | Package marker. |
| `system.py` | 3 | `load_dotenv()` only. Future entry point. |
| `config.py` | 133 | `LangGraphCoeConfig` (Pydantic). Sub-configs: `TierConfig`, `LLMConfig` (with `role_tiers` dict), `WebSearchConfig`, `EmbedderConfig`, `CorpusConfig`, `RetrieverConfig`, `RerankerConfig`, `WikidataConfig`. Loader: `LangGraphCoeConfig.from_yaml()`. |
| `config.yaml` | — | Default settings. Cluster hostnames (`n0142`, `n0385`, `n0999`) target an internal LiteLLM gateway; override for non-cluster runs. |
| `implementation_plan.md` | — | Design notes for the port. |
| `documents/WIKIDATA.md` | — | User + maintainer guide for the Wikidata tool. |

### Roles & LLM execution

| File | Lines | Key symbols |
|---|---:|---|
| `roles.py` | 896 | 17 `Role` objects (system prompt + Pydantic input/output models). See **§3 Role registry** below. |
| `llm.py` | 165 | `RoleModelRegistry` (role → tier → `ChatLiteLLM`, lazy). `execute_role_lc(registry, role, input, n, tier_override)` — n parallel `with_structured_output` calls + regex fallback. Helpers: `format_messages`, `parse_fallback`. |
| `parsing.py` | 284 | `extract_info_from_text(role, text)` — regex-driven fallback parser. Helpers: `extraction_type_from_annotation`, `_convert_value`, `_extract_field_with_regex`. |

### Tools (`langgraph_coe/tools/`)

All are LangChain `@tool` async functions. Each module exposes a one-shot `init_*` that must run before the tool is invoked.

| File | Lines | Tools / functions | Notes |
|---|---:|---|---|
| `__init__.py` | 28 | Re-exports. | |
| `retrieval.py` | 90 | `@tool corpus_search(query)`. `init_retrieval_pipeline(retriever_cfg, reranker_cfg)`, `get_corpus_retriever`, `_rerank_documents`. | FAISS index path override: `LANGGRAPH_CORPUS_INDEX_PATH`. |
| `web.py` | 99 | `@tool web_search(query)`. `init_web_search(cfg)`, `_crawl_page`. | Serper if `SERPER_API_KEY` set, else DuckDuckGo. |
| `wikidata.py` | 379 | `@tool link_entities`, `@tool fetch_and_prune_subgraph` (Stage A only), `@tool enrich_entities`. Factory `create_fetch_and_prune_tool(registry)` returns the Stage A+B variant for agents. Lifecycle: `init_wikidata`, `reset_wikidata_session`. | Per-question state in `ContextVar` (`_cv_visited_qids`, `_cv_hop_count`). **Reset between questions.** Three-layer loop prevention: visited set → hop counter → LangGraph `recursion_limit`. |
| `wikidata_backend.py` | 335 | HTTP/transport backend for the Wikidata client. | |
| `wikidata_client.py` | 736 | High-level Wikidata client (entity search, subgraph fetch, property/label lookups). | |
| `wikidata_properties.py` | 117 | Curated property allow-/deny-lists. | |

### Subgraphs (`langgraph_coe/graphs/`)

| File | Lines | Symbols |
|---|---:|---|
| `__init__.py` | 15 | Re-exports `build_kg_search_graph`, `run_kg_search_sync`, `run_kg_search_async`. |
| `kg_search.py` | 289 | `KGSearchState` (TypedDict). Builder: `build_kg_search_graph(registry)` → 3 nodes: `ner_agent` (uses `link_entities`), `triple_search_agent` (uses Stage A+B fetch tool), `enrich` (plain async, calls `enrich_entities`). Runners: `run_kg_search_sync`, `run_kg_search_async`. Helpers: `_parse_link_entities_tool_payloads`, `_parse_triple_tool_payloads` (parse `ToolMessage` trace for observability). |

### Tests (`langgraph_coe/tests/`)

Empty on disk — new tests should land here. Until `pyproject.toml`'s `testpaths` is updated from `["tests"]`, run with `pytest langgraph_coe/tests` explicitly. `asyncio_mode = "auto"`.

---

## 2. `wemg/` — legacy reference

Original pure-Python implementation. ~12.8k LOC. Kept for reference and for the `python -m wemg` CLI.

### Top-level

| File | Lines | Symbols |
|---|---:|---|
| `__main__.py` | 45 | CLI: `python -m wemg "question"`. |
| `config.py` | 318 | `WEMGConfig` and sub-configs. |
| `system.py` | 472 | `WEMGSystem`, `AnswerResult`, `answer_question`, `answer_questions_batch`. End-to-end orchestrator. |

### Reasoning (`wemg/reasoning/`)

| File | Lines | Symbols |
|---|---:|---|
| `nodes.py` | 339 | `NodeType`, `NodeState`, `ReasoningNode`, `MCTSNode`, `CoTNode`. Helpers: `check_correctness`, `make_*_state`, `add_node_content_to_memory`. |
| `mcts.py` | 412 | `mcts_search`, `select`, `expand`, `simulate`, `evaluate`, `get_answer`. |
| `cot.py` | 179 | `cot_search`, `cot_get_answer`, `generate_next_step`. |
| `generator.py` | 495 | `NodeGenerator` — produces subqa / final-answer / self-correct children. |
| `working_memory.py` | 680 | `WorkingMemory`, `_GraphStore`, `parse_graph_from_text`, LLM-driven pruning. |
| `interaction_memory.py` | 590 | Interaction-level memory store. |
| `memory.py` | 23 | Shared memory base. |

### Retrieval (`wemg/retrieval/`)

| File | Lines | Symbols |
|---|---:|---|
| `wikidata.py` | 1775 | Full Wikidata client (legacy, monolithic). |
| `freebase_client.py` | 704 | Freebase SPARQL client. |
| `freebase_preprocess.py` | 552 | Freebase data prep. |
| `graph_retriever_client.py` | 344 | Graph retriever HTTP client. |
| `web_search.py` | 286 | `WebSearchTool`, `WebSearchResult`, `KGEntity`, `crawl_page(s)`. Serper + DuckDuckGo backends. |
| `entity_linking.py` | 187 | `link_entities_llm`, `link_entities_azure`, `_wikipedia_url_to_qid`. |
| `virtuoso.py` | 190 | Virtuoso SPARQL adapter. |
| `corpus.py` | 134 | `CorpusRetriever` (FAISS). |
| `reranker.py` | 65 | Reranker HTTP client. |

### LLM layer (`wemg/llm/`)

| File | Lines | Symbols |
|---|---:|---|
| `roles.py` | 1019 | Legacy role registry — superset of `langgraph_coe/roles.py`. `execute_role`, `format_messages`, `parse_response`. |
| `client.py` | 528 | LiteLLM client wrapper. |
| `parsing.py` | 276 | Legacy regex fallback parser. |
| `cache.py` | 172 | Disk + in-memory LLM cache. |

### Evaluation (`wemg/evaluation/`)

| File | Lines | Symbols |
|---|---:|---|
| `runner.py` | 726 | `DatasetEvaluator`, artifact serialization (`_serialize_search_tree`, `_save_question_artifacts`), Freebase subgraph cache loaders. |
| `evaluate.py` | 173 | CLI entry: `python -m wemg.evaluation.evaluate ...`. `split_eval_overrides`. |
| `metrics.py` | 244 | EM / F1 / hop-level metrics. |
| `datasets.py` | 194 | Dataset loaders. |
| `artifacts.py` | 223 | Per-question artifact I/O. |
| `scripts/` | — | Dataset prep scripts. |

### Utils (`wemg/utils/`)

| File | Lines | Symbols |
|---|---:|---|
| `graph.py` | 511 | Graph manipulation utilities. |
| `text.py` | 58 | Text normalization helpers. |

---

## 3. Role registry — `langgraph_coe/roles.py`

17 roles. Tier assignment lives in `LLMConfig.role_tiers` (`config.py`). Each role: `Role(name, system_prompt, InputModel, OutputModel)`.

| Constant | Name (key in `role_tiers`) |
|---|---|
| `SUBQUESTION_GENERATOR` | `subquestion_generator` |
| `ANSWER_GENERATOR` | `answer_generator` |
| `QUERY_GENERATOR` | `query_generator` |
| `SELF_CORRECTOR` | `self_corrector` |
| `QUESTION_REPHRASER` | `question_rephraser` |
| `REASONING_SYNTHESIZER` | `reasoning_synthesizer` |
| `EVALUATOR` | `evaluator` |
| `MAJORITY_VOTER` | `majority_voter` |
| `FINAL_ANSWER_SYNTHESIZER` | `final_answer_synthesizer` |
| `CONSENSUS_EVALUATOR` | `consensus_evaluator` |
| `VERIFIER` | `verifier` |
| `EXTRACTOR` | `extractor` |
| `MEMORY_CONSOLIDATOR` | `memory_consolidation` |
| `NER` | `named_entity_recognition` |
| `RELATION_EXTRACTOR` | `relation_extraction` |
| `TRIPLE_PRUNER` | `triple_pruner` |

(`QueryGraphGenerator` input/output models exist but no `Role` constant is exported.)

---

## 4. Tests — `tests/`

Top-level `tests/` covers the legacy Wikidata client. Per `CLAUDE.md`, this tree was deleted on the `langchain` branch — present in working copy but not the merge target.

| File | Purpose |
|---|---|
| `conftest.py`, `wikidata/conftest.py` | Pytest fixtures (incl. fake backend wiring). |
| `wikidata/_fixtures.py`, `fake_backend.py`, `contracts.py` | Shared test fixtures, fake HTTP backend, contract assertions. |
| `test_async_correctness.py` | Async invariants. |
| `test_caching.py` | Cache layer. |
| `test_concurrency.py` | Concurrent task isolation (relevant to the `ContextVar` design in `langgraph_coe/tools/wikidata.py`). |
| `test_enrichment.py` | `enrich_entities` behavior. |
| `test_entity_linking.py` | `link_entities` behavior. |
| `test_khop_triples.py` | k-hop subgraph traversal. |
| `test_live_smoke.py` | Live API smoke (network). |
| `test_property_search.py` | Property allow/deny logic. |
| `test_rate_limit.py` | Rate limiter. |
| `test_tools_integration.py` | End-to-end tool wiring. |

---

## 5. Repo-level files

| Path | Purpose |
|---|---|
| `README.md` | Research framing (still describes legacy `wemg/`). Use for vocabulary, not implementation details. |
| `CLAUDE.md` → `.agents/AGENTS.md` | Project instructions for all coding agents (Claude/Codex/Gemini/Cursor/OpenCode). |
| `pyproject.toml` | Build config. Python `>=3.12`. `asyncio_mode = "auto"`, `log_cli=INFO`, `testpaths = ["tests"]`. |
| `agentsync.toml` | Regenerates symlink fan-out + MCP configs across agents. Do not hand-edit symlinked files. |
| `.mcp.json` / `.cursor/mcp.json` / `opencode.json` | Register the `docs-langchain` MCP server (`https://docs.langchain.com/mcp`) — use it for first-party LangChain/LangGraph API questions. |
| `.agents/skills/`, `.agents/commands/` | Shared agent skills/commands (currently empty). |
| `Framework-overview.pdf` | Untracked design doc. |
| `examples/` | Example scripts. |

---

## 6. Common entry points

```bash
pip install -e ".[dev]"                        # install with dev extras

python -m wemg "What is the capital of France?"   # legacy CLI (only path with one)

pytest                                          # legacy tests (testpaths=["tests"])
pytest langgraph_coe/tests                      # new tests once they exist
```

`langgraph_coe` has **no CLI yet** — drive it from a script that builds a `RoleModelRegistry`, initializes tools (`init_wikidata`, `init_retrieval_pipeline`, `init_web_search`), and invokes `build_kg_search_graph(registry).ainvoke(...)`.
