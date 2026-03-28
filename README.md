# WEMG — When Embedding Models Meet Graph RAG

WEMG is a question-answering stack that combines **graph-style retrieval** from Wikidata with **dense retrieval** (corpus embeddings or web search). Reasoning is driven by **Monte Carlo Tree Search (MCTS)** or **chain-of-thought (CoT)** search, with LLM agents in specialized roles (generation, evaluation, extraction, etc.).

---

## Main features

- **Dual retrieval**: Wikidata triples and entities plus text passages (FAISS corpus or web crawl).
- **Reasoning strategies**: MCTS with exploration, early stopping, and optional graph consensus rewards; CoT with configurable depth.
- **Reranking**: Optional cross-encoder / reranker pass over retrieved candidates.
- **Memory**: Working memory (text + knowledge graph) and optional **interaction memory** (embedding-backed retrieval across questions).
- **Evaluation**: Dataset runs with JSONL logs, per-question artifacts (search tree, textual memory, graph), Sub-EM, LLM **Acc**, Pass@k, resume, and rescoring.
- **Caching**: Redis-backed LLM response cache (optional).
- **Async-friendly internals**: Parallel LLM and batch answering APIs.

---

## Installation

### Requirements

- **Python ≥ 3.10** (see `pyproject.toml`).
- **LLM API**: OpenAI-compatible HTTP API (`llm.url`, `llm.api_key`).
- **Redis** (optional): recommended if `cache.enabled: true` in config.
- **Corpus mode** (optional): HuggingFace-format document shard dir + FAISS index paths in config.
- **Web search mode** (optional): Serper API key when `retriever.type: web_search`.

### From source

```bash
git clone https://github.com/hieum98/wemg.git
cd wemg

# Editable install (recommended)
pip install -e .

# With dev dependencies (pytest, python-dotenv, …)
pip install -e ".[dev]"
```

Or install pinned deps only:

```bash
pip install -r requirements.txt
```

This project is often used with a **conda** env named `wemg`; then prefix commands with `conda run -n wemg` if you use that layout.

---

## How to run

### One-off question (CLI)

```bash
# Uses default config: wemg/config.yaml (override with --config)
export API_KEY=...   # required unless set in YAML

python -m wemg "What is the capital of France?"

# Dotted overrides (same syntax as evaluation)
python -m wemg "Who directed Inception?" search.strategy=mcts llm.model_name=Qwen3-8B
```

### Python API

```python
from wemg import WEMGSystem

system = WEMGSystem()  # or WEMGSystem(config_path="path/to/config.yaml")
try:
    result = system.answer("What is the capital of France?")
    print(result.answer)
    print(result.concise_answer)
finally:
    system.close()
```

Convenience helpers (build a system, answer, then close):

```python
from wemg import answer_question, answer_questions_batch

text = answer_question("What is the capital of France?")
results = answer_questions_batch(["Q1?", "Q2?"], max_workers=4)
```

`WEMGSystem` also accepts `config_overrides=["llm.concurrency=16", ...]`, a `config_dict`, or a ready-made `WEMGConfig` instance.

---

## Configuration

Configuration is **YAML + Pydantic** (`wemg/config.py` validates `wemg/config.yaml`). Overrides use **dotted keys** and `key=value` tokens (booleans and numbers are parsed; `null`/`none` allowed).

- **Default file**: `wemg/config.yaml` (see `wemg.config.get_default_config_path()`).
- **Load in code**: `WEMGConfig.from_yaml(path, overrides=["search.strategy=cot"])`.
- **Environment**: `API_KEY`, `SERPER_API_KEY`, and `REDIS_PASSWORD` are merged into the tree when set (see `config.py`).

### Environment variables

| Variable | Used for |
|----------|-----------|
| `API_KEY` | `llm.api_key`, corpus embedder key, interaction-memory embedding key (if not set in YAML) |
| `SERPER_API_KEY` | `retriever.web_search.api_key` when using web search |
| `REDIS_PASSWORD` | `cache.password` |

### Config sections (fields)

**`llm`**

| Field | Role |
|-------|------|
| `model_name`, `url`, `api_key` | OpenAI-compatible client target |
| `client_type` | `openai` \| `azure` \| `anthropic` |
| `concurrency`, `max_retries` | Parallelism and retries |
| `generation.*` | `timeout`, `temperature`, `n`, `top_p`, `max_tokens`, `max_input_tokens`, `top_k`, `enable_thinking`, `random_seed` |

**`cache`** — Redis: `enabled`, `host`, `port`, `db`, `password`, `prefix`, `ttl`.

**`retriever`**

| Field | Role |
|-------|------|
| `type` | `corpus` \| `web_search` |
| `corpus.embedder.*` | Embedding model for dense retrieval (`model_name`, `url`, `api_key`, `embedder_type`: `openai` \| `huggingface`) |
| `corpus.corpus_path`, `corpus.index_path` | HF dataset dir + FAISS index |
| `web_search.*` | `api_key`, `top_k`, `crawl_full_text`, crawl rate limits, query/URL cache TTLs |

**`reranker`** — `enabled`, `model_name`, `url`, `api_key`, `top_k`, `concurrency`, `instruction`.

**`search`**

| Field | Role |
|-------|------|
| `strategy` | `mcts` \| `cot` |
| `mcts.num_iterations`, `max_tree_depth`, `exploration_weight` | MCTS loop |
| `mcts.use_golden_answer_for_reward` | Training/debug signal when gold answer is supplied |
| `mcts.min_graph_nodes_for_consensus`, `mcts.consensus_weight` | Blend graph consensus with answer evaluation reward |
| `mcts.early_termination.*` | `enabled`, `min_iterations`, `high_confidence_threshold`, `convergence_patience`, `semantic_sufficiency_count` |
| `cot.max_depth` | CoT expansion depth |

**`node_generation`** — Per-step generation: `n`, `n_subquestions`, `top_k_websearch`, `top_k_entities`, `n_hops`, `entity_linking_method` (`llm` \| `azure` + Azure fields), `rerank_kb_documents`, `triple_pruning_delta`, `triple_pruning_top_k`.

**`memory`** — See [Memory](#memory) below.

**`logging`** — `level`, `format`.

**`output`** — `include_reasoning`, `include_concise_answer`, `show_search_tree`, `verbose`.

---

## Memory

WEMG uses two complementary memory layers (see `wemg/reasoning/memory.py`, `working_memory.py`, `interaction_memory.py`).

### Working memory

- Holds **textual** evidence snippets and a **directed graph** of entities and relations (aligned with Wikidata-style triples).
- **`GlobalKnowledge`**: reward-gated shared store in MCTS; high-scoring branch **deltas** can be **absorbed** into global knowledge so other branches reuse facts.
- **Config** (`memory.working_memory`):
  - `max_textual_memory_tokens` — cap on text injected into prompts.
  - `absorption_min_reward` — minimum branch reward to promote a delta into global knowledge.
  - `absorption_top_k` — if nothing meets the threshold, still absorb the top-k branches by reward.

`AnswerResult` can expose `working_memory` and `global_knowledge` for inspection or logging.

### Interaction memory

- Optional **vector store** (Chroma) of past question–answer or interaction snippets, retrieved by embedding similarity to augment new questions.
- **Config** (`memory.interaction_memory`): `enabled`, `scope` (`question` = per-question lifecycle vs `dataset` = shared across a batch), `log_dir`, `save_to_file`, `db_path`, `collection_name`, `token_budget`, batch sizes, embedding model URL/name, embedding cache settings.

Enable and tune when you want **cross-turn** or **cross-example** context (e.g. `scope: dataset` for evaluation batches).

---

## Evaluation

Entry point:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate [OPTIONS] [OVERRIDES...]
```

- **`--config`**: YAML path; default is the package `config.yaml`.
- **Overrides**: space-separated `key=value`. Keys below are **evaluation** parameters; anything else is applied as **WEMG config** (dotted keys). A leading `+` on keys is ignored (Hydra-style compatibility).

Minimal example:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/bamboogle
```

With system overrides:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  dataset_name_or_path=cwq \
  output_path=results/cwq \
  max_examples=100 \
  resume=true \
  search.strategy=mcts \
  llm.model_name=Qwen3-8B
```

### Evaluation-only override keys

| Key | Default | Meaning |
|-----|---------|--------|
| `dataset_name_or_path` | *(required)* | Short name, HuggingFace id, local HF dataset dir, or `.json` / `.jsonl` |
| `output_path` | `./results` | Log, metrics, resolved `config.yaml`, `artifacts/` |
| `resume` | `true` | Skip questions whose `question` string already appears in `evaluation_log.jsonl` |
| `max_examples` | — | Truncate after load |
| `shuffle` | `false` | Shuffle (seed 42) before truncation |
| `question_column` / `answer_column` | `question` / `answer` | Column names |
| `max_concurrent` | auto | Parallelism for batch answering |
| `log_batch_size` | *(runner default)* | Checkpoint frequency to JSONL |
| `clear_kb_cache_every_n_batches` | `1` | Wikidata cache clear interval; `0`/`null` disables |
| `score_only` | `false` | Only score existing predictions (no generation) |
| `prediction_column` | `predicted_answer` | Used when `score_only=true` |

### Outputs under `output_path`

| Artifact | Purpose |
|----------|--------|
| `config.yaml` | Resolved config for reproducibility |
| `evaluation_log.jsonl` | One JSON object per question |
| `metrics.json` / `summary.txt` | Aggregated Sub-EM, Acc, counts, optional Pass@k |
| `artifacts/q_<idx>_<hash>_<slug>/` | `search_tree.json`, `working_memory_textual.json`, `working_memory_graph.pkl` |

**Metrics**: substring exact match (**Sub-EM**), LLM judge **Acc**, **Pass@k** when the run produces multiple samples.

More detail (rescoring from logs, profiling, artifact layout) lives in [`wemg/evaluation/README.md`](wemg/evaluation/README.md).

### Supported dataset short names

Graph-oriented: `cwq`, `webqsp`, `qald_10`, `hotpotqa_adv`, `grail_qa`.  
Text-oriented: `2wiki`, `hotpotqa`, `musique`, `bamboogle`, `frames`.  
Other HuggingFace ids or local paths work via `load_dataset_any` (`wemg/evaluation/datasets.py`).

---

## Helpers

### Evaluation artifacts and visualization

`wemg.evaluation.artifacts` provides loading and plotting helpers, for example:

- `load_question_artifacts`, `find_artifacts_entry` — locate rows in `evaluation_log.jsonl` and load saved trees/graphs.
- `load_graph_memory`, `load_search_tree_json`
- `print_saved_search_tree`
- `visualize_graph_memory` (static) / `visualize_graph_memory_interactive` (PyVis HTML; Jupyter or browser)

Example (interactive graph):

```python
from wemg.evaluation.artifacts import load_graph_memory, visualize_graph_memory_interactive

graph = load_graph_memory("results/.../working_memory_graph.pkl")
visualize_graph_memory_interactive(
    graph,
    title="Working memory",
    save_path="results/.../working_memory_graph.html",
    notebook_mode=True,
    open_in_browser=False,
)
```

### Test utilities (`tests/helpers`)

For developers running pytest:

- `tests.helpers.bootstrap`: `repo_root()`, `default_config_yaml()`, `load_test_env()` (loads repo `.env` without clobbering env), `corpus_paths_from_config_or_env` (uses `CORPUS_PATH` / `INDEX_PATH` overrides in tests).
- `tests.helpers.slow_integration_debug`: pretty-print helpers for slow integration tests (`to_debug_dict`, `working_memory_snapshot`, etc.).

---

## Development

```bash
# All tests (from repo root)
pytest tests/

# Skip slow tests
pytest tests/ -m "not slow"

# By kind
pytest tests/ -m "unit"
pytest tests/ -m "integration"
```

Use `pytest -s` with slow integration tests if you rely on debug printers in `tests.helpers.slow_integration_debug`.

---

## Performance and profiling

```bash
set -a && source .env && set +a   # if you keep secrets in .env

conda run -n wemg python -m cProfile -o results/profiling/eval.prof \
  -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/profiling/bamboogle
```

Inspect with `python -m pstats` or [snakeviz](https://jiffyclub.github.io/snakeviz/). The evaluator also emits timing logs around chunks when logging is set to INFO.

Practical knobs: raise `llm.concurrency`, keep Redis cache on, tune MCTS `early_termination` and `min_graph_nodes_for_consensus`, use `memory.interaction_memory.scope=dataset` when appropriate.

---

## Project structure

```
wemg/
├── config.yaml / config.py     # Defaults and Pydantic schema
├── system.py                   # WEMGSystem orchestration
├── llm/                        # Client, cache, roles, parsing
├── retrieval/                  # Corpus, web search, Wikidata, reranker, entity linking
├── reasoning/                  # MCTS, CoT, nodes, generator, working + interaction memory
├── evaluation/                 # CLI, runner, metrics, datasets, artifacts
└── utils/                      # Text, graph visualization helpers
```

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Add tests for new behavior  
4. Run `pytest tests/`  
5. Open a pull request  

---

## License

MIT — see the LICENSE file.

---

## Authors

- Hieu Man ([hieum@uoregon.edu](mailto:hieum@uoregon.edu))

---

## Citation

```bibtex
@software{wemg2026,
  title={WEMG: When Embedding Model Meet Graph RAG},
  author={Man, Hieu},
  year={2026}
}
```

---

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for LLM client patterns  
- [Wikidata](https://www.wikidata.org/) and SPARQL tooling for graph retrieval  

For issues or questions, use the GitHub issue tracker.
