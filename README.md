# WEMG: When Embedding Models Meet Graph RAG

WEMG is a research-oriented question-answering system that combines graph retrieval, dense retrieval, and LLM reasoning.

The central novelty is a Zettelkasten-inspired working memory: atomic notes with stable IDs, typed inter-note links, a lifecycle progression (fleeting → literature → permanent → structure), and emergent structure notes that emit gap questions to actively steer reasoning.

## Highlights

- **Zettelkasten working memory**: notes progress through a four-stage lifecycle; permanent notes are synthesized into structure notes that emit targeted gap questions fed back into the reasoning loop.
- **Typed note links**: ten link types (SUPPORTS, CONTRADICTS, DERIVED_FROM, GAP_FOR, ANSWERS, ELABORATES, PRECEDES, REFINES, ANALOGOUS_TO, CO_OCCURS_WITH) maintain rich inter-note relationships.
- **Graph proximity retrieval**: relevant notes are retrieved by walking entity-to-note mappings and note-graph edges; a ChromaDB semantic fallback kicks in when fewer than `k_min` notes are found.
- **Tiered LLM routing**: a cheap 4–8B model handles mechanical tasks (consolidation, entity linking, relation extraction); the primary model handles promotion decisions and structure generation.
- **Bidirectional memory**: evidence flows from text into the entity graph and from graph structure back into textual context.
- **Dual retrieval**: Wikidata/SPARQL graph evidence plus corpus (FAISS) or web search evidence.
- **Two reasoning strategies**: Monte Carlo Tree Search (MCTS) and chain-of-thought (CoT).
- **Evaluation pipeline**: dataset runs, metrics, and per-question artifacts.

## Architecture at a glance

Main request flow:

1. `WEMGSystem.answer(question)` in `wemg/system.py` selects `mcts` or `cot`.
2. `NodeGenerator` in `wemg/reasoning/generator.py` performs retrieval and node expansion.
3. Knowledge-base retrieval uses iterative 1-hop expansion (`k=1` per hop) with frontier reranking between hops.
4. `WorkingMemory.synchronize_memory()` in `wemg/reasoning/working_memory.py` runs the 6-step note lifecycle pipeline and returns gap questions.
5. Gap questions are injected back as `[Knowledge Gap]: …` fleeting notes, steering the next iteration.
6. The system returns an `AnswerResult`.

Core modules:

- `wemg/system.py`: orchestration, lifecycle, cheap-client instantiation.
- `wemg/reasoning/mcts.py`: MCTS search with gap-question injection.
- `wemg/reasoning/cot.py`: CoT search with gap-question injection.
- `wemg/reasoning/generator.py`: retrieval and generation pipeline.
- `wemg/reasoning/working_memory.py`: Zettelkasten note store, note graph, 6-step sync.
- `wemg/reasoning/note_store.py`: ChromaDB-backed `NoteVectorStore` for semantic note retrieval.
- `wemg/retrieval/wikidata.py`: Wikidata/SPARQL access with batching and cache.
- `wemg/llm/roles.py`: role definitions and structured I/O (including `PROMOTION_EVALUATOR`, `STRUCTURE_NOTE_GENERATOR`).

### Working memory note lifecycle

```text
add_textual_memory()
      │
      ▼  NoteType.FLEETING
      │
  synchronize_memory() — 6-step pipeline:
  Step 1  Ingest & tag        [cheap LLM]  assign UUIDs; deduplicate
  Step 2  Consolidate fleeting [cheap LLM]  surviving notes → LITERATURE
  Step 3  Entity linking       [cheap LLM]  NER + relation extraction + graph update
                                            → CO_OCCURS_WITH / SUPPORTS links
  Step 4  Promote lit→perm    [expensive]  PROMOTION_EVALUATOR checks corroboration;
                                            demotes on contradiction (CONTRADICTS link)
  Step 5  Structure notes     [expensive]  STRUCTURE_NOTE_GENERATOR when ≥ M new permanents;
                                            emits gap_questions[], creates DERIVED_FROM links
  Step 6  Selective retrieval  [no LLM]    graph proximity + semantic fallback
      │
      └── returns gap_questions: List[str]
```

## Installation

Requirements:

- Python 3.10+
- OpenAI-compatible LLM endpoint
- Optional Redis for LLM response caching
- Optional ChromaDB for note vector store (installed via `pip install -e ".[dev]"`)

Recommended setup:

```bash
conda create -n wemg python=3.10 -y
conda activate wemg
pip install -e ".[dev]"
```

Minimal setup:

```bash
pip install -e .
```

## Quick start

CLI:

```bash
conda run -n wemg python -m wemg "What is the capital of France?"
```

With runtime overrides:

```bash
conda run -n wemg python -m wemg "Who directed Inception?" search.strategy=mcts llm.model_name=Qwen3-8B
```

With a cheap LLM tier for consolidation tasks:

```bash
conda run -n wemg python -m wemg "question" \
  llm.cheap_model_name=Qwen3-8B \
  llm.cheap_url=http://localhost:4001/v1
```

Python API:

```python
from wemg import WEMGSystem

system = WEMGSystem()
try:
    result = system.answer("What is the capital of France?")
    print(result.answer)
    print(result.concise_answer)
finally:
    system.close()
```

## Configuration

- Default config file: `wemg/config.yaml`
- Schema and validation: `wemg/config.py` (`WEMGConfig`)
- Override format: dotted `key=value` arguments

Key working memory parameters:

| Key | Default | Description |
| --- | --- | --- |
| `memory.working_memory.promotion_corroboration_count` | `2` | Distinct-source confirmations required to promote literature → permanent |
| `memory.working_memory.structure_note_trigger_m` | `5` | New permanents added before a structure note is generated |
| `memory.working_memory.retrieval_k_min` | `3` | Min notes from graph proximity before semantic fallback |
| `memory.working_memory.note_store.enabled` | `true` | Enable ChromaDB-backed note vector store |
| `memory.working_memory.note_store.persist_dir` | `.note_store` | Persistence directory for ChromaDB |
| `llm.cheap_model_name` | `null` | Model for cheap-tier tasks; falls back to main model if unset |

Environment variables:

- `API_KEY`: LLM key and related embedding keys if unset in config.
- `SERPER_API_KEY`: web search key.
- `REDIS_PASSWORD`: Redis password.

## Evaluation

Run evaluation:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/bamboogle
```

See `wemg/evaluation/README.md` for CLI options, artifact layout, rescoring, and profiling.

## Testing

Run the full suite:

```bash
conda run -n wemg pytest tests/
```

Fast iteration run:

```bash
conda run -n wemg pytest tests/ -m "not slow_integration and not integration"
```

See `tests/README.md` for markers and environment setup.

## Project layout

```text
wemg/
  config.py           # WEMGConfig, WorkingMemoryConfig, NoteStoreConfig, LLMConfig (cheap tier)
  config.yaml
  system.py           # WEMGSystem orchestrator; instantiates cheap_client
  llm/
    roles.py          # Role definitions incl. PROMOTION_EVALUATOR, STRUCTURE_NOTE_GENERATOR
    client.py         # LLMClient with Redis cache and batch generation
  retrieval/
    wikidata.py       # Batched SPARQL with retries and disk cache
  reasoning/
    working_memory.py # Zettelkasten note store, 6-step sync, lifecycle, graph proximity retrieval
    note_store.py     # ChromaDB-backed NoteVectorStore
    generator.py      # NodeGenerator: iterative KB retrieval + generation
    mcts.py           # MCTS search with gap-question injection
    cot.py            # CoT search with gap-question injection
    nodes.py          # CoTNode, MCTSNode, NodeState
  evaluation/
  utils/
tests/
examples/
retriever_corpora/
```

## Troubleshooting

- If retrieval fails in corpus mode, verify `retriever.corpus.corpus_path` and `retriever.corpus.index_path`.
- If web search mode fails, verify `SERPER_API_KEY` and `retriever.type=web_search`.
- If cache is enabled but Redis is unavailable, check `memory.cache` connection settings.
- If the note store fails to initialize, verify ChromaDB is installed (`pip install chromadb`) and `memory.working_memory.note_store.persist_dir` is writable.
- Gap questions appear in debug logs (`[Knowledge Gap]: …`) after the first structure note is generated (requires ≥ `structure_note_trigger_m` permanent notes).

## Contributing

1. Create a feature branch.
2. Add or update tests in `tests/`.
3. Run relevant tests and then `pytest tests/`.
4. Open a pull request with clear change notes.

## License

MIT
