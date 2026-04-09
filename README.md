# WEMG: When Embedding Models Meet Graph RAG

WEMG is a research-oriented question-answering system that combines graph retrieval, dense retrieval, and LLM reasoning.

The central novelty of this research is memory management: a coordinated textual memory and graph memory that are continuously synchronized during reasoning.

## Highlights

WEMG is designed around graph-text memory coordination as the primary research contribution, not retrieval alone.

- Memory-centric reasoning core: synchronized textual memory and graph memory.
- Bidirectional memory updates: evidence flows from text into graph structure and from graph structure back into textual context.
- Search-time synchronization: memory coordination is integrated into each reasoning step, not only post-processing.
- Persistent memory state across reasoning steps improves decomposition, verification, and final synthesis.
- Textual memory captures evidence snippets, intermediate reasoning traces, and model-generated hypotheses.
- Graph memory preserves entity-relation structure for relational consistency and multi-hop reasoning.
- Dual retrieval: Wikidata graph evidence plus corpus or web evidence.
- Two reasoning strategies: Monte Carlo Tree Search (MCTS) and chain-of-thought (CoT).
- Evaluation pipeline: dataset runs, metrics, and per-question artifacts.

## Architecture at a glance

Main request flow:

1. `WEMGSystem.answer(question)` in `wemg/system.py` selects `mcts` or `cot`.
2. `NodeGenerator` in `wemg/reasoning/generator.py` performs retrieval and node expansion.
3. Knowledge-base retrieval uses iterative 1-hop expansion (`k=1` per hop).
4. `WorkingMemory.synchronize_memory()` in `wemg/reasoning/working_memory.py` performs graph-text memory coordination after each reasoning update.
5. The system returns an `AnswerResult`.

Core modules:

- `wemg/system.py`: orchestration and lifecycle.
- `wemg/reasoning/mcts.py`: MCTS search.
- `wemg/reasoning/cot.py`: CoT search.
- `wemg/reasoning/generator.py`: retrieval and generation pipeline.
- `wemg/retrieval/wikidata.py`: Wikidata/SPARQL access.
- `wemg/llm/roles.py`: role definitions and structured I/O.

## Installation

Requirements:

- Python 3.10+
- OpenAI-compatible LLM endpoint
- Optional Redis for caching

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

Example:

```bash
conda run -n wemg python -m wemg "question" search.strategy=cot node_generation.n_hops=2
```

Environment variables:

- `API_KEY`: LLM key and related embedding keys if unset in config.
- `SERPER_API_KEY`: web search key.
- `REDIS_PASSWORD`: Redis password.
- A repo-level `.env` file is loaded automatically when present.

## Self-hosted Wikidata endpoint (recommended)

For large retrieval workloads (or conversion scripts that map Freebase IDs to Wikidata IDs), use a local SPARQL endpoint instead of the public Wikidata Query Service.

References:

- qEndpoint Wikidata image: <https://github.com/the-qa-company/qEndpoint#qacompanyqendpoint-wikidata>
- Freebase/Wikidata conversion notebook: <https://github.com/yuancu/freebase-wikidata-convert/blob/main/conversion.ipynb>

### 1. Start a local Wikidata endpoint with Docker

Truthy dump (smaller):

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=6G qacompany/qendpoint-wikidata
```

Full Wikidata dump (larger):

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=10G --env HDT_BASE=wikidata_all qacompany/qendpoint-wikidata
```

Notes:

- Keep `MEM_SIZE >= 6G` for truthy and `>= 10G` for full dumps.
- First startup can take a long time because the HDT index is downloaded and initialized.
- Disk usage is large; plan capacity before running.

### 2. Verify endpoint is reachable

```bash
curl -H 'Accept: application/sparql-results+json' \
  http://localhost:1234/api/endpoint/sparql \
  --data-urlencode 'query=SELECT * WHERE { ?s ?p ?o } LIMIT 5'
```

The qEndpoint web UI is available at <http://localhost:1234>.

### 3. Use local endpoint in conversion scripts

The conversion flow in `freebase-wikidata-convert/conversion.ipynb` initializes:

```python
EntityConverter("http://localhost:1234/api/endpoint/sparql")
```

This is strongly recommended for bulk Freebase→Wikidata ID conversion. The public endpoint (`https://query.wikidata.org/sparql`) is much slower and may rate-limit large jobs.

### 4. Point WEMG retrieval to local endpoint (optional)

Current Wikidata SPARQL requests in WEMG default to the public endpoint constant in `wemg/retrieval/wikidata.py`.

If you want WEMG itself to query your local qEndpoint instance, update that endpoint to:

```text
http://localhost:1234/api/endpoint/sparql
```

After changing it, run tests:

```bash
conda run -n wemg pytest tests/retrieval/test_wikidata.py
```

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
  config.py
  config.yaml
  system.py
  llm/
  retrieval/
  reasoning/
  evaluation/
  utils/
tests/
examples/
retriever_corpora/
```

## Troubleshooting

- If retrieval fails in corpus mode, verify `retriever.corpus.corpus_path` and `retriever.corpus.index_path`.
- If web search mode fails, verify `SERPER_API_KEY` and `retriever.type=web_search`.
- If cache is enabled but Redis is unavailable, check cache connection settings.

## Contributing

1. Create a feature branch.
2. Add or update tests in `tests/`.
3. Run relevant tests and then `pytest tests/`.
4. Open a pull request with clear change notes.

## License

MIT
