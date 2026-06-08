# Evaluation CLI and artifacts

Run COE on a dataset, write predictions to JSONL, save artifacts per question, and compute aggregate metrics.

## Quick start

```bash
conda run -n coe python -m coe.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/bamboogle
```

Fast iteration example:

```bash
conda run -n coe python -m coe.evaluation.evaluate \
  dataset_name_or_path=cwq \
  output_path=results/cwq-dev \
  max_examples=50 \
  resume=true \
  search.strategy=mcts
```

## CLI usage

Entry point:

```bash
conda run -n coe python -m coe.evaluation.evaluate [OPTIONS] [OVERRIDES...]
```

Notes:

- `dataset_name_or_path` is required.
- `--config` selects a YAML config file; otherwise the default config is used.
- Extra `key=value` pairs are interpreted as evaluation keys or forwarded as COE config overrides.
- A leading `+` on keys is accepted for compatibility.
- A repo-level `.env` file is loaded automatically before config validation.

Example with overrides:

```bash
conda run -n coe python -m coe.evaluation.evaluate \
  dataset_name_or_path=qald_10 \
  output_path=results/qald_10 \
  max_examples=100 \
  resume=true \
  llm.model_name=Qwen3-8B \
  search.strategy=mcts
```

## Evaluation keys

| Key | Default | Description |
|-----|---------|-------------|
| `dataset_name_or_path` | required | Dataset name, HF id, local dataset dir, or JSON/JSONL path. |
| `output_path` | `./results` | Output folder for config, logs, metrics, and artifacts. |
| `resume` | `true` | Skip already logged questions by exact `question` match. |
| `max_examples` | none | Limit number of examples after load. |
| `shuffle` | `false` | Shuffle with seed 42 before truncation. |
| `question_column` | `question` | Input question field. |
| `answer_column` | `answer` | Gold answer field. |
| `max_concurrent` | auto | Worker cap for batch answering. |
| `log_batch_size` | runner default | Flush frequency for JSONL appends. |
| `clear_kb_cache_every_n_batches` | `1` | In-process KB cache clear interval. `0`/`None` disables. |
| `score_only` | `false` | Score an existing predictions dataset without generation. |
| `prediction_column` | `predicted_answer` | Prediction field name used by `score_only=true`. |

## Output files

Generated under `output_path`:

- `config.yaml`: resolved runtime config.
- `evaluation_log.jsonl`: one line per question.
- `metrics.json`: aggregate metrics.
- `summary.txt`: human-readable summary.
- `artifacts/q_<idx>_<hash>_<slug>/`: per-question artifact directory.

Typical artifact files:

- `search_tree.json`
- `working_memory_textual.json`
- `working_memory_graph.pkl`

## Rescoring from predictions

Use `score_only=true` to compute scores from existing predictions:

```bash
conda run -n coe python -m coe.evaluation.evaluate \
  score_only=true \
  dataset_name_or_path=/path/to/predictions.jsonl \
  output_path=results/rescore \
  question_column=question \
  answer_column=answer \
  prediction_column=predicted_answer
```

## Artifact helpers

Utilities in `coe.evaluation.artifacts` help locate and visualize saved trees/graphs.

Common helpers:

- `load_question_artifacts`
- `find_artifacts_entry`
- `print_saved_search_tree`
- `load_graph_memory`
- `visualize_graph_memory`
- `visualize_graph_memory_interactive`

## Profiling

Example with cProfile:

```bash
mkdir -p results/profiling
conda run -n coe python -m cProfile -o results/profiling/eval.prof \
  -m coe.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/profiling/bamboogle
```

Inspect profile:

```bash
conda run -n coe python -m pstats results/profiling/eval.prof
```

## Freebase

### Infrastructure setup

**1 — Freebase Virtuoso DB** (53 GB; ~100 GB RAM required at runtime)

```bash
conda activate coe
# Install once (skip if already installed)
conda install -c conda-forge virtuoso-opensource

cd /home/hieum/uonlp/coe
# Choose any local directory for the Freebase DB files
mkdir -p data/freebase
# Download virtuoso_db.zip from https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
curl -L "https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip?dl=1" -o data/freebase/virtuoso_db.zip
unzip -o data/freebase/virtuoso_db.zip -d data/freebase

# Start via COE wrapper (HTTP SPARQL on :3001)
python -m coe.retrieval.virtuoso start 3001 -d data/freebase/virtuoso_db

# Check server status
python -m coe.retrieval.virtuoso status 3001

# To stop
python -m coe.retrieval.virtuoso stop 3001
```

**2 — QID→MID mapping file** (required for Wikidata-search bridge)

Use the notebook helper in `draft.ipynb` to build `qid_to_mid.json` from
`fid2qid.pkl` (online source), then point config to that map:

```bash
node_generation.freebase_qid_to_mid_map_path=/path/to/qid_to_mid.json
node_generation.freebase_qid_to_mid_candidates=5
```

### Run evaluation

```bash
conda run -n coe python -m coe.evaluation.evaluate \
  dataset_name_or_path=grail_qa \
  node_generation.kb_source=freebase_live \
  node_generation.freebase_sparql_url=http://localhost:3001/sparql \
  node_generation.freebase_qid_to_mid_map_path=/path/to/qid_to_mid.json \
  node_generation.freebase_qid_to_mid_candidates=5 \
  output_path=results/grailqa_freebase_live
```

`metrics.json` will contain a `by_level` key with `overall`, `i.i.d.`, `compositional`, and `zero-shot` sub-metrics.

### Methodological difference from KBQA-o1

KBQA-o1 does **not** perform text-to-MID entity discovery at inference time.
Its entity candidates are **gold MIDs pre-extracted from gold SPARQL annotations** during preprocessing (`data_process.py` → `prompt_function()` → `entities` field).
The LLM only selects *which* gold MID to start with — it does not discover entities from question text.
The `ELQ_SERVICE_URL` in KBQA-o1's config is unused dead code (never called at inference).

COE performs **text→MID entity discovery** via LLM NER → Wikidata search → QID→MID map → SPARQL.
This difference should be noted when comparing results: KBQA-o1 has privileged access to gold entities; COE does not.

## Notes

- Resume logic uses exact `question` string matches.
- Graph artifacts are pickled NetworkX objects.
- LLM-based accuracy is aggregated in metrics outputs.
- For setup and architecture context, see the root `README.md`.
