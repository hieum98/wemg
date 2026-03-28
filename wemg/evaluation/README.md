# Evaluation CLI and artifacts

Run WEMG on a benchmark, stream results to JSONL, persist per-question reasoning artifacts, and aggregate metrics. This document covers the CLI, what is saved, rescoring from logs, visualization, and profiling.

## How to run

Entry point:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate [OPTIONS] [OVERRIDES...]
```

- **`--config`**: Path to WEMG YAML. If omitted, the project default config is used (see `wemg.config.get_default_config_path()`).
- **`OVERRIDES`**: Space-separated `key=value` tokens. Keys listed below are **evaluation** parameters; any other `key=value` is forwarded as a **WEMG config** override (same rules as elsewhere in the project). A leading `+` on a key is ignored (Hydra-style), e.g. `+output_path=...` works.

Minimal example:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/bamboogle
```

`dataset_name_or_path` is **required**. Values can be a known short name (see `wemg.evaluation.datasets`), a HuggingFace dataset id, a local HF dataset directory, or a `.json` / `.jsonl` file with the right columns.

Example with WEMG config overrides (everything after the eval keys applies to `WEMGSystem`):

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  dataset_name_or_path=qald-10 \
  output_path=results/qald-10 \
  max_examples=100 \
  resume=true \
  llm.model_name=Qwen/Qwen3-8B \
  search.strategy=mcts
```

### Evaluation override keys

| Key | Default | Meaning |
|-----|---------|--------|
| `dataset_name_or_path` | *(required)* | Dataset name, HF id, or path. |
| `output_path` | `./results` | Directory for log, metrics, `config.yaml`, and `artifacts/`. |
| `resume` | `true` | If the JSONL log exists, skip questions whose `question` string already appears in the log. |
| `max_examples` | *(none)* | Truncate dataset after load. |
| `shuffle` | `false` | Shuffle with seed 42 before truncation. |
| `question_column` | `question` | Question field name in the dataset. |
| `answer_column` | `answer` | Gold answer field name. |
| `max_concurrent` | *(auto)* | Max workers per batch chunk for `answer_batch` (capped by batch size and config). |
| `log_batch_size` | *(see runner)* | Questions per chunk before appending to the log; smaller = more frequent checkpoints. |
| `clear_kb_cache_every_n_batches` | `1` | Clear in-process Wikidata triple caches every N chunks; `0` or `None` disables. |
| `score_only` | `false` | Do **not** run the system; only score an existing dataset that already has predictions (see below). |
| `prediction_column` | `predicted_answer` | Column to read predictions from when `score_only=true`. |

When **`score_only=true`**, the tool loads `dataset_name_or_path` with `load_dataset_any`, then calls `DatasetEvaluator.score_from_predictions`. The dataset must expose `question_column`, `answer_column`, and `prediction_column`. It writes **`metrics.json`** under `output_path` only (no new JSONL rows). Useful if you exported predictions to JSONL and converted or merged them into a JSON file with the expected columns.

### Outputs under `output_path`

| Path | Purpose |
|------|--------|
| `config.yaml` | Resolved WEMG config after env and overrides (reproducibility). |
| `evaluation_log.jsonl` | One JSON object per line, per question (see below). |
| `metrics.json` | Aggregates: `mean_sub_em`, `mean_acc`, `total_questions`, `valid_questions`, and optional `pass_at_*` / `overall_pass_rate` when pass@k is present. |
| `summary.txt` | Same metrics as human-readable lines. |
| `artifacts/q_<idx>_<sha1>_<slug>/` | Per-question search tree, textual memory, graph (see below). |

---

## What is saved in `evaluation_log.jsonl`

Each line is one record. Typical fields:

| Field | Notes |
|-------|--------|
| `question` | Question string (also the resume key). |
| `correct_answer` | Gold answer from the dataset (may be string or list depending on source). |
| `predicted_answer` | What is scored: `concise_answer` or full `answer` from the system; or an `Error: ...` string on failure. |
| `full_answer` | Long-form system answer when present (success path). |
| `sub_em` | Substring exact match score in `[0, 1]` (see `compute_sub_em`). |
| `acc` | Always `null` in the log as written today; LLM-based Acc is computed after all chunks and rolled into **`metrics.json`** / **`summary.txt`** only (not patched back into JSONL lines). |
| `pass_at_k` | Pass@k rank when the run produced it; `null` otherwise. |
| `error` | Present when generation or scoring failed for that row. |
| `artifacts` | Object with `artifact_dir`, `search_tree_path`, `textual_memory_path`, `graph_memory_path` (paths may be `null` if nothing was saved). |
| `artifacts_error` | If persisting artifacts failed, this string is set instead of relying on silent failure. |

Order of lines follows append order while evaluation runs (dataset order within each chunk). For **resume**, already logged questions are skipped; the file is only appended for new work.

---

## Recomputing scores from the log

**Sub-EM / Acc / aggregates:** `score_from_predictions` recomputes per-example Sub-EM and Acc (via the evaluator LLM) and refreshes **`metrics.json`**. It does **not** rebuild **`pass_at_k`**—aggregates that depend on pass@k need the original multi-sample traces. Use this path when you need **per-question Acc** values, since those are not stored on each JSONL row after a full `evaluate` run.

### Option A: Python API (read JSONL, build a scoring dataset)

```python
import json
from pathlib import Path

from wemg.evaluation.runner import DatasetEvaluator
from wemg.system import WEMGSystem

output_path = Path("results/bamboogle")
log_path = output_path / "evaluation_log.jsonl"

rows = []
with open(log_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

dataset_for_scoring = [
    {
        "question": row.get("question", ""),
        "answer": row.get("correct_answer", ""),
        "predicted_answer": row.get("predicted_answer", ""),
    }
    for row in rows
]

system = WEMGSystem(config_path=str(output_path / "config.yaml"))
try:
    evaluator = DatasetEvaluator(system)
    metrics = evaluator.score_from_predictions(
        dataset_for_scoring,
        output_path=str(output_path),
        question_column="question",
        answer_column="answer",
        prediction_column="predicted_answer",
    )
finally:
    system.close()

print("Recomputed metrics:", metrics)
```

This **overwrites** `output_path/metrics.json`. Adjust paths and column names if your log used different keys.

### Option B: CLI `score_only` after exporting predictions

Convert or save a `.json` / `.jsonl` file whose rows have `question`, `answer`, and `predicted_answer`, then:

```bash
conda run -n wemg python -m wemg.evaluation.evaluate \
  score_only=true \
  dataset_name_or_path=/path/to/scoring_set.jsonl \
  output_path=results/rescore_run \
  question_column=question \
  answer_column=answer \
  prediction_column=predicted_answer
```

---

## Visualizing answers from the log (search tree and graph)

Paths live in each log row under `artifacts`. Helpers are in `wemg.evaluation.artifacts`.

### Load one question by index or exact question text

```python
from wemg.evaluation.artifacts import (
    load_question_artifacts,
    print_saved_search_tree,
    visualize_graph_memory,
    visualize_graph_memory_interactive,
)

output_path = "results/bamboogle"

data = load_question_artifacts(output_path, index=0)
# or: load_question_artifacts(output_path, question="… exact string …")
# or: load_question_artifacts(output_path, entry={...})  # raw dict from JSONL

tree = data["search_tree"]               # dict or None
textual_memory = data["textual_memory"]  # list[str] or None
graph = data["graph_memory"]             # networkx graph or None
paths = data["artifact_paths"]
```

### Search tree (text hierarchy)

`search_tree.json` holds `node_type`, `content`, `children`, and MCTS fields (`visits`, `value`) when present.

```python
print_saved_search_tree(data["artifact_paths"]["search_tree_path"])
# or print_saved_search_tree(tree)
```

### Graph memory (static PNG)

Uses `wemg.utils.graph.visualize_graph`. Writes an image when `save_path` is set.

```python
visualize_graph_memory(
    data["artifact_paths"]["graph_memory_path"],
    title="Question 0 — graph memory",
    save_path="tmp/q0_graph.png",
)
```

### Graph memory (interactive HTML)

Uses `visualize_graph_interactive`; in Jupyter, can embed an `IFrame`; `open_in_browser=True` opens the file URL.

```python
visualize_graph_memory_interactive(
    data["artifact_paths"]["graph_memory_path"],
    title="Question 0 — graph memory",
    save_path="tmp/q0_graph.html",
    notebook_mode=True,
    open_in_browser=False,
)
```

**Files on disk per question** (under `artifacts/.../`):

- `search_tree.json` — reasoning tree.
- `working_memory_textual.json` — list of textual memory entries (plus confirmed global facts when saved).
- `working_memory_graph.pkl` — pickled `networkx` graph (working memory composed with global-knowledge graph when applicable).

### Other helpers

- `find_artifacts_entry(output_path, index=..., question=...)`
- `load_search_tree_json(path)`
- `load_graph_memory(path)`

---

## Profiling

### cProfile (whole process)

Run the evaluator as a module and write a stats file:

```bash
mkdir -p results/profiling
conda run -n wemg python -m cProfile -o results/profiling/eval.prof \
  -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/profiling/bamboogle
```

Inspect (examples):

```bash
conda run -n wemg python -m pstats results/profiling/eval.prof
# sort cumulative / stats <n>
```

Or: `pip install snakeviz` then `snakeviz results/profiling/eval.prof`.

Keep **`--config`** before the overrides if you use a non-default config path:

```bash
conda run -n wemg python -m cProfile -o eval.prof \
  -m wemg.evaluation.evaluate --config /path/to/config.yaml \
  dataset_name_or_path=bamboogle output_path=results/pf
```

### In-run timing logs

`DatasetEvaluator.evaluate` logs `PROFSTEP` lines around each chunk (`chunk_start`, `answer_batch_done`, `append_logs_done`) with elapsed milliseconds—enable your logging configuration (e.g. `INFO`) to correlate wall time across stages without cProfile.

---

## Notes

- Resume deduplicates by **exact** `question` string; changing whitespace or wording produces a new run row.
- Graph files are pickle for full `networkx` attributes; load only with a compatible environment.
- `load_question_artifacts(...)` accepts a preloaded `entry` dict if you already parsed a line from `evaluation_log.jsonl`.
