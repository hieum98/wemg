# Evaluation Artifacts (Per-Question Reasoning State)

This module now persists per-question reasoning artifacts during evaluation so you can reload and inspect them later.

## Compute evaluation metrics

You can compute metrics in two ways:

### 1) Run full system (generate predictions + score)

Use the evaluation CLI to run WEMG and compute metrics in one step:

```bash
python -m wemg.evaluation.evaluate \
  dataset_name_or_path=bamboogle \
  output_path=results/bamboogle
```

Outputs written under `output_path`:

- `evaluation_log.jsonl`: per-question predictions and scores
- `metrics.json`: aggregate metrics (Sub-EM, Acc, pass@k when available)
- `summary.txt`: human-readable summary
- `artifacts/`: per-question reasoning artifacts

### 2) Recompute metrics from logfile entries (no full system re-run)

If you already have `evaluation_log.jsonl`, you can reload it and use
`DatasetEvaluator.score_from_predictions(...)` to recompute metrics:

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

# score_from_predictions expects "answer" by default.
# Map logfile's "correct_answer" -> "answer".
dataset_for_scoring = [
    {
        "question": row.get("question", ""),
        "answer": row.get("correct_answer", ""),
        "predicted_answer": row.get("predicted_answer", ""),
    }
    for row in rows
]

system = WEMGSystem()  # or WEMGSystem(config_path=str(output_path / "config.yaml"))
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

Note: `score_from_predictions` recomputes Sub-EM and Acc from predictions. It does not
recompute pass@k, because pass@k depends on multi-sample generation traces.

## What gets saved

When running `DatasetEvaluator.evaluate(...)`, each question writes artifacts under:

- `OUTPUT_PATH/artifacts/q_<index>_<sha1>_<slug>/`

Each question directory contains separate files:

- `search_tree.json`  
  Serialized reasoning tree (`node_type`, `content`, recursive `children`, and MCTS fields like `visits`/`value` when available).
- `working_memory_textual.json`  
  JSON list of textual memory entries.
- `working_memory_graph.pkl`  
  Pickled `networkx` graph from working memory.

Each JSONL record in `OUTPUT_PATH/evaluation_log.jsonl` includes:

- `artifacts.artifact_dir`
- `artifacts.search_tree_path`
- `artifacts.textual_memory_path`
- `artifacts.graph_memory_path`

If artifact persistence fails for a row, the log entry still exists and includes `artifacts_error`.

## Load helpers

Use `wemg.evaluation.artifacts`:

- `find_artifacts_entry(output_path, index=..., question=...)`
- `load_search_tree_json(path)`
- `load_graph_memory(path)`
- `load_question_artifacts(output_path, index=..., question=..., entry=...)`
- `print_saved_search_tree(tree_or_path)`
- `visualize_graph_memory(graph_or_path, title=..., save_path=...)`

## Notebook usage

```python
from wemg.evaluation.artifacts import (
    load_question_artifacts,
    print_saved_search_tree,
    visualize_graph_memory,
)

output_path = "results/bamboogle"

# Load artifacts for first logged question
data = load_question_artifacts(output_path, index=0)

# Access loaded content
tree = data["search_tree"]               # dict (serialized tree)
textual_memory = data["textual_memory"]  # list[str]
graph = data["graph_memory"]             # networkx graph

# Print tree in compact hierarchy form (similar to system tree print)
print_saved_search_tree(data["artifact_paths"]["search_tree_path"])

# Visualize graph memory (saves image if save_path is provided)
visualize_graph_memory(
    data["artifact_paths"]["graph_memory_path"],
    title="Question 0 Graph Memory",
    save_path="tmp/q0_graph.png",
)
```

## Profling the evaluation

```bash
# Load environment variables from .env file
set -a
source .env
set +a

# Run evaluation and profile the execution
python -m cProfile -o results/profiling/cprofile/profile.prof wemg.evaluation.evaluate \
    +dataset_name_or_path=<dataset_name_or_path> \
    +output_path=results/profiling/<dataset_name_or_path>
```

## Notes

- `load_question_artifacts(...)` can identify an entry by index or exact question string.
- Graph files are saved with pickle to preserve full `networkx` structure/attributes.
- Search trees are saved as JSON for easy manual inspection and forward compatibility.
