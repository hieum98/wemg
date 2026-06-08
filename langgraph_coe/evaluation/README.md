# langgraph_coe Evaluation

Benchmark the **langgraph_coe** system (LangGraph CoT/MCTS) and emit results in the
**exact same format** as the legacy `coe.evaluation` so the two are directly
comparable.

## Run

```bash
conda activate coe   # or use `uv run`
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=bamboogle \
    output_path=./results/lgc_bamboogle \
    search.strategy=mcts
```

The corpus index is ~99 GB and the live endpoints must be reachable, so run on a
node with the RAM + network (files are shared across nodes), e.g. `ssh n0162`.

## Arguments

**Evaluation keys** (same as legacy): `dataset_name_or_path`, `output_path`,
`resume` (default `true`), `score_only`, `max_examples`, `shuffle`,
`question_column`, `answer_column`, `level_column`, `max_concurrent`,
`log_batch_size`.

**Config overrides**: any dotted `langgraph_coe` config key, e.g.
`search.strategy=cot`, `search.mcts.num_iterations=8`,
`llm.tiers.heavy.api_base=http://n0152:30000/v1`, `cache.enabled=true`.
Hydra-style `+key=` prefixes are accepted and ignored.

## Output (identical to legacy)

Written under `output_path/`:

| File | Contents |
|------|----------|
| `evaluation_log.jsonl` | one row/question: `question`, `correct_answer`, `predicted_answer`, `full_answer`, `sub_em_short`, `sub_em_long`, `pass_at_k`, `acc_short`, `acc_long`, `level`, `artifacts` (or `error`) |
| `metrics.json` | `{"short_answer": …, "long_answer": …, "by_level"?: …}` |
| `summary.txt` | human-readable short/long metric block |
| `config.yaml` | resolved config (env + overrides) for reproducibility |
| `artifacts/q_XXXXX_<digest>_<slug>/` | `search_tree.json` (MCTS), `working_memory_textual.json`, `working_memory_graph.pkl` |

## How it differs from legacy internally

- **Backend**: drives the compiled CoT/MCTS LangGraph (`langgraph_coe.system`)
  instead of `COESystem`. The runtime (FAISS index + Wikidata client) is wired
  **once** and the graph is invoked per question inside a single event loop —
  the index is never reloaded per question.
- **Sub-EM / aggregates**: byte-for-byte the same logic as `coe.evaluation.metrics`.
- **Acc judge**: the `langgraph_coe` `EVALUATOR` role (same `rating/10` mapping).
- **`pass_at_k`**: the public `AnswerResult` envelope does not expose per-step
  pass signals, so this column is `null` (the key is kept for schema parity).
