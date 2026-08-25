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
| `artifacts/q_XXXXX_<digest>_<slug>/` | `search_tree.json` (MCTS), `working_memory_textual.json`, `working_memory_graph.pkl`, `plan.json` (when a plan ran) |

## Plan channel ablation

`search.plan` adds an explicit prose plan that conditions subquestion generation.
It is **off by default** so the baseline arm is byte-identical to the pre-plan
behaviour (the plan nodes are not added to the graph at all).

| arm | override | what it measures |
|-----|----------|------------------|
| **A0** | *(default)* `search.plan.enabled=false` | baseline |
| **A1** | `search.plan.enabled=true search.plan.replan_max=0` | static plan; the gate computes `plan_action` and logs it but never routes |
| **A2** | `search.plan.enabled=true search.plan.replan_max=2` | plan + replan on contested/falsified discharge |

```bash
# A0 vs A1 — run this pair first; if A1 ≈ A0 nothing downstream matters.
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/plan_a0 search.strategy=cot search.plan.enabled=false

python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/plan_a1 search.strategy=cot \
    search.plan.enabled=true search.plan.replan_max=0
```

Compare `metrics.json` sub-EM **and mean answer length** — `compute_sub_em` is a
substring test, so hedging inflates it. `artifacts/*/plan.json` carries the plan,
the ledger, the per-hop `plan_action_log` (`armed` records whether the router was
allowed to act) and `iteration_history`; A1's log alone gives the trigger's fire
rate. `2wiki` is the negative control — its comparison sets are enumerated in the
question, so referent multiplicity cannot fail there.

Related knobs: `search.mcts.branch_local_memory=true` snapshots memory per tree
node so a branch's retrieval writes stay in its subtree (default `false` keeps the
documented shared-memory parity); `retriever.enabled=false` skips corpus fan-out
when no local index is available.

## Smoke test

`python -m langgraph_coe.scripts.smoke_test` runs a wiring check (not an accuracy
run) against real services using `langgraph_coe/config.smoke.yaml` — Qwen3-32B on
Bedrock for every role, Wikidata, web search instead of the local corpus, no
reranker. It asserts a plan is produced and injected, that **no plan text reaches
`text_memory`**, and that both strategies terminate. See the header of that module
for prerequisites; `COE_SPARQL_ENDPOINT` overrides the SPARQL endpoint.

## How it differs from legacy internally

- **Backend**: drives the compiled CoT/MCTS LangGraph (`langgraph_coe.system`)
  instead of `COESystem`. The runtime (FAISS index + Wikidata client) is wired
  **once** and the graph is invoked per question inside a single event loop —
  the index is never reloaded per question.
- **Sub-EM / aggregates**: byte-for-byte the same logic as `coe.evaluation.metrics`.
- **Acc judge**: the `langgraph_coe` `EVALUATOR` role (same `rating/10` mapping).
- **`pass_at_k`**: the public `AnswerResult` envelope does not expose per-step
  pass signals, so this column is `null` (the key is kept for schema parity).
