# Results: what this work established, and what it did not

Entry point for the experimental record. The full evidence is in
[`plan_idea_and_results.md`](plan_idea_and_results.md); the design rationale and the
hypotheses that were tried and rejected are in
[`plan_channel_status_and_plan.md`](plan_channel_status_and_plan.md) (internal working log).

All numbers below are on `datasets/musique_depth.jsonl` (120 multi-hop questions) unless
stated, with Qwen3-32B on Bedrock. Paired tests throughout: two-sided exact **sign test** for
binary outcomes, **Wilcoxon signed-rank** for cost, bootstrap CIs on means.

---

## Read this first: the noise floor

Two runs of a **byte-identical configuration**, launched concurrently, differ by 1 row out of
117 — and **17 rows (14.5%) flip** between them (12% with reasoning off). At n = 117-234 this
design cannot resolve accuracy differences below roughly **±5 points**.

Nothing measured here exceeds ±4.3 points on accuracy. Every cost effect exceeds its own noise
by 10-100x. **The claims worth making are cost claims.**

---

## What is claimable

### 1. The plan reduces cost at unchanged accuracy — replicated in both inference regimes

| measurement | LLM calls | prompt tokens |
|---|---|---|
| CoT, reasoning off, current code | **-24.8, p < 0.0001** | -36,993, p = 0.0179 |
| CoT, reasoning off, `fix_*` (234 pairs) | -12.6, p = 0.0025 | not significant |
| MCTS plan-rollout, reasoning off (118 pairs) | **-36.2, p < 0.0001** | -69,756, p = 0.0023 |
| MCTS plan-rollout, reasoning on | -3.9, p = 0.0274 | -15,023, p = 0.0049 |

Four independent measurements, all surviving Wilcoxon, at accuracy that never moves.

### 2. Selective reasoning: identical accuracy for 40% of the reasoning spend

Reasoning on the roles that reason over evidence (`heavy`, `plan`), off the bounded
extraction/consolidation roles:

| | full reasoning | selective | |
|---|---|---|---|
| relaxed sub-EM | 21.37% | 22.22% | +0.85 pts, p = 1.0000 |
| reasoning tokens / q | 23,416 | **9,443** | **-13,974, p < 0.0001** |
| output tokens / q | 29,817 | **16,526** | -13,291, p < 0.0001 |

Mechanism confirmed, not assumed: memory items 3.71 -> 5.36, extracted facts 11.71 -> 13.71.

### 3. A shared plan harms tree search — a design result

Under MCTS with one plan shared across siblings: distinct subquestions per question 9.9 -> 6.3
(-36%), mean sibling-subtree overlap 10.2% -> 23.1%, at identical accuracy. A plan is a
variance-*reduction* device, and pUCT can only prefer one child over another when the children
differ. Hence `mcts_plan_scope="rollout"`, which keeps the plan's within-chain benefits and
preserves sibling diversity by construction.

### 4. One complete causal chain: the plan caused a wrong-answer mode, and it was removed

On questions with >= 2 intents, answering with a referent bound to a non-terminal intent and to
no terminal one — **87% of such answers are wrong**:

| | plan | no-plan | ratio | sign test |
|---|---|---|---|---|
| original | **10.6%** | 5.3% | 1.8x | **p = 0.0054** |
| after *omitting* scaffolding from findings | 8.5% | 4.7% | 1.8x | p = 0.0636 |
| after **labelling** it as an exclusion | **5.6%** | **5.6%** | **1.00x** | **p = 1.0000** |

The first fix failed instructively: the referent survives in `candidate_answers`, which both
arms share, and a *good* decomposition makes it the crispest candidate there. Synthesis had to be
*told* which referents are inputs, not left to infer it from an absence. This restored parity —
it did not make the plan beat no-plan.

### 5. Reasoning trades recall for precision, in a recall-bound pipeline

Same code, reasoning the only difference:

| | facts extracted | memory items | hops | 1-hop stops |
|---|---|---|---|---|
| reasoning ON | 11.5-11.7 | 3.3-3.7 | 2.69 | 23% |
| reasoning OFF | **18.7-18.8** | **6.5-7.0** | 3.51 | 3% |

Reasoning extracts ~38% fewer facts and retains ~50% fewer memory items. Extraction and
consolidation are recall tasks with asymmetric costs — a spurious fact merely occupies a slot, a
dropped fact is unrecoverable — so sharper filtering optimises the wrong objective. Confirmed by
intervention (turning reasoning off those roles reverses it) and by §14.13 (re-showing dropped
facts changes conversion by *exactly zero*).

---

## What is NOT claimable

### The plan does not improve accuracy

Eight well-powered measurements, consistently positive in CoT, never significant:

| contrast | gap | p |
|---|---|---|
| CoT, reasoning off, `lab_*` (234 pairs) | +2.14 pts | 0.5515 |
| CoT, reasoning off, `fix_*` (234 pairs) | +0.86 pts | 0.8830 |
| CoT, reasoning off, current code (117) | +3.42 pts | 0.4240 |
| CoT, reasoning **on** (117) | +1.71 pts | 0.8145 |
| MCTS plan-rollout, reasoning off (118) | +4.24 pts | 0.3593 |
| MCTS plan-rollout, reasoning **on** (117) | -1.71 pts | 0.7905 |

### Reasoning does not improve accuracy either

Clean 2x2, same code, same day, reasoning the only cross-arm difference:

| | plan ON | plan OFF |
|---|---|---|
| reasoning ON | 23.08% | 21.37% |
| reasoning OFF | **27.35%** | **23.93%** |

Reasoning is directionally **worse** in both arms (-2.56, -4.27 pts; p = 0.33-0.69) for ~19.5k
extra output tokens per question. **There is no plan x reasoning interaction.**

### Measured and rejected

Replanning (fired on 20/62 questions, +1 row — the ambiguity is in the world, not the plan);
four ledger-side fixes (the ledger is 0.49 lines against ~8.5 of synthesis input, so ~6% —
one ceiling explaining all four); candidate reordering; and re-surfacing consolidation-dropped
evidence (conversion identical, 15/22 in both arms, p = 1.0000).

---

## Limitations

1. **Statistical power.** ±5 points is the resolution at this n. The fix is more questions, not
   more interventions.
2. **Retrieval is the prior bottleneck.** Gold reaches memory on only 25-30% of questions (down
   from 41.4% before the multi-provider search chain), which caps everything downstream. ~78% of
   failures are retrieval-bound and no planning change can reach them.
3. **Live web dependence.** `web_search` leads with Wikipedia, the gold source for MuSiQue, and
   is rate-limited in practice. Absolute accuracy is not comparable across weeks.
4. **The reasoning axis is directional, not significant.** The mechanism (~50% less evidence) is
   solid and large; the accuracy consequence is inside the noise floor.

---

## Reproducing the numbers

```bash
source .venv/bin/activate          # NOT conda
[ -f .env ] && source .env         # provider keys; never inline them
```

Arms are selected on the command line; one config serves all of them. **Run at most two arms
concurrently** — Wikipedia is the head of the provider chain and a third arm drives it into
sustained 429s, degrading the runs already in flight. Both arms of a contrast must run
*concurrently*, never sequentially, or the 14.5% flip rate confounds the comparison.

```bash
EV="python -m langgraph_coe.evaluation.evaluate --config langgraph_coe/config.eval.yaml
    dataset_name_or_path=./datasets/musique_depth.jsonl level_column=level"

# CoT, plan on vs off
$EV output_path=./results/cot_plan_on  search.strategy=cot search.plan.enabled=true  search.plan.replan_max=0 &
$EV output_path=./results/cot_plan_off search.strategy=cot search.plan.enabled=false &

# MCTS, plan-rollout on vs off. max_concurrent=4 per arm: MCTS issues far more retrieval per
# question (~13 Wikipedia 429s/min at the default 8 against CoT's 2), and two arms at 4 give the
# same aggregate rate as one at 8 without serialising the contrast.
$EV output_path=./results/mcts_plan_on  search.strategy=mcts max_concurrent=4 \
   search.plan.enabled=true search.plan.replan_max=0 search.plan.mcts_plan_scope=rollout &
$EV output_path=./results/mcts_plan_off search.strategy=mcts max_concurrent=4 search.plan.enabled=false &
```

**Reasoning on.** `enable_thinking` is inert on Bedrock (it rides in `chat_template_kwargs`,
which only a self-hosted SGLang chat template reads). Use `reasoning_effort`, and set it on
**all five tiers**: `config.eval.yaml` declares the tier once as `&bedrock_tier` and aliases it,
but pydantic validates each key into its own `TierConfig`, so setting one leaves four
reasoning-off and silently produces a mixed arm. Only `"high"` engages the model. Raise
`max_tokens` with it — reasoning is billed *inside* it, and at the stock 4096 `open_ie` truncated
on 15% of its completions.

```bash
for t in heavy medium plan light classify; do
  R="$R llm.tiers.$t.reasoning_effort=high llm.tiers.$t.max_tokens=8192"; done
$EV output_path=./results/r_cot_plan_off search.strategy=cot search.plan.enabled=false $R &

# Selective reasoning (the recommended regime if reasoning is wanted at all):
$EV output_path=./results/r_cot_selective search.strategy=cot search.plan.enabled=false \
   llm.tiers.heavy.reasoning_effort=high llm.tiers.heavy.max_tokens=8192 \
   llm.tiers.plan.reasoning_effort=high  llm.tiers.plan.max_tokens=8192 \
   llm.role_tiers.memory_consolidation=medium &
```

`resume` defaults to true, so a re-run continues from an existing `evaluation_log.jsonl`.

### Reporting

```bash
python scripts/reason_report.py      cot_plan_on:cot_plan_off   # accuracy + cost, paired
python scripts/conversion_report.py  cot_plan_on:cot_plan_off   # retrieval vs conversion
python scripts/fix_report.py         cot_plan_on:cot_plan_off   # + the referent-leak rate
```

`reason_report.py` prints an **arm-validity block before any number** — what fraction of
completions actually reasoned, how many were truncated at `max_tokens`, and whether the arms
differ in any config key other than the intended intervention. It reads the `config.yaml` the
runner wrote back, not what was intended. This is not optional bookkeeping: it caught two arms
that launched in the wrong regime while printing the right label, and a reasoning toggle that
LiteLLM was silently dropping.

Every result directory contains the exact `config.yaml` used, plus per-question artifacts
(`plan.json`, `retrieval_log.json`, consolidated memory) under `artifacts/`.

### Metric note

Primary accuracy metric is **`sub_em_short_relaxed`**. Plain `sub_em` requires the gold string
verbatim, and three golds in this dataset arrive wrapped (`"at the city of Cairo, Illinois"`,
`"The Australian Ballet"`, `"four-year"`), so a correctly concise answer could never match them —
worth +1 to +3 rows in **every one of 27 runs** (+1.79 points). It is recorded *beside*
`sub_em_short`, never instead of it, so historical numbers are not silently restated.
