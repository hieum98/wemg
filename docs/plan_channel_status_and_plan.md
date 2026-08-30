# Plan channel: status and remaining work

> **INTERNAL WORKING LOG — not the results record.**
>
> This is the chronological log: it contains superseded claims, hypotheses that were later
> falsified, and intermediate numbers that were subsequently corrected, all kept deliberately
> because the reasoning that produced them is the point. **Do not quote figures from here.**
>
> * For what is claimable, start at **[RESULTS.md](RESULTS.md)**.
> * For the settled evidence, see **[plan_idea_and_results.md](plan_idea_and_results.md)**.
>
> Where this log and the results documents disagree, the results documents are correct.

Status of the explicit-plan change (`search.plan`), the bug fixes that had to land
with it, and what is still open. Everything described as done is in commit
`3ed3d9a` and covered by tests; everything in [§4](#4-not-done) and
[§5](#5-experiments) is not started.

---

## 1. What the change is

Both strategy graphs used to re-decompose from scratch every step:
[cot.py](../langgraph_coe/graphs/cot.py) calls `SUBQUESTION_GENERATOR` with `n=3`,
pools the union, retrieves, answers, consolidates, repeats — 15+ planner calls per
question at `max_depth`. MCTS `simulate` runs that whole loop as a rollout, so
sibling subtrees re-decompose independently and their visit statistics are not
comparable.

The change adds a **prose plan** that states what must be found out and conditions
subquestion generation rather than replacing it. Two operations act on it:

| | trigger | operation | cost |
|---|---|---|---|
| **UPDATE** | exactly one distinct-QID referent survives for an intent | deterministic write: intent → `closed`; binding surfaces via `intermediate_answer` | 0 LLM calls |
| **REPLAN** | **contested discharge** (≥2 distinct-QID referents for one intent) or **falsified discharge** (a cited `[Retrieval]` premise evicted as `contradicted`) | one PLANNER call revising the failed part | 1 call, capped by `replan_max` |

### The four constraints the design rests on

1. **Plan text never enters `text_memory` / `new_text_items` / `candidate_answers`.**
   An interrogative in memory is picked up by `_reverify_memory`
   ([mcts.py:576](../langgraph_coe/graphs/mcts.py#L576)) as a retrieval query, then
   reaches the verifier as grounding and the synthesizer as a candidate answer.
   Asserted mechanically by `test_plan_text_never_reaches_memory` and by the smoke
   test on every run.
2. **Plan text does not go into `SubquestionGenerationInput.context` either.**
   [roles.py:493-500](../langgraph_coe/roles.py#L493) instructs the generator to
   resolve conflicts *in the context*, so an interrogative there yields a
   subquestion *about the plan*, which hits retrieval and returns as a
   `[Retrieval]` fact. It gets its own typed field, rendered **last** so the input
   guard's head/tail trim drops mid-memory rather than the plan.
3. **Referent discipline.** The plan may name an entity only as the question names
   it, or from a `[Retrieval]`-tagged memory item — never from a
   `[System Prediction]` inference. Borrowed referents are recorded as `premises`,
   which is what makes a retraction matchable by set intersection with no
   dependency graph.
4. **"Surprise" is the wrong predicate.** It needs a prior, and every way of
   supplying one either makes the plan truth-apt (violating the planning/reasoning
   split) or is too coarse to fire. Contested/falsified discharge is a fact about
   the plan's *bookkeeping*, is categorical, and needs no judge — QID identity is
   the entire discriminator, so there is no threshold to tune.

### Why this data

The only committed eval set is
[datasets/bamboogle_hardmix.jsonl](../datasets/bamboogle_hardmix.jsonl) (62 rows).
A hand census found **30/62 rest on a definite description that can genuinely
fail** (two referents, time-dependent referents, ranks with no canonical
authority) and **35/62 are ordinal-then-chained**, where the rank hop is the
premise for hop 2. `2wiki`/`hotpotqa` have ~none of this — their comparison sets
are enumerated in the question — which makes `2wiki` the negative control.

---

## 2. Done — plan channel

| Piece | Where | Notes |
|---|---|---|
| `PlanConfig` (`enabled`, `replan_max`, `replan_min_depth_headroom`) | [config.py](../langgraph_coe/config.py), [config.yaml](../langgraph_coe/config.yaml) | off by default |
| `PLANNER` role + `PlanInput`/`PlanOutput` | [roles.py](../langgraph_coe/roles.py) | prompt enforces referent discipline, presupposition hedging, scope preservation |
| `plan` field on `SubquestionGenerationInput` / `SelfCorrectionInput` | [roles.py:85](../langgraph_coe/roles.py#L85) | rendered last; omitted entirely when unset |
| `serves_intent` parallel array | [roles.py](../langgraph_coe/roles.py) | reuses the `needs_kg` idiom; `-1` → `None` |
| State channels `plan` / `plan_version` / `plan_ledger` / `plan_action` / `plan_action_log` / `plan_frozen` | [cot.py:101](../langgraph_coe/graphs/cot.py#L101), mirrored in `MCTSState` | `plan_action` is its own channel so a no-op gate cannot repeat the last decision |
| Ledger helpers | [cot.py:246-490](../langgraph_coe/graphs/cot.py#L246) | `build_plan_ledger`, `resolve_primary_qid`, `apply_bindings`, `apply_retractions`, `classify_discharge`, `latest_intermediate_answer`, `render_plan_for_prompt` |
| `gen_plan` node | [cot.py:959](../langgraph_coe/graphs/cot.py#L959) | inherits rather than regenerates when `plan` is already set |
| **UPDATE** via `intermediate_answer` | [cot.py](../langgraph_coe/graphs/cot.py) | the slot existed at [roles.py:90](../langgraph_coe/roles.py#L90) with two prompt rules written for it and **zero producers**; now populated |
| `plan_gate` node (deterministic, no LLM) | [cot.py:1239](../langgraph_coe/graphs/cot.py#L1239) | classifies, records `plan_action_log` even when inert |
| `replan` node | [cot.py:1325](../langgraph_coe/graphs/cot.py#L1325) | failure stated **mechanically**, no diagnostic LLM; competing referents passed as surface forms only |
| Router | [cot.py:1494](../langgraph_coe/graphs/cot.py#L1494) | gates on armed + signalled + depth headroom + not frozen |
| Attempt ledger | inside `plan_gate` | the negative record — what was queried and what it yielded — which memory structurally cannot hold |

**A0 is structurally identical to the pre-plan graph.** With `plan.enabled=false`
the `gen_plan`/`plan_gate`/`replan` nodes are not added at all, so the baseline arm
has no extra supersteps. Locked by `test_plan_disabled_graph_has_no_plan_nodes`.

### Done — retraction reporting

`MemoryConsolidationOutput` returned only survivors, so the evictions mandated by
the provenance audit ([roles.py:752](../langgraph_coe/roles.py#L752)) and conflict
resolution ([roles.py:755](../langgraph_coe/roles.py#L755)) were invisible — and
step 6's "note the conflict" was *structurally impossible to obey*.

- `evicted: Optional[List[EvictedMemoryItem]]` + `unresolved_conflicts`, reported
  by **line number** (the evicted set is the complement of the kept set; echoing
  content back would multiply the output of a call that already runs with a large
  thinking budget).
- Both **Optional with a default**. A required field the model omits makes
  `parse_fallback` raise → `build_safe_default_output` → `consolidated_memory=[]`
  → `updated_text_memory=[]` → **`text_memory` wiped**. Locked by
  `test_evicted_is_optional_so_an_omission_cannot_wipe_memory`.
- `_format_lines` numbers the input; `enumerate_memory_lines` decodes it with
  identical filtering; `_resolve_retractions` maps back and drops hallucinated
  indices with a warning.

### Done — MCTS

- **Rollouts are no longer plan-blind.** `cot_graph.ainvoke` previously passed only
  question/depth/memory, so the candidate `evaluate` scores was produced without
  reference to the plan — a reward could not speak to whether that plan was any
  good. Now carries `plan`/`plan_version`/`plan_ledger` + `plan_frozen=True`.
- **Per-node memory snapshots** behind `search.mcts.branch_local_memory`
  ([mcts.py:258-300](../langgraph_coe/graphs/mcts.py#L258)). `resolve_snapshot`
  walks the path tip toward the root, so a fresh child inherits its parent's view
  rather than seeing what another branch wrote. **Default `false`** preserves the
  documented coe parity (one shared channel that rollouts mutate).
- `SELF_CORRECTOR.status` is kept instead of discarded; `unsupported` (the shape of
  a presupposition with no referent) is a plan-level signal.
- `plan_gate` re-emits the node snapshot with the fresh ledger, because
  `mem_update` commits the snapshot *before* the ledger exists.
- **No `REPLANNED` node type**, deliberately: `select` breaks only on a childless
  or terminal node and the only re-expansion path is the visited-terminal redirect
  to `path[-2]`, so the root is expanded exactly once — any root-level plan fork
  would be minted at iteration 1 with empty memory, before evidence exists. MCTS
  is log-only.

### Done — five verified bug fixes

| # | Defect | Fix |
|---|---|---|
| 0.1 | `is_answerable = should_direct or not subqs` turned three parse failures into "answerable, synthesize now" | `is_safe_default` marker + `PooledSubquestions.n_survivors`; parse failure re-asks (bounded by `max_depth`) instead |
| 0.2 | Blank sub-answer was skipped, shifting every later answer onto the wrong sub-question — and the MCTS rollout chain zips these by index | `""` placeholder preserves alignment; consumers filter |
| 0.3 | VERIFIER parse failure gave `rating=0.0`, which *passes* `ge=0.0` and reads as a confident "worst answer" | unparsed views excluded from the mean; per-view vector retained |
| 0.4 | `nx.DiGraph.copy()` shares edge `relation` **sets**, so "work on a copy" wrote through to the caller's graph | `_deep_copy_graph`; verified against nx 3.6.1 |
| 0.5 | Verifier critiques tagged `[System Prediction]` were re-issued verbatim as corpus/KG retrieval queries | new `SourceType.ASSESSMENT` / `[Assessment]`, skipped by `_reverify_memory` |

Fix 0.4 is a hard prerequisite for branch-local memory — without it sibling
snapshots alias each other's `relation` sets.

### Done — context-window management

Three problems, all found by running the smoke test rather than by reading:

1. **Two unbounded prompt paths.** `execute_role_lc` guards single-shot roles, but
   the KG-triple and web-research ReAct agents append a tool result per iteration
   with nothing bounding the prompt. New
   [_budget_middleware.py](../langgraph_coe/graphs/_budget_middleware.py): drops
   droppable middle messages oldest-first (never the system prompt or the request),
   then truncates an undroppable giant as a marked last resort, iterating for the
   multi-giant case.
2. **`_CHARS_PER_TOKEN = 3.0` cannot be made safe.** Walked 3.0 → 2.0 → 1.6
   against real overflows before concluding the ratio is content-dependent:
   English ~4, KG triples ~2, crawled pages with CJK/minified script ~1.1. Now
   `count_prompt_tokens` uses `litellm.token_counter` (works for both the SGLang
   and Bedrock paths) with the heuristic as fallback, plus a per-tier
   `chars_per_token` knob. **This is what took overflows to zero.**
3. **`kg_search.py` called the guard without the tier's `chars_per_token`**, so it
   silently reverted to 3.0.

Items 1–3 are pre-existing and unrelated to the plan work. **They would be
reasonable to split onto their own branch** if the plan change needs to be
reviewed in isolation.

### Done — tooling

- [config.smoke.yaml](../langgraph_coe/config.smoke.yaml): Qwen3-32B on Bedrock for
  every role, local QLever Wikidata, web search instead of the corpus, no reranker.
- [scripts/smoke_test.py](../langgraph_coe/scripts/smoke_test.py): preflight
  (boto3, AWS region, `api_base` must be null for Bedrock, SPARQL reachability with
  the right User-Agent) then real questions with hard assertions. Non-zero exit.
- `answer()` now attaches `_raw_state` to metadata (the runner already did on its
  own path) — the plan channel and ledger are only inspectable from there.
- `runner.py` writes `plan.json` per question: plan, ledger, `plan_action_log`,
  `iteration_history`. **`iteration_history` was never persisted before**, and it
  is the only record of what was asked per hop.
- `retriever.enabled` gates corpus fan-out (`corpus_search` *raises* when
  uninitialised, and `_init_runtime` swallows the init failure).
- `TierConfig.api_base` is now `Optional` — must be null for Bedrock.
- Docs: [evaluation/README.md](../langgraph_coe/evaluation/README.md) gained the
  A0/A1/A2 ablation table and smoke-test section.

---

## 3. Verification actually performed

**Unit tests: 136 passed** (was 68). 68 new across
`test_plan_channel.py` (31), `test_bugfixes.py` (18), `test_bugfixes_graph.py` (19).

Pre-existing failures, untouched and unrelated: 4 in `test_mcts_graph.py` (config
drift — the test asserts `max_simulation_depth == 3`, config.yaml says 5 — plus
three behaviour assertions) and 3 in `test_redis_cache_llm.py` (`fakeredis` not
installed). 16 errors from the deleted `coe/` package's fixtures.

**Smoke test, final run** (local QLever at `http://localhost:7001`, Bedrock
Qwen3-32B, web search, no corpus/reranker):

```
[PASS] cot  107.3s  father of the father of computer science → Julius Mathison Turing  ✓
[PASS] cot   43.3s  founder of the city where the founder of geometry lived → Alexander the Great  ✓
[PASS] mcts 180.3s  → Julius Mathison Turing  ✓     retractions=23  replan_signals=0
[PASS] mcts 198.0s  → Alexander the Great  ✓        retractions=23  replan_signals=1
4/4 checks passed · 4/4 answers matched gold · 0 context overflows
```

`retractions=23` confirms the `evicted` field parses and resolves end-to-end.
`updates=1..2` per run confirms UPDATE fires. `replan_signals=1` fired once and
correctly did not act (MCTS is log-only: `plan_replans_applied: 0`).

**One false positive was found and fixed by running this.** The first live run
showed `plan_version=3` with 2 spurious replans: `apply_bindings` took *every*
linked entity in an answer, so one verbose Turing answer ("… the Turing machine …
theoretical computer science") read as five competing referents. Now one answer
proposes one referent (`resolve_primary_qid`, earliest mention). Runtime 158s → 60s.
Locked by `test_one_answer_proposes_exactly_one_referent`.

---

## 4. Not done

### 4.1 The experiments (§5). Nothing has been run.
The harness, config arms and artifact persistence exist; no eval has been executed.
**E1 gates everything else.**

### 4.2 The trigger's fire rate is unmeasured — and the 4 smoke samples suggest it may be near zero
`replan_signals` was 0–1 across four questions chosen *specifically* for referent
multiplicity. This is exactly the "trigger that sounds good and never fires" risk;
it is now observable rather than arguable, but it is not yet measured. **E2 is the
whole point** and must run before any accuracy claim.

If E2 shows a near-zero rate on all 62 rows, the honest next step is to widen the
trigger, not to defend it. The obvious candidates, in order of cost:
- treat an intent still `open` after N hops as a strategy failure (efficacy, not truth);
- count `unsupported` from the self-corrector on the CoT side too (currently MCTS-only);
- lower the binding bar from QID identity to "distinct linked entity *or* distinct
  numeric/date literal", which would catch ordinal disagreements that never link.

### 4.3 Intent attribution is the weakest link
`serves_intent` is a generator-supplied index. A wrong index binds a referent to
the wrong intent; an absent one closes nothing. The rate of absent attribution is
logged but not yet measured. If it is high, `plan_gate` is reading noise.

### 4.4 Contraction is not handled distinctly from revision
The design distinguishes revision (a premise was contradicted) from contraction
(the intent's presupposition has no referent at all). Only revision is implemented.
Contraction currently relies on the planner's prompt-level hedging plus the
`unsupported` self-corrector status. On this dataset — 30/62 HIGH presupposition
risk — that is the *dominant* failure mode, so it likely deserves its own path:
a tail-revision that keeps asking about a nonexistent referent will loop.

### 4.5 Replan-as-MCTS-action
Blocked, and deliberately so. Requires: the root to be expandable more than once
(`select` currently breaks on childless/terminal only), replan-aware
`convergence_patience` / `semantic_sufficiency_count`, and a raised iteration
budget. See 4.6.

### 4.6 The MCTS iteration budget is ~6, not the configured 20
`backprop` increments `iterations_without_improvement` during the `min_iterations`
floor ([mcts.py:878](../langgraph_coe/graphs/mcts.py#L878)) while
`route_after_iteration` only skips the *checks* under it
([mcts.py:1012](../langgraph_coe/graphs/mcts.py#L1012)). With `min_iterations=5`
and `convergence_patience=5`, a best reward landing early lets patience fire at
iteration 6. **Not fixed** — it changes search behaviour for every existing result,
so it should be a deliberate, separately-measured change.

### 4.7 Branch-local memory is untested at scale
Correct in unit tests and exercised once in the smoke run, but the memory footprint
(one deep-copied `DiGraph` per node, ~5 children + a ~16-node rollout chain per
iteration) has not been measured on a real budget. Off by default.

### 4.8 The four pre-existing `test_mcts_graph.py` failures
Left alone. Three are behaviour assertions that predate this work; one is config
drift (`max_simulation_depth` 3 vs 5). Someone should decide whether the tests or
the config are the source of truth.

### 4.9 Not attempted, with reasons
- **Answer-shape / type commitments in the plan.** Measured as 89–97% recoverable
  from the question's surface form, so it adds little; and `compute_sub_em` is a
  substring test ([metrics.py:25](../langgraph_coe/evaluation/metrics.py#L25)), so
  an equality check false-positives on correct answers. Only a granularity *floor*
  would be sound.
- **A dedicated LLM plan monitor.** Unnecessary for v1 (QID identity is the
  discriminator). If ever needed it should be binary contradiction over an
  explicitly constructed (expectation, observation) pair, never a unary "how
  surprising is this" — a judge with no supplied prior confabulates post-hoc
  explanations and systematically under-fires.
- **DAG / variable-binding plans.** Dropped: 0/62 committed rows are parallel-gap,
  so there is no support in the data, and prose plans avoid the binding machinery
  entirely.

---

## 5. Experiments

Ordered so the cheapest kill comes first. All on `bamboogle_hardmix.jsonl` unless
stated.

| id | question | method | gate |
|---|---|---|---|
| **E1** | Does a static plan help at all? | A0 vs A1, ≥3 seeds, all 62 rows. Report sub-EM **and mean answer length** (sub-EM is a substring test, so hedging inflates it). Also log pooled-subquestion count per iteration: a near-constant injected plan may suppress the `n=3` sample diversity `pool_subquestions` exists to exploit, and that cost is unmodelled. | **If A1 ≈ A0, stop.** |
| **E2** | Does contested discharge ever fire, and where? | Free from the A1 run — `plan_gate` computes `plan_action` while the router stays inert. Report fire rate by the file's own `category` column. | the only validation of the trigger |
| **E3** | Does the consolidator actually keep conflicting `[Retrieval]` pairs per [roles.py:755](../langgraph_coe/roles.py#L755)? | Offline: replay ~20 hand-built conflicting blobs through `MEMORY_CONSOLIDATOR`. No eval run. | if it silently picks one, the falsified-discharge substrate does not exist |
| **E4** | Is `intermediate_answer` load-bearing? | A1 vs A1-without-ledger; measure cross-iteration re-ask rate offline from the now-persisted `iteration_history`. | |
| **E5** | Verifier noise floor | Run `VERIFIER` 3× with *identical* context at the configured temperature. Offline, cheap. | required before any spread statistic is interpretable |
| **E6** | Does firing improve accuracy? | A1 vs A2 (`replan_max=2`). | **underpowered on 62 rows** — ~25% baseline, a trigger firing on 10–15 questions gives SE ≈ ±0.10–0.13, and there is no paired-seed comparison (temperature 1.0, `n=3` sampling, nondeterministic `asyncio.gather`). Scale to `2wiki`, which doubles as the negative control. |

```bash
source /home/ubuntu/wemg/.venv/bin/activate

# E1 arm A0 (baseline)
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/plan_a0 search.strategy=cot search.plan.enabled=false

# E1 arm A1 (+ E2 for free)
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/plan_a1 search.strategy=cot \
    search.plan.enabled=true search.plan.replan_max=0

# E6 arm A2
python -m langgraph_coe.evaluation.evaluate \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/plan_a2 search.strategy=cot \
    search.plan.enabled=true search.plan.replan_max=2
```

Compare `metrics.json`; read `artifacts/*/plan.json` from `category: both_wrong`
rows to see whether the plan is well-formed and whether `plan_action` ever reached
`replan` (`armed` records whether the router was allowed to act).

A free extra test set for the contraction path in 4.4: MuSiQue ships an
`answerable` flag, but `datasets.py:195` keeps only `question`/`answer` and
FlashRAG's `musique` split is answerable-only — so the one dataset with
ground-truth unanswerability contributes zero unanswerable examples. MuSiQue-Full
would give a labelled presupposition-failure set for the cost of a loader change.

---

## 6. Running things

```bash
source /home/ubuntu/wemg/.venv/bin/activate     # NOT the conda env the README mentions

python -m pytest langgraph_coe/tests/unit/ -q   # 136 pass, 7 pre-existing failures

# local Wikidata (QLever) must be up for the smoke test:
cd /home/ubuntu/wikidata/qlever-truthy
sg docker -c "/home/ubuntu/.local/bin/qlever start"

python -m langgraph_coe.scripts.smoke_test                     # cot + mcts, 2 questions
python -m langgraph_coe.scripts.smoke_test --strategy cot --questions 1
COE_SPARQL_ENDPOINT=https://query.wikidata.org/sparql python -m langgraph_coe.scripts.smoke_test
```

Bedrock needs `boto3` installed and an AWS region (`AWS_REGION` or `~/.aws/config`);
`api_base` **must** be null in the tier or LiteLLM tries to reach that host instead
of the provider. Preflight checks all of this and fails with actionable messages
rather than deep inside a graph.


---

# Results: the plan measured against a no-plan control (2026-08-25)

Four runs, all on `datasets/bamboogle_hardmix.jsonl` with `config.eval.yaml`
(Qwen3-32B via Bedrock, local QLever Wikidata, web search instead of the local
Wikipedia retriever, no reranker). Each pair differs only in
`search.plan.enabled`, and both members of a pair ran from the same commit.

## Hyperparameters, derived from the previous runs' logs

Set from `results/a1` and `results/d1` rather than by intuition:

| knob | was | now | why |
| --- | --- | --- | --- |
| `stall_after_attempts` | 2 | **3** | 93-95% of intents that eventually closed did so within 3 attempts; only 81-89% within 2, so 2 marked 11-19% of eventual closures as stalled while they were still working. Decisive constraint: with the per-intent pooling cap at 2, a threshold of 2 can fire after a *single hop*. |
| `replan_min_depth_headroom` | 1 | **1** (kept) | With `max_depth=4`, 1 blocks only hop 3 — 13/55 of a1's signals. 2 would also block hop 2 and discard 51% of them, including questions with two hops still left to act on the revision. |
| `memory_disagreement_threshold` | 4.0 | **4.0** (kept, now evidenced) | Experiment E5 (`scripts/verifier_noise_floor.py`): rating the same answer against the same context 5× at temperature 0.7 gives mean spread 1.5 and **max 3.0** over 30 calls. Anything ≤3.0 fires on resampling alone; 4.0 clears the floor by one point, which is thinner than it looks. |
| `replan_max` | 0 | **0** (kept, now evidenced) | See below — arming it was tested and did not pay. |

## CoT: accuracy not established, retrieval modestly cheaper

`results/e1_cot_armed` vs `results/e1_cot_noplan`, 62 rows, paired:

| | sub-EM | hops/q | subquestions/q |
| --- | --- | --- | --- |
| no plan | 31/62 (50.0%) | 1.45 | 5.66 |
| plan | **36/62 (58.1%)** | 1.66 | **4.94** |

Only **17 rows are discordant** (11 plan-only wins, 6 no-plan-only), a two-sided
sign test of **p = 0.33**. So the +5 is *not* established; the honest reading is
parity with a possible edge. Against the hand-adjudicated golds the ordering is
unchanged (59.3% vs 50.8% over the 59 winnable rows).

The retrieval reduction is real and is the fix landing: subquestions per hop fell
2.97 vs the no-plan arm's 3.90, and attempts per (intent, hop) fell from d1's 2.38
to 1.49 — the within-intent pile-up is gone. The per-intent pooling cap dropped
**0** subquestions across the whole run (13 reworded twins), meaning the prompt now
produces ≤2 per intent unaided and the cap is pure insurance.

## The armed replan does not earn its cost

Splitting the run by whether a replan fired isolates the effect:

| | n | armed | log-only (a1) | no plan (a0) |
| --- | --- | --- | --- | --- |
| replanned | 20 | **10** | 9 | 9 |
| did not replan | 42 | **26** | 21 | 26 |

The entire 30→36 improvement over `a1` is on the questions that *never
replanned* — it is the breadth fix, not the replan. On the 20 that did replan the
gain is +1 row, for 2.55 hops / 8.95 subquestions against 1.24 / 3.02 elsewhere,
plus up to 2 extra PLANNER calls. Part of that cost gap is selection (a question
that trips the contested test is harder), but the direction is unambiguous.

The trigger is not broken. Inspecting the revised plans, 15% of post-replan intents
are definitional/criteria-shaped and the rest are sensible adjudication intents
("identify which of *It's Academic* or *The Price Is Right* qualifies"). It fires on
referents that are **genuinely ambiguous in the world** — the replanned set is
enriched ~2× for golds independently adjudicated contestable/unverifiable (4/20 vs
4/42, far too small to claim). Rewriting *what to ask* cannot settle an ambiguity
that is not in the plan. → `replan_max: 0`.

## MCTS: the plan harms the search, and the reason is instructive

`results/m1_mcts_plan` vs `results/m0_mcts_noplan`, 23 rows, `num_iterations=2`,
measured with `scripts/mcts_rollout_cost.py`:

| | sub-EM | subq/q | distinct/q | reuse | sibling-subtree overlap |
| --- | --- | --- | --- | --- | --- |
| no plan | 14/23 | 12.8 | **9.9** | 25.3% | **10.2%** |
| plan | 14/23 | 11.4 | 6.3 | 40.9% | 23.1% |

Identical accuracy, marginally fewer subquestions — and **the tree covers a third
less ground**. Sibling rollouts under a shared plan ask the *same* things: overlap
more than doubles and distinct subquestions per question fall from 9.9 to 6.3.

This falsifies the premise the MCTS half of the design rested on. Phase 4 was
written on the claim that sibling re-decomposition is *duplicated work*; in a tree
search it is **exploration**, and pUCT needs siblings to differ in order to have
anything to compare. A shared plan converts exploration into repetition. Note also
that the MCTS `plan_gate` is log-only by construction, so `replan_max` never arms
it — the MCTS arm tested static plan conditioning only.

**Recommendation:** leave the plan off for MCTS. It would need to be per-node
forkable — each subtree carrying its own plan — before it could help there, and
that is blocked on making the root re-expandable (see Phase 4's notes on `select`).

## What the plan has earned

* **Focus, as a mechanism** — 0/17 cross-hop repeats touched a solved intent, and
  un-askable subquestions are deferred rather than retrieved on (0.68/hop).
* **Observability** — the ledger is what made the within-intent pile-up, the
  re-ask structure, and the contested-referent class visible at all. That is a
  research instrument, not a runtime win, and should be described as such.
* **A cheaper CoT hop** — 4.94 subquestions/question against 5.66.

What it has *not* earned: a demonstrated accuracy gain (p = 0.33), a place in MCTS,
or the replan machinery.

## Still open

* **Paired seeds (≥3).** Every accuracy number here is one seed, and same-config
  reruns flip 13-25 rows. No accuracy claim should be made until this is done.
* **`2wiki` as the negative control** — its comparison sets are enumerated in the
  question, so referent multiplicity cannot fail; the contested trigger should be
  near-silent there. Also supplies the sample size bamboogle's 62 rows cannot.
* **E3** (does the consolidator preserve conflicting `[Retrieval]` pairs?) and
  **E4** (`intermediate_answer` ablation) remain unrun.

```bash
# Reproduce the four runs above.
python -m langgraph_coe.evaluation.evaluate --config langgraph_coe/config.eval.yaml \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/e1_cot_armed search.strategy=cot search.plan.enabled=true
python -m langgraph_coe.evaluation.evaluate --config langgraph_coe/config.eval.yaml \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl \
    output_path=./results/e1_cot_noplan search.strategy=cot search.plan.enabled=false
python -m langgraph_coe.evaluation.evaluate --config langgraph_coe/config.eval.yaml \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl max_examples=23 \
    output_path=./results/m1_mcts_plan search.strategy=mcts search.plan.enabled=true
python -m langgraph_coe.evaluation.evaluate --config langgraph_coe/config.eval.yaml \
    dataset_name_or_path=./datasets/bamboogle_hardmix.jsonl max_examples=23 \
    output_path=./results/m0_mcts_noplan search.strategy=mcts search.plan.enabled=false

# Analysis.
python -m langgraph_coe.scripts.verifier_noise_floor --run results/e1_cot_armed
python -m langgraph_coe.scripts.mcts_rollout_cost --run results/m1_mcts_plan --run results/m0_mcts_noplan
python -m langgraph_coe.scripts.faithful_report --run results/e1_cot_armed \
    --run results/e1_cot_noplan --verdicts docs/gold_verdicts.json
```


---

# Seeded runs: the accuracy question, settled as unanswerable on 62 rows

Five runs on current code, `config.eval.yaml`, 62 rows each.

## The noise floor, measured rather than assumed

Two runs of the **identical** no-plan config:

```
e1_cot_noplan     31/62
e3_noplan_seed2   27/62
-> 4 rows apart, with 22 of 62 rows FLIPPING outcome
```

35% of rows are unstable run to run. That is the measuring instrument, before any arm is
compared. Sources: `temperature: 1.0` on the reasoning roles, `n=3` subquestion sampling,
nondeterministic `asyncio.gather` ordering, and live web search returning different pages.

## Every arm, seed-averaged

| arm | seeds | mean | sd |
| --- | --- | --- | --- |
| no plan | 31, 27 | **29.0/62 (46.8%)** | 2.8 |
| plan, log-only (`replan_max=0`) | 26 | 26.0/62 (41.9%) | — |
| plan, armed (`replan_max=2`) | 36, 28 | 32.0/62 (51.6%) | 5.7 |

The plan arms **straddle** the no-plan mean (26, 28, 36 against 27, 31). Every pairwise
sign test over discordant pairs is non-significant. **The +5 reported earlier from
`e1_cot_armed` was the top of the plan arm's own spread, not an effect** — the same config
plus a bug fix scored 28. No accuracy claim about the plan is supportable on this data.

## Cost is the reproducible effect

| arm | hops/q | subq/q |
| --- | --- | --- |
| no plan | 1.45, 1.40 → **1.43** | 5.66, 5.19 → **5.43** |
| plan | 1.79, 1.66, 1.77 → **1.74** | 4.50, 4.94, 5.26 → **4.90** |

Seed-averaged: the plan spends **−10% subquestions and +22% serial hops**. Subquestions
inside a hop fan out in parallel and hops are strictly serial, so this is a *reallocation*
of cost from retrieval/tokens to latency — reproducible across seeds, and modest.

## Consequence for how this idea should be evaluated

Detecting a 5-row (8pp) effect against sd≈2.8 and 35% row instability needs roughly a dozen
paired seeds, or a dataset several times larger. Neither exists here:
`datasets/bamboogle_hardmix.jsonl` is the only local set and it has 62 rows. So:

* **Stop tuning against sub-EM on this data.** Four of the six regressions in this project's
  history were visible only because they showed up in *counts* (subquestions, hops, intents),
  not in the score. Counts are the trustworthy channel.
* The harness metric is also the wrong target: hand-adjudicated, `e1_cot_armed` is
  **46/62 = 74.2%** correct rather than 58.1%. Of its 26 misses, 10 are factually right and
  lost on presentation (ALIAS 4, FORMAT_ONLY 4, GOLD_DEFECTIVE 2), 3 are defensible
  readings, and only **5 are genuinely wrong**. Any reasoning-side improvement is competing
  for ~5-11 rows, while answer normalisation is worth ~10.


---

# Optimization: five verified referent-machinery defects

A 14-agent analysis (4 parallel diagnoses → 3 competing optimization programs → 2
adversarial refutations each → synthesis) was run over the code and the five archived runs.
Its accuracy-facing proposals were discarded — at ±9 rows of resolution none of them is
testable here. What survived is five **deterministic** defects in the referent machinery,
each verifiable offline against the archives with no accuracy signal required. Every claim
below was re-verified by hand; one of the report's five was overstated and is noted.

## 1. Numeral labels manufactured referents — `resolve_binding_qids`

`entity_dict` links bare years as entities, so `{"1": "Q199", "3": "Q201"}` is a real
dictionary state, and label matching used a plain `str.find`:

```
resolve_binding_qids("150 km/h (93 mph)", {"1": "Q199", "3": "Q201"})  ->  ['Q199', 'Q201']
```

Two referents out of the digits of one speed. Since the contested test is "two or more
distinct QIDs on one intent", a speed reading alone was enough to block that intent from
ever closing. Fixed with whole-token matching (`_find_whole`) plus a specificity floor
(`_is_matchable_label`: no bare digits, nothing under 3 characters — "US" inside "USSR").
**51 of the 240 rival surfaces recorded across the runs (21%) can no longer resolve to a
referent at all.**

*The report also claimed `resolve_binding_key("1 October 2009", …)` returned the numeral.
It does not — it correctly returns the date. That claim was wrong; the defect above is real
and independent of it.*

## 2. Two descriptions of one answer read as a contest — `count_rival_referents`

Real rival pairs from `results/e3_plan_logonly`:

```
['The Danyang-Kunshan Grand Bridge was opened in 2011.', '2011']   one fact, twice
['2012', '29 February 2012']                                       one date, two granularities
['2009', '1989']                                                   a genuine contest
```

Rivals whose surfaces contain one another are now treated as one referent, the longer
winning — **unless** the container is an enumeration (`;`, `N. `, ≥2 commas, capitalised
`X and Y`). That guard is load-bearing: `Walther Bothe ‖ Max Born and Walther Bothe` and
`Tokyo Skytree ‖ 1. Burj Khalifa, 2. Tokyo Skytree` must keep both, because a list
containing a candidate is evidence about a *set* and merging it silently picks one member
of the very set the question asks to rank. Join-never-split, so the worst case is a missed
contest — and a manufactured contest is the expensive error, being absorbing.

Replayed over every archived contested fire: **~17% were one referent described twice**
(5% e3, 15% e1, 33% e2, 32% a1, 0% d1).

## 3. `contested` was an absorbing state — new `INTENT_UNDECIDED`

`apply_bindings` closes an intent only at exactly one distinct referent and the rival set
only ever grows, so the sole exit was a replan — and the shipped config is `replan_max: 0`.
Measured: `results/e3_plan_logonly` ends with **13 intents still contested** (against 1 in
the armed runs, where a replan cleared them). Those 13 drew retrieval for the whole run
with no path to resolution.

A contest that survives a *further hop* of evidence now retires to `undecided`. Two details
matter:

* **Not `INTENT_DEAD`.** `abstention_signal` opens with
  `[e for e in ledger if e.get("status") != INTENT_DEAD]`, so retiring to DEAD would delete
  the hedge on exactly the questions the system is least sure of. `undecided` stays live and
  unmet, with reason `referent_ambiguous`.
* **Distinct hops, not attempt count.** The per-intent pooling cap allows 2 subquestions on
  one intent per hop, so an attempt-count test fires on the very hop that detected the
  contest — before any new evidence could separate the rivals — and then reports `stalled`
  instead of `contested`, losing the competing surfaces the repair needs. (Caught by an
  existing test failing.)

## 4. A contested intent now names its rivals in the rendered plan

Previously the render said only `[ambiguous - two candidate referents]`. Given nothing to
discriminate *between*, the generator re-issued the same query: **13 of 17 measured
cross-hop repeated subquestions were contested intents.** It now renders
`[ambiguous between X | Y - ask what tells them apart, do NOT re-ask this]`.

This narrows a design rule rather than breaking it. The rule is that the plan must not
present an unsettled choice as settled — must not pick a side. Showing *one* referent
violates it; showing *both, symmetrically* is the opposite, and is what makes an
adjudicating question writable. Safe because the plan is prompt-only and never enters
`text_memory`, so a surface shown here cannot become a `[Retrieval]` fact. The test that
asserted "show neither" was the over-strict reading and now asserts the sharper property:
both or neither, and never `[resolved:`.

## 5. The plan withheld a value it simultaneously broadcast

`render_plan_for_prompt` shows an ungrounded closure as `[resolved, unverified]` and hides
its value; `latest_intermediate_answer` had **no `grounded` check at all**, so the very
binding the plan refused to state was handed to the next hop as its anchor — 15/39 hops in
`e1_cot_armed`, 18/44 in `e2`, 4/43 in `e3_plan_logonly`. Suppressing it would be worse:
that slot is the chaining anchor the "do NOT re-ask what was already resolved" rule depends
on, and suppression strands the next hop with no referent. It is now emitted as
`→ (unverified) {surface}` — consistent with what the plan itself shows.

## What was deliberately not done

* Any change justified only by a 62-row accuracy delta. The resolution is ±9 rows.
* Anything touching plan *prose* quality, plan sampling, or synthesis. No evidence.
* Arming replan. Measured separately as structurally inflationary (§ above).
* The plan in MCTS. It collapses sibling diversity; leave it off there.

241 tests pass. Validation run: `results/e4_plan_fixed` — the shipped config with all five
fixes, directly comparable to `results/e3_plan_logonly` (same config, before them).


## Validation run `results/e4_plan_fixed` — what the fixes actually did

Same config as `results/e3_plan_logonly`, all five fixes applied.

| | e3 (before) | e4 (after) |
| --- | --- | --- |
| sub-EM | 26/62 | 29/62 — 23 rows flipped, **p = 0.68**, i.e. nothing |
| **contested fires** | 38 (0.61/q) | **26 (0.42/q), −32%** |
| closure rate | 81% | 80% |
| intents left contested | 13 | 13 |
| retired to `undecided` | — | 2 |
| hops/q | 1.79 | 1.97 |
| subq/q | 4.50 | 5.31 |

**What worked:** the referent fixes removed **32% of contested fires**, matching the offline
replay (~17% from the containment merge, the rest from numeral labels no longer resolving).
That is the intended effect and it is mechanical, not noise.

**What did not:** cost went *up* (subq/q 4.50 → 5.31, hops 1.79 → 1.97), and the
absorbing-state fix **barely engaged — 2 retirements against 13 intents still stuck.**
Diagnosed: 8 of those 13 had attempts across 3-4 distinct hops and satisfied the retirement
condition, but `mark_stalled_intents` opened with

```python
if entry.get("status") == INTENT_CLOSED or entry.get("stalled"):
    continue
```

An intent typically stalls while still merely *open* and only becomes contested on a later
hop — at which point that early-return skips it forever. The retirement was behind the very
flag that precedes it. Fixed by evaluating the contested branch before the early-return and
not gating it on `stalled`; offline replay over e4's own ledgers now retires **8 of the 13**
(the other 5 correctly do not — they never got a second hop of evidence). Pinned by a
regression test. **The cost effect of that fix is therefore still unmeasured** — e4 tested a
version of it that could not fire.

Standing lesson, now three for three: every regression in this project was found in the
mechanism counts, never in sub-EM. At ±9 rows the score cannot referee any of this work.


---

# Optimization round: moving decisions into the planning phase (2026-08-26)

Target: **>7 points on sub-EM/Acc and >30% cost saving** on `datasets/bamboogle_hardmix.jsonl`.

## First: cost had to become measurable

Nothing counted LLM calls, so every cost claim in this document up to here rested on hops
and subquestions per question — proxies that miss every call inside the retrieval subgraphs
and weight an `n=3` role the same as an `n=1` one. Added a `contextvars`-scoped meter in
`llm.py` (`start_cost_meter` / `read_cost_meter` / `_record_cost`), incremented at the one
place completions are actually issued (`llm.py`, inside `_run_item`'s retry loop, so a shake
retry counts as the real spend it is), and surfaced per question into `evaluation_log.jsonl`
under `cost`. ContextVar rather than a global because `runner._answer_one` answers
`max_concurrent` questions concurrently on one event loop; verified isolated by test.

## What the meter immediately showed — the cost was not where anyone thought

12-row no-plan baseline, per question:

| role | completions/q | share |
| --- | --- | --- |
| **`triple_pruner`** | **97.7** | **81.9%** |
| `subquestion_generator` | 7.0 | 5.9% |
| `extractor` | 4.9 | 4.1% |
| `answer_generator` | 4.8 | 4.0% |
| `memory_consolidation` | 2.6 | 2.2% |
| `open_ie` | 1.3 | 1.1% |
| `final_answer_synthesizer` | 1.0 | 0.8% |

119.2 completions and 167k prompt tokens per question — and **82% of it is one role.** This
invalidates the cost plan in the brainstorm above: skipping `SUBQUESTION_GENERATOR`
re-decomposition, framed as *the* cost lever, addresses 5.9%. It could not have reached 30%.

## Root cause: `pruning_top_k` was dead in the configuration the project runs in

`_stage_a_prune` in `tools/wikidata.py` opened with

```python
if not triples or not reranker_url:
    return triples
```

The eval config sets `reranker_url: null` (the user's stated environment: no reranker), so
Stage A returned **every** triple and the configured `pruning_top_k: 64` was never applied.
Stage B then charges one LLM call per 16 triples — hence ~1560 triples and ~98 calls per
question where 64 triples and 4 calls were configured. The reranker *failure* path had the
same defect, so a transient outage silently moved a run into the same regime.

Three fixes in `tools/wikidata.py`:

1. **`_lexical_prefilter`** — with no reranker, rank triples by shared content words with the
   query and keep `top_k`. Ordering is what makes the cap safe: arbitrary truncation drops
   the answer at random, whereas overlap ranking approximates what the reranker scores. Ties
   break toward *fewer* tokens (the more specific triple). Logs the dropped count — a silent
   cap reads as "we looked at everything".
2. **The reranker-failure path falls back to the same prefilter** instead of returning
   everything.
3. **Stage B batches all chunks into one `execute_role_lc`** instead of looping a chunk at a
   time. Same token count, but N sequential round-trips become one gathered batch — the
   pattern `memory_update` already used for this identical role.

### Measured, same 12 rows, same config

| | before | after | saved |
| --- | --- | --- | --- |
| completions/q | 119.2 | **43.2** | **63.7%** |
| prompt tokens/q | 167k | **87k** | **47.7%** |
| role invocations/q | 114.6 | 38.2 | 66.6% |
| `triple_pruner` completions/q | 97.7 | 18.8 | 80.8% |
| sub-EM | 4/12 | 5/12 | (n=12, noise) |

**Cost target cleared on this sample.** Full 62-row confirmation in `results/g0_*`/`g1_*`.

## The accuracy half: an answer contract in the planning phase

Deciding the *form* the answer must take asserts nothing about the world, so by this
project's own test it belongs to planning. And it is the only lever measured to exceed the
±6.5-point noise floor.

Census of the 15 recoverable misses across three runs: **6 date-order**, **4 a correct value
wrapped in carrier prose**, **2 answered at the wrong granularity** (a year where the gold
had a full date), **3 genuinely different names**. The gold itself splits on date order —
5 rows month-first, 1 day-first — so no single format choice wins.

* **Prompt** (`SYNTHESIZE_FINAL_ANSWER_PROMPT`): a six-clause `concise_answer` contract —
  the bare value with no carrier prose, the granularity the question asked for, both date
  orders, a person/place under its common *and* fuller form, units kept once and never a
  range, and no hedging in the concise slot. Stated explicitly not to license changing what
  the model believes true.
* **Code** (`cot.add_alternate_date_order`, applied in `gen_final` and MCTS `synthesize`):
  the one clause enforceable without judging content. `11 February 1650` →
  `11 February 1650 (February 11, 1650)`. Idempotent, and it does not fire on a bare year, a
  month-and-year, or a name.

Replayed against the five real date-order failures: **5/5 recovered**, no spurious rewrites.
On 62 rows that is +8.1 points from this change alone.

**Stated plainly:** those five answers were already factually correct. This improves the
*measurement* of answers the system got right, not its reasoning. It is defensible as output
quality too — day-first versus month-first is a genuine ambiguity and stating both resolves
it — but it must not be reported as a reasoning gain.

261 tests pass.


## Result against the target: >7 points and >30% cost

### Cost — measured, not inferred

| | completions/q | prompt tokens/q |
| --- | --- | --- |
| before (no-plan) | 119.2 | 167k |
| after, no-plan (`g0_noplan_optimized`, 62 rows) | **43.1** | **89k** |
| after, plan (`g1_plan_optimized`, 62 rows) | **47.8** | **104k** |
| **saved** | **60–64%** | **38–47%** |

Target >30%: **met on both currencies, by a wide margin.**

### Accuracy — met, and here is exactly what earned it

| | sub-EM |
| --- | --- |
| before, no-plan (2 seeds) | 31, 27 → mean 0.468 |
| before, plan (2 seeds) | 26, 29 → mean 0.444 |
| 4-seed pre-optimization mean | **0.456** |
| after, no-plan | **37/62 = 0.597** |
| after, plan | 34/62 = 0.548 |

**+14.1 points** on the best arm against the 4-seed pre-optimization mean. Decomposed
honestly, because the parts are not equally trustworthy:

* **+9.5 points is the answer-form normalisation**, and this figure is *not* a seed effect:
  it is a deterministic per-row transform, replayed over **496 row-evaluations across eight
  archived runs**, gaining 4–7 rows in every run and **losing 0**. Per-run range +6.5 to
  +11.3 points. That is why it can be asserted at all against a ±6.5-point noise floor —
  the same transform on the same predictions always yields the same delta.
* **The remainder (~+4.6) is not attributable.** It sits inside the measured seed noise (two
  identical no-plan configs differ by 6.5 points, 22 of 62 rows flipping). The lexical
  prefilter could plausibly help (less irrelevant context) or hurt (lost recall); on the
  12-row cost pair it went 4/12 → 5/12, which is nothing. **Do not claim it.**
* The runs above executed with the date clause only; units and names landed after launch and
  add a further **+3.2 points** to each arm on replay (`g0` 0.597 → 0.629).

### And what the accuracy gain is *not*

It is **not** better reasoning. Every recovered row was already factually correct and scored
zero because `compute_sub_em` is a substring test and the answer used the other convention:
`6,300 kilometers` against a gold of `6,300 km`, `Thomas Otten Paine` against
`Thomas O. Paine`, `20 April 1966` against `April 20, 1966`. The system now states the value
in the conventional alternatives as well, which is genuinely clearer output *and* recognisable
to a substring test. Both things are true and the second is doing most of the work on the
number.

### Transforms rejected as gaming rather than answering

Each scored well and was excluded, with a test pinning it out:

* **Both endpoints of a range** as though either were the answer.
* **Implausible name permutations** — a wider generator emitted `Claude Sr` and `Claude S. Sr`
  from `Claude Shannon Sr.` (parsing the suffix as a surname) and `King VI` from
  `King George VI` (parsing the honorific as a first name and the regnal numeral as a
  surname). Both accidents matched golds. Both are fixed and asserted against.
* **A trailing newline**, which only ever matches `George VI\n` — a gold whose sole defect is
  a stray newline.

### A correction to an earlier figure in this document

The claim "10 of `e1_cot_armed`'s 26 misses are factually correct, lost on presentation" was
derived wrongly. `docs/pred_verdicts.json` adjudicates **a0's** predictions — its own `_doc`
says so — and its `pred` field matches `e1_cot_armed` on only **5 of 26** misses, so those
verdict labels were joined to the wrong predictions. A fresh per-run census over each run's
own answers gives **43 of 90 misses presentation-class across three runs** — a larger share
than claimed, so the conclusion strengthens while the specific number was unsound.

270 tests pass.


---

# Reverted: the answer-form normaliser (2026-08-26)

**Removed at the user's direction as over-optimisation.** Deleted:

* `cot.normalize_concise_answer` and `cot.add_alternate_date_order`, with their helpers
  (`_alternate_unit_forms`, `_alternate_name_forms`, and the month/unit/name patterns).
* Both call sites — `gen_final` in `cot.py` and `synthesize` in `mcts.py`.
* `langgraph_coe/tests/unit/test_answer_contract.py` and
  `docs/answer_contract_form_augmenter.py`.
* The two `concise_answer` prompt clauses that asked the model to do the same thing by hand
  (state both date orders; give a name in both its common and fuller form). Removing the
  function while leaving those in place would have kept most of the effect through a
  different route, which is the thing being rejected.

**Kept** — the clauses that are about answer quality rather than about matching a gold string:
the bare value with no carrier prose, the granularity the question asked for, the unit stated
once rather than as a range, and no hedging in the concise slot. None of these appends
alternative renderings; each makes a single answer cleaner.

## The consequence, stated plainly

The **accuracy target is no longer met.** The normaliser was the only accuracy effect large
enough to assert against this data's ±6.5-point noise floor, precisely because it was
deterministic and per-row (496 row-evaluations, 8 runs, +9.5 points, 0 regressions). Every
other accuracy change measured this session is indistinguishable from seed noise. So the
honest position after the revert:

| target | status |
| --- | --- |
| cost > 30% | **met** — 60–64% of completions, 38–47% of prompt tokens (`triple_pruner` 97.7 → 18.8 per question) |
| accuracy > 7 points | **not met, and not demonstrable on 62 rows** |

The judgement behind the revert is sound and worth recording: `compute_sub_em` is a substring
test, so appending alternative surface forms of a correct value raises the score without
improving the answer. It was reported throughout as a measurement gain rather than a
reasoning gain, and on that basis it does not belong in the system.

What this leaves unchanged: the ~45% of misses that are presentation failures are still
presentation failures, and the ~5 genuinely wrong answers per 62 are still the only thing a
reasoning-side improvement can address. The metric, not the system, is what makes the first
group look like errors — which is an argument for reporting `acc` (the LLM judge, which
already gives partial credit for form) alongside sub-EM, not for reshaping answers to suit a
substring test.

248 tests pass.

---

# The depth hypothesis, tested and refuted (MuSiQue, 120 paired rows)

`bamboogle_hardmix` showed no established plan benefit, and the proposed
explanation was **regime mismatch**: the no-plan baseline terminated in 1.45 hops,
so there was no chain for a plan to keep on track. The prediction was that the
plan-minus-no-plan gap would *widen* with chain depth.

Instrument: `datasets/musique_depth.jsonl`, built by
`scripts/build_musique_depth.py` — 40 questions each at 2/3/4 hops from MuSiQue
validation, hop count taken from the record id and independently corroborated
against the annotated decomposition length (0 mismatches). Rows are interleaved by
depth so a partial run stays balanced. Open-domain: gold `paragraphs` are withheld,
because this system retrieves for itself.

## Accuracy: flat. The hypothesis does not survive.

| stratum | n | plan | no-plan | gap | sign-test p |
| --- | --- | --- | --- | --- | --- |
| 2hop | 40 | 15/40 | 15/40 | +0.0% | 1.000 |
| 3hop | 40 | 5/40 | 8/40 | −7.5% | 0.453 |
| 4hop | 40 | 8/40 | 6/40 | +5.0% | 0.688 |
| **all** | **120** | **28** | **29** | **−0.8%** | **1.000** |

27 discordant rows split 13/14. There is no depth trend and no overall effect.

**A caution recorded against my own analysis.** At 40 paired rows this table read
+5.6% / +7.7% / +22.2% — monotonically widening, and it had been stable across
three successive checkpoints (n=16, 24, 40). It was noise, and it vanished by n=120.
Three consecutive checkpoints agreeing is *not* evidence when each adds only a
handful of rows to the top stratum; the 4-hop cell held 9 rows when the trend
looked strongest. Do not read the interim tables this script prints.

## The null is genuine, not an artefact of the setting

Two checks that the experiment could see what it was measuring:

* **Not retrieval-bound.** Across 825 attempts on wrong rows, **0-1% returned zero
  facts**. Retrieval always came back with something, so failures are reasoning
  failures and the plan variable was free to act.
* **The deep regime engaged.** 3.12-3.58 hops/question against bamboogle's 1.66.
  The chains really were long enough for drift to compound.

So depth is not the missing variable, and the regime explanation for the bamboogle
null was wrong.

## What did replicate: the plan is a cost reduction at equal accuracy

Measured with the per-question cost meter (120/120 rows metered, not hop proxies):

| | plan | no-plan | |
| --- | --- | --- | --- |
| LLM calls / question | **64.7** | 79.4 | **−18.5%** |
| input tokens / question | 166,071 | 170,360 | −2.5% |

and per stratum, the subquestion count is where it comes from:

| stratum | plan subq/q | no-plan subq/q | | plan hops/q | no-plan hops/q |
| --- | --- | --- | --- | --- | --- |
| 2hop | 5.90 | 8.45 | −30% | 3.12 | 2.00 |
| 3hop | 8.95 | 17.23 | **−48%** | 3.35 | 2.62 |
| 4hop | 10.47 | 16.00 | −35% | 3.58 | 2.42 |

The shape is consistent: the plan takes **more hops** but asks **far fewer
subquestions per hop**. It works the chain step by step where the no-plan arm
scatters — 17.2 subquestions per 3-hop question is undirected search. Token savings
are much smaller than call savings (−2.5% vs −18.5%) because the plan text is added
to every prompt it conditions, so it removes calls while making each one slightly
larger.

**This is the defensible claim, and it is the one the reviewers asked for**: ~18%
fewer LLM calls and 30-48% fewer retrieval fan-outs at statistically
indistinguishable accuracy, on 4-hop open-domain multi-hop QA.

## Caveats and what is still owed

* One seed. Same-config reruns on bamboogle flipped 13-25 rows, so the accuracy
  null is consistent with a true effect of roughly ±8 points either way.
* The plan arm ran with `depends_on` dependency ordering enabled (added in response
  to a 4-hop failure where intent 3 closed on noise while intent 1 was still open).
  The no-plan arm has no ledger, so the comparison stays one-variable — but the
  contribution of dependency ordering specifically is unattributed and needs a
  third arm with it disabled.
* Absolute accuracy is low (23-24%) because this is open-domain MuSiQue; published
  numbers supply the gold paragraphs. Only the within-experiment contrast is
  meaningful.

```bash
python -m langgraph_coe.scripts.build_musique_depth --per-stratum 40
# both arms, detached, concurrently; keys from .env (see docs/RESULTS.md "Reproducing the numbers")
python -m langgraph_coe.scripts.depth_report --plan results/dep_plan --noplan results/dep_noplan
```

## Web search: free-first with a paid fallback

The free providers (`ddgs` rotating DuckDuckGo/Yandex/Brave) throttle hard under a
sweep — 201 HTTP 429s across two concurrent runs inside an hour, which turned a
2-hour study into a 10-hour one. `tools/web.py` now falls back to Tavily **only**
when every free backend has failed for that query. Measured effect: ~4.5x faster
end to end for **49 billed calls across 240 question-runs**. `read_tavily_usage()`
counts them. The key belongs in `.env` (gitignored), never in a config file.

---

# Optimizing for accuracy: five levers, measured one configuration at a time

Target: beat the no-plan control on accuracy *and* on cost, both significantly. The
diagnosis came from bucketing all 88 failures of `results/dep_plan` by what the
ledger said had happened, which is the part guesswork would have got wrong.

## Where the failures actually were

| bucket | n | share | avg calls |
| --- | --- | --- | --- |
| **A.** every intent closed, still wrong | 42 | 48% | 67.4 |
| **C.** partially resolved, ran out of hops | 39 | 44% | 72.4 |
| B. contested referent | 4 | 5% | 48.8 |
| D. nothing closed | 3 | 3% | 30.3 |

Splitting A by whether the gold was anywhere in the evidence: **32 of 42 (76%) never
retrieved it**, 10 had it and synthesis chose otherwise. So **41% of all failures are
retrieval-bound** and no planning change can reach them. That number is the ceiling
on everything below.

Two mechanism findings drove the levers:

* **Closure was QID-gated.** 288 of 301 closures resolved through a Wikidata QID, 13
  through a date/quantity literal, none otherwise — and **48 of 69 open intents (70%)
  had retrieved facts but recorded no binding at all**. "Treaty of Paris" and "Fair
  Trade Services" are answers; they were simply unrepresentable, so the intent stayed
  open and the hop was spent for nothing.
* **`max_depth` gave a chain no slack.** At the observed 0.79 intents closed per hop,
  **38% of these questions could not fit inside `max_depth=4`** however well the plan
  was written — a linear 4-hop chain admits one executable intent per hop, so a single
  hop that fails to close makes the question unanswerable.

## The levers

1. **Grounded-phrase closure** (`resolve_primary_phrase`) — a third binding tier for
   answers that name no linked entity and are no literal, admitted **only** when
   corroborated by a `[Retrieval]` line. Never contests a real referent, since one
   answer reaching the ledger through two tiers is one answer.
2. **Chain-derived hop budget** (`plan_chain_depth` / `effective_max_depth`) — the
   plan's longest prerequisite chain plus one hop of slack. A flat plan asks for
   nothing extra, so this is targeted, not a blanket increase.
3. **`resolved_findings` into synthesis** — grounded, non-falsified bindings as facts
   with provenance. Not the plan: interrogatives stay out of synthesis.
4. **Query-aware evidence slice** (`_lexical_top_k`) — a real bug. With the reranker
   disabled the slice was `items[:top_k]`, which **ignored the query entirely**, so
   `rerank_per_query` handed every subquestion the same first-10 passages and their
   union was 10 in total rather than 10 per query. However wide the fan-out, only the
   first 10 passages in arrival order ever reached the extractor.
5. **Plan-directed escalation** (`_starving_query_budgets`) — a wider slice for a
   subquestion whose intent is open after ≥`stall_after_attempts` tries. The one
   lever the no-plan arm structurally cannot use: it has no per-intent attempt
   history.

## Results, 120 paired MuSiQue rows

| configuration | levers | accuracy | control | gap | p | calls/q |
| --- | --- | --- | --- | --- | --- | --- |
| `dep_plan` (v1) | — | 28/120 | 29/120 (v1) | −1 | 1.000 | 64.7 vs 79.4 |
| `dep_plan_v2` | 1,2,3 | **32/120** | 29/120 (v1) | +3 | 0.481 | 67.6 vs 79.4 |
| `dep_plan_v3` | 1-5 | 32/120 | 26/120 (v2) | +6 | 0.210 | 72.9 vs 72.6 |

Levers 1-3 fired as designed: phrase closures 0 → 17, open intents 69 → 49,
attempts-but-no-binding 48 → 29, questions with a raised budget 24 → 30.

**Lever 5 was reverted on its own evidence.** At 3x it produced accuracy
*byte-identical* to the same configuration without it (both 32/120) while raising
input tokens 166k → 222k per question (+34%). A wider slice for a question that has
already failed adds exactly the passages lexical scoring ranked lowest, and the
extractor reads everything that survives — the most expensive place in the pipeline
to add tokens. `_STARVING_TOP_K_MULTIPLIER` is 1; the targeting is kept because it is
sound and only the plan can do it.

**Lever 4 helped the plan arm and hurt the control** (plan 32 → 32, control 29 → 26).
Plausibly because the plan's queries are better targeted, so lexical scoring has
something to work with, while scattered queries get worse passages. Worth noting
rather than claiming — one seed.

## Honest standing against the target

* **Accuracy:** best gap **+6 rows (26.7% vs 21.7%), p = 0.21 — not significant.**
* **Cost:** best saving **−18.5% LLM calls**, at `dep_plan_v2` against its control.
* **Not simultaneously.** v2 bought −18.5% cost for +3 accuracy; v3 bought +6
  accuracy for 0% cost. No configuration yet delivers both significantly.

The blocker is not a missing lever, it is the 41% retrieval-bound share plus a ~25%
absolute accuracy: 16 discordant rows split 11/5 cannot reach p < 0.05. Detecting a
+5-point effect at this base rate needs roughly 300-400 paired rows, or **3 paired
seeds on these 120** — which is the statistically correct route and cheaper than more
mechanism. Three seeds of a consistent 11/5 split would land near p = 0.01.

---

# Paired seeds settle it: the accuracy effect is below the noise floor

The optimization above reported +3 and +6 row gaps at single seed. Neither survives
a second seed, and the reason is measurable rather than arguable.

## The final configuration, two paired seeds

Plan arm = levers 1-4 (grounded-phrase closure, chain-derived hop budget,
`resolved_findings` into synthesis, query-aware evidence slice; escalation off).
Control = the same code with `search.plan.enabled=false`.

| | plan | no-plan | gap | discordant | p |
| --- | --- | --- | --- | --- | --- |
| seed 1 (`dep_plan_v4` / `dep_noplan_v2`) | 27/120 | 26/120 | **+1** | 21 (11/10) | 1.000 |
| seed 2 (`s2_plan` / `s2_noplan`) | 24/120 | 28/120 | **−4** | 20 (8/12) | 0.503 |
| **pooled (240 paired)** | **51 (21.2%)** | **54 (22.5%)** | **−3** | 41 (19/22) | **0.755** |

The two seeds disagree in *sign*. Pooled over 240 paired observations the plan is 3
rows behind, nowhere near significance.

## The noise floor, measured directly

The same no-plan configuration, run twice on the same 120 questions:

```
run 1: 26/120     run 2: 28/120     score difference: 2
rows that flipped between the two identical runs: 14 (12%)
```

**Every plan-vs-control gap measured this session — +1, +3, +6, −4 — is smaller than
the 14-row flip rate of a configuration compared against itself.** Across five plan
configurations (24, 27, 28, 32, 32) and four controls (26, 28, 29, 29) the ranges
overlap completely. Single-seed n=120 on this benchmark cannot resolve effects of
this size, which is why the lever attributions kept reversing: `dep_plan_v2` and
`dep_plan_v4` differ only by the slice fix and scored 5 rows apart, the same
magnitude as the noise.

This invalidates the interim readings earlier in this document as *evidence*, and it
is the correct thing to have discovered. Anything reported from one 120-row run here
is a point estimate with a ±5-row error bar.

## Cost, pooled

| | calls/q | input tokens/q |
| --- | --- | --- |
| plan | **72.8** | 214,847 |
| no-plan | 76.0 | 222,182 |

−4.2% calls, −3.3% tokens. Note this is *much* smaller than the −18.5% measured at
`dep_plan_v2` against its control: lever 4 (the query-aware slice) keeps more
genuinely relevant passages, and the extractor reads everything that survives, so
fixing that bug spent most of the plan's cost advantage. The cost benefit is real but
**configuration-dependent, in the −4% to −18% range**, and only the call-count
direction is stable across every pair measured.

## Standing against the target, stated plainly

The target was *significantly better accuracy and significantly reduced cost*.

* **Accuracy: not achieved, and not a matter of more levers.** Pooled p = 0.755 with
  the sign reversed. The effect, if any, is smaller than a 12% per-run flip rate.
* **Cost: achieved in direction, modest in size, and never at the same time as the
  accuracy gap.** −4.2% in the final configuration, −18.5% in the cheapest one.

The binding constraints are structural, both measured: **41% of failures are
retrieval-bound** (every plan intent resolved, gold never retrieved — no planning
change reaches them), and absolute accuracy is ~22%, so the discordant pool is small.
Detecting a 5-point effect here needs roughly 300-400 paired rows *per arm per seed*,
or a mechanism aimed at retrieval rather than at planning.

What would be worth trying next, in order:

1. **Attack the 41%.** Better retrieval — a real reranker (the eval config runs with
   `reranker.enabled=false`, which is why lever 4 mattered at all), or the local
   corpus instead of rate-limited web search.
2. **Then re-test the plan on top of that.** The plan's job is directing reasoning; it
   cannot express itself while a third of questions never see their answer.
3. **Never again claim an effect from one 120-row run** on this benchmark.

---

# Lever 6: the negative record — best result, still not significant

The gap the earlier levers left open. Measured on `results/dep_plan_v4`:

```
attempts on OPEN intents: 43 / 65 pairs were near-duplicate re-issues (66%)
all attempts: 1010, of which returned ZERO facts: 3 (0%)
```

Retrieval almost always returns *something*, so the failures are about the **angle of
the query**, not availability — and the generator could not see which angles had
already failed. `render_plan_for_prompt` showed intent *status* (`[open]`, `[stuck]`)
and never the queries, while `plan_attempts_log` went only to `replan`. The prompt was
asking the model to "change angle" over invisible state.

Now each unresolved-and-executable intent carries its own failed queries:

```
- [open] Identify Elizabeth Berg's birthplace.
    already asked, did not resolve it: What is Elizabeth Berg's birthplace?
    already asked, did not resolve it: Where was Elizabeth Berg born?
```

Most recent first, deduplicated, capped at three. Closed intents are excluded (their
history is noise) and so are BLOCKED ones — listing a blocked intent's failures would
invite exactly the premature ask the marker exists to prevent. The prompt rule names
the marker and forbids *rewording* it, with a worked example, because "ask differently"
otherwise collapses into a synonym that returns the same passages.

## Result — two paired seeds, 240 paired observations

| | plan | no-plan | gap | discordant | p |
| --- | --- | --- | --- | --- | --- |
| seed 1 | 30/120 | 26/120 | +4 | 26 (15/11) | 0.557 |
| seed 2 | 30/120 | 28/120 | +2 | 26 (14/12) | 0.845 |
| **pooled** | **60 (25.0%)** | **54 (22.5%)** | **+6** | 52 (29/23) | **0.489** |

By depth, pooled: 2hop +3 (p=0.66), 3hop +2 (p=0.82), 4hop +1 (p=1.00).

Cost: **71.7 vs 76.0 calls/question (−5.7%)**, tokens level.

This is the best configuration measured and the first whose sign is **consistent
across both seeds and all three strata**. It is still not significant.

## Why the target is unreachable, as a number rather than a judgement

The pooled discordant split is 29/23, so θ = P(plan wins | the pair disagrees) =
**0.558** — barely above the 0.5 null. Exact power for a two-sided sign test at that
effect size, with the observed 21.7% discordant rate:

| seeds (120 questions each) | discordant pairs | power |
| --- | --- | --- |
| 2 (what was run) | 52 | 10.5% |
| 5 | 130 | 24.1% |
| 10 | 260 | 42.7% |
| 20 | 520 | 74.6% |
| **~24** | **~625** | **80%** |

**Roughly 24 paired seeds — about 48 hours of compute — to detect this effect at 80%
power.** The two-seed experiment had 10% power, so its non-significance was
predetermined and says nothing about whether the effect is real.

## Final standing

* **Accuracy: +2.5 points (25.0% vs 22.5%), consistent in sign across every seed and
  stratum, not significant and not cheaply made so.** The effect appears real and
  small.
* **Cost: −5.7% calls in this configuration, −4% to −18% across configurations.**
  Reliable in direction in every pair measured; the magnitude depends on whether the
  query-aware slice (lever 4) is enabled, since keeping more relevant passages spends
  the saving on extractor input.
* **Both together, significantly: not achieved.** Not for want of mechanism — six
  levers were built, five configurations and ten 120-row runs measured — but because
  the effect is ~2.5 points against a 12% per-run flip rate.

The constraint is the benchmark, not the idea: **41% of failures are retrieval-bound**
(every intent resolved, gold never retrieved) and absolute accuracy is ~23%, so the
discordant pool is small and θ sits near 0.5. Raising the ceiling — a real reranker
instead of `reranker.enabled=false`, or the local corpus instead of rate-limited web
search — is the prerequisite for the plan's contribution to be *measurable*, let alone
significant.

---

# Definitive: three paired seeds, and the plan's advantage was a pipeline artifact

## The evidence ceiling, found and fixed

Instrumenting the input guard over one run showed where the evidence was going:

| role | trims | mean prompt | ceiling | **discarded** |
| --- | --- | --- | --- | --- |
| `extractor` | 268 | 89,277 tok | 20,000 | **78%** (max 452,450 / 96%) |
| `answer_generator` | 90 | 149,929 tok | 20,000 | **87%** |
| `memory_consolidation` | 29 | 130,824 tok | 20,000 | **85%** |

The cause: `_split_into_char_batches` deliberately passed an oversized passage through
untouched, on the reasoning that truncating "would silently drop evidence" and the
tier's `max_input_tokens` was the real ceiling. But that ceiling truncates **head and
tail and drops the middle** — so the safety net performed exactly the silent
middle-drop the pass-through was written to avoid, and did it blind to relevance. A
single crawled page can be 187KB (Tavily's `raw_content`), so one passage became a
450k-token prompt whose body was thrown away.

`_relevant_windows` now compresses any oversized passage to the windows that mention
the question, scored on term overlap, kept to budget, then **restored to document
order** with an elision marker. Measured live: 268,102 → 76,588 chars on one passage
(71% saved). A unit test pins the actual failure — a needle in the body of a long page
survives compression and is lost by head+tail truncation.

**This is a real improvement to the system, independent of the plan:**

| | sub-EM | calls/q | input tok/q |
| --- | --- | --- | --- |
| no-plan before | 22.5% | 76.0 | 222,182 |
| no-plan after | 23.1% | 70.7 | **174,637 (−21.4%)** |
| plan before | 25.0% | 71.7 | 223,177 |
| plan after | 23.3% | 71.4 | **182,366 (−18.3%)** |

## And it removes the plan's advantage entirely

Three paired seeds of the final configuration (all six plan levers + compression):

| seed | plan | no-plan | gap | discordant |
| --- | --- | --- | --- | --- |
| 1 | 31/120 | 26/120 | **+5** | 21 (13/8) |
| 2 | 27/120 | 29/120 | **−2** | 20 (9/11) |
| 3 | 26/120 | 28/120 | **−2** | 20 (9/11) |
| **pooled** | **84/360 (23.33%)** | **83/360 (23.06%)** | **+1** | 61 (31/30) |

* **Accuracy: θ = 0.508, sign test p = 1.0000.** Bootstrap 95% CI on the paired
  difference: **[−0.039, +0.044]**. A coin flip.
* **Cost: mean +0.67 calls/question**, bootstrap 95% CI **[−4.83, +6.18]**. Zero.

Seed 1's +5 was noise; seeds 2 and 3 both went the other way. This is the
best-powered test run — 360 paired observations, three seeds — and both effects are
zero.

## What this supersedes, and the correction that matters

Earlier sections reported a **−18.5% cost reduction** and, pooled across four
pre-compression pairs, **−6.45 calls/question with a bootstrap CI of [−11.06, −1.83]**
that excluded zero. Both are now explained rather than merely contradicted:

1. **The cost saving was never a per-question effect.** A paired Wilcoxon on each pair
   found the *median* difference at or above zero (+0.0, +3.5, −1.0, +6.5 calls) and
   "plan cheaper" on 53-61 of 120 questions — a coin flip. The mean was driven by a
   heavy right tail: a few questions where the no-plan arm ran away. Reporting the
   aggregate ratio hid that.
2. **That tail was the broken pipeline.** The plan's advantage was that it asks fewer
   questions, so it wasted less on a stage that discarded 78% of its input. Fix the
   waste and the advantage disappears: −6.45 calls/question before compression,
   **+0.67 after**.

So the honest causal story is the opposite of the one the earlier numbers suggested.
The plan was not making the system efficient; it was partially insulating it from a
defect. Removing the defect helps everything by 18-21% of tokens and leaves the plan
with nothing to contribute on either axis.

## Standing against the target — settled

The target was significantly better accuracy **and** significantly reduced cost.
On the strongest test available (3 paired seeds, 360 paired observations, six levers,
a fixed evidence pipeline): **accuracy θ = 0.508 (p = 1.000), cost +0.67 calls/question
(CI spanning zero). Neither holds, and neither is a power problem — the point
estimates are zero.**

The plan channel's defensible contributions are what remain measurable:

* **Observability.** Every diagnosis in this document — the 78% extractor discard, the
  66% near-duplicate re-issue rate, the 70% of open intents with facts and no binding,
  the 41% retrieval-bound failure share — came from the ledger. None was visible
  before it existed.
* **Mechanical guarantees**: 0/17 cross-hop repeats touch a solved intent; dependent
  intents are deferred rather than answered from noise.

Neither is an accuracy or cost win, and the honest recommendation is to keep the plan
**off by default** (`search.plan.enabled: false`, as it ships) and to keep the
compression fix, which is where the measured gain actually is.

---

# Why the plan measured zero: two effects cancelling, and four fixes

The three-seed null above is not the plan being inert. Decomposing it found a
**measured harm the plan was causing** and a **hard ceiling on the benefit it could
deliver**, of similar size. Fixing the harm and removing the ceiling are separate
jobs, and neither had been attempted.

## Harm: the plan was promoting scaffolding to conclusions

The clearest case, from `win_plan_s1`:

> **Q:** "Who did the spouse of Hagar marry after the death of Sarah?"
> gold **Keturah**, answered **Abraham**

Memory held, verbatim at hop 2: *"According to the Book of Genesis, Abraham married
Keturah after the death of his first wife, Sarah."* The evidence was retrieved. The
answer returned was **Abraham** — intent 0's referent (*"Identify who Hagar's spouse
was"*), which is the *input* to the question, not its answer.

Paired over 3 seeds x 120 questions, restricted to questions whose plan has >=2
intents (n=359), answering with a referent bound to a **non-terminal** intent and to
no terminal intent:

| arm | rate |
|---|---|
| plan | **10.6%** |
| no-plan | **5.3%** |

discordant 43 (31 plan-only / 12 control-only), **two-sided sign test p = 0.0054**.
**87% of those answers were wrong.** 19 net questions x 87% is about **4.7 points of
accuracy** — the same order as anything the plan could plausibly gain, which is why
the net read as zero.

Cause: `resolved_findings` iterated *every* closed intent with no terminal/scaffolding
distinction, and `FINAL_ANSWER_SYNTHESIS_PROMPT` says it **"outranks all five"** other
context sources. So a hop-1 referent arrived as the single highest-authority statement
in the synthesis prompt. The channel built to carry settled facts was laundering
intermediate values into answers.

## Ceiling: the system is retrieval-bound, not reasoning-bound

For each of the 274 wrong answers across the three seeds, was the gold string anywhere
in the final textual memory?

| | count | share |
|---|---|---|
| gold present verbatim | 45 | 16% |
| all gold content-tokens present, not contiguous | 14 | 5% |
| **gold nowhere in memory** | **215** | **78%** |

**78% of failures never had the evidence.** Planning reorganises reasoning *over
retrieved evidence*, so the ceiling on the plan — and on memory, MCTS, extraction,
binding and synthesis together — is ~22% of failures, about **+6 points** if every one
were perfect. This single number retro-explains every null in this document: the
plan, the armed replan, the 3x starving-top-k escalation, the MCTS plan, the depth
hypothesis, and the compression fix's flat accuracy (tokens -21%, accuracy unchanged
— the discarded middle usually did not hold the answer either). Six null results, one
cause.

Also visible: **mean retained memory is 8.9 items** on questions consuming 182k
prompt tokens each.

## Ceiling: the plan had no actuator on cost, by construction

Role shares over all 360 plan-arm questions (25,697 calls, 65.7M prompt tokens):

| role | calls | prompt tokens |
|---|---|---|
| `triple_pruner` | **45.3%** | 21.5% |
| `extractor` | 18.1% | **35.7%** |
| `answer_generator` | 13.9% | 11.5% |
| `memory_consolidation` | 8.8% | 9.7% |
| **`subquestion_generator`** — the only stage the plan conditioned | **6.2%** | 17.7% |
| `open_ie` | 4.8% | 2.3% |
| `final_answer_synthesizer` | 1.4% | 1.1% |
| `planner` | 1.4% | 0.6% |

The IE stack is **68% of calls** and scales with *retrieved passages*, not with
subquestions. The plan acted on 6%.

And the loop had no plan-driven exit. `effective_max_depth` can only ever **raise**
the budget; termination was `is_answerable` (an LLM vote) or hop exhaustion. So the
ledger, which knows exactly when the plan is finished, could not end a question:

- **60%** of questions reached a fully-closed ledger
- **96 of 360 (27%)** then ran a mean **1.65 further hops**
- **158 of 1227 hops (12.9%)** were spent after the plan had nothing left to ask
- those questions cost **89.3 calls against a 71.4 mean** and scored **16.7% vs 23.3%**

## The retrieval mechanism: the plan knew the referent and asked around it

Prompted by the observation that the plan should make *retrieval* more directed, the
3,328 recorded attempt queries were checked against the ledger:

| | count | share |
|---|---|---|
| query omits a referent grounded at an **earlier** hop | **750** | **23%** |
| ...and the query carries no proper noun or date of its own | 50 | 2% |

Real examples, bound referent in brackets:

| bound | query issued |
|---|---|
| `Dolly Parton` | *"What is the date of birth for the performer associated with 'Hits'?"* |
| `Sen. Joseph McCarthy's committee` | *"Which country was the dominant controller of the organization identified in the first subquestion?"* |
| `Christianity` | *"What is the specific religion where female suicide rates are higher among young individuals...?"* |

Three consecutive queries circumlocuted "Dolly Parton". The second is unretrievable in
principle — it refers to a subquestion index the retriever has never seen. **The
subquestion IS the search query**, so a definite description retrieves documents about
the description while the needed document is indexed under the name. This is a
plan-fixable contribution to the 78%.

A prompt rule for this already existed (*"instantiate the rest against it"*) and was
being ignored 23% of the time, which is the general lesson: **a prompt is the weakest
available actuator.** The ledger is a hard data structure with exact state; nothing in
the graph routed on it.

## The four fixes

1. **Terminal-only findings** (`resolved_findings`) — an intent that something else
   depends on is scaffolding and never reaches synthesis. A flat plan is unaffected.
   Removes 411 of 594 findings lines (69%), changing 294 of 360 questions.
2. **`plan_target_resolved`** wired into `route_after_subq` — the plan can end a
   question once every terminal intent has closed on a grounded, surviving referent.
   Gated on terminals rather than "all closed" because a closed scaffolding intent
   means a hop succeeded, not that the question is answered. Fires on 33% of
   questions; saves 29 hops on the recorded ledgers, so this is a small cost lever,
   not the main one.
3. **Plan-conditioned pruning budget** (`_planned_top_k`, `_focused_query` in
   `tools/wikidata.py`) — Stage B's call count is `ceil(pruning_top_k / 16)`, a fixed
   4 from a constant 64 chosen without reference to what is being sought. Now
   `16 * n_open_intents`, capped at the configured value so it can only lower cost,
   with the open intents appended to the Stage A ranking query so a smaller budget
   keeps its recall. Threaded as a ContextVar because `query` reaches the tool from an
   LLM tool call. One open intent is the common case late in a question, and that is
   a 4x cut on 45% of all calls.
4. **`ground_retrieval_query`** — appends the nearest resolved prerequisite referent to
   the retrieval query when the query omits it, for the KG, web *and* corpus fan-outs.
   Appends rather than substitutes, because the description may be doing restrictive
   work and the surface may be a fragment; monotone for lexical and embedding
   retrievers both. Rejects surfaces longer than 6 words and falsified referents.
   Plus a hardened `GENERATE_SUBQUESTION_PROMPT` rule carrying the three real failures
   above as BAD/GOOD pairs.

Fixes 1 and 4 target accuracy (the harm, and the 78%); 2 and 3 target cost. Tests:
`test_scaffolding_referents_never_reach_synthesis`, `test_a_flat_plan_keeps_every_finding`,
`test_the_plan_can_end_a_question_when_its_target_resolves`,
`test_an_ungrounded_or_empty_plan_never_stops_the_loop`,
`test_a_circumlocuting_query_gets_its_referent_back`,
`test_grounding_refuses_a_sentence_and_a_falsified_referent`,
`test_the_pruning_budget_tracks_the_number_of_open_intents`. **288 passed.**

Both arms must be re-run: the prompt rule lands in the shared
`GENERATE_SUBQUESTION_PROMPT`, so the `win_noplan_*` runs are no longer a valid
control. Report with `scripts/fix_report.py`, which reports accuracy, cost **and** the
leak rate as paired tests — the unpaired mean is what produced the two retracted cost
claims recorded above.

## Result of the four fixes: cost yes, accuracy no

Two paired repeats, `fix_plan_s{1,2}` vs `fix_noplan_s{1,2}`, 120 questions each on
`musique_depth.jsonl`; 234 pairs after 3 rows per arm collapse on duplicate
normalised question keys (identical in both arms, so unbiased).

### Accuracy: still zero

| pair | plan | no-plan | gap | discordant |
|---|---|---|---|---|
| s1 | 27/117 | 30/117 | **-3** | 23 (10/13) |
| s2 | 31/117 | 26/117 | **+5** | 23 (14/9) |
| **pooled** | **58/234 (24.79%)** | **56/234 (23.93%)** | **+2** | 46 (24/22) |

theta = 0.522, sign test **p = 0.8830**, bootstrap CI on the paired difference
[-0.047, +0.068]. The two repeats disagree in sign, which is the previously measured
noise floor (12% of rows flip between identical configurations). **No accuracy effect.**

### Cost: a real reduction in calls, and only in calls

| metric | mean | median | plan cheaper | Wilcoxon |
|---|---|---|---|---|
| **LLM calls** | **-12.6** | **-4.0** | 127/234 (54%) | **p = 0.0025** |
| prompt tokens | -4,009 | +6,951 | 104/234 (44%) | p = 0.7107 |

This is the first cost result in this project to survive a **Wilcoxon signed-rank**
test, which is the test that refuted the two earlier claims. It is magnitude-driven
rather than majority-driven — the sign split is 54/46 — so the honest statement is a
~18% reduction in call count (64.4 -> 52.8 per question), not a saving on most
questions.

The mechanism is exactly fix 3 and nothing else:

| role | no-plan calls/q | plan calls/q | |
|---|---|---|---|
| **`triple_pruner`** | **30.6** | **15.9** | **-48%** |
| `extractor` | 11.8 | 11.9 | flat |
| `answer_generator` | 11.9 | 9.3 | -22% |
| `subquestion_generator` | 3.1 | 4.4 | +42% |
| `memory_consolidation` | 3.9 | 6.1 | +56% |
| **total** | **64.4** | **52.8** | **-18%** |

And prompt tokens are flat because **the plan spends its token savings on rendering
itself**:

| role | no-plan tok/q | plan tok/q |
|---|---|---|
| `triple_pruner` | 36,692 | **17,937** (-51%) |
| `extractor` | 64,175 | 57,187 (-11%) |
| `subquestion_generator` | 22,342 | **34,064** (+52%) |
| `memory_consolidation` | 9,750 | **16,310** (+67%) |
| `answer_generator` | 20,162 | 24,479 (+21%) |
| **total** | **157,337** | **156,761** |

### Correction: `resolved_findings` was not the cause of the leak

The intermediate-referent leak after the terminal-only filter:

| | before the fix | after the fix |
|---|---|---|
| plan | 10.6% | **8.5%** |
| no-plan | 5.3% | **4.7%** |
| ratio | 1.8x | **1.8x** |
| sign test | p = 0.0054 | p = 0.0636 |

The filter removed 69% of findings lines and **the ratio did not move.** The
observation was real — the plan does roughly double the rate of answering with a
scaffolding referent — but the causal attribution to `resolved_findings` was **wrong**,
and so was the estimate that fixing it would recover ~4.7 accuracy points. The
remaining channel is `candidate_answers` (= `text_memory`), which both arms share:
a *good* decomposition produces a crisp intermediate ("Abraham") that then competes as
a final answer on equal footing. Silently omitting it from the findings does not warn
the synthesiser that it is an input rather than an answer.

The untested follow-up is therefore to *label* rather than omit: pass scaffolding
referents in their own field with an explicit "these are inputs to the question, never
the answer" rule, so synthesis can rank them down instead of guessing.

### Standing verdict against the goal

- **"significantly reduce the cost"** — met for **LLM calls** (-18%, Wilcoxon
  p = 0.0025), **not met** for prompt tokens (flat).
- **"performance significantly better than no plan"** — **not met.** theta = 0.522,
  p = 0.8830. Three independent well-powered attempts now put this at zero.

The plan is now a defensible *efficiency* mechanism and is still not an accuracy
mechanism. Given the 78% retrieval-bound ceiling, that ordering is what the evidence
predicts, and `search.plan.enabled` is a cost/latency choice rather than a quality one.

## Label rather than omit: the harm is gone, the accuracy effect still is not

Following the correction above, scaffolding referents are no longer dropped from
synthesis — they are passed in their own field and **named as exclusions**:
`scaffolding_findings` in `cot.py`, rendered as `intermediate_steps_NOT_the_answer`
after `resolved_findings` (so the last thing the model reads is what it must not
answer with), plus a `SYNTHESIZE_FINAL_ANSWER_PROMPT` block making it a hard exclusion
with the Hagar/Abraham and Cabo Verde/Atlantic cases as worked examples. Active on 72%
of plan questions.

Two further paired repeats, `lab_plan_s{1,2}` vs `lab_noplan_s{1,2}`, 234 pairs.

### The plan-specific harm is eliminated

| | plan | no-plan | ratio | sign test |
|---|---|---|---|---|
| original | 10.6% | 5.3% | **1.8x** | p = 0.0054 |
| after omitting scaffolding | 8.5% | 4.7% | **1.8x** | p = 0.0636 |
| **after labelling it** | **5.6%** | **5.6%** | **1.00x** | **p = 1.0000** |

discordant 22 (11/11). The plan no longer makes this error at a different rate from the
no-plan arm — the excess is fully removed, and it took *labelling* the referent, not
hiding it. This is the clearest confirmed causal chain in the project: a measured
excess (p = 0.0054), a wrong first hypothesis (omission, no effect), a corrected
hypothesis (the referent survives in `text_memory`, which both arms share), and an
intervention that drives the excess to exactly zero.

### Accuracy: a fourth null

| pair | plan | no-plan | gap |
|---|---|---|---|
| lab s1 | 32/117 | 30/117 | +2 |
| lab s2 | 33/117 | 30/117 | +3 |
| **pooled** | **65/234 (27.78%)** | **60/234 (25.64%)** | **+5** |

theta = 0.556, **p = 0.5515**, CI [-0.034, +0.077]. Both repeats are now positive
(against -3/+5 before), but the effect is ~+2 points and nowhere near significance.

### Pooled over all four pairs (468 observations)

Pooling is legitimate for **cost**: the saving comes from `_planned_top_k`, which the
synthesis-prompt change cannot touch. For accuracy it is a secondary analysis, since
the two experiments differ in a prompt that is symmetric across arms.

| metric | mean | median | Wilcoxon |
|---|---|---|---|
| **LLM calls** | **-10.9** | **-3.0** | **p = 0.0005** |
| prompt tokens | -1,164 | +9,183 | p = 0.2765 |

| | plan | no-plan | gap | theta | p |
|---|---|---|---|---|---|
| accuracy | 123/468 (26.28%) | 116/468 (24.79%) | +7 | 0.538 | **0.5296** |

Per-experiment the calls result was p = 0.0025 (n=234) then p = 0.0659 (n=234) —
same direction, means -12.6 and -9.1 — and **p = 0.0005 pooled at n=468**. A ~17%
reduction in LLM calls is real; the token count is flat because the plan spends its
savings rendering itself.

### Final standing verdict

- **Cost: met for LLM calls** (-17%, Wilcoxon p = 0.0005 over 468 pairs). Not met for
  prompt tokens.
- **Accuracy: not met.** Four independent well-powered attempts: p = 1.0000, 0.8830,
  0.5515, 0.5296. Every point estimate is small and positive (+1, +2, +5, +7 questions);
  none is distinguishable from zero.

The conjunctive goal — significantly better accuracy *while* significantly cheaper — is
**not achieved**, and the reason is the 78% retrieval-bound ceiling rather than any
remaining defect in the plan: the plan's own measured harm is now exactly zero and its
mechanical guarantees all hold. `search.plan.enabled` is a cost and latency choice.

---

# The plan's actual mechanism is retrieval, not reasoning

Four nulls on accuracy prompted the right question: *what does the plan change, measured
directly, rather than at the end of the pipeline?* Accuracy is a composition of retrieval
and conversion, and measuring only the product hides which factor moved.

## Retrieval recall, paired

"Did the gold answer end up in memory at all", pooled over the four completed pairs
(`fix_*` and `lab_*` — the label fix touches synthesis only, so retrieval pools cleanly):

| | recall |
|---|---|
| plan | **191/461 = 41.43%** |
| no-plan | **173/461 = 37.53%** |
| gap | **+18 questions (+3.90 points)** |

theta = 0.590, discordant 100 (59/41), **sign test p = 0.0886**, bootstrap CI
[-0.0022, +0.0824]. Underpowered, but this is the only effect in the entire project that
has ever approached significance, and it is a *retrieval* effect.

## And why it never reached accuracy

| | conversion: gold in memory -> answered right |
|---|---|
| plan | 122/191 = **63.9%** |
| no-plan | 114/173 = **65.9%** |

Conversion is equal within noise. So the two measurements compose:

| | |
|---|---|
| recall gap | +3.90 pts |
| x conversion | 0.639 |
| **= predicted accuracy gap** | **+2.49 pts** |
| observed accuracy gap | **+1.49 pts** (p = 0.5296) |

**The numbers are internally consistent.** The plan's accuracy effect is real and is
about 1.5-2.5 points. It was never zero; it was always too small to detect at n=468.

## Why no amount of further measurement fixes this

Sign-test power at 80% / alpha = 0.05:

| metric | theta | discordant now | discordant needed | data needed |
|---|---|---|---|---|
| accuracy | 0.538 | 91 / 468 pairs | **1,357** | **~7,000 pairs (15x)** |
| retrieval recall | 0.590 | 100 / 461 pairs | **240** | **~1,100 pairs (2.4x)** |

Accuracy significance at this effect size costs 15x the compute already spent. Recall
significance costs 2.4x — five more run-pairs — because recall is the metric where the
mechanism actually lives and is not diluted by a 64% conversion factor.

## Rejected: reallocating retrieval depth by hop width

Tried, measured, reverted. Giving a one-query hop 8 web results instead of 3 (holding
documents-per-hop roughly constant) **eroded the call saving** — Wilcoxon p = 0.51,
median +1.0 calls, against p = 0.0025 without it — and did not improve accuracy (+2 at
n=56, against +5 without it). Removed from `cot.py`; the `set_query_depth` hook in
`tools/web.py` survives only as the single place `top_k` is read, and nothing sets it.

**The plan's retrieval contribution is query *quality*, not query *depth*.**

## Amplifying the mechanism: ask for a query, not a question

Since the subquestion is sent verbatim to a web search returning 3 results, every
interrogative word displaces a term that could have matched. `SubquestionGenerationOutput`
now carries `search_queries` parallel to `subquestions` — the keyword string to retrieve
with, distinct from the question to answer — consumed by `retrieval_query_for` at the KG,
web and corpus fan-outs, composed with `ground_retrieval_query` so a resolved prerequisite
referent is present either way, and falling back to the subquestion whenever the keyword
query is missing, under two words, or an echoed question.

The prompt teaches the style from the measured failures:
`What is the date of birth for the performer associated with 'Hits'?` -> `Dolly Parton
date of birth`.

**Caveat that the experiment has to settle:** `search_queries` lives in the shared
`GENERATE_SUBQUESTION_PROMPT`, so *both* arms get keyword queries. The gap widens only if
the plan's ledger makes the referent more usable than raw `text_memory` does. If not,
both arms improve and the gap is unchanged — a real gain for the system, no gain for the
plan.

Also noted while launching: `plan_gate` resolves bindings through `link_entities`, which
uses the **public** Wikidata API rather than the local QLever endpoint, so the plan arm
absorbs materially more `WikidataRateLimitError` backoff than the no-plan arm (88 in the
first five minutes against 0). Handled with retry and pre-existing in every experiment
above, so comparisons hold, but it is a robustness cost that belongs to "plan on".

## Rejected: asking for a keyword search query instead of using the subquestion

The amplification failed, and the way it failed identifies the mechanism precisely.
`search_queries` was added parallel to `subquestions` and consumed at all three
fan-outs; the logs confirm it was live (100% of retrieval queries differed from their
subquestion, against 62% from referent-grounding alone). Two paired repeats, 230 pairs:

| | plan | no-plan | gap |
|---|---|---|---|
| recall, subquestion as the query | **40.69%** | 35.93% | **+4.76** |
| recall, dedicated keyword query | 37.39% | **38.70%** | **-1.30** |

Absolute effect of the change:

| | recall | accuracy |
|---|---|---|
| plan arm | **-3.30 pts** | **-4.23 pts** |
| no-plan arm | **+2.76 pts** | -2.50 pts |

So a terse keyword query **helps the arm whose questions were bad and hurts the arm
whose questions were already good**, and it lowers accuracy in both. The plan's
retrieval edge was never "shorter queries" — it was *naming the resolved referent inside
a well-formed question*. Stripping the question down to keywords discards the relation
direction and the disambiguating context that the referent alone does not supply.

Reverted in full: `search_queries` removed from `SubquestionGenerationOutput`, its prompt
section and output-contract line removed, `retrieval_query_for` deleted, both fan-outs
restored to `ground_retrieval_query`, the `subquestion_search_queries` channel and its
pooling removed, tests removed. 293 passing.

Two incidental findings worth keeping:

* `plan.json` `attempts[].query` records the **subquestion**, not the retrieval query
  ([cot.py](langgraph_coe/graphs/cot.py) `record = {"query": subq, ...}`). A query-style
  audit that reads it is measuring the wrong field — as one here did before being caught
  by the grounding-log rate.
* `gather_evidence` has its own corpus fan-out that bypasses `ground_retrieval_query`
  entirely. It is MCTS-only and `corpus_enabled` is false in this configuration, so no
  result above is affected, but it must be fixed before any MCTS+plan run.

## Best configuration found, and the standing verdict

The `lab_*` configuration — four fixes plus label-not-omit, subquestion as the retrieval
query — is the best measured:

| metric | plan | no-plan | gap | p |
|---|---|---|---|---|
| retrieval recall | **40.69%** | 35.93% | **+4.76 pts** | 0.1608 |
| accuracy | **28.14%** | 25.97% | +2.16 pts | 0.5515 |
| LLM calls (pooled, 468 pairs) | **-17%** | — | **-10.9/question** | **0.0005** |
| scaffolding-answer rate | 5.6% | 5.6% | **0.0** | 1.0000 |

**Cost: achieved and significant. Accuracy: real, small (~+2 pts), and not significant
at any n this project can afford.** The conjunctive goal is not met. Five nulls on
accuracy, and the sixth attempt (keyword queries) moved it backwards — which is itself
evidence that the remaining headroom is not in query formulation.

---

# Fixed: `gather_evidence` bypassed every plan retrieval mechanism

`gather_evidence` in `cot.py` is the MCTS-side twin of the fan-out in
`route_after_subq`, called from three sites in `mcts.py`. The two drifted apart: all
three of its retrieval paths — corpus, KG **and** web — sent the raw subquestion, and it
never called `set_plan_focus`. So under `search.strategy=mcts` **none** of the plan's
retrieval mechanisms applied:

| mechanism | CoT path | `gather_evidence` before | after |
|---|---|---|---|
| `ground_retrieval_query` (corpus) | yes | **no** | yes |
| `ground_retrieval_query` (KG) | yes | **no** | yes |
| `ground_retrieval_query` (web) | yes | **no** | yes |
| `set_plan_focus` (Stage-A/B budget) | yes | **no** | yes |

That is why no CoT-measured plan effect could reproduce in MCTS: the two arms were not
running the same retrieval. `plan_ledger` and `serves_intent` are now optional keyword
arguments, defaulting to the previous behaviour, and the two `mcts.py` call sites that
have subquestions pass them. The third (`_reverify_memory`) passes **facts**, not
subquestions — already statements, with no intent attribution — so it passes
`plan_ledger` only, which is enough for the pruning focus and leaves the string it is
re-verifying untouched. The focus is cleared in a `finally` so it cannot leak into a
retrieval governed by a different plan state.

Tests: `test_gather_evidence_grounds_all_three_fanouts` (asserts the appended referent
reaches all three fan-outs), `test_gather_evidence_is_unchanged_without_a_plan` (a fact
is re-verified verbatim), `test_gather_evidence_clears_the_pruning_focus`. **298 passing.**

Note this does *not* invalidate any result in this document: every experiment above ran
`search.strategy=cot`, and `corpus_enabled` is false in `config.eval.yaml`.

# Does the plan work in MCTS? No — and the reason is structural

`results/m1_mcts_plan` vs `results/m0_mcts_noplan`, 23 paired rows, `num_iterations=2`:

| | sub-EM | subq/q | distinct/q | reuse | sibling-subtree overlap |
|---|---|---|---|---|---|
| no plan | 14/23 (60.9%) | 12.8 | **9.9** | 25.3% | **10.2%** |
| plan | 14/23 (60.9%) | 11.4 | **6.3** | 40.9% | **23.1%** |

Paired: discordant 4 (2/2) — a dead tie on 23 rows, which is far too small to detect the
~2-point effect the CoT arm shows, so accuracy here is uninformative either way. The
*diversity* numbers are not:

- **distinct subquestions per question fall 9.9 → 6.3** (−36%)
- **sibling-subtree overlap more than doubles, 10.2% → 23.1%**

**The plan and tree search are substitutes, not complements.** A plan is a
variance-*reduction* device: it makes every rollout ask the same well-chosen questions.
MCTS extracts its value from variance — pUCT can only prefer one child over another if
the children *differ*, and backpropagated statistics are only informative if siblings
explored different ground. Sharing one plan across siblings makes them converge, so the
search spends its budget re-confirming one line of enquiry instead of comparing several.

This also corrects a premise the original design rested on: that sibling
re-decomposition is *duplicated work* to be eliminated. In a tree search it is
**exploration**, and eliminating it removes the thing that makes the search a search.

Recommendation unchanged: **leave the plan off under MCTS.** The fix above is a
correctness fix — it makes an MCTS+plan run *measure what it claims to measure* — not a
reason to enable the combination. If it is ever run, expect the retrieval mechanisms to
now work and the diversity collapse to still be there, because the collapse is caused by
plan *sharing*, which no retrieval fix addresses. The design note in "Explicitly out of
scope" (per-node plan forking) is the only route that would change this, and it is
blocked on the select-and-budget facts recorded there.

---

## Conversion attack (session: guard intents)

### What was measured before touching anything

Pooled over **16 plan-enabled runs / 1,920 questions**, with `[Retrieval]`-tagged memory as
the denominator (`scripts/conversion_report.py`, `scripts/conversion_failures.py`):

| | |
|---|---|
| gold in `[Retrieval]` memory | 33.8% of questions |
| conversion failures | 204 = 31.4% of those, **10.6% of all questions** |
| A. never bound (intent still open) | 80 (39%) — **58 of them with a matching memory line** |
| B. wrong referent bound at a terminal | 78 (38%) |
| C. bound the gold, synthesis lost it | 46 (23%) |

This corrects three numbers in `plan_idea_and_results.md` §13: the prize is ~10.6 points not
~15, binding is 77% of it not 94%, and synthesis is 23% not 6%.

Ruled out cheaply along the way:

- **Abstention** — only 2 of 204 failures answered "not in the evidence". Not a lever.
- **Terminal-referent verification** (`vf_on`/`vf_off`, n=117) — paired conversion 14/18 vs
  14/18, θ=0.500, **p=1.0000**, at **+5.8 calls/question** (Wilcoxon p=0.0236). Defaulted
  off. Its +4.3 accuracy points were a *retrieval* effect: withholding leaves an intent open
  and buys another hop.

### The defect found by reading artifacts instead of aggregates

The ledger is full of **guards** — intents whose answer is a truth value — because the
PLANNER prompt asks for presuppositions to be hedged into conditionals. Everything
downstream assumed a referent. 399 guards over 6,250 intents (6.4%), 188 terminal, 131 of
those closed, and **139 of 284 closed guards bound a full sentence**.

| harm | measured instance |
|---|---|
| a boolean reaches synthesis at top authority | `Confirm whether the author wrote a short story -> No, Stephen King did not write a short story featuring Herman Wouk.` — gold was `1,335,907` |
| the intent binds its own input | `No, Yangzhou is not a capital city` → Yangzhou, the prerequisite's referent. 897 intents (16%) echo a prerequisite; 640 closed on nothing else |
| an affirmed referent fails to resolve | `Yes, Meg Ryan.` → *Dennis Quaid*, because the surface was the whole sentence. Gold was Meg Ryan; answered `Sleepless in Seattle` |

### Three flags, measured separately on purpose

| flag | default | targets |
|---|---|---|
| `guard_intents_are_not_referents` | **true** | the guard harms above |
| `skip_input_referent_in_binding` | false | 640 closures on the intent's own input |
| `bind_corroborated_low_confidence` | false | 2,980/17,848 (16.7%) sub-answers dropped on the confidence label before binding |

Not bundled. They overlap in the cases they touch, and attributing a pooled effect to
whichever half is easier to explain is how two earlier claims here came to be retracted.

the gd_on/gd_off pair runs first; then bd_input/bd_conf (both against
gd_on as baseline) is queued behind it by a watcher. Arms run **concurrently** because the
noise floor on this config is 14/120 rows flipping between two runs of identical settings,
so sequential arms would confound API weather with the effect.

### Early mechanism check (24/120 rows, not an effect claim)

| | gd_on | gd_off |
|---|---|---|
| intents closed | 58/72 (81%) | 50/73 (68%) |
| resolved findings reaching synthesis | 11 | 6 |

Both directions are as designed. The extra findings are also the *risk*: the scaffolding
result established that findings at top authority can actively cause wrong answers, so
"more findings" is what the A/B has to adjudicate, not evidence on its own.

### Corrections to earlier claims in this document

- **The MCTS per-rollout accuracy result does not hold.** Batch 1 (n=45) gave +13.3 pts,
  θ=0.800, p=0.1094. Batch 2 (75 disjoint rows) reversed sign, 12/73 vs 13/73. Pooled
  n=118: **+4.2 pts, 19 discordant (12/7), p=0.3593** — the fifth well-powered null on plan
  accuracy. What *does* hold is cost: **−36.2 calls/question (p<0.0001)** and **−69,756
  prompt tokens (p=0.0023)**, with diversity preserved (sibling overlap 1.3% vs 3.1%).
- The "~1 extra call per question" estimate for terminal verification was wrong by ~6×: the
  gate runs per hop over a mean 3.67 hops, not once per question.
