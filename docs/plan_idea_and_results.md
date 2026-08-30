# The plan channel: idea, mechanisms, and measured results

**Start at [RESULTS.md](RESULTS.md)** for the one-page summary of what is and is not
claimable, the noise floor that bounds every accuracy number here, and the commands to
reproduce each figure. This document is the full evidence behind it.

A consolidated account. The chronological working log — including superseded claims and
the reasoning that produced them — is [plan_channel_status_and_plan.md](plan_channel_status_and_plan.md);
this document is the settled version.

**Headline.** The plan delivers a **significant cost reduction** (−17% LLM calls,
Wilcoxon p = 0.0005 over 468 paired questions) and a **real but not significant accuracy
gain** (+1.5 points, p = 0.5296). Its mechanism is **retrieval**, not reasoning: it gets
the gold answer into memory on 41.4% of questions against 37.5% without
(p = 0.0886). It also *caused* a specific class of wrong answer, which was diagnosed and
eliminated. Under MCTS a shared plan **harms** the search and must not be enabled.

---

## 1. The idea

A *plan* is prose stating **what still needs to be found out** — interrogative, never
assertive. It lives in its own state channel, reaches the model only through dedicated
prompt fields, and **never enters `text_memory`**. That last constraint is load-bearing:
an interrogative in memory becomes a retrieval query, then a verifier-grounding target,
then a synthesis candidate.

Alongside the prose sits a **ledger**: one entry per *intent*, carrying its status, its
dependencies (`depends_on`), the referent it resolved to (`bindings`), the memory facts it
cites (`premises`), and the queries already tried against it (`attempts`).

The prose conditions the model. **The ledger is the part that does the work** — it is exact
state that graph code can route on. The single most useful lesson from this project:

> A prompt is the weakest available actuator. A rule saying "instantiate the plan against
> memory" already existed and was being ignored on 23% of issued queries. The ledger is a
> data structure; code can enforce what a prompt only requests.

### The original hypotheses, and how they fared

| # | Hypothesis | Outcome |
|---|---|---|
| 1 | With a plan the reasoning is more focused — no re-asking what is solved | **Held mechanically.** 0/17 cross-hop repeats target a solved intent. |
| 2 | The plan knows which subquestions are answerable *now*, so it won't fire them all at once | **Held.** Dependent intents defer instead of answering from noise. |
| 3 | The plan saves cost in CoT | **Held, after being built to.** −17% calls, but only once the plan gained an actuator on the expensive stage. |
| 4 | The plan gives MCTS better rollouts | **Refuted for a shared plan** — it collapses search diversity. |

Hypothesis 3 is worth dwelling on: it was *false* for eighteen months of measurement and
became true only when the plan was given control of the retrieval budget. Focus alone
bought nothing, because the stage the plan influenced was 6% of the spend.

---

## 2. Where the cost actually is

Measured over 360 plan-arm questions (25,697 calls, 65.7M prompt tokens):

| role | calls | prompt tokens |
|---|---|---|
| `triple_pruner` | **45.3%** | 21.5% |
| `extractor` | 18.1% | **35.7%** |
| `answer_generator` | 13.9% | 11.5% |
| `memory_consolidation` | 8.8% | 9.7% |
| **`subquestion_generator`** ← all the plan originally touched | **6.2%** | 17.7% |
| `open_ie` | 4.8% | 2.3% |
| `final_answer_synthesizer` | 1.4% | 1.1% |
| `planner` | 1.4% | 0.6% |

The IE stack is **68% of calls** and scales with *retrieved passages*, not with
subquestions. Any "the plan asks fewer questions" story is capped at 6% before it starts.

---

## 3. Where the accuracy ceiling is

For each of 274 wrong answers, was the gold string anywhere in final memory?

| | count | share |
|---|---|---|
| present verbatim | 45 | 16% |
| all gold content-tokens present | 14 | 5% |
| **absent entirely** | **215** | **78%** |

**78% of failures never had the evidence.** So the ceiling on *every* reasoning-side
lever together — plan, memory, MCTS, extraction, binding, synthesis — is ~22% of failures,
about **+6 accuracy points**. This one number retro-explains six independent nulls:
the plan, the armed replan, 3× rerank escalation, the MCTS plan, the depth hypothesis, and
the compression fix's flat accuracy.

The document surface is the reason: `retriever.enabled: false` and `web_search.top_k: 3`,
at a measured 2.55 queries per hop — **~7.6 documents per hop, ~24 per question**, from
which a mean **7.7–8.9 memory items** survive.

Of the 22% that *is* addressable, **conversion is 64%**: 122 of 191 questions whose memory
held the gold were answered correctly. Metric artifacts are negligible and symmetric
(1–2 per run under article/punctuation normalisation).

---

## 4. The mechanisms, and what each is worth

### 4.1 Plan-conditioned pruning budget — **the entire cost win**

`triple_pruner` runs `ceil(pruning_top_k / 16)` calls per KG fetch: a fixed **4**, from a
constant 64 chosen without reference to what was being sought. Stage A already reduces
1,780,510 raw triples to 141,248 (7.9%), so the 64 is what Stage B pays for.

`_planned_top_k` scales it to `16 × open_intents`, capped at the configured value so it can
only ever lower cost, with the open intents appended to the Stage-A ranking query so a
smaller budget keeps its recall. Threaded as a ContextVar because the query reaches the
tool from an LLM tool call.

Measured live: 76 focused fetches (46 at `top_k` 16, 26 at 32, 4 at 48) → **−64% Stage-B
calls** on those fetches.

> A rejected hypothesis worth recording: dropping zero-overlap triples before Stage B.
> Measured 99% of kept triples share a content word with the query, so the saving was 1%.

### 4.2 Referent grounding — the retrieval mechanism

The subquestion **is** the search query. Measured over 3,328 issued queries, **23% omitted
a referent the plan had already bound at an earlier hop** (35% of the 1,738 queries where a
short grounded prerequisite referent existed). Real examples, bound referent in brackets:

| bound | query issued |
|---|---|
| `Dolly Parton` | *"What is the date of birth for the performer associated with 'Hits'?"* (×3) |
| `Sen. Joseph McCarthy's committee` | *"…the organization identified in the first subquestion?"* |
| `Christianity` | *"What is the specific religion where female suicide rates are higher…?"* |

The second is unretrievable in principle — it cites a subquestion index the retriever has
never seen. `ground_retrieval_query` **appends** (never substitutes) the nearest resolved
prerequisite referent, rejecting surfaces over six words and falsified referents.

A bug found late by audit and fixed: the KG fan-out gate tested the **ungrounded**
subquestion (`_subq_hits_known_entity(sq, …)`) while sending the grounded query — so the KG
was skipped for exactly the queries grounding had just made KG-answerable.

### 4.3 Terminal-only findings and labelled scaffolding — a harm, diagnosed and removed

The plan *caused* a class of wrong answer. Restricted to questions with ≥2 intents,
answering with a referent bound to a **non-terminal** intent and to no terminal one:

| | plan | no-plan | ratio | sign test |
|---|---|---|---|---|
| original | **10.6%** | 5.3% | 1.8× | **p = 0.0054** |
| after *omitting* scaffolding from findings | 8.5% | 4.7% | 1.8× | p = 0.0636 |
| after **labelling** it as an exclusion | **5.6%** | **5.6%** | **1.00×** | **p = 1.0000** |

**87% of those answers were wrong.** The canonical case:

> *"Who did the spouse of Hagar marry after the death of Sarah?"* — gold **Keturah**,
> answered **Abraham**. Memory held, verbatim at hop 2: *"Abraham married Keturah after the
> death of his first wife, Sarah."*

`resolved_findings` fed **every** closed intent to synthesis, and the synthesis prompt ranks
it above all other context — so a hop-1 referent arrived as the highest-authority statement
in the prompt.

**The first fix did not work, and the reason matters.** Omitting scaffolding removed 69% of
findings lines and moved the ratio not at all: the referent survives in
`candidate_answers` (= `text_memory`), which both arms share, and a *good* decomposition
makes it the **crispest** candidate there — resolving it cleanly is what unlocked the later
hops. Synthesis had to be *told* which referents are inputs, not left to infer it from an
absence. `scaffolding_findings` now names them under
`intermediate_steps_NOT_the_answer`, rendered last so the final thing the model reads is
what it must not answer with.

This is the project's one complete causal chain: measured excess → wrong first hypothesis →
corrected hypothesis → intervention driving the excess to exactly zero.

### 4.4 Ledger-driven termination — a small cost lever

Termination was `is_answerable` (an LLM vote) or hop exhaustion; `effective_max_depth` can
only ever *raise* the budget. So the ledger could not end a question:

- **60%** of questions reached a fully-closed ledger
- **96 of 360 (27%)** then ran a mean **1.65 further hops**
- **158 of 1,227 hops (12.9%)** were spent after the plan had nothing left to ask
- those questions cost **89.3 calls against a 71.4 mean** and scored **16.7% vs 23.3%**

`plan_target_resolved` ends the question once every *terminal* intent closes on a grounded,
surviving referent — gated on terminals rather than "all closed", because a closed
scaffolding intent means a hop succeeded, not that the question is answered. Fires on 33%
of questions and saves 29 hops: a small lever, not the main one.

### 4.5 Supporting mechanisms

- **Dependency ordering.** `depends_on` with a sanitizer rejecting self- and
  forward-references; `is_executable` walks the whole ancestor chain, so a 4-hop plan does
  not unblock intent 3 when intent 2 closed on evidence that itself depended on an
  unresolved intent 1.
- **Binding tiers.** QID → normalized literal → grounded phrase, admitted only when
  corroborated by retrieval. Measured closures: **96% QID, 4% literal**.
- **Pool caps.** Near-duplicate filter at 0.95 similarity and a 2-per-intent cap. Both are
  backstops: the generator's diversity instruction largely works, and the twin filter drops
  only ~3%.
- **`N_PLANS = 1`.** Sampling 3 plans and keeping one was measured to consume the
  alternatives **0/62 times**. A replan conditioned on *why* the plan failed beats an iid
  draw made before any evidence existed.
- **`replan_max = 0`.** On the 20 questions that replanned, armed scored 10 against 9 for
  both log-only and no-plan, while costing 2.55 hops / 8.95 subquestions against 1.24 /
  3.02 elsewhere. The trigger fires on referents that are *genuinely ambiguous in the
  world*, and rewriting what to ask cannot settle that.

---

## 5. Methodology

Every claim is a **paired** comparison on the same questions, because the unpaired mean
produced two retracted claims (§7).

- **Accuracy / recall / leak:** two-sided exact **sign test** on discordant pairs
  (McNemar), plus a bootstrap CI on the mean paired difference.
- **Cost:** **Wilcoxon signed-rank** — the test that refuted the earlier claims. A
  bootstrap CI on the *mean* can exclude zero while the median sits at zero and "plan
  cheaper" runs at a coin flip; `scripts/fix_report.py` now prints both and flags that
  disagreement explicitly.
- **Noise floor:** the same no-plan config run twice scores 26 vs 28 with **14 of 120 rows
  (12%) flipping** — larger than most effects measured here. Two arms disagreeing in sign
  across repeats is expected, not informative.
- **Verifier noise floor:** 30 identical VERIFIER calls at temperature 0.7 span **3.0
  rating points**, which bounds any spread statistic.
- **Data:** `datasets/musique_depth.jsonl` — 120 MuSiQue rows, 40 each at 2/3/4 hops,
  sub-types proportional, interleaved by depth so partial runs stay balanced.
- **Repeats** are independent draws, not seeded: `seed` is null in every config, and
  nondeterminism comes from temperature and asyncio completion order.

---

## 6. Results

### 6.1 Cost — achieved

Pooled over 468 paired questions. Pooling is legitimate here: the saving comes from
`_planned_top_k`, which no other change touches.

| metric | mean | median | plan cheaper | Wilcoxon |
|---|---|---|---|---|
| **LLM calls** | **−10.9** | **−3.0** | 250/468 (53%) | **p = 0.0005** ✅ |
| prompt tokens | −1,164 | +9,183 | 206/468 (44%) | p = 0.2765 ❌ |

**64.4 → 52.8 calls per question, −17%.** Magnitude-driven rather than
majority-driven (the sign split is 53/47), so the honest statement is "−17% on average",
not "cheaper on most questions". Per-experiment: p = 0.0025, then p = 0.0659, then
p = 0.0005 pooled — same direction throughout.

**The mechanism is one role:**

| role | no-plan | plan | |
|---|---|---|---|
| **`triple_pruner`** | **30.6** | **15.9** | **−48%** |
| `answer_generator` | 11.9 | 9.3 | −22% |
| `extractor` | 11.8 | 11.9 | flat |
| `subquestion_generator` | 3.1 | 4.4 | +42% |
| `memory_consolidation` | 3.9 | 6.1 | +56% |
| **total** | **64.4** | **52.8** | **−18%** |

**Tokens are flat because the plan spends its savings rendering itself:**

| role | no-plan tok/q | plan tok/q |
|---|---|---|
| `triple_pruner` | 36,692 | **17,937** (−51%) |
| `extractor` | 64,175 | 57,187 (−11%) |
| `subquestion_generator` | 22,342 | **34,064** (+52%) |
| `memory_consolidation` | 9,750 | **16,310** (+67%) |
| `answer_generator` | 20,162 | 24,479 (+21%) |
| **total** | **157,337** | **156,761** |

### 6.2 Retrieval recall — the plan's real effect

"Did the gold answer reach memory at all", pooled over 461 pairs:

| | recall |
|---|---|
| **plan** | **191/461 = 41.43%** |
| no-plan | 173/461 = 37.53% |
| gap | **+18 questions (+3.90 points)** |

θ = 0.590, discordant 100 (59/41), **p = 0.0886**, bootstrap CI [−0.002, +0.082]. The only
effect in the project that has ever approached significance.

### 6.3 Accuracy — real, small, not significant

| experiment | plan | no-plan | gap | p |
|---|---|---|---|---|
| three-seed baseline | 84/360 (23.33%) | 83/360 (23.06%) | +1 | 1.0000 |
| four fixes | 58/234 (24.79%) | 56/234 (23.93%) | +2 | 0.8830 |
| + labelled scaffolding | 65/234 (27.78%) | 60/234 (25.64%) | +5 | 0.5515 |
| **pooled (468)** | **123/468 (26.28%)** | **116/468 (24.79%)** | **+7** | **0.5296** |

### 6.4 The numbers are internally consistent

| | |
|---|---|
| retrieval recall gap | +3.90 pts |
| × conversion (63.9% plan vs 65.9% no-plan — equal) | 0.639 |
| **⇒ predicted accuracy gap** | **+2.49 pts** |
| observed accuracy gap | **+1.49 pts** |

The accuracy effect was **never zero**. It is ~1.5–2.5 points, exactly what a +3.9-point
recall gain converts to, and it was always too small to detect at this n.

### 6.5 Why more measurement will not fix that

Sign-test power, 80% at α = 0.05:

| metric | θ | discordant now | needed | data needed |
|---|---|---|---|---|
| accuracy | 0.538 | 91 / 468 pairs | **1,357** | **~7,000 pairs (15×)** |
| retrieval recall | 0.590 | 100 / 461 pairs | **240** | **~1,100 pairs (2.4×)** |

Accuracy significance at this effect size costs 15× the compute already spent. **Recall
significance costs 2.4× — five more run-pairs** — because recall is where the mechanism
lives, undiluted by a 64% conversion factor.

---

## 7. Retracted claims

Recorded because the reasoning that produced them is a reusable warning.

1. **"−18.5% cost", then "−6.45 calls/question with a CI excluding zero."** Both were
   heavy-right-tail artifacts of the *unpaired mean*. A paired Wilcoxon put the median at
   or above zero in all four pre-compression pairs (+0.0, +3.5, −1.0, +6.5) with "plan
   cheaper" on 53–61 of 120 questions — a coin flip. **The tail was a broken pipeline**
   (§8): the plan wasted less on a stage discarding 78% of its input. Fix the stage and the
   advantage vanishes — −6.45 calls before, **+0.67 after**.
2. **"`resolved_findings` causes the leak; fixing it recovers ~4.7 points."** The leak was
   real and significant; the attribution was wrong. Omission changed the ratio not at all
   (§4.3).
3. **"MCTS shows identical accuracy under a plan."** Wrong framing. 4 discordant rows of 23
   (p = 1.0) against a ~12% same-config flip rate is **uninformative**, not identical.
4. **"The +6 accuracy gap is stable across three checkpoints."** It was noise at n=40.
5. **A "regime mismatch" explanation for the bamboogle null** — simply wrong.
6. **"Sibling re-decomposition is duplicated work."** In a tree search it is *exploration*
   (§9).

---

## 8. The pipeline defect the plan helped find

The plan's best contribution was diagnostic. Instrumenting the input guard:

| role | mean prompt | ceiling | discarded |
|---|---|---|---|
| `extractor` | 89,277 | 20,000 | **78%** (max 452,450 / 96%) |
| `answer_generator` | 149,929 | 20,000 | 87% |
| `memory_consolidation` | 130,824 | 20,000 | 85% |

`_split_into_char_batches` deliberately passed oversized passages through, reasoning that
truncation "would silently drop evidence" and the tier ceiling was the real limit. But that
ceiling keeps **head and tail and drops the middle** — performing the exact silent
middle-drop the pass-through existed to prevent, relevance-blind. One 187 KB web page
became a 450k-token prompt whose body was discarded.

`_relevant_windows` compresses oversized passages to query-relevant windows in document
order:

| | sub-EM | input tok/q |
|---|---|---|
| no-plan before | 22.5% | 222,182 |
| no-plan after | 23.1% | **174,637 (−21.4%)** |
| plan after | — | −18.3% |

**This is the largest single win in the project and it is plan-independent.** Accuracy was
flat, which §3 explains: the discarded middle usually did not hold the answer either.

---

## 9. MCTS: a shared plan harms the search

`results/m1_mcts_plan` vs `m0_mcts_noplan`, 23 paired rows, `num_iterations=2`:

| | sub-EM | subq/q | distinct/q | reuse | sibling overlap |
|---|---|---|---|---|---|
| no plan | 14/23 | 12.8 | **9.9** | 25.3% | **10.2%** |
| plan | 14/23 | 11.4 | **6.3** | 40.9% | **23.1%** |

Accuracy is **uninformative** (4 discordant, p = 1.0, below the noise floor). Diversity is
not: distinct subquestions per question **9.91 → 6.30, paired p = 0.0266**; sibling overlap
more than doubles; distinct subquestions per tree node falls 0.67 → 0.49 (−27%).

**Why, structurally.** This search has exactly one stochastic branching operator:
`_gen_subqa` samples `SUBQUESTION_GENERATOR` at n=3, and the pooled subquestions *are* the
node's children. The tree's action space is the support of that sampling distribution — and
the plan is injected into that same call. A plan is a variance-**reduction** device;
pUCT can only prefer one child over another when the children differ, and backpropagated
statistics only inform if siblings explored different ground. Narrowing the branching
distribution is right in CoT (fewer redundant serial hops) and self-defeating in a tree.

Worse, `N_PLANS = 1` and `gen_plan` runs **once at the root over empty memory**, so a
tree-scoped plan **cannot vary across the tree by construction**.

### 9.1 Per-rollout plan scope

`search.plan.mcts_plan_scope` now defaults to **`"rollout"`**: the tree holds no plan (the
root `gen_plan`/`plan_gate` nodes are not even added), and each rollout runs the CoT loop
and plans from **its own branch memory**. The plan's benefits are all *within* one
reasoning chain and none require siblings to share anything, so plans differ where branches
differ and diversity is preserved by construction. The winning rollout's ledger is carried
out on a separate `rollout_plan_ledger` channel — feeding synthesis (so the
scaffolding exclusion survives) while nothing inherits it.

**Result: per-rollout planning is a large, significant cost saving at accuracy that is
positive but not distinguishable from noise.** Three arms, 45 rows, `max_simulation_depth=2` (depth 2 rather than the eval
config's 1, because at depth 1 no referent is bound at an earlier hop so referent
grounding can never fire):

| arm | sub-EM | distinct subq/q | reuse | sibling overlap |
|---|---|---|---|---|
| no plan | 12/45 = 26.7% | 17.6 | 24.2% | 3.1% |
| **tree-shared plan** | 11/45 = 24.4% | **9.7** | 35.1% | **17.9%** |
| **per-rollout plan** | **18/45 = 40.0%** | 15.3 | **15.4%** | **1.3%** |

Paired against the no-plan arm (n=45):

| | accuracy | calls | prompt tokens |
|---|---|---|---|
| **per-rollout** | **+13.3 pts** (theta = 0.800, 10 discordant 8/2, p = 0.1094) | **-36.5/q, Wilcoxon p = 0.0005** | **-75,004/q, p = 0.0060** |
| tree-shared | -2.2 pts (theta = 0.455, p = 1.0000) | -66.2/q, p < 0.0001 | -135,323/q, p < 0.0001 |

Three things follow.

1. **The diversity mechanism is confirmed in both directions.** Tree scope reproduces the
   collapse (distinct subquestions 17.6 -> 9.7, sibling overlap 3.1% -> 17.9%); rollout
   scope *preserves* it (15.3 distinct, and overlap of 1.3% — lower than no-plan at all).
   Sharing was the whole problem, exactly as the structural argument predicted.
2. **The accuracy effect did not survive the power it needed.** A second batch of 75 rows
   (`mx2_np` / `mx2_roll`, disjoint questions) took the paired n from 45 to 118 — and
   reversed sign within that batch (12/73 vs 13/73). Pooled:

   | | accuracy | calls | prompt tokens |
   |---|---|---|---|
   | per-rollout, **pooled n=118** | **+4.2 pts** (25.42% vs 21.19%), 19 discordant (12/7), theta = 0.632, **p = 0.3593** | **-36.2/q, Wilcoxon p < 0.0001** | **-69,756/q, p = 0.0023** |

   So the +13.3 points from batch 1 was a small-sample artifact, and the honest reading is
   **equal accuracy at 36 fewer calls and ~70k fewer prompt tokens per question**. That is
   the same shape as the CoT result (§7) and it is the *cost* half of the goal, now
   reproduced under tree search at n=118 with p < 0.0001.

   This is the fifth well-powered null on plan accuracy. The pattern is consistent enough
   to state plainly: the plan buys efficiency, not correctness.
3. **The tree-shared arm is cheaper still and worse.** That is the cleanest possible
   demonstration that its saving comes from *doing less search*, not from searching better.

Caveat on absolute numbers: these runs postdate the local entity-linking fallback (§13.1)
and the multi-provider search chain, so they are not comparable to the CoT figures earlier
in this document. The plan-vs-no-plan *contrast* is internally valid — the arms differ
only in `search.plan.enabled` and `mcts_plan_scope`.

### 9.2 The MCTS path was never running the plan

An audit of every retrieval site found **19 confirmed bypasses**, two reachable in the
shipped CoT config. Most consequentially, `gather_evidence` — used *only* by MCTS — sent
the raw subquestion on **all three** fan-outs and never set the pruning focus:

| mechanism | CoT path | `gather_evidence` before | after |
|---|---|---|---|
| `ground_retrieval_query` (corpus / KG / web) | ✓ | ✗ | ✓ |
| `set_plan_focus` | ✓ | ✗ | ✓ |

So `m0`/`m1` were never a valid MCTS+plan comparison — and in any case they **predate all
five mechanisms** (HEAD `ad403c6`). Also fixed: `effective_max_depth` silently deepening
rollouts past `max_simulation_depth`; a rollout ending via `plan_target_resolved` not
counting as a sufficiency vote; `_gen_subqa` dropping intent attribution when writing the
SUB_QA node, so self-correction re-retrieved circumlocuting sub-questions verbatim; and
`_reverify_memory` omitting the ledger, which did not leave the focus alone but *disabled*
it (`set_plan_focus([], 0)`), paying 64-candidate pruning while siblings paid 16.

---

## 10. Rejected levers, with their numbers

Each was implemented, measured, and reverted.

| lever | result |
|---|---|
| **Keyword `search_queries`** instead of the subquestion | Plan recall **−3.30 pts**, no-plan recall **+2.76**; accuracy fell in both (−4.23 / −2.50); the plan's recall gap went **+4.76 → −1.30**. A terse query helps the arm whose questions were bad and hurts the arm whose questions were already good. The plan's edge was *naming the referent inside a well-formed question*, not brevity. |
| **Concentrating web `top_k` by hop width** | Eroded the call saving (Wilcoxon **p = 0.51**, median **+1.0** calls, vs p = 0.0025 without) and did not help accuracy (+2 vs +5). Query *quality*, not *depth*. |
| **3× rerank escalation for starving intents** | Byte-identical accuracy, input tokens 166k → 222k (**+34%**). |
| **Armed replan** (`replan_max > 0`) | 10 vs 9 correct on the 20 questions that replanned, at 2.55 hops / 8.95 subquestions against 1.24 / 3.02. |
| **Sampling 3 plans** | Alternatives consumed **0/62 times**. |
| **Dropping zero-overlap triples** before Stage B | 99% of kept triples already overlap; saving 1%. |
| **Gold-paragraph depth test** | Withdrawn: 0 of 19 probe attempts returned zero facts, so retrieval was not returning nothing. |

---

## 11. Configuration

```yaml
search:
  plan:
    enabled: false               # ships off; see §12
    replan_max: 0                # armed replan measured worthless
    replan_min_depth_headroom: 1
    stall_after_attempts: 3
    memory_disagreement_threshold: 4.0   # verifier noise floor is 3.0 points
    mcts_plan_scope: rollout     # "tree" collapses search diversity
```

Every value is evidenced above. When `enabled: false`, the plan nodes are not added to
either graph, so the baseline is structurally identical to the pre-plan system.

---

## 12. What to claim, and what not to

**Defensible:**

- **−17% LLM calls at equal accuracy**, with a named mechanism (plan-conditioned pruning
  budget) and a paired Wilcoxon at p = 0.0005 over 468 questions.
- **−21% prompt tokens** from query-relevant compression, accuracy held (§8) —
  plan-independent, and the largest single win.
- **A plan-caused failure mode identified and eliminated**: 1.8× excess at p = 0.0054 driven
  to exact parity, with the intermediate wrong hypothesis documented.
- **A quantified bottleneck decomposition**: 78% retrieval-bound, and a conversion block
  worth 10.6% of all questions, re-measured at n=204 and decomposed into 77% binding /
  23% synthesis (§13.4).

**Not defensible:** any claim that planning improves multi-hop accuracy. Four
well-powered attempts, all null; the effect is real at ~1.5–2.5 points and needs ~7,000
paired questions to demonstrate.

**Defensible (new):** the cost saving reproduces under tree search. Per-rollout planning is
-36.2 calls and -69,756 prompt tokens per question at n=118, Wilcoxon p < 0.0001 and
p = 0.0023, with search diversity *preserved* (sibling overlap 1.3% vs no-plan's 3.1%).

**Not defensible:** anything about the plan helping tree search on *accuracy*. A shared plan
measurably hurts it (-2.2 pts with diversity collapse); the per-rollout variant is +4.2 pts
at p = 0.3593 — the fifth well-powered null.

`search.plan.enabled` is a **cost and latency knob, not a quality one.**

The highest-leverage remaining work is not planning. It is **retrieval** — 78% of failures
never had the evidence, and the surface is three web results per query — followed by
**conversion**, where 31% of successfully retrieved gold still fails to become the answer
(§13.4). Within conversion, the identified defect is that a *guard* intent's yes/no answer
was being treated as a referent (§13.6).

---

## 13. Conversion: where retrieved gold fails to become the answer

§3 established that 36% of questions whose memory held the gold were still answered
wrong — ~15% of all questions, the largest single addressable block left. Attributing
those 70 failures to a pipeline stage:

| stage | share |
|---|---|
| **synthesis never used the gold at all** | **79%** |
| synthesis reasoned to it in prose, the concise field lost it | 21% |

But the deeper attribution inverts the obvious reading. Checking whether the plan had
already handed synthesis the answer at top authority:

| | share |
|---|---|
| **terminal intent never closed** — no `resolved_findings` at all | **53%** |
| **closed on the wrong referent** — findings present, gold absent | **41%** |
| gold *was* at top authority and synthesis ignored it | **6%** |

**94% of conversion failure is binding, not synthesis.** Two of the three sampled cases in
that last 6% are metric artifacts (`The Australian Ballet` vs `Australian Ballet`), so
synthesis ignoring a correct top-authority finding is close to negligible.

### 13.1 The 53%: binding starved by throttled entity linking

Binding is QID-first — measured tiers across 240 questions: **QID 685, literal 52,
phrase 92** — so it runs through `link_entities`, and that call hits the **public**
Wikidata API. `wikidata.sparql_endpoint` points at local QLever, which covers k-hop
triples but *not* the search-ranking service:

| call | endpoint | status |
|---|---|---|
| k-hop triples (`fetch_outgoing`) | localhost:7001 | local |
| **label → QID (`wbsearchentities`)** | wikidata.org/w/api.php | **throttled** |
| **labels/descriptions (`wbgetentities`)** | wikidata.org/w/api.php | **throttled** |

Measured across two evaluation runs: **3,805 name lookups lost to HTTP 429** (2,160
distinct), about **8 per question**. Consequences: 82% of intents closed, 105 unclosed
intents *had* retrieved facts and bound nothing (0.44/question), and only 58% of
questions closed every intent.

**Fixed** with local SPARQL fallbacks — public API first, local second:

- `search_entities_local` ranks label matches by outgoing-statement count as a prominence
  proxy, and **declines rather than guesses** unless the top candidate dominates the
  runner-up by ≥8×. Validated on 30 names drawn from actual 429 failures:

  | threshold | accepted | correct | wrong | precision | coverage |
  |---|---|---|---|---|---|
  | 1× | 23 | 20 | 3 | 87% | 77% |
  | 4× | 18 | 17 | 1 | 94% | 60% |
  | **8×** | **15** | **15** | **0** | **100%** | **50%** |

  Every disagreement with the API was a short ambiguous label — `ABC` (1.3×), `State`
  (1.0×), `The Book Thief` (4.9×) — while correct matches separate sharply (`Dolly
  Parton` 25.7×, most with no runner-up at all). Ordering matters: the API ranks by
  relevance and this ranks by prominence, agreeing on only 67% of top-1 overall, so
  local-*first* would be wrong ~10% of the time, and **a wrong QID is worse than no QID
  because it manufactures a false binding**.
- `get_entity_details_local` recovers labels and English descriptions by exact QID, so no
  gate is needed. Without a label, `entity_dict` entries are bare `{"qid": …}`, which
  silently disables `_known_entity_labels` — the KG fan-out gate stops firing on
  already-linked entities — and label-based binding resolution.

Live validation: **23 of 24 names (96%)** that had previously all failed with 429 now
resolve (`Mississippi River → Q1497`, `Molotov–Ribbentrop Pact → Q130796`). At scale in
the MCTS arms: **0 rate-limit failures** against 314/154/118 before, with 67/39/62 local
rescues. Tests in `tests/unit/test_wikidata_local_fallback.py`; **319 passing**.

Caveat: this changes behaviour for *both* arms, so every completed measurement above was
taken under broken entity linking. The effect on closure, conversion and accuracy was never
isolated in its own experiment; because it is symmetric across arms it does not confound any
paired contrast, but it is one reason absolute accuracy is not comparable with pre-fix runs.
See [RESULTS.md](RESULTS.md) "Limitations".

### 13.2 The 41%: closed on the wrong referent, with the rival never examined

`plan_gate` builds binding candidates **only** from `current_subanswers_concise` — one
answer per subquestion. So when the answer generator picks wrong, that single value is the
only candidate, `count_rival_referents` sees exactly one referent, and the intent closes
on the wrong value with **no contest detected** — while the correct rival sits in memory
unexamined. Two measured cases:

| question | memory contained | answered |
|---|---|---|
| county of Kimbrough Memorial Stadium | *"Canyon is the county seat of **Randall County**"* **and** *"Canyon is … in **Lubbock County**"* — two different Canyons | Lubbock |
| wife of a *Here Comes the Boom* cast member in *Grown Ups* | *"Eric Lamonsoff (Kevin James) is married to Sally Lamonsoff (**Maria Bello**)"* | Salma Hayek (Adam Sandler's on-screen wife) |

The other observed synthesis-side mechanisms, for completeness: **granularity mismatch**
(asked for a state, answered `Portugal` where memory held `Lisbon District`);
**range aggregation** (memory listed Yugoslavia 1943–1992, FRY 1992–2003, Serbia and
Montenegro 2003–2006; the answer merged them into 1943–2006); and **over-abstraction**
(memory held *"three different relationships he had in the past"* verbatim, answered *"a
former romantic partner"*).

**Deliberately not implemented yet.** Deriving rival candidates from the evidence set
rather than the single sub-answer would expose these contests — but a contested intent
does not *close*, and with `replan_max: 0` there is no repair path, so over-detection
would convert 41%-wrong-answer into more 53%-never-closed. That trade needs the
§13.1 measurement first, and a discrimination step second.

### 13.3 Order of work

1. **Measure the entity-linking fix** — it attacks the 53% bucket, is already implemented,
   and carries no over-detection risk.
2. **Then** consider evidence-derived rival candidates, which need a discrimination step
   (not merely a replan) to be worth their cost.

### 13.4 Both buckets re-measured after the linking fix — and re-decomposed

The §13.1 caveat is now discharged. Re-running the decomposition over **1,920 questions
across 16 plan-enabled runs** (a much larger sample than the original 70, and taken after
the local-SPARQL fallbacks and the new multi-provider search chain):

| | before (n=70) | **after (n=204)** |
|---|---|---|
| gold in `[Retrieval]` memory | — | **33.8%** of questions |
| conversion failures | ~15% of all questions | **10.6%** of all questions |
| never bound (intent still open) | 53% | **39%** — of which 28% had a matching memory line |
| wrong referent bound at a terminal | 41% | **38%** |
| bound the gold, synthesis lost it | 6% | **23%** |

Three corrections to §13 fall out of this, and they matter:

1. **The prize is ~10.6 points, not ~15.** Part of the original block was retrieval
   failure recorded as conversion failure, and part has already been harvested.
2. **"94% is binding" was too high.** It is **77%** (39% + 38%). Synthesis is 23%, not 6%
   — the original 6% was measured only against *top-authority findings being ignored*,
   which is a much narrower question than "synthesis lost it".
3. **Metric artifacts are not negligible inside this bucket.** `four-year` vs `four
   years`, `The Australian Ballet` vs `Australian Ballet` — the substring test scores these
   wrong. They inflate every conversion-failure count by a few percent.

### 13.5 The terminal-referent check: measured, null, defaulted off

`SELF_CORRECTOR` verifying each terminal referent against the hop's evidence before
closing — §13.2's proposed attack on the wrong-referent bucket. `vf_on` vs `vf_off`, 117
paired questions, plan enabled in both so nothing else differs:

| metric | vf_on | vf_off | test |
|---|---|---|---|
| **paired conversion** (gold in memory in *both* arms) | **14/18** | **14/18** | θ=0.500, **p = 1.0000** |
| gold in memory | 29.9% | 24.8% | — |
| conversion, marginal | 74.3% | 65.5% | — |
| accuracy | 23.08% | 18.80% | 21 discordant (13/8), p = 0.3833 |
| calls/question | +5.8 | — | Wilcoxon **p = 0.0236** (worse) |

**It moved conversion by exactly zero.** The marginal conversion gap (74.3% vs 65.5%) is
entirely a *denominator* effect: the arm retrieved more gold. And the mechanism for that is
incidental — withholding an unverified referent leaves the intent open, which buys another
hop of search. An extra hop is a cheaper way to buy an extra hop.

Cost went the wrong way by 5.8 calls/question, of which 3.03 are `SELF_CORRECTOR` itself.
The design note's estimate of "~1 extra call per question" was wrong: the gate runs **per
hop**, over a mean 3.67 hops, not once per question.

Defaulted to `false`. The code stays — the role is now wired into CoT, which it was not
before — but it fails the cost half of the goal and delivers nothing on the metric it was
built for.

### 13.6 The actual defect: a guard's answer is a truth value, not a referent

Reading the artifacts rather than the aggregates found the mechanism. The PLANNER prompt
instructs presuppositions to be hedged into conditionals — *"determine whether she had a
spouse; if so, identify who"* — so **the ledger is full of intents whose answer is yes or
no**, and every consumer of the ledger assumed a referent. Over 1,920 questions / 6,250
intents:

| | count |
|---|---|
| guard (yes/no) intents | **399** (6.4% of intents) |
| ...that are **terminal** | 188 (47% of guards) |
| ...terminal **and closed** | 131 |
| closed guards binding a **full sentence** | **139 / 284** (49%) |
| intents whose binding **echoes a prerequisite's referent** | 897 (16%) |

Three distinct harms, all from the same cause:

- **A boolean arrives at synthesis as the highest-authority candidate answer.**
  `resolved_findings` emits `f"{intent} -> {surface}"` with no shape check, and its own
  docstring records that this block is ranked above every other context source. Measured
  output: `Confirm whether the author wrote a short story featuring Herman Wouk -> No,
  Stephen King did not write a short story featuring Herman Wouk.` on a question whose gold
  was `1,335,907`. This is the *same mechanism* that made scaffolding referents cost ~4.7
  points — fixed once for scaffolding, recurring with booleans.
- **The intent binds its own input.** The only linked entity in *"No, Yangzhou is not a
  capital city"* is Yangzhou — the referent the prerequisite already bound. So the guard
  either closes on the subject, or the subject becomes a **manufactured rival** against the
  real answer. Measured: 24 intents were contested *solely* by such an echo and would
  otherwise have closed; 640 closed on nothing but an echo.
- **An affirmed referent fails to resolve at all.** `Yes, Meg Ryan.` resolved to *Dennis
  Quaid* — the subject, mentioned in the sibling candidate — because the surface was the
  whole sentence. Memory held *"Dennis Quaid is married to Meg Ryan"* verbatim and the gold
  was `Meg Ryan`; the answer returned was `Sleepless in Seattle`.

The fix is deterministic and adds **no LLM call**:

| change | effect |
|---|---|
| `is_polarity_intent` | recognises a guard from its interrogative shape |
| guard binds `pol:true` / `pol:false` | one key, so it still closes cheaply but can never contest |
| `strip_affirmation` before resolution | `Yes, Meg Ryan.` → `Meg Ryan`, which resolves |
| `resolved_findings` skips guards and sentences | a truth value is never a candidate answer |
| `ground_retrieval_query` skips guards | *"no"* never enters a retrieval query |
| `terminal_intents` excludes guards | a boolean is not one of the plan's targets |
| `plan_target_resolved` refuses a guard | a closed guard cannot end the loop |
| PLANNER prompt | a whether/if check may never be the plan's last step |

Two design points worth recording:

- **A guard always closes**, deliberately. A guard that cannot close blocks
  `plan_target_resolved` for the rest of the question, and at `replan_max: 0` that is
  permanent — buying hops that cannot help, because the guard is not what the question
  asks. A guard answered both ways records `polarity_conflict` rather than contesting.
- **`terminal_intents` returns the guards when *every* terminal is a guard** (2.8% of
  questions — a malformed plan) rather than an empty list, and `plan_target_resolved` then
  refuses to stop. The caller decides; the boolean never stands in for the answer.

One caution against reading this as settled: the *observational* accuracy split runs the
other way — questions whose plan has a terminal guard scored 33.7% vs 22.8% without. That
is confounded (hedged plans may attach to questions the model has more purchase on, and
extra terminal intents add text that a substring metric can hit), which is exactly why the
fix is behind `search.plan.guard_intents_are_not_referents` and measured as an A/B rather
than shipped on the strength of the mechanism.

### 13.7 The guard fix: mechanism confirmed, outcome null — and why

`gd_on` vs `gd_off`, 117 paired questions, concurrent arms, plan enabled in both:

| metric | gd_on | gd_off | test |
|---|---|---|---|
| **paired conversion** | 17/23 | 18/23 | θ=0.333, **p = 1.0000** |
| accuracy | 26/117 = 22.22% | 26/117 = 22.22% | 18 discordant (9/9), **p = 1.0000** |
| calls/question | −5.4 | — | Wilcoxon **p = 0.0874** (n.s.; the mean CI excludes 0, the paired test does not) |
| prompt tokens | −12,725 | — | Wilcoxon p = 0.0782 (n.s., same heavy tail) |
| intermediate-referent leak | **1.7%** | 5.2% | 8 discordant (2/6), p = 0.2891 |

The mechanism moved exactly as designed and the outcome did not move at all:

| | gd_on | gd_off |
|---|---|---|
| intents closed (first 24 rows) | 81% | 68% |
| questions where any resolved finding reaches synthesis | **48%** | 37% |
| mean resolved findings per question | **0.49** | 0.41 |

**And those last two rows are the explanation.** `candidate_answers` is the *entire*
consolidated memory — a measured 7.7–8.9 items — while `resolved_findings` is **0.49 lines
per question**. So the ledger supplies roughly **6% of what synthesis reads**, even though
the prompt ranks it first. That is a hard ceiling on every ledger-side fix, and it accounts
for three nulls in a row (terminal verification, the guard fix, and the leak reduction that
did not reach significance).

Kept on by default anyway, and labelled honestly: it is **correctness housekeeping, not a
measured improvement**. A closed intent whose referent is the string "No, Stephen King did
not write a short story featuring Herman Wouk" is wrong bookkeeping whatever the accuracy
does, it is directionally cheaper rather than more expensive, and it takes the
intermediate-referent leak from 5.2% to 1.7%.

### 13.8 Conversion is mostly a retrieval failure wearing a conversion costume

Two measurements settle what the remaining 10.6% actually is.

**Hop position cannot discriminate.** Over 104 failures where both the gold line and the
answered line are locatable in memory, the gold sits at a *later* hop 27 times, an earlier
hop 14 times, and **the same hop 63 times (61%)**. So a "prefer later hops" prompt rule
would be inapplicable to most cases and directionally right in only a quarter of them. Not
built.

**The wrong answer is usually the better-supported one.** Over 75 failures where the gold
and the answered line are distinct, scoring each line's content-word overlap with the
question:

| | share |
|---|---|
| the **answered** line matches the question better — defensible on the evidence | **47%** |
| tie | 39% |
| the **gold** line matches the question better — a genuine selection error | **15%** |

Synthesis picks the better-supported line about **three times more often** than it makes a
selection error. Memory contains the gold *string* while the surrounding evidence points
harder at a rival — two different Canyons, Richland vs Lexington County, Phoenix Raceway vs
Tucson Raceway Park. The *discriminating* fact was never retrieved.

**So the conversion block is not a bookkeeping problem and not a prompt problem.** Of the
10.6% of questions in it, roughly 15% — about **1.6 points of the total** — is a selection
error that better reasoning over existing memory could fix. The rest needs the
discriminating evidence retrieved, which is the same conclusion §12 reached from the other
direction: **retrieval is the bottleneck, and 78% of failures never had the evidence.**

This is why the conversion prize looked large in aggregate and resisted three targeted
fixes. The aggregate counted questions where the gold string happened to appear in memory;
it did not ask whether the evidence *distinguished* the gold from its rivals, and usually
it did not.

### 13.9 Two further binding hypotheses, both retired by measurement

**Low-confidence attrition is not a lost opportunity.** `plan_gate` drops a sub-answer whose
self-reported `confidence_level` is low before binding — 22–28% of all offered answers,
the largest single source of attrition. The hypothesis was that a `[Retrieval]`-corroborated
answer should override the model's own hedge. Implemented behind
`bind_corroborated_low_confidence` and measured on the live arm: the rescue fired on **1 of
85 drops**. The label and corroboration agree almost perfectly — ~99% of low-confidence
answers are *also* uncorroborated — so the gate was right and there is nothing to recover.
Left off.

**Contested-intent discrimination would address ~2% of the block.** §13.2 deferred
evidence-derived rival candidates pending a discrimination step. Sized before building: of
1,320 questions only 48 (3.6%) have a contested intent at all, and of 128 conversion failures
just **2 (2%)** do. Even generously — deriving rivals from the evidence rather than the single
sub-answer would surface a rival pair in the 59% of failures where the gold and the answered
line are both present as distinct memory lines — §13.8 already showed that in 47% of those
the evidence supports the *wrong* rival. **Detection is feasible; adjudication needs a fact
that was never retrieved.** Not built.

### 13.10 Final tally: five ledger-side fixes, five nulls, one explanation

Every measurement in this attack, all paired on 117 questions with concurrent arms:

| fix | conversion (paired) | accuracy | cost | default |
|---|---|---|---|---|
| `verify_terminal_referents` | 14/18 vs 14/18, **p=1.0000** | +4.3 pts, p=0.3833 | **+5.8 calls, p=0.0236** | **off** |
| `guard_intents_are_not_referents` | 17/23 vs 18/23, **p=1.0000** | ±0, p=1.0000 | −5.4 calls, p=0.0874 n.s. | on¹ |
| `skip_input_referent_in_binding` | 15/19 vs 15/19, **p=1.0000** | −4, p=0.4807 | flat, p=0.5427 | **off** |
| `bind_corroborated_low_confidence` | 18/22 vs 17/22, **p=1.0000** | +1, p=1.0000 | flat, p=0.2680 | **off** |
| contested-intent discrimination | not built — 2% of failures | — | — | — |

¹ correctness housekeeping only: it takes the intermediate-referent leak from 5.2% to 1.7%
and is directionally cheaper, but its outcome effect is a measured zero.

**One explanation covers all five.** `candidate_answers` is the entire consolidated memory
(7.7–8.9 items); `resolved_findings` is 0.49 lines per question. The ledger is **~6% of what
synthesis reads**. A fix that makes the ledger more correct cannot move an answer that is
being selected from the other 94%.

The corollary is the useful part. Conversion failure decomposes as:

| | share of the 10.6% block |
|---|---|
| the evidence supports the **wrong** candidate — the discriminating fact was never retrieved | ~47% |
| the evidence is ambiguous between them | ~39% |
| **a genuine selection error over adequate evidence** | **~15%** (≈1.6 points of the total) |

So the conversion block was never worth ~15 points, and after this decomposition it is worth
about **1.6** to anything short of better retrieval. §12's ranking stands unchanged and is now
supported from a second direction: **retrieval is the bottleneck** — three web results per
query, 78% of failures never holding the evidence at all.

### 13.11 A shipped improvement: sub-EM was scoring correct answers wrong

The one defect in this investigation that was both real and worth fixing turned out to be in
the *measurement*, and it was found by reading the failures rather than the aggregates.

`compute_sub_em` asks whether the gold string appears in the prediction **verbatim**. Three
gold answers in `datasets/musique_depth.jsonl` arrive wrapped, so a correctly concise answer
can never match them:

| gold | prediction | scored |
|---|---|---|
| `at the city of Cairo, Illinois` | `Cairo, Illinois` | **0** |
| `The Australian Ballet` | `Australian Ballet` | **0** |
| `four-year` | `four years` | **0** |

Measured across **27 runs / 3,240 questions**: 58 answers reclassified, **+1 to +3 in every
single run**, mean +2.15 — **+1.79 points**.

Two consequences, and the second is the one that matters:

1. Every absolute accuracy number in this document is **~1.8 points low**.
2. Those questions are scored wrong in *both* arms of every paired comparison, so they never
   enter the discordant pool. Up to 3 discordant pairs were being discarded per A/B —
   sign-test power spent on a string-matching artifact, in a project whose central difficulty
   has been insufficient power.

`compute_sub_em_relaxed` strips only the gold's determiner/preposition/`<head noun> of`
wrapper, never its content, and **refuses a one-token residue**. That guard is what separates
the two structurally identical cases: `at the city of Cairo, Illinois` leaves `cairo illinois`
and is allowed; `the state of Washington` leaves the bare `washington`, which `Washington
D.C.` would wrongly satisfy, so the strict result stands. Verified to leave every genuine
failure wrong — underspecified dates (`1929` for `11 February 1929`), ranges (`1970s` for
`From the 1950s to the 1970s`), granularity (`Latin` for `Medieval Latin`), and rival
referents (`Lexington` for `Richland County`). 18 unit tests in
`tests/unit/test_metrics_relaxed.py`.

**Recorded beside sub-EM, never in place of it** (`sub_em_short_relaxed` in the evaluation
log), so historical numbers stay comparable. Re-running every A/B in this document under the
corrected metric changes **no** sign and **no** significance verdict:

| comparison | sub-EM | relaxed |
|---|---|---|
| guard fix | 26v26, d=9/9, p=1.000 | 28v28, d=9/9, p=1.000 |
| terminal verification | 27v22, d=13/8, p=0.383 | 30v24, d=14/8, p=0.286 |
| skip input referent | 22v26, d=7/11, p=0.481 | 25v28, d=7/10, p=0.629 |
| low-confidence rescue | 27v26, d=9/8, p=1.000 | 28v28, d=9/9, p=1.000 |
| MCTS per-rollout (batch 1) | 18v12, d=8/2, p=0.109 | 18v12, d=8/2, p=0.109 |
| MCTS per-rollout (batch 2) | 12v13, d=4/5, p=1.000 | 14v14, d=5/5, p=1.000 |

### 13.12 Two more attacks sized and declined

**Discrimination retrieval.** §13.8 said the discriminating fact was never retrieved, so the
obvious move is to fetch it: on a rival pair, issue one query aimed at separating them. Probed
offline on 14 real failures before building (`scripts/discriminate_probe.py`) because a 120-row
A/B costs two hours and a probe costs minutes. Two query forms, both failed:

- `"{question} {rival_a} or {rival_b}"` returned text containing **neither** rival on 12 of
  14. A 20-word multi-hop question is a bad search query and appending "A or B" does not
  rescue it — that run measured the query, not the surface.
- fetching **each rival's own page** and scoring it against the question separated them on
  **0 of 14**.

Reading those 14 cases is what redirected the session: most are not rival referents at all
but **answer-type** errors — `Veruca Salt` (the character) for `Julie Dawn Cole` (the actor),
`United States Navy` for `Sea, Air, and Land`. Which led to §13.11.

**Mechanical precision upgrade.** For the 86 genuinely underspecified answers, widen the
answer to a superstring found in `[Retrieval]` memory. Sized before building: the exact gold
form is recoverable for **18 (21%)**, but some *other* superstring exists for **51 (59%)** —
so a naive widening fires wrongly about three times as often as rightly, for a ceiling of
~0.75 points. Not built. Note also that the granularity rule it would enforce **already
exists** in `SYNTHESIZE_FINAL_ANSWER_PROMPT` (rule 2, with the `1797` / `4 March 1797`
example) and is being violated, so restating it in the prompt is not the fix either.

### 13.13 The 94% channel tested directly — also null

Every fix in §13.10 acted on the ledger, which is ~6% of the synthesiser's input. The obvious
objection is that the ledger was never the right lever, so the ordering of
`candidate_answers` — the whole of `text_memory`, and the other 94% — was tested too.

**Choosing the ordering.** Seven candidate signals were scored on 149 conversion failures
where the gold-bearing memory line and the line the answer came from are distinct:

| signal | gold | rival | p | |
|---|---|---|---|---|
| memory position later | 101 | 48 | <0.0001 | favours gold |
| hop later | 60 | 28 | 0.0008 | favours gold |
| `[Retrieval]` provenance | 6 | 0 | 0.0312 | favours gold |
| line shorter | 72 | 77 | 0.7433 | nothing |
| contains a date | 20 | 16 | 0.6177 | nothing |
| idf-weighted question overlap | 35 | 78 | 0.0001 | **favours rival** |
| question content-word overlap | 28 | 71 | <0.0001 | **favours rival** |

The last two are worth stating plainly: **relevance ranking would have made conversion
worse.** The wrong candidate is usually the one that looks more like the question — which is
also why it was chosen.

**Two checks that tempered the winning signal**, both run before shipping:

- Length-matching. A short prediction (`1929`) matches an early line more readily than a
  longer gold string does. Requiring ≥2 content tokens on both sides takes the result from
  101/48 at p<0.0001 to **67/41 at p=0.0157**, displacement 0.089 of the list — under one
  position in an ~8-item list.
- An *unconditioned* comparison **disagrees in sign**: across all questions, the gold sits at
  mean relative position 0.425 against the wrong answer's 0.450.

**Result.** `ro_on` vs `ro_off`, 117 paired questions, plan disabled in both so nothing else
differs:

| metric | ro_on | ro_off | test |
|---|---|---|---|
| **paired conversion** | 18/26 = 69.2% | 21/26 = 80.8% | gap **−3**, θ=0.286, p=0.4531 |
| accuracy | 22.22% | 23.93% | 22 discordant (10/12), p=0.8318 |

Null, and directionally **negative** — the direction the unconditioned check had pointed to.
Left off. The positional asymmetry is real but too small to act on, and the synthesiser is not
meaningfully primacy-biased over a list this short.

**So the conclusion of §13.10 now rests on both channels, not one.** Seven interventions:
four on the ledger, one on the ordering of the 94% channel, one on discrimination retrieval,
one on precision — all null. Only the measurement fix (§13.11) was real. Conversion is bound
by what the evidence *distinguishes*, not by how the system bookkeeps or orders it.

### 13.14 Consolidation loss — the channel that was actually missing

§13.13 tested the ledger (6% of the synthesiser's input) and the ordering of
`candidate_answers` (the other 94%). Both null. Neither touched material that **never
reached synthesis at all**.

`extracted_facts` is cleared by `increment` every hop, so the pre-consolidation evidence was
unobservable after a run and this question had never been asked. A new append-only
`retrieval_log` channel (persisted as `retrieval_log.json` per question) makes it answerable.
Measured on 60 instrumented questions, plan disabled (`results/cl_probe`):

| | |
|---|---|
| gold in **retrieved** facts (pre-consolidation) | 22/60 = **36.7%** |
| gold in **consolidated** `text_memory` | 19/60 = **31.7%** |
| **lost by consolidation** | **6 = 10.0% of all questions** |
| of those 6, answered wrong | **6 — all of them** |

**27% of the questions whose retrieval found the gold lose it before synthesis sees it, and
none recovered.** `MEMORY_CONSOLIDATOR` keeps a mean **0.56** of the retrieved facts — as low
as **0.08** on individual questions, where 24 facts became 2 memory items — and it makes those
retention decisions per hop, under four removal rules, without knowing which fact the final
answer will need. The six losses are ordinary multi-hop answers, not noise:
`University of South Carolina` → answered `Clemson University`; `Mario Andretti` → `Nigel
Mansell`; `English Channel` → `Doggerland`.

This also reconciles a discrepancy: §6.2 measured gold-reaching-memory at 41.43%, and the
recent runs sit near 30%. Part of that gap is consolidation, and part is that the runs since
the multi-provider search chain landed are hitting Wikipedia 429s — worth separating, and not
yet done.

`dropped_evidence` surfaces **only** the discarded facts to synthesis (capped at 25, most
recent first, appended **last** under an explicit lower-reliability label — the scaffolding
result established that content at top authority gets returned as the answer whether or not it
deserves to be). Kept facts are excluded because they are already in `candidate_answers`.
Paraphrase-tolerant at 0.8 content-token overlap, since the consolidator rewrites and an exact
test would call almost everything dropped.

**Status: ADJUDICATED IN [§14.13](#1413-dropped-evidence-the-outstanding-experiment-finally-adjudicated--and-it-is-a-null)
— it is a null, and the hypothesis below is refuted.** On 117 pairs, accuracy -2.56 pts
(p = 0.6072) and, on the 22 questions where both arms held the gold, conversion is *identical*
at 15/22 each (p = 1.0000). Read §14.13 rather than this section's expectations.

The original run (`de_on`/`de_off`) died at 16/120 when the LLM host `n0142` disappeared from
DNS mid-run — an infrastructure failure, not a result. Two things were flagged to watch, both
recorded here because the first turned out to be wrong:

- `dropped_evidence` is resurfacing **13.9 facts/question**, more than the ~8 consolidated
  items. That is above the ~44% implied by the 0.56 retention ratio, so the overlap test is
  under-detecting paraphrased survivors and paying some redundancy. Bounded by the cap.
- The conversion **denominator must not move**: nothing is added to `text_memory`, so
  "gold in memory" is unchanged and any recovered question shows up as accuracy rather than
  inflating the denominator. If the denominator does move, the measurement is wrong.

## 14. Reasoning on: what changes, and what does not

Every result above this section was measured with **reasoning off**. `config.eval.yaml` pins
`enable_thinking: false`, but on Bedrock that field is inert either way — it rides in
`chat_template_kwargs`, which only a self-hosted SGLang chat template reads. Bedrock's actual
switch is `additionalModelRequestFields={"reasoning_effort": "high"}`, now exposed as
`TierConfig.reasoning_effort`.

### 14.1 Two ways this experiment could have measured nothing

Both were found by probing rather than by reading documentation, and both are silent:

- **LiteLLM drops `reasoning_effort` unless it is allow-listed.** Passing it alone measured
  byte-identical to reasoning-off — 30 output tokens, no reasoning block. It needs
  `allowed_openai_params=["reasoning_effort"]` alongside it. Without that the "on" arm *is*
  the off arm, and the contrast reads as a clean null.
- **Only `"high"` engages the model.** The Bedrock gateway advertises
  high/low/max/medium/minimal/none/xhigh, but that is the gateway's enum, not qwen3-32b's:
  none/low/medium return 4-6 output tokens with no reasoning, minimal/xhigh fail in the
  backend, max is rejected. Anything but `high` looks enabled and measures as disabled.

The Anthropic-style `{"thinking": {"type": "enabled", ...}}` block is also inert here: it
returns 200 and does nothing. It is not a fallback signal.

Because these fail quietly, `scripts/reason_report.py` refuses to report a contrast without
first printing an **arm-validity block**: what fraction of completions actually returned
reasoning, how many were truncated, and whether the two arms differ in any config key other
than the plan flag. Measured on the arms below: reasoning fired on **99.6-99.9%** of
completions, truncation **0.02-0.05%**, and the ablation check passes.

### 14.2 The token budget is a trap, and it was measured, not guessed

Reasoning tokens are billed **inside** `max_tokens`, arriving as
`usage.completion_tokens_details.reasoning_tokens`. So an over-long trace truncates the JSON
payload rather than the trace, the role falls back to `build_safe_default_output`, and the arm
loses accuracy for a budget reason that has nothing to do with reasoning quality.
`thinking_budget` cannot prevent this — its logit processor needs an SGLang server — yet
`build_request_kwargs` still returns a dill-pickled blob for the Bedrock model name, so it
would ship an unusable param. Only `max_tokens` binds.

A calibration pair on the same 6 questions:

| `max_tokens` | truncated completions | where |
|---|---|---|
| 4096 (stock) | 2 = **1.17%** | `open_ie`, **2/13 = 15%** of its completions |
| **8192** | **0 = 0.00%** | — |

So the raise was necessary, not precautionary. Cost of reasoning on this workload:
**~19.5k reasoning tokens per question, 77% of all output tokens.**

### 14.3 CoT, plan on vs off, reasoning on: a null

120 questions on `musique_depth.jsonl`, both arms concurrent, 117 pairs after 3 rows collapse
on duplicate normalised question keys (identical in both arms, so unbiased).

| | relaxed sub-EM | |
|---|---|---|
| plan on | 27/117 | 23.08% |
| plan off | 25/117 | 21.37% |
| **gap** | **+2** | **+1.71 pts** |

discordant 18 (10/8), sign test **p = 0.8145**, bootstrap CI [-0.051, +0.086]. Every cost
metric is also null: calls -0.9 (p = 0.32), prompt tokens -5,510 (p = 0.96), reasoning tokens
+1,258 (p = 0.44).

This matches the reasoning-off verdict (`lab_*`: +5/234, p = 0.5515) — **reasoning does not
change the plan's value.** The plan is a null with reasoning and a null without it.

### 14.4 A retracted intermediate reading, recorded because it is the methodological point

At 32 completed pairs the same contrast read **-12.50 pts** (9/32 vs 13/32), and a 2x2 against
`lab_*` on those same 32 questions appeared to show a clean interaction — reasoning helping the
no-plan arm (+18.7 pts) while the plan arm lost ground — supporting a tidy story in which plan
and reasoning are substitutes, both imposing structure on decomposition.

At 117 pairs that became **+1.71 pts**. The interaction was sampling noise and the hypothesis
is withdrawn. This is the same 12%-of-rows-flip noise floor that governs everything else in
this project, and 32 rows is well inside it. Recorded rather than deleted because the reading
was *interesting*, which is exactly when a partial result is most likely to be believed.

### 14.5 Reasoning is directionally worse, and the axis is confounded

On the 117 questions common to all six arms:

| | reasoning OFF (`lab_*` pooled, n=234) | reasoning ON (n=117) |
|---|---|---|
| plan | 27.78% | 23.08% |
| no-plan | 25.64% | 21.37% |

All four paired comparisons are negative (-4.3 to -5.1 pts) and **none is significant**
(p = 0.38-0.46). This axis is **not a clean contrast** and must not be quoted as one: `lab_*`
ran on older code *and* in an older retrieval regime. **Settled in §14.14** with a reasoning-off
pair on current code, which reproduces the direction (-2.56 and -4.27 pts) without the confound.
Read §14.14, not this table.

### 14.6 Where the loss is: retrieval, not conversion

Using the `retrieval_log` channel, which separates "never retrieved" from "retrieved and
discarded":

| | gold retrieved | gold in memory | lost to consolidation |
|---|---|---|---|
| reasoning OFF, `cl_probe` (n=60) | 36.7% | 31.7% | 10.0% |
| reasoning OFF, `de_on` (n=16, current code) | 43.8% | 43.8% | 6.2% |
| **reasoning ON** (n=117) | **29.1%** | **25.6%** | 9.4% |

Conversion is **unchanged** — 80.0% against 80.6% marginal, and the paired reading favours
reasoning-on (83.3% vs 72.2%). Reasoning does not mishandle evidence it holds; it retrieves
less of it. Two accompanying distributional shifts, both large:

- **Consolidated memory items: 3.5 against 6.1.** Not code drift: ten reasoning-off runs on
  current code sit in a tight 5.62-6.68 band (older code 7.12-7.92), and reasoning-on is far
  outside it at 3.00-3.71.
- **Hops: 2.69 against 3.33-3.65**, driven by early termination — reasoning-on stops after a
  single hop on **23%** of questions against **0-5%** for every reasoning-off run.

### 14.7 What the hop shift does NOT license

The obvious inference — reasoning declares the question answerable too early, truncating the
multi-hop chain — **is not supported by this data**. Split by hop count, the 1-hop questions
score *above* the run average:

| | 1 hop | 2-3 hops | 4+ hops |
|---|---|---|---|
| reasoning ON | **32.0%** (23% of qs) | 35.6% | 7.5% |
| `lab_plan_s1` | 25.0% (3%) | 40.0% | 13.2% |

So early stopping is **selection, not damage**: the model stops early on questions it can
actually answer, and the questions that run to 4+ hops are the ones that were never going to
converge (7.5-13.2% in every regime). Attributing the retrieval loss to early termination
would be reading a confound. What survives is the localisation — retrieval, not conversion —
and the fact that reasoning buys none of the accuracy it is charged 77% of output tokens for.

### 14.8 MCTS, plan-rollout on vs off, reasoning on: the reasoning-off verdict replicates

120 questions on `musique_depth.jsonl`, both arms concurrent at `max_concurrent=4`, 117 pairs.
`mcts_plan_scope=rollout` set explicitly rather than relying on the default.

| | relaxed sub-EM | |
|---|---|---|
| plan-rollout on | 20/117 | 17.09% |
| plan-rollout off | 22/117 | 18.80% |
| **gap** | **-2** | **-1.71 pts** |

discordant 14 (6/8), sign test **p = 0.7905**, CI [-0.077, +0.043]. **Accuracy: null.**

Cost, paired per question — and here the plan-rollout wins:

| metric | mean | median | Wilcoxon |
|---|---|---|---|
| **LLM calls** | **-3.9** | -3.0 | **p = 0.0274** |
| **prompt tokens** | **-15,023** | -13,402 | **p = 0.0049** |
| completions | -3.9 | -3.0 | p = 0.0509 |
| completion tokens | -3,058 | -3,998 | p = 0.0952 |
| reasoning tokens | -2,003 | -2,656 | p = 0.1625 |

Side by side with the reasoning-off measurement (`mx_*` + `mx2_*`, n = 118):

| | reasoning OFF | reasoning ON |
|---|---|---|
| accuracy gap | +4.24 pts, p = 0.3593 | -1.71 pts, p = 0.7905 |
| calls | -36.2, p < 0.0001 | -3.9, p = 0.0274 |
| prompt tokens | -69,756, p = 0.0023 | -15,023, p = 0.0049 |

The **qualitative conclusion replicates exactly**: accuracy is a null, the cost reduction is
real and survives Wilcoxon. What changes is the *magnitude* — the saving shrinks about 9x,
because the baseline it is saving against shrank. Reasoning-on MCTS spends **53 completions per
question against the historical 127**, the same early-termination shift seen in CoT (§14.6), so
there is far less duplicated work left for a shared rollout plan to remove.

### 14.9 Answer to the question this section was opened to settle

**Reasoning changes neither verdict.**

| contrast | reasoning OFF | reasoning ON |
|---|---|---|
| CoT, plan on vs off | null (+2/234, p = 0.5515) | **null (+2/117, p = 0.8145)** |
| MCTS, plan-rollout on vs off | accuracy null; calls -36.2 | **accuracy null; calls -3.9 (p = 0.027)** |

Both plan contrasts were nulls on accuracy with reasoning off, and both remain nulls with
reasoning on. The MCTS cost saving survives in both regimes. This is now the **sixth and
seventh** well-powered null on plan accuracy in this project, and they are the first two
measured under a different inference regime — which is worth more than another repeat at the
same settings, because it rules out "the plan would help a stronger reasoner" as an
explanation for the earlier nulls.

Reasoning is also not itself an improvement: it costs 77% of output tokens and is directionally
negative on accuracy in both arms. The honest summary is that on this workload it buys earlier
termination and less retrieved evidence, at ~4x the output-token price.

### 14.10 Selective reasoning: the one intervention here that is worth keeping

§14.6 localised the harm of reasoning to *retrieval*, and to two roles in particular: memory
consolidation (items 6.1 -> 3.5) and `open_ie`, the only role to truncate at the stock budget.
Neither reasons over evidence — one compresses, the other extracts. So reasoning was left on
the roles that actually reason and taken off the rest.

Reasoning is a **per-tier** setting, so this needs no code change. Keep it on `heavy`
(`subquestion_generator`, `answer_generator`, `self_corrector`, `final_answer_synthesizer`,
`web_researcher`) and `plan` (`planner`); leave `medium`/`light`/`classify` off. That leaves
`memory_consolidation`, which sits on `heavy` and would keep reasoning, so it is remapped to
`medium` — the tiers are all aliases of one Bedrock tier here, so the remap changes reasoning
and nothing else.

`r_cot_selective` vs `r_cot_plan_off`, 117 pairs, plan off in both:

| | full reasoning | selective | |
|---|---|---|---|
| relaxed sub-EM | 25/117 = 21.37% | **26/117 = 22.22%** | +0.85 pts, p = 1.0000 |
| reasoning tokens / q | 23,416 | **9,443** | **-13,974, p < 0.0001** |
| output tokens / q | 29,817 | **16,526** | -13,291, p < 0.0001 |
| LLM calls / q | 29.1 | 26.2 | -2.9, p = 0.0079 |
| prompt tokens / q | — | — | -7,171, p = 0.0425 |

**A 59.7% cut in reasoning spend and 44.6% in output tokens, at no accuracy cost**, with every
cost metric surviving Wilcoxon. This is the only intervention in this section that improves
anything.

The predicted mechanism also holds, which is what distinguishes this from a lucky cost result:

| | mem items | retrieved facts | gold retrieved | gold in memory |
|---|---|---|---|---|
| full reasoning | 3.71 | 11.71 | 29.1% | 25.6% |
| **selective** | **5.36** | **13.71** | **35.0%** | **29.9%** |
| reasoning off (`cl_probe` / `de_on`) | 5.88 / 6.00 | 16.97 / 14.75 | 36.7% / 43.8% | 31.7% / 43.8% |

Selective reasoning returns memory occupancy and retrieved gold to roughly the reasoning-off
band. Note the asymmetry: gold-in-memory recovers +4.3 points but accuracy only +0.85, because
conversion — not retrieval — is the binding constraint on those particular questions, which is
consistent with everything in §13.

Two caveats, both checked rather than assumed. The arms differ in `max_tokens` on the
reasoning-off tiers (4096 vs 8192), which is inert here: truncation is 0.05% against 0.03%,
`open_ie` 2/227, so the ceiling never bound. And `reasoning_effort` fired on 60.7% of
completions by design, not by accident — `scripts/reason_report.py` now distinguishes an
intentionally-partial arm from a silently-mixed one, and scopes the ablation check by contrast
type, because judging a reasoning contrast by a plan ablation's rules reports a correct arm as
broken.

**Recommendation.** Reasoning stays off by default: it buys no accuracy and costs 4x the output
tokens. If it is enabled for other reasons, enable it *selectively* — the same accuracy is
available for 40% of the reasoning spend.

### 14.11 The reasoning-on noise floor, and what it does to every number above

Every flip estimate in this project came from reasoning-off repeats. `r_cot_plan_off` and
`r_cot_plan_off_s2` are the same measurement under reasoning: identical config, launched
concurrently, 117 paired questions.

| | relaxed sub-EM |
|---|---|
| repeat 1 | 25/117 = 21.37% |
| repeat 2 | 26/117 = 22.22% |
| absolute gap | **1 row (0.85 pts)** |
| **rows that flipped** | **17/117 = 14.5%** (8/9) |

(The two configs differ only in the `memory_consolidator` -> `memory_consolidation` key rename
of §14.12, which is behaviourally identical: both resolve the role to `heavy`, one explicitly
and one through `_get_tier`'s fallback.)

So the reasoning-on flip rate is **14.5%**, slightly above the 12% measured with reasoning off.
Set the session's effects against it:

| effect | rows | points | p |
|---|---|---|---|
| noise floor, identical configs | 1 | 0.85 | 1.0000 |
| CoT, plan on vs off | +2 | +1.71 | 0.8145 |
| MCTS, plan-rollout on vs off | -2 | -1.71 | 0.7905 |
| selective vs full reasoning | +1 | +0.85 | 1.0000 |

**Every accuracy effect measured in this section is the same size as the gap between two runs
of the same configuration.** That is the correct way to read the nulls: not "the plan does
nothing" in some strong sense, but "at n = 117 this design resolves effects of roughly ±5 points
and every candidate came in under ±2". It is also why the §14.4 retraction was inevitable in
hindsight — a 4-point reading at n = 32 is well inside a band this wide.

The corollary for the cost results is the opposite, and worth stating because it is easy to
lose: the cost effects are **10 to 100 times** their own noise and survive Wilcoxon
(selective reasoning -13,974 reasoning tokens at p < 0.0001; MCTS plan-rollout -3.9 calls at
p = 0.0274). Cost is where this system is measurable; accuracy, at this sample size, is not.

### 14.12 A dead config key found on the way

`config.eval.yaml` mapped `memory_consolidator: heavy`, but the registered Role name is
`memory_consolidation` (`MEMORY_CONSOLIDATOR` is the module constant, not the role's `.name`).
So the entry matched nothing and the role reached `heavy` only through `_get_tier`'s
unknown-role fallback. Harmless in effect — the fallback is the same tier — but any future
attempt to remap that role by editing the misspelled key would silently have done nothing,
which is exactly the sort of thing that costs a day. Fixed; `config.yaml` already spelled it
correctly.

### 14.13 Dropped evidence: the outstanding experiment, finally adjudicated — and it is a null

§13.13 left this measured-but-unadjudicated: `de_on`/`de_off` died at 16/120. Re-run here as
`rd_on` against `r_cot_selective`, identical in every respect except
`synthesis_sees_dropped_evidence`, both under selective reasoning (the regime worth improving).

| | relaxed sub-EM | |
|---|---|---|
| dropped evidence shown | 23/117 = 19.66% | |
| control | 26/117 = 22.22% | **-3 rows, -2.56 pts, p = 0.6072** |

Cost behaved exactly as predicted, which is at least evidence the mechanism was wired
correctly: prompt tokens +10,378 (p = 0.15), calls -0.2, completions -0.0 — one longer context
on one call, and nothing else.

**Conversion, which is the metric this intervention actually targets:**

| | gold in memory | conversion (marginal) | conversion (paired) |
|---|---|---|---|
| `rd_on` | 29/117 = 24.8% | 20/29 = 69.0% | **15/22 = 68.2%** |
| control | 35/117 = 29.9% | 22/35 = 62.9% | **15/22 = 68.2%** |

On the 22 questions where **both** arms held the gold, the two arms convert *identically* —
15/22 each, discordant 4 (2/2), p = 1.0000. Re-surfacing the discarded facts changed nothing.

So the hypothesis of §13.13 is **refuted**. The gold was retrieved, consolidation threw it
away, and putting it back in front of the synthesiser did not recover it. The most likely reason
is the one already established for scaffolding in §8: position and authority dominate. The
dropped facts are appended last under an explicit *lower-reliability* label, which is what makes
them safe, and apparently also what makes them inert.

**A correction to this project's own method.** This experiment's launcher asserted that the conversion
denominator *must not move*, since nothing is added to `text_memory` — and offered that as a
validity check. The denominator moved anyway, by 5.1 points. The assertion was wrong: retrieval
is nondeterministic (different sampled subquestions, different live web results), so
gold-in-memory varies run to run regardless of an intervention that provably cannot touch it —
the reasoning-on noise floor of §14.11 is 14.5% of rows. The *marginal* conversion comparison is
therefore not trustworthy across runs, and only the **paired** reading — same question, both
arms holding the gold — supports a conclusion. That is what conversion_report.py reports both
for, and this is the case that shows why.

### 14.14 The clean 2x2: same code, same day, reasoning as the only cross-arm difference

§14.5 could only compare reasoning-on against `lab_*`, which ran on older code in an older
retrieval regime, and said so. `r_cot_plan_{on,off}_noreason` remove that confound: identical
code, same day, same two-arm concurrency, reasoning off. 117 questions common to all six arms.

| | plan ON | plan OFF |
|---|---|---|
| **reasoning ON** | 27/117 = 23.08% | 25/117 = 21.37% |
| **reasoning OFF** | **32/117 = 27.35%** | 28/117 = 23.93% |

| for scale | |
|---|---|
| reasoning ON, plan OFF, repeat 2 | 26/117 = 22.22% |
| selective reasoning, plan OFF | 26/117 = 22.22% |

**The reasoning axis** (paired per question, positive = reasoning helps):

| comparison | gap | discordant | p |
|---|---|---|---|
| plan off | **-2.56 pts** | 25 (11/14) | 0.6900 |
| plan on | **-4.27 pts** | 17 (6/11) | 0.3323 |
| selective vs reasoning off | -1.71 pts | 20 (9/11) | 0.8238 |

**The plan axis** (positive = the plan helps):

| comparison | gap | discordant | p |
|---|---|---|---|
| reasoning ON | +1.71 pts | 18 (10/8) | 0.8145 |
| reasoning OFF | +3.42 pts | 14 (9/5) | 0.4240 |

**Noise floor**, identical configs: -0.85 pts, discordant 17 (8/9), p = 1.0000.

Four things follow, and the fourth is the one that matters most:

1. **Reasoning does not help, in either arm.** Every comparison is negative, -1.71 to -4.27
   points, now with the code and retrieval-regime confounds removed. None is significant, so
   the claim is *directional*: reasoning is not an improvement on this workload, and it costs
   ~19.5k extra output tokens per question to not be one.
2. **The plan does not help either, in either regime.** +1.71 and +3.42 points, p = 0.81 and
   p = 0.42. Consistently positive across four independent measurements now, never significant.
3. **There is no interaction.** The plan is mildly positive with and without reasoning; reasoning
   is mildly negative with and without the plan. The §14.4 substitutes hypothesis is dead on the
   full sample, not merely unsupported.
4. **Every accuracy cell is inside the noise floor.** The floor is 17 discordant rows between
   identical configs; the effects sit on 14-25 discordant rows. At n = 117 this design cannot
   resolve anything smaller than roughly ±5 points, and nothing measured here exceeds ±4.3. The
   correct reading is not "these interventions do nothing" but "if any of them does something,
   it is smaller than this experiment can see" — and the way to change that is more questions,
   not more interventions.

The cost side is the opposite in every respect, which is the actionable half of this section:

| | effect | p |
|---|---|---|
| plan, reasoning OFF | **-24.8 calls**, -36,993 prompt tokens | < 0.0001, 0.0179 |
| plan-rollout, MCTS reasoning ON | -3.9 calls, -15,023 prompt tokens | 0.0274, 0.0049 |
| selective vs full reasoning | **-13,974 reasoning tokens** | < 0.0001 |

Cost effects here are 10-100x their own noise and survive Wilcoxon. Accuracy effects are not
distinguishable from re-running the same configuration. **Any claim this project makes should be
a cost claim.**

### 14.15 Why reasoning costs accuracy: it trades recall for precision in a recall-bound pipeline

§14.6 localised the loss to retrieval, but its evidence was cross-regime (against `cl_probe`
and `de_on`). The `*_noreason` arms allow the same comparison on **identical code with reasoning
as the only difference**, and the mechanism turns out to be far larger and cleaner than the
accuracy effect it produces:

| plan off | facts extracted | memory items |
|---|---|---|
| reasoning ON | 11.71 | 3.71 |
| reasoning OFF | **18.81** | **6.50** |

| plan on | facts extracted | memory items | hops | 1-hop stops |
|---|---|---|---|---|
| reasoning ON | 11.51 | 3.29 | 2.69 | 23% |
| reasoning OFF | **18.73** | **7.00** | 3.51 | 3% |

**Reasoning extracts ~38% fewer facts and retains ~50% fewer memory items**, reproduced across
two independent arm pairs. Unlike every accuracy number in this section, these are not inside the
noise floor — they are 40-50% relative effects.

**The explanation that fits: reasoning makes the model more selective, and this pipeline is
recall-bound.** `EXTRACTOR`, `OPEN_IE` and `MEMORY_CONSOLIDATOR` are recall tasks with an
asymmetric cost structure — a spurious fact is cheap, because it merely occupies a memory slot,
whereas a dropped fact is unrecoverable. Reasoning improves *judgment*, and judgment applied to
"is this fact relevant?" yields tighter filtering, i.e. it optimises precision where the binding
constraint is recall. `MEMORY_CONSOLIDATOR` compounds this by deciding per hop, without knowing
which fact the final answer will need, so sharper *local* pruning is worse *globally*.

Two independent results support that chain over a mere correlation:

- **The intervention reverses it.** Taking reasoning off exactly those roles (§14.10) restored
  memory items 3.71 -> 5.36 and extracted facts 11.71 -> 13.71 at no accuracy cost, while leaving
  it on the roles that genuinely reason over evidence.
- **The drops are unrecoverable, so they cost something.** §14.13 re-showed the discarded facts
  to the synthesiser and conversion moved by *exactly zero* (15/22 in both arms).

**Ruled out:** the rival explanation that reasoning simply consumed the output budget and
truncated payloads. Truncation measured 0.02-0.05%.

**And the asymmetry is the interesting part.** Evidence volume falls ~50% while accuracy falls
only 2.6-4.3 points and conversion stays flat at ~80%. So most of what reasoning pruned really
was redundant — it prunes *well*, and occasionally prunes the gold. That is simultaneously why
the harm is real and why it is small enough to hide inside a 14.5% noise floor.
