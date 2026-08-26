# How CoT and MCTS actually run, and where the plan enters

Real output, not a sketch. Everything below is captured from two live runs of

> **Who was the father of the father of computer science?** — gold `Julius Mathison Turing`
> (`datasets/bamboogle_hardmix.jsonl`, category `v5_wrong_unrun`)

against local QLever Wikidata + web search, Qwen3-32B on Bedrock, `search.plan.enabled=true`.
Reproduce with:

```bash
source /home/ubuntu/wemg/.venv/bin/activate
python -m langgraph_coe.scripts.smoke_test --strategy cot  --questions 1 --dump /tmp/trees
python -m langgraph_coe.scripts.smoke_test --strategy mcts --questions 1 --dump /tmp/trees
```

---

## 1. The first thing to know: CoT has no tree

| | CoT | MCTS |
|---|---|---|
| shape | **linear loop** | **search tree** |
| `state["tree"]` | `{}` — empty, always | 17 nodes on this run |
| trajectory record | `iteration_history` (1 entry/hop) | the tree itself + `current_path` |
| what repeats | one hop: decompose → retrieve → answer → consolidate | select → expand → simulate → evaluate → backprop |

So "the CoT tree" doesn't exist. CoT's structure is a **sequence of hops**, each of which
fans out *within* the hop (one retrieval per subquestion, in parallel) but never branches
across hops. MCTS is the one that branches — and its `simulate` step runs a whole CoT loop
as a rollout, which is why the two are described together.

---

## 2. CoT — the plan conditions each hop

```
START
  │
  ├─ gen_plan ..................... ONE PLANNER call. Writes `plan` (prose) + `plan_ledger`.
  │                                 Skipped entirely if `plan` is already set (MCTS rollouts).
  ▼
  gen_subq  ◀════════════════════╗  Prompt gets: question + memory + PLAN + intermediate_answer
  │                              ║  (n=3 completions, pooled)
  ├─ route ──▶ gen_final ──▶ END  ║  answerable, or depth ≥ max_depth
  ▼                              ║
  fan out per subquestion        ║  corpus (gated) + Wikidata KG (gated) + web
  ▼                              ║
  rerank → extract_relevant      ║  EXTRACTOR → atomic facts
  ▼                              ║
  gen_subanswers                 ║  one ANSWER_GENERATOR call per subquestion
  ▼                              ║
  mem_update                     ║  MemoryUpdateGraph → consolidated memory + `retractions`
  ▼                              ║
  plan_gate ...................... DETERMINISTIC, no LLM. Binds referents, classifies.
  │                              ║
  ├─ replan ─────────────────────╢  only if armed + signalled + headroom  (ONE PLANNER call)
  ▼                              ║
  increment ═════════════════════╝  bump depth, clear per-hop scratch
```

### What the run produced

**Plan** (`plan_version=1`, 2 intents):

```
To answer the question, we must first identify who is referred to as the
'father of computer science,' then determine the identity of their father.
```

Note what the plan does **not** say. It never names Turing — that is the referent
discipline: the plan may name an entity only as the question names it, or from a
`[Retrieval]`-tagged memory item. It says *"identify who is referred to as"*, an
interrogative with no truth value.

**Ledger after hop 0:**

```
intent[0] closed  closed_at=0   Establish which person is called the father of computer science.
    bound Q7251        hop=0    "Alan Mathison Turing"
    tried n_facts=5             "Who is commonly referred to as the father of computer science?"

intent[1] closed  closed_at=0   Determine the identity of the father of the person identified …
    bound Q20895930    hop=0    "Julius Mathison Turing"
    tried n_facts=5             "Who was the father of the person identified as the father of …"
```

**Trajectory** (`iteration_history`, hop 0, `plan_action=update`):

```
Q: Who is commonly referred to as the father of computer science?
A: Alan Mathison Turing is commonly referred to as the father of computer science…

Q: Who was the father of the person identified as the father of computer science?
A: The father of Alan Mathison Turing … was Julius Mathison Turing.
```

**Gate decision:**

```
hop=0  action=update  reason=update  seen=2  unattributed=0  low_confidence=0  armed=True
```

**Memory** (note the provenance tags — the plan is *absent* from all of it):

```
[hop=1] [Retrieval]: Alan Mathison Turing is widely considered … theoretical computer science.
[hop=1] [Retrieval]: Alan Mathison Turing is often considered … modern computer science.
[hop=1] [Retrieval]: Charles Babbage is considered by some to merit the title 'father of the computer'.
[hop=1] [Retrieval]: The father of Alan Mathison Turing was Julius Mathison Turing.
```

Answer: **`Julius Mathison Turing`** ✓

### The five places the plan touches the run

1. **`gen_plan`** writes it, once. Inherited rather than regenerated if already present.
2. **`gen_subq`** receives it in a *dedicated* `plan` field — never inside `context`, because
   the generator is instructed to resolve conflicts found in the context, and an
   interrogative there produces a subquestion *about the plan*.
   Rendered **last** in the prompt so the input guard's head/tail trim eats mid-memory, not
   the plan. Intents are annotated `[resolved]` / `[open]` / `[ambiguous]` — statuses only,
   never bound values.
3. **`intermediate_answer`** — this is the whole of UPDATE, and the plan prose is
   **never** enriched with the retrieved value (see §4.10 of the status doc: a bound
   referent in the plan channel would be a world-claim with no provenance tag, no hop
   tag and no eviction path). The binding travels through this typed slot instead. When intent[0] closed on
   `Alan Mathison Turing`, the next `gen_subq` call received that binding through this typed
   slot, whose prompt rule (*"do NOT re-ask what was already resolved"*) already existed and
   had **zero producers** before this change.
4. **`plan_gate`** binds referents and classifies. No LLM call.
5. **`replan`** rewrites the failed part — one PLANNER call, and only when armed.

And one place it deliberately **does not** touch: `FINAL_ANSWER_SYNTHESIZER`. At synthesis
the only question is which candidate is true; an unmet intent sitting next to candidate
answers invites treating a correct answer as deficient.

---

## 3. MCTS — one plan, shared by the whole tree

```
START
  │
  ├─ gen_plan ..................... ONE PLANNER call for the whole search.
  ▼                                 Seeds the root snapshot when branch_local_memory=true.
  select  ◀══════════════════════╗  pUCT traversal root → leaf
  ▼                              ║
  expand                         ║  resolve_snapshot(path) → this branch's memory + PLAN
  │                              ║  _reverify_memory, then by leaf type:
  │                              ║    USER_QUESTION  → _gen_subqa [+ _gen_final]
  │                              ║    SUB_QA         → _gen_subqa + _gen_self_correct
  │                              ║    SELF_CORRECTED → _gen_subqa
  ▼                              ║
  simulate ....................... runs the WHOLE CoT graph as a rollout,
  │                              ║  now carrying plan + plan_frozen=True
  ▼                              ║
  evaluate                       ║  3 VERIFIER views → reward ∈ [-1, 1]
  ▼                              ║
  backprop                       ║  visits/value along current_path
  ▼                              ║
  mem_update                     ║  consolidate; commit branch snapshot
  ▼                              ║
  plan_gate ...................... log-only here (no replan edge — see below)
  │                              ║
  └─ route ──▶ synthesize ──▶ END ╝  else loop to select
```

### The real tree from this run (17 nodes, 2 iterations)

`v` = visits, `Q` = value/visits, `*` = on the final `current_path`.

```
ROOT      v=2 Q=+0.73 * Q: Who was the father of the father of computer science?
|-- SUB_QA    v=1 Q=+0.83   Who is commonly referred to as the father of computer [...]
|   `-- SUB_QA    v=1 Q=+0.83   Who is commonly referred to as the father of computer [...]
|       `-- SUB_QA    v=1 Q=+0.83   Who was the father of the person identified as the [...]
|           `-- SUB_QA    v=1 Q=+0.83   Who is widely recognized as the father of computer [...]
|               `-- FINAL     v=1 Q=+0.83   ANS: The father of the father of computer science was [...]
`-- SUB_QA    v=1 Q=+0.63 * Who was the father of the person identified as the [...]
    |-- SUB_QA    v=1 Q=+0.63 * Who is referred to as the father of computer science? -> [...]
    |   `-- SUB_QA    v=1 Q=+0.63 * Who is referred to as the father of computer science? -> [...]
    |       `-- SUB_QA    v=1 Q=+0.63 * Who was the father of the person identified as the [...]
    |           `-- SUB_QA    v=1 Q=+0.63 * Who is commonly referred to as the father of computer [...]
    |               `-- FINAL     v=1 Q=+0.63 * ANS: The question 'Who was the father of the father of [...]
    |-- SUB_QA    v=0 Q=  .     Who was the father of the person identified as the [...]
    |-- SUB_QA    v=0 Q=  .     Who is considered the father of computer science? -> [...]
    |-- SUB_QA    v=0 Q=  .     Who was the father of the person considered the father [...]
    |-- SUB_QA    v=0 Q=  .     Who is commonly referred to as the father of computer [...]
    `-- SELF_CORR v=0 Q=  .     Who was the father of the person identified as the [...]
```

Three things this makes concrete:

**The long chains are rollouts, not deliberate depth.** Each `SUB_QA → SUB_QA → … → FINAL`
run is one `simulate` call: the CoT rollout's `iteration_history` is unpacked into one node
per `(subquestion, subanswer)` pair and wired linearly under the expanded child. So the tree
is *two iterations deep in decisions* but five deep in nodes.

**Most nodes are never visited.** 5 of 17 have `v=0` — expansion mints ~5 children plus a
rollout chain per iteration, and traversal only ever walks one path. This is the concrete
form of the budget problem: the unvisited frontier outruns the iteration budget.

**Sibling coherence is what the shared plan buys.** Both branches ask the *same* two
questions in different words ("commonly referred to as" / "referred to as" / "widely
recognized as"), because both inherit the same plan. Before the plan they re-decomposed
independently, so siblings pursued unrelated decompositions and their visit statistics were
not comparable.

### The plan's own trace on this run

```
PLAN (v1):
  To answer the question, we first need to establish who is referred to as the
  father of computer science. Once that identity is known, we can investigate
  who their father was.

LEDGER:
  intent[0] closed closed_at=7  Establish which person is referred to as the father of …
      bound Q7251       hop=7   "Alan Mathison Turing is referred to as …"
      tried n_facts=3 ×5, n_facts=7 ×1      ← six attempts before it closed
  intent[1] closed closed_at=7  Determine the identity of the father of the person …
      bound Q20895930   hop=7   "Julius Mathison Turing"

GATE:
  iter=6  action=replan  reason=stalled   seen=2  armed=False   ← fired, did not act
  iter=7  action=update  reason=update    seen=5  armed=False
```

That `reason=stalled` at iteration 6 is the **stall branch firing on real data**: intent[0]
had accumulated 5 attempts without closing, which is the *efficacy* failure — the intent was
well-formed, the framing just wasn't landing. `armed=False` because MCTS is log-only, and at
iteration 7 the intent closed anyway.

Answer: **`Julius Mathison Turing`** ✓

### Why MCTS is log-only

`replan` is not an MCTS action, deliberately. `select` breaks only on a childless or
terminal node, and the only re-expansion path is the visited-terminal redirect to
`path[-2]` — so **the root is expanded exactly once**. Any root-level plan fork would
therefore be minted at iteration 1, with empty memory, before any evidence exists. The
signal is recorded so its rate is measurable before that structure is changed.

---

## 4. UPDATE vs REPLAN, on this example

```
                     ┌─────────────────────────────────────────┐
   hop answers ──────▶  resolve_binding_key(answer)            │
                     │    QID via entity_dict, else a literal  │
                     └───────────────┬─────────────────────────┘
                                     ▼
                     ┌───────────────────────────────────────────────┐
                     │  how many DISTINCT keys for this one intent?  │
                     └───┬───────────────┬───────────────────────┬───┘
                         │ exactly 1     │ ≥ 2                   │ 0
                         ▼               ▼                       ▼
                     ┌────────┐   ┌──────────────┐      ┌──────────────────┐
                     │ UPDATE │   │  CONTESTED   │      │ attempts ≥ N ?   │
                     │ close  │   │  → REPLAN    │      │  → STALLED       │
                     │ 0 LLM  │   │  discriminate│      │  → REPLAN        │
                     └────────┘   └──────────────┘      │  (contraction)   │
                                                        └──────────────────┘
       plus, independently:  a cited premise evicted as `contradicted`
                             → FALSIFIED → REPLAN (re-establish it)
```

Precedence is **contested > falsified > stalled**, most specific first.

On this question both intents resolved to exactly one QID, so both hit UPDATE and the
answer came straight out. On the *third largest stadium* row from the same dataset the first
intent bound two distinct QIDs (`United States` and `India`), which is contested → the
replan added a step to fix the ranking criterion → the answer came out correct.

Worth knowing where this **cannot** help: on the *"only cruise line that flies the American
flag"* row the model closed the intent on hop 1 with a single, confident, **wrong** referent.
All three branches are facts about the plan's bookkeeping, and a confidently-wrong binding
leaves the bookkeeping looking healthy. Catching that needs evidence-checking on the
reasoning side, which is where it belongs.

---

## 5. Inspecting your own runs

`--dump DIR` writes one JSON per question with `plan`, `plan_ledger`, `plan_action_log`,
`iteration_history`, `text_memory` and the flattened `tree`. During a real eval,
`runner.py` writes the same content to
`<output_path>/artifacts/q_*/plan.json` automatically.

```bash
python -m langgraph_coe.scripts.smoke_test --strategy mcts --questions 1 --dump /tmp/trees
python - <<'EOF'
import json, textwrap
d = json.load(open('/tmp/trees/mcts_....json'))
kids = {}
for nid, n in d['tree'].items():
    kids.setdefault(n['parent_id'], []).append(nid)
def draw(nid, pre="", last=True, top=True):
    n = d['tree'][nid]; c = n.get('content') or {}
    txt = c.get('question') or c.get('final_answer') or \
          f"{c.get('sub_question','')} -> {c.get('sub_answer','')}"
    vis = n.get('visits') or 0
    q = f"{(n.get('value') or 0)/vis:+.2f}" if vis else "  .  "
    print(pre + ("" if top else ("`-- " if last else "|-- ")) +
          f"[{n['node_type']:14}] v={vis} Q={q}  {textwrap.shorten(str(txt), 70)}")
    ch = kids.get(nid, [])
    cp = pre if top else pre + ("    " if last else "|   ")
    for i, x in enumerate(ch):
        draw(x, cp, i == len(ch)-1, top=False)
draw(d['root_id'])
EOF
```

Related: [plan_channel_status_and_plan.md](plan_channel_status_and_plan.md) for what is
done, what is not, and the experiment plan.
