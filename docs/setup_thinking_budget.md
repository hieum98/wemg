# Setting Up the Thinking-Token Budget for `langgraph_coe`

This guide covers capping the number of **reasoning ("thinking") tokens** a Qwen
model spends per call. Wiring is in `langgraph_coe/thinking_budget.py`,
`langgraph_coe/config.py` (`TierConfig.thinking_budget`), and
`langgraph_coe/llm.py` (`RoleModelRegistry.get_model_by_tier`).

> **TL;DR**
> 1. Launch SGLang with `--enable-custom-logit-processor` and `--reasoning-parser qwen3`.
> 2. Make sure the model's `<think>`/`</think>` token ids are in `THINK_TOKEN_IDS`
>    (`langgraph_coe/thinking_budget.py`). Qwen3.5 and Qwen3 are already there.
> 3. In `langgraph_coe/config.yaml`, on a tier with `enable_thinking: true`, add
>    `thinking_budget: <N>` (keep it **well below** `max_tokens`).
> 4. Done — `RoleModelRegistry` forwards a custom logit processor that forces
>    `</think>` after `N` thinking tokens.

---

## Why this is needed (and why it isn't one line)

SGLang exposes **no** request parameter to limit *reasoning* tokens — unlike
Anthropic's `thinking.budget_tokens`. Specifically:

- `chat_template_kwargs: {thinking_budget: N}` is **silently ignored** by SGLang.
- The Qwen team's own docs say a true thinking budget lives only in Alibaba's
  hosted service, not in open-source serving frameworks.

The only working mechanism is a **custom logit processor**: a small callable
shipped with each request that forces the `</think>` token once `N` thinking
tokens have been emitted. SGLang ships a built-in `Qwen3ThinkingBudgetLogitProcessor`,
but its token ids are hardcoded for the *original* Qwen3 family and **silently
no-op** on Qwen3.5 (different ids). This module provides a model-agnostic
replacement.

---

## Prerequisites

- **SGLang server launched with `--enable-custom-logit-processor`.**
  Without it, the budget is silently ignored (and a malformed processor can
  crash the worker — see Troubleshooting).
- `--reasoning-parser qwen3` so reasoning is separated into `reasoning_content`
  (already required for the rest of the stack).
- The model's `<think>` / `</think>` / newline token ids registered in
  `THINK_TOKEN_IDS` (`langgraph_coe/thinking_budget.py`). Shipped:

  | Model substring | `<think>` | `</think>` | newline | Verified |
  |-----------------|-----------|------------|---------|----------|
  | `qwen3.5`       | 248068    | 248069     | 198     | ✅ live (Qwen3.5-4B) |
  | `qwen3-next`    | 151667    | 151668     | 198     | ⚠️ inherited from Qwen3 |
  | `qwen3`         | 151667    | 151668     | 198     | ✅ (original Qwen3 ids) |

  Matching is case-insensitive on the model name with any provider prefix
  (`openai/`) stripped. Unknown models → budget skipped with a warning (never a
  silent wrong-id no-op).

> The app venv (`uv`, Python 3.13) does **not** need `sglang` or `dill` at
> runtime — the processor is shipped as a pre-serialized blob (see
> [Regenerating the blob](#regenerating-the-processor-blob)).

---

## 1. Launch SGLang with custom logit processors enabled

```bash
# server-side conda env (the one running SGLang)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --reasoning-parser qwen3 \
  --enable-custom-logit-processor \
  --host 0.0.0.0 --port 30000
```

Verify the flag took effect (a *valid* no-op processor returns `200`; the server
stays up):

```bash
curl -s http://<host>:30000/v1/models   # should list the model
```

---

## 2. Enable the budget in config

Edit `langgraph_coe/config.yaml`. The budget only applies to tiers with
`enable_thinking: true`:

```yaml
llm:
  tiers:
    light:
      model_name: openai/Qwen/Qwen3.5-4B
      api_base: http://n0152:30000/v1
      temperature: 0.2
      max_tokens: 2048
      enable_thinking: true     # required for the budget to do anything
      thinking_budget: 512      # cap reasoning at ~512 tokens; MUST be < max_tokens
```

Rules of thumb:

- **`thinking_budget` < `max_tokens`, with headroom.** `max_tokens` still bounds
  *thinking + answer combined*; the budget must leave room for the answer (and,
  for roles with structured output, the JSON). A budget of ~512 with
  `max_tokens: 2048` is comfortable; `budget ≈ max_tokens` is not.
- **Don't set it too small.** Structure stays valid even at tiny budgets, but
  *answer quality* degrades (at `budget=32` the model failed to recall a fact it
  got right at `budget=256`). Give reasoning-heavy roles (`answer_generator`,
  `self_corrector`, `final_answer_synthesizer`) a generous budget; reserve tight
  caps for bounded roles.
- `thinking_budget: null` (default) = no cap.

---

## 3. How it flows

`RoleModelRegistry.get_model_by_tier` (`langgraph_coe/llm.py`):

1. Builds `model_kwargs` with `chat_template_kwargs.enable_thinking`.
2. If `enable_thinking and thinking_budget is not None`, calls
   `thinking_budget.build_request_kwargs(model_name, budget)` and merges:
   ```python
   {
     "custom_logit_processor": "<pre-serialized blob>",
     "custom_params": {
       "thinking_budget": 512,
       "think_start_id": 248068, "think_end_id": 248069, "newline_id": 198,
     },
   }
   ```
3. `ChatLiteLLM` forwards both to SGLang (verified to survive the
   LangChain → LiteLLM → SGLang path).

At decode time the processor counts tokens since `<think>`; once the budget is
spent it forces a newline then `</think>`, ending reasoning. The existing
`_content_to_text` strips the `thinking` block, and `with_structured_output`'s
fallback parser reads the JSON answer as usual.

```python
# Smoke test against a live server
import asyncio
from langgraph_coe.config import LLMConfig
from langgraph_coe.llm import RoleModelRegistry
from langchain_core.messages import HumanMessage

cfg = LLMConfig()
t = cfg.tiers["light"]
t.model_name, t.api_base, t.api_key = "openai/Qwen/Qwen3.5-4B", "http://n0152:30000/v1", "EMPTY"
t.enable_thinking, t.thinking_budget, t.max_tokens = True, 48, 700

m = cfg and RoleModelRegistry(cfg).get_model_by_tier("light")
print("custom_params:", m.model_kwargs.get("custom_params"))

async def main():
    r = await m.ainvoke([HumanMessage(content="How many primes between 1 and 200? Reason step by step.")])
    rc = (r.additional_kwargs or {}).get("reasoning_content") or ""
    print("reasoning_chars:", len(rc))   # budgeted ≈ small; unbudgeted is much larger
asyncio.run(main())
```

---

## Structured output

The budget is **compatible with** `with_structured_output` (the
`execute_role_lc` path) and tends to make it **more reliable** under a tight
`max_tokens`:

| Config (`max_tokens`) | Structured output | Note |
|-----------------------|-------------------|------|
| `budget=256` (1024)   | ✅ valid           | good answer |
| `budget=32` (1024)    | ✅ valid           | answer quality drops |
| `budget=None` (1024)  | ❌ fails           | unbounded thinking exhausts `max_tokens`, empty content |
| `budget=None` (4096)  | ✅ valid           | needed the larger ceiling |

The forced `</think>` ends reasoning early, leaving tokens for the JSON answer.
Keep the budget low enough that `max_tokens - thinking_budget` comfortably fits
the structured response.

---

## Regenerating the processor blob

The processor is a `dill`-pickled class committed at
`langgraph_coe/thinking_budget_clp.json`. **`dill` bytecode is Python-version
specific:** a blob pickled in the app venv (Py3.13) **cannot** be unpickled by
the SGLang server (Py3.12). So the committed blob is generated with the
**server's** Python, and you only need to regenerate it if
`_ThinkingBudgetLogitProcessor.__call__` changes (token ids are data in
`custom_params`, not baked into the blob):

```bash
# run with the SERVER's Python (the SGLang env), from repo root
PYTHONPATH=. /path/to/sglang-env/bin/python -m langgraph_coe.thinking_budget --regen
```

It prints the interpreter version it used — that must match the server's.

Adding a new model family: append a `(substring, (start, end, newline))` entry to
`THINK_TOKEN_IDS`. Get the ids from the tokenizer; **no blob regen needed**:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
print(tok.encode("<think>", add_special_tokens=False),   # -> [248068]
      tok.encode("</think>", add_special_tokens=False),  # -> [248069]
      tok.encode("\n", add_special_tokens=False))         # -> [198]
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Reasoning never truncates | `thinking_budget` wrong-model ids no-op, or thinking off | Verify model in `THINK_TOKEN_IDS`; set `enable_thinking: true` |
| Budget ignored entirely | Server launched without `--enable-custom-logit-processor` | Relaunch SGLang with the flag |
| Empty content / parse failure | `max_tokens` too small for thinking + answer | Lower `thinking_budget` or raise `max_tokens` |
| Whole SGLang worker dies (SIGQUIT) | Malformed `custom_logit_processor` payload (bad/version-mismatched blob) | Never hand-edit the blob; regen with the server's Python |
| `non-hexadecimal number / no locals found` on load | Blob pickled with a different Python than the server | `--regen` with the server's interpreter |
| Answer quality dropped | Budget too small | Raise `thinking_budget` for reasoning-heavy roles |
| Unknown-model warning in logs | Model not in `THINK_TOKEN_IDS` | Add its token ids (budget is skipped, not wrong, until then) |

---

## Related source files

| Path | Role |
|------|------|
| `langgraph_coe/thinking_budget.py` | Processor, `THINK_TOKEN_IDS`, `build_request_kwargs`, `--regen` |
| `langgraph_coe/thinking_budget_clp.json` | Pre-serialized (server-Python) processor blob |
| `langgraph_coe/config.py` | `TierConfig.thinking_budget` |
| `langgraph_coe/config.yaml` | Per-tier `thinking_budget` operator toggle |
| `langgraph_coe/llm.py` | `RoleModelRegistry.get_model_by_tier` forwards the processor |
