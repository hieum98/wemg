# Generation / Sampling Parameters for `langgraph_coe`

This guide covers the per-tier sampling knobs that control LLM decoding. They
live on `TierConfig` (`langgraph_coe/config.py`) and are forwarded by
`RoleModelRegistry.get_model_by_tier` (`langgraph_coe/llm.py`) through
`ChatLiteLLM` → LiteLLM → SGLang.

> **TL;DR**
> Set any of these on a tier in `langgraph_coe/config.yaml`; **omit** a knob to
> use the server's default (the code only sends knobs you set):
> ```yaml
> llm:
>   tiers:
>     light:
>       temperature: 1.0
>       top_p: 0.95
>       top_k: 20
>       min_p: 0.0
>       presence_penalty: 1.5
>       frequency_penalty: 0.0
>       repetition_penalty: 1.0
>       seed: 8888
> ```

---

## Knobs

| Knob | Type | Default | OpenAI-standard? | Notes |
|------|------|---------|------------------|-------|
| `temperature` | float | `0.7` (per tier) | yes | Constructor arg on `ChatLiteLLM`. |
| `top_p` | float | `0.95` | yes | Nucleus sampling. Always sent. |
| `top_k` | int | `None` (off) | no | Forwarded to SGLang request body. `1` ≈ greedy. |
| `min_p` | float | `None` (off) | no | Must be in `[0, 1]` (SGLang validates). |
| `presence_penalty` | float | `None` (off) | yes | Penalize tokens already present. |
| `frequency_penalty` | float | `None` (off) | yes | Penalize by token frequency. |
| `repetition_penalty` | float | `None` (off) | no | `1.0` = no effect; `>1` discourages repeats. |
| `seed` | int | `None` | yes | Reproducible sampling (best-effort on SGLang). |
| `max_tokens` | int | `8192` (per tier) | yes | Total output cap (thinking + answer). |
| `enable_thinking` | bool | `True` | — | Qwen reasoning toggle (`chat_template_kwargs`). |
| `thinking_budget` | int | `None` | — | Cap on reasoning tokens — see [setup_thinking_budget.md](setup_thinking_budget.md). |

**Defaults note:** these are the `TierConfig` field defaults
(`langgraph_coe/config.py`). The shipped `config.yaml` sets some explicitly per
tier; anything not set there falls back to these.

### OpenAI-standard vs SGLang-only

`top_k`, `min_p`, and `repetition_penalty` are **not** OpenAI Chat Completions
parameters. LiteLLM forwards them to SGLang's request body unchanged — verified
live (SGLang's own validator rejects e.g. `min_p=5.0` with
*"min_p must be in [0, 1]"*, proving the value reaches the server rather than
being dropped client-side). The OpenAI-standard knobs (`temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `seed`) are passed natively.

---

## How it's wired

`langgraph_coe/llm.py`, `RoleModelRegistry.get_model_by_tier`:

```python
model_kwargs = {"top_p": cfg.top_p}
for name, value in (("top_k", cfg.top_k), ("min_p", cfg.min_p),
                    ("presence_penalty", cfg.presence_penalty),
                    ("frequency_penalty", cfg.frequency_penalty),
                    ("repetition_penalty", cfg.repetition_penalty),
                    ("seed", cfg.seed)):
    if value is not None:               # unset knob -> server default
        model_kwargs[name] = value
model_kwargs["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}
ChatLiteLLM(model=..., temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            model_kwargs=model_kwargs, ...)
```

`temperature` and `max_tokens` are `ChatLiteLLM` constructor args; every other
sampling knob rides in `model_kwargs`. `ChatLiteLLM`'s native `top_p`/`top_k`
fields default to `None`, so routing them via `model_kwargs` causes no
duplicate-key conflict.

---

## Setting values

### Per tier in YAML (recommended)

Add knobs under `llm.tiers.<tier>` in `langgraph_coe/config.yaml`. Tiers are
independent — e.g. give `heavy` exploratory sampling and `classify` near-greedy:

```yaml
llm:
  tiers:
    heavy:
      temperature: 1.0
      top_p: 0.95
      min_p: 0.0
    classify:
      temperature: 0.0
      top_k: 1          # greedy for tiny fixed-shape outputs
```

### Dotted runtime override

Per project convention (`key=value`):

```
llm.tiers.heavy.top_k=20 llm.tiers.heavy.presence_penalty=1.5
```

### In code

```python
from langgraph_coe.config import LLMConfig
cfg = LLMConfig()
cfg.tiers["heavy"].top_k = 20
cfg.tiers["heavy"].repetition_penalty = 1.1
```

---

## Guidance

- **Omit, don't neutralize.** Leaving a knob unset (`None`) is not the same as
  setting a "neutral" value — unset means SGLang picks the default; an explicit
  value always overrides. Only set knobs you intend to control.
- **Penalties interact.** `presence_penalty`/`frequency_penalty` (additive,
  logit-space) and `repetition_penalty` (multiplicative) both fight repetition;
  combining aggressive values can degrade fluency. Start from one.
- **Reproducibility.** Set `seed` (and keep temperature/top_p fixed) for
  repeatable sampling; SGLang honors it best-effort, not bit-exact across
  batching.
- **Per-tier intent.** Reasoning tiers (`heavy`/`medium`) tolerate higher
  temperature/`top_p`; bounded `classify` roles do better near-greedy
  (`temperature: 0.0` or `top_k: 1`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `min_p must be in [0, 1]` (BadRequest) | Out-of-range value | Use `0 ≤ min_p ≤ 1` |
| Knob seems ignored | Left unset (`None`) | Set it explicitly on the tier |
| `top_k`/`min_p` rejected as unknown | Backend isn't SGLang/OpenAI-compatible w/ these | Drop the non-OpenAI knobs for that backend |
| Repetitive output | No penalty set | Add `repetition_penalty: 1.1` or `frequency_penalty` |
| Non-reproducible runs | `seed` unset or temperature high | Set `seed` and lower temperature |

---

## Related source files

| Path | Role |
|------|------|
| `langgraph_coe/config.py` | `TierConfig` sampling fields |
| `langgraph_coe/config.yaml` | Per-tier operator values |
| `langgraph_coe/llm.py` | `get_model_by_tier` forwards knobs to SGLang |
| `docs/setup_thinking_budget.md` | The `thinking_budget` knob (reasoning-token cap) |
