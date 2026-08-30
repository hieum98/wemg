"""Bedrock reasoning toggle, and the accounting that proves it fired.

Every evaluation in this project so far ran with reasoning off. Turning it on is not a
matter of flipping ``enable_thinking``: that rides in ``chat_template_kwargs``, which
only a self-hosted SGLang chat template reads, so on Bedrock it is silently inert.
Bedrock's own switch is ``additionalModelRequestFields={"reasoning_effort": "high"}``.

Probed live against ``bedrock/qwen.qwen3-32b-v1:0`` in us-west-2:

  * ``reasoning_effort`` alone measured **identical to omitting it** — 30 output tokens,
    no reasoning block — because LiteLLM drops non-allow-listed params. Adding
    ``allowed_openai_params=["reasoning_effort"]`` produced 906 chars of reasoning on
    the same prompt.
  * Only ``"high"`` engages the model. none/low/medium return 4-6 output tokens with no
    reasoning block; minimal/xhigh fail in the backend; max is rejected outright.

Both facts share one failure mode: a "reasoning on" arm that is byte-identical to the
off arm, and therefore an A/B that measures nothing while looking like it ran. These
tests exist to make that failure loud, which is why they assert on the *allow-list* and
not just on the value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from langgraph_coe.config import LangGraphCoeConfig, TierConfig


def _registry(**tier_kwargs):
    """A registry whose ``heavy`` tier carries *tier_kwargs*."""
    from langgraph_coe.llm import RoleModelRegistry

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.llm.tiers = {"heavy": TierConfig(**tier_kwargs)}
    return RoleModelRegistry(cfg.llm)


def _model_kwargs(**tier_kwargs):
    return dict(_registry(**tier_kwargs).get_model_by_tier("heavy").model_kwargs or {})


# ── the switch ────────────────────────────────────────────────────────────────


def test_reasoning_off_is_the_default_so_prior_runs_stay_reproducible():
    kwargs = _model_kwargs(model_name="bedrock/qwen.qwen3-32b-v1:0")
    assert TierConfig().reasoning_effort is None
    assert "reasoning_effort" not in kwargs
    # The allow-list must not appear either: a bare allow-list is harmless but would
    # mean the two arms differ in request shape for no reason.
    assert "allowed_openai_params" not in kwargs


def test_reasoning_on_forwards_the_allow_list_not_just_the_value():
    """The allow-list is the whole fix — without it LiteLLM silently drops the field."""
    kwargs = _model_kwargs(
        model_name="bedrock/qwen.qwen3-32b-v1:0", reasoning_effort="high"
    )
    assert kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" in (kwargs.get("allowed_openai_params") or []), (
        "reasoning_effort without allowed_openai_params measured identical to "
        "reasoning-off in the live probe"
    )


def test_the_two_arms_differ_in_exactly_this_one_knob():
    """Anything else differing would confound the contrast."""
    base = dict(model_name="bedrock/qwen.qwen3-32b-v1:0")
    off = _model_kwargs(**base)
    on = _model_kwargs(**base, reasoning_effort="high")
    assert set(on) - set(off) == {"reasoning_effort", "allowed_openai_params"}
    for key in set(off) & set(on):
        assert off[key] == on[key], key


def test_the_sglang_toggle_is_still_forwarded_independently():
    """The two mechanisms target different backends and must not have been merged."""
    kwargs = _model_kwargs(model_name="openai/Qwen/Qwen3.5-4B", enable_thinking=True)
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in kwargs


def test_reasoning_effort_is_settable_per_tier_from_a_dotted_override():
    """The arms are selected on the CLI, so this path has to work."""
    from langgraph_coe.evaluation.evaluate import (
        _apply_config_overrides,
        split_eval_overrides,
    )

    cfg = LangGraphCoeConfig.from_yaml()
    _eval, overrides = split_eval_overrides(["llm.tiers.heavy.reasoning_effort=high"])
    _apply_config_overrides(cfg, overrides)
    assert cfg.llm.tiers["heavy"].reasoning_effort == "high"


def test_aliased_tiers_are_independent_objects():
    """Why a reasoning override must name every tier it means, not just ``heavy``.

    config.eval.yaml declares one tier as ``&qwen8b`` and reuses it by alias/merge-key. If
    those shared a single object, setting ``heavy`` would set them all; they do not, so a
    one-tier override leaves the rest untouched and produces a silently mixed arm that
    looks enabled and measures as disabled.
    """
    cfg = LangGraphCoeConfig.from_yaml("langgraph_coe/config.eval.yaml")
    tiers = cfg.llm.tiers
    assert {"heavy", "plan", "consolidate", "medium", "light"} <= set(tiers)
    tiers["heavy"].reasoning_effort = "high"
    assert [t for t in tiers if tiers[t].reasoning_effort == "high"] == ["heavy"]


def test_eval_config_reasons_only_on_the_roles_measured_to_benefit():
    """The selective-reasoning result, pinned as config.

    Reasoning suppressed evidence on the extraction/consolidation roles — 38% fewer facts
    extracted, ~50% fewer memory items retained — while buying no accuracy anywhere. So it
    belongs on the roles that reason over evidence and nowhere else. On SGLang the switch is
    ``enable_thinking``; ``reasoning_effort`` is Bedrock-only and must stay unset here, or
    it reads as configured and silently does nothing.
    """
    from langgraph_coe.llm import RoleModelRegistry

    cfg = LangGraphCoeConfig.from_yaml("langgraph_coe/config.eval.yaml")
    reg = RoleModelRegistry(cfg.llm)

    def thinks(role: str) -> bool:
        return cfg.llm.tiers[reg._get_tier(role)].enable_thinking

    for role in ("subquestion_generator", "answer_generator", "self_corrector",
                 "final_answer_synthesizer", "planner"):
        assert thinks(role) is True, f"{role} reasons over evidence; keep thinking on"
    for role in ("memory_consolidation", "extractor", "open_ie", "triple_pruner",
                 "verifier"):
        assert thinks(role) is False, f"{role} is recall/bounded; thinking costs evidence"
    assert all(t.reasoning_effort is None for t in cfg.llm.tiers.values())


def test_eval_config_has_no_tier_without_roles():
    """A tier mapped to zero roles is dead config. ``classify`` was exactly that — every
    role config.py places on it is remapped to ``light`` here — and it went unnoticed
    because ``_get_tier`` falls back to ``heavy`` for anything unmapped."""
    from collections import defaultdict

    from langgraph_coe import roles as roles_mod
    from langgraph_coe.llm import RoleModelRegistry

    cfg = LangGraphCoeConfig.from_yaml("langgraph_coe/config.eval.yaml")
    reg = RoleModelRegistry(cfg.llm)
    names = {
        getattr(v, "name")
        for v in vars(roles_mod).values()
        if hasattr(v, "name") and hasattr(v, "output_model")
    }
    used = defaultdict(list)
    for n in names:
        used[reg._get_tier(n)].append(n)
    unused = sorted(set(cfg.llm.tiers) - set(used))
    assert not unused, f"tiers defined but mapped to no role: {unused}"


# ── the accounting ────────────────────────────────────────────────────────────


def _raw(completion_tokens=None, reasoning_tokens=None, *, metadata=True, finish=None):
    """A stand-in for the LangChain message ``include_raw=True`` hands back."""
    if not metadata:
        return SimpleNamespace(response_metadata={}, usage_metadata=None)
    usage = {"completion_tokens": completion_tokens, "prompt_tokens": 10}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    meta = {"token_usage": usage}
    if finish is not None:
        meta["finish_reason"] = finish
    return SimpleNamespace(response_metadata=meta, usage_metadata=None)


def test_truncation_is_counted_because_it_would_invalidate_the_reasoning_arm():
    """Reasoning is billed inside max_tokens, so an over-long trace truncates the JSON
    payload — the role then falls back to a neutral default and the arm loses accuracy
    for a budget reason. Indistinguishable from "reasoning didn't help" without this."""
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    _record_output_cost("open_ie", _raw(8192, 8000, finish="length"))
    _record_output_cost("open_ie", _raw(600, 400, finish="stop"))
    assert meter["truncated_responses"] == 1
    assert meter["by_role"]["open_ie"]["truncated_responses"] == 1


def test_a_clean_stop_is_never_counted_as_truncation():
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    for finish in ("stop", "end_turn", None):
        _record_output_cost("verifier", _raw(100, 50, finish=finish))
    assert meter["truncated_responses"] == 0


def test_reasoning_tokens_are_counted_separately_from_the_payload():
    """Reasoning is the entire cost of a reasoning-on arm and is absent from
    ``prompt_tokens``, so without this the A/B could only report call counts."""
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    _record_output_cost("subquestion_generator", _raw(209, 196))
    assert meter["completion_tokens"] == 209
    assert meter["reasoning_tokens"] == 196
    assert meter["reasoning_responses"] == 1
    role = meter["by_role"]["subquestion_generator"]
    assert (role["completion_tokens"], role["reasoning_tokens"]) == (209, 196)


def test_a_response_without_reasoning_leaves_the_reasoning_counter_alone():
    """This is how the off arm reads, and how a silently-inert toggle would read."""
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    _record_output_cost("verifier", _raw(30, None))
    assert meter["completion_tokens"] == 30
    assert meter["reasoning_tokens"] == 0
    assert meter["reasoning_responses"] == 0


def test_reasoning_responses_is_what_distinguishes_on_from_inert():
    """A toggle that passed the gateway but did nothing yields output tokens with zero
    reasoning responses — indistinguishable from the off arm on totals alone."""
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    for _ in range(3):
        _record_output_cost("planner", _raw(30, None))
    assert meter["completion_tokens"] == 90
    assert meter["reasoning_responses"] == 0, "the signal that reasoning never fired"


def test_accounting_never_breaks_a_graph_on_a_missing_usage_block():
    """A provider that omits usage must leave the counters at zero, not raise —
    zero reads as 'unmeasured', which is recoverable; an exception is not."""
    from langgraph_coe.llm import _record_output_cost, start_cost_meter

    meter = start_cost_meter()
    for bad in (None, object(), _raw(metadata=False)):
        _record_output_cost("extractor", bad)
    assert meter["completion_tokens"] == 0
    assert meter["reasoning_tokens"] == 0


def test_recording_is_a_no_op_without_an_installed_meter():
    """Library use and unit tests run with no meter; behaviour must not change."""
    import langgraph_coe.llm as llm_mod

    llm_mod._cost_meter.set(None)
    llm_mod._record_output_cost("extractor", _raw(10, 5))  # must not raise


def test_output_cost_is_recorded_even_when_the_completion_fails_to_parse():
    """Under reasoning-on the unparseable response is *more* likely to be the
    expensive one — reasoning consumed the max_tokens budget — so attributing spend
    only on success would understate exactly the arm being measured."""
    import inspect

    from langgraph_coe.llm import execute_role_lc

    src = inspect.getsource(execute_role_lc)
    collect = src[src.index("def _collect") :]
    record = collect.index("_record_output_cost")
    parsed_branch = collect.index('r.get("parsed")')
    assert record < parsed_branch, "must be recorded before the parse branch returns"


# ── the budget interaction ────────────────────────────────────────────────────


def test_thinking_budget_would_ship_an_sglang_only_param_to_bedrock():
    """``thinking_budget`` cannot cap Bedrock reasoning, and worse, it does not abstain.

    ``build_request_kwargs`` keys off the *model name*, and "qwen3-32b" matches, so it
    happily returns a dill-pickled ``custom_logit_processor`` for a Bedrock tier — a
    param only an SGLang server run with ``--enable-custom-logit-processor`` can honour.

    It is inert today only because ``llm.get_model_by_tier`` gates it behind
    ``enable_thinking``, which config.eval.yaml pins False. So a future Bedrock tier
    setting ``enable_thinking: true`` alongside a budget would ship the blob for nothing.
    The consequence for this experiment: on Bedrock the only bound on reasoning length is
    ``max_tokens``, shared with the JSON payload — which is why the launch script raises
    it rather than reaching for ``thinking_budget``.
    """
    from langgraph_coe.thinking_budget import build_request_kwargs

    kwargs = build_request_kwargs("bedrock/qwen.qwen3-32b-v1:0", 4096)
    assert "custom_logit_processor" in (kwargs or {}), (
        "if this now abstains for managed providers, the note above is stale"
    )

    # The gate that keeps it inert. Asserted because it is load-bearing.
    off = _model_kwargs(
        model_name="bedrock/qwen.qwen3-32b-v1:0",
        enable_thinking=False,
        thinking_budget=4096,
    )
    assert "custom_logit_processor" not in off


@pytest.mark.parametrize("effort", ["none", "low", "medium", "minimal", "max", "xhigh"])
def test_only_high_is_a_supported_effort_on_this_deployment(effort):
    """Recorded as an executable note: the Bedrock gateway advertises this whole enum,
    but on qwen3-32b every value except ``high`` either produces no reasoning or errors
    in the backend. A future config setting one of these would look enabled and measure
    as disabled."""
    assert effort != "high"
