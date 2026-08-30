#!/usr/bin/env python
"""Paired report for the reasoning-on arms, plus the checks that make the arm valid.

Two things this reports that ``fix_report.py`` cannot:

1. **Arm validity.** A reasoning arm has three silent failure modes, each of which makes
   the contrast measure something other than what it claims:
     * the toggle never fired (LiteLLM drops ``reasoning_effort`` unless allow-listed,
       and only ``"high"`` engages qwen3-32b) → the "on" arm IS the off arm;
     * responses were truncated at ``max_tokens`` (reasoning is billed inside it) → the
       role fell back to a neutral default, so the arm lost accuracy for a budget reason;
     * the two arms disagree on any knob other than the plan flag.
   None of these show up in an accuracy number, so they are printed first and loudly.
   Reporting a contrast whose validity block is dirty is reporting a confound.

2. **Output-side cost.** Reasoning is 3/4 of output tokens and is invisible in
   ``prompt_tokens``, which is all the older reports counted. Without it, thinking scores
   as free.

Primary accuracy metric is ``sub_em_short_relaxed``: three golds in this dataset arrive
wrapped ("at the city of Cairo, Illinois") and can never match a correctly concise
answer, which cost +1 to +3 rows in all 27 runs measured. Plain ``sub_em_short`` is
printed alongside for continuity with the historical numbers.

Usage:
  python scripts/reason_report.py r_cot_plan_on:r_cot_plan_off
  python scripts/reason_report.py r_cot_plan_on:r_cot_plan_off --vs a1:a0
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_report import boot_ci, sign_test, wilcoxon  # noqa: E402
from fix_report import load as _load_strict  # noqa: E402


def load(run: str) -> Dict[str, dict]:
    """Rows for a run, or {} if it has not flushed its first chunk yet.

    Arms are read while still in flight — the runner appends in chunks of 8 — so a missing
    log means "not started", not "broken". Crashing there would make the report unusable
    for exactly the monitoring it is most needed for.
    """
    try:
        return _load_strict(run)
    except FileNotFoundError:
        return {}

COST_KEYS = ("calls", "completions", "prompt_tokens", "completion_tokens", "reasoning_tokens")


def _cost(rec: dict) -> dict:
    return rec.get("cost") or {}


def arm_facts(run: str) -> dict:
    """Everything needed to judge whether an arm is what it says it is."""
    rows = list(load(run).values())
    cfg_path = f"results/{run}/config.yaml"
    cfg_text = open(cfg_path).read() if os.path.exists(cfg_path) else ""
    tiers_on = cfg_text.count("reasoning_effort: high")
    tiers_total = cfg_text.count("model_name:")
    agg = {k: 0 for k in COST_KEYS}
    agg["reasoning_responses"] = 0
    agg["truncated_responses"] = 0
    for r in rows:
        c = _cost(r)
        for k in list(agg):
            agg[k] += c.get(k, 0) or 0
    return {
        "run": run,
        "rows": len(rows),
        "tiers_on": tiers_on,
        "tiers_total": tiers_total,
        "max_tokens": sorted({
            int(line.split(":")[1]) for line in cfg_text.splitlines()
            if line.strip().startswith("max_tokens:")
        }) or [None],
        **agg,
    }


def _flatten(obj, prefix="") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj
    return out


# Keys a contrast is ALLOWED to differ on — the intervention being ablated. Anything else
# differing means the arms are not a clean ablation and the gap cannot be attributed.
# Deliberately a small, explicit allow-list rather than a pattern: the whole value of this
# check is that an unlisted difference is loud, so a new experiment flag should have to be
# added here consciously.
_EXPECTED_DIFFS = {
    "search.plan.enabled",
    "search.strategy",  # differs between contrasts, never within one
    "search.cot.synthesis_sees_dropped_evidence",
    "search.cot.recent_evidence_first",
    "search.plan.verify_terminal_referents",
    "search.plan.mcts_plan_scope",
}


def _is_reasoning_contrast(run_a: str, run_b: str) -> bool:
    """Is this pair a reasoning-regime contrast rather than a plan ablation?

    The two kinds of comparison have different legitimate differences, and judging one by
    the other's rules reports a correct arm as broken. A pair involving `selective` or a
    `_noreason` control is *supposed* to differ on the reasoning knobs.
    """
    return any(
        tag in r for r in (run_a, run_b) for tag in ("selective", "_noreason")
    )


# Reasoning-regime contrasts may additionally differ on these. `role_tiers.*` is included
# because taking reasoning off one role means remapping it to a reasoning-off tier, and
# `max_tokens` because reasoning is billed inside it.
_REASONING_DIFF_PREFIXES = (
    "llm.tiers.",
    "llm.role_tiers.",
)


def config_diff(run_a: str, run_b: str) -> List[str]:
    """Config keys on which two arms disagree, excluding the intended ablation."""
    try:
        import yaml
    except ImportError:
        return []
    pa, pb = f"results/{run_a}/config.yaml", f"results/{run_b}/config.yaml"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return []
    fa = _flatten(yaml.safe_load(open(pa)) or {})
    fb = _flatten(yaml.safe_load(open(pb)) or {})
    reasoning = _is_reasoning_contrast(run_a, run_b)
    bad = []
    for k in sorted(set(fa) | set(fb)):
        if k in _EXPECTED_DIFFS or k.endswith("output_path"):
            continue
        if reasoning and k.startswith(_REASONING_DIFF_PREFIXES):
            continue
        if fa.get(k) != fb.get(k):
            bad.append(f"{k}: {fa.get(k)!r} vs {fb.get(k)!r}")
    return bad


def print_ablation_check(pairs: List[str]) -> bool:
    """Confirm each pair differs only in what it is meant to differ in."""
    ok = True
    for pair in pairs:
        a, b = pair.split(":")
        bad = config_diff(a, b)
        kind = "reasoning knobs" if _is_reasoning_contrast(a, b) else "the plan flag"
        if bad:
            ok = False
            print(f"  {a} vs {b}: ** {len(bad)} UNINTENDED CONFIG DIFFERENCE(S) **")
            for line in bad[:8]:
                print(f"      {line}")
        else:
            print(f"  {a} vs {b}: differ only in {kind}  ✓")
    return ok


def print_validity(facts: List[dict], expect_reasoning_diff: bool = False) -> bool:
    print("── ARM VALIDITY (read this before any number below) ──────────────")
    ok = True
    for f in facts:
        compl = f["completions"] or 1
        fired = f["reasoning_responses"] / compl
        trunc = f["truncated_responses"] / compl
        print(f"  {f['run']:<18} rows {f['rows']:>4}  "
              f"reasoning tiers {f['tiers_on']}/{f['tiers_total']}  "
              f"max_tokens {f['max_tokens']}")
        print(f"  {'':<18} reasoning fired on {f['reasoning_responses']}/{compl} "
              f"completions = {fired:.1%}")
        print(f"  {'':<18} truncated at max_tokens: {f['truncated_responses']} "
              f"({trunc:.2%})")
        if 0 < fired < 0.98:
            print(f"  {'':<18} (partial: reasoning on some tiers only)")
        if trunc > 0.01:
            print(f"  {'':<18} ** {trunc:.1%} TRUNCATED: payloads cut off, so this arm "
                  f"lost accuracy to the token budget, not to reasoning **")
            ok = False
    mt = {tuple(f["max_tokens"]) for f in facts}
    if len(mt) > 1:
        # Whether this actually confounds anything is decidable, not a matter of opinion:
        # max_tokens only bites when it truncates. If truncation is ~0 in both arms the
        # ceiling never bound and the difference is inert, so report it as a caveat with
        # the evidence rather than voiding the comparison.
        worst = max(
            f["truncated_responses"] / (f["completions"] or 1) for f in facts
        )
        if worst > 0.01:
            print(f"  ** arms disagree on max_tokens {mt} AND truncation reaches "
                  f"{worst:.1%} — confounded **")
            ok = False
        else:
            print(f"  note: arms disagree on max_tokens {mt}, but truncation is "
                  f"{worst:.2%} in both — the ceiling never bound, so this is inert")
    # Arms with no completions yet are excluded: an in-flight arm has fired on 0 of 0
    # completions, which is indistinguishable from reasoning-off on totals alone and would
    # flag every mid-run report as a reasoning contrast.
    started = [f for f in facts if f["completions"]]
    # Partial reasoning is only a confound when the arms DISAGREE about it. Judging each
    # arm against "should be ~100%" was wrong twice: it flagged the deliberately-partial
    # `selective` arm as broken, and then flagged its sibling too because the exemption was
    # keyed on the run *name*. What actually matters is the spread between arms — two arms
    # that both reason on 60.7% of completions are perfectly comparable to each other,
    # whereas 99% against 60% is a reasoning contrast wearing another label.
    fractions = [f["reasoning_responses"] / (f["completions"] or 1) for f in started]
    if fractions and (max(fractions) - min(fractions)) > 0.05:
        if expect_reasoning_diff:
            print(f"  reasoning differs by design ({min(fractions):.1%} vs "
                  f"{max(fractions):.1%}) — that IS the intervention here")
        else:
            print(f"  ** arms disagree on how much they reasoned "
                  f"({min(fractions):.1%} vs {max(fractions):.1%}): this is a REASONING "
                  f"contrast, not the contrast it claims to be **")
            ok = False
    print(f"  verdict: {'arms comparable' if ok else 'DO NOT REPORT AS-IS'}")
    print()
    return ok


def compare(pairs: List[str], label: str) -> dict:
    n = 0
    hit = {"a": 0, "b": 0}
    hit_raw = {"a": 0, "b": 0}
    disc_a = disc_b = 0
    disc_raw_a = disc_raw_b = 0
    acc_diffs: List[float] = []
    cost_diffs: Dict[str, List[float]] = {k: [] for k in COST_KEYS}
    per_pair: List[str] = []

    for pair in pairs:
        pa, pb = pair.split(":")
        A, B = load(pa), load(pb)
        keys = sorted(set(A) & set(B))
        a_hit = b_hit = 0
        for k in keys:
            ra, rb = A[k], B[k]
            ea = float(ra.get("sub_em_short_relaxed") or ra.get("sub_em_short") or 0.0)
            eb = float(rb.get("sub_em_short_relaxed") or rb.get("sub_em_short") or 0.0)
            ra_raw = float(ra.get("sub_em_short") or 0.0)
            rb_raw = float(rb.get("sub_em_short") or 0.0)
            n += 1
            a_hit += ea > 0
            b_hit += eb > 0
            hit["a"] += ea > 0
            hit["b"] += eb > 0
            hit_raw["a"] += ra_raw > 0
            hit_raw["b"] += rb_raw > 0
            if ea > 0 and eb == 0:
                disc_a += 1
            elif eb > 0 and ea == 0:
                disc_b += 1
            if ra_raw > 0 and rb_raw == 0:
                disc_raw_a += 1
            elif rb_raw > 0 and ra_raw == 0:
                disc_raw_b += 1
            acc_diffs.append((1.0 if ea > 0 else 0.0) - (1.0 if eb > 0 else 0.0))
            ca, cb = _cost(ra), _cost(rb)
            for key in COST_KEYS:
                cost_diffs[key].append((ca.get(key, 0) or 0) - (cb.get(key, 0) or 0))
        per_pair.append(
            f"  {pa:<18} {a_hit:>3}/{len(keys):<4} vs {pb:<18} {b_hit:>3}/{len(keys):<4}"
            f"  gap {a_hit - b_hit:+3d}"
        )

    if not n:
        print(f"  no overlapping questions for {label}")
        return {}

    print(f"── {label} ─────────────────────────────────────────")
    print("\n".join(per_pair))
    d = disc_a + disc_b
    p = sign_test(disc_a, disc_b)
    lo, hi = boot_ci(acc_diffs)
    print(f"  paired questions: {n}")
    print(f"  relaxed sub-EM   on {hit['a']}/{n} = {100 * hit['a'] / n:.2f}%   "
          f"off {hit['b']}/{n} = {100 * hit['b'] / n:.2f}%   "
          f"gap {hit['a'] - hit['b']:+d} ({100 * (hit['a'] - hit['b']) / n:+.2f} pts)")
    print(f"  discordant {d} ({disc_a}/{disc_b})  sign test p = {p:.4f}  "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    print(f"  bootstrap 95% CI on paired difference: [{lo:+.4f}, {hi:+.4f}]")
    p_raw = sign_test(disc_raw_a, disc_raw_b)
    print(f"  [plain sub_em_short, for continuity: {hit_raw['a']}/{n} vs "
          f"{hit_raw['b']}/{n}, discordant {disc_raw_a + disc_raw_b} "
          f"({disc_raw_a}/{disc_raw_b}), p = {p_raw:.4f}]")
    print()
    print("  cost, paired per question (positive = the ON arm spends more):")
    for key in COST_KEYS:
        ds = [float(x) for x in cost_diffs[key]]
        if not any(ds):
            print(f"    {key:<18} (not recorded in these runs)")
            continue
        s = sorted(ds)
        med = s[len(s) // 2]
        clo, chi = boot_ci(ds)
        wp, wn = wilcoxon(ds)
        print(f"    {key:<18} mean {sum(ds) / len(ds):+,.1f}  median {med:+,.1f}  "
              f"CI [{clo:+,.1f}, {chi:+,.1f}]  Wilcoxon p = {wp:.4f} (n={wn})"
              f"{'  SIG' if wp < 0.05 else ''}")
        if (clo < 0 and chi < 0) and wp >= 0.05:
            print(f"    {'':<18} ** mean CI excludes 0 but Wilcoxon does not: heavy "
                  f"tail, not a per-question saving **")
    print()
    return {"n": n, "on": hit["a"], "off": hit["b"], "disc": (disc_a, disc_b), "p": p}


def main(argv: List[str]) -> None:
    pairs: List[str] = []
    vs_pairs: List[str] = []
    sink = pairs
    for a in argv:
        if a == "--vs":
            sink = vs_pairs
            continue
        sink.append(a)

    runs = sorted({r for pair in pairs for r in pair.split(":")})
    facts = [arm_facts(r) for r in runs]
    # A reasoning-regime contrast is SUPPOSED to differ in reasoning, so the
    # validity check is told which kind of comparison it is judging.
    expect_rdiff = any(_is_reasoning_contrast(*p.split(':')) for p in pairs)
    clean = print_validity(facts, expect_reasoning_diff=expect_rdiff)
    print("── ABLATION CHECK (arms may differ only in the intervention) ─────")
    clean = print_ablation_check(pairs) and clean
    print()
    if not clean:
        print("!! One or more checks failed. Numbers below describe a confounded "
              "comparison — fix the arms before quoting anything from here.\n")

    # Labelled from what the arms actually did, not from what was intended — the label
    # is the first thing a reader trusts, so it must not assert reasoning that never fired.
    mode = "ON" if any(f["reasoning_responses"] for f in facts) else "OFF"
    on = compare(pairs, f"REASONING {mode}: plan on vs plan off")
    if vs_pairs:
        off = compare(vs_pairs, "REASONING OFF (historical): plan on vs plan off")
        if on and off:
            print("── DID REASONING CHANGE THE PLAN'S VALUE? ───────────────────────")
            g_on = 100 * (on["on"] - on["off"]) / on["n"]
            g_off = 100 * (off["on"] - off["off"]) / off["n"]
            print(f"  plan gap with reasoning ON : {g_on:+.2f} pts "
                  f"(n={on['n']}, p={on['p']:.4f})")
            print(f"  plan gap with reasoning OFF: {g_off:+.2f} pts "
                  f"(n={off['n']}, p={off['p']:.4f})")
            print(f"  difference-in-differences  : {g_on - g_off:+.2f} pts")
            print("  NOT a paired test: the two contrasts are different runs, so this "
                  "is a descriptive comparison of two effect sizes, not evidence that "
                  "reasoning changed the plan's value.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1:])
