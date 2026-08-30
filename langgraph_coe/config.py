from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class TierConfig(BaseModel):
    """LLM configuration for one tier."""

    model_name: str = "Qwen3-Next-80B-A3B-Thinking-FP8"
    # None for managed providers that route by model name rather than URL
    # (e.g. ``bedrock/qwen.qwen3-32b-v1:0``): passing an api_base there makes
    # LiteLLM try to reach that host instead of the provider endpoint.
    api_base: Optional[str] = "http://n0142:4000/v1"
    api_key: Optional[str] = None  # inherits from top-level llm.api_key
    temperature: float = 0.7
    max_tokens: int = 8192
    max_input_tokens: int = 100000
    # Chars-per-token used by the input guard to estimate prompt size. Lower is
    # more conservative (trims earlier). 3.0 suits the SGLang/Qwen setup, but
    # KG-heavy prompts (QIDs, triples, JSON) tokenize closer to ~2.1 chars/token,
    # so a model whose real window is tight needs this lowered or the guard
    # under-trims and the provider rejects the request outright.
    chars_per_token: float = 3.0
    top_p: float = 0.95
    # Extra sampling controls. None = omit from the request so SGLang's own
    # default applies (don't send a neutral value you don't intend). top_k,
    # min_p and repetition_penalty are non-OpenAI knobs forwarded to SGLang's
    # request body via model_kwargs; presence/frequency_penalty and seed are
    # OpenAI-standard. See docs/setup_generation_params.md.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    enable_thinking: bool = True
    # Cap on *reasoning* tokens (SGLang custom logit processor forces </think>
    # once spent). None = no cap. Only takes effect when enable_thinking is True
    # and the model's think-token ids are known (see langgraph_coe.thinking_budget).
    # Must be < max_tokens, which still bounds thinking + answer combined.
    thinking_budget: Optional[int] = None
    # Reasoning toggle for *managed providers* (Bedrock), which ignore the SGLang
    # mechanism above: ``enable_thinking`` rides in ``chat_template_kwargs``, which
    # only a self-hosted chat template reads, so on Bedrock it is silently inert and
    # ``thinking_budget``'s logit processor cannot run at all.
    #
    # Bedrock's own switch is ``additionalModelRequestFields={"reasoning_effort": ...}``.
    # Probed live against bedrock/qwen.qwen3-32b-v1:0 in us-west-2:
    #
    #   * Only "high" engages it. The gateway advertises an enum of high/low/max/
    #     medium/minimal/none/xhigh, but that is the *gateway's* enum, not the model's:
    #     none/low/medium return 4-6 output tokens and no reasoning block, minimal and
    #     xhigh pass the gateway then fail in the backend, and max is rejected outright.
    #     So this field is effectively None | "high" here — anything else is a silent
    #     no-op that would make a "reasoning on" arm a duplicate of the off arm.
    #   * LiteLLM drops the field unless it is allow-listed, hence the
    #     ``allowed_openai_params`` companion in ``llm.get_model_by_tier``. Passing
    #     ``reasoning_effort`` alone measured identical to omitting it (30 output
    #     tokens, no reasoning) — a passthrough that fails silently.
    #   * The Anthropic-style ``{"thinking": {...}}`` block is inert: it returns 200
    #     with no reasoning. It is not a fallback signal.
    #
    # Reasoning tokens are billed inside ``max_tokens`` (they arrive as
    # ``completion_tokens_details.reasoning_tokens``), so raising this without raising
    # max_tokens truncates the JSON payload instead of the reasoning. Calibrated on this
    # workload: at 4096 ``open_ie`` truncated on 15% of its completions; at 8192, zero.
    #
    # **Measured, and left off by default.** Reasoning costs ~19.5k tokens/question =
    # **77% of all output tokens** and bought no measured accuracy. On CoT over 117 pairs
    # (``r_cot_plan_*``) it left the plan contrast a null (+1.71 pts, p = 0.8145, i.e. the
    # same verdict as reasoning-off) and was itself directionally *negative* against the
    # reasoning-off runs, -4.3 to -5.1 pts at p = 0.38-0.46 — suggestive but confounded,
    # because those runs predate both current code and the current retrieval regime.
    #
    # Where the loss sits is clearer than whether it is real: **retrieval, not
    # conversion.** Gold reached memory on 25.6% of questions against 30.8-43.8%
    # reasoning-off, while conversion held at 80.0% vs 80.6%. Two large distributional
    # shifts accompany it — consolidated memory items 3.5 vs 6.1 (ten reasoning-off runs
    # on current code span a tight 5.62-6.68), and single-hop terminations on 23% of
    # questions against 0-5%. The tempting inference that early stopping truncates the
    # multi-hop chain is NOT supported: those 1-hop questions score *above* the run
    # average (32.0%), so the early stop is selection, not damage.
    #
    # **If you do want reasoning, enable it SELECTIVELY.** On the roles that reason over
    # evidence it is harmless; on the extraction and consolidation roles it suppresses
    # evidence, because those are recall tasks with asymmetric costs (a spurious fact
    # merely occupies a memory slot, a dropped fact is unrecoverable) and reasoning makes
    # the model filter harder — 38% fewer facts extracted, ~50% fewer memory items kept.
    # Reasoning on heavy+plan only, with memory_consolidation moved off the reasoning
    # tier, gave identical accuracy for 59.7% less reasoning spend (p < 0.0001).
    # config.eval.yaml is wired that way; see docs/RESULTS.md and §14.10/§14.15 of
    # docs/plan_idea_and_results.md.
    #
    # See docs/plan_idea_and_results.md §14.
    reasoning_effort: Optional[str] = None
    max_retries: int = 3
    timeout: int = 300


class LLMConfig(BaseModel):
    """Extended LLM config with tier-based model selection."""

    api_key: Optional[str] = None
    tiers: Dict[str, TierConfig] = {
        "heavy": TierConfig(
            model_name="Qwen3-Next-80B-A3B-Thinking-FP8",
            temperature=0.7,
            max_tokens=8192,
            enable_thinking=True,
        ),
        "medium": TierConfig(
            model_name="Qwen3-Next-80B-A3B-Thinking-FP8",
            temperature=0.4,
            max_tokens=4096,
            enable_thinking=False,
        ),
        "light": TierConfig(
            model_name="Qwen3-32B-FP8",
            temperature=0.2,
            max_tokens=2048,
            enable_thinking=False,
        ),
        # Minimal tier for roles whose output is small and bounded
        # (e.g. a list of ≤16 indices, a short entity list). Thinking off, tight
        # token ceiling — these cannot need more, so the cap only trims worst-case
        # decode and lets the server schedule tighter. Never put a reasoning or
        # open-ended-list role here.
        "classify": TierConfig(
            model_name="Qwen3-32B-FP8",
            temperature=0.0,
            max_tokens=1024,
            enable_thinking=False,
        ),
        # Planning tier. A plan makes no claim about the world, so there is nothing
        # to be right about and nothing to verify — which makes breadth, not
        # precision, the thing worth buying. Hotter than the reasoning tiers so the
        # n=3 samples actually differ; ``select_plan`` then picks among them. Keep
        # reasoning roles cold and verified.
        "plan": TierConfig(
            model_name="Qwen3-Next-80B-A3B-Thinking-FP8",
            temperature=1.0,
            max_tokens=4096,
            enable_thinking=True,
        ),
    }
    role_tiers: Dict[str, str] = {
        # Heavy reasoning roles
        "answer_generator": "heavy",
        "self_corrector": "heavy",
        "final_answer_synthesizer": "heavy",
        "subquestion_generator": "heavy",
        "reasoning_synthesizer": "heavy",
        "planner": "plan",
        # Medium structured-output roles
        "memory_consolidation": "medium",
        "relation_extraction": "medium",
        "open_ie": "medium",
        # verifier stays on medium pending the accuracy eval (its rating is the
        # MCTS reward; do not shrink its budget until measured).
        "verifier": "medium",
        "evaluator": "medium",
        # Bounded-output classification roles: tiny, fixed-shape outputs.
        "triple_pruner": "classify",  # keep_indices: ≤16 ints
        "named_entity_recognition": "classify",  # short entity list
        "question_rephraser": "classify",  # single rephrased query
        # Light extraction / agent roles
        "web_researcher": "light",
        "extractor": "light",
        # KG subgraph: LangChain create_agent tool-calling agents
        "kg_ner_agent": "light",
        "kg_triple_search_agent": "medium",
    }


class WebSearchConfig(BaseModel):
    # Off by default for paper-parity (the reference framework keeps web search
    # disabled in its fair-comparison setup; KG + corpus are the active surfaces).
    # When False, ``CoTGraph`` skips the ``web_one`` ReAct fan-out entirely.
    enabled: bool = False
    api_key: Optional[str] = None
    top_k: int = 5
    crawl_full_text: bool = True
    max_crawl_requests_per_second: float = 2.0
    max_queries_per_agent: int = 2
    # Paid last-resort provider, used ONLY when the free providers are all blocked.
    # Prefer the ``TAVILY_API_KEY`` environment variable over setting this: a key in
    # a YAML file gets committed. Leave both unset to disable the fallback entirely,
    # in which case a blocked search returns empty as it does today.
    tavily_api_key: Optional[str] = None
    # Brave Search API. Keyed and quota'd (2k queries/month on the free tier), so it
    # sits behind the unmetered providers. ``BRAVE_API_KEY`` env var takes precedence.
    brave_api_key: Optional[str] = None

    # ``providers`` is the fallback chain, tried in order until one returns a
    # non-empty result. Names must be keys of ``tools.web._PROVIDERS``:
    #
    #   builtin   Serper when ``api_key``/``SERPER_API_KEY`` is set, else the free
    #             ``ddgs`` rotation. The legacy slot, and the one unit tests stub.
    #   searxng   Local SearXNG container (``setup/searxng_up.sh``). Unmetered.
    #   wikipedia MediaWiki search API. Unmetered, no key, returns page text inline.
    #   brave     Brave Search API. Needs ``brave_api_key``.
    #   tavily    Billed. Needs ``tavily_api_key``.
    #
    # The default is deliberately the pre-existing behavior (builtin → tavily) so
    # that enabling the newer providers is an explicit, per-config decision rather
    # than something that silently changes what a run retrieves. See
    # ``config.eval.yaml`` for the chain the sweeps actually use.
    providers: List[str] = Field(default_factory=lambda: ["builtin", "tavily"])
    # Base URL of the local SearXNG instance. Only read when ``searxng`` is in
    # ``providers``. ``SEARXNG_URL`` env var takes precedence.
    searxng_url: str = "http://localhost:8080"
    # MediaWiki site queried by the ``wikipedia`` provider.
    wikipedia_lang: str = "en"


class CacheRedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    llm_db: int = 0
    wikidata_db: int = 1


class CacheWikidataConfig(BaseModel):
    entity_ttl: int = 2_592_000
    search_ttl: int = 604_800
    triples_ttl: int = 604_800
    enrich_ttl: int = 2_592_000


class CacheWebConfig(BaseModel):
    ttl: int = 86_400


class CacheConfig(BaseModel):
    enabled: bool = False
    redis: CacheRedisConfig = Field(default_factory=CacheRedisConfig)
    wikidata: CacheWikidataConfig = Field(default_factory=CacheWikidataConfig)
    web: CacheWebConfig = Field(default_factory=CacheWebConfig)


class MCTSConfig(BaseModel):
    """Knobs for the MCTS strategy graph."""

    num_iterations: int = 15
    exploration_weight: float = 2.0
    max_tree_depth: int = 10
    # Per-rollout CoTGraph depth. Lower than coe's 5 because each rollout step
    # now runs a full CoT iteration (retrieval + rerank + extractor + memory).
    max_simulation_depth: int = 3
    # Floor on iterations before any early-termination condition (high-confidence,
    # semantic-sufficiency, convergence-patience) may fire. ``num_iterations``
    # always wins as the hard cap. Honored by ``route_after_iteration``.
    min_iterations: int = 5
    # Minimum tree depth (root = 0) before expand emits a FINAL_ANSWER child from
    # ``_gen_final``. Matches coe ``should_explore = depth < 2`` — skip synthesis
    # while memory is still shallow (legacy skipped retrieval there; we skip the
    # whole final-answer expansion call).
    final_answer_min_depth: int = 2
    high_confidence_threshold: float = 0.9
    convergence_patience: int = 5
    semantic_sufficiency_count: int = 5
    # LangGraph superstep budget for one MCTS run. Each iteration is ~7 supersteps
    # (select→expand→simulate→evaluate→backprop→mem_update→route); size with
    # headroom over ``num_iterations × 7``.
    recursion_limit: int = 150
    # Snapshot ``text_memory`` / ``graph_memory`` / ``entity_dict`` per tree node so
    # a branch's retrieval writes stay inside its own subtree.
    #
    # Default False preserves documented coe parity: memory is a single shared
    # channel that rollouts mutate directly (see the module docstring). That
    # sharing is what makes a rejected branch's evidence permanent — its writes
    # survive even though its value is discarded, and ``evaluate``'s
    # memory-grounded verifier views then score later candidates against them. Any
    # design that treats a subtree as cheap to explore (e.g. replan-as-action)
    # needs this on, so the flag is what makes the two regimes comparable.
    branch_local_memory: bool = False


class CoTConfig(BaseModel):
    """Knobs for the standalone CoT strategy graph."""

    # Maximum decomposition depth (CoT loop iterations) before forced synthesis.
    # Drives ``route_after_subq``; without it the loop finalizes immediately.
    max_depth: int = 5
    # LangGraph superstep budget for one CoT run. Each iteration is ~8 supersteps
    # (gen_subq→fan-out→rerank→extract→subanswers→mem_update→increment); size with
    # headroom over ``max_depth × 8``.
    recursion_limit: int = 75

    # Present the LATEST evidence first in ``candidate_answers``.
    #
    # ``candidate_answers`` is the whole of ``text_memory`` in oldest-first order, rendered as
    # an explicit numbered list, and it is ~94% of what the synthesiser reads (``resolved_
    # findings`` is 0.49 lines per question). Measured over 149 conversion failures where the
    # gold-bearing memory line and the line the answer came from are distinct, length-matched
    # so a short prediction cannot win by matching an early line more readily: the gold sits
    # **later** in memory **67 times against 41, p = 0.0157**, by a mean 0.089 of the list.
    # (Unmatched the same test reads 101/48 at p < 0.0001, which overstates it; and an
    # unpaired check across all questions puts the gold at mean position 0.425 against the
    # wrong answer's 0.450, i.e. disagrees in sign. So this is a modest effect on a weak
    # prior, kept off until the A/B rules.)
    #
    # Two rival orderings were tested on the same cases and both favour the WRONG line, so
    # relevance ranking is deliberately NOT used here: question content-word overlap (gold 28
    # / rival 71, p < 0.0001) and idf-weighted overlap (35 / 78, p = 0.0001).
    #
    # **Measured and left off.** `ro_on` vs `ro_off`, 117 paired questions, plan disabled in
    # both: paired conversion 18/26 vs 21/26 — **gap -3**, theta=0.286, p=0.4531 — and accuracy
    # 22.22% vs 23.93%, 22 discordant (10/12), p=0.8318. Null and directionally *negative*,
    # which is the direction the unpaired positional check had already pointed to. So the
    # positional asymmetry is real but too small to act on, and the synthesiser is not
    # meaningfully primacy-biased over an ~8-item list.
    #
    # Recorded so this is not retried: of the seven interventions measured against conversion,
    # this is the one that tested the 94% channel directly, and it moved nothing.
    recent_evidence_first: bool = False

    # Show the synthesiser the retrieved facts that CONSOLIDATION DROPPED.
    #
    # Measured on a 60-question instrumented run (``results/cl_probe``, via the new
    # ``retrieval_log`` channel, which accumulates every extracted fact because
    # ``extracted_facts`` is cleared each hop and nothing else preserved it):
    #
    #   gold in retrieved facts (pre-consolidation)   22/60 = 36.7%
    #   gold in consolidated text_memory              19/60 = 31.7%
    #   **lost by consolidation**                     **6 = 10.0% of all questions**
    #   of those 6, answered wrong                    **6 (all of them)**
    #
    # So 27% of the questions whose retrieval found the gold lose it before synthesis ever
    # sees it, and none recovered. MEMORY_CONSOLIDATOR keeps a mean 0.56 of the retrieved
    # facts — as low as 0.08 on individual questions — deciding per hop, without knowing
    # which fact the final answer will need.
    #
    # This is the only conversion channel that was not exhausted: the ledger is ~6% of the
    # synthesiser's input and four ledger fixes measured null; candidate ordering is the other
    # 94% and measured null too. Dropped evidence is neither — it is material that never
    # reached synthesis at all.
    #
    # Only the *dropped* facts are appended, capped at 25, and placed LAST with an explicit
    # lower-reliability label: the scaffolding result established that content at top
    # authority gets returned as the answer whether it deserves to be or not.
    #
    # **MEASURED AND REFUTED — left off.** ``rd_on`` vs ``r_cot_selective``, 117 pairs,
    # identical but for this flag: accuracy 23/117 vs 26/117 (-2.56 pts, p = 0.6072), and on
    # the 22 questions where *both* arms held the gold the two convert **identically**,
    # 15/22 each, discordant 4 (2/2), p = 1.0000. Cost moved exactly as designed (prompt
    # tokens +10,378, calls -0.2), so the mechanism was wired correctly — it simply does
    # nothing. Putting consolidation-discarded evidence back in front of the synthesiser
    # does not recover it; the most likely reason is the §8 position-and-authority result,
    # i.e. the same last-place lower-reliability framing that makes this safe also makes it
    # inert. See docs/plan_idea_and_results.md §14.13.
    synthesis_sees_dropped_evidence: bool = False



class PlanConfig(BaseModel):
    """Knobs for the explicit plan channel (planning/reasoning separation).

    A *plan* is prose stating what to find out. It conditions
    ``SUBQUESTION_GENERATOR`` and ``SELF_CORRECTOR`` via a dedicated prompt field
    and never enters ``text_memory`` — an interrogative in memory becomes a
    retrieval query, then verifier grounding, then a synthesis candidate.

    Two operations act on it. **UPDATE** is a deterministic, LLM-free write that
    closes a plan intent and surfaces its resolved binding through the
    ``intermediate_answer`` slot. **REPLAN** is one PLANNER call, fired by
    ``plan_gate`` on *contested discharge* (two or more distinct-QID referents
    survive for one intent) or *falsified discharge* (a fact the plan cited was
    evicted by retrieval).

    **On by default, and the reason is cost, not accuracy.** The accuracy effect is a
    measured null across eight paired experiments (see below and docs/RESULTS.md); what
    replicates is a substantial reduction in spend at unchanged accuracy — CoT -24.8 LLM
    calls per question (p < 0.0001) and -36,993 prompt tokens (p = 0.0179), MCTS
    plan-rollout -3.9 calls (p = 0.0274) and -15,023 prompt tokens (p = 0.0049). Four
    independent measurements, all surviving Wilcoxon, in both reasoning regimes.

    Set ``enabled=false`` to recover the A0 baseline, which is byte-identical to the
    pre-plan behaviour: when this is False ``gen_plan``/``plan_gate``/``replan`` are not
    added to the graph at all, so the ablation stays meaningful in both directions.

    **Measured effect, 62 rows, paired, identical code:** in CoT the plan scored
    36/62 against 31/62 without it and used 4.94 subquestions per question against
    5.66 — but only 17 rows were discordant (11 to 6), a two-sided sign test of
    p = 0.33, so the accuracy difference is not established.

    **In MCTS it is actively harmful to the search, and the reason matters.** With
    the plan, sibling rollouts ask the same things: mean sibling-subtree overlap
    23.1% against 10.2%, and distinct subquestions per question 6.3 against 9.9 — so
    the tree covers roughly a third less ground for the same cost, at identical
    accuracy (14/23 both). The design treated sibling re-decomposition as duplicated
    work; in a tree search it is *exploration*, and a shared plan converts it into
    repetition.

    That harm is specific to ``mcts_plan_scope="tree"`` and is why the default is
    ``"rollout"``: each rollout plans from its own branch memory, so plans differ where
    the branches differ and sibling diversity is preserved by construction. Under
    ``"rollout"`` the MCTS contrast is an accuracy null with a real cost saving, in both
    reasoning regimes — so the plan is safe to leave on for MCTS too.
    """

    enabled: bool = True
    # 0 keeps ``plan_gate`` in log-only mode: ``plan_action`` is computed and
    # recorded, but the router never takes the replan edge.
    #
    # Measured, not merely cautious. Armed at 2 on the 62-row set, replan fired on 20
    # questions and scored 10 against 9 for both the log-only and the no-plan arm on
    # those same 20 — while costing 2.55 hops / 8.95 subquestions against 1.24 / 3.02
    # on the questions that did not replan, plus up to 2 extra PLANNER calls. The
    # trigger is not broken; it fires on referents that are genuinely ambiguous in the
    # world, and rewriting *what to ask* cannot settle that.
    replan_max: int = 0
    # Refuse to replan within this many iterations of ``max_depth`` — a plan
    # rewritten on the last hop cannot be acted on.
    replan_min_depth_headroom: int = 2
    # Attempts against one intent before it counts as stalled. This is the only
    # trigger branch that fires when nothing surprising happens, so it is what keeps
    # the plan from being revised never rather than rarely: contested and falsified
    # discharge both need an *event*, and an intent that quietly returns nothing
    # produces none.
    #
    # 3, from the run logs: 93-95% of intents that eventually closed did so within 3
    # attempts, but only 81-89% within 2. It must also stay above the per-intent
    # pooling cap (``cot._MAX_PER_INTENT``), or a single hop of ordinary work
    # exhausts the budget and every unresolved intent reads as stalled.
    stall_after_attempts: int = 3
    # MCTS only: minimum gap between the closed-book verifier rating and the lowest
    # memory-grounded one before it counts as the evidence disagreeing with the
    # answer.
    #
    # Measured against the verifier's own noise floor (E5,
    # ``scripts/verifier_noise_floor.py``): five identical calls per case at
    # temperature 0.7 spread a mean of 1.5 and a max of 3.0 over 30 calls. A
    # threshold at or below 3.0 therefore fires on resampling alone. 4.0 clears the
    # observed floor by one point, which is a thinner margin than it looks — the max
    # over more samples is likely higher.
    memory_disagreement_threshold: float = 4.0
    # Verify a *terminal* intent's referent against this hop's evidence with
    # ``SELF_CORRECTOR`` before closing on it, replacing the answer when the evidence
    # supports a different one and withholding it when nothing is corroborated.
    #
    # Measured motivation: on 70 questions whose memory held the gold answer and whose
    # final answer was still wrong, the terminal intent had closed on the *wrong*
    # referent 67% of the time, and only 3% of those were flagged by the consolidator's
    # conflict detection — the rest were silent. The cause is structural: binding
    # candidates come from one ``current_subanswers_concise`` entry per subquestion, so
    # when the answer generator picks wrong that value is the only candidate and
    # ``count_rival_referents`` sees no rivalry to contest.
    #
    # Terminal intents only. ``SELF_CORRECTOR`` is reused rather than a new role: it
    # already returns correct/partial/incorrect/unsupported plus a refinement, and was
    # wired only into MCTS despite this docstring claiming the plan conditions it.
    #
    # **Measured and defaulted OFF.** ``vf_on`` vs ``vf_off``, 117 paired questions, both
    # arms with the plan enabled so nothing else differs:
    #
    #   * paired conversion — of questions whose memory held the gold in BOTH arms, the
    #     fraction answered correctly — 14/18 vs 14/18. theta=0.500, p=1.0000. Zero.
    #   * accuracy +4.3 pts (23.08% vs 18.80%), 21 discordant (13/8), p=0.3833. The gain
    #     is real but comes from *retrieval*, not conversion: gold-in-memory rose 24.8% ->
    #     29.9%, because withholding an unverified referent leaves the intent open and
    #     buys another hop of search. An extra hop is a cheaper way to buy an extra hop.
    #   * cost +5.8 calls/question, Wilcoxon p=0.0236 — 3.03 of them SELF_CORRECTOR, the
    #     rest the extra hops. The estimate above of "~1 extra call" was wrong: the gate
    #     runs per hop over a mean 3.67 hops, not once per question.
    #
    # So it fails the cost half of the goal outright and delivers nothing on the metric
    # it was built for. Kept because the code is sound and the role is now wired into CoT
    # for future use, but off by default.
    verify_terminal_referents: bool = False

    # Treat a *guard* — an intent whose answer is a truth value, "determine whether she
    # had a spouse" — as a guard rather than as a referent-bearing intent. The PLANNER
    # prompt asks for presuppositions to be hedged into conditionals, so guards are
    # generated deliberately, and every consumer of the ledger assumed a referent.
    #
    # Measured over 1,920 questions / 6,250 intents: 399 guards (6.4% of intents), 188 of
    # them terminal, 131 of those closed, and 139 of 284 closed guards bound a *full
    # sentence* as their referent. Three concrete harms:
    #
    #   * ``resolved_findings`` is ranked above every other source in the synthesis
    #     prompt, so a closed guard arrived there as "Confirm whether the author wrote a
    #     short story -> No, Stephen King did not write a short story featuring Herman
    #     Wouk." on a question whose gold was 1,335,907. Same mechanism that made
    #     scaffolding referents cost ~4.7 points.
    #   * resolving the answer as a referent bound the intent's own *input* — "No,
    #     Yangzhou is not a capital city" -> Yangzhou — which either closed the intent on
    #     the subject or manufactured a rival against the real answer. 16% of intents with
    #     bindings echoed a prerequisite's referent.
    #   * "Yes, Meg Ryan." resolved to Dennis Quaid (the subject) rather than Meg Ryan,
    #     because the whole sentence was the surface. The affirmation is now stripped
    #     before resolution.
    #
    # Off restores the previous behaviour exactly, which is what makes the A/B a single
    # variable rather than a code-version comparison.
    #
    # **Measured: mechanism confirmed, outcome null.** `gd_on` vs `gd_off`, 117 paired
    # questions, concurrent arms: paired conversion 17/23 vs 18/23 (p=1.0000), accuracy
    # 26/117 vs 26/117 (18 discordant 9/9, p=1.0000), calls -5.4/question (Wilcoxon
    # p=0.0874, n.s.), intermediate-referent leak 1.7% vs 5.2% (p=0.2891).
    #
    # The reason it cannot move accuracy is structural: ``candidate_answers`` is the entire
    # consolidated memory (7.7-8.9 items) while ``resolved_findings`` is 0.49 lines per
    # question, so the ledger supplies ~6% of what synthesis reads however correct it is.
    # That ceiling explains this null and the ``verify_terminal_referents`` one.
    #
    # Kept on as **correctness housekeeping, not a measured improvement**: closing an intent
    # on the string "No, Stephen King did not write a short story featuring Herman Wouk" is
    # wrong whatever the accuracy does, it is directionally cheaper rather than dearer, and
    # it takes the intermediate-referent leak from 5.2% to 1.7%.
    guard_intents_are_not_referents: bool = True

    # Stop an intent binding the referent its own prerequisites already bound. The QID
    # resolver takes the *earliest* linked entity in the answer, which is right for a
    # concise answer and exactly backwards for a sentence one: in "Dennis Quaid is married
    # to Meg Ryan" the earliest entity is the subject the intent was asked *about* and the
    # answer sits in the predicate. Measured on 5,593 intents with bindings, 897 (16%)
    # resolved to a prerequisite's referent and 640 closed on nothing else.
    #
    # Separate from ``guard_intents_are_not_referents`` and defaulted OFF so the two are
    # measured independently rather than as one bundle — they overlap in the cases they
    # touch, and attributing a pooled effect to whichever half is cheaper to explain is
    # how the earlier retracted claims in this project happened.
    #
    # **Measured and left off.** `bd_input` vs `gd_on`, 117 paired questions: paired
    # conversion 15/19 vs 15/19 (gap 0, p=1.0000), accuracy 18.80% vs 22.22% — gap **-4**,
    # 18 discordant (7/11), theta=0.389, p=0.4807 — and cost flat (Wilcoxon p=0.5427). Not
    # significant in either direction, but with zero conversion effect and a negative
    # accuracy sign there is no case for enabling it. The mechanism is real (640 intents
    # closed on nothing but their own input referent); it just does not reach the answer,
    # for the same reason as every other ledger fix here — see
    # ``guard_intents_are_not_referents`` on the ~6% ceiling.
    skip_input_referent_in_binding: bool = False

    # Let a sub-answer whose self-reported ``confidence_level`` is low still bind a
    # referent when a ``[Retrieval]`` line corroborates it.
    #
    # ``plan_gate`` discards low-confidence answers outright. But ``ANSWER_GENERATOR`` is
    # told to "always try your best to answer the question even if the context is
    # incomplete or missing" (rule 3), so a low label frequently reports doubt about the
    # *context*, not about the referent — and when a [Retrieval] line does corroborate the
    # answer, that doubt has already been resolved by evidence. Measured over 6,453 hops,
    # 2,980 of 17,848 sub-answers (16.7%) were dropped this way before binding — the
    # largest single source of attrition, ahead of missing intent attribution (6.6%).
    #
    # Corroboration is the arbiter, so the original concern survives: an *uncorroborated*
    # guess still cannot bind, and so still cannot compete with a grounded answer.
    #
    # **Measured and left off: the premise was wrong.** On the live arm the rescue fired on
    # **1 of 85** low-confidence drops. The self-reported label and [Retrieval] corroboration
    # agree almost perfectly — ~99% of low-confidence answers are also uncorroborated — so
    # the gate was not discarding recoverable answers and there is nothing here to recover.
    # The 22-28% attrition is real but the answers were genuinely unsupported.
    bind_corroborated_low_confidence: bool = False

    # Where the plan lives under MCTS: ``"tree"`` (one plan, generated at the root and
    # inherited by every node and rollout) or ``"rollout"`` (no plan in the tree; each
    # rollout runs the CoT loop and generates its own).
    #
    # ``"tree"`` was measured to harm the search. Over 23 paired rows at
    # ``num_iterations=2``, sharing one plan cut distinct subquestions per question from
    # 9.9 to 6.3 (-36%) and more than doubled sibling-subtree overlap, 10.2% -> 23.1%,
    # at identical accuracy. The cause is structural: a plan is a variance-*reduction*
    # device, and pUCT can only prefer one child over another when the children differ.
    # Sharing one plan across siblings makes them converge, so the tree re-confirms one
    # line of enquiry instead of comparing several.
    #
    # ``"rollout"`` separates the two effects. The plan's benefits are all *within* a
    # single reasoning chain — grounded retrieval queries, a scoped triple-pruner
    # budget, scaffolding excluded from synthesis — and none of them require siblings to
    # share anything. Each rollout plans from its own branch memory, so plans differ
    # where the branches differ and diversity is preserved by construction.
    mcts_plan_scope: str = "rollout"


class SearchConfig(BaseModel):
    """Top-level strategy selection (``system.py`` reads ``strategy``)."""

    strategy: str = "mcts"  # "mcts" | "cot"
    cot: CoTConfig = Field(default_factory=CoTConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)


class MemoryConfig(BaseModel):
    """Working-memory limits used by ``MemoryUpdateGraph`` and the CoT extractor."""

    max_textual_memory_tokens: int = 16_384
    prune_batch_size: int = 16
    # CoT extractor: split joined reranked passages into char-budgeted batches
    # so the EXTRACTOR's input never exceeds the model's context window.
    # Default ≈ 24k characters ≈ 7k tokens — well under any tier's max_input_tokens
    # and leaves room for system prompt + thinking + structured output.
    extractor_max_input_chars: int = 24_000


class EmbedderConfig(BaseModel):
    model_name: str = "Qwen3-Embedding-4B"
    url: str = "http://n0385:4000/v1"
    api_key: Optional[str] = None
    # Qwen3-Embedding is asymmetric: passages are embedded raw (build_index.py) but
    # queries must carry this instruction prefix ({query} placeholder).
    query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: {query}"
    )


class CorpusConfig(BaseModel):
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    # LangChain bundle (<name>.faiss + <name>.pkl) or raw HF-datasets index.
    index_path: str = (
        "/gpfs/projects/uonlp/hieum/wemg/retriever_corpora/Qwen3-4B-Emb-index.faiss"
    )
    search_k: int = 10  # FAISS fetch depth before optional rerank / caller top_k cap
    # HF hub id or local ``load_from_disk`` path (same field name as legacy ``coe``).
    corpus_path: Optional[str] = "Hieuman/wiki23-processed"
    # Alias used in raw-index FAISS layout; kept in sync with ``corpus_path`` when one is set.
    corpus_dataset: Optional[str] = None
    corpus_split: str = "train"
    text_column: str = "contents"

    @model_validator(mode="after")
    def _sync_corpus_source_fields(self) -> "CorpusConfig":
        """Allow either ``corpus_path`` or ``corpus_dataset`` in YAML (legacy vs langgraph name)."""
        path = (self.corpus_path or "").strip() or None
        dataset = (self.corpus_dataset or "").strip() or None
        resolved = path or dataset
        if resolved is None:
            return self
        return self.model_copy(
            update={"corpus_path": resolved, "corpus_dataset": resolved}
        )

    def resolved_corpus_source(self) -> Optional[str]:
        """HF dataset / disk path for raw-index docstore lookup."""
        return self.corpus_dataset or self.corpus_path


class RetrieverConfig(BaseModel):
    # Corpus retrieval is the recall floor for every subquestion, so it is on by
    # default. Set False when no local FAISS index / embedding server is available:
    # ``corpus_search`` raises when the pipeline is uninitialised, and
    # ``_init_runtime`` deliberately swallows the init failure, so without this
    # flag the graphs would fan out into a guaranteed RuntimeError on every hop.
    enabled: bool = True
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen3-Reranker-4B"
    url: str = "http://n0999:30002/v1"
    api_key: Optional[str] = "EMPTY"
    top_k: int = 10
    # Qwen3-Reranker SGLang template (<Instruct>: …); sent as JSON ``instruct``.
    instruction: Optional[str] = (
        "Given a web search query, retrieve relevant passages that answer the query."
    )


class WikidataConfig(BaseModel):
    """Configuration for the Wikidata knowledge-graph tools."""

    # Local QEndpoint (e.g. ``http://127.0.0.1:30162/api/endpoint/sparql`` via SSH tunnel).
    # None → public ``https://query.wikidata.org/sparql``.
    sparql_endpoint: Optional[str] = None

    # SPARQL / Wikipedia rate limits
    max_sparql_rps: float = 2.0
    max_wikipedia_rps: float = 10.0
    triple_cache_max_entries: int = 5000

    # Loop prevention: maximum number of fetch_and_prune_subgraph calls per question
    max_hops: int = 3

    # Stage A pruning knobs (reranker-based)
    reranker_url: Optional[str] = "http://n0999:30002/v1"
    reranker_model: Optional[str] = "Qwen3-Reranker-4B"
    # Task instruction sent to Qwen3-Reranker as the ``instruct`` field. Tells the
    # model what "relevant" means for triple selection; without it the reranker
    # saturates (~0.998 flat scores) on short, near-identical triple strings.
    reranker_instruction: Optional[str] = (
        "Given a question, retrieve the knowledge-graph triples whose relation and "
        "object are needed to answer it."
    )
    pruning_top_k: int = 64  # max triples kept after Stage A
    pruning_delta: float = 0.05  # score tolerance below the top score


class LangGraphCoeConfig(BaseModel):
    """Root settings for ``langgraph_coe`` loaded from ``config.yaml``."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    wikidata: WikidataConfig = Field(default_factory=WikidataConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @staticmethod
    def default_yaml_path() -> Path:
        return Path(__file__).resolve().parent / "config.yaml"

    @classmethod
    def from_yaml(
        cls, path: Optional[Path | str] = None, *, merge_api_key_env: bool = True
    ) -> LangGraphCoeConfig:
        """Load from YAML; when *path* is omitted, use ``default_yaml_path()`` if it exists."""

        p = Path(path) if path is not None else cls.default_yaml_path()
        raw: Dict[str, Any] | None = None
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        merged = cls.model_validate(raw or {})
        if merge_api_key_env:
            key = (
                merged.llm.api_key
                or os.environ.get("API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            merged.llm.api_key = key
            if not merged.retriever.corpus.embedder.api_key:
                merged.retriever.corpus.embedder.api_key = key
            if not merged.reranker.api_key:
                merged.reranker.api_key = key or "EMPTY"

            # Optional runtime overrides (YAML in config.yaml is the default source of truth).
            env_index = os.environ.get("LANGGRAPH_CORPUS_INDEX_PATH") or os.environ.get(
                "SAR_CORPUS_INDEX_PATH"
            )
            if env_index:
                merged.retriever.corpus.index_path = env_index

            env_corpus = (
                os.environ.get("LANGGRAPH_CORPUS_DATASET")
                or os.environ.get("LANGGRAPH_CORPUS_PATH")
                or os.environ.get("SAR_CORPUS_DATASET")
                or os.environ.get("SAR_CORPUS_PATH")
            )
            if env_corpus:
                merged.retriever.corpus.corpus_path = env_corpus
                merged.retriever.corpus.corpus_dataset = env_corpus

            env_embed_url = os.environ.get("LANGGRAPH_TEST_EMBED_URL")
            if env_embed_url:
                merged.retriever.corpus.embedder.url = env_embed_url

            env_embed_model = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL")
            if env_embed_model:
                merged.retriever.corpus.embedder.model_name = env_embed_model

            env_rerank_url = os.environ.get("LANGGRAPH_TEST_RERANK_URL")
            if env_rerank_url:
                merged.reranker.url = env_rerank_url

            env_rerank_model = os.environ.get("LANGGRAPH_TEST_RERANK_MODEL")
            if env_rerank_model:
                merged.reranker.model_name = env_rerank_model

        return merged
