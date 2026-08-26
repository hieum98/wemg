"""LangGraph-CoE: LLM Execution Layer with tier-based model selection."""

import asyncio
import contextvars
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from .config import LLMConfig
from .parsing import extract_info_from_text, extraction_type_from_annotation
from .roles import Role
from .thinking_budget import build_request_kwargs

logger = logging.getLogger(__name__)

# When a role is executed over a LIST of inputs, the items are independent so
# they run concurrently up to this cap (a bounded ``asyncio.gather``). Previously
# list items were awaited strictly one-at-a-time, which serialized e.g. answering
# the N sub-questions in an MCTS/CoT expand into N sequential round-trips. Set
# ``LANGGRAPH_ROLE_ITEM_CONCURRENCY=1`` to restore the old sequential behavior.
_ROLE_ITEM_CONCURRENCY = max(
    1, int(os.environ.get("LANGGRAPH_ROLE_ITEM_CONCURRENCY", "16"))
)

# Legacy parity: when an item produces no parseable structured output, retry the
# whole batch with "shaken" sampling params (new seed + nudged temperature/top_p)
# up to this many attempts before giving up. The fixed per-tier ``seed`` makes a
# naive resample deterministic, so varying the seed is what actually yields a
# different sample. Override via ``LANGGRAPH_STRUCT_PARSE_RETRIES``.
_STRUCT_PARSE_RETRIES = max(
    1, int(os.environ.get("LANGGRAPH_STRUCT_PARSE_RETRIES", "3"))
)


def format_messages(role: Role, item: BaseModel) -> List[Any]:
    """Format prompt messages for a given role and input item."""
    system_prompt = role.system_prompt
    user_prompt = str(item)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


# Fallback chars-per-token used to estimate prompt size when the tokenizer is
# unavailable. Only a fallback: the ratio is content-dependent in a way no single
# constant survives — natural English runs ~4, KG triples and entity IDs ~2, and
# crawled pages containing CJK or minified script measured ~1.1 chars/token in
# practice, i.e. a 32k-char payload the heuristic scored as 10k tokens was really
# 29k. Prefer :func:`count_prompt_tokens`, which asks the tokenizer.
_CHARS_PER_TOKEN = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Cost meter
# ──────────────────────────────────────────────────────────────────────────────
#
# Nothing counted LLM calls, so no cost claim about this system was provable — the only
# available proxies were hops and subquestions per question, which miss every call inside
# the retrieval subgraphs and mis-weight a role sampled at n=3 against one sampled at n=1.
#
# A ContextVar rather than a module global because ``runner._answer_one`` answers
# ``max_concurrent`` questions concurrently on one event loop, so a global would blend
# them. It follows the per-question ``reset_*_session`` pattern already documented there.
# Reads are free and misses are silent: when no meter is installed (unit tests, library
# use) every hook below is a no-op, so behaviour cannot change.
_cost_meter: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "langgraph_coe_cost_meter", default=None
)


def start_cost_meter() -> Dict[str, Any]:
    """Install a fresh per-question meter and return it (call once per question)."""
    meter: Dict[str, Any] = {"calls": 0, "completions": 0, "prompt_tokens": 0, "by_role": {}}
    _cost_meter.set(meter)
    return meter


def read_cost_meter() -> Optional[Dict[str, Any]]:
    return _cost_meter.get()


def _record_cost(role_name: str, completions: int, prompt_tokens: Optional[int]) -> None:
    """Add one role invocation to the active meter, if any.

    ``completions`` is the number of *actual* model responses requested — ``n`` per
    attempt, summed over shake retries — not the number of ``execute_role_lc`` calls. A
    role at n=3 costs three times a role at n=1 and the meter has to say so.
    """
    meter = _cost_meter.get()
    if meter is None:
        return
    meter["calls"] += 1
    meter["completions"] += int(completions)
    if prompt_tokens:
        meter["prompt_tokens"] += int(prompt_tokens) * int(completions)
    slot = meter["by_role"].setdefault(
        role_name, {"calls": 0, "completions": 0, "prompt_tokens": 0}
    )
    slot["calls"] += 1
    slot["completions"] += int(completions)
    if prompt_tokens:
        slot["prompt_tokens"] += int(prompt_tokens) * int(completions)
_TRUNCATION_NOTICE = "\n\n…[input truncated to fit the model context budget]…\n\n"


def count_prompt_tokens(messages: Sequence[Any], model: Optional[str]) -> Optional[int]:
    """Exact prompt token count via the model's tokenizer, or None if unavailable.

    ``litellm.token_counter`` resolves the right tokenizer per provider, so this
    is the only measure that holds across the SGLang and Bedrock paths and across
    content types. Returns None (rather than raising) when the model is unknown or
    litellm cannot tokenize it, so callers fall back to the char heuristic.
    """
    if not model or not messages:
        return None
    try:
        import litellm

        payload = [
            {
                "role": getattr(m, "type", None) == "system" and "system" or "user",
                "content": _content_to_text(getattr(m, "content", "")),
            }
            for m in messages
        ]
        return int(litellm.token_counter(model=model, messages=payload))
    except Exception:  # noqa: BLE001 — estimation must never break a graph node
        logger.debug("token_counter unavailable for model=%r", model, exc_info=True)
        return None


def _truncate_messages_to_budget(
    messages: List[Any],
    max_input_tokens: Optional[int],
    role_name: str,
    chars_per_token: float = _CHARS_PER_TOKEN,
    model: Optional[str] = None,
) -> List[Any]:
    """Cap the prompt size at ``max_input_tokens``.

    Only the final (user) message is trimmed — earlier messages (the role's
    system prompt) are preserved whole. The user payload is trimmed by keeping
    its head and tail and dropping the middle, so the question and any trailing
    instructions survive while the bulky accumulated memory/retrieval in the
    middle is dropped. Emits a WARNING when it fires — never a silent drop.

    When ``model`` is given, the char budget is derived from an **exact** token
    count for the current payload rather than a fixed chars-per-token guess. The
    guess cannot be made safe with one constant: the same 32k-char payload is 10k
    tokens of English and 29k tokens of CJK-bearing crawled markup, so a heuristic
    tuned for one silently under-trims the other.
    """
    if not max_input_tokens or max_input_tokens <= 0 or not messages:
        return messages

    actual_tokens = count_prompt_tokens(messages, model)
    if actual_tokens is not None and actual_tokens > 0:
        total_chars = sum(len(_content_to_text(m.content)) for m in messages)
        if actual_tokens <= max_input_tokens:
            return messages
        # Measured ratio for *this* payload, floored so a pathological ratio still
        # produces a positive budget.
        measured = max(0.5, total_chars / actual_tokens)
        budget_chars = int(max_input_tokens * measured)
    else:
        budget_chars = int(max_input_tokens * chars_per_token)
    sizes = [len(_content_to_text(m.content)) for m in messages]
    total = sum(sizes)
    if total <= budget_chars:
        return messages
    # Everything except the last message (the system prompt) is kept whole; the
    # last message is the trimmable user payload.
    fixed = sum(sizes[:-1])
    user_budget = budget_chars - fixed - len(_TRUNCATION_NOTICE)
    last = messages[-1]
    user_text = _content_to_text(last.content)
    if user_budget <= 0:
        logger.warning(
            "[input_guard] role=%s: non-user messages (%d chars) already exceed "
            "the budget (max_input_tokens=%d ≈ %d chars); sending unmodified — "
            "expect a context-length error.",
            role_name,
            fixed,
            max_input_tokens,
            budget_chars,
        )
        return messages
    head = user_budget // 2
    tail = user_budget - head
    truncated = user_text[:head] + _TRUNCATION_NOTICE + user_text[-tail:]
    logger.warning(
        "[input_guard] role=%s prompt ≈%d tok > max_input_tokens=%d; trimmed "
        "user payload %d→%d chars (head=%d, tail=%d).",
        role_name,
        int(total / chars_per_token),
        max_input_tokens,
        len(user_text),
        len(truncated),
        head,
        tail,
    )
    new_messages = list(messages[:-1])
    new_messages.append(type(last)(content=truncated))
    return new_messages


def _unwrap_serialized_generation(text: str) -> str:
    """Recover the real model output when ``content`` is a serialized generation.

    Reasoning models under SGLang/litellm sometimes serialize the entire
    ``ChatGeneration`` (a ``{"lc": 1, ..., "kwargs": {"text": "<real output>",
    "message": {...thinking...}}}`` blob) into the message ``content`` string.
    The user-facing output then lives, JSON-escaped, in ``kwargs.text`` while the
    message content holds only the thinking trace — so structured parsing and the
    regex fallback both fail on the wrapper. Unwrap to ``kwargs.text`` (which
    ``json.loads`` returns already-unescaped) so downstream parsing sees the
    clean output. Returns *text* unchanged when it isn't such a wrapper.
    """
    if not text.lstrip().startswith('{"lc"'):
        return text
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    kwargs = obj.get("kwargs") if isinstance(obj, dict) else None
    if isinstance(kwargs, dict) and isinstance(kwargs.get("text"), str) and kwargs["text"].strip():
        return kwargs["text"]
    return text


def _content_to_text(content: Any) -> str:
    """Coerce a LangChain message ``content`` field to a plain string.

    Qwen3 (and other reasoning-capable models) under SGLang's OpenAI-compatible
    API return ``content`` as a list of typed blocks
    (e.g. ``[{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]``).
    Pull out the user-facing text segments and drop ``thinking``/``reasoning``
    blocks; fall back to ``str()`` for anything we don't recognise so we never
    silently lose information.
    """
    if isinstance(content, str):
        return _unwrap_serialized_generation(content)
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype in ("thinking", "reasoning", "redacted_thinking"):
                continue
            for key in ("text", "content", "value"):
                if key in block and isinstance(block[key], str):
                    parts.append(block[key])
                    break
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def parse_fallback(role: Role, raw_text: Any) -> Optional[BaseModel]:
    """Fallback parsing using regex extraction if structured output fails."""
    text = _content_to_text(raw_text)

    keys = list(role.output_model.model_fields.keys())
    specs = [
        extraction_type_from_annotation(f.annotation)
        for f in role.output_model.model_fields.values()
    ]
    value_types = [s[0] for s in specs]
    field_optional = [s[1] for s in specs]

    try:
        parsed_dict = extract_info_from_text(
            text, keys, value_types, field_optional=field_optional
        )
    except Exception:
        # Log the raw text the parser choked on so the failure mode is
        # observable (e.g. truncated thinking, no JSON emitted at all).
        preview = (
            text
            if len(text) < 4000
            else f"{text[:2000]}\n…[truncated {len(text) - 4000} chars]…\n{text[-2000:]}"
        )
        logger.error(
            "[parse_fallback] role=%r could not extract fields %s from raw "
            "content (len=%d):\n----- BEGIN RAW -----\n%s\n----- END RAW -----",
            role.name,
            keys,
            len(text),
            preview,
        )
        raise

    return role.output_model(**parsed_dict)


# Attribute stamped on the neutral structures returned by
# :func:`build_safe_default_output`. Callers use :func:`is_safe_default` to tell a
# retry-exhausted parse failure apart from a genuine completion whose zero value
# happens to be meaningful — the two are otherwise indistinguishable, and
# conflating them turns an outage into a confident signal (a VERIFIER rating of
# 0.0 reads as "worst possible answer"; an empty ``subquestions`` list reads as
# "nothing left to ask"). Set via ``object.__setattr__`` so it lives outside the
# pydantic field set and never appears in ``model_dump()``.
_SAFE_DEFAULT_ATTR = "__coe_safe_default__"


def mark_safe_default(obj: BaseModel) -> BaseModel:
    """Stamp ``obj`` as a neutral default (see :data:`_SAFE_DEFAULT_ATTR`)."""
    object.__setattr__(obj, _SAFE_DEFAULT_ATTR, True)
    return obj


def is_safe_default(obj: Any) -> bool:
    """True when ``obj`` is a neutral default, not a parsed model completion."""
    return bool(getattr(obj, _SAFE_DEFAULT_ATTR, False))


def build_safe_default_output(role: Role) -> BaseModel:
    """Construct a neutral, valid instance of a role's output model.

    Used as a last resort when no completion parses after all shake retries: we
    return this empty/neutral structure (warning loudly) instead of raising, so a
    single un-parseable LLM response never breaks the surrounding graph loop. The
    caller treats it as a no-result for that node. Required scalar fields get a
    type-appropriate zero value; lists get ``[]``; optional/defaulted fields are
    left to pydantic. Falls back to ``model_construct`` if a constrained field
    (e.g. a regex-validated literal) rejects the zero value.

    The result is stamped by :func:`mark_safe_default` so callers can distinguish
    it from a real completion — ``is_safe_default(out)`` is the only reliable way
    to detect "the model never answered" downstream.
    """
    defaults: Dict[str, Any] = {}
    zero_by_type = {
        "str": "",
        "Literal": "",
        "int": 0,
        "float": 0.0,
        "bool": False,
        "list": [],
        "List": [],
    }
    for name, field in role.output_model.model_fields.items():
        vtype, _opt = extraction_type_from_annotation(field.annotation)
        if vtype in ("list", "List"):
            # Always [] (never None) — even for optional list fields — so callers
            # that iterate the result can't trip over a None and break the loop.
            defaults[name] = []
        elif field.is_required():
            defaults[name] = zero_by_type.get(vtype, None)
        # else: optional scalar — leave pydantic's configured default
    try:
        return mark_safe_default(role.output_model(**defaults))
    except Exception:  # noqa: BLE001 — constrained field; skip validation
        return mark_safe_default(role.output_model.model_construct(**defaults))


class RoleModelRegistry:
    """Maps role names → ChatLiteLLM instances via tier indirection.

    Lazily creates one ChatLiteLLM per unique tier config.
    """

    def __init__(self, llm_config: LLMConfig):
        self._tiers = llm_config.tiers
        self._role_tiers = llm_config.role_tiers
        self._api_key = llm_config.api_key
        self._instances: Dict[str, ChatLiteLLM] = {}  # tier_name → instance

    def _get_tier(self, role_name: str) -> str:
        return self._role_tiers.get(role_name, "heavy")

    def get_max_input_tokens(
        self, role_name: str, tier_override: Optional[str] = None
    ) -> int:
        """Prompt-token ceiling for a role's tier (enforced by the input guard)."""
        tier = tier_override or self._get_tier(role_name)
        cfg = self._tiers.get(tier) or self._tiers.get("heavy")
        return cfg.max_input_tokens if cfg else 0

    def get_chars_per_token(
        self, role_name: str, tier_override: Optional[str] = None
    ) -> float:
        """Chars-per-token estimate for a role's tier (input-guard conversion)."""
        tier = tier_override or self._get_tier(role_name)
        cfg = self._tiers.get(tier) or self._tiers.get("heavy")
        return float(getattr(cfg, "chars_per_token", _CHARS_PER_TOKEN) or _CHARS_PER_TOKEN)

    def get_model_name(
        self, role_name: str, tier_override: Optional[str] = None
    ) -> Optional[str]:
        """LiteLLM model id for a role's tier — the tokenizer needs it by name."""
        tier = tier_override or self._get_tier(role_name)
        cfg = self._tiers.get(tier) or self._tiers.get("heavy")
        return getattr(cfg, "model_name", None) if cfg else None

    def get_model_by_tier(self, tier: str) -> ChatLiteLLM:
        """Get or create the ChatLiteLLM for a specific tier."""
        if tier not in self._tiers:
            logger.warning(
                f"Tier '{tier}' not found in config, falling back to 'heavy'."
            )
            tier = "heavy"

        if tier not in self._instances:
            cfg = self._tiers[tier]

            # Qwen3/SGLang exposes thinking-mode toggle via chat_template_kwargs.
            # Forward it so configured tiers actually take effect.
            model_kwargs: Dict[str, Any] = {"top_p": cfg.top_p}

            # Optional sampling controls — only forward when set so an unset knob
            # keeps SGLang's own default. The non-OpenAI knobs (top_k, min_p,
            # repetition_penalty) ride through LiteLLM to SGLang's request body.
            for _name, _value in (
                ("top_k", cfg.top_k),
                ("min_p", cfg.min_p),
                ("presence_penalty", cfg.presence_penalty),
                ("frequency_penalty", cfg.frequency_penalty),
                ("repetition_penalty", cfg.repetition_penalty),
                ("seed", cfg.seed),
            ):
                if _value is not None:
                    model_kwargs[_name] = _value

            model_kwargs["chat_template_kwargs"] = {
                "enable_thinking": cfg.enable_thinking,
            }

            # Optional reasoning-token budget: SGLang has no native param, so we
            # ship a custom logit processor that forces </think> once the budget
            # is spent (server must run with --enable-custom-logit-processor).
            # Only meaningful while thinking is on. build_request_kwargs returns
            # None (and warns) for models whose think-token ids we don't know.
            if cfg.enable_thinking and cfg.thinking_budget is not None:
                budget_kwargs = build_request_kwargs(
                    cfg.model_name, cfg.thinking_budget
                )
                if budget_kwargs:
                    model_kwargs.update(budget_kwargs)

            self._instances[tier] = ChatLiteLLM(
                model=cfg.model_name,
                api_base=cfg.api_base,
                api_key=cfg.api_key or self._api_key,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                max_retries=cfg.max_retries,
                timeout=cfg.timeout,
                model_kwargs=model_kwargs,
            )
        return self._instances[tier]

    def get_model(self, role_name: str) -> ChatLiteLLM:
        """Get the ChatLiteLLM instance for a role based on its tier."""
        tier = self._get_tier(role_name)
        return self.get_model_by_tier(tier)

    def get_structured(self, role: Role) -> Runnable:
        """Get a Runnable that returns the role's structured output."""
        model = self.get_model(role.name)
        return model.with_structured_output(role.output_model)


async def execute_role_lc(
    registry: RoleModelRegistry,
    role: Role,
    input_data: Union[BaseModel, List[BaseModel]],
    n: int = 1,
    tier_override: Optional[str] = None,
) -> Tuple[Union[BaseModel, List[BaseModel], List[List[BaseModel]]], Dict]:
    """LangChain-native role execution.

    Args:
        registry: RoleModelRegistry for tier-based model selection
        role: Role with system_prompt, input_model, output_model
        input_data: Single or list of Pydantic input models
        n: Number of completions per input
        tier_override: Force a specific tier (for retry escalation)

    Returns:
        Tuple of (results, log_data).
    """
    is_single = isinstance(input_data, BaseModel)
    items = [input_data] if is_single else input_data

    # Get model — use override tier if escalating
    if tier_override:
        model = registry.get_model_by_tier(tier_override)
    else:
        model = registry.get_model(role.name)

    chain = model.with_structured_output(role.output_model, include_raw=True)
    max_input_tokens = registry.get_max_input_tokens(role.name, tier_override)
    # Tolerate registry stand-ins in tests that predate this accessor.
    chars_per_token = getattr(
        registry, "get_chars_per_token", lambda *_a, **_k: _CHARS_PER_TOKEN
    )(role.name, tier_override)
    guard_model = getattr(registry, "get_model_name", lambda *_a, **_k: None)(
        role.name, tier_override
    )

    def _shaken_chain(attempt: int) -> Runnable:
        """Rebuild the structured chain with perturbed sampling for a retry.

        ``attempt == 0`` uses the configured params. Later attempts vary the
        ``seed`` (the tiers pin a fixed seed, so resampling otherwise reproduces
        the same failure) and nudge ``temperature``/``top_p`` upward (capped),
        mirroring legacy's failure-escalation. ``model_copy`` keeps the cached
        client; the overridden knobs ride through to SGLang per call.
        """
        if attempt == 0:
            return chain
        base_kwargs = dict(getattr(model, "model_kwargs", None) or {})
        base_seed = base_kwargs.get("seed")
        base_kwargs["seed"] = (base_seed if isinstance(base_seed, int) else 0) + 1000 * attempt
        base_top_p = base_kwargs.get("top_p", 0.95) or 0.95
        base_kwargs["top_p"] = min(1.0, base_top_p + 0.05 * attempt)
        base_temp = getattr(model, "temperature", 1.0)
        new_temp = min(1.0, (base_temp if base_temp is not None else 1.0) + 0.1 * attempt)
        perturbed = model.model_copy(
            update={"temperature": new_temp, "model_kwargs": base_kwargs}
        )
        return perturbed.with_structured_output(role.output_model, include_raw=True)

    def _collect(results: List[Any]) -> Tuple[List[BaseModel], List[Exception]]:
        """Pull parsed models out of a gather() batch, catching per-completion
        parse failures so one bad completion can't abort an otherwise good batch."""
        parsed: List[BaseModel] = []
        errors: List[Exception] = []
        for r in results:
            if isinstance(r, Exception):
                errors.append(r)
                continue
            try:
                if isinstance(r, dict):
                    if isinstance(r.get("parsed"), role.output_model):
                        parsed.append(r["parsed"])
                        continue
                    raw = r.get("raw")
                    if raw is not None and hasattr(raw, "content"):
                        fallback = parse_fallback(role, raw.content)
                        if fallback:
                            parsed.append(fallback)
                elif isinstance(r, role.output_model):
                    parsed.append(r)
            except Exception as e:  # noqa: BLE001 — keep other completions alive
                errors.append(e)
        return parsed, errors

    async def _run_item(item: BaseModel) -> List[BaseModel]:
        """Execute one input item (n completions, gathered) → its parsed outputs.

        Retries the whole batch with shaken sampling params up to
        ``_STRUCT_PARSE_RETRIES`` times when nothing parses (legacy parity)."""
        messages = format_messages(role, item)  # system + user only
        messages = _truncate_messages_to_budget(
            messages, max_input_tokens, role.name, chars_per_token, guard_model
        )

        # Counted once per item, before the retry loop, so the prompt is tokenized once.
        item_prompt_tokens = count_prompt_tokens(messages, guard_model)

        last_errors: List[Exception] = []
        for attempt in range(_STRUCT_PARSE_RETRIES):
            attempt_chain = _shaken_chain(attempt)
            # Parallel execution for N completions of this one item.
            tasks = [attempt_chain.ainvoke(messages) for _ in range(n)]
            # Recorded before awaiting: a shake retry is real spend even when it fails,
            # and attributing it only on success would understate the cost of a role
            # whose output is hard to parse.
            _record_cost(role.name, n, item_prompt_tokens)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parsed, last_errors = _collect(results)
            if parsed:
                if attempt > 0:
                    logger.info(
                        "[%s] recovered structured output after %d shake retr%s",
                        role.name,
                        attempt,
                        "y" if attempt == 1 else "ies",
                    )
                return parsed
            if attempt < _STRUCT_PARSE_RETRIES - 1:
                logger.warning(
                    "[%s] no parseable structured output (attempt %d/%d); "
                    "shaking sampling params (seed/temperature/top_p) and retrying",
                    role.name,
                    attempt + 1,
                    _STRUCT_PARSE_RETRIES,
                )

        # All attempts failed. Do NOT raise — a single un-parseable response must
        # not break the surrounding graph loop. Warn loudly (with the underlying
        # error so outages stay observable) and return a neutral, valid default
        # structure for the caller to treat as a no-result.
        reason = repr(last_errors[0]) if last_errors else "no parseable completion"
        logger.warning(
            "[%s] no valid structured output after %d shaken attempts (%s); "
            "returning a neutral default so the loop continues",
            role.name,
            _STRUCT_PARSE_RETRIES,
            reason,
        )
        return [build_safe_default_output(role)]

    # Items in a list are independent, so run them concurrently (bounded). A
    # single item (or single-input call) takes the trivial path. Order of
    # ``items`` is preserved in the returned results.
    if len(items) <= 1:
        per_item: List[List[BaseModel]] = [await _run_item(items[0])] if items else []
    else:
        sem = asyncio.Semaphore(_ROLE_ITEM_CONCURRENCY)

        async def _bounded(it: BaseModel) -> List[BaseModel]:
            async with sem:
                return await _run_item(it)

        gathered = await asyncio.gather(
            *[_bounded(it) for it in items], return_exceptions=True
        )
        # Surface the first failure (matches the old sequential raise-on-error),
        # after all in-flight requests have settled (no orphaned tasks).
        for g in gathered:
            if isinstance(g, Exception):
                raise g
        per_item = list(gathered)

    all_results = []
    log_entries = []
    for item, parsed in zip(items, per_item):
        if n == 1:
            all_results.append(parsed[0])
        else:
            all_results.append(parsed)
        log_entries.append((str(item), str(parsed[0])))

    log_data = {role.name: log_entries}

    return (all_results[0] if is_single else all_results), log_data
