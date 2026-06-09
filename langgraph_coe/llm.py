"""LangGraph-CoE: LLM Execution Layer with tier-based model selection."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

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


def format_messages(role: Role, item: BaseModel) -> List[Any]:
    """Format prompt messages for a given role and input item."""
    system_prompt = role.system_prompt
    user_prompt = str(item)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


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
        return content
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

    async def _run_item(item: BaseModel) -> List[BaseModel]:
        """Execute one input item (n completions, gathered) → its parsed outputs."""
        messages = format_messages(role, item)  # system + user only

        # Parallel execution for N completions of this one item.
        tasks = [chain.ainvoke(messages) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parsed: List[BaseModel] = []
        for r in results:
            if isinstance(r, dict):
                # Langchain return format with include_raw=True
                if "parsed" in r and isinstance(r["parsed"], role.output_model):
                    parsed.append(r["parsed"])
                elif "raw" in r and hasattr(r["raw"], "content"):
                    fallback = parse_fallback(role, r["raw"].content)
                    if fallback:
                        parsed.append(fallback)
            elif isinstance(r, role.output_model):
                # Just in case some provider returns the object directly
                parsed.append(r)

        if not parsed:
            # All N completions failed — raise so RetryPolicy catches it.
            errors = [r for r in results if isinstance(r, Exception)]
            raise (
                errors[0]
                if errors
                else RuntimeError(f"No valid output for {role.name}")
            )
        return parsed

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
