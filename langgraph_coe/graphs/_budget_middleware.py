"""Keep a ReAct agent's growing message list inside the model's context window.

``execute_role_lc`` guards single-shot role calls (``_truncate_messages_to_budget``),
but a tool-calling agent built with ``langchain.agents.create_agent`` appends a
message per tool result and calls the model again — so nothing bounds the prompt
across iterations. On a large-window deployment that is invisible; on a tighter
one the provider rejects the request outright, e.g.::

    BedrockException - This model's maximum context length is 32768 tokens.
    However, you requested 4096 output tokens and your prompt contains at least
    28673 input tokens.

which surfaces as a hard failure of the whole KG-search or web-research step
rather than as degraded evidence.

This middleware drops whole messages from the middle of the conversation, oldest
first, keeping the system prompt, the original request, and the most recent turns
— the ones the model actually needs to decide its next tool call. Whole messages
rather than truncated text because a tool_call/tool_result pair must stay
consistent: a tool result whose content is chopped mid-JSON is worse than one
that is absent.

Register alongside :data:`strip_reasoning_middleware`::

    middleware=[strip_reasoning_middleware, make_budget_middleware(registry, role)]
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import BaseMessage

from ..llm import _CHARS_PER_TOKEN, _TRUNCATION_NOTICE, count_prompt_tokens

logger = logging.getLogger(__name__)

# How many of the newest messages are never dropped. The agent's next decision
# depends on the most recent tool results, so trimming those defeats the purpose.
_KEEP_TAIL = 4


def _content_len(message: BaseMessage) -> int:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += len(block)
            elif isinstance(block, dict):
                for key in ("text", "content", "value"):
                    value = block.get(key)
                    if isinstance(value, str):
                        total += len(value)
                        break
        return total
    return len(str(content))


def _is_droppable(message: BaseMessage) -> bool:
    """Only tool results and assistant turns may be dropped.

    Keeping every human/system message means the agent never loses the request it
    is working on — the failure mode where an agent forgets its own task is worse
    than one where it forgets an intermediate observation.
    """
    return getattr(message, "type", None) in {"tool", "ai"}


def trim_messages_to_char_budget(
    messages: Sequence[BaseMessage], budget_chars: int
) -> List[BaseMessage]:
    """Drop droppable middle messages, oldest first, until under ``budget_chars``.

    Returns the input unchanged when it already fits, or when nothing droppable
    remains — an over-budget prompt of system + human messages is a caller
    problem, and silently mangling it would hide that.
    """
    if budget_chars <= 0 or not messages:
        return list(messages)

    kept: List[BaseMessage] = list(messages)
    total = sum(_content_len(m) for m in kept)
    if total <= budget_chars:
        return kept

    # Indices eligible for dropping: droppable, and outside the protected tail.
    protected_from = max(0, len(kept) - _KEEP_TAIL)
    dropped = 0
    for idx in range(protected_from):
        if total <= budget_chars:
            break
        message = kept[idx]
        if not _is_droppable(message):
            continue
        total -= _content_len(message)
        kept[idx] = None  # type: ignore[call-overload]
        dropped += 1

    result = [m for m in kept if m is not None]
    if dropped:
        logger.warning(
            "[budget_guard] agent prompt over budget (%d chars > %d); dropped %d "
            "intermediate message(s), %d remain",
            sum(_content_len(m) for m in messages),
            budget_chars,
            dropped,
            len(result),
        )
    if total <= budget_chars:
        return result

    # Last resort: the remaining prompt is dominated by a single message we are
    # not allowed to drop — typically the agent's request, which embeds the whole
    # accumulated memory or triple set. Shrink the largest such message in place
    # rather than let the provider reject the call, keeping its head and tail so
    # the instruction at the top and any trailing constraint both survive.
    # Iterate: one oversized message is the common case, but several can survive
    # the drop pass (e.g. a large request plus a protected recent tool result), and
    # shrinking only the largest would leave the prompt over budget.
    remaining = total
    for _ in range(len(result)):
        shrunk = _truncate_largest(result, budget_chars, remaining)
        if shrunk is result:
            break  # no further progress possible
        result = shrunk
        remaining = sum(_content_len(m) for m in result)
        if remaining <= budget_chars:
            break
    if remaining > budget_chars:
        logger.warning(
            "[budget_guard] still %d chars over budget after dropping and "
            "truncating — expect a context-length error",
            remaining - budget_chars,
        )
    return result


def _truncate_largest(
    messages: List[BaseMessage], budget_chars: int, total: int
) -> List[BaseMessage]:
    """Head/tail-truncate the largest non-system message to fit ``budget_chars``.

    System messages are exempt: they carry the role instructions, and a truncated
    instruction produces confidently wrong tool calls rather than a visible
    failure. Only string content is truncated — structured (multimodal) content is
    left alone rather than risk producing an invalid block list.

    Returns the input object unchanged (identity-comparable) when no progress is
    possible, which is how the caller's loop terminates.
    """
    biggest_idx, biggest_len = -1, 0
    for idx, message in enumerate(messages):
        if getattr(message, "type", None) == "system":
            continue
        if not isinstance(getattr(message, "content", None), str):
            continue
        # Already shrunk on a previous pass — truncating it again would spend the
        # allowance on the same message and never terminate.
        if _TRUNCATION_NOTICE in message.content:
            continue
        size = _content_len(message)
        if size > biggest_len:
            biggest_idx, biggest_len = idx, size
    if biggest_idx < 0:
        return messages

    # Everything except the message we are about to shrink. When several oversized
    # messages remain, ``others`` still exceeds the budget on this pass, so cap the
    # allowance at an even share instead of giving up — the caller loops and each
    # pass shrinks one more.
    others = total - biggest_len
    allowance = budget_chars - others - len(_TRUNCATION_NOTICE)
    if allowance <= 0:
        n_shrinkable = sum(
            1
            for m in messages
            if getattr(m, "type", None) != "system"
            and isinstance(getattr(m, "content", None), str)
            and _TRUNCATION_NOTICE not in m.content
        )
        allowance = (
            budget_chars // max(1, n_shrinkable + 1)
        ) - len(_TRUNCATION_NOTICE)
    if allowance <= 0 or allowance >= biggest_len:
        return messages

    text = messages[biggest_idx].content
    head = allowance // 2
    tail = allowance - head
    shrunk = text[:head] + _TRUNCATION_NOTICE + text[-tail:] if tail else text[:head]
    logger.warning(
        "[budget_guard] truncated the largest message %d→%d chars to fit the "
        "%d-char budget",
        biggest_len,
        len(shrunk),
        budget_chars,
    )
    out = list(messages)
    try:
        out[biggest_idx] = messages[biggest_idx].model_copy(
            update={"content": shrunk}
        )
    except Exception:  # pragma: no cover - defensive, mirrors sanitize_messages
        messages[biggest_idx].content = shrunk
    return out


def make_budget_middleware(registry: Any, role_name: str):
    """Middleware capping an agent's prompt at ``role_name``'s tier budget.

    Reads ``max_input_tokens`` and ``chars_per_token`` from the registry so the
    bound follows the same per-tier config the single-shot guard uses.
    """
    max_input_tokens = 0
    chars_per_token = _CHARS_PER_TOKEN
    model_name = None
    try:
        max_input_tokens = int(registry.get_max_input_tokens(role_name) or 0)
        getter = getattr(registry, "get_chars_per_token", None)
        if getter is not None:
            chars_per_token = float(getter(role_name) or _CHARS_PER_TOKEN)
        name_getter = getattr(registry, "get_model_name", None)
        if name_getter is not None:
            model_name = name_getter(role_name)
    except Exception:  # noqa: BLE001 — test stand-ins may not implement these
        logger.debug("[budget_guard] registry has no budget info for %r", role_name)

    @wrap_model_call
    async def budget_middleware(request, handler):
        if max_input_tokens <= 0:
            return await handler(request)

        # Prefer an exact token count for this payload: a chars-per-token constant
        # cannot be made safe, since crawled pages carrying CJK or minified script
        # tokenize near 1.1 chars/token while English runs ~4, so any single ratio
        # under-trims one of them into a hard context-length error.
        actual = count_prompt_tokens(request.messages, model_name)
        if actual is not None and actual > 0:
            if actual <= max_input_tokens:
                return await handler(request)
            total_chars = sum(_content_len(m) for m in request.messages)
            measured = max(0.5, total_chars / actual)
            budget_chars = int(max_input_tokens * measured)
        else:
            budget_chars = int(max_input_tokens * chars_per_token)

        trimmed = trim_messages_to_char_budget(request.messages, budget_chars)
        if trimmed is request.messages or list(trimmed) == list(request.messages):
            return await handler(request)
        return await handler(request.override(messages=trimmed))

    return budget_middleware


__all__ = [
    "make_budget_middleware",
    "trim_messages_to_char_budget",
]
