"""Shared middleware to make thinking-enabled models safe in ReAct loops.

Reasoning content blocks (``type: "thinking"`` / ``"reasoning"`` /
``"redacted_thinking"``) emitted by thinking-enabled models (e.g. Qwen3.5 on
SGLang) are valid in a model *response*, but must be dropped before the assistant
turn is replayed to the model on the next iteration of a tool-calling loop:
SGLang's OpenAI schema only accepts text/image_url/video_url/audio_url content
parts and rejects ``type: "thinking"`` parts, which breaks any multi-turn agent.

Any ``langchain.agents.create_agent`` ReAct agent backed by a thinking-enabled
tier must register :data:`strip_reasoning_middleware`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import BaseMessage

_REASONING_BLOCK_TYPES = {"thinking", "reasoning", "redacted_thinking"}


def _clean_content(content: Any) -> Any:
    """Drop reasoning blocks; collapse a pure-text remainder to a plain string."""
    if not isinstance(content, list):
        return content
    cleaned: List[Any] = []
    text_parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in _REASONING_BLOCK_TYPES:
                continue
            cleaned.append(block)
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        else:
            cleaned.append(block)
    if not cleaned:
        return ""
    if all(isinstance(b, dict) and b.get("type") == "text" for b in cleaned):
        return "".join(text_parts)
    return cleaned


def sanitize_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Return *messages* with reasoning content stripped (tool calls preserved)."""
    out: List[BaseMessage] = []
    for msg in messages:
        new_content = _clean_content(getattr(msg, "content", None))
        ak = getattr(msg, "additional_kwargs", None)
        drop_ak = isinstance(ak, dict) and "reasoning_content" in ak
        if new_content is not getattr(msg, "content", None) or drop_ak:
            update: Dict[str, Any] = {"content": new_content}
            if drop_ak:
                update["additional_kwargs"] = {
                    k: v for k, v in ak.items() if k != "reasoning_content"
                }
            try:
                msg = msg.model_copy(update=update)
            except Exception:  # pragma: no cover - defensive
                msg.content = new_content
        out.append(msg)
    return out


@wrap_model_call
async def strip_reasoning_middleware(request, handler):
    """Strip prior-turn reasoning blocks before each model call (SGLang-safe)."""
    return await handler(request.override(messages=sanitize_messages(request.messages)))
