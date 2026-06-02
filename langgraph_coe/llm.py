"""LangGraph-CoE: LLM Execution Layer with tier-based model selection."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from langchain_litellm import ChatLiteLLM

from .config import LLMConfig
from .roles import Role
from .parsing import extract_info_from_text, extraction_type_from_annotation

logger = logging.getLogger(__name__)


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
        # Print the raw text the parser choked on so the failure mode is
        # observable (e.g. truncated thinking, no JSON emitted at all).
        preview = text if len(text) < 4000 else f"{text[:2000]}\n…[truncated {len(text)-4000} chars]…\n{text[-2000:]}"
        print(
            f"[parse_fallback] role={role.name!r} could not extract fields {keys} "
            f"from raw content (len={len(text)}):\n----- BEGIN RAW -----\n{preview}\n----- END RAW -----",
            flush=True,
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
            logger.warning(f"Tier '{tier}' not found in config, falling back to 'heavy'.")
            tier = "heavy"
            
        if tier not in self._instances:
            cfg = self._tiers[tier]

            # Qwen3/SGLang exposes thinking-mode toggle via chat_template_kwargs.
            # Forward it so configured tiers actually take effect.
            model_kwargs: Dict[str, Any] = {"top_p": cfg.top_p}
            model_kwargs["chat_template_kwargs"] = {
                "enable_thinking": cfg.enable_thinking,
            }

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

    all_results = []
    log_entries = []
    
    for item in items:
        messages = format_messages(role, item)  # system + user only
        
        # Parallel execution for N completions
        tasks = [chain.ainvoke(messages) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        parsed = []
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
            # All N completions failed — raise so RetryPolicy catches it
            errors = [r for r in results if isinstance(r, Exception)]
            raise errors[0] if errors else RuntimeError(f"No valid output for {role.name}")
            
        if n == 1:
            all_results.append(parsed[0])
            log_entries.append((str(item), str(parsed[0])))
        else:
            all_results.append(parsed)
            log_entries.append((str(item), str(parsed[0])))

    log_data = {role.name: log_entries}
    
    return (all_results[0] if is_single else all_results), log_data
