"""Text utilities for token counting and context formatting."""

from typing import Any, Dict, List, Union

import tiktoken


def approximate_token_count(messages: Union[List[Dict[str, str]], str]) -> int:
    """Approximate token count using tiktoken cl100k_base encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    else:
        return sum(len(encoding.encode(m["content"])) for m in messages)


def truncate_text_to_max_tokens(
    text: str, max_tokens: int, encoding_name: str = "cl100k_base"
) -> str:
    """Return text truncated to at most max_tokens (cl100k_base by default)."""
    if max_tokens <= 0 or not text:
        return text if max_tokens > 0 else ""
    encoding = tiktoken.get_encoding(encoding_name)
    ids = encoding.encode(text)
    if len(ids) <= max_tokens:
        return text
    return encoding.decode(ids[:max_tokens])


def format_context(
    memory: str = None,
    graph_memory: str = None,
    retrieval_info: List[str] = None,
    reasoning_trace: str = None,
) -> str:
    """Format context string from memory, graph memory, retrieval info, and reasoning trace.

    Combines:
    - memory text directly
    - retrieval_info as "- [Retrieval]: {content}" lines
    - graph_memory as a structured knowledge section
    - reasoning_trace as a separate section
    Joins non-empty sections with "\\n---------------\\n"
    """
    if retrieval_info is None:
        retrieval_info = []
    retrieval_lines = [f"- [Retrieval]: {content.strip()}" for content in retrieval_info]
    retrieval_text = "\n".join(retrieval_lines)

    fact_parts = [memory, retrieval_text]
    fact_context = "\n".join(p for p in fact_parts if p)
    fact_context = "**Information** \n" + fact_context if fact_context else ""

    graph_context = f"**Structured Knowledge (Graph)**\n{graph_memory}" if graph_memory else ""
    reasoning_context = f"**Reasoning Trace** \n{reasoning_trace}" if reasoning_trace else ""

    sections = [fact_context, graph_context, reasoning_context]
    return "\n---------------\n".join(s for s in sections if s)
