"""Agents module for WEMG.

This package provides:
- LLM agents for generation and embedding
- Retriever agents for corpus-based retrieval
- Role definitions for various LLM tasks
- Tool integrations (web search, Wikidata)
"""
from wemg.agents.base_llm_agent import BaseLLMAgent, BaseClient, OpenAIClient
from wemg.agents.retriever_agent import RetrieverAgent

__all__ = [
    'BaseLLMAgent',
    'BaseClient', 
    'OpenAIClient',
    'RetrieverAgent',
]

