"""Procedure modules for reasoning operations.

This package contains reusable procedures for:
- Role execution
- Retrieval operations
- Node generation
- OpenIE parsing
"""
from wemg.runners.procedures.base_role_execution import execute_role
from wemg.runners.procedures.retrieval import (
    explore,
    retrieve_from_web,
    retrieve_from_kb,
    retrieve_entities_from_kb,
    retrieve_triples,
)
from wemg.runners.procedures.node_generator import (
    NodeGenerator,
    GenerationResult,
)
from wemg.runners.procedures.openie import parse_graph_from_text

__all__ = [
    'execute_role',
    'explore',
    'retrieve_from_web',
    'retrieve_from_kb',
    'retrieve_entities_from_kb',
    'retrieve_triples',
    'NodeGenerator',
    'GenerationResult',
    'parse_graph_from_text',
]
