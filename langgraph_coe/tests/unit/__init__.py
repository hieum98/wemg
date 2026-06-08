"""Unit tests: fast, hermetic, no network.

Every graph/role here runs against the stubs and spies in
``langgraph_coe.tests._fixtures`` (and ``fakeredis`` for cache tests), so the
whole package runs with no live LLM/embedder/SPARQL/Redis. This is the default
tier — run it with::

    pytest langgraph_coe/tests/unit
"""
