"""Integration tests: exercise one subsystem against a live dependency.

These hit real services — an LLM/embedder endpoint, a SPARQL backend, or a real
Redis — usually with the *other* surfaces stubbed so each test isolates one
component. They self-skip when their endpoint is unreachable (see
``langgraph_coe.tests._servers.endpoint_alive``), so the package is safe to
collect anywhere. Point them at your deployment via the ``LANGGRAPH_TEST_*``
environment variables, then::

    pytest langgraph_coe/tests/integration
"""
