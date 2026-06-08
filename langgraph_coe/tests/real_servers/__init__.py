"""Real-server tests: full end-to-end against the live stack.

Unlike ``integration`` (one subsystem, others stubbed), these drive a whole
strategy — CoT / MCTS / the system orchestrator — through real model endpoints
with minimal stubbing, the closest thing to a production smoke test. Slowest
tier; self-skips when endpoints are unreachable. Run with::

    pytest langgraph_coe/tests/real_servers
"""
