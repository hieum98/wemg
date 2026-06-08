# langgraph_coe tests

Tests are grouped by **what they depend on**, not by build phase:

| Directory        | Needs                                   | Speed   |
|------------------|-----------------------------------------|---------|
| `unit/`          | nothing (stubs + `fakeredis`)           | fast    |
| `integration/`   | one live dependency (LLM / SPARQL / Redis), rest stubbed | medium |
| `real_servers/`  | the full live stack (real model endpoints) | slow    |

Within each directory the filename names the subsystem under test
(`test_kg_search_*`, `test_cot_*`, `test_mcts_*`, …).

## Running

```bash
# default tier — no servers required
pytest langgraph_coe/tests/unit

# one subsystem against a live dependency
pytest langgraph_coe/tests/integration

# full end-to-end smoke
pytest langgraph_coe/tests/real_servers

# everything; integration/real_servers self-skip when endpoints are down
pytest langgraph_coe/tests
```

`integration/` and `real_servers/` resolve their endpoints from `LANGGRAPH_TEST_*`
environment variables (e.g. `LANGGRAPH_TEST_LLM_URL`, `LANGGRAPH_TEST_EMBED_URL`,
`LANGGRAPH_TEST_SPARQL_URL`), falling back to the defaults in
[`_servers.py`](./_servers.py). When an endpoint is unreachable the whole module
skips with a message showing the SSH tunnel to open.

## Shared infrastructure (at the package root)

- [`conftest.py`](./conftest.py) — fixtures (`config`, `fake_redis`,
  `mini_backend`, `init_wikidata_tools`, …) and custom-marker registration
  (`integration`, `requires_wikidata`, `slow_integration`).
- [`_fixtures.py`](./_fixtures.py) — stubs/spies for hermetic unit tests
  (fake chat models, fake ReAct agents, `ToolSpy`, config-override logging).
- [`_servers.py`](./_servers.py) — `endpoint_alive()` probe + default test
  endpoints shared by the live-server suites.
