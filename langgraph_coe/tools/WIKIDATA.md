# Wikidata Tool — User & Maintainer Guide

Async Wikidata + Wikipedia client for the LangGraph CoE agent. Lives in four
files under `langgraph_coe/tools/`:

| File | Role | Lines |
|---|---|---|
| [`wikidata_backend.py`](wikidata_backend.py) | `WikidataBackend` Protocol + `HTTPWikidataBackend` (httpx wire layer) | ~335 |
| [`wikidata_client.py`](wikidata_client.py) | `WikidataClient` — cache, rate-limit, single-flight, k-hop traversal | ~736 |
| [`wikidata.py`](wikidata.py) | LangChain `@tool` wrappers used by the agent (`link_entities`, `enrich_entities`, `fetch_and_prune_subgraph`) | ~379 |
| [`wikidata_properties.py`](wikidata_properties.py) | Bundled `PROPERTY_LABELS` + `DEFAULT_PROPERTIES` data | ~117 |

Tests live in [`tests/wikidata/`](../../tests/wikidata/) (91 contract tests, run in <2 min without network).

---

## Quick start

```python
from langgraph_coe.tools.wikidata_client import WikidataClient

client = WikidataClient()                       # uses HTTPWikidataBackend by default

# 1. Entity linking: text → top-k QIDs
entities = await client.link_entities("Berlin", top_k=3)
# → [WikidataEntity(qid="Q64", label="Berlin", ...), ...]

# 2. Enrichment: QID → full record (+ optional Wikipedia text)
enriched = await client.enrich_entities(["Q64", "Q183"], get_details=True)

# 3. Property search: text → top-k PIDs
props = await client.search_properties("capital", top_k=2)
# → [WikidataProperty(pid="P36", label="capital", ...), ...]

# 4. K-hop subgraph
triples = await client.get_k_hop_triples("Q64", k=1, bidirectional=True, enrich=True)
# → list[WikiTriple] with subject/relation/object as full WikidataEntity/Property objects
```

All four methods accept either a single string or a list. List input returns a
list-of-lists (per-input results); single input returns a flat list.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LangGraph agent                                                 │
│  ├── @tool link_entities          ────┐                          │
│  ├── @tool fetch_and_prune_subgraph ──┤                          │
│  └── @tool enrich_entities       ─────┤                          │
└────────────────────────────────────────┼──────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  WikidataClient   (wikidata_client.py)                           │
│                                                                  │
│  • LRU L1 caches per primitive                                   │
│  • Redis L2 cache (optional)                                     │
│  • Single-flight coalescing per cache key                        │
│  • Concurrency semaphore + per-endpoint rate limiters            │
│  • 429 / Retry-After retry                                       │
│  • K-hop traversal, frontier capping, dedup                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │ WikidataBackend Protocol (7 methods)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  HTTPWikidataBackend  (wikidata_backend.py)                      │
│                                                                  │
│  • httpx.AsyncClient (NO requests, NO SPARQLWrapper)             │
│  • query.wikidata.org/sparql        (fetch_outgoing/incoming)    │
│  • www.wikidata.org/w/api.php       (wbsearchentities / get…)    │
│  • en.wikipedia.org/w/api.php       (page extracts)              │
└──────────────────────────────────────────────────────────────────┘
```

The **client owns orchestration** (caches, batching, retry, traversal). The
**backend owns the wire** (HTTP + JSON parsing). They communicate over a
small `WikidataBackend` Protocol with seven async methods.

This split is the single most important design choice: keep it. It makes the
client testable in isolation (`FakeWikidataBackend` in
[`tests/wikidata/fake_backend.py`](../../tests/wikidata/fake_backend.py))
and lets the wire layer be swapped without touching cache/retry/traversal
logic.

---

## `WikidataClient` constructor

```python
WikidataClient(
    *,
    backend: Optional[WikidataBackend] = None,    # default: HTTPWikidataBackend()
    max_sparql_rps: float = 2.0,                  # SPARQL endpoint rate (Wikidata enforces ~2)
    max_wikipedia_rps: float = 10.0,              # MediaWiki REST rate (wikidata.org API + Wikipedia)
    lru_capacity: int = 5000,                     # per-cache LRU size (7 caches total)
    redis: Optional["redis.Redis"] = None,        # L2 cache (optional)
    redis_ttl_seconds: int = 86400,               # L2 TTL (24h default)
    concurrency_limit: int = 10,                  # max in-flight backend calls
)
```

**Rate limit routing:**
- `_sparql_limiter` (`max_sparql_rps`) — only `fetch_outgoing`, `fetch_incoming`
- `_wiki_limiter` (`max_wikipedia_rps`) — everything else: `search_entities_text`,
  `get_entity_details`, `search_properties_text`, `get_property_details`,
  `get_wikipedia_contents`

Treat `max_sparql_rps=2.0` as the upper bound for the public SPARQL endpoint;
do not raise it without a documented private endpoint. Wikidata will 429 you.

---

## The four functions

### `link_entities(names, *, top_k=1)`

```python
# Single name
[e1, e2] = await client.link_entities("Berlin", top_k=2)

# Batch (returns list-of-lists, one per input)
[[e_berlin], [e_paris]] = await client.link_entities(["Berlin", "Paris"], top_k=1)

# QID short-circuit (no API call, returns the QID wrapped)
[e] = await client.link_entities("Q64")
```

Internally: dedup names → `search_entities_text` per unique name (single-flight
coalesces concurrent identical queries) → one batched `get_entity_details` for
all resolved QIDs across the call.

### `enrich_entities(qids, *, get_details=False)`

```python
# Without wikipedia content
[e] = await client.enrich_entities(["Q64"])           # label/desc/aliases only

# With wikipedia article extracts
[e] = await client.enrich_entities(["Q64"], get_details=True)
```

Returns one `WikidataEntity` per input qid, **input order preserved**. Missing
QIDs come back as bare stubs (`WikidataEntity(qid=q)`) — no exception.

### `search_properties(query, *, top_k=1)`

```python
[p_capital] = await client.search_properties("capital", top_k=1)
[p_p36] = await client.search_properties("P36")     # PID short-circuit
```

Filters out **uninformative properties** (where `label == pid` or `label` is
missing).

Well-known PIDs (in [`PROPERTY_LABELS`](wikidata_properties.py)) are served
**O(1) from the bundled dict** — no backend call. This is a deliberate
optimization for the ~90 properties the agent uses most.

### `get_k_hop_triples(qids, *, k=1, bidirectional=True, enrich=True)`

```python
triples = await client.get_k_hop_triples("Q64", k=2, bidirectional=True, enrich=True)
# Returns list[WikiTriple] — 2-hop neighborhood of Q64, outgoing + incoming edges.

# Per-seed results
per_seed = await client.get_k_hop_triples(["Q64", "Q183"], k=1)
# Returns list[list[WikiTriple]] — per_seed[0] only contains triples reachable from Q64.
```

**Invariants:**
- Per-seed visited sets are disjoint; concurrent seeds don't pollute each other.
- Each hop's aggregate frontier is capped at `MAX_ENTITIES_PER_HOP = 500`.
- Cycles terminate (visited-QID dedup).
- `enrich=False` returns bare `WikidataEntity(qid=...)` / `WikidataProperty(pid=...)`
  stubs and makes **zero** enrichment backend calls.

---

## Agent integration (LangChain `@tool` wrappers)

`wikidata.py` exposes three `BaseTool` instances and a session-reset helper:

```python
from langgraph_coe.tools.wikidata import (
    init_wikidata,
    link_entities,
    enrich_entities,
    fetch_and_prune_subgraph,
    create_fetch_and_prune_tool,
    reset_wikidata_session,
)
from langgraph_coe.config import LangGraphCoeConfig

cfg = LangGraphCoeConfig.from_yaml()
init_wikidata(cfg.wikidata)             # call ONCE at startup; builds the singleton client

# Now the tools are callable
await link_entities.ainvoke({"entity_names": ["Berlin"]})

# At the start of every NEW question/run:
reset_wikidata_session()                # clears per-question visited-QID + hop budget
```

**Stage A / Stage B pruning** (`fetch_and_prune_subgraph`):
- **Stage A** (always on if `reranker_url` set in config): scores triples
  against the question via the reranker endpoint, keeps top-K within a
  score-delta band.
- **Stage B** (opt-in): LLM-based pruning via the `TRIPLE_PRUNER` role. Use
  `create_fetch_and_prune_tool(registry)` instead of the plain `@tool` to
  enable Stage B inside an agent.

**Hop budget**: each call to `fetch_and_prune_subgraph` increments a counter.
After `cfg.wikidata.max_hops` calls, the tool returns a budget-exhausted marker
instead of fetching. Reset between questions with `reset_wikidata_session()`.

---

## Caching: L1 (LRU) + L2 (Redis)

Seven LRU caches in the client, each sized at `lru_capacity` (default 5000):

| Cache | Key | Value |
|---|---|---|
| `_entities` | qid | `EntityRecord` dict |
| `_properties` | pid | `PropertyRecord` dict |
| `_outgoing` | qid | `list[(pid, obj_qid_or_literal)]` |
| `_incoming` | qid | `list[(pid, subj_qid)]` |
| `_wiki` | wikipedia_title | extract text |
| `_entity_search` | `(query, top_k)` | `list[qid]` |
| `_property_search` | `(query, top_k)` | `list[pid]` |

Redis is opt-in. If provided, the client follows **L1 → L2 → backend** on
reads and **writes to both** on cache populate.

```python
import redis
r = redis.Redis(host="localhost", port=6379)
client = WikidataClient(redis=r, redis_ttl_seconds=86400)
```

Redis failures are **silently ignored** — `_redis_get`/`_redis_set` catch all
exceptions. The client never fails because Redis is down; it falls back to L1.

Redis stores **JSON-encoded** values. Tuple-typed values (outgoing/incoming
edges) round-trip via `redis_normalize=lambda lst: [tuple(e) for e in lst]`
declared at the call site in `_fetch_outgoing` / `_fetch_incoming`.

**Cache key conventions** (collision-safe across functions):
- `wiki:ent:<qid>` — entity details
- `wiki:prop:<pid>` — property details
- `wiki:out:<qid>` — outgoing edges
- `wiki:in:<qid>` — incoming edges
- `wiki:wp:<title>` — wikipedia content

---

## Single-flight coalescing

`_coalesce(key, fetch_fn)` ensures **N concurrent identical fetches → 1
backend call**. Followers `await` the leader's future. Critical for agent
batches where many parallel questions all need the same entity.

The keys are deterministic strings:
- `search_e:<name_lower>:<top_k>` for entity searches
- `search_p:<query_lower>:<top_k>` for property searches
- `ent:<sorted,csv,qids>` for entity details batches
- `out:<sorted,csv,qids>` for outgoing fetches
- `in:<sorted,csv,qids>` for incoming fetches
- `prop:<sorted,csv,pids>` for property details batches
- `wp:<sorted,csv,titles>` for wikipedia batches

**Cancellation handling** (be careful before changing this logic):
- If the **follower** is cancelled, propagate (`task.cancelling() > 0` check).
- If the **leader** is cancelled, the future is cancelled and the next
  follower retries as the new leader. No follower is left hanging.

See [`tests/wikidata/test_async_correctness.py`](../../tests/wikidata/test_async_correctness.py)
for the canonical cancellation correctness tests.

---

## Retry + 429 / Retry-After

`_call_with_retry(limiter, fn, *args, **kwargs)`:
- Up to `MAX_RETRIES = 3` attempts.
- On `WikidataRateLimitError`, sleeps `e.retry_after` if set, else
  `2 ** attempt` seconds.
- **Releases the concurrency semaphore before sleeping** — otherwise the
  retry path deadlocks at `concurrency_limit=1`. Do NOT move the sleep inside
  the `async with self._semaphore` block.

---

## Maintenance

### Add a new backend (e.g., a cached/replay/mocked variant)

1. Implement the seven async methods in the `WikidataBackend` Protocol.
2. Pass an instance via `WikidataClient(backend=MyBackend(...))`.
3. No changes to the client.

Example shells in [`tests/wikidata/fake_backend.py`](../../tests/wikidata/fake_backend.py)
(`FakeWikidataBackend`) and
[`langgraph_coe/tools/wikidata_backend.py`](wikidata_backend.py)
(`HTTPWikidataBackend`).

### Add a well-known PID (avoid round-tripping for it)

Add to [`wikidata_properties.py`](wikidata_properties.py)::

```python
PROPERTY_LABELS["P1234"] = {
    "label": "new property",
    "description": "...",
}
DEFAULT_PROPERTIES.append("P1234")     # optional, only if the agent should prefer it
```

The client will serve it O(1) from the bundled dict; no backend call.

### Add a new function to `WikidataClient`

1. Add the primitive(s) to the `WikidataBackend` Protocol in
   [`wikidata_backend.py`](wikidata_backend.py).
2. Implement in `HTTPWikidataBackend` and in `FakeWikidataBackend`
   (`tests/wikidata/fake_backend.py`).
3. Add a method on `WikidataClient` that:
   - Normalizes single-vs-list input.
   - Uses `self._fetch_cached(...)` for batched/cached/coalesced fetches.
   - Wraps direct backend calls in `self._call_with_retry(self._wiki_limiter, ...)`.
   - Returns properly typed pydantic models built via `_make_entity` / `_make_property`.
4. Add contract tests under [`tests/wikidata/test_<your_feature>.py`](../../tests/wikidata/).

Don't add a new private cache field unless the data has a natural per-key
identity — most additions can share existing caches.

### Run the test suite

```bash
# Fast contract tests (no network) — should be <2 min
pytest tests/wikidata -v -m "not requires_wikidata"

# Live smoke tests (hits real Wikidata; run manually, not in CI)
pytest tests/wikidata/test_live_smoke.py -m requires_wikidata -v
```

Test layout:
- `test_entity_linking.py` / `test_enrichment.py` / `test_property_search.py` /
  `test_khop_triples.py` — per-function correctness + batching.
- `test_caching.py` — L1 LRU + L2 Redis (uses `fakeredis`).
- `test_rate_limit.py` — 429/Retry-After, RPS spacing (uses `VirtualClock`).
- `test_concurrency.py` — `asyncio.gather` semantics, in-flight cap.
- `test_async_correctness.py` — single-flight, cancellation, deadlock prevention,
  event-loop responsiveness.
- `test_tools_integration.py` — `@tool` wrappers, ContextVar isolation,
  Stage A/B pruning.
- `test_live_smoke.py` — 4 live-API sanity tests (marker-gated).

---

## Gotchas

### LangChain `BaseTool.ainvoke` isolates `ContextVar` mutations

`BaseTool.ainvoke` runs the tool coro inside `asyncio.create_task(coro,
context=ctx)` (see `coro_with_context` in `langchain_core/runnables/utils.py`).
Each `.ainvoke(...)` runs in a child Task with a COPY of the parent's context.
**A `ContextVar.set(...)` inside the tool does NOT propagate back** to the
caller's task.

**The fix already in `wikidata.py`:** bind a single mutable `_SessionState`
object to a ContextVar in the parent task. Child tasks inherit the same
object reference; mutations to its fields (`.visited`, `.hop_count`)
propagate via shared identity. Concurrent agent runs each call
`reset_wikidata_session()` from their own task, which rebinds locally for
isolation.

If you ever find yourself doing `ContextVar.set()` from inside a `@tool`
function and expecting the parent task to see it: **you've been bitten by
this**. Mutate the existing object instead.

### Rate-limit routing is not symmetric

Only `fetch_outgoing` and `fetch_incoming` go through `_sparql_limiter`.
Everything else — including `get_entity_details` (a wbgetentities REST call)
— goes through `_wiki_limiter`. If you misroute a new method through the
SPARQL limiter, you'll spuriously pay the 2 RPS penalty on Wikidata REST
calls (see the test `test_sparql_and_wikipedia_rates_independent`).

### Semaphore must be released before retry sleep

In `_call_with_retry`, the `async with self._semaphore` block is exited
**before** `await asyncio.sleep(retry_after)`. If you move the sleep inside
the semaphore, retries will deadlock at `concurrency_limit=1` because the
retrying task holds the only slot while sleeping.

### Backend exceptions must subclass `WikidataRateLimitError` for retry

The retry loop catches `WikidataRateLimitError` specifically. If you add a
backend that signals rate-limiting via a different exception type, either
re-raise as `WikidataRateLimitError(retry_after=...)` or update
`_call_with_retry`. Don't catch `Exception` — non-rate-limit errors should
propagate immediately.

### Tuple ↔ list serialization for outgoing/incoming

Edges are stored as `list[tuple[str, str]]` in L1 LRU. JSON serialization
flattens tuples to lists, so when reading from Redis, the client re-normalizes
via `redis_normalize=lambda lst: [tuple(e) for e in lst]`. If you add a new
cache whose values contain tuples, pass `redis_normalize` to `_fetch_cached`.

### `is_valid_qid` / `is_valid_pid` accept underscores

`_QID_RE = r"^Q[A-Za-z0-9_]+$"` accepts test mock IDs like `Q_DIVERSE_0` in
addition to real `Q\d+` Wikidata IDs. The production data only ever uses
`Q\d+`, but the relaxed regex makes tests cleaner. Don't tighten this without
updating the test fixtures.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Hop budget never exhausts in a multi-tool agent run | Test/code calls `_cv_hop_count.set(...)` inside `ainvoke` | Use `_get_session().hop_count` mutation; see ContextVar gotcha above |
| All requests serialized despite high `concurrency_limit` | Rate limit interval too high (low RPS); limiter `async with self._lock` serializes acquire | Raise the RPS knob; check which limiter the call routes through |
| `WikidataRateLimitError` on the first call | Likely a real 429 from Wikidata (you exceeded RPS) | Lower `max_sparql_rps` / `max_wikipedia_rps`; check headers in `HTTPWikidataBackend._raise_on_rate_limit` |
| Concurrent identical calls produce N backend hits | `_coalesce` keys don't match (different `top_k`, casing, ordering) | Inspect the coalesce key construction in the relevant `_link_one` / `_fetch_*` site |
| Redis cache reads return stale `(rel, obj)` shaped data | Forgot to pass `redis_normalize` for tuple-shaped values | Add `redis_normalize=lambda lst: [tuple(e) for e in lst]` to the `_fetch_cached` call |
| `test_per_task_session_isolation` fails | Sync fixture set the ContextVar before the test's task started | Make the fixture `async def` |
| Stage A reranker warnings spam in tests | Reranker URL set in config but unreachable | Either set `reranker_url=None` in test config or `respx`-mock the endpoint |

---

## Performance

Rough characteristics on the FakeWikidataBackend (no network):
- 50 concurrent `link_entities("Berlin")` → 1 search call + 1 detail call.
- 1000 concurrent `get_k_hop_triples("Q64")` → 1 backend call (single-flight).
- 500 distinct concurrent seeds → 500 fetches, bounded by `concurrency_limit`.
- k=2 over a 600-node fan-out → frontier capped at 500 at hop 2.

Production latency is dominated by the wire layer; the client adds ~ms-level
overhead per call. The semaphore + rate limiter bound throughput at
roughly `min(concurrency_limit × completion_rate, RPS_limit × concurrent_acquisition)`.

---

## Background reading

- The original messy implementation lives in [`wemg/retrieval/wikidata.py`](../../wemg/retrieval/wikidata.py)
  (1700+ lines, sync+threading, kept only for legacy callers in `wemg/reasoning/`).
  Do not extend it — extend the new client instead.
- Contract test design rationale: see the approved plan at
  `~/.claude/plans/i-neet-to-write-velvety-thunder.md` (target-spec test suite,
  punchlist-driven rewrite).
- The LangChain ainvoke ContextVar isolation behavior is documented in a memory:
  `~/.claude/projects/-home-hieum-projects-wemg/memory/project_langchain_ainvoke_context_isolation.md`.
