# Wikidata Tools (`langgraph_coe`)

User and maintainer guide for the async Wikidata + Wikipedia stack used by the
LangGraph CoE agent (KG search, CoT/MCTS retrieval, and `@tool` wrappers).

> **TL;DR**
> 1. Configure `wikidata` in `langgraph_coe/config.yaml` (optional local SPARQL tunnel).
> 2. Call `init_wikidata(cfg.wikidata, cache=...)` once at startup — or use `system.answer()` which does this when `cache.enabled: true`.
> 3. Call `reset_wikidata_session()` at the start of each new question.
> 4. Agent tools: `link_entities`, `fetch_and_prune_subgraph`, `enrich_entities`.
> 5. Persistent Redis (`wd:*` keys): see [setup_redis_cache.md](setup_redis_cache.md).

---

## Source layout

| File | Role | ~Lines |
|------|------|--------|
| `langgraph_coe/tools/wikidata_backend.py` | `WikidataBackend` protocol + `HTTPWikidataBackend` (httpx) | 343 |
| `langgraph_coe/tools/wikidata_client.py` | `WikidataClient` — LRU, Redis L2, rate limits, k-hop | 827 |
| `langgraph_coe/tools/wikidata.py` | LangChain `@tool` wrappers + session/hop budget | 392 |
| `langgraph_coe/tools/wikidata_properties.py` | Bundled `PROPERTY_LABELS` / `DEFAULT_PROPERTIES` | 117 |

Contract tests: `tests/wikidata/` (fast, no network by default).

LangGraph integration smoke: `langgraph_coe/tests/phase0/test_wikidata_local_integration.py` (local QEndpoint SPARQL).

---

## Configuration

`langgraph_coe/config.yaml` → `wikidata`:

```yaml
wikidata:
  # null → public https://query.wikidata.org/sparql
  # Local QEndpoint (SSH tunnel): see deploy_local_wikidata-v2.md
  sparql_endpoint: null
  max_sparql_rps: 2.0
  max_wikipedia_rps: 10.0
  triple_cache_max_entries: 5000
  max_hops: 3                    # max fetch_and_prune_subgraph calls per question
  reranker_url: null             # Stage A pruning (/v1/rerank); null = skip Stage A
  reranker_model: null
  pruning_top_k: 64
  pruning_delta: 0.05
```

| Field | Effect |
|-------|--------|
| `sparql_endpoint` | Passed to `HTTPWikidataBackend` for `fetch_outgoing` / `fetch_incoming` only |
| `max_hops` | Tool-level hop budget (not k-hop depth inside one fetch) |
| `reranker_url` | Stage A scores triple strings before optional Stage B LLM prune |

Entity search, property metadata, and Wikipedia still hit **public** Wikidata/Wikipedia APIs even when SPARQL is local.

---

## Quick start (library)

```python
from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.tools.wikidata_client import WikidataClient

cfg = LangGraphCoeConfig.from_yaml()
client = WikidataClient(
    sparql_endpoint=cfg.wikidata.sparql_endpoint,
    max_sparql_rps=cfg.wikidata.max_sparql_rps,
    max_wikipedia_rps=cfg.wikidata.max_wikipedia_rps,
    lru_capacity=cfg.wikidata.triple_cache_max_entries,
)

entities = await client.link_entities("Berlin", top_k=3)
enriched = await client.enrich_entities(["Q64"], get_details=True)
props = await client.search_properties("capital", top_k=2)
triples = await client.get_k_hop_triples("Q64", k=1, bidirectional=True, enrich=True)
```

List inputs return **list-of-lists** (one inner list per input); single strings return a flat list.

---

## Quick start (agent tools)

```python
from langgraph_coe.config import LangGraphCoeConfig
from langgraph_coe.tools.wikidata import (
    init_wikidata,
    link_entities,
    enrich_entities,
    fetch_and_prune_subgraph,
    create_fetch_and_prune_tool,
    reset_wikidata_session,
)

cfg = LangGraphCoeConfig.from_yaml()
init_wikidata(cfg.wikidata)   # pass cache=... when Redis enabled (see setup_redis_cache.md)

reset_wikidata_session()      # once per new question
await link_entities.ainvoke({"entity_names": ["Berlin"]})
```

`system.answer()` calls `_init_runtime()` → `init_wikidata(cfg.wikidata, cache=shared_cache)` when `cache.enabled: true`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LangGraph agent (kg_search / CoT / MCTS)                        │
│  ├── @tool link_entities                                         │
│  ├── @tool fetch_and_prune_subgraph                              │
│  └── @tool enrich_entities                                       │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  wikidata.py — hop budget, visited QIDs (ContextVar session)     │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  WikidataClient                                                  │
│  • L1 LRU (7 tables)                                             │
│  • L2 RedisDictCache via cache=  → wd:* keys (recommended)       │
│  • L2 raw redis via redis=       → wiki:* keys (_fetch_cached)   │
│  • Single-flight, semaphore, SPARQL vs wiki rate limiters        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ WikidataBackend (7 async methods)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  HTTPWikidataBackend (httpx)                                     │
│  • SPARQL endpoint (outgoing/incoming edges)                     │
│  • wikidata.org w/api.php (search, wbgetentities, …)             │
│  • en.wikipedia.org w/api.php (extracts)                         │
└──────────────────────────────────────────────────────────────────┘
```

Keep the **client / backend split**: test the client with `FakeWikidataBackend` (`tests/wikidata/fake_backend.py`); swap wire behavior without touching caches or traversal.

---

## `WikidataClient` constructor

```python
WikidataClient(
    *,
    backend: Optional[WikidataBackend] = None,
    sparql_endpoint: Optional[str] = None,   # builds HTTPWikidataBackend when set
    max_sparql_rps: float = 2.0,
    max_wikipedia_rps: float = 10.0,
    lru_capacity: int = 5000,
    cache: Optional[Any] = None,             # RedisDictCache — wd:* keys (used by system.py)
    redis: Optional[Any] = None,             # raw redis.Redis — wiki:* keys in _fetch_cached
    redis_ttl_seconds: int = 86400,
    concurrency_limit: int = 10,
)
```

**Rate limit routing**

- `_sparql_limiter` — `fetch_outgoing`, `fetch_incoming` only (~2 RPS for public endpoint).
- `_wiki_limiter` — entity/property search, `get_entity_details`, Wikipedia fetches.

Do not raise `max_sparql_rps` on the public Wikidata SPARQL URL without a private endpoint.

**User-Agent (public API):** `HTTPWikidataBackend` sends a Wikimedia-compliant UA with contact info (not `bot/<random>` — that returns 403). Override via env:

```bash
export WIKIDATA_CONTACT="contact/you@example.edu"   # default: contact/hieum@uoregon.edu
# or full string:
export WIKIDATA_USER_AGENT="COE/0.2.0 (langgraph_coe; contact/you@example.edu) python-httpx"
```

---

## Public API

### Core methods (agent + library)

| Method | Purpose |
|--------|---------|
| `link_entities(names, top_k=1)` | Text → QIDs; batched `get_entity_details` |
| `enrich_entities(qids, get_details=False)` | QID → `WikidataEntity`; optional Wikipedia body |
| `search_properties(query, top_k=1)` | Text/PID → properties; bundled PIDs are O(1) |
| `get_k_hop_triples(qids, k=1, bidirectional=True, enrich=True)` | Multi-hop neighborhood; frontier cap 500/hop |

**`get_k_hop_triples` invariants:** disjoint per-seed visited sets; cycles stop via QID dedup; `enrich=False` skips enrichment backend calls.

### Cache-aware helpers (Redis `wd:*` when `cache=` set)

Used by tests and future call sites; wired through `_cache_get` / `_cache_set`:

| Method | Redis key |
|--------|-----------|
| `get_entity(qid)` | `wd:entity:{qid}` |
| `search_entities(name, top_k=…)` | `wd:search:{name.lower()}:{top_k}` |
| `get_triples(qid)` | `wd:triples:{qid}` |
| `get_wikipedia_content(qid)` | `wd:enrich:{qid}` |

**Not cached:** pruning output from `fetch_and_prune_subgraph` (query-specific).

`link_entities` / `get_k_hop_triples` still benefit from **L1 LRU** and optional legacy `redis=` `wiki:*` keys inside `_fetch_cached`; production `system.py` only passes `cache=`, not `redis=`.

---

## Agent tools (`wikidata.py`)

### `link_entities`

Resolves names to QIDs; maintains a **run-wide** `entity_cache` dict (name → QID) across questions in addition to per-question session state.

### `fetch_and_prune_subgraph`

- Fetches **1-hop** triples per call (`get_k_hop_triples(..., k=1, bidirectional=False)`).
- **Stage A** (if `wikidata.reranker_url` set): HTTP POST to `{reranker_url}/rerank`, keep top tier within `pruning_delta` of best score, cap `pruning_top_k`.
- **Stage B** (optional): `create_fetch_and_prune_tool(registry)` enables LLM `TRIPLE_PRUNER` batches.
- **Hop budget:** `session.hop_count` vs `max_hops`; over budget returns a marker string instead of fetching.

### `enrich_entities`

Wraps `enrich_entities` on the client for tool I/O.

### Session reset

```python
reset_wikidata_session()  # new _SessionState: visited=∅, hop_count=0
```

Must run at the start of each question. See **ContextVar gotcha** below.

---

## Caching

### L1 — in-process LRU (always on)

| LRU field | Key | Value |
|-----------|-----|--------|
| `_entities` | qid | entity record dict |
| `_properties` | pid | property record dict |
| `_outgoing` / `_incoming` | qid | edge lists |
| `_wiki` | wikipedia title | extract text |
| `_entity_search` | `(text, top_k)` | list of QIDs |
| `_property_search` | `(text, top_k)` | list of PIDs |

### L2 — `cache=` (`RedisDictCache`, recommended)

Enabled via `system.py` when `cache.enabled: true`. Keys on Redis db `wikidata_db` (default **1**). TTLs from `cache.wikidata.*` in config.

Full setup: **[setup_redis_cache.md](setup_redis_cache.md)**.

### L2 — legacy `redis=` (`wiki:*` prefixes)

If you pass `redis=redis.Redis(...)` directly to `WikidataClient`, `_fetch_cached` also read/writes:

- `wiki:ent:{qid}`, `wiki:prop:{pid}`, `wiki:out:{qid}`, `wiki:in:{qid}`, `wiki:wp:{title}`

This path is **not** used by `init_wikidata` / `system.answer()` today (only `cache=` is). Prefer `cache=` + `wd:*` for new work.

Redis failures never raise — client falls back to L1 then backend.

---

## Local offline SPARQL

For zero-latency k-hop on a clone, deploy QEndpoint and tunnel to it:

**[deploy_local_wikidata-v2.md](deploy_local_wikidata-v2.md)**

Then set:

```yaml
wikidata:
  sparql_endpoint: http://127.0.0.1:30162/api/endpoint/sparql
```

Integration test (from repo root):

```bash
ssh -fN -L 30162:n0162:1234 t2   # example tunnel
export LANGGRAPH_TEST_SPARQL_URL=http://127.0.0.1:30162/api/endpoint/sparql
uv run pytest langgraph_coe/tests/phase0/test_wikidata_local_integration.py -v
```

---

## Single-flight coalescing

`_coalesce(key, fetch_fn)` — concurrent identical fetches share one backend call. Keys include `search_e:…`, `ent:…`, `out:…`, etc.

Cancellation: follower cancel propagates; leader cancel lets the next waiter become leader. See `tests/wikidata/test_async_correctness.py`.

---

## Retry and 429

`_call_with_retry`: up to 3 attempts on `WikidataRateLimitError`; sleeps `retry_after` or `2**attempt`. **Semaphore is released before sleep** — required to avoid deadlock at `concurrency_limit=1`.

---

## Tests

```bash
# Fast contract suite (no network)
uv run pytest tests/wikidata -v -m "not requires_wikidata"

# Live public API smoke
uv run pytest tests/wikidata/test_live_smoke.py -m requires_wikidata -v

# LangGraph @tools + local SPARQL (tunnel required)
uv run pytest langgraph_coe/tests/phase0/test_wikidata_local_integration.py -v

# Redis wd:* cache wiring (fakeredis)
uv run pytest langgraph_coe/tests/phase0/test_redis_cache_wikidata.py -v
```

| Test module | Focus |
|-------------|--------|
| `test_entity_linking.py` | `link_entities` |
| `test_enrichment.py` | `enrich_entities` |
| `test_property_search.py` | `search_properties` |
| `test_khop_triples.py` | `get_k_hop_triples` |
| `test_caching.py` | L1 + legacy `redis=` |
| `test_rate_limit.py` | 429 / RPS |
| `test_concurrency.py` | Semaphore / gather |
| `test_async_correctness.py` | Single-flight / cancel |
| `test_tools_integration.py` | `@tool` + ContextVar + pruning |

---

## Maintainer notes

### New backend

Implement the seven `WikidataBackend` methods; pass `WikidataClient(backend=…)`.

### New well-known PID

Add to `wikidata_properties.py` → `PROPERTY_LABELS` (and optionally `DEFAULT_PROPERTIES`).

### New client method

1. Extend protocol + `HTTPWikidataBackend` + `FakeWikidataBackend`.
2. Use `_fetch_cached` / `_call_with_retry` / `_coalesce` as appropriate.
3. Add `tests/wikidata/test_<feature>.py`.

---

## Gotchas

### LangChain `BaseTool.ainvoke` and ContextVar

`ainvoke` runs the tool in a child task with a **copy** of the parent context. Do not expect `ContextVar.set()` inside a tool to update the parent.

**Fix:** bind one mutable `_SessionState` per question; mutate `.visited` / `.hop_count` on that object. `reset_wikidata_session()` rebinds a fresh object in the current task.

### Rate-limit routing

Only SPARQL edge fetches use `_sparql_limiter`. `get_entity_details` uses `_wiki_limiter`.

### Tuple JSON in legacy `redis=` path

Outgoing/incoming edges use `redis_normalize=lambda lst: [tuple(e) for e in lst]` in `_fetch_cached`.

### Relaxed QID/PID regex

`Q[A-Za-z0-9_]+` supports test fixtures like `Q_DIVERSE_0`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| Hop budget never exhausts | ContextVar set inside tool | Mutate `_get_session().hop_count` |
| All calls serialized | Low RPS / wrong limiter | Check SPARQL vs wiki routing |
| 429 on first call | Real Wikidata throttle | Lower RPS; backoff |
| N duplicate backend hits | Coalesce key mismatch | Align `top_k`, casing, batch ordering |
| Stage A warnings | `reranker_url` unreachable | Set `null` or deploy reranker ([deploy_reranker_server.md](deploy_reranker_server.md)) |
| SPARQL timeout locally | Tunnel down / wrong URL | See [deploy_local_wikidata-v2.md](deploy_local_wikidata-v2.md) |
| `wd:*` never in Redis | `cache.enabled: false` | [setup_redis_cache.md](setup_redis_cache.md) |

---

## Legacy code

Do not extend `coe/retrieval/wikidata.py` (sync legacy stack). New behavior belongs in `langgraph_coe/tools/`.

---

## Related docs

| Doc | Topic |
|-----|--------|
| [setup_redis_cache.md](setup_redis_cache.md) | Redis for `wd:*`, LLM db 0, `web:*` |
| [deploy_local_wikidata-v2.md](deploy_local_wikidata-v2.md) | Offline QEndpoint SPARQL |
| [deploy_reranker_server.md](deploy_reranker_server.md) | Stage A reranker endpoint |
| `langgraph_coe/implementation_plan.md` §3.4 | Cache design rationale |
