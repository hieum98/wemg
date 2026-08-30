# Co-Evolving Graph and Text Memory for Training-Free Multi-Hop Question Answering

COE is a research-oriented question-answering system that combines graph retrieval, dense retrieval, and LLM reasoning.

The central novelty of this research is memory management: a coordinated textual memory and graph memory that are continuously synchronized during reasoning.

> **Experimental record: [`docs/RESULTS.md`](docs/RESULTS.md).** One page covering what the
> measurements do and do not support, the noise floor that bounds every accuracy figure
> (14.5% of rows flip between identical configurations), and the exact commands to reproduce
> each number. Read it before quoting any result from this repository.

The implementation is the **`langgraph_coe`** package, built on
[LangGraph](https://github.com/langchain-ai/langgraph). (An earlier hand-rolled `coe`
package has been removed; `langgraph_coe` is the only implementation.)

## Highlights

COE is designed around graph-text memory coordination as the primary research contribution, not retrieval alone.

- Memory-centric reasoning core: synchronized textual memory and graph memory.
- Bidirectional memory updates: evidence flows from text into graph structure and from graph structure back into textual context.
- Search-time synchronization: memory coordination is integrated into each reasoning step, not only post-processing.
- Persistent memory state across reasoning steps improves decomposition, verification, and final synthesis.
- Textual memory captures evidence snippets, intermediate reasoning traces, and model-generated hypotheses.
- Graph memory preserves entity-relation structure for relational consistency and multi-hop reasoning.
- Triple retrieval: Wikidata graph evidence plus corpus and web evidence.
- Two reasoning strategies: Monte Carlo Tree Search (MCTS) and chain-of-thought (CoT).
- An explicit **plan channel** conditioning decomposition, with a per-intent ledger (on by
  default; it reduces cost at unchanged accuracy — see [`docs/RESULTS.md`](docs/RESULTS.md)).
- Evaluation pipeline: dataset runs, metrics, and per-question artifacts.

## Architecture at a glance

Main request flow:

1. `answer(question)` in `langgraph_coe/system.py` builds the initial state and selects the `cot` or `mcts` graph.
2. The strategy graph decomposes the question, fans out retrieval, and reasons over the result. CoT: `gen_plan → gen_subq → [KG | web | corpus] → rerank → extract → gen_subanswers → mem_update → plan_gate → loop`.
3. Knowledge-base retrieval uses iterative 1-hop expansion, with an LLM or cross-encoder pruning stage over candidate triples.
4. `MemoryUpdateGraph` performs graph-text memory coordination after each reasoning step: consolidating textual memory and merging extracted triples into the graph.
5. The system returns an `AnswerResult`.

Core modules:

- `langgraph_coe/system.py`: top-level `answer()` / `answer_batch()` entry points and runtime init.
- `langgraph_coe/graphs/`: compiled LangGraph strategies (`cot.py`, `mcts.py`) plus KG search, web research, and memory-update subgraphs.
- `langgraph_coe/tools/`: Wikidata, dense retrieval, web search, and cache tools.
- `langgraph_coe/roles.py`: role definitions and structured I/O schemas.
- `langgraph_coe/llm.py`: tiered role→model routing, prompt-budget guard, cost metering.
- `langgraph_coe/config.py` + `config.yaml`: `LangGraphCoeConfig` schema and tier configuration.
- `langgraph_coe/evaluation/`: dataset evaluation CLI and runner.

For the LangGraph system design (state, reducers, CoT/MCTS graph topologies, subgraphs) see
[`docs/design_langgraph_coe.md`](docs/design_langgraph_coe.md). For the package guide see
[`langgraph_coe/README.md`](langgraph_coe/README.md).

## Installation

Requirements:

- Python 3.10+
- An OpenAI-compatible LLM endpoint (local SGLang, or a managed provider via LiteLLM)
- Optional Redis for caching

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Use the virtualenv, not conda — some older docs in this repo still say `conda run -n coe`,
which is stale.

## Quick start

The public API is async. Answer a single question:

```python
import asyncio
from langgraph_coe.system import answer

result = asyncio.run(answer("Who directed Inception?"))
print(result.answer)
print(result.concise_answer)
```

Answer many questions concurrently:

```python
import asyncio
from langgraph_coe.system import answer_batch

results = asyncio.run(answer_batch(["Q1", "Q2", "Q3"], max_workers=4))
```

## Configuration

- Default config file: `langgraph_coe/config.yaml`
- Evaluation config: `langgraph_coe/config.eval.yaml` (locally-served Qwen3 via SGLang)
- Schema and validation: `langgraph_coe/config.py` (`LangGraphCoeConfig`)
- Override format: dotted `key=value` arguments

LLM roles are organized into **tiers**, so model size and reasoning can be matched to what
each role actually does. `config.eval.yaml` uses:

| tier | model | reasoning | roles |
| --- | --- | --- | --- |
| `heavy` | Qwen3-8B | **on** | subquestion generation, answering, self-correction, final synthesis, web research |
| `plan` | Qwen3-8B | **on** | planner |
| `consolidate` | Qwen3-8B | off | memory consolidation |
| `medium` | Qwen3-4B | off | extraction, verification, scoring |
| `light` | Qwen3-1.7B | off | NER, relation extraction, open IE, triple pruning, query rephrasing |

Reasoning is enabled **selectively, and that is a measured choice**: globally it costs ~19.5k
output tokens per question and buys no accuracy, and on the extraction/consolidation roles it
actively suppresses evidence (38% fewer facts extracted, ~50% fewer memory items retained),
because those are recall tasks and reasoning makes the model filter harder. Selective
reasoning gave identical accuracy for 59.7% less reasoning spend. See
[`docs/RESULTS.md`](docs/RESULTS.md).

On SGLang the switch is `enable_thinking` (forwarded as `chat_template_kwargs`, which the
Qwen3 chat template reads), and the server must be launched with `--reasoning-parser qwen3`.
`reasoning_effort` is the Bedrock-only equivalent and is inert on SGLang.

Environment variables:

- `API_KEY` / `OPENAI_API_KEY`: LLM key and related embedding keys if unset in config.
- `SERPER_API_KEY`: web search key (drives the `builtin` provider).
- `BRAVE_API_KEY`, `TAVILY_API_KEY`: keys for the `brave` and `tavily` providers.
- `SEARXNG_URL`: overrides `web_search.searxng_url` (default `http://localhost:8080`).
- `COE_SPARQL_ENDPOINT`: overrides `wikidata.sparql_endpoint`.
- `REDIS_PASSWORD`: Redis password.
- A repo-level `.env` file is loaded automatically when present.

## Setup scripts

Infrastructure launchers live in [`setup/`](setup/):

```bash
./setup/sglang_up.sh           # start the Qwen3-8B / 4B / 1.7B servers
./setup/sglang_up.sh --check   # report which endpoints are reachable
./setup/sglang_up.sh --down    # stop them

./setup/searxng_up.sh          # start the local SearXNG container
./setup/searxng_up.sh --down   # stop and remove it

./setup/wikidata_up.sh         # probe the local Wikidata SPARQL endpoint
./setup/wikidata_up.sh status  # container, restart policy, index integrity
./setup/wikidata_up.sh build   # rebuild the index from the dump (~1h40m)
```

`wikidata_up.sh status` is worth running before a sweep: it reports whether the index
directory the container bind-mounts still exists on the host. If it does not, QLever keeps
answering from unlinked inodes it holds open and looks perfectly healthy, but the endpoint
cannot survive a stop or a reboot — and `stop` refuses to run in that state rather than
destroying a ~1h40m rebuild.

## Web search providers

`web_search.providers` in the config is a fallback chain, tried in order until one
returns results. Order it unmetered → metered → billed; see the provider-chain note
in `langgraph_coe/tools/web.py` for the measured recall behind the default order.

| Provider | Cost | Notes |
| --- | --- | --- |
| `wikipedia` | unmetered | MediaWiki API, no key; returns page text inline (no crawl) |
| `searxng` | unmetered | Local container; needs `./setup/searxng_up.sh` |
| `builtin` | Serper credits, or free `ddgs` when unkeyed | Legacy slot |
| `brave` | 2k queries/month free | Needs `BRAVE_API_KEY` |
| `tavily` | billed | Needs `TAVILY_API_KEY` |

`wikipedia` leads the chain deliberately: MuSiQue's gold paragraphs are drawn from
Wikipedia, so it is the gold source rather than a fallback. It is also rate-limited in
practice — sustained parallel sweeps draw HTTP 429s, after which the chain falls through to
a non-gold source and retrieval quality drops. Run **at most two evaluation arms
concurrently**, and cap `max_concurrent` for MCTS (see `docs/RESULTS.md`).

Note that `ddgs` (the unkeyed `builtin` path) is unreachable from some hosts —
DuckDuckGo DNS is intercepted on the eval box and the connect times out — which is
why the chain exists. `read_provider_usage()` reports per-provider call counts so
spend against the finite buckets is visible after a run.

## Self-hosted Wikidata endpoint (recommended)

For large retrieval workloads (or conversion scripts that map Freebase IDs to Wikidata IDs), use a local SPARQL endpoint instead of the public Wikidata Query Service.

The deployment used for the reported results is described in
[`docs/deploy_local_wikidata-v2.md`](docs/deploy_local_wikidata-v2.md). A qEndpoint-based
alternative is below.

References:

- qEndpoint Wikidata image: <https://github.com/the-qa-company/qEndpoint#qacompanyqendpoint-wikidata>
- Freebase/Wikidata conversion notebook: <https://github.com/yuancu/freebase-wikidata-convert/blob/main/conversion.ipynb>

### 1. Start a local Wikidata endpoint with Docker

Truthy dump (smaller):

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=6G qacompany/qendpoint-wikidata
```

Full Wikidata dump (larger):

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=10G --env HDT_BASE=wikidata_all qacompany/qendpoint-wikidata
```

Notes:

- Keep `MEM_SIZE >= 6G` for truthy and `>= 10G` for full dumps.
- First startup can take a long time because the HDT index is downloaded and initialized.
- Disk usage is large; plan capacity before running.

### 2. Verify endpoint is reachable

```bash
curl -H 'Accept: application/sparql-results+json' \
  http://localhost:1234/api/endpoint/sparql \
  --data-urlencode 'query=SELECT * WHERE { ?s ?p ?o } LIMIT 5'
```

The qEndpoint web UI is available at <http://localhost:1234>.

### 3. Use local endpoint in conversion scripts

The conversion flow in `freebase-wikidata-convert/conversion.ipynb` initializes:

```python
EntityConverter("http://localhost:1234/api/endpoint/sparql")
```

This is strongly recommended for bulk Freebase→Wikidata ID conversion. The public endpoint (`https://query.wikidata.org/sparql`) is much slower and may rate-limit large jobs.

### 4. Point retrieval at the local endpoint

Set `wikidata.sparql_endpoint` in the config (or export `COE_SPARQL_ENDPOINT`) to the
endpoint's SPARQL path — for qEndpoint that is:

```text
http://localhost:1234/api/endpoint/sparql
```

Set it to `null` to fall back to public Wikidata. Note that entity *label* lookups
(`wbsearchentities` / `wbgetentities`) hit the public MediaWiki API rather than SPARQL and
are rate-limited independently; `langgraph_coe/tools/wikidata_backend.py` resolves them
against the local endpoint when the public API answers 429.

Then run the Wikidata tool tests:

```bash
pytest langgraph_coe/tests/unit/test_wikidata_local_fallback.py
```

## Evaluation

```bash
python -m langgraph_coe.evaluation.evaluate \
  --config langgraph_coe/config.eval.yaml \
  dataset_name_or_path=./datasets/musique_depth.jsonl \
  level_column=level \
  output_path=./results/my_run
```

Accepts dotted config overrides such as `search.strategy=cot`,
`search.mcts.num_iterations=8`, `search.plan.enabled=false`,
`llm.tiers.heavy.api_base=http://localhost:30000/v1`, and `cache.enabled=true`. The
resolved config is written to `<output_path>/config.yaml` — read that back rather than
trusting the command line, since it is what the run actually used.

Outputs `evaluation_log.jsonl`, `metrics.json`, `summary.txt`, and per-question
`artifacts/` (plan ledger, retrieval log, consolidated memory, working-memory graph).
`resume` defaults to true, so re-running continues from an existing log.

See `langgraph_coe/evaluation/README.md` for full CLI options and artifact details, and
[`docs/RESULTS.md`](docs/RESULTS.md) for the reporting scripts in `scripts/`.

## Testing

Run the unit suite:

```bash
pytest langgraph_coe/tests/unit -q
```

Full suite, including integration tests that need live servers:

```bash
pytest langgraph_coe/tests
```

See `langgraph_coe/tests/README.md` for markers and environment setup.

## Project layout

```text
langgraph_coe/        # the implementation
  README.md           # package guide (architecture, API, config, setup)
  config.py           # LangGraphCoeConfig schema
  config.yaml         # default config
  config.eval.yaml    # evaluation config (local SGLang Qwen3 tiers)
  system.py           # answer() / answer_batch() entry points
  llm.py              # tiered role->model routing, prompt guard, cost meter
  roles.py            # role definitions and structured I/O
  graphs/             # cot.py, mcts.py + KG / web / memory-update subgraphs
  tools/              # wikidata, retrieval, web search, cache
  evaluation/         # evaluation CLI + README
  scripts/            # dataset builders, smoke test, offline probes
  tests/              # unit / integration / real_servers + README
docs/                 # setup guides + the experimental record
  RESULTS.md          # START HERE: what is claimable, and what is not
  plan_idea_and_results.md         # full evidence behind RESULTS.md
  plan_channel_status_and_plan.md  # internal working log (do not quote)
setup/                # infrastructure launchers (sglang_up.sh, searxng_up.sh)
scripts/              # analysis + reporting over results/ (see RESULTS.md)
  reason_report.py    # paired accuracy/cost report + arm-validity checks
  conversion_report.py, fix_report.py, ...
datasets/             # evaluation sets
examples/
```

## Troubleshooting

- If an LLM call fails to connect, check the tier endpoints: `./setup/sglang_up.sh --check`.
  Each tier has its own `api_base`, so one dead server breaks only the roles on that tier.
- If retrieval fails in corpus mode, verify `retriever.corpus.corpus_path` and
  `retriever.corpus.index_path`. `corpus_search` **raises** when its pipeline is
  uninitialised rather than returning empty, so a missing index fails every hop.
  The wiki23 index is ~99 GB and is loaded without mmap — see the RAM caveat in
  `config.eval.yaml` before enabling it on a new host.
- If web search returns nothing, check the logs for `no results from any provider`:
  that means the whole `web_search.providers` chain is down. Confirm SearXNG is up
  (`./setup/searxng_up.sh`) and that the keyed providers have not hit their quotas
  (Tavily answers HTTP 432 when its plan limit is reached).
- If reasoning appears inline in answers as `<think>...</think>`, the SGLang server was
  launched without `--reasoning-parser qwen3`.
- If cache is enabled but Redis is unavailable, check cache connection settings.

## Contributing

1. Create a feature branch.
2. Add or update tests under `langgraph_coe/tests/`.
3. Run relevant tests and then `pytest langgraph_coe/tests/unit -q`.
4. Open a pull request with clear change notes.

## License

MIT
