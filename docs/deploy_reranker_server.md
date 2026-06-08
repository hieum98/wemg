# Deploying the Qwen3-Reranker Server (SGLang)

This guide walks through deploying the **Qwen3-Reranker** model behind SGLang's
Cohere-compatible `/v1/rerank` endpoint, which the `langgraph_coe` corpus
retrieval pipeline (`langgraph_coe/tools/retrieval.py`) calls to reorder FAISS
candidates by relevance.

> **TL;DR — the one gotcha that breaks everything:**
> Qwen3-Reranker scores by decoder-only `yes`/`no` logprobs, so it must run in
> **generation mode**. Launch it **with** `--chat-template` and **without**
> `--is-embedding`. Getting this wrong produces either constant/degenerate scores
> or an HTTP 400. See [Troubleshooting](#troubleshooting).

---

## Prerequisites

- A GPU node (the reference deployment uses a single **H100 80GB** for
  `Qwen3-Reranker-4B`).
- A conda env with SGLang installed (reference: env `sglang`, SGLang `0.5.9`).
- The chat template shipped in this repo:
  `wemg/utils/sglang-qwen3-reranker.jinja`.

The template frames each (query, document) pair for `yes`/`no` scoring:

```jinja
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ instruct | default("Given a web search query, retrieve relevant passages that answer the query.") }}
<Query>: {{ messages[0]["content"] }}
<Document>: {{ messages[1]["content"] }}<|im_end|>
<|im_start|>assistant{{ '\n' }}
```

---

## Launch the Server

Run on the GPU node (here, `n0997`, port `30000`):

```bash
conda activate sglang

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-4B \
  --served-model-name Qwen3-Reranker-4B \
  --chat-template /gpfs/projects/uonlp/hieum/wemg/wemg/utils/sglang-qwen3-reranker.jinja \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
```

Flag-by-flag:

| Flag | Why |
| --- | --- |
| `--chat-template …qwen3-reranker.jinja` | **Mandatory.** Without it SGLang does not recognize the model as a decoder-only reranker and returns HTTP 400. |
| `--is-embedding` | **Must be OMITTED.** Qwen3-Reranker uses generation-mode logprob scoring; with this flag the `/rerank` path returns constant/garbage scores (or a 400 in newer SGLang). |
| `--trust-remote-code` | Recommended by the SGLang docs for Qwen3-Reranker. |
| `--disable-radix-cache` | Recommended; avoids prefix-cache scoring artifacts across requests. |
| `--served-model-name` | The `model` id clients send (`Qwen3-Reranker-4B`); must match `reranker.model_name` in config. |

> **Note:** Swap `Qwen/Qwen3-Reranker-4B` for `Qwen3-Reranker-0.6B` / `-8B` to
> trade quality for memory. Update `--served-model-name` **and** the client
> config (`reranker.model_name`) to match.

Reference: <https://docs.sglang.io/docs/supported-models/rerank_models>

---

## Connecting from a Workstation (SSH Tunnel)

If you run tests/clients off-node, forward the port:

```bash
# Map local 30002 → n0997:30000
ssh -fN -L 30002:n0997:30000 <login-host>
```

The client then targets `http://localhost:30002/v1`.

---

## Verify the Deployment

A reachable server is **not** the same as a correctly-scoring one. Use the probe
script — it posts curated query/document contrasts and checks that the relevant
passage ranks #1 with separable scores:

```bash
conda activate wemg

python langgraph_coe/scripts/probe_reranker_server.py \
  --url http://n0997:30000/v1 \
  --model Qwen3-Reranker-4B
```

A healthy server prints **continuously varying** scores and:

```
SUMMARY: 6 passed, 0 failed, 6 total
```

Example of a healthy response (note the spread, not two frozen values):

```
#1 idx=0 score=0.97145585  'Paris is the capital and largest city of France.' <-- expected #1
#2 idx=2 score=0.89405171  'Cricket is a bat-and-ball game ...'
#3 idx=1 score=0.80957725  'The Python programming language ...'
```

The pytest equivalent (exercises the real client code path):

```bash
LANGGRAPH_TEST_RERANK_URL=http://n0997:30000/v1 \
LANGGRAPH_TEST_RERANK_MODEL=Qwen3-Reranker-4B \
pytest langgraph_coe/tests/phase0/test_retrieval_integration.py -v -s
```

---

## Wiring into `langgraph_coe`

The reranker is configured by `RerankerConfig` in `langgraph_coe/config.py`:

```python
class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen3-Reranker-4B"     # must match --served-model-name
    url: str = "http://n0999:30002/v1"        # base incl. /v1; client appends /rerank
    api_key: Optional[str] = "EMPTY"
    top_k: int = 10                            # passages kept after rerank
    instruction: Optional[str] = (             # sent as JSON `instruct`
        "Given a web search query, retrieve relevant passages that answer the query."
    )
```

Point `url` at your deployment (e.g. `http://n0997:30000/v1` on-node, or
`http://localhost:30002/v1` through the tunnel). The client
(`call_sglang_reranker`) POSTs `{model, query, documents, instruct}` to
`<url>/rerank` and returns `[(index, score), …]` sorted descending.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| **HTTP 400**: `"Detected Qwen3 reranker chat template, but the server is not in generation mode. Please relaunch without --is-embedding"` | Launched with `--is-embedding`. | Remove `--is-embedding`, keep `--chat-template`. |
| **HTTP 400** / model not recognized as reranker | `--chat-template` missing. | Add `--chat-template …sglang-qwen3-reranker.jinja`. |
| **Flat/constant scores** (every doc gets the same value, e.g. all `0.17328…`) | Server reachable but not discriminating — usually `--is-embedding` set, or wrong/missing template. | Relaunch in generation mode with the correct template; re-run the probe. |
| **Wrong winner, only two distinct score values** | Same as above — degenerate embedding-mode output ranked by position, not relevance. | Same fix; probe should then show varied scores and `6 passed`. |
| Probe `no result rows` / connection error | Server still loading, crashed, or wrong host/port. | Check `GET <url>/models` returns 200; inspect the server process / GPU memory. |

Inspect the running process to confirm the flags:

```bash
ssh n0997 'ps aux | grep launch_server | grep -v grep'
# Expect: --chat-template ...sglang-qwen3-reranker.jinja  AND NO  --is-embedding
```
