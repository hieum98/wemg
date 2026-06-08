"""Real KGSearchGraph integration — live QEndpoint SPARQL + Qwen LLM.

Exercises ``build_kg_search_graph`` end-to-end against:
  - QEndpoint Wikidata SPARQL at ``n0162:1234``
  - SGLang LLM at ``n0152:30000`` (all KG-search tiers)
  - Generation / sampling params from ``langgraph_coe/config.yaml`` (per tier)

Each graph node (``ner_agent``, ``triple_search_agent``, ``enrich``) prints its
full state update. LLM completions and tool calls inside the triple-search ReAct
agent are printed via ``astream_events``.

Run manually (from repo root)::

    uv run pytest langgraph_coe/tests/phase0/test_kg_search_integration.py -v -s

Optional env overrides::

    LANGGRAPH_TEST_SPARQL_URL   default ``http://n0162:1234/api/endpoint/sparql``
    LANGGRAPH_TEST_LLM_URL      default ``http://n0152:30000/v1``
    API_KEY / OPENAI_API_KEY    LLM auth (loaded from repo-root ``.env``)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import httpx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, WikidataConfig
from langgraph_coe.graphs.kg_search import build_kg_search_graph
from langgraph_coe.llm import RoleModelRegistry
from langgraph_coe.tools.wikidata import init_wikidata, reset_wikidata_session

from .._fixtures import log_config_override, override_tier_endpoint

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

SPARQL_URL = os.environ.get(
    "LANGGRAPH_TEST_SPARQL_URL",
    "http://n0162:1234/api/endpoint/sparql",
)
LLM_URL = os.environ.get("LANGGRAPH_TEST_LLM_URL", "http://n0152:30000/v1")

# Reranker for Stage-A pruning. Defaults to the config.yaml endpoint/model; an
# env var can repoint it at an SSH tunnel without editing config.yaml.
_WIKIDATA_CFG_DEFAULTS = LangGraphCoeConfig.from_yaml().wikidata
RERANKER_URL = (
    os.environ.get("LANGGRAPH_TEST_RERANKER_URL") or _WIKIDATA_CFG_DEFAULTS.reranker_url
)
RERANKER_MODEL = (
    os.environ.get("LANGGRAPH_TEST_RERANKER_MODEL")
    or _WIKIDATA_CFG_DEFAULTS.reranker_model
)

# KG subgraph roles → tiers (see config.yaml role_tiers).
_KG_SEARCH_TIERS = ("light", "medium", "classify")

# Proven in ``test_wikidata_local_integration`` (UOregon → Q766145, founded P571).
_QUERY = "When was the University of Oregon founded?"


def _sparql_alive(url: str) -> bool:
    probe = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                url,
                params={"query": probe},
                headers={"Accept": "application/sparql-results+json"},
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            return bool(data.get("results", {}).get("bindings"))
    except Exception:
        return False


def _llm_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{url.rstrip('/')}/models")
            return resp.status_code == 200
    except Exception:
        return False


def _reranker_alive(url: str | None, model: str | None) -> bool:
    """Probe the exact ``/rerank`` endpoint Stage-A uses with a trivial payload."""
    if not url:
        return False
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url.rstrip('/')}/rerank",
                json={
                    "model": model or "reranker",
                    "query": "ping",
                    "documents": ["alpha", "beta"],
                },
            )
            return resp.status_code == 200
    except Exception:
        return False


_SPARQL_UP = _sparql_alive(SPARQL_URL)
_LLM_UP = _llm_alive(LLM_URL)
_RERANKER_UP = _reranker_alive(RERANKER_URL, RERANKER_MODEL)
_STACK_UP = _SPARQL_UP and _LLM_UP and _RERANKER_UP

_skip_reason = (
    f"KGSearchGraph integration requires QEndpoint SPARQL ({SPARQL_URL}, "
    f"up={_SPARQL_UP}), LLM ({LLM_URL}, up={_LLM_UP}) and reranker "
    f"({RERANKER_URL}, up={_RERANKER_UP})."
)

requires_kg_search_stack = pytest.mark.skipif(
    not _STACK_UP,
    reason=_skip_reason,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_wikidata,
    requires_kg_search_stack,
]


def _integration_cfg() -> LangGraphCoeConfig:
    return LangGraphCoeConfig.from_yaml()


def _wikidata_cfg() -> WikidataConfig:
    """config.yaml ``wikidata`` block; only env/endpoint fields deviate (logged)."""
    cfg = _integration_cfg().wikidata.model_copy(deep=True)
    # Endpoint wiring: the SSH-tunnelled / env SPARQL URL replaces config.yaml's.
    cfg.sparql_endpoint = log_config_override(
        "wikidata.sparql_endpoint",
        cfg.sparql_endpoint,
        SPARQL_URL,
        reason="LANGGRAPH_TEST_SPARQL_URL / tunnelled QEndpoint for this run",
    )
    # Stage-A rerank enabled: use the config.yaml reranker unless an env var
    # repoints it at a tunnel (logged only when it deviates from config.yaml).
    cfg.reranker_url = log_config_override(
        "wikidata.reranker_url",
        cfg.reranker_url,
        RERANKER_URL,
        reason="LANGGRAPH_TEST_RERANKER_URL / tunnelled reranker for this run",
    )
    cfg.reranker_model = log_config_override(
        "wikidata.reranker_model",
        cfg.reranker_model,
        RERANKER_MODEL,
        reason="LANGGRAPH_TEST_RERANKER_MODEL for this run",
    )
    # max_hops / pruning_top_k / max_sparql_rps all come from config.yaml.
    return cfg


def _build_registry() -> RoleModelRegistry:
    """Registry with config.yaml generation params; KG tiers → ``LLM_URL``."""
    cfg = _integration_cfg()
    for tier_name in _KG_SEARCH_TIERS:
        cfg.llm.tiers[tier_name] = override_tier_endpoint(
            cfg,
            tier_name,
            api_base=LLM_URL,
            reason="KG-search integration LLM endpoint (LANGGRAPH_TEST_LLM_URL)",
        )
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or cfg.llm.api_key
        or "EMPTY"
    )
    cfg.llm.api_key = api_key
    return RoleModelRegistry(cfg.llm)


def _truncate(value: Any, *, max_len: int = 1200) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return f"{value[:max_len]}… [{len(value)} chars total]"
    if isinstance(value, list):
        return [_truncate(v, max_len=max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate(v, max_len=max_len) for k, v in value.items()}
    return value


def _print_block(title: str, payload: Any) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)
    if isinstance(payload, (dict, list)):
        text = json.dumps(_truncate(payload), indent=2, ensure_ascii=False, default=str)
    else:
        text = str(payload)
    print(text, flush=True)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in ("thinking", "reasoning", "redacted_thinking"):
                    key = "thinking" if "thinking" in block else "reasoning"
                    val = block.get(key) or block.get("text") or block.get("content")
                    if val:
                        parts.append(f"[{btype}]\n{val}")
                    continue
                for key in ("text", "content", "value"):
                    if isinstance(block.get(key), str):
                        parts.append(block[key])
                        break
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _print_tier_summary() -> None:
    cfg = _integration_cfg()
    rows: list[dict[str, Any]] = []
    for tier_name in _KG_SEARCH_TIERS:
        t = cfg.llm.tiers[tier_name]
        rows.append(
            {
                "tier": tier_name,
                "model": t.model_name,
                "api_base": LLM_URL,
                "temperature": t.temperature,
                "max_tokens": t.max_tokens,
                "enable_thinking": t.enable_thinking,
                "thinking_budget": t.thinking_budget,
                "top_p": t.top_p,
                "top_k": t.top_k,
                "min_p": t.min_p,
                "presence_penalty": t.presence_penalty,
                "frequency_penalty": t.frequency_penalty,
                "repetition_penalty": t.repetition_penalty,
                "seed": t.seed,
            }
        )
    _print_block("LLM tier config (generation params from config.yaml)", rows)


async def _run_graph_with_verbose_trace(
    graph: Any,
    initial: Mapping[str, Any],
) -> Dict[str, Any]:
    """Stream the graph once; print node updates + agent/tool events."""
    final: Dict[str, Any] = dict(initial)
    node_names = {"ner_agent", "triple_search_agent", "enrich"}

    async for event in graph.astream_events(initial, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        data = event.get("data") or {}
        meta = event.get("metadata") or {}
        node = meta.get("langgraph_node")

        if kind == "on_chain_end" and node in node_names:
            output = data.get("output")
            if isinstance(output, dict):
                _print_block(f"NODE OUTPUT: {node}", output)
                final.update(output)

        elif kind == "on_chat_model_end":
            output = data.get("output")
            content = getattr(output, "content", output)
            text = _content_to_text(content)
            label = f"LLM ({node or name})"
            _print_block(label, text)

        elif kind == "on_tool_end":
            tool_output = data.get("output")
            tool_input = data.get("input")
            _print_block(
                f"TOOL END: {name}",
                {
                    "input": _truncate(tool_input, max_len=800),
                    "output": _truncate(tool_output, max_len=2000),
                },
            )

    return final


@pytest.fixture(autouse=True)
def _wire_wikidata():
    from langgraph_coe.tools import wikidata as wd_mod

    wd_mod._wikidata_client = None
    wd_mod._wikidata_config = None
    init_wikidata(_wikidata_cfg())
    reset_wikidata_session()
    yield
    wd_mod._wikidata_client = None
    wd_mod._wikidata_config = None


@requires_kg_search_stack
async def test_kg_search_graph_live_verbose():
    """Full KG search: NER → link → triple agent → enrich; print every step."""
    _print_block(
        "integration setup",
        {
            "sparql_endpoint": SPARQL_URL,
            "llm_url": LLM_URL,
            "query": _QUERY,
            "kg_search_tiers": list(_KG_SEARCH_TIERS),
        },
    )
    _print_tier_summary()

    registry = _build_registry()
    graph = build_kg_search_graph(registry)

    initial = {
        "subquery": _QUERY,
        "original_query": _QUERY,
        "context": "",
        "known_entities_summary": "",
        "errors": [],
    }

    final = await _run_graph_with_verbose_trace(graph, initial)

    _print_block("FINAL STATE (merged)", final)

    errors = final.get("errors") or []
    assert not errors, f"unexpected graph errors: {errors!r}"

    linked = final.get("linked_entities") or []
    assert linked, "NER + link_entities should resolve at least one entity"

    qids = final.get("qids_for_triples") or []
    assert qids, f"expected seed QIDs after linking; linked={linked!r}"

    triples = final.get("triples") or []
    if not triples:
        print(
            "\n[warn] triple_search_agent returned no triple strings "
            "(pruner may have dropped all candidates); see TOOL/LLM trace above.",
            flush=True,
        )

    labels = final.get("enriched_entity_labels") or []
    assert labels, "enrich node should return entity labels"
