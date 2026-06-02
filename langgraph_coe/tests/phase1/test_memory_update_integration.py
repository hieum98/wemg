"""Phase 1 integration smoke test against real model endpoints.

Hits the actual Qwen LLM + embedder deployed via SGLang on the SLURM cluster
(routed through SSH tunnels: ``localhost:30172`` → ``n0172:30000`` for the LLM,
``localhost:30164`` → ``n0164:30000`` for the embedder). Only runs when both
endpoints are reachable, otherwise skipped.

The test exercises the full ``MemoryUpdateGraph`` (consolidation pre, OpenIE
extraction + entity linking, merge/prune, textualise, optional consolidation
post) end-to-end on a small synthetic memory blob, with the Wikidata
``link_entities`` tool mocked so the test stays self-contained (no network
calls beyond the model endpoints).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
import networkx as nx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, TierConfig
from langgraph_coe.llm import RoleModelRegistry


LLM_URL = os.environ.get("LANGGRAPH_TEST_LLM_URL", "http://localhost:30172/v1")
LLM_MODEL = os.environ.get("LANGGRAPH_TEST_LLM_MODEL", "openai/Qwen/Qwen3-8B")
EMBED_URL = os.environ.get("LANGGRAPH_TEST_EMBED_URL", "http://localhost:30164/v1")
EMBED_MODEL = os.environ.get("LANGGRAPH_TEST_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


def _endpoint_alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{url.rstrip('/')}/models")
            return resp.status_code == 200
    except Exception:
        return False


def _real_endpoints_available() -> bool:
    return _endpoint_alive(LLM_URL) and _endpoint_alive(EMBED_URL)


pytestmark = pytest.mark.skipif(
    not _real_endpoints_available(),
    reason=(
        "Real model endpoints unreachable. Open SSH tunnels:\n"
        "  ssh -fN -L 30172:n0172:30000 -L 30164:n0164:30000 t2"
    ),
)


def _build_real_registry() -> RoleModelRegistry:
    cfg = LangGraphCoeConfig.from_yaml()
    # All tiers point at the same Qwen3-8B endpoint — we want one round-trip per
    # role call, not three different deployments. Thinking stays ON: disabling it
    # tanks Qwen3 quality on reasoning-heavy roles (consolidation, relation
    # extraction, pruning), so the fallback parser must handle multi-block
    # ``content`` responses instead.
    tier = TierConfig(
        model_name=LLM_MODEL,
        api_base=LLM_URL,
        api_key="EMPTY",
        temperature=0.2,
        # Thinking-mode Qwen3 spends thousands of tokens reasoning before
        # emitting the JSON output; an under-budget cap truncates mid-thought
        # and leaves no parseable answer.
        max_tokens=16384,
        max_input_tokens=32768,
        top_p=0.95,
        enable_thinking=True,
        max_retries=2,
        timeout=300,
    )
    cfg.llm.tiers = {"heavy": tier, "medium": tier, "light": tier}
    cfg.llm.api_key = "EMPTY"
    return RoleModelRegistry(cfg.llm)


class _StubLinkEntities:
    """Stand-in for the Wikidata ``link_entities`` tool used by the graph.

    Returns a deterministic QID mapping for known surface forms; falls back to
    an empty list for unknown names. Keeps the integration test independent of
    the live Wikidata SPARQL endpoint so failures isolate to the LLM path.
    """

    name = "link_entities"

    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping
        self.invocations: List[Any] = []

    async def ainvoke(self, inp: Any, *args: Any, **kwargs: Any) -> List[Dict[str, str]]:
        self.invocations.append(inp)
        names = inp.get("entity_names", inp) if isinstance(inp, dict) else inp
        out: List[Dict[str, str]] = []
        for name in names or []:
            qid = self._mapping.get(str(name))
            if qid:
                out.append({"name": str(name), "qid": qid, "description": ""})
        return out


async def test_memory_update_graph_runs_against_real_qwen(monkeypatch, caplog):
    """Full graph executes end-to-end with the real Qwen3-8B LLM."""
    from langgraph_coe.graphs import memory_update as mem_mod

    stub = _StubLinkEntities({
        "France": "Q142",
        "Paris": "Q90",
        "Eiffel Tower": "Q243",
    })
    monkeypatch.setattr(mem_mod, "link_entities", stub, raising=False)

    registry = _build_real_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    state: Dict[str, Any] = {
        "question": "What is the capital of France and what famous monument is there?",
        "new_text_items": [
            "Paris is the capital of France.",
            "The Eiffel Tower is a famous landmark located in Paris.",
        ],
        "new_raw_triples": [],
        "current_text_memory": [
            "France is a country located in Western Europe.",
        ],
        "current_graph": nx.DiGraph(),
        "entity_dict": {},
    }

    final = await graph.ainvoke(state)
    print(final)

    # ── structural assertions ────────────────────────────────────────────────
    for key in (
        "consolidated_memory",
        "extracted_relations",
        "linked_entities",
        "updated_text_memory",
        "updated_graph",
        "updated_entity_dict",
    ):
        assert key in final, f"missing state key: {key}"

    assert isinstance(final["updated_text_memory"], list)
    assert isinstance(final["updated_graph"], nx.DiGraph)
    assert isinstance(final["updated_entity_dict"], dict)

    # ── consolidation produced something non-trivial ────────────────────────
    assert final["updated_text_memory"], "post-consolidation produced empty memory"
    joined = " ".join(final["updated_text_memory"]).lower()
    assert "paris" in joined and "france" in joined, (
        f"expected key entities in consolidated memory, got: {final['updated_text_memory']!r}"
    )

    # ── relation extraction yielded at least one relation ───────────────────
    assert final["extracted_relations"], "open_ie returned no relations"

    # ── entity linking either resolved via stub or fell back gracefully ─────
    assert stub.invocations, "graph did not invoke link_entities tool"

    # ── graph mutation isolation ────────────────────────────────────────────
    assert final["updated_graph"] is not state["current_graph"]


async def test_memory_update_graph_handles_existing_entity_dict(monkeypatch):
    """Known entities in ``entity_dict`` should not trigger redundant link calls."""
    from langgraph_coe.graphs import memory_update as mem_mod
    from langgraph_coe.roles import WikidataEntity

    stub = _StubLinkEntities({"Paris": "Q90"})
    monkeypatch.setattr(mem_mod, "link_entities", stub, raising=False)

    registry = _build_real_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    final = await graph.ainvoke({
        "question": "Where is Paris?",
        "new_text_items": ["Paris is in France."],
        "new_raw_triples": [],
        "current_text_memory": [],
        "current_graph": nx.DiGraph(),
        "entity_dict": {
            "Q142": WikidataEntity(qid="Q142", label="France", description="country"),
        },
    })

    assert "Q142" in final["updated_entity_dict"], "pre-existing QID was dropped"
    print(final)

    if stub.invocations:
        # Whichever names get sent must not include 'France' — already known.
        invoked = stub.invocations[0]
        names = invoked.get("entity_names", []) if isinstance(invoked, dict) else invoked
        assert "France" not in (names or []), (
            f"link_entities should skip names matching known entity_dict labels; saw {names!r}"
        )
