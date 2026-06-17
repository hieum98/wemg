"""Integration smoke test — live ``MemoryUpdateGraph`` (LLM + Wikidata).

Exercises ``build_memory_update_graph`` end-to-end against:
  - SGLang LLM at ``n0152:30000`` (all tiers; defaults from ``config.yaml``)
  - Live Wikidata ``link_entities`` via ``init_wikidata`` (search API + QEndpoint)

Run manually (from repo root)::

    uv run pytest langgraph_coe/tests/phase1/test_memory_update_integration.py -v -s

Optional env overrides::

    LANGGRAPH_TEST_LLM_URL override tier ``api_base`` (default: config.yaml)
    LANGGRAPH_TEST_SPARQL_URL override ``wikidata.sparql_endpoint``
    API_KEY / OPENAI_API_KEY LLM auth (loaded from repo-root ``.env``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import httpx
import networkx as nx
import pytest

from langgraph_coe.config import LangGraphCoeConfig, WikidataConfig
from langgraph_coe.llm import RoleModelRegistry
from langgraph_coe.tools.wikidata import init_wikidata, reset_wikidata_session

from .._fixtures import log_config_override, override_tier_endpoint
from .._servers import endpoint_alive as _endpoint_alive

_REPO_ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Roles exercised by MemoryUpdateGraph → tiers (see config.yaml role_tiers).
_MEMORY_UPDATE_TIERS = ("medium", "classify")

SPARQL_URL = os.environ.get(
    "LANGGRAPH_TEST_SPARQL_URL",
    LangGraphCoeConfig.from_yaml().wikidata.sparql_endpoint,
)
LLM_URL = os.environ.get(
    "LANGGRAPH_TEST_LLM_URL",
    LangGraphCoeConfig.from_yaml().llm.tiers["medium"].api_base,
)


def _integration_cfg() -> LangGraphCoeConfig:
    return LangGraphCoeConfig.from_yaml()


def _wikidata_cfg() -> WikidataConfig:
    cfg = _integration_cfg().wikidata.model_copy(deep=True)
    cfg.sparql_endpoint = log_config_override(
        "wikidata.sparql_endpoint",
        cfg.sparql_endpoint,
        SPARQL_URL,
        reason="LANGGRAPH_TEST_SPARQL_URL / live QEndpoint for this run",
    )
    return cfg


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


_LLM_UP = _endpoint_alive(LLM_URL)
_SPARQL_UP = _sparql_alive(SPARQL_URL)
_STACK_UP = _LLM_UP and _SPARQL_UP

_skip_reason = (
    f"MemoryUpdateGraph integration requires LLM ({LLM_URL}, up={_LLM_UP}) and "
    f"QEndpoint SPARQL ({SPARQL_URL}, up={_SPARQL_UP})."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_wikidata,
    pytest.mark.skipif(not _STACK_UP, reason=_skip_reason),
]


def _build_registry() -> RoleModelRegistry:
    """Registry with config.yaml generation params; memory tiers → ``LLM_URL``."""
    cfg = _integration_cfg()
    for tier_name in _MEMORY_UPDATE_TIERS:
        cfg.llm.tiers[tier_name] = override_tier_endpoint(
            cfg,
            tier_name,
            api_base=LLM_URL,
            reason="MemoryUpdate integration LLM endpoint (LANGGRAPH_TEST_LLM_URL)",
        )
    api_key = (
        os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or cfg.llm.api_key
        or "EMPTY"
    )
    cfg.llm.api_key = api_key
    return RoleModelRegistry(cfg.llm)


@pytest.fixture
def live_wikidata():
    """Initialise the real Wikidata client for ``link_entities``."""
    reset_wikidata_session()
    init_wikidata(_wikidata_cfg())
    yield
    reset_wikidata_session()


async def test_memory_update_graph_runs_against_real_servers(live_wikidata):
    """Full graph executes end-to-end with live LLM + Wikidata entity linking."""
    from langgraph_coe.graphs import memory_update as mem_mod

    registry = _build_registry()
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

    assert final["updated_text_memory"], "post-consolidation produced empty memory"
    joined = " ".join(final["updated_text_memory"]).lower()
    assert "paris" in joined and "france" in joined, (
        f"expected key entities in consolidated memory, got: {final['updated_text_memory']!r}"
    )

    assert final["extracted_relations"], "open_ie returned no relations"
    assert final["linked_entities"], "link_entities resolved no entities"

    # France and Paris should link to well-known QIDs when Wikidata is reachable.
    linked_qids = set(final["linked_entities"].values())
    assert "Q142" in linked_qids or "Q90" in linked_qids, (
        f"expected France (Q142) or Paris (Q90) among linked QIDs; got {linked_qids!r}"
    )

    assert final["updated_graph"] is not state["current_graph"]
    assert final["updated_graph"].number_of_edges() >= 1, (
        "expected at least one pruned relation in the updated graph"
    )


async def test_memory_update_graph_handles_existing_entity_dict(live_wikidata):
    """Known entities in ``entity_dict`` should not be re-sent to link_entities."""
    from langgraph_coe.graphs import memory_update as mem_mod
    from langgraph_coe.roles import WikidataEntity

    registry = _build_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    final = await graph.ainvoke(
        {
            "question": "Where is Paris?",
            "new_text_items": ["Paris is in France."],
            "new_raw_triples": [],
            "current_text_memory": [],
            "current_graph": nx.DiGraph(),
            "entity_dict": {
                "Q142": WikidataEntity(
                    qid="Q142", label="France", description="country"
                ),
            },
        }
    )

    assert "Q142" in final["updated_entity_dict"], "pre-existing QID was dropped"
    print(final)

    # France is already known via entity_dict label — only Paris (or similar) should link.
    linked_names = {k.lower() for k in final["linked_entities"]}
    assert "france" not in linked_names, (
        f"link_entities should skip France (already in entity_dict); linked {linked_names!r}"
    )


def _seed_graph_with_qid_edge(
    src_key: str, src_name: str, dst_key: str, dst_name: str, relation: str
) -> nx.DiGraph:
    """Build a DiGraph with one QID-keyed edge, matching ``_add_triple_to_graph``."""
    g = nx.DiGraph()
    g.add_node(src_key, name=src_name, qid=src_key)
    g.add_node(dst_key, name=dst_name, qid=dst_key)
    g.add_edge(src_key, dst_key, relation={relation})
    return g


async def test_memory_update_raw_triples_dedup_and_non_mutation(live_wikidata):
    """new_raw_triples coercion + dedup against a pre-populated graph.

    Covers branches the happy-path tests never reach:
      - ``_coerce_raw_triple_to_relation`` on a raw dict and a ``Relation``
      - ``_relation_already_in_graph`` filtering an edge already in the graph
        so the (expensive) pruner never re-examines it
      - ``merge_and_prune`` working on a *copy* — the caller's graph is untouched
    """
    from langgraph_coe.graphs import memory_update as mem_mod
    from langgraph_coe.roles import Relation

    registry = _build_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    # Pre-existing edge: France --has_capital--> Paris (QID-keyed).
    seeded = _seed_graph_with_qid_edge("Q142", "France", "Q90", "Paris", "has_capital")

    final = await graph.ainvoke(
        {
            "question": "What is France's capital and which countries does it border?",
            # No text items → isolate the raw-triple path (open_ie returns nothing).
            "new_text_items": [],
            "new_raw_triples": [
                # Duplicate of the seeded edge (as a raw dict) → must be filtered.
                {
                    "subject": "France",
                    "subject_id": "Q142",
                    "relation": "has_capital",
                    "object": "Paris",
                    "object_id": "Q90",
                },
                # Genuinely new edge (as a Relation) → eligible for the pruner.
                Relation(
                    subject="France",
                    subject_id="Q142",
                    relation="shares_border_with",
                    object="Germany",
                    object_id="Q183",
                ),
            ],
            "current_text_memory": [],
            "current_graph": seeded,
            "entity_dict": {},
        }
    )
    print(final)

    dup_triple = "France — has_capital — Paris"
    assert dup_triple not in final["kept_triples"], (
        "duplicate edge reached the pruner — dedup filter (_relation_already_in_graph) failed"
    )

    # Non-mutation: the caller's graph object is unchanged.
    assert final["updated_graph"] is not seeded
    assert seeded.number_of_edges() == 1, (
        "caller's graph was mutated by merge_and_prune"
    )
    assert "shares_border_with" not in str(seeded.edges(data=True))

    # The pre-existing edge survives the copy into the updated graph.
    updated = final["updated_graph"]
    assert updated.has_edge("Q142", "Q90"), "pre-existing edge lost in updated graph"
    assert "has_capital" in updated.edges["Q142", "Q90"]["relation"]


async def test_memory_update_empty_input_routes_to_finalize(live_wikidata):
    """Fully empty input must take the ``finalize_memory`` branch without error.

    Exercises every empty-guard (consolidate_pre/open_ie/link_entities/
    merge_and_prune early returns) and the ``route_after_textualize`` →
    ``finalize_memory`` edge that the happy-path tests never hit.
    """
    from langgraph_coe.graphs import memory_update as mem_mod

    registry = _build_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    final = await graph.ainvoke(
        {
            "question": "Anything?",
            "new_text_items": [],
            "new_raw_triples": [],
            "current_text_memory": [],
            "current_graph": nx.DiGraph(),
            "entity_dict": {},
        }
    )
    print(final)

    assert final["updated_text_memory"] == [], "empty input should yield empty memory"
    assert not final.get("kept_triples"), "no triples should be kept on empty input"
    assert final["updated_graph"].number_of_edges() == 0
    assert final["linked_entities"] == {}


async def test_memory_update_pruner_batching_across_chunks(live_wikidata):
    """Feed > prune_batch_size new triples to exercise multi-chunk pruning.

    Catches off-by-one / misalignment bugs in the
    ``zip(chunk_relations, outputs)`` + per-batch ``keep_indices`` mapping in
    ``merge_and_prune`` (default batch size is 16, so 18 triples → 2 chunks).
    """
    from langgraph_coe.graphs import memory_update as mem_mod
    from langgraph_coe.roles import Relation

    registry = _build_registry()
    graph = mem_mod.build_memory_update_graph(registry)

    # 18 distinct, surface-form triples (no QIDs → no linking needed).
    raw_triples = [
        Relation(
            subject=f"Entity{i}",
            relation="related_to",
            object=f"Topic{i}",
            context=f"Entity{i} is related to Topic{i}.",
        )
        for i in range(18)
    ]
    valid_strings = {f"Entity{i} — related_to — Topic{i}" for i in range(18)}

    final = await graph.ainvoke(
        {
            "question": "Summarise the relationships between the listed entities and topics.",
            "new_text_items": [],
            "new_raw_triples": raw_triples,
            "current_text_memory": [],
            "current_graph": nx.DiGraph(),
            "entity_dict": {},
        }
    )
    print(final["kept_triples"])

    kept = final["kept_triples"]
    assert isinstance(kept, list)
    # No chunk-misalignment: every kept triple is one of the inputs (not a
    # cross-batch index that resolved to the wrong relation).
    assert set(kept).issubset(valid_strings), (
        f"kept triple not among inputs — batch/keep_indices misalignment: "
        f"{set(kept) - valid_strings!r}"
    )
    assert len(kept) <= 18
    assert final["updated_graph"].number_of_edges() == len(set(kept))
