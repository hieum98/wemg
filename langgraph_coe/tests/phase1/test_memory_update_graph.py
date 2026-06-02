"""Phase 1 §4 — MemoryUpdateGraph target specs.

The graph lives at ``langgraph_coe.graphs.memory_update`` and decomposes legacy
``synchronize_memory`` into deterministic LangGraph nodes:

  1. ``consolidate_pre``
  2. ``open_ie`` + ``link_entities`` (Wikidata tool)
  3. ``merge_and_prune``
  4. ``textualize_graph``
  5. ``consolidate_post`` or ``finalize_memory`` (when no kept triples)

These tests are target specs: they describe the behavior Phase 1 must reach.
They skip cleanly only while the new graph module itself is absent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import networkx as nx
import pytest

from langgraph_coe import roles as roles_mod


def _import_module():
    """Phase 1 introduces ``langgraph_coe.graphs.memory_update``."""
    try:
        from langgraph_coe.graphs import memory_update as mem_mod  # type: ignore
    except ImportError:
        pytest.skip("Phase 1 §4 introduces langgraph_coe.graphs.memory_update")
    if not hasattr(mem_mod, "build_memory_update_graph"):
        pytest.skip("build_memory_update_graph not implemented yet (§4 target)")
    return mem_mod


def _registry():
    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(LangGraphCoeConfig().llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    return registry


def _memory_out(*items: str) -> roles_mod.MemoryConsolidationOutput:
    return roles_mod.MemoryConsolidationOutput(
        consolidated_memory=[
            roles_mod.MemoryItem(
                content=item,
                provenance=roles_mod.SourceType.SYSTEM_PREDICTION.value,
                hop_depth=None,
            )
            for item in items
        ]
    )


def _open_ie_out(
    *,
    entities: Sequence[str] = (),
    relations: Sequence[roles_mod.Relation] = (),
) -> roles_mod.OpenIEOutput:
    return roles_mod.OpenIEOutput(
        entities=[roles_mod.Entity(id=None, name=name, description=None) for name in entities],
        relations=list(relations),
    )


def _relation(
    subject: str,
    relation: str,
    object_: str,
    *,
    subject_id: str | None = None,
    object_id: str | None = None,
) -> roles_mod.Relation:
    return roles_mod.Relation(
        subject=subject,
        subject_id=subject_id,
        relation=relation,
        object=object_,
        object_id=object_id,
        context=None,
    )


class RoleExecutorSpy:
    """Async stand-in for ``execute_role_lc`` with per-role canned outputs."""

    def __init__(
        self,
        *,
        consolidation_outputs: Sequence[roles_mod.MemoryConsolidationOutput] | None = None,
        relations: Sequence[roles_mod.Relation] | None = None,
        entities: Sequence[str] = (),
        prune_keep_indices: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._consolidation_outputs = list(consolidation_outputs or [_memory_out("memory")])
        self._relations = list(relations or [])
        self._entities = list(entities)
        self._prune_keep_indices = [list(v) for v in (prune_keep_indices or [])]

    async def __call__(
        self,
        registry: Any,
        role: roles_mod.Role,
        input_data: Any,
        n: int = 1,
        tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        self.calls.append(
            {
                "role": role.name,
                "input": input_data,
                "n": n,
                "tier_override": tier_override,
            }
        )

        if role.name == "memory_consolidation":
            if len(self._consolidation_outputs) > 1:
                return self._consolidation_outputs.pop(0), {}
            return self._consolidation_outputs[0], {}

        if role.name == "open_ie":
            return _open_ie_out(entities=self._entities, relations=self._relations), {}

        if role.name == "triple_pruner":
            inputs = input_data if isinstance(input_data, list) else [input_data]
            outputs = []
            for idx, item in enumerate(inputs):
                if self._prune_keep_indices:
                    keep = self._prune_keep_indices.pop(0)
                else:
                    keep = list(range(len(getattr(item, "triples", []))))
                outputs.append(roles_mod.TriplePruneOutput(keep_indices=keep))
            return (outputs if isinstance(input_data, list) else outputs[0]), {}

        raise AssertionError(f"Unexpected role call: {role.name}")

    def role_inputs(self, role_name: str) -> List[Any]:
        return [c["input"] for c in self.calls if c["role"] == role_name]


class LinkEntitiesSpy:
    """Small tool-like object for the graph's direct ``link_entities`` call."""

    name = "link_entities"

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping
        self.invocations: List[Any] = []

    async def ainvoke(self, inp: Any, *args: Any, **kwargs: Any) -> List[Dict[str, str]]:
        self.invocations.append(inp)
        names = inp.get("entity_names", inp) if isinstance(inp, dict) else inp
        return [
            {
                "name": str(name),
                "qid": self.mapping[str(name)],
                "description": f"{name} entity",
            }
            for name in names
            if str(name) in self.mapping
        ]


def _state(
    *,
    graph: nx.DiGraph | None = None,
    entity_dict: Dict[str, Any] | None = None,
    current_text_memory: Sequence[str] = ("Existing memory item.",),
    new_text_items: Sequence[str] = ("New retrieved item.",),
    new_raw_triples: Sequence[Any] = (),
) -> Dict[str, Any]:
    return {
        "question": "What is the capital of France?",
        "new_text_items": list(new_text_items),
        "new_raw_triples": list(new_raw_triples),
        "current_text_memory": list(current_text_memory),
        "current_graph": graph or nx.DiGraph(),
        "entity_dict": dict(entity_dict or {}),
    }


def _input_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(_input_text(v) for v in value)
    if hasattr(value, "memory"):
        return str(value.memory)
    if hasattr(value, "text"):
        return str(value.text)
    if hasattr(value, "triples"):
        return "\n".join(map(str, value.triples))
    return str(value)


def _pruner_batch_sizes(executor: RoleExecutorSpy) -> List[int]:
    sizes: List[int] = []
    for value in executor.role_inputs("triple_pruner"):
        inputs = value if isinstance(value, list) else [value]
        sizes.extend(len(getattr(item, "triples", [])) for item in inputs)
    return sizes


def _graph_mentions(graph: nx.DiGraph, token: str) -> bool:
    for node, data in graph.nodes(data=True):
        if token in str(node) or token in str(data):
            return True
    for src, dst, data in graph.edges(data=True):
        if token in f"{src} {dst} {data}":
            return True
    return False


def _node_token_count(graph: nx.DiGraph, token: str) -> int:
    return sum(
        1
        for node, data in graph.nodes(data=True)
        if token in str(node) or token in str(data)
    )


def _install_spies(
    monkeypatch: pytest.MonkeyPatch,
    mem_mod: Any,
    executor: RoleExecutorSpy,
    *,
    link_mapping: Dict[str, str] | None = None,
) -> LinkEntitiesSpy:
    link_spy = LinkEntitiesSpy(link_mapping or {})
    monkeypatch.setattr(mem_mod, "execute_role_lc", executor, raising=False)
    monkeypatch.setattr(mem_mod, "link_entities", link_spy, raising=False)
    return link_spy


async def test_graph_compiles_with_expected_state_keys(monkeypatch):
    """``build_memory_update_graph(registry)`` surfaces the §4.1 state fields."""
    mem_mod = _import_module()
    executor = RoleExecutorSpy(
        consolidation_outputs=[
            _memory_out("France has Paris as its capital."),
            _memory_out("France has Paris as its capital after graph merge."),
        ],
        relations=[_relation("France", "capital", "Paris")],
        entities=["France", "Paris"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(_state())

    for key in (
        "consolidated_memory",
        "extracted_relations",
        "linked_entities",
        "updated_text_memory",
        "updated_graph",
        "updated_entity_dict",
    ):
        assert key in final, f"MemoryUpdateState must surface '{key}' (§4.1)"

    assert isinstance(final["updated_text_memory"], list)
    assert isinstance(final["updated_graph"], nx.DiGraph)
    assert isinstance(final["updated_entity_dict"], dict)


async def test_pre_and_post_consolidation_use_memory_consolidation_role(monkeypatch):
    """The graph performs the two consolidation passes specified in §4.2."""
    mem_mod = _import_module()
    executor = RoleExecutorSpy(
        consolidation_outputs=[
            _memory_out("Consolidated pre memory."),
            _memory_out("Consolidated post memory with graph triples."),
        ],
        relations=[_relation("France", "capital", "Paris")],
        entities=["France", "Paris"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(
        _state(
            current_text_memory=["Current memory fact."],
            new_text_items=["New answer fact."],
        )
    )

    consolidation_inputs = executor.role_inputs("memory_consolidation")
    assert len(consolidation_inputs) == 2, (
        "MemoryUpdateGraph must call memory_consolidation before extraction "
        "and again after textualizing graph triples (§4.2)"
    )
    assert "Current memory fact." in _input_text(consolidation_inputs[0])
    assert "New answer fact." in _input_text(consolidation_inputs[0])
    assert "Subject: France | Relation: capital | Object: Paris" in _input_text(
        consolidation_inputs[1]
    )
    assert final["updated_text_memory"] == ["Consolidated post memory with graph triples."]


async def test_skips_post_consolidation_when_no_kept_triples(monkeypatch):
    """When no new triples survive pruning, skip the post-consolidation LLM call."""
    mem_mod = _import_module()
    current_graph = nx.DiGraph()
    current_graph.add_edge("France", "Paris", relation="capital")

    executor = RoleExecutorSpy(
        consolidation_outputs=[
            _memory_out("Consolidated pre only."),
            _memory_out("Should not reach post consolidation."),
        ],
        relations=[_relation("France", "capital", "Paris")],
        entities=["France", "Paris"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(_state(graph=current_graph))

    assert len(executor.role_inputs("memory_consolidation")) == 1
    assert final["updated_text_memory"] == ["Consolidated pre only."]


async def test_open_ie_consumes_consolidated_memory(monkeypatch):
    """OpenIE must start from ``consolidated_memory``, not raw input."""
    mem_mod = _import_module()
    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("Only the consolidated memory should be parsed.")],
        relations=[_relation("France", "capital", "Paris")],
        entities=["France", "Paris"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    await graph.ainvoke(
        _state(
            current_text_memory=["raw current memory"],
            new_text_items=["raw new item"],
        )
    )

    open_ie_text = _input_text(executor.role_inputs("open_ie")[0])
    assert "Only the consolidated memory should be parsed." in open_ie_text
    assert "raw current memory" not in open_ie_text
    assert "raw new item" not in open_ie_text
    assert len(executor.role_inputs("open_ie")) == 1


async def test_link_entities_skips_entities_already_in_entity_dict(monkeypatch):
    """The entity-linking branch should not re-resolve known QIDs (§4.2)."""
    mem_mod = _import_module()
    known_france = roles_mod.WikidataEntity(
        qid="Q142",
        label="France",
        description="country in Western Europe",
    )
    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("France has Paris as its capital."), _memory_out("post")],
        relations=[_relation("France", "capital", "Paris", subject_id="Q142")],
        entities=["France", "Paris"],
    )
    link_spy = _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(_state(entity_dict={"Q142": known_france}))

    assert link_spy.invocations, "link_entities should be called for unresolved names"
    invoked_names = link_spy.invocations[0].get("entity_names", [])
    assert invoked_names == ["Paris"], (
        "link_entities must skip names already represented in entity_dict; "
        f"saw {invoked_names!r}"
    )
    assert "Q142" in final["updated_entity_dict"]
    assert "Q90" in final["updated_entity_dict"]


async def test_merge_prunes_only_new_edges_and_does_not_mutate_input_graph(monkeypatch):
    """``merge_and_prune`` runs the pruner on newly proposed edges only (§4.2)."""
    mem_mod = _import_module()
    current_graph = nx.DiGraph()
    current_graph.add_edge("France", "Paris", relation="capital")

    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("France facts."), _memory_out("post")],
        relations=[
            _relation("France", "capital", "Paris"),
            _relation("France", "contains", "Lyon"),
        ],
        entities=["France", "Paris", "Lyon"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Paris": "Q90", "Lyon": "Q456"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(_state(graph=current_graph))

    pruner_text = "\n".join(_input_text(v) for v in executor.role_inputs("triple_pruner"))
    assert "contains" in pruner_text
    assert "capital" not in pruner_text, (
        "triple_pruner must receive newly proposed edges only, not existing graph edges"
    )
    assert final["updated_graph"] is not current_graph
    assert not _graph_mentions(current_graph, "Lyon"), "input current_graph must not be mutated"
    assert _graph_mentions(final["updated_graph"], "Lyon")


async def test_merge_collapses_duplicate_nodes_by_qid(monkeypatch):
    """Nodes representing the same Wikidata QID collapse to one graph node (§4.2)."""
    mem_mod = _import_module()
    current_graph = nx.DiGraph()
    current_graph.add_node("French Republic", qid="Q142", name="French Republic")

    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("France contains Lyon."), _memory_out("post")],
        relations=[
            _relation("France", "contains", "Lyon", subject_id="Q142", object_id="Q456"),
        ],
        entities=["France", "Lyon"],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"Lyon": "Q456"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(
        _state(
            graph=current_graph,
            entity_dict={
                "Q142": roles_mod.WikidataEntity(qid="Q142", label="France"),
                "Q456": roles_mod.WikidataEntity(qid="Q456", label="Lyon"),
            },
        )
    )

    assert _node_token_count(final["updated_graph"], "Q142") == 1, (
        "merge_and_prune must collapse duplicate graph nodes that share a QID"
    )
    assert _graph_mentions(final["updated_graph"], "Lyon")


async def test_triple_pruner_batches_new_edges_at_sixteen(monkeypatch):
    """Newly proposed edges are pruned in batches of 16 (§4.2)."""
    mem_mod = _import_module()
    relations = [
        _relation(f"Subject {i}", f"relation_{i}", f"Object {i}")
        for i in range(17)
    ]
    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("many relations"), _memory_out("post")],
        relations=relations,
        entities=[f"Subject {i}" for i in range(17)] + [f"Object {i}" for i in range(17)],
    )
    _install_spies(monkeypatch, mem_mod, executor, link_mapping={})

    graph = mem_mod.build_memory_update_graph(_registry())
    await graph.ainvoke(_state())

    assert _pruner_batch_sizes(executor) == [16, 1], (
        "triple_pruner must be called on newly proposed edges in batches of 16"
    )


async def test_pruner_keep_indices_control_graph_and_textualized_edges(monkeypatch):
    """Only triples retained by ``triple_pruner.keep_indices`` enter graph/text memory."""
    mem_mod = _import_module()
    executor = RoleExecutorSpy(
        consolidation_outputs=[_memory_out("candidate facts"), _memory_out("post")],
        relations=[
            _relation("France", "borders", "Spain"),
            _relation("France", "capital", "Paris"),
            _relation("France", "currency", "Euro"),
        ],
        entities=["France", "Spain", "Paris", "Euro"],
        prune_keep_indices=[[1]],
    )
    _install_spies(
        monkeypatch,
        mem_mod,
        executor,
        link_mapping={"France": "Q142", "Spain": "Q29", "Paris": "Q90"},
    )

    graph = mem_mod.build_memory_update_graph(_registry())
    final = await graph.ainvoke(_state())

    post_input = _input_text(executor.role_inputs("memory_consolidation")[1])
    assert "Subject: France | Relation: capital | Object: Paris" in post_input
    assert "borders" not in post_input
    assert "currency" not in post_input
    assert _graph_mentions(final["updated_graph"], "Paris")
    assert not _graph_mentions(final["updated_graph"], "Spain")
    assert not _graph_mentions(final["updated_graph"], "Euro")


def test_memory_config_exposes_textual_memory_token_budget():
    """§4.2 requires ``config.memory.max_textual_memory_tokens`` (default 16384)."""
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    assert hasattr(cfg, "memory"), "Phase 1 must add LangGraphCoeConfig.memory"
    assert cfg.memory.max_textual_memory_tokens == 16_384
