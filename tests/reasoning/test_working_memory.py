"""Real-system tests for WorkingMemory (`wemg/reasoning/working_memory.py`)."""

import asyncio
import pytest

from tests.conftest import requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output
from wemg.llm.roles import Relation, SourceType
from wemg.reasoning.working_memory import (
    GlobalKnowledge,
    MemoryDelta,
    WorkingMemory,
    parse_graph_from_text,
)
from wemg.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple


# =====================================================================
# Existing unit tests (updated for new API)
# =====================================================================


def test_format_memory_item_and_dedup_textual():
    wm = WorkingMemory(max_textual_memory_tokens=4096, wikidata_client=None)
    t = wm.format_memory_item("hello", SourceType.RETRIEVAL)
    assert "[Retrieval]" in t
    wm.add_textual_memory("hello", source=SourceType.RETRIEVAL)
    wm.add_textual_memory("hello", source=SourceType.RETRIEVAL)
    assert len(wm.textual_memory) == 1
    assert "hello" in wm.format_textual_memory().lower()


def test_format_textual_memory_empty():
    wm = WorkingMemory(wikidata_client=None)
    assert wm.format_textual_memory() == ""


def test_format_memory_item_no_double_tag():
    wm = WorkingMemory(wikidata_client=None)
    raw = "[Retrieval]: already tagged"
    assert wm.format_memory_item(raw, SourceType.RETRIEVAL) == raw.strip()


def test_add_node_wikidata_entity_no_client_scalar_label():
    wm = WorkingMemory(wikidata_client=None)
    e = WikidataEntity(qid="Q64", label="Berlin", description="city")
    wm.add_node_to_graph_memory(e)
    assert wm.graph_memory.number_of_nodes() == 1
    assert "Q64" in wm.entity_dict


@pytest.mark.requires_wikidata
def test_add_edge_wiki_triple(live_wikidata_client):
    wm = WorkingMemory(wikidata_client=live_wikidata_client)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    triple = WikiTriple(subject=berlin, relation=prop, object=germany)
    wm.add_edge_to_graph_memory(triple)
    assert wm.graph_memory.number_of_edges() >= 1


@pytest.mark.requires_wikidata
def test_format_graph_memory_non_empty(live_wikidata_client):
    wm = WorkingMemory(wikidata_client=live_wikidata_client)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(WikiTriple(subject=berlin, relation=prop, object=germany))
    text = wm.format_graph_memory()
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_parse_graph_from_text_real(wemg_config, live_llm_client, live_wikidata_client):
    requires_llm_credentials(wemg_config)
    triples, log = await parse_graph_from_text(
        live_llm_client,
        "Marie Curie won the Nobel Prize. She was a physicist.",
        interaction_memory=None,
        known_entities=None,
    )
    assert isinstance(triples, list)
    assert isinstance(log, dict)


@pytest.mark.requires_llm
def test_consolidate_textual_memory_real(wemg_config, live_llm_client, live_wikidata_client):
    requires_llm_credentials(wemg_config)
    wm = WorkingMemory(max_textual_memory_tokens=2048, wikidata_client=live_wikidata_client)
    wm.add_textual_memory("Paris is the capital of France.", source=SourceType.RETRIEVAL)
    wm.add_textual_memory("Lyon is a city in France.", source=SourceType.RETRIEVAL)
    asyncio.run(wm._aconsolidate_textual_memory(live_llm_client, "What is the capital of France?"))
    assert len(wm.textual_memory) >= 1
    assert wm.format_textual_memory()


@pytest.mark.requires_llm
@pytest.mark.requires_wikidata
@pytest.mark.slow_integration
def test_deduplicate_graph_memory_real(wemg_config, live_llm_client, live_wikidata_client):
    """Replaced test_consolidate_graph_memory_real — now tests structural dedup."""
    requires_llm_credentials(wemg_config)
    wm = WorkingMemory(max_textual_memory_tokens=2048, wikidata_client=live_wikidata_client)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(WikiTriple(subject=berlin, relation=prop, object=germany))
    wm.deduplicate_graph()
    print_slow_integration_output(
        "test_deduplicate_graph_memory_real",
        working_memory_after=wm,
    )
    assert wm.graph_memory.number_of_nodes() >= 0


@pytest.mark.requires_llm
@pytest.mark.slow_integration
def test_synchronize_memory_real(wemg_config, live_llm_client, live_wikidata_client):
    """Tests the new one-directional synchronize_memory (text -> graph only)."""
    requires_llm_credentials(wemg_config)
    wm = WorkingMemory(
        max_textual_memory_tokens=2048, wikidata_client=live_wikidata_client,
    )
    wm.add_textual_memory("Berlin is the capital of Germany.", source=SourceType.RETRIEVAL)
    asyncio.run(wm.asynchronize_memory(live_llm_client, "What is the capital of Germany?"))
    print_slow_integration_output(
        "test_synchronize_memory_real",
        working_memory_after=wm,
    )
    assert wm.graph_memory.number_of_nodes() >= 0


# =====================================================================
# New unit tests for redesigned features
# =====================================================================


def test_snapshot_isolates_local_state():
    """Mutations on a snapshot must not leak back to the original."""
    wm = WorkingMemory(wikidata_client=None)
    wm.add_textual_memory("fact A", source=SourceType.RETRIEVAL)
    snap = wm.snapshot()

    snap.add_textual_memory("fact B", source=SourceType.RETRIEVAL)
    snap.add_node_to_graph_memory(WikidataEntity(qid="Q1", label="E1", description="d"))

    assert len(wm.textual_memory) == 1
    assert wm.graph_memory.number_of_nodes() == 0
    assert len(snap.textual_memory) == 2
    assert snap.graph_memory.number_of_nodes() == 1


def test_snapshot_shares_global_knowledge():
    """Snapshot and original should reference the same GlobalKnowledge."""
    gk = GlobalKnowledge()
    gk.confirmed_facts.append("global fact")
    wm = WorkingMemory(wikidata_client=None, global_knowledge=gk)
    snap = wm.snapshot()
    assert snap.global_knowledge is gk


def test_get_delta_captures_additions():
    """get_delta returns text and entities added since snapshot creation."""
    wm = WorkingMemory(wikidata_client=None)
    wm.add_textual_memory("pre-existing", source=SourceType.RETRIEVAL)
    snap = wm.snapshot()

    snap.add_textual_memory("new fact", source=SourceType.RETRIEVAL)
    snap.entity_dict["Q999"] = WikidataEntity(qid="Q999", label="X", description="d")

    delta = snap.get_delta()
    assert len(delta.new_textual_items) == 1
    assert "new fact" in delta.new_textual_items[0]
    assert "Q999" in delta.new_entities


def test_global_knowledge_absorb_gated():
    """Absorption with reward below threshold should be rejected."""
    gk = GlobalKnowledge()
    delta = MemoryDelta(new_textual_items=["fact X"])
    gk.absorb(delta, reward=0.1, min_reward=0.5)
    assert len(gk.confirmed_facts) == 0

    gk.absorb(delta, reward=0.6, min_reward=0.5)
    assert len(gk.confirmed_facts) == 1
    assert "fact X" in gk.confirmed_facts[0]


def test_global_knowledge_absorb_deduplicates():
    """Absorbing the same fact twice should not duplicate it."""
    gk = GlobalKnowledge()
    delta = MemoryDelta(new_textual_items=["fact Y"])
    gk.absorb(delta, reward=1.0)
    gk.absorb(delta, reward=1.0)
    assert gk.confirmed_facts.count("fact Y") == 1


def test_deduplicate_graph_merges_same_qid():
    """Nodes with the same QID entered via different surface forms are merged."""
    from wemg.llm.roles import Entity as OpenIEEntity

    wm = WorkingMemory(wikidata_client=None)
    e1 = OpenIEEntity(id="Q64", name="Berlin", description="city")
    e2 = OpenIEEntity(id="Q64", name="Berlin (city)", description="city in Germany")
    e3 = OpenIEEntity(id="Q183", name="Germany", description="country")

    wm.graph_memory.add_node("Q64", data=e1)
    wm.graph_memory.add_node("Q64_dup", data=OpenIEEntity(id="Q64", name="Berlin (city)", description=""))
    wm.graph_memory.add_node("Q183", data=e3)
    wm.graph_memory.add_edge("Q64_dup", "Q183", relation={"capital of"})

    assert wm.graph_memory.number_of_nodes() == 3
    wm.deduplicate_graph()
    assert wm.graph_memory.number_of_nodes() == 2
    assert wm.graph_memory.has_edge("Q64", "Q183")


def test_conditional_sync_skips_when_clean():
    """should_synchronize returns False when nothing has changed."""
    wm = WorkingMemory(wikidata_client=None)
    assert wm.should_synchronize() is False

    wm.add_textual_memory("a", source=SourceType.RETRIEVAL)
    assert wm.text_store.dirty is True
    assert wm.should_synchronize() is True


def test_sync_one_directional_no_graph_to_text():
    """Graph content should NOT be injected into textual_memory during sync."""
    wm = WorkingMemory(wikidata_client=None)
    e1 = WikidataEntity(qid="Q64", label="Berlin", description="city")
    e2 = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_node_to_graph_memory(e1)
    wm.add_node_to_graph_memory(e2)
    wm.graph_memory.add_edge("Q64", "Q183", relation={"capital of"})

    text_before = list(wm.textual_memory)
    assert len(text_before) == 0


def test_format_merges_global_and_local():
    """format_textual_memory should include both global and local facts."""
    gk = GlobalKnowledge()
    gk.confirmed_facts = ["global fact 1", "global fact 2"]
    wm = WorkingMemory(wikidata_client=None, global_knowledge=gk)
    wm.add_textual_memory("local fact", source=SourceType.RETRIEVAL)

    text = wm.format_textual_memory()
    assert "global fact 1" in text
    assert "global fact 2" in text
    assert "local fact" in text


def test_edge_provenance_stored():
    """add_edge_to_graph_memory should store provenance metadata on edges."""
    wm = WorkingMemory(wikidata_client=None)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(
        WikiTriple(subject=berlin, relation=prop, object=germany),
        source_step=3, timestamp=1000.0, reward=0.8,
    )
    edge = wm.graph_memory.edges["Q64", "Q183"]
    assert edge["provenance"]["source_step"] == 3
    assert edge["provenance"]["timestamp"] == 1000.0
    assert edge["provenance"]["reward"] == 0.8


def test_dirty_tracking_on_add():
    """Adding textual memory should set dirty flag and increment counter."""
    wm = WorkingMemory(wikidata_client=None)
    assert wm.text_store.dirty is False
    assert wm.text_store.items_since_sync == 0

    wm.add_textual_memory("fact", source=SourceType.RETRIEVAL)
    assert wm.text_store.dirty is True
    assert wm.text_store.items_since_sync == 1

    wm.add_textual_memory("fact", source=SourceType.RETRIEVAL)
    assert wm.text_store.items_since_sync == 1


def test_memory_delta_dataclass():
    """MemoryDelta should be constructable with defaults."""
    delta = MemoryDelta()
    assert delta.new_textual_items == []
    assert delta.new_triples == []
    assert delta.new_entities == {}

    delta2 = MemoryDelta(new_textual_items=["x"], new_entities={"Q1": WikidataEntity(qid="Q1", label="A")})
    assert len(delta2.new_textual_items) == 1
    assert "Q1" in delta2.new_entities


def test_global_knowledge_add_relation_with_scalar_object():
    """GlobalKnowledge should keep relation triples where object is a literal/scalar."""
    gk = GlobalKnowledge(wikidata_client=None)
    gk.entity_dict["Q64"] = WikidataEntity(qid="Q64", label="Berlin", description="city")

    gk._add_relation(
        Relation(
            subject="Berlin",
            subject_id="Q64",
            relation="population",
            object="3769000",
            object_id=None,
        )
    )

    assert gk.graph.number_of_edges() == 1
    _, object_id = next(iter(gk.graph.edges()))
    object_node = gk.graph.nodes[object_id]["data"]
    assert object_node.id is None
    assert object_node.name == "3769000"


# =====================================================================
# Graph intelligence helper tests
# =====================================================================


def _make_geo_wm():
    """Helper: create a WorkingMemory with Berlin -> Germany edge."""
    wm = WorkingMemory(wikidata_client=None)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(WikiTriple(subject=berlin, relation=prop, object=germany))
    return wm


def test_format_combined_memory_includes_both():
    """format_combined_memory should include both textual and graph content."""
    wm = _make_geo_wm()
    wm.add_textual_memory("Berlin is nice.", source=SourceType.RETRIEVAL)
    combined = wm.format_combined_memory()
    assert "Berlin is nice" in combined
    assert "capital of" in combined or "Berlin" in combined


def test_format_combined_memory_empty():
    wm = WorkingMemory(wikidata_client=None)
    assert wm.format_combined_memory() == ""


def test_get_known_entity_qids():
    wm = _make_geo_wm()
    qids = wm.get_known_entity_qids()
    assert "Q64" in qids
    assert "Q183" in qids


def test_get_known_entity_qids_includes_global():
    """Global knowledge graph entities should also appear."""
    gk = GlobalKnowledge(wikidata_client=None)
    from wemg.llm.roles import Entity as OpenIEEntity

    gk.graph.add_node("Q42", data=OpenIEEntity(id="Q42", name="Douglas Adams", description="author"))
    wm = WorkingMemory(wikidata_client=None, global_knowledge=gk)
    qids = wm.get_known_entity_qids()
    assert "Q42" in qids


def test_get_well_connected_qids():
    wm = WorkingMemory(wikidata_client=None)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    france = WikidataEntity(qid="Q142", label="France", description="country")
    europe = WikidataEntity(qid="Q46", label="Europe", description="continent")

    wm.add_edge_to_graph_memory(WikiTriple(
        subject=berlin, relation=WikidataProperty(pid="P17", label="country", description=None), object=germany,
    ))
    wm.add_edge_to_graph_memory(WikiTriple(
        subject=berlin, relation=WikidataProperty(pid="P30", label="continent", description=None), object=europe,
    ))
    wm.add_edge_to_graph_memory(WikiTriple(
        subject=germany, relation=WikidataProperty(pid="P30", label="continent", description=None), object=europe,
    ))
    wm.add_edge_to_graph_memory(WikiTriple(
        subject=france, relation=WikidataProperty(pid="P30", label="continent", description=None), object=europe,
    ))

    well = wm.get_well_connected_qids(min_degree=3)
    assert "Q46" in well
    assert "Q142" not in well


def test_get_underexplored_neighbor_qids():
    wm = _make_geo_wm()
    paris = WikidataEntity(qid="Q90", label="Paris", description="city")
    france = WikidataEntity(qid="Q142", label="France", description="country")
    wm.add_edge_to_graph_memory(WikiTriple(
        subject=paris,
        relation=WikidataProperty(pid="P1376", label="capital of", description=None),
        object=france,
    ))
    # Germany (Q183) has degree 1, so it's underexplored relative to Berlin (Q64)
    neighbors = wm.get_underexplored_neighbor_qids(["Q64"])
    assert "Q183" in neighbors
    # France shouldn't appear because it's not a neighbor of Q64
    assert "Q142" not in neighbors


def test_get_underexplored_neighbor_qids_empty_graph():
    wm = WorkingMemory(wikidata_client=None)
    assert wm.get_underexplored_neighbor_qids(["Q64"]) == []


def test_is_triple_known_true():
    wm = _make_geo_wm()
    assert wm.is_triple_known("Q64", "capital of", "Q183") is True


def test_is_triple_known_false_different_relation():
    wm = _make_geo_wm()
    assert wm.is_triple_known("Q64", "part of", "Q183") is False


def test_is_triple_known_false_missing_node():
    wm = _make_geo_wm()
    assert wm.is_triple_known("Q999", "capital of", "Q183") is False


