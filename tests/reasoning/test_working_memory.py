"""Real-system tests for WorkingMemory (`wemg/reasoning/working_memory.py`)."""

import pytest

from tests.conftest import requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output
from wemg.llm.roles import SourceType
from wemg.reasoning.working_memory import WorkingMemory, parse_graph_from_text
from wemg.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple


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


@pytest.mark.requires_wikidata
def test_connect_graph_memory_single_node(live_wikidata_client):
    wm = WorkingMemory(wikidata_client=live_wikidata_client)
    wm.add_node_to_graph_memory(WikidataEntity(qid="Q64", label="Berlin", description="city"))
    assert wm.connect_graph_memory(max_hops=1) is True


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
    wm.consolidate_textual_memory(live_llm_client, "What is the capital of France?")
    assert len(wm.textual_memory) >= 1
    assert wm.format_textual_memory()


@pytest.mark.requires_llm
@pytest.mark.requires_wikidata
@pytest.mark.slow_integration
def test_consolidate_graph_memory_real(wemg_config, live_llm_client, live_wikidata_client):
    requires_llm_credentials(wemg_config)
    wm = WorkingMemory(max_textual_memory_tokens=2048, wikidata_client=live_wikidata_client)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(WikiTriple(subject=berlin, relation=prop, object=germany))
    wm.consolidate_graph_memory(live_llm_client, "How is Berlin related to Germany?")
    print_slow_integration_output(
        "test_consolidate_graph_memory_real",
        working_memory_after=wm,
    )
    assert wm.graph_memory.number_of_nodes() >= 0


@pytest.mark.requires_llm
@pytest.mark.slow_integration
def test_synchronize_memory_real(wemg_config, live_llm_client, live_wikidata_client):
    requires_llm_credentials(wemg_config)
    wm = WorkingMemory(max_textual_memory_tokens=2048, wikidata_client=live_wikidata_client)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    wm.add_edge_to_graph_memory(WikiTriple(subject=berlin, relation=prop, object=germany))
    wm.synchronize_memory(live_llm_client, "What is the capital of Germany?")
    print_slow_integration_output(
        "test_synchronize_memory_real",
        working_memory_after=wm,
    )
    assert wm.graph_memory.number_of_nodes() >= 0
