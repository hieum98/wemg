"""Real-system tests for NodeGenerator (wemg/reasoning/generator.py)."""

import pytest

from tests.conftest import requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output
from wemg.llm.roles import AnswerGenerationOutput
from wemg.reasoning.generator import GenerationResult, merge_logs
from wemg.retrieval.wikidata import WikidataEntity, WikidataProperty, WikiTriple


def test_merge_logs_empty_and_overlap():
    assert merge_logs() == {}
    assert merge_logs({}, None) == {}
    a = {"R": [("i", "o")]}
    b = {"R": [("i2", "o2")], "S": [("x", "y")]}
    m = merge_logs(a, b)
    assert len(m["R"]) == 2
    assert m["S"] == [("x", "y")]


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_node_generator_generate_answer_no_explore(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    live_node_generator.working_memory.add_textual_memory("France is a country in Europe.")
    result = await live_node_generator.generate_answer(
        "In one short phrase, name a country in Europe.",
        should_explore=False,
    )
    assert isinstance(result, GenerationResult)
    assert result.retrieved_triples == []
    assert result.retrieved_entities == []
    assert isinstance(result.log_data, dict)
    assert result.answers
    assert isinstance(result.answers[0], AnswerGenerationOutput)
    assert result.answers[0].answer


@pytest.mark.requires_llm
@pytest.mark.requires_wikidata
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_retrieve_from_kb(wemg_config, live_node_generator):
    """Live entity linking + Wikidata k-hop triples + triple pruner (no mocks)."""
    requires_llm_credentials(wemg_config)
    triples, entities, log = await live_node_generator._retrieve_from_kb(
        "What is the capital of France?"
    )
    print_slow_integration_output(
        "test_node_generator_retrieve_from_kb",
        triples=triples,
        entities=entities,
        log=log,
    )
    assert isinstance(log, dict)
    assert isinstance(triples, list)
    assert isinstance(entities, list)
    for t in triples:
        assert isinstance(t, WikiTriple)
    for e in entities:
        assert isinstance(e, WikidataEntity)


@pytest.mark.requires_llm
@pytest.mark.requires_wikidata
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_explore(wemg_config, live_node_generator):
    """Full _explore path: query gen, web/corpus retrieval, KB triples, entity enrichment."""
    requires_llm_credentials(wemg_config)
    pytest.importorskip("SPARQLWrapper")
    documents, triples, entities, log = await live_node_generator._explore(
        "What is the capital of France?"
    )
    print_slow_integration_output(
        "test_node_generator_explore",
        documents=documents,
        triples=triples,
        entities=entities,
        log=log,
    )
    assert isinstance(log, dict)
    assert isinstance(documents, list)
    assert isinstance(triples, list)
    assert isinstance(entities, list)
    for d in documents:
        assert isinstance(d, str)
    for t in triples:
        assert isinstance(t, WikiTriple)
    for e in entities:
        assert isinstance(e, WikidataEntity)


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_generate_answer_with_explore(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    result = await live_node_generator.generate_answer(
        "What is the capital of France?",
        should_explore=True,
    )
    print_slow_integration_output(
        "test_node_generator_generate_answer_with_explore",
        result=result,
    )
    assert isinstance(result, GenerationResult)
    assert isinstance(result.log_data, dict)
    assert result.answers
    assert isinstance(result.answers[0], AnswerGenerationOutput)


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_node_generator_generate_subquestion(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    unanswerable, should_direct, log = await live_node_generator.generate_subquestion(
        "What factors influence ocean tides?"
    )
    assert isinstance(unanswerable, list)
    assert isinstance(should_direct, bool)
    assert isinstance(log, dict)


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_node_generator_generate_rephrase(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    rephrased, log = await live_node_generator.generate_rephrase("What is photosynthesis?")
    assert isinstance(rephrased, list)
    assert isinstance(log, dict)
    if rephrased:
        assert all(isinstance(s, str) for s in rephrased)


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_update_working_memory(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    berlin = WikidataEntity(qid="Q64", label="Berlin", description="city")
    germany = WikidataEntity(qid="Q183", label="Germany", description="country")
    prop = WikidataProperty(pid="P1376", label="capital of", description=None)
    triple = WikiTriple(subject=berlin, relation=prop, object=germany)
    gr = GenerationResult(
        answers=[],
        information_items=["Berlin is the capital of Germany."],
        retrieved_triples=[triple],
        retrieved_entities=[berlin, germany],
        log_data={},
    )
    live_node_generator.update_working_memory(gr)
    print_slow_integration_output(
        "test_node_generator_update_working_memory",
        working_memory_after=live_node_generator.working_memory,
    )
    assert "Q64" in live_node_generator.working_memory.entity_dict
    assert live_node_generator.working_memory.graph_memory.number_of_nodes() >= 1


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_generate_self_correction(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    result = await live_node_generator.generate_self_correction(
        "What is the capital of Italy?",
        "Milan",
    )
    print_slow_integration_output(
        "test_node_generator_generate_self_correction",
        result=result,
    )
    assert isinstance(result, GenerationResult)
    assert isinstance(result.log_data, dict)
    assert isinstance(result.answers, list)


@pytest.mark.requires_llm
@pytest.mark.integration
@pytest.mark.slow_integration
@pytest.mark.asyncio
async def test_node_generator_generate_synthesis(wemg_config, live_node_generator):
    requires_llm_credentials(wemg_config)
    result = await live_node_generator.generate_synthesis("What causes seasons on Earth?")
    print_slow_integration_output(
        "test_node_generator_generate_synthesis",
        result=result,
    )
    assert isinstance(result, GenerationResult)
    assert isinstance(result.log_data, dict)
    assert isinstance(result.answers, list)
