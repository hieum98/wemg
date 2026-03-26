"""Real-system tests for NodeGenerator (wemg/reasoning/generator.py)."""

import pytest

from tests.conftest import requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output
from wemg.reasoning import generator as generator_module
from wemg.llm.roles import AnswerGenerationOutput
from wemg.reasoning.generator import GenerationResult, NodeGenerator, merge_logs
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


def _make_triples(n: int) -> list[WikiTriple]:
    rel = WikidataProperty(pid="P1", label="related to", description=None)
    return [
        WikiTriple(subject=f"S{i}", relation=rel, object=f"O{i}")
        for i in range(n)
    ]


def test_stage_a_triple_pruning_delta_and_top_k():
    triples = _make_triples(4)

    class DummyReranker:
        def get_scores(self, query, documents, instruction=None):
            return [0, 1, 2, 3], [0.90, 0.87, 0.84, 0.10]

    ng = NodeGenerator(
        client=None,
        retriever=None,
        wikidata_client=None,
        working_memory=None,
        reranker=DummyReranker(),
        triple_pruning_delta=0.08,
        triple_pruning_top_k=2,
    )

    kept = ng._stage_a_prune_triples("q", triples)
    assert kept == [triples[0], triples[1]]


@pytest.mark.asyncio
async def test_prune_triples_stage_b_runs_on_stage_a_survivors(monkeypatch):
    triples = _make_triples(4)

    class DummyReranker:
        def get_scores(self, query, documents, instruction=None):
            return [0, 1, 2, 3], [0.90, 0.87, 0.84, 0.10]

    ng = NodeGenerator(
        client=None,
        retriever=None,
        wikidata_client=None,
        working_memory=None,
        reranker=DummyReranker(),
        triple_pruning_delta=0.08,
        triple_pruning_top_k=2,
    )

    captured = {}

    async def fake_execute_role(**kwargs):
        captured["inputs"] = kwargs["input_data"]
        out = type("Out", (), {"keep_indices": [0]})()
        return [[out]], {"TRIPLE_PRUNER": [("in", "out")]}

    monkeypatch.setattr(generator_module, "execute_role", fake_execute_role)

    kept, log = await ng._prune_triples("q", triples)
    assert len(captured["inputs"]) == 1
    assert len(captured["inputs"][0].triples) == 2
    assert len(kept) == 1
    assert "TRIPLE_PRUNER" in log


@pytest.mark.asyncio
async def test_prune_triples_without_reranker_uses_all_inputs(monkeypatch):
    triples = _make_triples(3)
    ng = NodeGenerator(
        client=None,
        retriever=None,
        wikidata_client=None,
        working_memory=None,
        reranker=None,
    )

    captured = {}

    async def fake_execute_role(**kwargs):
        captured["inputs"] = kwargs["input_data"]
        out = type("Out", (), {"keep_indices": [0, 1]})()
        return [[out]], {"TRIPLE_PRUNER": [("in", "out")]}

    monkeypatch.setattr(generator_module, "execute_role", fake_execute_role)

    kept, log = await ng._prune_triples("q", triples)
    assert len(captured["inputs"][0].triples) == 3
    assert len(kept) == 2
    assert "TRIPLE_PRUNER" in log
