"""End-to-end WEMGSystem with real services."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_corpus_paths, requires_llm_credentials
from tests.helpers.slow_integration_debug import print_slow_integration_output

# Single-sentence multi-hop questions (resolve an intermediate entity, then the target).
_MULTIHOP_MCTS_SMOKE_QUESTION = (
    "What is the capital of the nation whose most populous city is Istanbul?"
)
_MULTIHOP_COT_SMOKE_QUESTION = (
    "Who was the first person to walk on Earth's only natural satellite?"
)
_MULTIHOP_BATCH_SMOKE_QUESTION = ["What ocean borders the continent along whose western edge the Andes run?",
"If increasing CO2 raises global mean temperature through the greenhouse effect, why might regional precipitation patterns change unevenly rather than uniformly worldwide?"]



def test_wemg_system_init_from_yaml(wemg_config_path, wemg_config):
    requires_llm_credentials(wemg_config)
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)
    from wemg.system import WEMGSystem

    system = WEMGSystem(config_path=wemg_config_path)
    assert system.cfg.llm.model_name
    system.close()


@pytest.mark.requires_llm
@pytest.mark.slow_integration
@pytest.mark.integration
def test_wemg_system_answer_mcts_smoke(wemg_config_path, wemg_config, smoke_system_overrides):
    requires_llm_credentials(wemg_config)
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)
    from wemg.system import WEMGSystem

    overrides = list(smoke_system_overrides) + [
        "search.strategy=mcts",
        "output.include_reasoning=true",
    ]
    system = WEMGSystem(config_path=wemg_config_path, config_overrides=overrides)
    try:
        result = system.answer(_MULTIHOP_MCTS_SMOKE_QUESTION, question_id="pytest_mcts")
        breakpoint()
        print("\n--- MCTS search tree ---")
        if result.search_tree is not None:
            result.search_tree.print_tree()
        print_slow_integration_output("test_wemg_system_answer_mcts_smoke", result=result)
        assert result.question == _MULTIHOP_MCTS_SMOKE_QUESTION
        assert result.answer or result.concise_answer
        assert result.metadata.get("strategy") == "mcts"
        assert result.working_memory is not None
    finally:
        system.close()


@pytest.mark.requires_llm
@pytest.mark.slow_integration
@pytest.mark.integration
def test_wemg_system_answer_cot_smoke(wemg_config_path, wemg_config, smoke_system_overrides):
    requires_llm_credentials(wemg_config)
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)
    from wemg.system import WEMGSystem

    overrides = list(smoke_system_overrides) + [
        "search.strategy=cot",
        "output.include_reasoning=true",
    ]
    system = WEMGSystem(config_path=wemg_config_path, config_overrides=overrides)
    try:
        result = system.answer(_MULTIHOP_COT_SMOKE_QUESTION, question_id="pytest_cot")
        breakpoint()
        print("\n--- CoT reasoning tree ---")
        if result.search_tree is not None:
            result.search_tree.print_tree()
        print_slow_integration_output("test_wemg_system_answer_cot_smoke", result=result)
        assert result.question == _MULTIHOP_COT_SMOKE_QUESTION
        assert result.answer or result.concise_answer
        assert result.metadata.get("strategy") == "cot"
    finally:
        system.close()


@pytest.mark.requires_llm
@pytest.mark.slow_integration
def test_answer_questions_batch_smoke(wemg_config_path, wemg_config, smoke_system_overrides):
    requires_llm_credentials(wemg_config)
    if wemg_config.retriever.type == "corpus":
        requires_corpus_paths(wemg_config)
    from wemg.system import answer_questions_batch

    overrides = list(smoke_system_overrides) + [
        "search.strategy=cot",
        "output.include_reasoning=true",
    ]
    results = answer_questions_batch(
        _MULTIHOP_BATCH_SMOKE_QUESTION,
        config_path=wemg_config_path,
        config_overrides=overrides,
        max_workers=2,
    )
    assert len(results) == len(_MULTIHOP_BATCH_SMOKE_QUESTION)
    for i, r in enumerate(results):
        print(f"\n--- CoT reasoning tree (batch item {i}) ---")
        if r.search_tree is not None:
            r.search_tree.print_tree()
    print_slow_integration_output(
        "test_answer_questions_batch_smoke",
        results=results,
    )
    assert results[0].question == _MULTIHOP_BATCH_SMOKE_QUESTION[0]
    assert results[0].answer or results[0].concise_answer


# ---------------------------------------------------------------------------
# Freebase subgraph mode — unit tests (no LLM / SPARQL)
# ---------------------------------------------------------------------------

_FREEBASE_CACHE_ENTRY = {
    "id": "fb_test_001",
    "question": "Which team did Rodriguez play for in 2005?",
    "q_entity_qids": ["Q0"],
    "entity_dict": {
        "Q0": {"qid": "Q0", "label": "Alex Rodriguez", "description": "", "is_cvt_record": False},
        "Q1": {"qid": "Q1", "label": "New York Yankees", "description": "", "is_cvt_record": False},
    },
    "property_dict": {
        "P0": {"pid": "P0", "label": "batting stats team", "description": ""},
    },
    "triples": [["Q0", "P0", "Q1"]],
    "embeddings": {},
}


def _make_freebase_system(wemg_config_path, extra_overrides=None):
    """Return an initialised WEMGSystem configured for freebase_subgraph mode."""
    from wemg.system import WEMGSystem

    overrides = [
        "node_generation.kb_source=freebase_subgraph",
        "search.strategy=cot",
        "search.cot.max_depth=1",
        "node_generation.n_subquestions=1",
        "node_generation.top_k_entities=1",
        "memory.interaction_memory.enabled=false",
        "output.show_search_tree=false",
        "output.include_reasoning=false",
        "cache.enabled=false",
    ]
    if extra_overrides:
        overrides += extra_overrides
    return WEMGSystem(config_path=wemg_config_path, config_overrides=overrides)


def test_freebase_system_config_kb_source(wemg_config_path):
    """Config correctly reflects kb_source=freebase_subgraph after override."""
    system = _make_freebase_system(wemg_config_path)
    assert system.cfg.node_generation.kb_source == "freebase_subgraph"
    system.close()


def test_freebase_system_creates_graph_retriever_client(wemg_config_path, monkeypatch):
    """answer() with subgraph_cache_entry creates a GraphRetrieverClient, not WikidataClient."""
    from wemg.retrieval.graph_retriever_client import GraphRetrieverClient
    from wemg.retrieval.wikidata import WikidataClient

    created_clients = []

    original_init = GraphRetrieverClient.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created_clients.append(self)

    monkeypatch.setattr(GraphRetrieverClient, "__init__", tracking_init)

    # Patch answer_with_cot to avoid real LLM call
    import asyncio

    async def fake_cot(*args, **kwargs):
        return "Answer: New York Yankees", [], None

    monkeypatch.setattr("wemg.system.cot_search", fake_cot)

    # Also patch cot_get_answer
    monkeypatch.setattr(
        "wemg.system.cot_get_answer",
        lambda *a, **kw: ("New York Yankees", "New York Yankees"),
    )

    system = _make_freebase_system(wemg_config_path)
    system._initialize()

    try:
        result = system.answer(
            "Which team did Rodriguez play for in 2005?",
            question_id="fb_pytest",
            subgraph_cache_entry=_FREEBASE_CACHE_ENTRY,
        )
        assert created_clients, "GraphRetrieverClient was never instantiated"
        # The created client must have loaded the cache entry
        client = created_clients[-1]
        assert "Q0" in client._entities_by_qid
        assert "Q1" in client._entities_by_qid
    finally:
        system.close()


def test_freebase_system_no_cache_entry_uses_wikidata(wemg_config_path, monkeypatch):
    """When subgraph_cache_entry=None, the normal WikidataClient path is used."""
    from wemg.retrieval.graph_retriever_client import GraphRetrieverClient

    graph_client_calls = []

    original_init = GraphRetrieverClient.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        graph_client_calls.append(True)

    monkeypatch.setattr(GraphRetrieverClient, "__init__", tracking_init)

    async def fake_cot(*args, **kwargs):
        return "Answer: X", [], None

    monkeypatch.setattr("wemg.system.cot_search", fake_cot)
    monkeypatch.setattr(
        "wemg.system.cot_get_answer",
        lambda *a, **kw: ("X", "X"),
    )

    system = _make_freebase_system(wemg_config_path)
    system._initialize()

    try:
        system.answer("Test question", question_id="no_cache", subgraph_cache_entry=None)
        assert not graph_client_calls, "GraphRetrieverClient should NOT be created when cache_entry=None"
    finally:
        system.close()


def test_answer_batch_forwards_cache_entries(wemg_config_path, monkeypatch, tmp_path):
    """answer_batch passes the correct per-index cache entry to each answer() call."""
    received_entries = []

    original_answer = None

    def spy_answer(self, question, question_id=None, golden_answer=None, subgraph_cache_entry=None):
        received_entries.append(subgraph_cache_entry)
        from wemg.system import AnswerResult

        return AnswerResult(question=question, answer="X", concise_answer="X")

    from wemg.system import WEMGSystem

    monkeypatch.setattr(WEMGSystem, "answer", spy_answer)

    system = _make_freebase_system(wemg_config_path)
    entries = [_FREEBASE_CACHE_ENTRY, None]
    system.answer_batch(
        ["Q1?", "Q2?"],
        question_ids=["id1", "id2"],
        subgraph_cache_entries=entries,
        max_workers=1,
    )

    assert len(received_entries) == 2
    assert any(e == _FREEBASE_CACHE_ENTRY for e in received_entries)
    assert any(e is None for e in received_entries)
    system.close()


def test_answer_batch_cache_entries_length_mismatch_raises(wemg_config_path):
    """Providing mismatched subgraph_cache_entries length raises ValueError."""
    system = _make_freebase_system(wemg_config_path)
    system._initialize()
    try:
        with pytest.raises(ValueError, match="subgraph_cache_entries length"):
            system.answer_batch(
                ["Q1?", "Q2?"],
                subgraph_cache_entries=[_FREEBASE_CACHE_ENTRY],  # length 1, not 2
                max_workers=1,
            )
    finally:
        system.close()


def test_freebase_cache_entry_jsonl_round_trip(tmp_path):
    """Cache entries serialise to JSONL and reload cleanly."""
    cache_file = tmp_path / "test_cache.jsonl"
    with open(cache_file, "w") as f:
        f.write(json.dumps(_FREEBASE_CACHE_ENTRY) + "\n")

    with open(cache_file) as f:
        loaded = json.loads(f.readline())

    from wemg.retrieval.graph_retriever_client import GraphRetrieverClient

    client = GraphRetrieverClient()
    seeds = client.load_from_cache(loaded)
    assert seeds == ["Q0"]
    assert client._entities_by_qid["Q1"].label == "New York Yankees"
