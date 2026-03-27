"""End-to-end WEMGSystem with real services."""

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
