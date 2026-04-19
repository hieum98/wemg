"""Live `execute_role` tests using config-backed LLM client."""

import pytest
import pydantic

from tests.conftest import requires_llm_credentials
from wemg.llm.client import LLMClient
from wemg.llm.roles import (
    SUBQUESTION_GENERATOR,
    SubquestionGenerationInput,
    SubquestionGenerationOutput,
    execute_role,
)


def _make_client(wemg_config):
    requires_llm_credentials(wemg_config)
    gen = wemg_config.llm.generation
    return LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        concurrency=wemg_config.llm.concurrency,
        max_retries=wemg_config.llm.max_retries,
        timeout=gen.timeout,
        temperature=gen.temperature,
        n=gen.n,
        top_p=gen.top_p,
        min_p=gen.min_p,
        max_tokens=gen.max_tokens,
        max_input_tokens=gen.max_input_tokens,
        top_k=gen.top_k,
        presence_penalty=gen.presence_penalty,
        repetition_penalty=gen.repetition_penalty,
        enable_thinking=gen.enable_thinking,
        random_seed=gen.random_seed,
        cache_config={"enabled": False},
    )


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_execute_role_single_input_real(wemg_config):
    client = _make_client(wemg_config)
    try:
        inp = SubquestionGenerationInput(
            question="What is the capital of France?",
            context="General knowledge.",
        )
        results, log_data = await execute_role(
            client=client,
            role=SUBQUESTION_GENERATOR,
            input_data=inp,
            n=1,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, SubquestionGenerationOutput) for r in results)
        assert SUBQUESTION_GENERATOR.name in log_data
        entries = log_data[SUBQUESTION_GENERATOR.name]
        assert isinstance(entries, list)
        assert len(entries) == 1
        input_str, raw_output = entries[0]
        assert input_str == str(inp)
        assert isinstance(raw_output, str)
        assert raw_output != ""
    finally:
        client.close()


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_execute_role_batch_input_real(wemg_config):
    client = _make_client(wemg_config)
    try:
        inp1 = SubquestionGenerationInput(
            question="What is the capital of France?",
            context="Geography.",
        )
        inp2 = SubquestionGenerationInput(
            question="What is the capital of Germany?",
            context="Geography.",
        )
        results, log_data = await execute_role(
            client=client,
            role=SUBQUESTION_GENERATOR,
            input_data=[inp1, inp2],
            n=1,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for res in results:
            assert isinstance(res, list)
            if res:
                assert all(isinstance(r, SubquestionGenerationOutput) for r in res)
        assert SUBQUESTION_GENERATOR.name in log_data
        entries = log_data[SUBQUESTION_GENERATOR.name]
        assert len(entries) == 2
        assert entries[0][0] == str(inp1)
        assert entries[1][0] == str(inp2)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_execute_role_rejects_wrong_input_type():
    class OtherInput(pydantic.BaseModel):
        text: str

    wrong_input = OtherInput(text="not a SubquestionGenerationInput")
    with pytest.raises(AssertionError):
        await execute_role(
            client=None,
            role=SUBQUESTION_GENERATOR,
            input_data=wrong_input,
        )


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_execute_role_log_data_non_empty_real(wemg_config):
    client = _make_client(wemg_config)
    try:
        inp = SubquestionGenerationInput(
            question="List sub-questions to understand climate change.",
            context="General science.",
        )
        results, log_data = await execute_role(
            client=client,
            role=SUBQUESTION_GENERATOR,
            input_data=inp,
            n=2,
        )
        assert isinstance(results, list)
        assert SUBQUESTION_GENERATOR.name in log_data
        entries = log_data[SUBQUESTION_GENERATOR.name]
        assert len(entries) == 1
        input_str, raw_output = entries[0]
        assert input_str == str(inp)
        assert isinstance(raw_output, str)
        assert raw_output != ""
    finally:
        client.close()
