"""Real-model tests for execute_role in wemg.llm.roles.

Uses a live LLM endpoint configured via environment variables and existing
Role definitions (e.g., SUBQUESTION_GENERATOR). These tests are marked
with @pytest.mark.requires_llm so they can be selectively run.
"""

import os
import asyncio

import pytest

from wemg.llm.client import LLMClient
from wemg.llm.roles import (
    SUBQUESTION_GENERATOR,
    SubquestionGenerationInput,
    SubquestionGenerationOutput,
    execute_role,
)
from conftest import get_llm_env, requires_llm
import pydantic


def _make_client():
    api_key, url, model = get_llm_env()
    if not url:
        pytest.skip("LLM_URL or OPENAI_BASE_URL not set")
    return LLMClient(
        model_name=model,
        url=url,
        api_key=api_key,
        max_retries=1,
        max_tokens=32768,
        temperature=0.0,
    )


@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_execute_role_single_input_real():
    """Single SubquestionGenerationInput with real model."""
    requires_llm()
    client = _make_client()
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
async def test_execute_role_batch_input_real():
    """Two SubquestionGenerationInput instances in a batch."""
    requires_llm()
    client = _make_client()
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
    """execute_role should assert when input type does not match role.input_model."""

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
async def test_execute_role_log_data_non_empty_real():
    """execute_role with n=2 returns non-empty log_data for real model."""
    requires_llm()
    client = _make_client()
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

