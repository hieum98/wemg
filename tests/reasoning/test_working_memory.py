import pytest

from wemg.reasoning.working_memory import WorkingMemory


@pytest.mark.asyncio
async def test_arun_consolidation_fallback_uses_schema_valid_provenance(monkeypatch):
    from wemg.llm import roles as roles_module

    async def fake_execute_role(*args, **kwargs):
        # Force fallback path in WorkingMemory._arun_consolidation
        return [], {}

    monkeypatch.setattr(roles_module, "execute_role", fake_execute_role)

    wm = WorkingMemory()
    output, _ = await wm._arun_consolidation(
        client=object(),
        question="Who is older?",
        raw_memory="- [System Prediction]: Sample item",
    )

    assert output.consolidated_memory
    assert output.consolidated_memory[0].provenance == "System Prediction"
