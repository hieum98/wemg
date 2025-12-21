from typing import Dict, Optional, Tuple, Union, List
import pydantic

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.roles.base_role import BaseLLMRole
from wemg.runners.interaction_memory import InteractionMemory


async def execute_role(llm_agent: BaseLLMAgent, role: BaseLLMRole, input_data: Union[pydantic.BaseModel, List[pydantic.BaseModel]], interaction_memory: Optional[InteractionMemory] = None, **kwargs):
    is_single = False
    if isinstance(input_data, pydantic.BaseModel):
        input_data = [input_data]
        is_single = True
    all_messages = []
    for item in input_data:
        expected_type = role.input_model
        assert isinstance(item, expected_type), f"Input data must be of type {expected_type.__name__}"
        messages = await role.format_messages_async(item, interaction_memory=interaction_memory)
        all_messages.append(messages)

    kwargs = {
        'n': kwargs.pop('n', 1),
        'output_schema': role.output_model,
        **kwargs
    }
    response, raw_response = llm_agent.generator_role_execute(all_messages, **kwargs)

    input_data = [str(item) for item in input_data]
    assert len(input_data) == len(raw_response), "input_data and raw_response must have the same length"
    to_log_data: Dict[str, List[Tuple[str, str]]] = {
        role.role_name: [tuple(pair) for pair in zip(input_data, raw_response) if pair[1]]
    }
    parsed_response: List[List[pydantic.BaseModel]] = []
    for res in response:
        r = []
        for item in res:
            parsed_item = role.parse_response(item)
            if parsed_item:
                r.append(parsed_item)
        parsed_response.append(r)
    if is_single:
        return parsed_response[0], to_log_data
    else:
        return parsed_response, to_log_data