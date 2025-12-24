import logging
import os
from typing import List, Optional, Type, Dict, Union
import pydantic

from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.parsing import extract_info_from_text

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class BaseLLMRole:
    def __init__(self, system_prompt: str, input_model: Type[pydantic.BaseModel], output_model: Type[pydantic.BaseModel]):
        self.system_prompt = system_prompt
        self.input_model = input_model
        self.output_model = output_model
        self.role_name = self.__class__.__name__
    
    def format_messages(
            self, 
            input_data: pydantic.BaseModel, 
            interaction_memory: Optional[InteractionMemory] = None,
            ) -> List[Dict[str, str]]:
        """
        Synchronous version - use format_messages_async for concurrent access.
        """
        # Check the formatting of input_data
        assert isinstance(input_data, self.input_model), f"Input data must be of type {self.input_model.__name__}"
        # Check the first message in history should be the system prompt of this role
        system_message = {"role": "system", "content": self.system_prompt}
        if interaction_memory is None:
            history = [system_message]
        else:
            in_context_examples = interaction_memory.get_examples(role=self.role_name, query=str(input_data))
            in_context_examples = sum(in_context_examples, []) # Flatten the list of lists
            if in_context_examples:
                logger.warning("The interaction memory provided examples for this role. Ensure that all examples are compatible with this role.")
                logger.info(f"In-context examples: {'\n-'.join([msg['content'] for msg in in_context_examples])}")
            assert 'system' not in [msg['role'] for msg in in_context_examples], "in_context_examples should not contain a system message."
            history = [system_message] + in_context_examples
        user_message = {"role": "user", "content": str(input_data)}
        return history + [user_message]
    
    async def format_messages_async(
            self, 
            input_data: pydantic.BaseModel, 
            interaction_memory: Optional[InteractionMemory] = None,
            ) -> List[Dict[str, str]]:
        """
        Async version with read lock - allows multiple concurrent format_messages calls.
        Use this when calling from async procedures like answer_question.
        """
        # Check the formatting of input_data
        assert isinstance(input_data, self.input_model), f"Input data must be of type {self.input_model.__name__}"
        # Check the first message in history should be the system prompt of this role
        system_message = {"role": "system", "content": self.system_prompt}
        if interaction_memory is None:
            history = [system_message]
        else:
            # Use async version with read lock for concurrent access
            in_context_examples = await interaction_memory.get_examples_async(role=self.role_name, query=str(input_data))
            in_context_examples = sum(in_context_examples, []) # Flatten the list of lists
            if in_context_examples:
                logger.warning("The interaction memory provided examples for this role. Ensure that all examples are compatible with this role.")
                logger.info(f"In-context examples: {'\n-'.join([msg['content'] for msg in in_context_examples])}")
            assert 'system' not in [msg['role'] for msg in in_context_examples], "in_context_examples should not contain a system message."
            history = [system_message] + in_context_examples
        user_message = {"role": "user", "content": str(input_data)}
        return history + [user_message]
    
    def log_interaction(self, input_data: pydantic.BaseModel, response: str, interaction_memory: InteractionMemory):
        """Log the interaction between the user and the model. Synchronous version."""
        if interaction_memory:
            interaction_memory.log_turn(
                role=self.role_name,
                user_input=str(input_data),
                assistant_output=response
            )
        logger.info(f"Interaction for role {self.role_name}: user_input={input_data}, assistant_output={response}")

    async def log_interaction_async(self, input_data: pydantic.BaseModel, response: str, interaction_memory: InteractionMemory):
        """Log the interaction between the user and the model. Async version with write lock."""
        if interaction_memory:
            await interaction_memory.log_turn_async(
                role=self.role_name,
                user_input=str(input_data),
                assistant_output=response
            )
        logger.info(f"Interaction for role {self.role_name}: user_input={input_data}, assistant_output={response}")

    def parse_response(self, response: Union[str, Dict[str, str], pydantic.BaseModel]) -> Optional[pydantic.BaseModel]:
        """Parse the model response into the output model."""
        if isinstance(response, self.output_model):
            return response
        elif isinstance(response, dict):
            try:
                return self.output_model(**response)
            except:
                logger.error(f"Failed to parse response into {self.output_model.__name__}. Response dict: {response}")
                output = self.output_model(
                    **{key: response.get(key, None) for key in self.output_model.model_fields.keys()}
                )
                return output
        elif isinstance(response, str):
            keys = self.output_model.model_fields.keys()
            value_types = [field.annotation.__name__ for field in self.output_model.model_fields.values()]
            parsed_dict = extract_info_from_text(response, keys, value_types)
            try:
                return self.output_model(**parsed_dict)
            except:
                logger.error(f"Failed to parse response into {self.output_model.__name__}. Parsed dict: {parsed_dict}. Response text: {response}")
                return None

def _create_role(name: str, prompt: str, input_model, output_model, description: str = ""):
    """Factory function to create role classes."""
    class Role(BaseLLMRole):
        def __init__(self):
            super().__init__(prompt, input_model, output_model)
            self.role_name = name
    Role.name = name
    Role.description = description
    Role.__name__ = name
    return Role
