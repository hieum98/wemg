import logging
import os
from typing import List, Optional, Type, Dict, Union
import pydantic

from wemg.utils.parsing import extract_info_from_text

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class BaseLLMRole:
    def __init__(self, system_prompt: str, input_model: Type[pydantic.BaseModel], output_model: Type[pydantic.BaseModel]):
        self.system_prompt = system_prompt
        self.input_model = input_model
        self.output_model = output_model
    
    def format_messages(
            self, 
            input_data: pydantic.BaseModel, 
            history: Optional[List[Dict[str, str]]] = None,
            ) -> List[Dict[str, str]]:
        # Check the formatting of input_data
        assert isinstance(input_data, self.input_model), f"Input data must be of type {self.input_model.__name__}"
        # Check the first message in history should be the system prompt of this role
        system_message = {"role": "system", "content": self.system_prompt}
        if history is None:
            history = [system_message]
        else:
            assert history[0]["role"] == "system" and history[0]["content"] == self.system_prompt, \
                f"The first message in history must be the system prompt of {self.__class__.__name__}. Got: {history[0]} but expected: {system_message}"
        user_content = "\n\n".join([f"{key}: {value}" for key, value in input_data.model_dump().items()])
        user_message = {"role": "user", "content": user_content}
        return history + [user_message]

    def parse_response(self, response: Union[str, Dict[str, str], pydantic.BaseModel]) -> Union[pydantic.BaseModel, Dict, str]:
        """Parse the model response into the output model."""
        if isinstance(response, self.output_model):
            return response
        elif isinstance(response, dict):
            return self.output_model(**response)
        elif isinstance(response, str):
            keys = self.output_model.model_fields.keys()
            value_types = [field.annotation.__name__ for field in self.output_model.model_fields.values()]
            parsed_dict = extract_info_from_text(response, keys, value_types)
            try:
                return self.output_model(**parsed_dict)
            except:
                raise ValueError(f"Failed to parse response into {self.output_model.__name__}. Parsed dict: {parsed_dict}. Response text: {response}")


