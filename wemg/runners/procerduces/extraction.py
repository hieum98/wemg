import logging
import os
from typing import List

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))



def extract_entities_from_text(
        llm_agent: BaseLLMAgent,
        text: str,
        ) -> List[roles.open_ie.Entity]:
    # NER Extraction
    logger.info("Starting NER extraction from text.")
    ner_role = roles.open_ie.NERRole()
    ner_messages = ner_role.format_messages(input_data=roles.open_ie.NERInput(text=text))
    ner_kwargs = {
        'output_schema': roles.open_ie.NEROutput,
        'n': 1, # only one response
    }
    ner_response = llm_agent.generator_role_execute(ner_messages, **ner_kwargs)[0][0]
    ner_output: roles.open_ie.NEROutput = ner_role.parse_response(ner_response)
    entities = ner_output.entities
    entities = list(set(entities))  # deduplicate entities
    logger.info(f"Extracted entities: {entities}")
    return entities


def extract_relations_from_text(
        llm_agent: BaseLLMAgent,
        text: str,
        entities: List[roles.open_ie.Entity] = None,
        ) -> List[roles.open_ie.Relation]:
    # OpenIE Relation Extraction
    logger.info("Starting relation extraction from text.")
    relation_extraction_role = roles.open_ie.RelationExtractionRole()
    input_data=roles.open_ie.RelationExtractionInput(text=text, entities=[ent.name for ent in entities] if entities else None)
    messages = relation_extraction_role.format_messages(input_data=input_data)
    open_ie_kwargs = {
        'output_schema': roles.open_ie.RelationExtractionOutput,
        'n': 1, # only one response
    }
    response = llm_agent.generator_role_execute(messages, **open_ie_kwargs)[0][0]
    output: roles.open_ie.RelationExtractionOutput = relation_extraction_role.parse_response(response)
    relations = output.relations
    relations = list(set(relations))  # deduplicate relations
    logger.info(f"Extracted relations: {relations}")
    return relations
