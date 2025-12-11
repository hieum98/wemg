import logging
import os
from typing import List, Optional, Type
import pydantic

from wemg.agents.roles.base_role import BaseLLMRole


NER_EXTRACTION_PROMPT = """You are an expert Named Entity Recognition (NER) specialist. Your task is to identify and extract all named entities from the provided text and ensure completeness and consistency.

## Instructions:
1. **Determine if the text is a question**: If the text is a question, focus ONLY on extracting entities that are essential to answering or understanding the question. Skip entities that are not directly relevant to the question's intent.
2. **For non-question text**: Read the text and identify all named entities.
3. Define precise boundaries: include modifiers that are part of the name (e.g., "Prime Minister Boris Johnson"), exclude articles unless part of official names (e.g., "The Beatles")
4. Handle ambiguity using context (e.g., "Apple" as organization vs. fruit, "Washington" as person vs. location")
5. Extract unique entities only once, if they appear multiple times in the text, i.e, if an entity is mentioned more than once, extract it only one time.
6. Do NOT extract pronouns or common descriptive phrases

## Key Rules:
- Only extract actual named entities, not common nouns or pronouns
- **For questions**: Prioritize entities that are central to answering the question
- Ensure no overlap between entities; if one entity is part of another, extract only the most complete version
- DO NOT include nested entities when meaningful (e.g., "New York" within "New York City")
- For ambiguous cases, provide brief justification
- For each entity, provide short description or context for clarity
"""

class NERInput(pydantic.BaseModel):
    text: str = pydantic.Field(..., description="The input text from which to extract named entities.")

class Entity(pydantic.BaseModel):
    name: str = pydantic.Field(..., description="The name of the extracted entity.")
    description: Optional[str] = pydantic.Field(None, description="A brief description or context for the entity.")
    is_scalar: bool = pydantic.Field(..., description="Indicates whether the entity is scalar (e.g., date, quantity) or not.")

class NEROutput(pydantic.BaseModel):
    entities: List[Entity] = pydantic.Field(..., description="A list of extracted named entities.")


RELATION_EXTRACTION_PROMPT = """You are an expert Relation Extraction specialist. Your task is to identify and extract all meaningful relationships between named entities in the provided text, ensuring accuracy and completeness.

## Instructions:
1. Identify all entity pairs that have a direct relationship expressed in the text
2. For each relationship, extract:
   - Subject entity (the entity performing the action or being described)
   - Relation type (the nature of the relationship)
   - Object entity (the entity being acted upon or related to)
3. Only extract relationships explicitly stated or strongly implied in the text
4. If an entity list is provided, focus on relationships involving those entities, but include other relevant entities if they form important relationships
5. Use clear, concise relation types that accurately reflect the relationship described in the text

## Key Rules:
- Extract only factual relationships explicitly mentioned in the text
- Avoid inferring relationships not directly stated
- Use consistent relation naming (lowercase with underscores)
- Each relationship should involve exactly two entities (subject and object)
- If the same relationship is mentioned multiple times, extract it only once
- Provide brief supporting evidence for each extracted relationship
"""

class Relation(pydantic.BaseModel):
    subject: str = pydantic.Field(..., description="The subject entity in the relationship.")
    relation: str = pydantic.Field(..., description="The type of relationship between the subject and object.")
    object: str = pydantic.Field(..., description="The object entity in the relationship.")
    evidence: Optional[str] = pydantic.Field(None, description="Brief supporting evidence from the text for the relationship.")

class RelationExtractionInput(pydantic.BaseModel):
    text: str = pydantic.Field(..., description="The input text from which to extract relationships.")
    entities: Optional[List[str]] = pydantic.Field(None, description="An optional list of entities to focus on for relationship extraction.")

class RelationExtractionOutput(pydantic.BaseModel):
    relations: List[Relation] = pydantic.Field(..., description="A list of extracted relationships between entities.")


class NERRole(BaseLLMRole):
    name: str = "named_entity_recognition"
    description: str = "Role for extracting named entities from text."
    def __init__(
            self, 
            system_prompt = NER_EXTRACTION_PROMPT, 
            input_model = NERInput, 
            output_model = NEROutput
        ):
        super().__init__(system_prompt, input_model, output_model)

class RelationExtractionRole(BaseLLMRole):
    name: str = "relation_extraction"
    description: str = "Role for extracting relationships between named entities from text."
    def __init__(
            self, 
            system_prompt = RELATION_EXTRACTION_PROMPT, 
            input_model = RelationExtractionInput, 
            output_model = RelationExtractionOutput
        ):
        super().__init__(system_prompt, input_model, output_model)
