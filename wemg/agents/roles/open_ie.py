"""Open Information Extraction roles for NER and relation extraction.

This module defines LLM roles for extracting named entities and
relationships from text.
"""
import logging
import os
from typing import List, Optional
import pydantic

from wemg.agents.roles.base_role import _create_role

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


# =============================================================================
# Prompts
# =============================================================================

NER_PROMPT = """You are an expert Named Entity Recognition specialist. Extract all named entities from the text.

Instructions:
1. If text is a question, focus ONLY on entities essential to answering it
2. Define precise boundaries (include modifiers like "Prime Minister Boris Johnson")
3. Handle ambiguity using context (e.g., "Apple" as company vs. fruit)
4. Extract unique entities only once

Rules:
- Only extract actual named entities, not common nouns or pronouns
- No overlapping entities; extract most complete version
- For each entity, provide brief description for clarity
"""

RELATION_EXTRACTION_PROMPT = """You are an expert Relation Extraction specialist. Extract all meaningful relationships between entities. Each relationship should be self-contained, i.e., understandable on its own without needing to refer back to the original text or entities. You should prioritize simple relationships over complex relationships.

Instructions:
1. Identify entity pairs with direct relationships
2. Break down complex relationships into simpler relationships.
3. Only extract explicitly stated or strongly implied relationships
4. Use clear, concise relation types
5. Make sure extracted relationships are self-contained and not duplicated.

"""


# =============================================================================
# Input/Output Models
# =============================================================================

class Entity(pydantic.BaseModel):
    name: str = pydantic.Field(..., description="Entity name.")
    description: Optional[str] = pydantic.Field(None, description="Brief description of the entity.")
    is_scalar: bool = pydantic.Field(..., description="Whether entity is scalar (date, quantity, etc.).")

    def __hash__(self):
        return hash(self.name)


class NERInput(pydantic.BaseModel):
    text: str = pydantic.Field(..., description="Text to extract entities from.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class NEROutput(pydantic.BaseModel):
    entities: List[Entity] = pydantic.Field(..., description="Extracted named entities.")


class Relation(pydantic.BaseModel):
    subject: str = pydantic.Field(..., description="Subject entity.")
    relation: str = pydantic.Field(..., description="Relationship type.")
    object: str = pydantic.Field(..., description="Object entity.")
    context: Optional[str] = pydantic.Field(None, description="The context in which the relationship is stated. It should be self-contained and not rely on the original text or entities.")
    
    def __hash__(self):
        return hash((self.subject, self.relation, self.object))
    
    def __str__(self):
        text = f"Subject: {self.subject}\nRelation: {self.relation}\nObject: {self.object}"
        return text + (f"\nContext: {self.context}" if self.context else "")


class RelationExtractionInput(pydantic.BaseModel):
    text: str = pydantic.Field(..., description="Text to extract relationships from.")
    entities: Optional[List[str]] = pydantic.Field(None, description="Entities to focus on.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class RelationExtractionOutput(pydantic.BaseModel):
    relations: List[Relation] = pydantic.Field(..., description="Extracted relationships.")


# =============================================================================
# Role Classes
# =============================================================================

NERRole = _create_role(
    "named_entity_recognition",
    NER_PROMPT,
    NERInput,
    NEROutput,
    "Named entity recognition"
)

RelationExtractionRole = _create_role(
    "relation_extraction",
    RELATION_EXTRACTION_PROMPT,
    RelationExtractionInput,
    RelationExtractionOutput,
    "Relation extraction"
)
