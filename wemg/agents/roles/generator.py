"""Generator roles for question answering and reasoning.

This module defines LLM roles for various generation tasks including
subquestion generation, answer generation, query generation, and more.
"""
import logging
import os
from typing import List, Optional
import pydantic

from wemg.agents.roles.base_role import _create_role
from wemg.agents.tools.wikidata import PROPERTY_LABELS

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


# =============================================================================
# Prompts
# =============================================================================

GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.

## Core Principle: The generated subquestion must NOT be answerable using the provided context.

## Instructions:
1. Analyze the Main Question: Identify core intent, key entities, and required information.
2. Map Context to Requirements: Check if context contains all required facts.
3. Decision Point:
   - If YES (Sufficient): No subquestion needed.
   - If NO (Insufficient): Proceed to generate subquestion.
4. If Insufficient:
   a. Identify the Core Knowledge Gap
   b. Formulate an atomic, relevant, self-contained subquestion. Make sure each subquestion is fully understandable on its own without needing to refer back to the original question or context.
   c. VALIDATE: Ensure subquestion CANNOT be answered by context. If it can be answered by context, generate a new subquestion.
"""

ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. Deliver a direct, accurate answer with transparent, step-by-step reasoning.

## Instructions:
1. Analyze the question and identify key components.
2. If context provided, extract all relevant information.
3. Resolve information gaps using your knowledge. You can use your own knowledge or your own internet search.
4. Synthesize a clear, well-reasoned answer. State assumptions clearly if you made any assumptions or used your own internet search.
"""

SELF_CORRECT_PROMPT = """You are an expert in answer verification and refinement. Given a question, proposed answer, and context, verify correctness and provide a refined response.

## Instructions:
1. Parse question requirements
2. Extract relevant facts from context
3. Evaluate proposed answer: Systematically assess the proposed answer to determine if the answer is:
    -  CORRECT:  Accurate, complete, and well-supported
    -  PARTIAL:  Correct but incomplete or lacking detail
    -  INCORRECT:  Contains factual errors or logical flaws
    -  UNSUPPORTED:  Cannot be verified against available context
4. Generate refined answer
"""

REPHRASE_QUESTION_PROMPT = """You are a Question Refiner that transforms unclear questions into precise, clear questions. Make sure the rephrased question is fully understandable on its own without needing to refer back to the original question.

## Principles:
1. Clarity First: Eliminate ambiguity and jargon.
2. Preserve Intent: Don't alter the core inquiry.
3. Enhance Answerability: Make specific and self-contained.

## Instructions:
1. Identify key subject and core action
2. Note vague terms or confusing structure
3. Rewrite to be clear and unambiguous
"""

SYNTHESIZE_PROMPT = """You are a specialized AI for multi-step reasoning. Perform a single, focused reasoning step by analyzing context and producing a consolidated synthesis.

## Instructions:
1. Analyze the main question objective
2. Review all facts in context
3. Determine the Next Logical Step : Based on context and question, decide on the most valuable reasoning action to perform. Your action should be one of the following:
    - Synthesize a Causal or Temporal Link: Connect multiple facts to explain why something happened or to establish a sequence of events.
    - Identify a Core Relationship: Integrate disparate pieces of information to define the relationship between key entities or concepts.
    - Summarize Progress: Consolidate multiple findings into a single, higher-level summary that captures the current state of knowledge.
    - Identify a Contradiction: If the context contains conflicting information, highlight the discrepancy.
    - Formulate a Hypothesis: Propose a plausible conclusion that logically follows from the context but may need further validation in subsequent steps.
    - Assess Sufficiency: If the context provides enough information to directly answer the main question, state this clearly and formulate the definitive answer.
    - Articulate the Conclusion: Generate a single, dense paragraph that clearly states your new conclusion. This thought must be self-contained and understandable without referencing the full context again.

## Critical Constraints:
1. No External Information: Do NOT introduce any facts, assumptions, or information not present in the context.
2. No New questions: Do not ask for new information. Your role is to synthesize, not to query.
"""

GENERATE_QUERIES_PROMPT = """You are a Reasoning Engine that deconstructs user input into precise, self-contained search queries.

## Principles:
1. Self-Contained: Each query understandable without original input.
2. Atomic: One single fact per query.
3. Essential & Non-Redundant: Every query necessary and unique.

## Instructions:
1. Parse the Input: 
    - If the input is a question: Identify its type (e.g., factual, comparative, causal, temporal), key entities, and the required reasoning steps. 
    - If the input is a statement: Deconstruct it into its core, verifiable claims. Identify the key entities and the asserted relationships between them. 
2. Generate Strategic Queries: Formulate a list of search queries that are necessary to answer/verify the input. Each query is a building block to reach the final answer that follows the guiding principles above. The goal is to provide the user with all the search components they would need to solve the problem from scratch.
3. Ensure Self-Containment: Each query must be understandable and answerable on its own. Make sure each query is self-contained and does not rely on the original input, other queries, or any external context. Rewrite queries that are not self-contained.
4. Review for Completeness and Non-Redundancy: Ensure that the set of queries collectively covers all necessary information to answer/verify the input without any overlap or unnecessary duplication.
"""

# Format reference relations from PROPERTY_LABELS for the prompt
_REFERENCE_RELATIONS = "\n".join([
    f"- {prop_data['label']}: {prop_data['description']}"
    for prop_data in sorted(PROPERTY_LABELS.values(), key=lambda x: x['label'])
])

GENERATE_STRUCTURED_QUERIES = f"""You are a Query Generator that decomposes questions into structured (subject, relation) queries for knowledge graph retrieval. Each query represents a single fact lookup needed to answer or verify the input.

## Reference Relations (commonly used in knowledge graphs):
{_REFERENCE_RELATIONS}

## Principles:
0. Analyze the input: Identify the key entities and the required relationships between them.
1. Relation Constraints: Use provided relations only if specified. You also should consider the relations that are not provided but are commonly used in the knowledge graph such as those in the reference relations above (this is just for reference, you can ignore it if you think it is not relevant).
2. Entity Focus: Prioritize provided entities as subjects if specified
3. Atomicity: One piece of information per query. For the query required complex relationship, you can break down the complex relationship into multiple simple relationships to form the query.
4. Self-Containment: Fully specified entities, no pronouns
5. Completeness: All queries needed for the answer
6. Non-Redundancy: Each query seeks unique information that provides clues to answer the question, DO NOT generate redundant or unhelpful queries that are not required to answer the question.

"""


# =============================================================================
# Input/Output Models
# =============================================================================

class SubquestionGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class SubquestionGenerationOutput(pydantic.BaseModel):
    is_answerable: bool = pydantic.Field(..., description="If the main question can be answered with context.")
    subquestion: Optional[str] = pydantic.Field(None, description="Generated subquestion if not answerable.")


class AnswerGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class AnswerGenerationOutput(pydantic.BaseModel):
    answer: str = pydantic.Field(..., description="The final answer.")
    concise_answer: str = pydantic.Field(..., description="A concise version of the answer.")
    reasoning: str = pydantic.Field(..., description="The reasoning process behind the final answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$", description="Confidence level.")


class QueryGeneratorInput(pydantic.BaseModel):
    input_text: str = pydantic.Field(..., description="Input text to deconstruct into queries.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class QueryGeneratorOutput(pydantic.BaseModel):
    queries: List[str] = pydantic.Field(..., description="Generated search queries.")


class SelfCorrectionInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    proposed_answer: str = pydantic.Field(..., description="The proposed answer to verify.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class SelfCorrectionOutput(pydantic.BaseModel):
    status: str = pydantic.Field(..., pattern=r"^(correct|partial|incorrect|unsupported)$")
    refined_answer: str = pydantic.Field(..., description="The refined answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$")


class QuestionRephraserInput(pydantic.BaseModel):
    context: Optional[str] = pydantic.Field(None, description="Context for rephrasing.")
    original_question: str = pydantic.Field(..., description="The original question.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class QuestionRephraserOutput(pydantic.BaseModel):
    rephrased_question: str = pydantic.Field(..., description="The rephrased question.")


class ReasoningSynthesizeInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The main question.")
    context: str = pydantic.Field(..., description="Facts or previous reasoning steps.")

    def __str__(self):
        return f"Question:\n{self.question}\n\nContext:\n{self.context}"


class ReasoningSynthesizeOutput(pydantic.BaseModel):
    is_answerable: bool = pydantic.Field(..., description="If question can be answered with context.")
    step_conclusion: str = pydantic.Field(..., description="Synthesized conclusion.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$")


class Query(pydantic.BaseModel):
    subject: str = pydantic.Field(..., description="Subject entity in the query.")
    relation: str = pydantic.Field(..., description="Relation to query.")
    reasoning: Optional[str] = pydantic.Field(None, description="Reasoning for inclusion.")

    def __hash__(self):
        return hash((self.subject, self.relation))


class QueryGraphGeneratorInput(pydantic.BaseModel):
    input_text: str = pydantic.Field(..., description="Input to deconstruct.")
    entities: Optional[List[str]] = pydantic.Field(None, description="Entities to focus on.")
    relations: Optional[List[str]] = pydantic.Field(None, description="Relations to use.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class QueryGraphGeneratorOutput(pydantic.BaseModel):
    queries: List[Query] = pydantic.Field(..., description="Generated structured queries.")


# =============================================================================
# Role Classes
# =============================================================================


# Create role classes using factory
SubquestionGenerator = _create_role(
    "subquestion_generator",
    GENERATE_SUBQUESTION_PROMPT,
    SubquestionGenerationInput,
    SubquestionGenerationOutput,
    "Subquestion generation for multi-hop QA"
)

AnswerGenerator = _create_role(
    "answer_generator", 
    ANSWER_PROMPT,
    AnswerGenerationInput,
    AnswerGenerationOutput,
    "Answer generation for multi-hop QA"
)

QueryGenerator = _create_role(
    "query_generator",
    GENERATE_QUERIES_PROMPT,
    QueryGeneratorInput,
    QueryGeneratorOutput,
    "Query generation for information retrieval"
)

SelfCorrector = _create_role(
    "self_corrector",
    SELF_CORRECT_PROMPT,
    SelfCorrectionInput,
    SelfCorrectionOutput,
    "Answer verification and refinement"
)

QuestionRephraser = _create_role(
    "question_rephraser",
    REPHRASE_QUESTION_PROMPT,
    QuestionRephraserInput,
    QuestionRephraserOutput,
    "Question clarification"
)

ReasoningSynthesizer = _create_role(
    "reasoning_synthesize",
    SYNTHESIZE_PROMPT,
    ReasoningSynthesizeInput,
    ReasoningSynthesizeOutput,
    "Multi-step reasoning synthesis"
)

StructuredQueryGenerator = _create_role(
    "structured_query_generator",
    GENERATE_STRUCTURED_QUERIES,
    QueryGraphGeneratorInput,
    QueryGraphGeneratorOutput,
    "Structured query generation for KG retrieval"
)
