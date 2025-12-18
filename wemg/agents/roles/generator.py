import logging
import os
from typing import List, Optional, Type
import pydantic

from wemg.agents.roles.base_role import BaseLLMRole

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.

## Core Principle: The generated subquestion must NOT be answerable using the provided context. If a logical subquestion can be answered by the context, it is not a true knowledge gap, and you must look for the next piece of missing information.

## Step-by-Step Instructions:
1.  Analyze the Main Question:  Deconstruct the question to identify its core intent (e.g., factual lookup, comparison, causal link), key entities, and the information required for a complete answer.
2.  Map Context to Requirements:* Systematically check if the provided context contains all the facts, entities, and relationships identified in Step 1.
3.  Decision Point: Assess Answerability: 
   - If YES (Context is Sufficient): The main question can be fully and confidently answered. No subquestion is needed.
   - If NO (Context is Insufficient): The context is missing at least one critical piece of information. Proceed to the next steps.
4.  If the Context is Insufficient, Execute the Following: 
   a. Identify the Core Knowledge Gap: Pinpoint the most immediate and crucial piece of missing information. This is the first thing you would need to look up to start solving the main question.
   b. Formulate the Subquestion: Create a clear, self-contained question that precisely targets this single knowledge gap. The subquestion should be:
      * Atomic: Asks for one fact.
      * Relevant: Its answer is essential for answering the main question.
      * Non-Anaphoric: Understandable without reading the main question or context (e.g., avoid pronouns like "he" or "it").
   c. CRITICAL VALIDATION: Before finalizing, you must verify that your formulated subquestion CANNOT be answered by the provided context. If it can be, you have made an error. You must re-evaluate the knowledge gap and formulate a different subquestion that targets information truly missing from the context.
"""

class SubquestionGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class SubquestionGenerationOutput(pydantic.BaseModel):
    is_answerable: bool = pydantic.Field(..., description="Indicates if the main question can be answered with the provided context.")
    subquestion: Optional[str] = pydantic.Field(None, description="The generated subquestion if the main question is not answerable with the context.")
    reasoning: str = pydantic.Field(..., description="The reasoning process behind the determination of answerability and subquestion generation.")


ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning. 

## Instructions:
1.  Question Analysis:  Carefully read and understand the question. Identify key components and clarify what is being asked.
2.  Context Utilization:   If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3.  Information Gap Resolution:  If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps. Attempt to answer these queries based on your own knowledge or do the research/ web-search yourself.
4.  Answer Formulation:  Synthesize all gathered information to construct a clear, helpfull, well-reasoned answer. If you make any assumptions, clearly state them in your reasoning and and lower down the output's confidence level. If multiple interpretations of the question are possible, address each one separately.
"""

class AnswerGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class AnswerGenerationOutput(pydantic.BaseModel):
    answer: str = pydantic.Field(..., description="The final answer to the question.")
    concise_answer: str = pydantic.Field(..., description="A concise version of the final answer.")
    reasoning: str = pydantic.Field(..., description="The reasoning process behind the answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$", description="The confidence level of the answer (high, medium, low).")


GENERATE_QUERIES_FOR_RETRIEVER = """"You are a highly advanced Reasoning Engine. Your primary function is to deconstruct a user's Input (a question or statement) into a series of precise, self-contained, and essential search queries. The goal is to generate queries that, when answered, provide all the necessary facts to answer/verify the Input.

## Guiding Principles for Queries
1. The Zero-Synthesis Principle (Most Important): You MUST NOT introduce any new information, entities, or concepts that are not explicitly present in the original Input.
2. Fully Self-Contained: The query MUST BE SELF-CONTAINED, meaning it should be understandable and answerable without needing to refer to the original Input, other queries, or any external context. It should not rely on any anaphoric references or ambiguous terms 
3. Atomic: Each query must ask for one single, indivisible fact. Deconstruct questions containing conjunctions ("and", "or") or multiple attributes into separate queries.
4. Essential & Non-Redundant: Every query must be necessary for the final answer, and must seek a unique piece of information not covered by other queries.

## Instructions:
1. Parse the Input: 
    - If the input is a question: Identify its type (e.g., factual, comparative, causal, temporal), key entities, and the required reasoning steps. 
    - If the input is a statement: Deconstruct it into its core, verifiable claims. Identify the key entities and the asserted relationships between them. Note that, a statement can be a declarative sentence, a claim, or a QA pair. If the input is a QA pair, treat it as a statement with an implied question.
2. Generate Strategic Queries: Formulate a list of search queries to resolve the Input. Each query is a building block to reach the final answer that follows the guiding principles above. Do not try to replace them with the "answer" you think they represent. The goal is to provide the user with all the search components they would need to solve the problem from scratch.
3. Ensure Self-Containment: Each query must be understandable and answerable on its own. Check that no query relies on the original Input, other queries, or any external context. 
4. Review for Completeness and Non-Redundancy: Ensure that the set of queries collectively covers all necessary information to answer/verify the Input without any overlap or unnecessary duplication.
"""

class QueryGeneratorInput(pydantic.BaseModel):
    input_text: str = pydantic.Field(..., description="The input text (question or statement) to be deconstructed into queries.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class QueryGeneratorOutput(pydantic.BaseModel):
    queries: List[str] = pydantic.Field(..., description="The list of generated search queries.")


SELF_CORRECT_PROMPT = """You are an expert assistant specializing in rigorous answer verification and question answering. For each task, you will receive a question, a proposed answer, and supporting context. Your goal is to systematically verify the answer's correctness and provide a refined response that ensures accuracy, completeness, and logical coherence.

## Instructions:
1.  Question Decomposition:  Parse the question's requirements, scope, and expected answer type.
2.  Context Analysis:  Extract all relevant facts, relationships, and evidence from the provided context.
3.  Answer Evaluation:  Systematically assess the proposed answer to determine if the answer is:
    -  CORRECT:  Accurate, complete, and well-supported
    -  PARTIAL:  Correct but incomplete or lacking detail
    -  INCORRECT:  Contains factual errors or logical flaws
    -  UNSUPPORTED:  Cannot be verified against available context
4.  Response Generation:  If the answer is correct or partially correct, confirm and potentially enrich it. If it is incorrect or unsupported, provide a refined answer
"""

class SelfCorrectionInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    proposed_answer: str = pydantic.Field(..., description="The proposed answer to be verified.")
    context: Optional[str] = pydantic.Field("Not provided", description="The context for the question.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class SelfCorrectionOutput(pydantic.BaseModel):
    status: str = pydantic.Field(..., pattern=r"^(correct|partial|incorrect|unsupported)$", description="The evaluation status of the proposed answer.")
    refined_answer: str = pydantic.Field(..., description="The refined answer after verification.")
    reasoning: str = pydantic.Field(..., description="The reasoning process behind the evaluation and refinement of the answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$", description="The confidence level of the refined answer (high, medium, low).")


REPHRASE_QUESTION_PROMPT = """You are a Prompt Refiner, an AI expert skilled at transforming unclear or complex questions into precise, answerable queries. Your primary goal is to enhance the clarity and effectiveness of questions while preserving their original intent.

## Guiding Principles:
1. Clarity First: Eliminate ambiguity, jargon, and convoluted phrasing. Use simple, direct language.
2. Preserve Intent: The rephrased question must ask the same thing as the original. Do not add new concepts or alter the core inquiry.Do not try to replace them with the "answer" you think they represent.
3. Enhance for Answerability: Structure the question to be specific and self-contained, guiding a clear path to the answer.

## Instructions:
1. Deconstruct: Identify the key subject, the core action, and any important details or constraints in the original question.
2. Pinpoint Problems: Note any vague terms, confusing sentence structure, or multiple questions combined into one.
3. Rephrase and Refine: Rewrite the question to be clear, concise, and unambiguous.
"""

class QuestionRephraserInput(pydantic.BaseModel):
    original_question: str = pydantic.Field(..., description="The original question to be rephrased.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class QuestionRephraserOutput(pydantic.BaseModel):
    rephrased_question: str = pydantic.Field(..., description="The rephrased, clearer version of the original question.")
    reasoning: str = pydantic.Field(..., description="The reasoning process behind the rephrasing of the question.")


GENERATE_STRUCTURED_QUERIES = """You are a Reasoning Engine that decomposes questions or statements into structured (subject, relation) queries for knowledge graph retrieval. Each query represents a single fact lookup needed to answer or verify the input.

## Core Principles
1. **Relation Constraints**: If relations are provided, ONLY use those exact relations. If not provided, use standard relation names.
2. **Entity Focus**: Prioritize provided entities as subjects. Only introduce new entities if reasonably necessary and explicitly mentioned in the input.
3. **Atomicity**: Each query requests ONE piece of information. No compound relations.
4. **Self-Containment**: Use fully specified entities, not pronouns or context-dependent references.
5. **Completeness**: Generate all queries needed for the answer.
6. **Non-Redundancy**: Each query must seek unique information and not duplicate another query.

## Instructions
1. **Analyze Input**: Identify key entities, target information, and reasoning hops needed
2. **Generate Queries**: For each fact needed, create an atomic (subject, relation) pair respecting any provided constraints
3. **Validate**: Ensure queries use correct relations (if constrained), are self-contained, atomic, complete, and non-redundant
"""

class QueryGraphGeneratorInput(pydantic.BaseModel):
    input_text: str = pydantic.Field(..., description="The input text (question or statement) to be deconstructed into queries.")
    entities: Optional[List[str]] = pydantic.Field(None, description="An optional list of entities to focus on for query generation.")
    relations: Optional[List[str]] = pydantic.Field(None, description="An optional list of relations to use for query generation.")

    def __str__(self):
        return "\n\n".join([f"{key}:\n{value}" for key, value in self.model_dump().items()])

class Query(pydantic.BaseModel):
    subject: str = pydantic.Field(..., description="The subject entity in the query.")
    relation: str = pydantic.Field(..., description="The relation should be queried for the subject.")
    reasoning: Optional[str] = pydantic.Field(None, description="The reasoning behind the inclusion of this query.")

    def __hash__(self):
        return hash((self.subject, self.relation))

class QueryGraphGeneratorOutput(pydantic.BaseModel):
    queries: List[Query] = pydantic.Field(..., description="A list of generated queries.")


class SubquestionGenerator(BaseLLMRole):
    name = "subquestion_generator"
    description = "Subquestion generation role for multi-hop question answering."

    def __init__(
            self, 
            system_prompt: str = GENERATE_SUBQUESTION_PROMPT, 
            input_model: Type[pydantic.BaseModel] = SubquestionGenerationInput, 
            output_model: Type[pydantic.BaseModel] = SubquestionGenerationOutput
            ):
        super().__init__(system_prompt, input_model, output_model)


class AnswerGenerator(BaseLLMRole):
    name = "answer_generator"
    description = "Answer generation role for multi-hop question answering."

    def __init__(
            self, 
            system_prompt: str = ANSWER_PROMPT, 
            input_model: Type[pydantic.BaseModel] = AnswerGenerationInput, 
            output_model: Type[pydantic.BaseModel] = AnswerGenerationOutput
            ):
        super().__init__(system_prompt, input_model, output_model)


class QueryGenerator(BaseLLMRole):
    name = "query_generator"
    description = "Query generation role for information retrieval."

    def __init__(
            self, 
            system_prompt: str = GENERATE_QUERIES_FOR_RETRIEVER, 
            input_model: Type[pydantic.BaseModel] = QueryGeneratorInput, 
            output_model: Type[pydantic.BaseModel] = QueryGeneratorOutput
            ):
        super().__init__(system_prompt, input_model, output_model)


class SelfCorrector(BaseLLMRole):
    name = "self_corrector"
    description = "Self-correction role for answer verification and refinement."

    def __init__(
            self, 
            system_prompt: str = SELF_CORRECT_PROMPT, 
            input_model: Type[pydantic.BaseModel] = SelfCorrectionInput, 
            output_model: Type[pydantic.BaseModel] = SelfCorrectionOutput
            ):
        super().__init__(system_prompt, input_model, output_model)


class QuestionRephraser(BaseLLMRole):
    name = "question_rephraser"
    description = "Question rephrasing role for clarifying user questions."

    def __init__(
            self, 
            system_prompt: str = REPHRASE_QUESTION_PROMPT, 
            input_model: Type[pydantic.BaseModel] = QuestionRephraserInput, 
            output_model: Type[pydantic.BaseModel] = QuestionRephraserOutput
            ):
        super().__init__(system_prompt, input_model, output_model)


class StructuredQueryGenerator(BaseLLMRole):
    name = "structured_query_generator"
    description = "Structured query generation role for knowledge graph retrieval."

    def __init__(
            self, 
            system_prompt: str = GENERATE_STRUCTURED_QUERIES, 
            input_model: Type[pydantic.BaseModel] = QueryGraphGeneratorInput, 
            output_model: Type[pydantic.BaseModel] = QueryGraphGeneratorOutput
            ):
        super().__init__(system_prompt, input_model, output_model)

