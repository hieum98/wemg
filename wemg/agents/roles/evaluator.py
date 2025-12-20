"""Evaluator roles for answer assessment and synthesis.

This module defines LLM roles for evaluating answers, majority voting,
and synthesizing final answers from multiple candidates.
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

JUDGE_ANSWER_PROMPT = """You are an expert evaluator. Rate from 0.0 to 10.0 how well the system_answer resolves the user_question.

Criteria:
- Helpfulness & Relevance: How does the answer address the user's core need?
- Correctness: Is information accurate (using correct_answer as reference if available else use your own knowledge or your own internet search)? If correct_answer matches system_answer, give 10.0.
"""

MAJORITY_VOTE_PROMPT = """You are an expert at evaluating answers. Given a question and answers, determine the final answer based on majority voting.

Instructions:
1. Analyze the question
2. Identify underlying consensus across responses
3. Synthesize a single, accurate answer based on majority consensus
"""

SYNTHESIZE_FINAL_ANSWER_PROMPT = """You are an expert in argumentative synthesis. Construct a superior answer by analyzing and integrating candidate answers.

Phase I - Deconstruction:
- Break down each candidate into conclusion, premises, reasoning path
- Assess factual accuracy, logical soundness, sufficiency

Phase II - Conflict Resolution:
- Map convergence and divergence points
- Adjudicate conflicts using hierarchy: authoritative sources > logical soundness > majority

Phase III - Synthesis:
- Build new superior reasoning path
- State final answer
- Self-critique and refine
"""


# =============================================================================
# Input/Output Models
# =============================================================================

class AnswerEvaluationInput(pydantic.BaseModel):
    user_question: str = pydantic.Field(..., description="The user's original question.")
    system_answer: str = pydantic.Field(..., description="The system's answer.")
    correct_answer: Optional[str] = pydantic.Field("Not available", description="The known correct answer.")

    def __str__(self):
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class AnswerEvaluationOutput(pydantic.BaseModel):
    rating: float = pydantic.Field(..., ge=0.0, le=10.0, description="Rating from 0.0 to 10.0.")
    reasoning: str = pydantic.Field(..., description="Reasoning behind the rating.")


class MajorityVoteInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question.")
    answers: List[str] = pydantic.Field(..., description="List of candidate answers.")

    def __str__(self):
        answers_str = "\n\n".join(f"{i+1}.\n {a}" for i, a in enumerate(self.answers))
        return f"question:\n{self.question}\n\nanswers:\n{answers_str}"


class MajorityVoteOutput(pydantic.BaseModel):
    final_answer: str = pydantic.Field(..., description="Final answer by majority voting.")
    concise_answer: str = pydantic.Field(..., description="Concise version.")
    reasoning: str = pydantic.Field(..., description="Reasoning behind the final answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$")


class FinalAnswerSynthesisInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    candidate_answers: List[str] = pydantic.Field(..., description="Candidate answers.")

    def __str__(self):
        candidates_str = "\n\n".join(f"{i+1}.\n {c}" for i, c in enumerate(self.candidate_answers))
        return f"question:\n{self.question}\n\ncandidate_answers:\n{candidates_str}"


class FinalAnswerSynthesisOutput(pydantic.BaseModel):
    final_answer: str = pydantic.Field(..., description="Synthesized final answer.")
    concise_answer: str = pydantic.Field(..., description="Concise version.")
    reasoning: str = pydantic.Field(..., description="Integrated reasoning behind the final answer.")
    confidence_level: str = pydantic.Field(..., pattern=r"^(high|medium|low)$")


# =============================================================================
# Role Classes
# =============================================================================

Evaluator = _create_role(
    "evaluator",
    JUDGE_ANSWER_PROMPT,
    AnswerEvaluationInput,
    AnswerEvaluationOutput,
    "Evaluation role for multi-hop question answering."
)

MajorityVoter = _create_role(
    "majority_voter",
    MAJORITY_VOTE_PROMPT,
    MajorityVoteInput,
    MajorityVoteOutput,
    "Majority voting role for multi-hop question answering."
)

FinalAnswerSynthesizer = _create_role(
    "final_answer_synthesizer",
    SYNTHESIZE_FINAL_ANSWER_PROMPT,
    FinalAnswerSynthesisInput,
    FinalAnswerSynthesisOutput,
    "Final answer synthesis role for multi-hop question answering."
)
