"""Role definitions for langgraph_coe.

Each ``Role`` bundles a name, system prompt, Pydantic input model, and Pydantic
output model. ``langgraph_coe/llm.py:RoleModelRegistry`` maps the role's name
to a tier (``heavy`` / ``medium`` / ``light``); see ``config.py:LLMConfig``.

Active callers today live in Phase 1–3 subgraphs. A handful of roles
(``QUESTION_REPHRASER``, ``REASONING_SYNTHESIZER``, ``EVALUATOR``,
``EXTRACTOR``) are kept without a current caller — their prompts are tuned
and they fit planned use cases (alternative MCTS expansion strategies,
offline answer evaluation harness, standalone information extraction). Roles
fully dropped from the port (``QUERY_GENERATOR``, ``MAJORITY_VOTER``,
``CONSENSUS_EVALUATOR``) are listed in ``implementation_plan.md`` §3.6.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type

import pydantic


class WikidataEntity(pydantic.BaseModel):
    qid: str
    label: str
    description: str = ""


logger = logging.getLogger(__name__)


# =============================================================================
# Role dataclass
# =============================================================================


@dataclass
class Role:
    name: str
    system_prompt: str
    input_model: Type[pydantic.BaseModel]
    output_model: Type[pydantic.BaseModel]


class _KeyedFieldsInput(pydantic.BaseModel):
    """Base for inputs whose prompt body is one ``key:\\n value`` block per field.

    Renders every field in declaration order, blank-line separated — the default
    shape several role prompts expect. Subclasses needing a custom layout (e.g.
    context-first ordering for prefix-cache reuse) simply override ``__str__``.
    """

    def __str__(self) -> str:
        return "\n\n".join(f"{k}:\n{v}" for k, v in self.model_dump().items())


class _EntityLinkingInput(pydantic.BaseModel):
    """Base for OpenIE-style inputs: free ``text`` plus optional ``known_entities``.

    Renders the known-entity reference block (when present) above the text — the
    shape the NER / relation-extraction / open-IE prompts are tuned for. Each
    subclass declares its own ``text`` field so it can document what to extract.
    """

    known_entities: Optional[List["WikidataEntity"]] = pydantic.Field(
        None,
        description="Optional reference list of known entities with Wikidata QIDs. Use only when certain; never guess IDs.",
    )

    def __str__(self) -> str:
        if not self.known_entities:
            return f"Text:\n{self.text}"
        known = "\n".join(
            f"- {e.qid}:{e.label}, {e.description}"
            for e in self.known_entities
            if e.qid and e.label
        )
        return f"Known entities:\n{known}\n---\nText:\n{self.text}"


# =============================================================================
# Generator I/O Models
# =============================================================================


class SubquestionGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field(
        "Not provided", description="The context for the question."
    )
    intermediate_answer: Optional[str] = pydantic.Field(
        None,
        description="Result of the previous reasoning hop to anchor the next subquestion.",
    )

    def __str__(self):
        parts = [f"question:\n{self.question}", f"context:\n{self.context}"]
        if self.intermediate_answer:
            parts.append(f"intermediate_answer:\n{self.intermediate_answer}")
        return "\n\n".join(parts)


class SubquestionGenerationOutput(pydantic.BaseModel):
    is_answerable: bool = pydantic.Field(
        ..., description="If the main question can be answered with context."
    )
    subquestions: Optional[List[str]] = pydantic.Field(
        None,
        description="Generated subquestions which are atomic, self-contained and diverse.",
    )
    needs_kg: Optional[List[bool]] = pydantic.Field(
        None,
        description=(
            "Parallel array to `subquestions` (same length and order). "
            "True if subquestions[i] targets a specific named entity's relation or "
            "attribute and should additionally query the knowledge graph; False if it "
            "is best served by document retrieval alone (no named entity, or a "
            "definitional / conceptual gap)."
        ),
    )


class AnswerGenerationInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    context: Optional[str] = pydantic.Field(
        "Not provided", description="The context for the question."
    )

    def __str__(self):
        # Context-first ordering: in every batched call site (CoT ``gen_subanswers``,
        # MCTS ``_gen_subqa``) the same ``context`` is reused across all S subquestions
        # while ``question`` varies. Emitting the shared context block first lets
        # SGLang RadixAttention serve it from the prefix cache on S-1 of the S calls
        # (§3a). Reordering is prompt-shape only; the model sees the same fields.
        return f"context:\n{self.context}\n\nquestion:\n{self.question}"


class AnswerGenerationOutput(pydantic.BaseModel):
    answer: str = pydantic.Field(..., description="The final answer.")
    concise_answer: str = pydantic.Field(
        ...,
        description="A concise version of the answer, only include the answer for the question.",
    )
    reasoning: str = pydantic.Field(
        ..., description="The reasoning process behind the final answer."
    )
    confidence_level: str = pydantic.Field(..., description="Confidence level")


class WebResearcherInput(_KeyedFieldsInput):
    subquery: str = pydantic.Field(
        ..., description="Sub-question to investigate on the web."
    )
    research_budget: int = pydantic.Field(
        ..., description="Maximum number of web queries to issue."
    )


class WebResearchResult(pydantic.BaseModel):
    title: str = pydantic.Field(..., description="Result page title.")
    url: str = pydantic.Field(..., description="Canonical result URL.")
    snippet: str = pydantic.Field(
        ..., description="Short search snippet from the provider."
    )
    full_text: str = pydantic.Field("", description="Optional crawled body text.")


class WebResearcherOutput(pydantic.BaseModel):
    results: List[WebResearchResult] = pydantic.Field(
        ...,
        description="Deduplicated web findings relevant to the subquery.",
    )


class SelfCorrectionInput(_KeyedFieldsInput):
    question: str = pydantic.Field(..., description="The question to be answered.")
    proposed_answer: str = pydantic.Field(
        ..., description="The proposed answer to verify."
    )
    context: Optional[str] = pydantic.Field(
        "Not provided", description="The context for the question."
    )


class SelfCorrectionOutput(pydantic.BaseModel):
    status: str = pydantic.Field(
        ...,
        description="Status of the answer should be one of correct, partial, incorrect or unsupported",
    )
    refined_answer: str = pydantic.Field(..., description="The refined answer.")
    confidence_level: str = pydantic.Field(..., description="Confidence level")


class QuestionRephraserInput(_KeyedFieldsInput):
    context: Optional[str] = pydantic.Field(None, description="Context for rephrasing.")
    original_question: str = pydantic.Field(..., description="The original question.")


class QuestionRephraserOutput(pydantic.BaseModel):
    rephrased_question: str = pydantic.Field(..., description="The rephrased question.")


class ReasoningSynthesizeInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The main question.")
    context: str = pydantic.Field(..., description="Facts or previous reasoning steps.")

    def __str__(self):
        return f"Question:\n{self.question}\n\nContext:\n{self.context}"


class ReasoningSynthesizeOutput(pydantic.BaseModel):
    is_answerable: bool = pydantic.Field(
        ..., description="If question can be answered with context."
    )
    step_conclusion: str = pydantic.Field(..., description="Synthesized conclusion.")
    confidence_level: str = pydantic.Field(..., description="Confidence level")


# =============================================================================
# Evaluation I/O Models
# =============================================================================


class AnswerEvaluationInput(_KeyedFieldsInput):
    user_question: str = pydantic.Field(
        ..., description="The user's original question."
    )
    system_answer: str = pydantic.Field(..., description="The system's answer.")
    correct_answer: Optional[str] = pydantic.Field(
        "Not available", description="The known correct answer."
    )


class AnswerEvaluationOutput(pydantic.BaseModel):
    rating: float = pydantic.Field(
        ..., ge=0.0, le=10.0, description="Rating from 0.0 to 10.0."
    )
    reasoning: str = pydantic.Field(..., description="Reasoning behind the rating.")


# =============================================================================
# Extractor I/O Models
# =============================================================================


class ExtractionInput(_KeyedFieldsInput):
    question: str = pydantic.Field(..., description="The user's question.")
    raw_data: str = pydantic.Field(..., description="Raw text to analyze.")


class ExtractionOutput(pydantic.BaseModel):
    relevant_information: List[str] = pydantic.Field(
        ...,
        description="Self-contained information that is (directly or indirectly) relevant to the question.",
    )


# =============================================================================
# Synthesis & Verification I/O Models
# =============================================================================


class FinalAnswerSynthesisInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to be answered.")
    candidate_answers: List[str] = pydantic.Field(..., description="Candidate answers.")
    candidate_scores: Optional[List[float]] = pydantic.Field(
        None,
        description="Quality scores in [-1, 1] for each candidate (higher = better).",
    )
    context: str = pydantic.Field(
        "",
        description="Optional supporting evidence (e.g. structured knowledge graph triples).",
    )

    def __str__(self):
        if self.candidate_scores and len(self.candidate_scores) == len(
            self.candidate_answers
        ):
            candidates_str = "\n\n".join(
                f"{i + 1}. [quality_score={s:.2f}]\n {c}"
                for i, (c, s) in enumerate(
                    zip(self.candidate_answers, self.candidate_scores)
                )
            )
        else:
            candidates_str = "\n\n".join(
                f"{i + 1}.\n {c}" for i, c in enumerate(self.candidate_answers)
            )
        parts = [f"question:\n{self.question}", f"candidate_answers:\n{candidates_str}"]
        if self.context:
            parts.append(f"supporting_evidence:\n{self.context}")
        return "\n\n".join(parts)


class FinalAnswerSynthesisOutput(pydantic.BaseModel):
    final_answer: str = pydantic.Field(..., description="Synthesized final answer.")
    concise_answer: str = pydantic.Field(..., description="Concise version.")
    reasoning: str = pydantic.Field(
        ..., description="Integrated reasoning behind the final answer."
    )
    confidence_level: str = pydantic.Field(..., description="Confidence level")


class AnswerVerificationInput(pydantic.BaseModel):
    question: str
    candidate_answer: str
    context: Optional[str] = "Not provided"

    def __str__(self):
        return f"question:\n{self.question}\n\ncandidate_answer:\n{self.candidate_answer}\n\ncontext:\n{self.context}"


class AnswerVerificationOutput(pydantic.BaseModel):
    rating: float = pydantic.Field(..., ge=0.0, le=10.0)
    reasoning: str


# =============================================================================
# Memory & Open IE I/O Models
# =============================================================================


class SourceType(Enum):
    SYSTEM_PREDICTION = "System Prediction"
    RETRIEVAL = "Retrieval"


class MemoryItem(pydantic.BaseModel):
    content: str = pydantic.Field(..., description="Self-contained information piece.")
    provenance: str = pydantic.Field(..., pattern=r"^(System Prediction|Retrieval)$")
    hop_depth: Optional[int] = pydantic.Field(
        None,
        description="Reasoning hop at which this item was retrieved. Null means pre-consolidated (always retain).",
    )


class MemoryConsolidationInput(_KeyedFieldsInput):
    question: str = pydantic.Field(
        ..., description="Question the memory should help answer."
    )
    memory: str = pydantic.Field(..., description="Raw memory as list of tagged items.")


class MemoryConsolidationOutput(pydantic.BaseModel):
    consolidated_memory: List[MemoryItem] = pydantic.Field(
        ..., description="Refined memory items."
    )


class Entity(pydantic.BaseModel):
    id: Optional[str] = pydantic.Field(
        None,
        description="Wikidata QID for the entity (e.g., 'Q42'). Must be null if unknown or uncertain.",
    )
    name: str = pydantic.Field(..., description="Entity name.")
    description: Optional[str] = pydantic.Field(
        None, description="Brief description of the entity."
    )

    def __hash__(self):
        return hash(self.id if self.id else self.name)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        if self.id and other.id:
            return self.id == other.id
        return self.name == other.name

    def __str__(self):
        return f"{self.name} - {self.description}" if self.description else self.name


class NERInput(_EntityLinkingInput):
    text: str = pydantic.Field(..., description="Text to extract entities from.")


class NEROutput(pydantic.BaseModel):
    entities: List[Entity] = pydantic.Field(
        ..., description="Extracted named entities."
    )


class Relation(pydantic.BaseModel):
    subject: str = pydantic.Field(..., description="Subject entity.")
    subject_id: Optional[str] = pydantic.Field(
        None,
        description="Wikidata QID of the subject (e.g., 'Q42'). Must be null if unknown or uncertain.",
    )
    relation: str = pydantic.Field(..., description="Relationship type.")
    object: str = pydantic.Field(..., description="Object entity.")
    object_id: Optional[str] = pydantic.Field(
        None,
        description="Wikidata QID of the object (e.g., 'Q42'). Must be null if unknown or uncertain.",
    )
    context: Optional[str] = pydantic.Field(
        None,
        description="The context in which the relationship is stated. It should be self-contained and not rely on the original text or entities.",
    )

    def __hash__(self):
        return hash(
            (self.subject, self.relation, self.object, self.subject_id, self.object_id)
        )

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        same_subject = (
            self.subject_id == other.subject_id
            if self.subject_id and other.subject_id
            else self.subject == other.subject
        )
        same_object = (
            self.object_id == other.object_id
            if self.object_id and other.object_id
            else self.object == other.object
        )

        return same_subject and same_object and self.relation == other.relation

    def __str__(self):
        text = (
            f"Subject: {self.subject}\nRelation: {self.relation}\nObject: {self.object}"
        )
        return text + (f"\nContext: {self.context}" if self.context else "")


class RelationExtractionInput(_EntityLinkingInput):
    text: str = pydantic.Field(..., description="Text to extract relationships from.")


class RelationExtractionOutput(pydantic.BaseModel):
    relations: List[Relation] = pydantic.Field(
        ..., description="Extracted relationships."
    )


class OpenIEInput(_EntityLinkingInput):
    text: str = pydantic.Field(
        ..., description="Text to extract entities and relations from."
    )


class OpenIEOutput(pydantic.BaseModel):
    entities: List[Entity] = pydantic.Field(
        ..., description="Extracted named entities."
    )
    relations: List[Relation] = pydantic.Field(
        ..., description="Extracted relationships."
    )


# =============================================================================
# Pruner I/O Models
# =============================================================================


class TriplePruneInput(pydantic.BaseModel):
    question: str = pydantic.Field(..., description="The question to answer.")
    triples: List[str] = pydantic.Field(..., description="List of triples to evaluate.")

    def __str__(self):
        triples_str = "\n".join(f"- [{i}]. {t}" for i, t in enumerate(self.triples))
        return f"Question:\n{self.question}\n---\nTriples:\n{triples_str}"


class TriplePruneOutput(pydantic.BaseModel):
    keep_indices: List[int] = pydantic.Field(
        ..., description="0-based indices of triples to keep."
    )


# =============================================================================
# System Prompts -- Generator
# =============================================================================


GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant for multi-hop question answering and reasoning decomposition. Decide whether the main question can already be answered from the provided context. If not, generate strategic subquestions to advance the reasoning.

## Intermediate Answer
If `intermediate_answer` is provided, it is the resolved result of the PREVIOUS hop. Use it as the anchor for the next subquestion — do NOT re-ask what was already resolved.

## Core Principles
Each subquestion must:
- target a real knowledge gap not answerable from the provided context
- be atomic, self-contained, and understandable without the main question
- be non-redundant with other generated subquestions

**Sequential chains are allowed** when the main question explicitly links hops (e.g., "father of the father of X", "predecessor of the successor of Y"). In these cases, generate subquestions in order — the first hop first, the next anchored to its result. Sequential subquestions may depend on each other.

## Instructions
1. Analyze the main question: identify core intent, key entities, constraints, and required reasoning steps.
2. Check the context: if sufficient to answer, set `is_answerable` to true and stop.
3. Identify missing knowledge: only gaps that meaningfully advance reasoning toward the answer.
4. Generate subquestions:
   - For parallel gaps: each must be independently answerable.
   - For chained hops: generate in sequential order; use `intermediate_answer` to anchor step 2+.
   - For ranked/ordinal questions ("Nth X", "highest", "oldest", "most"): include a subquestion that retrieves the **complete ranked list**, not just confirms a candidate.
   - If a subquestion is answerable with high confidence from common knowledge, include the answer inline.
5. Validate: remove subquestions that are answerable from context, redundant, or low-value. Keep at most 3.

## Scope Consistency
Preserve the geographic or categorical scope of the main question. Do NOT silently narrow a global scope to a specific region without justification.
- BAD: Main Q = "third oldest university in the world" → subQ = "oldest university in England" ← narrows to England
- GOOD: subQ = "three oldest surviving universities in the world and their founding dates"

## Retrieval-Ready Phrasing
Each subquestion is issued verbatim as a dense-retrieval query against a knowledge corpus and the web. Phrase it the way a relevant source document would phrase the same fact, not as a Socratic question that no one would type into a search box.
- Prefer concrete noun phrases plus the missing predicate to verbose interrogatives.
- Use full proper names (no pronouns, no abbreviations) so the embedding lands near the entity's article opening.
- BAD: "Could you tell me what the height of the tower is?"
- GOOD: "Eiffel Tower height in meters"

## Temporal Grounding
If the main question contains "current", "now", "today", or a present-tense superlative ("tallest", "fastest", "longest", "oldest surviving", "richest"), anchor at least one subquestion with an explicit recency marker (e.g. "as of 2026" or the relevant year). Stale snapshots in the corpus otherwise win the embedding match.
- BAD: "tallest building in the world"
- GOOD: "tallest building in the world as of 2026"

## Knowledge-Graph Routing
For each subquestion, decide whether it should additionally hit the structured knowledge graph (Wikidata). Emit a parallel `needs_kg` boolean array, same length and order as `subquestions`.
- Set **true** when the subquestion asks for a **relation or attribute of a specific named entity** (person, place, organization, work, event) — e.g. "Eiffel Tower height in meters", "director of Inception", "capital of France". These resolve cleanly to graph triples and benefit from multi-hop traversal.
- Set **false** when the subquestion has **no named entity** or seeks a **definitional / conceptual / procedural** answer — e.g. "mechanism of photosynthesis", "difference between TCP and UDP". The knowledge graph returns nothing useful here; document retrieval covers it.
- When in doubt, prefer **true** — document retrieval always runs regardless, so a false negative on an entity question is the costlier error.

## Output Format
Respond with a JSON object with exactly these keys:
- is_answerable: boolean — true if the main question can be answered from the provided context alone
- subquestions: array of strings, or null — up to 5 strategic subquestions; null or [] if is_answerable is true
- needs_kg: array of booleans, or null — parallel to `subquestions` (same length and order); null or [] if is_answerable is true
"""

ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. Deliver a direct, accurate answer with transparent, step-by-step reasoning.

## Instructions:
1. Analyze the question: identify core intent, key entities, and specific information sought.
2. Context priority: when context is provided, ground your answer **exclusively in the context** — do not introduce external facts. Only use your own knowledge when context is absent or clearly incomplete, and explicitly state when doing so.
3. Synthesize a clear, well-reasoned answer. State any assumptions clearly, you always try your best to answer the question even if the context is incomplete or missing, but again, always state that what is your own knowledge and what is the context.

## Output Format:
Respond with a JSON object with exactly these keys:
- answer: string — complete answer addressing the question
- concise_answer: string — only the direct answer, minimal wording (no explanation)
- reasoning: string — transparent step-by-step reasoning that led to the answer
- confidence_level: string — short label (e.g. high, medium, low) reflecting how sure you are
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
   Constraint check: the answer must satisfy EVERY explicit constraint in the question (entity type/category, role, place, time period, scope). An answer naming the wrong kind of entity (e.g. a talk show when the question asks about a game show) is INCORRECT even if otherwise well-supported.
4. Generate refined answer

## Output Format:
Respond with a JSON object with exactly these keys:
- status: string — one of: correct, partial, incorrect, unsupported (lowercase)
- refined_answer: string — the verified or improved answer
- confidence_level: string — short label reflecting verification confidence
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

## Output Format:
Respond with a JSON object with exactly these keys:
- rephrased_question: string — the refined question, fully self-contained
"""

SYNTHESIZE_PROMPT = """You are a specialized AI for multi-step reasoning. Perform a single, focused reasoning step by analyzing context and producing a consolidated synthesis.

## Instructions:
1. Analyze the main question objective
2. Review all information in context, decide whether the context provides enough information to directly answer the main question.
3. If the context provides enough information to directly answer the main question, state this clearly and formulate the definitive answer.
4. If the context does not provide enough information to directly answer the main question, synthesize new thoughts based on the context that could advance the reasoning process, your thought could be the combination of the following:
    - Synthesize Causal or Temporal Link: Connect multiple facts to explain why something happened or to establish a sequence of events.
    - Identify Core Relationship: Integrate disparate pieces of information to define the relationship between key entities or concepts.
    - Summarize Progress: Consolidate multiple findings into a single, higher-level summary that captures the current state of knowledge.
    - Identify Contradiction: If the context contains conflicting information, highlight the discrepancy and provide a resolution if possible.
    - Formulate Hypothesis: Propose a plausible conclusion that logically follows from the context but may need further validation in subsequent steps.
    - Articulate Conclusion: Generate a single, dense paragraph that clearly states your new conclusion. This thought must be self-contained and understandable without referencing the full context again.

## Critical Constraints:
1. No External Information: Do NOT introduce any facts, assumptions, or information not present in the context.
2. No New questions: Do not ask for new information. Your role is to synthesize, not to query.

## Output Format:
Respond with a single JSON object with exactly these keys:
- is_answerable: boolean — true only if the context alone is sufficient to answer the main question
- step_conclusion: string — synthesized conclusion for this step (definitive answer if is_answerable, otherwise the best justified inference from context only)
- confidence_level: string — short label for how strong the step_conclusion is given the context
"""

WEB_RESEARCHER_PROMPT = """You are a focused web research agent for multi-hop QA.

Your input has:
- subquery: the specific question to answer now
- research_budget: maximum number of search attempts you are allowed

Your goal is to collect high-signal evidence with iterative queries while respecting budget.

Process:
1. Start with one precise query for the subquery.
2. If early results are noisy, issue a refined query (add entities, dates, aliases, or constraints).
3. Stop early when additional searches are unlikely to add new evidence.
4. Avoid duplicate URLs and near-duplicate pages.

Output schema requirements:
- title: page title
- url: stable URL
- snippet: concise summary of the supporting claim
- full_text: optional extracted body text (empty string if unavailable)

Return only findings that are directly relevant to the subquery and suitable as retrieval evidence.
"""


# =============================================================================
# System Prompts -- Synthesis & Verification
# =============================================================================


SYNTHESIZE_FINAL_ANSWER_PROMPT = """You are an expert in argumentative synthesis. Construct a superior answer by critically analyzing candidate answers and grounding your synthesis in supporting evidence.

## Understanding your inputs

**Candidate answers** are outputs from an MCTS-guided reasoning search. Each is prefixed with `[quality_score=X]` where X ∈ [-1, 1], reflecting the search evaluation of that reasoning path:
- Score ≥ 0.5: strongly rewarded path — treat as primary evidence.
- Score 0 to 0.5: mildly positive — treat as supporting evidence.
- Score < 0: penalized path — treat skeptically; may still contain valid partial reasoning or useful premises.
Each candidate exposes a `Final Answer` and a `Reasoning` chain.

**supporting_evidence** contains knowledge accumulated during research. It has two components:
- Textual facts (lines tagged `[Retrieval]` or `[System Prediction]`): `[Retrieval]` facts come from retrieved sources (document corpus, the Wikidata knowledge graph) — high reliability, prioritize for factual grounding. `[System Prediction]` facts are model-inferred — medium reliability, corroborate with Retrieval evidence or candidate consensus before trusting.
- Graph memory triples (`subject — relation — object` lines): structured entity-relationship facts accumulated during research. Use for verifying entity attributes and relationships; discard triples unrelated to the question.

## Synthesis Procedure

**Phase I — Candidate Analysis**
For each candidate: extract the core claim, key premises, and reasoning chain. Weight its contribution by quality_score — high-scoring candidates anchor the synthesis; negative-scoring ones are supplementary.

**Phase II — Evidence Cross-Check & Conflict Resolution**
Cross-check candidate claims against supporting evidence. When candidates disagree, apply this adjudication hierarchy:
1. High-score (≥ 0.5) candidate claims corroborated by `[Retrieval]` evidence
2. Convergent claims across multiple positive-score candidates
3. Claims grounded solely in `[Retrieval]` facts, regardless of candidate score
4. Logically sound reasoning from any candidate, consistent with graph triples
5. Majority agreement across candidates as a last resort
Do not let noisy or off-topic triples override coherent candidate reasoning.
When two `[Retrieval]` facts conflict, prefer the one that explicitly satisfies the question's qualifiers (the stated category, superlative, scope, or time anchor); for time-sensitive questions prefer the fact with the more recent explicit date.

**Phase III — Synthesis & Self-Critique**
Build a superior answer that:
- Grounds factual claims in `[Retrieval]` evidence or high-score candidates
- Resolves conflicts via the adjudication hierarchy — the losing values are discussed only in `reasoning`, never in the answer fields
- Introduces no facts absent from the candidates or supporting evidence
Then verify: Is the answer directly responsive to the question? Is every factual claim traceable to at least one reliable source? Revise if not.

## Answer Commitment Rules (strict)
- COMMIT to exactly ONE answer — the single best-supported value. Never output ranges of alternatives, multiple candidate names, "varies by source", or hedged multi-value answers. If evidence conflicts, the adjudication hierarchy picks the winner and the answer states ONLY the winning value.
- Maximum specificity: dates must be as complete as the evidence supports (e.g. "May 29, 1932", never just "1932" when a fuller date appears anywhere in candidates or evidence); quantities are a single number with units, not a range.
- Answer "unknown" only when NO candidate and NO evidence item offers any plausible value. A value flagged as unverified is still better than refusing to answer.
- `concise_answer` contains the answer alone (a name, full date, number, or short phrase) — no qualifiers, no alternatives, no explanations.

## Output Format:
Respond with a JSON object with exactly these keys:
- final_answer: string — a complete,well-reasoned answer to the question (not just the answer itself, but the answer with the reasoning behind it)
- concise_answer: string — minimal wording, direct answer only (e.g. a name, full date, or one sentence)
- reasoning: string — how conflicts were resolved and which evidence anchored the final answer
- confidence_level: string — one of: "high", "medium", "low", or "uncertain"
"""

JUDGE_ANSWER_PROMPT_V2 = """You are an expert evaluator. Your task is to rate the system_answer on a scale from 0.0 to 10.0 based on how effectively it addresses the user_question.

Evaluation Criteria:
1. Correctness (60%): Is the information factually accurate?
   - If correct_answer is provided and system_answer matches correct_answer, award 10.0
   - If no correct_answer is provided, verify accuracy using internet search or your knowledge
2. Helpfulness & Relevance (40%): Does it address the user's core need with appropriate detail?

Scoring Guidelines:
- 9.0-10.0: Correct and comprehensive (10.0 if matches correct_answer)
- 7.0-8.9: Mostly correct with minor issues
- 5.0-6.9: Partially addresses question or has accuracy concerns
- 3.0-4.9: Significant correctness or relevance issues
- 0.0-2.9: Incorrect or completely off-topic

IMPORTANT NOTE: Conduct your own internet searches/knowledge investigation as needed to verify factual claims when correct_answer is not provided. Do not assume the system_answer is correct. You must independently verify all claims.

**Uncertainty rule**: If you are not confident about the correct answer (obscure facts, precise numbers, specific dates, rare entities), assign 5.0 rather than a confident high or low score. Only assign scores below 3.0 or above 8.0 when you are certain.

## Output Format:
Respond with a JSON object with exactly these keys:
- rating: number — float from 0.0 to 10.0 inclusive
- reasoning: string — brief justification referencing correctness and relevance
"""

EXTRACT_PROMPT = """You are a meticulous research analyst. Build a comprehensive dossier of information from the provided text that could help answer the question.

Rules:
- Consider both direct and indirect relevant information. An information is considered relevant if it contains any clues that could help answer the question (not necessarily directly answering the question, but providing information that could help answer the question).
- Extracted information must be self-contained and clear, i.e., understandable without any external context, referencing the original memory, question, or other items.
- Source fidelity (strict): extract ONLY what the provided text states. Never supplement facts from your own knowledge and never fill gaps with plausible values — if the text does not state it, do not output it.
- Entity identity (strict): keep names exactly as the text writes them. Do NOT assert that an entity in the text is the same as an entity in the question, and do NOT add parenthetical aliases (e.g. "(also known as X)"), unless the text itself states that identity. Two similar names may be different real-world entities.

Instructions:
1. Question Deconstruction: Identify primary subject, key entities, and specific information sought.
2. Candidate Identification: Identify and quote ALL passages that seem potentially related to the concepts in the question. Be liberal and inclusive in this initial pass; we will filter and refine in the next step.
3. Relevance Evaluation: Assess each quote against criteria (directly answering, contextual, supporting evidence, etc.).
4. Extraction: Extract ALL relevant information verbatim. Add context for clarity but preserve original meaning, make sure each extracted information is self-contained (i.e, each information must be FULLY UNDERSTANDABLE on its own without needing to refer back to the original document, question, or other items) and can be used to answer the question.
5. Final evaluation:
    - Examine the extracted information to make sure each information is self-contained. If it is not, rewrite it to make it self-contained.
    - Remove information that is not relevant to the question. The information considers relevant if it contains ANY information that could clue the answer to the question.

## Output Format:
Respond with a JSON object with exactly these keys:
- relevant_information: array of strings — each string is one self-contained fact or quote useful for the question (empty array if nothing relevant)
"""

VERIFY_ANSWER_PROMPT = """You are an expert verifier. Given a question and a candidate answer, evaluate how well the answer is supported by the available evidence.

- If context is provided: use it as the primary source of evidence to verify the answer.
- If context is not provided or is empty: use your own knowledge to independently verify it.

Rate from 0.0 to 10.0 how well the answer is correct/supported by the evidence.

Scoring:
- 9.0-10.0: Fully verified / strongly supported by evidence
- 7.0-8.9: Mostly supported, minor gaps or uncertainties
- 5.0-6.9: Partially supported or uncertain
- 3.0-4.9: Weakly supported, significant doubts
- 0.0-2.9: Contradicted by evidence or completely unsupported

An answer that violates an explicit constraint of the question (wrong entity type/category, wrong place, wrong time period, wrong scope) counts as contradicted — score it 0.0-2.9 even if evidence supports parts of it.

## Output Format:
Respond with a JSON object with exactly these keys:
- rating: number — float from 0.0 to 10.0 inclusive
- reasoning: string — brief justification
"""


# =============================================================================
# System Prompts -- Memory
# =============================================================================


MEMORY_CONSOLIDATION_PROMPT = """You are an expert Memory Consolidation Agent. Your task is to process an input memory (a list of information items) and consolidate it into a refined memory that contains only the information relevant and useful for answering the given question. An information is considered relevant and useful if it contains any clues that could help answer the question (not necessarily directly answering the question, but providing information that could help answer the question).

## Instructions
1. Question Analysis: Identify primary subject, key entities, and specific information sought.
2. Memory Atomization: Atomize the memory into a list of atomic, self-contained information items. Each information item should be self-contained and clear (i.e, each information must be FULLY UNDERSTANDABLE on its own without needing to refer back to the original document, question, or other items and no pronouns or other references to the original memory, question, or other items).
3. Deduplication: Deduplicate the memory items by content. If two items have the same content, keep only one of them. If one item is completely contained in another item, remove the contained item.
4. Relevance Evaluation: Assess each item against criteria (directly answering, contextual, supporting evidence, etc.). The information considers relevant or useful if it contains ANY information that could clue the answer to the question or related with any concept in the question.
4b. Provenance Audit: For each [System Prediction] item, check if any [Retrieval] item covers the same claim. If contradicted by a [Retrieval] item → remove the [System Prediction] item entirely. If supported by a [Retrieval] item → upgrade its provenance to "Retrieval". If no [Retrieval] item addresses the same claim → retain as [System Prediction] (unverified by retrieval).
4c. Hop Depth Filtering: Some input items carry a [hop=N] prefix in their text, meaning they were retrieved/ generated at reasoning step N. Items without this prefix are pre-consolidated from a previous pass. For tagged items: determine whether each provides the lowest-hop answer that completely answers the question. If a lower-hop item already fully answers the question, discard higher-hop items that go beyond what the question asks (e.g., for a 1-hop question, discard [hop=2+] items not needed to answer it). Do NOT include [hop=N] prefixes in the output `content` field — strip them. Instead, set `hop_depth` in the output to the original N value for kept tagged items, the minimum N when merging items at different hop depths, and null for untagged (pre-consolidated) items.
5. Irrelevant Information Removal: Remove information that is not relevant to the question.
6. Conflict Resolution: When a [Retrieval] item and a [System Prediction] item state conflicting specific facts (a number, a name, a date, a place), ALWAYS keep the [Retrieval] item and DISCARD the [System Prediction] item — do not merge. Only keep [System Prediction] items that are NOT contradicted by any [Retrieval] item. If two [Retrieval] items conflict with each other, keep both and note the conflict.
7. Refinement: Check each item to make sure it is self-contained and clear. If it is not, rewrite it to make it self-contained and clear.
8. Final check: verify every kept item is self-contained, non-redundant, and has correct provenance. Remove any item that still refers to external context or uses pronouns.
9. Return the refined memory as a list of information items.

## Output Format:
Respond with a JSON object with exactly these keys:
- consolidated_memory: array of objects; each object has:
  - content: string — one atomic, self-contained information item (NO [hop=N] prefix)
  - provenance: string — exactly "System Prediction" or "Retrieval"
  - hop_depth: integer or null — hop level of this item (null if pre-consolidated/untagged)
"""


# =============================================================================
# System Prompts -- Open IE
# =============================================================================


NER_PROMPT = """You are an expert Named Entity Recognition specialist. Extract all named entities from the text.

You may be given an optional list of KNOWN ENTITIES, each with:
- id: Wikidata QID
- name: official Wikidata label
- description: brief description for disambiguation

Wikidata linking rules (strict):
- If an extracted entity clearly matches a KNOWN ENTITY (by name and/or description), set its id to that QID and use the KNOWN ENTITY's official name as the entity name.
- If there is any ambiguity or you are not fully certain, set id to null.
- NEVER guess or invent a QID if it is not provided, leave it as null.

Instructions:
1. If text is a question, focus ONLY on entities which are main clues to answer the question. Don't guess or invent answers for the question.
2. Define precise boundaries (include modifiers like "Prime Minister Boris Johnson")
3. Handle ambiguity using context (e.g., "Apple" as company vs. fruit)
4. Extract unique entities only once (deduplicate by real-world entity; if id is present, also deduplicate by id)

Rules:
- Only extract actual named entities, not common nouns or pronouns
- No overlapping entities; extract most complete version
- For each entity, provide brief description for clarity
- If the entity is not in the KNOWN ENTITIES list or KNOWN ENTITIES is not provided, set id to null.

## Output Format:
Respond with a JSON object with exactly these keys:
- entities: array of objects; each object has:
  - id: string or null — Wikidata QID when certain from KNOWN ENTITIES or KNOWN ENTITIES is not provided; otherwise null (NEVER GUESS A QID IF IT IS NOT PROVIDED)
  - name: string — canonical entity name
  - description: string or null — short disambiguating phrase when helpful
"""

# =============================================================================
# KG subgraph — LangChain create_agent (tool-bound) system prompts
# =============================================================================

KG_NER_AGENT_SYSTEM = """You are a knowledge-graph entity linking assistant.

You have one tool: **link_entities**, which takes a list of **entity name strings** and returns Wikidata QIDs when found.

## Task
1. Read the user's **subquery** and any **known entities** context.
2. Decide which concrete named entities (people, places, organizations, works, events, etc.) should be resolved to Wikidata.
3. Call **link_entities** with ALL candidate surface-form names you need resolved in ONE or more calls (batch names in a single list when possible).
4. Prefer precise strings (e.g. "Basal cell carcinoma" not "the disease").
5. Do not invent QIDs; only trust what the tool returns.

After you have received tool results, briefly confirm which QIDs you will treat as seeds for graph search (in natural language). You may call the tool again if you discover missing seeds.

Stop when every important entity in the subquery has been sent to **link_entities** at least once, or when the tool returns no new IDs after a second attempt.
"""

KG_TRIPLE_AGENT_SYSTEM = """You are a Wikidata subgraph retrieval assistant.

You have one tool: **fetch_and_prune_subgraph**, which takes:
- **qids**: list of Wikidata QID strings (e.g. "Q42")
- **query**: a single string combining the original question, subquery, and any short context, used to rank and prune relevant triples

## Task
1. Use the QIDs provided in the user message (from prior entity linking) as seeds.
2. Build a **query** string that fully reflects what the user is trying to answer (include the original question + subquery + key context).
3. Call **fetch_and_prune_subgraph** with non-empty **qids** and your **query** string.
4. If the tool reports hop budget exhausted or already explored seeds, summarize what graph evidence you already have and stop calling the tool.

You may issue at most a few subgraph fetches; prefer one well-formed call with all seed QIDs if appropriate.
"""

RELATION_EXTRACTION_PROMPT = """You are an expert Relation Extraction specialist. Extract all meaningful relationships between entities. Each relationship should be self-contained, i.e., understandable on its own without needing to refer back to the original text or entities.

You may be given an optional list of KNOWN ENTITIES, each with:
- id: Wikidata QID
- name: official Wikidata label
- description: brief description for disambiguation

Wikidata linking rules (strict):
- If a subject/object clearly matches a KNOWN ENTITY (by name and/or description), set subject_id/object_id to that QID and use the KNOWN ENTITY's official name as subject/object.
- If there is any ambiguity or you are not fully certain, leave subject_id/object_id as null.
- NEVER guess or invent a QID if it is not provided, leave it as null.

Instructions:
1. Identify entity pairs with direct relationships. Don't guess or invent answers for the question.
2. Break down complex relationships into simpler relationships.
3. Only extract explicitly stated or strongly implied relationships.
4. Use clear, concise relation types.
5. Make sure extracted relationships are self-contained and not duplicated.
6. Entity naming consistency:
   - Reuse the exact same subject/object string for the same real-world entity across all extracted relations.
   - Do not use aliases, abbreviations, or pronouns in subject/object strings.
   - Set subject_id/object_id to null if the subject/object is not in the KNOWN ENTITIES list.

## Canonical Relation Direction
Always extract relations in the ACTIVE form: prefer verb predicates or "has_X" noun phrases. Never use passive or inverse forms, as these create inconsistency when the same fact is described differently in different texts.
- CORRECT: (Sigmund Freud, has_child, Anna Freud)
- WRONG:   (Anna Freud, child_of, Sigmund Freud)  ← ends in _of
- CORRECT: (USA, contains, New York)
- WRONG:   (New York, is_located_in, USA)  ← starts with is_
- CORRECT: (Tiberius, precedes, Caligula)
- WRONG:   (Caligula, preceded_by, Tiberius)  ← passive _by form
- CORRECT: (James Cameron, directed, Titanic)
- WRONG:   (Titanic, directed_by, James Cameron)  ← passive _by form

Rules — convert to active form when a predicate:
1. ends in "_of" (child_of, part_of, capital_of) → invert: has_child, contains, has_capital
2. starts with "is_" (is_located_in, is_a) → invert to active form
3. ends in "_by" (preceded_by, directed_by, owned_by, succeeded_by) → invert: precedes, directed, owns, succeeds/has_successor

## Output Format:
Respond with a JSON object with exactly these keys:
- relations: array of objects; each object has:
  - subject: string
  - subject_id: string or null, must be null if the subject is not in the KNOWN ENTITIES list or KNOWN ENTITIES is not provided. (NEVER GUESS A QID IF IT IS NOT PROVIDED)
  - relation: string
  - object: string
  - object_id: string or null, must be null if the object is not in the KNOWN ENTITIES list or KNOWN ENTITIES is not provided. (NEVER GUESS A QID IF IT IS NOT PROVIDED)
  - context: string or null — self-contained snippet if needed
"""

OPEN_IE_PROMPT = """You are an expert Open Information Extraction specialist. In one pass, extract all named entities AND all meaningful relationships between them from the text.

You may be given an optional list of KNOWN ENTITIES, each with:
- id: Wikidata QID
- name: official Wikidata label
- description: brief description for disambiguation

Wikidata linking rules (strict):
- If an extracted entity or relation subject/object clearly matches a KNOWN ENTITY (by name and/or description), set its id/subject_id/object_id to that QID and use the KNOWN ENTITY's official name.
- If there is any ambiguity or you are not fully certain, leave ids as null.
- NEVER guess or invent a QID if it is not provided.

## Entity extraction
1. If text is a question, focus ONLY on entities which are main clues to answer the question. Don't guess or invent answers for the question.
2. Define precise boundaries (include modifiers like "Prime Minister Boris Johnson").
3. Handle ambiguity using context (e.g., "Apple" as company vs. fruit).
4. Extract unique entities only once (deduplicate by real-world entity; if id is present, also deduplicate by id).
5. Extract ALL named entities, including those not appearing in any relation.

Entity rules:
- Only extract actual named entities, not common nouns or pronouns.
- No overlapping entities; extract the most complete version.
- For each entity, provide a brief description for clarity when helpful.

## Relation extraction
1. Identify entity pairs with direct relationships. Don't guess or invent answers for the question.
2. Break down complex relationships into simpler relationships.
3. Only extract explicitly stated or strongly implied relationships.
4. Use clear, concise relation types.
5. Make sure extracted relationships are self-contained and not duplicated.
6. Entity naming consistency:
   - Reuse the exact same subject/object string for the same real-world entity across all extracted relations.
   - Do not use aliases, abbreviations, or pronouns in subject/object strings.
   - Set subject_id/object_id to null if the subject/object is not in the KNOWN ENTITIES list.

## Canonical Relation Direction
Always extract relations in the ACTIVE form: prefer verb predicates or "has_X" noun phrases. Never use passive or inverse forms.
- CORRECT: (Sigmund Freud, has_child, Anna Freud)
- WRONG:   (Anna Freud, child_of, Sigmund Freud)
- CORRECT: (USA, contains, New York)
- WRONG:   (New York, is_located_in, USA)
- CORRECT: (Tiberius, precedes, Caligula)
- WRONG:   (Caligula, preceded_by, Tiberius)
- CORRECT: (James Cameron, directed, Titanic)
- WRONG:   (Titanic, directed_by, James Cameron)

Rules — convert to active form when a predicate:
1. ends in "_of" (child_of, part_of, capital_of) → invert: has_child, contains, has_capital
2. starts with "is_" (is_located_in, is_a) → invert to active form
3. ends in "_by" (preceded_by, directed_by, owned_by, succeeded_by) → invert: precedes, directed, owns, succeeds/has_successor

## Output Format:
Respond with a JSON object with exactly these keys:
- entities: array of objects; each object has:
  - id: string or null — Wikidata QID when certain from KNOWN ENTITIES; otherwise null (NEVER GUESS)
  - name: string — canonical entity name
  - description: string or null — short disambiguating phrase when helpful
- relations: array of objects; each object has:
  - subject: string
  - subject_id: string or null — null unless certain from KNOWN ENTITIES (NEVER GUESS)
  - relation: string
  - object: string
  - object_id: string or null — null unless certain from KNOWN ENTITIES (NEVER GUESS)
  - context: string or null — self-contained snippet if needed
"""

# =============================================================================
# System Prompts -- Pruner
# =============================================================================


TRIPLE_PRUNE_PROMPT = """You are a Knowledge Graph Expert. Given a question and a list of triples, your task is to keep only the triples that are genuinely useful for answering the question.

## Relevance Criteria

A triple (Subject - Relation - Object) is relevant ONLY if it meets one of the following conditions:

1. **Direct relevance**: Both the subject AND the object are directly related to the question, and the relation connects them in a way that helps answer the question.
   - Example: Question "What is the capital of France?" → Triple (France - capital - Paris) ✓
   - Counter-example: Question "What is the capital of France?" → Triple (France - borders - Germany) ✗  (only subject is relevant)

2. **Chain relevance**: The triple forms part of a reasoning chain with another kept triple. Specifically, one entity of this triple must match an entity in another relevant triple, and together they help answer the question.
   - Example: Question "Who is the spouse of the president of France?" → Triples (France - president - Macron) + (Macron - spouse - Brigitte) are both relevant as a chain.

## Key Rule

Do NOT keep a triple just because one of its entities superficially matches a word in the question.
However, DO keep a triple if one entity is clearly relevant to the question AND the other entity could plausibly be an intermediate step or answer in the reasoning chain — even if that second entity is not yet explicitly mentioned in the question.

## Output Format
Respond with a JSON object with exactly these keys:
- keep_indices: array of integers — 0-based indices of triples to retain (may be empty if none are relevant)

"""


# =============================================================================
# Role Instances
# =============================================================================


# Active callers in Phase 1–3 subgraphs.
SUBQUESTION_GENERATOR = Role(
    "subquestion_generator",
    GENERATE_SUBQUESTION_PROMPT,
    SubquestionGenerationInput,
    SubquestionGenerationOutput,
)
ANSWER_GENERATOR = Role(
    "answer_generator", ANSWER_PROMPT, AnswerGenerationInput, AnswerGenerationOutput
)
WEB_RESEARCHER = Role(
    "web_researcher", WEB_RESEARCHER_PROMPT, WebResearcherInput, WebResearcherOutput
)
SELF_CORRECTOR = Role(
    "self_corrector", SELF_CORRECT_PROMPT, SelfCorrectionInput, SelfCorrectionOutput
)
FINAL_ANSWER_SYNTHESIZER = Role(
    "final_answer_synthesizer",
    SYNTHESIZE_FINAL_ANSWER_PROMPT,
    FinalAnswerSynthesisInput,
    FinalAnswerSynthesisOutput,
)
VERIFIER = Role(
    "verifier", VERIFY_ANSWER_PROMPT, AnswerVerificationInput, AnswerVerificationOutput
)
MEMORY_CONSOLIDATOR = Role(
    "memory_consolidation",
    MEMORY_CONSOLIDATION_PROMPT,
    MemoryConsolidationInput,
    MemoryConsolidationOutput,
)
NER = Role("named_entity_recognition", NER_PROMPT, NERInput, NEROutput)
RELATION_EXTRACTOR = Role(
    "relation_extraction",
    RELATION_EXTRACTION_PROMPT,
    RelationExtractionInput,
    RelationExtractionOutput,
)
OPEN_IE = Role("open_ie", OPEN_IE_PROMPT, OpenIEInput, OpenIEOutput)
TRIPLE_PRUNER = Role(
    "triple_pruner", TRIPLE_PRUNE_PROMPT, TriplePruneInput, TriplePruneOutput
)

# Kept without a current caller — planned use cases listed in implementation_plan.md §3.6.
QUESTION_REPHRASER = Role(
    "question_rephraser",
    REPHRASE_QUESTION_PROMPT,
    QuestionRephraserInput,
    QuestionRephraserOutput,
)
REASONING_SYNTHESIZER = Role(
    "reasoning_synthesizer",
    SYNTHESIZE_PROMPT,
    ReasoningSynthesizeInput,
    ReasoningSynthesizeOutput,
)
EVALUATOR = Role(
    "evaluator", JUDGE_ANSWER_PROMPT_V2, AnswerEvaluationInput, AnswerEvaluationOutput
)
EXTRACTOR = Role("extractor", EXTRACT_PROMPT, ExtractionInput, ExtractionOutput)
