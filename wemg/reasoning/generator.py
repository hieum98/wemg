"""Node generation pipeline: retrieval + LLM reasoning for creating reasoning nodes."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from wemg.llm.roles import (
    execute_role,
    ANSWER_GENERATOR, SUBQUESTION_GENERATOR, QUERY_GENERATOR,
    SELF_CORRECTOR, QUESTION_REPHRASER, REASONING_SYNTHESIZER,
    EXTRACTOR, TRIPLE_PRUNER, 
    AnswerGenerationInput, SubquestionGenerationInput, QueryGeneratorInput,
    SelfCorrectionInput, QuestionRephraserInput, ReasoningSynthesizeInput,
    ExtractionInput, TriplePruneInput,
)
from wemg.retrieval.wikidata import (
    WikidataClient,
    WikidataEntity,
    WikiTriple,
    filter_entities_relevant_to_text,
)
from wemg.retrieval.entity_linking import link_entities_llm, link_entities_azure
from wemg.utils.text import format_context
from wemg.llm.roles import SourceType

logger = logging.getLogger(__name__)

# Max triples per TRIPLE_PRUNER LLM call when pruning large lists.
_PRUNE_TRIPLES_BATCH_SIZE = 64


@dataclass
class GenerationResult:
    answers: List[Any] = field(default_factory=list)
    retrieved_triples: List[WikiTriple] = field(default_factory=list)
    retrieved_entities: List[WikidataEntity] = field(default_factory=list)
    information_items: List[str] = field(default_factory=list)
    log_data: Dict = field(default_factory=dict)


def merge_logs(*logs: Dict) -> Dict:
    """Merge multiple role log dicts."""
    merged = {}
    for log in logs:
        if not log:
            continue
        for key, value in log.items():
            merged.setdefault(key, []).extend(value)
    return merged


class NodeGenerator:
    """Generates reasoning node content using retrieval + LLM.
    
    Used by both MCTS and CoT strategies.
    """
    
    def __init__(
        self,
        client,  # LLMClient
        retriever,  # WebSearchTool or CorpusRetriever
        wikidata_client,  # WikidataClient
        working_memory,  # WorkingMemory
        interaction_memory=None,  # Optional InteractionMemory
        reranker=None,  # Optional Reranker for kb_documents
        **kwargs,
    ):
        self.wikidata_client = wikidata_client
        self.client = client
        self.retriever = retriever
        self.reranker = reranker
        self.working_memory = working_memory
        self.interaction_memory = interaction_memory
        self.kwargs = kwargs
    
    async def generate_answer(self, question: str, should_explore: bool = True) -> GenerationResult:
        """Generate answer via retrieval + extraction + LLM answer generation.
        
        Pipeline:
        1. If should_explore: run explore() to get web results + Wikidata triples
        2. Extract relevant info from web results using Extractor role
        3. Build context from memory + extracted info + KB triples
        4. Generate answer using AnswerGenerator role
        """
        if should_explore:
            documents, retrieved_triples, retrieved_entities, exploration_log = await self._explore(question)
            documents = list(set(documents))
            extractor_inputs = [ExtractionInput(question=question, raw_data=data) for data in documents]
            if extractor_inputs:
                extracted_results, extractor_log = await execute_role(
                    client=self.client, role=EXTRACTOR, input_data=extractor_inputs,
                    interaction_memory=self.interaction_memory, n=1
                )
                all_extractions = sum(extracted_results, [])
                info_from_web = []
                for item in all_extractions:
                    if hasattr(item, 'relevant_information') and item.relevant_information:
                        info_from_web.extend(item.relevant_information)
            else:
                info_from_web = []
                extractor_log = {}
        else:
            info_from_web = []
            retrieved_triples = []
            retrieved_entities = []
            exploration_log = {}
            extractor_log = {}
        
        info_from_kb = [str(t) for t in retrieved_triples]
        all_info = info_from_web + info_from_kb
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_info)
        
        qa_input = AnswerGenerationInput(question=question, context=context)
        answers, qa_log = await execute_role(
            client=self.client, role=ANSWER_GENERATOR, input_data=qa_input,
            interaction_memory=self.interaction_memory, n=self.kwargs.get('n', 1)
        )
        
        return GenerationResult(
            answers=answers,
            retrieved_triples=retrieved_triples,
            retrieved_entities=retrieved_entities,
            information_items=all_info,
            log_data=merge_logs(exploration_log, extractor_log, qa_log),
        )
    
    async def generate_subquestion(self, user_question: str) -> Tuple[List[str], bool, Dict]:
        """Generate subquestions. Returns (subquestions, should_direct_answer, log_data)."""
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        subq_input = SubquestionGenerationInput(question=user_question, context=context)
        n_subq = self.kwargs.get('n_subquestions', 3)
        subquestions, log = await execute_role(
            client=self.client, role=SUBQUESTION_GENERATOR, input_data=subq_input,
            interaction_memory=self.interaction_memory, n=n_subq
        )
        
        unanswerable = []
        for sq in subquestions:
            if not sq.is_answerable and sq.subquestions:
                unanswerable.extend(sq.subquestions)
        unanswerable = list(set(unanswerable)) if unanswerable else []
        
        answerable_count = sum(1 for sq in subquestions if sq and sq.is_answerable)
        should_direct = (answerable_count / len(subquestions)) > 0.5 if subquestions else False
        
        return unanswerable, should_direct, log
    
    async def generate_rephrase(self, question: str) -> Tuple[List[str], Dict]:
        """Generate rephrased questions. Returns (rephrased_questions, log_data)."""
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        input_data = QuestionRephraserInput(context=context, original_question=question)
        rephrased, log = await execute_role(
            client=self.client, role=QUESTION_REPHRASER, input_data=input_data,
            interaction_memory=self.interaction_memory, n=self.kwargs.get('n', 1)
        )
        return [r.rephrased_question for r in rephrased], log
    
    async def generate_self_correction(self, sub_question: str, sub_answer: str) -> GenerationResult:
        """Generate self-corrected answer."""
        step_objective = f"Verify and correct the answer for: {sub_question}.\nProposed answer: {sub_answer}"
        documents, triples, retrieved_entities, exploration_log = await self._explore(step_objective)
        
        extractor_inputs = [ExtractionInput(question=sub_question, raw_data=d) for d in documents]
        if extractor_inputs:
            extracted, extractor_log = await execute_role(
                client=self.client, role=EXTRACTOR, input_data=extractor_inputs,
                interaction_memory=self.interaction_memory, n=1
            )
            info = []
            for item in sum(extracted, []):
                if hasattr(item, 'relevant_information'):
                    info.extend(item.relevant_information)
        else:
            info = []
            extractor_log = {}
        
        all_info = info + [str(t) for t in triples]
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_info)
        
        correction_input = SelfCorrectionInput(question=sub_question, proposed_answer=sub_answer, context=context)
        corrections, qa_log = await execute_role(
            client=self.client, role=SELF_CORRECTOR, input_data=correction_input,
            interaction_memory=self.interaction_memory, n=self.kwargs.get('n', 1)
        )
        
        return GenerationResult(
            answers=corrections, retrieved_triples=triples,
            retrieved_entities=retrieved_entities,
            information_items=all_info,
            log_data=merge_logs(exploration_log, extractor_log, qa_log),
        )
    
    async def generate_synthesis(self, user_question: str) -> GenerationResult:
        """Generate reasoning synthesis."""
        memory = self.working_memory.format_textual_memory()
        context = format_context(memory=memory)
        
        synth_input = ReasoningSynthesizeInput(question=user_question, context=context)
        outputs, reasoning_log = await execute_role(
            client=self.client, role=REASONING_SYNTHESIZER, input_data=synth_input,
            interaction_memory=self.interaction_memory, n=self.kwargs.get('n', 1)
        )
        
        return GenerationResult(answers=outputs, log_data=reasoning_log)
    
    def update_working_memory(self, result: GenerationResult) -> None:
        """Update working memory with generation results."""
        for item in result.information_items:
            self.working_memory.add_textual_memory(item, source=SourceType.RETRIEVAL)

        entity_dict = {e.qid: e for e in result.retrieved_entities}
        self.working_memory.entity_dict.update(entity_dict)
        for entity in result.retrieved_entities:
            self.working_memory.add_node_to_graph_memory(entity)
        for triple in result.retrieved_triples:
            self.working_memory.add_edge_to_graph_memory(triple)
    
    # =========================================================================
    # Retrieval pipeline
    # =========================================================================
    
    async def _explore(self, question: str) -> Tuple[List[str], List[WikiTriple], List[WikidataEntity], Dict]:
        """Combined web + KB retrieval pipeline.
        
        1. Generate search queries from question
        2. Web search for each query (parallel)
        3. Entity linking + Wikidata triple retrieval
        4. Prune triples and entities
        5. Return (documents, triples, entities, log_data)
        """
        query_input = QueryGeneratorInput(input_text=question)
        responses, query_log = await execute_role(
            client=self.client, role=QUERY_GENERATOR, input_data=query_input,
            interaction_memory=self.interaction_memory, n=1
        )
        queries = responses[0].queries if responses else [question]

        tasks = [self._retrieve_from_web(q) for q in queries]
        web_results = await asyncio.gather(*tasks)
        documents = list(set(sum(web_results, [])))
        
        tasks = [self._retrieve_from_kb(q) for q in queries]
        results = await asyncio.gather(*tasks)
        triples = []
        entities = []
        kb_log = {}
        for t, e, l in results:
            triples.extend(t)
            entities.extend(e)
            kb_log = merge_logs(kb_log, l)
        entities = list(set(entities))
        # Get details for all entities
        assert self.wikidata_client is not None and isinstance(self.wikidata_client, WikidataClient), "WikidataClient must be provided"
        all_entities = [self.working_memory.entity_dict.get(e.qid) if self.working_memory.entity_dict.get(e.qid) else e 
                        for e in entities]
        all_entities = self.wikidata_client.enrich_entities(all_entities, get_details=True)
        kb_documents = []
        for entity in all_entities:
            if hasattr(entity, 'wikipedia_content') and entity.wikipedia_content:
                kb_documents.append(entity.wikipedia_content)
        kb_documents = list(set(kb_documents))

        if self.reranker and self.kwargs.get("rerank_kb_documents", True) and kb_documents:
            top_kb = self.reranker.rerank(question, kb_documents)
        else:
            top_kb = kb_documents
        documents = list(set(documents + top_kb))
        all_log = merge_logs(query_log, kb_log)
        return documents, triples, all_entities, all_log
    
    async def _retrieve_from_web(self, query: str) -> List[str]:
        """Retrieve documents from web or corpus."""
        top_k = self.kwargs.get('top_k_websearch', 5)
        
        if hasattr(self.retriever, 'asearch'):
            output = await self.retriever.asearch(query, top_k=top_k)
            return [f"{r.title}\n{r.snippet}\n{r.full_text}" for r in output.results] if output.is_success else []
        elif hasattr(self.retriever, 'search'):
            output = self.retriever.search(query, top_k=top_k)
            return [f"{r.title}\n{r.snippet}\n{r.full_text}" for r in output.results] if output.is_success else []
        elif hasattr(self.retriever, 'retrieve'):
            contents, _ = self.retriever.retrieve(query, top_k=top_k)
            return contents if isinstance(contents, list) else [contents]
        else:
            logger.warning(f"Unknown retriever type: {type(self.retriever)}")
            return []
    
    async def _retrieve_from_kb(self, question: str) -> Tuple[List[WikiTriple], List, Dict]:
        """Retrieve triples from Wikidata knowledge base."""
        entity_linking_method = self.kwargs.get('entity_linking_method', 'llm')
        top_k_entities = self.kwargs.get('top_k_entities', 1)
        n_hops = self.kwargs.get('n_hops', 1)
        assert self.wikidata_client is not None and isinstance(self.wikidata_client, WikidataClient), "WikidataClient must be provided"

        known_entities = filter_entities_relevant_to_text(
            [e for e in self.working_memory.entity_dict.values() if isinstance(e, WikidataEntity)],
            question,
        )
        if entity_linking_method == "azure":
            entities, entity_dict, link_log = await link_entities_azure(
                question, self.wikidata_client,
                endpoint=self.kwargs.get('azure_endpoint'),
                key=self.kwargs.get('azure_key'),
            )
        else:
            entities, entity_dict, link_log = await link_entities_llm(
                self.client, question, self.wikidata_client,
                top_k_entities=top_k_entities,
                interaction_memory=self.interaction_memory,
                known_entities=known_entities,
                reranker=self.reranker,
            )

        # collect known question entities
        question_entities = [self.working_memory.entity_dict.get(qid) for qid in entity_dict.values()
                             if self.working_memory.entity_dict.get(qid)]
        question_entities = set(question_entities + entities)
        question_entities = [e for e in question_entities if isinstance(e, WikidataEntity)]
        qids = [e.qid for e in question_entities]
        if not qids:
            return [], [], link_log
        # retrieve triples from Wikidata
        triples = await self.wikidata_client.aget_k_hop_triples(
            qids, k=n_hops, bidirectional=True, enrich=True
        )
        if isinstance(triples, list) and triples and isinstance(triples[0], list):
            triples = sum(triples, [])
        triples = list(set(triples)) if triples else []
        all_entities = entities # keep all entities from the question
        if triples:
            # triples, prune_log = await self._prune_triples(question, triples)
            all_entities = list(set(all_entities + self._collect_entities_from_triples(triples)))
            # link_log = merge_logs(link_log, prune_log)
        return triples, all_entities, link_log
    
    async def _prune_triples(self, question: str, triples: List[WikiTriple]) -> Tuple[List[WikiTriple], Dict]:
        """Prune triples to keep only question-relevant ones."""
        if not triples:
            return [], {}
        chunks = [
            triples[i : i + _PRUNE_TRIPLES_BATCH_SIZE]
            for i in range(0, len(triples), _PRUNE_TRIPLES_BATCH_SIZE)
        ]
        inputs = [
            TriplePruneInput(question=question, triples=[str(t) for t in chunk])
            for chunk in chunks
        ]
        responses, log = await execute_role(
            client=self.client, role=TRIPLE_PRUNER, input_data=inputs,
            interaction_memory=self.interaction_memory, n=1
        )
        # List input → list of per-item parsed outputs (each item is n completions).
        kept: List[WikiTriple] = []
        for chunk, out_list in zip(chunks, responses):
            out = out_list[0] if out_list else None
            if not out or not hasattr(out, 'keep_indices'):
                kept.extend(chunk)
                continue
            keep = set(out.keep_indices)
            kept.extend(chunk[i] for i in range(len(chunk)) if i in keep)
        kept = list(set(kept))
        return kept, log

    @staticmethod
    def _collect_entities_from_triples(triples: List[WikiTriple]) -> List[WikidataEntity]:
        """Collect unique WikidataEntity from triples."""
        seen = set()
        entities = []
        for t in triples:
            if hasattr(t.subject, 'qid') and t.subject.qid not in seen:
                seen.add(t.subject.qid)
                entities.append(t.subject)
            if hasattr(t.object, 'qid') and t.object.qid not in seen:
                seen.add(t.object.qid)
                entities.append(t.object)
        return entities
