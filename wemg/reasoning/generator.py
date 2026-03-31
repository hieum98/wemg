"""Node generation pipeline: retrieval + LLM reasoning for creating reasoning nodes."""

import asyncio
import logging
import time
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
_PRUNE_TRIPLES_BATCH_SIZE = 16


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

    def _build_context(
        self,
        retrieval_info: List[str] = None,
        reasoning_trace: str = None,
    ) -> str:
        """Build LLM context from textual memory and retrieval."""
        memory = self.working_memory.format_textual_memory()
        return format_context(
            memory=memory,
            retrieval_info=retrieval_info,
            reasoning_trace=reasoning_trace,
        )
    
    async def generate_answer(self, question: str, should_explore: bool = True) -> GenerationResult:
        """Generate answer via retrieval + extraction + LLM answer generation.
        
        Pipeline:
        1. If should_explore: run explore() to get web results + Wikidata triples
        2. Extract relevant info from web results using Extractor role
        3. Build context from textual memory + graph memory + extracted info + KB triples
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
        context = self._build_context(retrieval_info=all_info)
        
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
    
    async def generate_subquestion(self, user_question: str, intermediate_answer: str = None) -> Tuple[List[str], bool, Dict]:
        """Generate subquestions. Returns (subquestions, should_direct_answer, log_data)."""
        context = self._build_context()
        subq_input = SubquestionGenerationInput(question=user_question, context=context, intermediate_answer=intermediate_answer)
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
        context = self._build_context()
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
        context = self._build_context(retrieval_info=all_info)
        
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
        context = self._build_context()
        
        synth_input = ReasoningSynthesizeInput(question=user_question, context=context)
        outputs, reasoning_log = await execute_role(
            client=self.client, role=REASONING_SYNTHESIZER, input_data=synth_input,
            interaction_memory=self.interaction_memory, n=self.kwargs.get('n', 1)
        )
        
        return GenerationResult(answers=outputs, log_data=reasoning_log)
    
    def update_working_memory(self, result: GenerationResult, *, source_step: int = None, hop_depth: int = None) -> None:
        """Update working memory with generation results."""
        for item in result.information_items:
            self.working_memory.add_textual_memory(item, source=SourceType.RETRIEVAL, hop_depth=hop_depth)

        self.working_memory.register_retrieved_triples(
            result.retrieved_triples, result.retrieved_entities
        )
    
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
        q_short = question[:120].replace("\n", " ")
        t_explore_start = time.perf_counter()
        logger.info("PROFSTEP explore stage=start question=%s", q_short)

        t0 = time.perf_counter()
        query_input = QueryGeneratorInput(input_text=question)
        responses, query_log = await execute_role(
            client=self.client, role=QUERY_GENERATOR, input_data=query_input,
            interaction_memory=self.interaction_memory, n=1
        )
        queries = responses[0].queries if responses else [question]
        queries = list(set(queries))
        logger.info(
            "PROFSTEP explore stage=query_generator elapsed_ms=%.1f queries=%d question=%s",
            (time.perf_counter() - t0) * 1000.0,
            len(queries),
            q_short,
        )

        t0 = time.perf_counter()
        tasks = [self._retrieve_from_web(q) for q in queries]
        web_results = await asyncio.gather(*tasks)
        documents = list(set(sum(web_results, [])))
        logger.info(
            "PROFSTEP explore stage=web_retrieval elapsed_ms=%.1f docs=%d question=%s",
            (time.perf_counter() - t0) * 1000.0,
            len(documents),
            q_short,
        )
        
        t0 = time.perf_counter()
        tasks = [self._retrieve_from_kb(q) for q in queries]
        results = await asyncio.gather(*tasks)
        triples = []
        entities = []
        kb_log = {}
        for t, e, l in results:
            triples.extend(t)
            entities.extend(e)
            kb_log = merge_logs(kb_log, l)
        logger.info(
            "PROFSTEP explore stage=kb_retrieval elapsed_ms=%.1f triples=%d entities=%d question=%s",
            (time.perf_counter() - t0) * 1000.0,
            len(triples),
            len(entities),
            q_short,
        )

        entities = list(set(entities))
        triples = list(set(triples))
        # prune triples
        triples, prune_log = await self._prune_triples(question, triples)
        entities = list(set(entities + self._collect_entities_from_triples(triples)))

        assert self.wikidata_client is not None and isinstance(self.wikidata_client, WikidataClient), "WikidataClient must be provided"
        all_entities = [self.working_memory.entity_dict.get(e.qid) if self.working_memory.entity_dict.get(e.qid) else e 
                        for e in entities]
        t0 = time.perf_counter()
        all_entities = await self.wikidata_client.aenrich_entities(all_entities, get_details=True)
        logger.info(
            "PROFSTEP explore stage=enrich_entities elapsed_ms=%.1f entities=%d question=%s",
            (time.perf_counter() - t0) * 1000.0,
            len(all_entities),
            q_short,
        )

        kb_documents = []
        for entity in all_entities:
            if hasattr(entity, 'wikipedia_content') and entity.wikipedia_content:
                kb_documents.append(entity.wikipedia_content)
        kb_documents = list(set(kb_documents))

        t0 = time.perf_counter()
        if self.reranker and self.kwargs.get("rerank_kb_documents", True) and kb_documents:
            top_kb = self.reranker.rerank(question, kb_documents)
        else:
            top_kb = kb_documents
        logger.info(
            "PROFSTEP explore stage=rerank_kb_documents elapsed_ms=%.1f kb_docs=%d top_docs=%d question=%s",
            (time.perf_counter() - t0) * 1000.0,
            len(kb_documents),
            len(top_kb),
            q_short,
        )
        documents = list(set(documents + top_kb))
        all_log = merge_logs(query_log, kb_log, prune_log)
        logger.info(
            "PROFSTEP explore stage=done elapsed_ms=%.1f docs=%d triples=%d entities=%d question=%s",
            (time.perf_counter() - t_explore_start) * 1000.0,
            len(documents),
            len(triples),
            len(all_entities),
            q_short,
        )
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
        """Retrieve triples from Wikidata knowledge base.

        Graph-aware enhancements:
        - Expands the query QID set with underexplored graph neighbours.
        - Filters out triples already present in the graph before pruning.
        """
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

        question_entities = [self.working_memory.entity_dict.get(qid) for qid in entity_dict.values()
                             if self.working_memory.entity_dict.get(qid)]
        question_entities = set(question_entities + entities)
        question_entities = [e for e in question_entities if isinstance(e, WikidataEntity)]
        qids = [e.qid for e in question_entities]
        if not qids:
            return [], [], link_log

        neighbor_qids = self.working_memory.get_underexplored_neighbor_qids(qids)
        all_qids = list(dict.fromkeys(qids + neighbor_qids))

        # Iterative 1-hop + rerank loop.
        # Each iteration calls aget_k_hop_triples(k=1) which always uses the fast
        # batched SPARQL path (_get_k_hop_multi_seed_one_hop) instead of the slow
        # per-entity BFS triggered by k>1.
        TOP_K_FRONTIER = 64
        MAX_FRONTIER = 500

        current_qids = list(all_qids)
        all_triples: List[WikiTriple] = []
        all_visited_qids: set = set(current_qids)

        for hop in range(n_hops):
            if not current_qids:
                break

            hop_results = await self.wikidata_client.aget_k_hop_triples(
                current_qids, k=1, bidirectional=False, enrich=True
            )
            # Flatten per-seed lists into a single deduplicated list.
            hop_triples: List[WikiTriple] = []
            if isinstance(hop_results, list) and hop_results and isinstance(hop_results[0], list):
                for per_seed in hop_results:
                    hop_triples.extend(per_seed)
            else:
                hop_triples = list(hop_results) if hop_results else []
            hop_triples = self._deduplicate_triples(hop_triples)

            if not hop_triples:
                break

            # Between hops: rerank to prune the frontier to TOP_K_FRONTIER triples.
            # Skip on the last hop — full pruning happens later in _prune_triples.
            if self.reranker is not None and hop < n_hops - 1:
                triple_docs = [str(t) for t in hop_triples]
                sorted_indices, _ = await asyncio.to_thread(
                    self.reranker.get_scores, question, triple_docs
                )
                hop_triples = [hop_triples[i] for i in sorted_indices[:TOP_K_FRONTIER]]

            all_triples.extend(hop_triples)

            # Extract object QIDs from this hop's triples as seeds for the next hop.
            next_qids: set = set()
            for t in hop_triples:
                if isinstance(t.object, WikidataEntity) and t.object.qid:
                    next_qids.add(t.object.qid)
            next_qids -= all_visited_qids
            if len(next_qids) > MAX_FRONTIER:
                next_qids = set(list(next_qids)[:MAX_FRONTIER])
            current_qids = list(next_qids)
            all_visited_qids.update(current_qids)

        triples = self._deduplicate_triples(all_triples)
        return triples, entities, link_log

    def _stage_a_prune_triples(self, question: str, triples: List[WikiTriple]) -> List[WikiTriple]:
        """Stage A: reranker score filtering with delta and top-k cap."""
        if not triples or not self.reranker:
            return triples

        triple_docs = [str(t) for t in triples]
        sorted_indices, scores = self.reranker.get_scores(question, triple_docs)
        if not sorted_indices or not scores:
            return triples

        delta = float(self.kwargs.get("triple_pruning_delta", 0.05))
        top_k = int(self.kwargs.get("triple_pruning_top_k", 64))
        if top_k <= 0:
            top_k = len(triples)

        top_score = scores[0]
        kept_indices = [
            idx
            for idx, score in zip(sorted_indices, scores)
            if score >= (top_score - delta)
        ]
        if not kept_indices:
            kept_indices = [sorted_indices[0]]
        if len(kept_indices) > top_k:
            kept_indices = kept_indices[:top_k]

        return [triples[i] for i in kept_indices]
    
    async def _prune_triples(self, question: str, triples: List[WikiTriple]) -> Tuple[List[WikiTriple], Dict]:
        """Prune triples with Stage A reranker, then Stage B TRIPLE_PRUNER."""
        if not triples:
            return [], {}
        stage_a_kept = self._stage_a_prune_triples(question, triples)

        chunks = [
            stage_a_kept[i : i + _PRUNE_TRIPLES_BATCH_SIZE]
            for i in range(0, len(stage_a_kept), _PRUNE_TRIPLES_BATCH_SIZE)
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
    def _deduplicate_triples(triples: List[WikiTriple]) -> List[WikiTriple]:
        """Return triples with duplicates removed (by subject QID, relation PID, object QID/str)."""
        seen: set = set()
        result: List[WikiTriple] = []
        for t in triples:
            key = (
                t.subject.qid,
                t.relation.pid,
                t.object.qid if isinstance(t.object, WikidataEntity) else str(t.object),
            )
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

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
