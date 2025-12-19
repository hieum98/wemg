import math
import random
from typing import Dict, List, Optional
from pyparsing import Union
from typing_extensions import TypedDict
import asyncio

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search, wikidata
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
# from wemg.runners.procerduces.extraction import extract_info_from_raw_text
# from wemg.runners.procerduces.generation import answer_question
from wemg.runners.procerduces.base_role_excercution import execute_role
from wemg.runners.procerduces.retrieval import explore
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.preprocessing import format_context


class MCTSReasoningNode(BaseReasoningNode):
    def __init__(
            self,
            node_state: NodeState,
            parent: Optional['MCTSReasoningNode'] = None,
            children: Optional[list['MCTSReasoningNode']] = None,
            max_depth: int = 10,
    ):
        super().__init__(node_state=node_state, parent=parent, children=children, max_depth=max_depth)
        self.children: List["MCTSReasoningNode"]
        self.parent: Optional["MCTSReasoningNode"]
        self.node_state = node_state  # type: NodeState
        self.value: float = 0.0
        self.visits: int = 0

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return average_reward + exploration_weight * exploration_term
    
    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def _merge_results(self, results: List[tuple]) -> tuple:
        """Merge results from multiple concurrent tasks.
        
        Each result can be either:
        - 5-tuple: (nodes, triples, entity_dict, property_dict, log)
        - 2-tuple: (nodes, log)
        """
        all_nodes = []
        all_triples = []
        all_entity_dict = {}
        all_property_dict = {}
        all_logs = {}
        
        for result in results:
            if len(result) == 5:
                nodes, triples, entity_dict, property_dict, log = result
                all_triples.extend(triples)
                all_entity_dict.update(entity_dict)
                all_property_dict.update(property_dict)
            else:  # len(result) == 2
                nodes, log = result
            
            all_nodes.extend(nodes)
            for key in log:
                all_logs.setdefault(key, []).extend(log[key])
        
        return all_nodes, all_triples, all_entity_dict, all_property_dict, all_logs

    def generate_children(
            self,
            llm_agent: BaseLLMAgent, 
            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
            working_memory: WorkingMemory, 
            interaction_memory: Optional[InteractionMemory] = None,
            **node_generation_kwargs
            ):
        """Generate children nodes for this node."""
        common_kwargs = {
            'llm_agent': llm_agent,
            'working_memory': working_memory,
            'interaction_memory': interaction_memory,
            **node_generation_kwargs
        }
        retriever_kwargs = {**common_kwargs, 'retriever_agent': retriever_agent}
        
        if self.depth > self.max_depth:
            results = asyncio.run(asyncio.gather(
                self._generate_final_answer(**retriever_kwargs)
            ))
        elif self.node_state.node_type == NodeType.USER_QUESTION:
            results = asyncio.run(asyncio.gather(
                self._generate_final_answer(**retriever_kwargs),
                self._generate_subqa(**retriever_kwargs)
            ))
        elif self.node_state.node_type == NodeType.SUB_QA_NODE:
            results = asyncio.run(asyncio.gather(
                self._rephrase(**common_kwargs),
                self._generate_subqa(**retriever_kwargs),
                self._self_correct(**retriever_kwargs),
                self._strengthen(**common_kwargs)
            ))
        elif self.node_state.node_type == NodeType.REPHRASED_QUESTION_NODE:
            results = asyncio.run(asyncio.gather(
                self._generate_subqa(**retriever_kwargs)
            ))
        elif self.node_state.node_type == NodeType.SELF_CORRECTED_NODE:
            results = asyncio.run(asyncio.gather(
                self._generate_subqa(**retriever_kwargs),
                self._strengthen(**common_kwargs)
            ))
        elif self.node_state.node_type == NodeType.SYNTHESIS_NODE:
            results = asyncio.run(asyncio.gather(
                self._generate_subqa(**retriever_kwargs)
            ))
        else:
            raise ValueError(f"Unsupported node type: {self.node_state.node_type}")
        nodes, retrieved_triples, entity_dict, property_dict, to_log_data = self._merge_results(results)
        
        # Update interaction memory
        if interaction_memory and to_log_data:
            for key, value in to_log_data.items():
                model_input, model_output = zip(*value)
                interaction_memory.log_turn(
                    role=key,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )
        return nodes, retrieved_triples, entity_dict, property_dict
    
    async def _generate_answer(self, 
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            question: str,
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ):
        """Generate answer given question and memory."""
        in_memory_entities = list(set(working_memory.entity_dict.values()))
        in_memory_relations = list(set(working_memory.property_dict.values()))
        
        # Explore external resources
        retrieved_documents, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=question,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=node_generation_kwargs.get('top_k_websearch', 5),
            top_k_entities=node_generation_kwargs.get('top_k_entities', 1),
            top_k_properties=node_generation_kwargs.get('top_k_properties', 1),
            n_hops=node_generation_kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=node_generation_kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=interaction_memory
        )

        # Extract information from web search results
        extractor_input = [roles.extractor.ExtractionInput(question=question, raw_data=data) for data in retrieved_documents]
        extracted_info_from_websearch, extractor_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_input,
            interaction_memory=interaction_memory,
            n=1
        )
        all_extraction: List[roles.extractor.ExtractionOutput] = sum([], extracted_info_from_websearch)
        info_from_websearch = []
        for item in all_extraction:
            if item.decision == "relevant":
                info_from_websearch.extend(item.information)

        # Build context
        info_from_kb = [str(t) for t in retrieved_triples]
        all_retrieved_info = info_from_websearch + info_from_kb
        # reasoning_trace = self.get_reasoning_trace()
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)

        qa_input = roles.generator.AnswerGenerationInput(question=question, context=context)
        answers, qa_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.AnswerGenerator(),
            input_data=qa_input,
            interaction_memory=interaction_memory,
            n=node_generation_kwargs.get('n', 1)
        )
        answers: List[roles.generator.AnswerGenerationOutput]

        # Process_log
        all_log_keys = set(list(qa_log.keys()) + list(extractor_log.keys()) + list(exploration_log.keys()))
        to_log_data = {key: qa_log.get(key, []) + extractor_log.get(key, []) + exploration_log.get(key, []) \
                       for key in all_log_keys}
        return answers, retrieved_triples, entity_dict, property_dict, to_log_data
    
    async def _generate_final_answer(self, 
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ):
        answers, retrieved_triples, entity_dict, property_dict, to_log_data = await self._generate_answer(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=self.user_question,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            **node_generation_kwargs
        )
        nodes: List[MCTSReasoningNode] = []
        for answer in answers:
            node_state = NodeState(node_type=NodeType.FINAL_ANSWER,
                                   content={
                                       "user_question": self.user_question,
                                       "final_answer": answer.answer,
                                       "concise_answer": answer.concise_answer,
                                       "reasoning": answer.reasoning,
                                   })
            node = MCTSReasoningNode(node_state=node_state, max_depth=self.max_depth)
            nodes.append(node)
        return nodes, retrieved_triples, entity_dict, property_dict, to_log_data

    async def _generate_subqa(self,
                        llm_agent: BaseLLMAgent, 
                        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                        working_memory: WorkingMemory, 
                        interaction_memory: Optional[InteractionMemory] = None,
                        **node_generation_kwargs
                        ):
        """Generate sub-qa pairs to advance the reasoning process."""
        if self.node_state.node_type == NodeType.REPHRASED_QUESTION_NODE:
            subquestion = self.node_state.content['sub_question']
            should_direct_answer = 0
            subq_log = {}
        else:
            # reasoning_trace = self.get_reasoning_trace()
            memory = working_memory.format_textual_memory()
            context = format_context(memory=memory)
            subq_gen_input = roles.generator.SubquestionGenerationInput(question=self.user_question, context=context)
            subquestions, subq_log = await execute_role(
                llm_agent=llm_agent,
                role=roles.generator.SubquestionGenerator(),
                input_data=subq_gen_input,
                interaction_memory=interaction_memory,
                n=node_generation_kwargs.get('n', 1)
            )
            subquestions: List[roles.generator.SubquestionGenerationOutput]
            # Random choice a subq with is_answerable==False
            unanswerable_subquestions = [subquestion for subquestion in subquestions if not subquestion.is_answerable]
            subquestion = None
            if unanswerable_subquestions:
                subquestion = random.choice(unanswerable_subquestions).subquestion
            should_direct_answer = [subquestion.is_answerable for subquestion in subquestions if subquestion]
            should_direct_answer = sum(should_direct_answer) / len(should_direct_answer) if should_direct_answer else 0

        if should_direct_answer > 0.5 or not subquestion:
            nodes, retrieved_triples, entity_dict, property_dict, answer_log = await self._generate_final_answer(
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                question=self.user_question,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                **node_generation_kwargs
            )
        else:
            answers, retrieved_triples, entity_dict, property_dict, answer_log = await self._generate_answer(
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                question=subquestion,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                **node_generation_kwargs
            )
            nodes: List[MCTSReasoningNode] = []
            for answer in answers:
                node_state = NodeState(
                    node_type=NodeType.SUB_QA_NODE,
                    content={
                        'user_question': self.user_question,
                        'sub_question': subquestion,
                        'sub_answer': answer.answer,
                        }
                )
                nodes.append(MCTSReasoningNode(node_state=node_state, max_depth=self.max_depth))

        all_log_keys = set(list(answer_log.keys()) + list(subq_log.keys()))
        to_log_data = {key: answer_log.get(key, []) + subq_log.get(key, []) for key in all_log_keys}
        return nodes, retrieved_triples, entity_dict, property_dict, to_log_data
    
    async def _rephrase(self,
                llm_agent: BaseLLMAgent, 
                working_memory: WorkingMemory, 
                interaction_memory: Optional[InteractionMemory] = None,
                **node_generation_kwargs
                  ):
        """Generate rephrased question nodes."""
        if self.node_state.node_type == NodeType.USER_QUESTION:
            question = self.user_question
        elif self.node_state.node_type == NodeType.SUB_QA_NODE:
            question = self.node_state.content['sub_question']
        else:
            raise ValueError(f"Rephrase not supported for node type: {self.node_state.node_type}")
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory)
        rephrase_input = roles.generator.QuestionRephraserInput(context=context, original_question=question)
        rephrased_question, rephrase_log = await execute_role(
                llm_agent=llm_agent,
                role=roles.generator.QuestionRephraser(),
                input_data=rephrase_input,
                interaction_memory=interaction_memory,
                n=node_generation_kwargs.get('n', 1)
            )
        rephrased_question: List[roles.generator.QuestionRephraserOutput]
        nodes = []
        for item in rephrased_question:
            node_state = NodeState(
                node_type=NodeType.REPHRASED_QUESTION_NODE,
                content={
                    'user_question': self.user_question,
                    'sub_question': item.rephrased_question
                }
            )
            nodes.append(MCTSReasoningNode(node_state=node_state, max_depth=self.max_depth))
        return nodes, rephrase_log
    
    async def _self_correct(self,
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ):
        """Generate self-corrected question nodes."""
        sub_question = self.node_state.content.get('sub_question')
        sub_answer = self.node_state.content.get('sub_answer')
        if not sub_question or not sub_answer:
            return [], {}
        
        step_objective = f"Verify and correct the answer for the sub-question: {sub_question}.\nProposed answer: {sub_answer}"
        in_memory_entities = list(set(working_memory.entity_dict.values()))
        in_memory_relations = list(set(working_memory.property_dict.values()))
        
        # Explore external resources
        retrieved_documents, retrieved_triples, entity_dict, property_dict, exploration_log = await explore(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=step_objective,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=node_generation_kwargs.get('top_k_websearch', 5),
            top_k_entities=node_generation_kwargs.get('top_k_entities', 1),
            top_k_properties=node_generation_kwargs.get('top_k_properties', 1),
            n_hops=node_generation_kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=node_generation_kwargs.get('use_question_for_graph_retrieval', True),
            interaction_memory=interaction_memory
        )

        # Extract information from web search results
        extractor_input = [roles.extractor.ExtractionInput(question=sub_question, raw_data=data) for data in retrieved_documents]
        extracted_info_from_websearch, extractor_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.extractor.Extractor(),
            input_data=extractor_input,
            interaction_memory=interaction_memory,
            n=1
        )
        all_extraction: List[roles.extractor.ExtractionOutput] = sum([], extracted_info_from_websearch)
        info_from_websearch = []
        for item in all_extraction:
            if item.decision == "relevant":
                info_from_websearch.extend(item.information)
        
        # Build context
        info_from_kb = [str(t) for t in retrieved_triples]
        all_retrieved_info = info_from_websearch + info_from_kb
        # reasoning_trace = self.get_reasoning_trace()
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info)

        self_correct_input = roles.generator.SelfCorrectionInput(question=sub_question, proposed_answer=sub_answer, context=context)
        answers, qa_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.SelfCorrector(),
            input_data=self_correct_input,
            interaction_memory=interaction_memory,
            n=node_generation_kwargs.get('n', 1)
        )
        answers: List[roles.generator.SelfCorrectionOutput]
        nodes: List[MCTSReasoningNode] = []
        for answer in answers:
            node_state = NodeState(node_type=NodeType.SELF_CORRECTED_NODE,
                                   content={
                                       'user_question': self.user_question,
                                       'sub_question': sub_question,
                                       'sub_answer': answer.refined_answer
                                   })
            nodes.append(MCTSReasoningNode(node_state=node_state, max_depth=self.max_depth))

        # process log
        all_log_keys = set(list(qa_log.keys()) + list(extractor_log.keys()) + list(exploration_log.keys()))
        to_log_data = {key: qa_log.get(key, []) + extractor_log.get(key, []) + exploration_log.get(key, []) \
                       for key in all_log_keys}
        return nodes, retrieved_triples, entity_dict, property_dict, to_log_data
    
    async def _strengthen(self,
                    llm_agent: BaseLLMAgent, 
                    working_memory: WorkingMemory, 
                    interaction_memory: Optional[InteractionMemory] = None,
                    **node_generation_kwargs
                    ):
        """Generate strengthened reasoning"""
        # reasoning_trace = self.get_reasoning_trace()
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory)
        reasoner_input = roles.generator.ReasoningSynthesizeInput(question=self.user_question, context=context)
        reasoner_output, reasoning_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.ReasoningSynthesizer(),
            input_data=reasoner_input,
            interaction_memory=interaction_memory,
            n=node_generation_kwargs.get('n', 1)
        )
        reasoner_output: List[roles.generator.ReasoningSynthesizeOutput]
        nodes = []
        for output in reasoner_output:
            if output.is_answerable:
                node_state = NodeState(node_type=NodeType.FINAL_ANSWER,
                                       content={
                                           "user_question": self.user_question,
                                            "final_answer": output.step_conclusion,
                                       })
            else:
                node_state = NodeState(node_type=NodeType.SYNTHESIS_NODE,
                                    content={
                                        'user_question': self.user_question,
                                        'synthesized_reasoning': output.step_conclusion
                                    })
            nodes.append(MCTSReasoningNode(node_state=node_state, max_depth=self.max_depth))
        return nodes, reasoning_log

class MCTSSearchTree(TypedDict):
    root: MCTSReasoningNode
    explored_nodes: set[MCTSReasoningNode]


def select(tree: MCTSSearchTree, exploration_weight=1.0) -> List[MCTSReasoningNode]:
    """Select a path from the root to a leaf node using the UCT algorithm."""
    path = []
    node = tree['root']
    while True:
        # 1. if node is unvisited or terminal, return the path
        if not node.children:
            path.append(node)
            return path
        
        # 2. a node has children, but not all of them are explored, select a random unexplored child
        # Note: In this implementation, we need to explore all children before going deeper
        unexplored_children = [child for child in node.children if child.visits == 0]
        if unexplored_children:
            node = random.choice(unexplored_children)
            path.append(node)
            return path
        
        # 3. all children are explored, select the child with the highest UCT score
        uct_scores = [child.upper_confidence_bound(exploration_weight) for child in node.children]
        node = node.children[uct_scores.index(max(uct_scores))]
        path.append(node)


def expand(
        tree: MCTSSearchTree, 
        working_memory: WorkingMemory, 
        interaction_memory: Optional[InteractionMemory] = None
        ):
    """Expand the tree by generating children nodes for the selected node using the working memory and interaction memory."""
    selected_path = select(tree)
    leaf = selected_path[-1]

    