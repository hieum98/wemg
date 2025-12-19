import math
import random
from typing import Dict, List, Optional, Tuple
from pyparsing import Union
from typing_extensions import TypedDict
import asyncio

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search, wikidata
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
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

    def _merge_results(self, results: List[tuple]):
        """Merge results from multiple concurrent tasks.
        
        Each result can be either:
        - 5-tuple: (nodes, triples, entity_dict, property_dict, log)
        - 2-tuple: (nodes, log)
        """
        all_nodes: List[MCTSReasoningNode] = []
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
            is_cot_simulation: bool = False,
            **node_generation_kwargs
            ) -> List["MCTSReasoningNode"]:
        """Generate children nodes for this node and update working memory."""
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
        elif is_cot_simulation:
            # Simulate to get the answer by only generate subqa node
            results = asyncio.run(asyncio.gather(
                self._generate_subqa(**retriever_kwargs)
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
        
        # Update working memory with retrieved information
        working_memory.entity_dict.update(entity_dict)
        working_memory.property_dict.update(property_dict)
        
        # Add retrieved triples to graph memory
        for triple in retrieved_triples:
            working_memory.add_edge_to_graph_memory(triple)
        
        # Add node content to textual memory based on node type
        for child in nodes:
            if child.node_state.node_type == NodeType.SUB_QA_NODE:
                sub_q = child.node_state.content.get('sub_question', '')
                sub_a = child.node_state.content.get('sub_answer', '')
                if sub_q and sub_a:
                    working_memory.add_textual_memory(
                        f"Q: {sub_q}; A: {sub_a}",
                        source=roles.extractor.SourceType.SYSTEM_PREDICTION
                    )
            elif child.node_state.node_type == NodeType.SELF_CORRECTED_NODE:
                sub_q = child.node_state.content.get('sub_question', '')
                sub_a = child.node_state.content.get('sub_answer', '')
                if sub_q and sub_a:
                    working_memory.add_textual_memory(
                        f"Q: {sub_q}; A: {sub_a}",
                        source=roles.extractor.SourceType.SYSTEM_PREDICTION
                    )
            elif child.node_state.node_type == NodeType.SYNTHESIS_NODE:
                synthesis = child.node_state.content.get('synthesized_reasoning', '')
                if synthesis:
                    working_memory.add_textual_memory(
                        f"{synthesis}",
                        source=roles.extractor.SourceType.SYSTEM_PREDICTION
                    )
        
        # Update interaction memory
        if interaction_memory and to_log_data:
            for key, value in to_log_data.items():
                model_input, model_output = zip(*value)
                interaction_memory.log_turn(
                    role=key,
                    user_input=list(model_input),
                    assistant_output=list(model_output)
                )
        return nodes
    
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
        if not node.children or node.is_terminal():
            path.append(node)
            return path
        
        # 2. a node has children, but not all of them are explored, select a random unexplored child
        # Note: In this implementation, we need to explore all children before going deeper
        unexplored_children = [child for child in node.children if child not in tree['explored_nodes']]
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
        selected_node: MCTSReasoningNode,
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory, 
        interaction_memory: Optional[InteractionMemory] = None,
        is_cot_simulation: bool = False,
        **node_generation_kwargs
        ) -> List[MCTSReasoningNode]:
    """Expand the tree by generating children nodes for the selected node."""
    # Skip expansion for terminal nodes
    if selected_node.is_terminal():
        tree['explored_nodes'].add(selected_node)
        return []
    
    # Generate children nodes - this also updates working memory
    children_nodes = selected_node.generate_children(
        llm_agent=llm_agent,
        retriever_agent=retriever_agent,
        working_memory=working_memory,
        interaction_memory=interaction_memory,
        is_cot_simulation=is_cot_simulation,
        **node_generation_kwargs
    )
    
    # Add children to the selected node by setting their parent
    # In anytree, setting child.parent automatically updates parent.children
    for child in children_nodes:
        child.parent = selected_node
    
    tree['explored_nodes'].add(selected_node)
    
    return children_nodes


def simulate(
        node: MCTSReasoningNode,
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory] = None,
        is_cot_simulation: bool = True,
        max_simulation_depth: int = 5,
        **node_generation_kwargs
        ) -> MCTSReasoningNode:
    """Simulate a random playout from the given node to a terminal state.
    
    This performs a rollout using chain-of-thought (CoT) simulation when is_cot_simulation=True,
    which generates only subqa nodes until reaching a final answer. When is_cot_simulation=False,
    it uses standard simulation by randomly selecting from all generated children node types.
    
    Working memory is updated automatically by generate_children() during the simulation.
    """
    current_node = node
    simulation_depth = 0
    
    while not current_node.is_terminal() and simulation_depth < max_simulation_depth:
        # Generate children - this also updates working memory internally
        children_nodes = current_node.generate_children(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            working_memory=working_memory,
            interaction_memory=interaction_memory,
            is_cot_simulation=is_cot_simulation,
            **node_generation_kwargs
        )
        
        if not children_nodes:
            # No children generated, terminate simulation
            break
        
        # Select a random child for simulation
        current_node = random.choice(children_nodes)
        
        # Link the child to parent for trajectory tracking
        current_node.parent = node if simulation_depth == 0 else current_node.parent
        
        simulation_depth += 1
    
    return current_node


async def evaluate(
        node: MCTSReasoningNode,
        llm_agent: BaseLLMAgent,
        interaction_memory: Optional[InteractionMemory] = None,
        golden_answer: Optional[str] = None
        ) -> float:
    """Evaluate a terminal node and return a reward score."""
    if node.node_state.node_type != NodeType.FINAL_ANSWER:
        # Non-terminal nodes get a default low reward
        return 0.1
    
    user_question = node.user_question
    final_answer = node.node_state.content.get('final_answer', '')
    
    # Use golden_answer from node if available
    correct_answer = golden_answer or node.golden_answer or "Not available"
    
    # Prepare evaluation input
    eval_input = roles.evaluator.AnswerEvaluationInput(
        user_question=user_question,
        system_answer=final_answer,
        correct_answer=correct_answer
    )
    
    # Execute evaluation
    eval_results, eval_log = await execute_role(
        llm_agent=llm_agent,
        role=roles.evaluator.Evaluator(),
        input_data=eval_input,
        interaction_memory=interaction_memory,
        n=1
    )
    
    if eval_results:
        eval_output: roles.evaluator.AnswerEvaluationOutput = eval_results[0]
        # Normalize rating from 0-10 to 0-1
        reward = eval_output.rating / 10.0
    else:
        # Default reward if evaluation fails
        reward = 0.5
    
    return reward


def evaluate_sync(
        node: MCTSReasoningNode,
        llm_agent: BaseLLMAgent,
        interaction_memory: Optional[InteractionMemory] = None,
        golden_answer: Optional[str] = None
        ) -> float:
    """Synchronous wrapper for the evaluate function."""
    return asyncio.run(evaluate(node, llm_agent, interaction_memory, golden_answer))


def mcts_search(
        question: str,
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        working_memory: WorkingMemory,
        interaction_memory: Optional[InteractionMemory] = None,
        num_iterations: int = 10,
        exploration_weight: float = 1.0,
        is_cot_simulation: bool = True,
        max_tree_depth: int = 10,
        max_simulation_depth: int = 5,
        golden_answer: Optional[str] = None,
        **node_generation_kwargs
        ) -> Tuple[Dict, MCTSSearchTree]:
    """Run Monte Carlo Tree Search to find the best reasoning path.
    
    The MCTS algorithm consists of four phases:
    1. Selection: Select a path from root to a leaf using UCT
    2. Expansion: Expand the selected leaf by generating children
    3. Simulation: Simulate a rollout from the expanded node to a terminal state
    4. Backpropagation: Update values along the path based on simulation result
    
    Args:
        question: The user's question (root node content)
        llm_agent: The language model agent
        retriever_agent: The retriever agent for external knowledge
        working_memory: The working memory containing context
        interaction_memory: Optional interaction memory for logging
        num_iterations: Number of MCTS iterations to run
        exploration_weight: Weight for exploration in UCT formula
        is_cot_simulation: If True, use chain-of-thought style simulation
        max_simulation_depth: Maximum depth for simulation rollouts
        golden_answer: Optional ground truth answer for evaluation
        **node_generation_kwargs: Additional arguments for node generation
        
    Returns:
        Tuple of (best terminal node content, explored search tree)
    """
    # Initialize the search tree
    root_node_state = NodeState(node_type=NodeType.USER_QUESTION, content={'user_question': question})
    root_node = MCTSReasoningNode(node_state=root_node_state, max_depth=max_tree_depth)
    tree: MCTSSearchTree = {
        'root': root_node,
        'explored_nodes': set()
    }
    
    best_terminal_node: Optional[MCTSReasoningNode] = None
    best_reward: float = -float('inf')
    
    for iteration in range(num_iterations):
        # Phase 1: Selection - Select path from root to leaf
        path = select(tree, exploration_weight)
        selected_node = path[-1]
        
        # Phase 2: Expansion - Generate children for selected node
        # Working memory is updated automatically by expand/generate_children
        if not selected_node.is_terminal():
            children = expand(
                tree=tree,
                selected_node=selected_node,
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                is_cot_simulation=False,  # Use full expansion, not CoT
                **node_generation_kwargs
            )
            
            # Select a child for simulation (prefer unexplored)
            if children:
                selected_node = random.choice(children)
        
        # Phase 3: Simulation - Rollout to terminal state
        # Working memory is updated automatically by simulate/generate_children
        if not selected_node.is_terminal():
            terminal_node = simulate(
                node=selected_node,
                llm_agent=llm_agent,
                retriever_agent=retriever_agent,
                working_memory=working_memory,
                interaction_memory=interaction_memory,
                is_cot_simulation=is_cot_simulation,
                max_simulation_depth=max_simulation_depth,
                **node_generation_kwargs
            )
        else:
            terminal_node = selected_node
        
        # Phase 4: Evaluation and Backpropagation
        reward = evaluate_sync(
            node=terminal_node,
            llm_agent=llm_agent,
            interaction_memory=interaction_memory,
            golden_answer=golden_answer
        )
        
        # Backpropagate reward through the path
        for node in path:
            node.backpropagate(reward)

        # Sync the working memory after each iteration
        working_memory.synchronize_memory(
            llm_agent=llm_agent,
            question=question,
            interaction_memory=interaction_memory
        )
        
        # Track best terminal node
        if terminal_node.is_terminal() and reward > best_reward:
            best_reward = reward
            best_terminal_node = terminal_node
    
    return best_terminal_node.node_state.content, tree


def get_answer(tree: MCTSSearchTree,
               llm_agent: BaseLLMAgent,
               interaction_memory: Optional[InteractionMemory] = None
               ) -> str:
    """Get the final answer from the explored tree."""
    terminal_nodes: List[MCTSReasoningNode] = []
    def collect_terminals(node: MCTSReasoningNode):
        if node.is_terminal():
            terminal_nodes.append(node)
        for child in node.children:
            collect_terminals(child)
    collect_terminals(tree['root'])
    
    if not terminal_nodes:
        return "No final answer found."

    all_answers = []
    for node in terminal_nodes:
        final_answer = node.node_state.content.get('final_answer', '')
        if final_answer:
            all_answers.append(final_answer)
    try:
        answer_synthesis_input = roles.evaluator.FinalAnswerSynthesisInput(
            question=tree['root'].user_question,
            candidate_answers=all_answers
        )
        synthesis_results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.FinalAnswerSynthesizer(),
            input_data=answer_synthesis_input,
            interaction_memory=interaction_memory,
            n=1
        ))
        if synthesis_results:
            synthesis_output: roles.evaluator.FinalAnswerSynthesisOutput = synthesis_results[0]
            answer = synthesis_output.final_answer
            short_answer = synthesis_output.concise_answer
        else:
            raise ValueError("No synthesis results returned.")
    except:
        # Fallback: Majority vote
        majority_vote_input = roles.evaluator.MajorityVoteInput(
            question=tree['root'].user_question,
            answers=all_answers
        )
        vote_results, log = asyncio.run(execute_role(
            llm_agent=llm_agent,
            role=roles.evaluator.MajorityVoter(),
            input_data=majority_vote_input,
            interaction_memory=interaction_memory,
            n=1
        ))
        if vote_results:
            vote_output: roles.evaluator.MajorityVoteOutput = vote_results[0]
            answer = vote_output.final_answer
            short_answer = vote_output.concise_answer
        else:
            answer = "Unable to determine final answer."
            short_answer = answer
    
    # Process logging
    if log and interaction_memory:
        for key, value in log.items():
            model_input, model_output = zip(*value)
            interaction_memory.log_turn(
                role=key,
                user_input=list(model_input),
                assistant_output=list(model_output)
            )
    return answer, short_answer

