import math
import random
from typing import List, Optional
from pyparsing import Union
from typing_extensions import TypedDict
import pydantic

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents.tools import web_search
from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState
# from wemg.runners.procerduces.extraction import extract_info_from_raw_text
# from wemg.runners.procerduces.generation import answer_question
from wemg.runners.procerduces.base_role_excercution import execute_role
from wemg.runners.procerduces.retrieval import explore
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory
from wemg.utils.preprocessing import format_context


class MCTSNodeState(NodeState):
    pass


class MCTSReasoningNode(BaseReasoningNode):
    def __init__(
            self,
            node_state: MCTSNodeState,
            parent: Optional['MCTSReasoningNode'] = None,
            children: Optional[list['MCTSReasoningNode']] = None,
            max_depth: int = 10,
    ):
        super().__init__(node_state=node_state, parent=parent, children=children, max_depth=max_depth)
        self.children: List["MCTSReasoningNode"]
        self.parent: Optional["MCTSReasoningNode"]
        self.node_state = node_state  # type: MCTSNodeState
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

    def generate_children(
            self,
            llm_agent: BaseLLMAgent,
            working_memory: WorkingMemory, 
            interaction_memory: Optional[InteractionMemory] = None,
            **node_generation_kwargs
            ):
        """Generate children nodes for this node."""
    
    async def _generate_answer(self, 
                            llm_agent: BaseLLMAgent, 
                            retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
                            question: str,
                            working_memory: WorkingMemory, 
                            interaction_memory: Optional[InteractionMemory] = None,
                            **node_generation_kwargs
                            ) -> List[roles.generator.AnswerGenerationOutput]:
        """Generate children which is the answer node."""
        in_memory_entities = list(set(working_memory.entity_dict.values()))
        in_memory_relations = list(set(working_memory.property_dict.values()))
        
        # Explore external resources
        retrieved_documents, retrieved_triples, entity_dict, property_dict = await explore(
            llm_agent=llm_agent,
            retriever_agent=retriever_agent,
            question=question,
            entities=in_memory_entities,
            relations=in_memory_relations,
            top_k_websearch=node_generation_kwargs.get('top_k_websearch', 5),
            top_k_entities=node_generation_kwargs.get('top_k_entities', 1),
            top_k_properties=node_generation_kwargs.get('top_k_properties', 1),
            n_hops=node_generation_kwargs.get('n_hops', 1),
            use_question_for_graph_retrieval=node_generation_kwargs.get('use_question_for_graph_retrieval', True)
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
        reasoning_trace = self.get_reasoning_trace()
        memory = working_memory.format_textual_memory()
        context = format_context(memory=memory, retrieval_info=all_retrieved_info, reasoning_trace=reasoning_trace)

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
        all_log_keys = set(list(qa_log.keys()) + list(extractor_log.keys()))
        to_log_data = {key: qa_log.get(key, []) + extractor_log.get(key, []) for key in all_log_keys}
        return answers, retrieved_triples, entity_dict, property_dict, to_log_data
    
    def _generate_subqa(self):
        """Generate children which are sub-question nodes."""
    
    def _rephase(self):
        """Generate children which are rephrased question nodes."""
    
    def _self_correct(self):
        """Generate children which are self-corrected question nodes."""
    
    def _strengthen(self):
        """Generate children which are strengthened reasoning"""


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

    