from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import pydantic
from anytree import NodeMixin


class NodeType(Enum):
    # Node type for user question, i.e., the root node
    USER_QUESTION = "USER_QUESTION"
    # Node type for final answer of the user question, i.e., the terminal node  
    FINAL_ANSWER = "FINAL_ANSWER"
    # Node type for subQA, i.e., the intermediate node for the sub-question and sub-answer 
    SUB_QA_NODE = "SUBQUESTION" 
    # Node type for rephrase question, i.e., the intermediate node for rephrased question. 
    # This node must be followed by a SUBQUESTION node and be generated from a USER_QUESTION or SUBQUESTION node.
    REPHRASED_QUESTION_NODE = "REPHASE_QUESTION"
    # Node type for self-correcting reasoning, i.e., the intermediate node for self-correcting reasoning.
    # This node must be generated from a SUBQUESTION node
    SELF_CORRECTED_NODE = "SELF_CORRECT"
    # Node type for reasoning strengthening, i.e., the intermediate node for reasoning strengthening.  
    # This node must be generated from a SUBQUESTION or SELF_CORRECTED_NODE node
    SYNTHESIS_NODE = "SYNTHESIS"


class NodeState(pydantic.BaseModel):
    node_type: NodeType = pydantic.Field(..., description="The type of the node.")
    content: Dict[str, str] = pydantic.Field(
        ..., 
        description="The content of the node, which varies based on the node type."
    )

    # check post-init conditions
    @pydantic.model_validator(mode='after')
    def check_content(self) -> 'NodeState':
        assert 'user_question' in self.content, "All nodes must have 'user_question' in content."
        if self.node_type == NodeType.FINAL_ANSWER:
            assert 'final_answer' in self.content, "FINAL_ANSWER node must have 'final_answer' in content."
        elif self.node_type == NodeType.SUB_QA_NODE:
            assert 'sub_question' in self.content and 'sub_answer' in self.content, "SUB_QA_NODE must have 'sub_question' and 'sub_answer' in content."
        elif self.node_type == NodeType.REPHRASED_QUESTION_NODE:
            assert 'sub_question' in self.content, "REPHRASED_QUESTION_NODE must have 'sub_question' in content."
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            assert 'sub_question' in self.content and 'sub_answer' in self.content, "SELF_CORRECTED_NODE must have 'sub_question' and 'sub_answer' in content."
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            assert 'synthesized_reasoning' in self.content, "SYNTHESIS_NODE must have 'synthesized_reasoning' in content."
       
        return self        

    def __str__(self):
        if self.node_type == NodeType.FINAL_ANSWER:
            text = f"Final Answer: {self.content['final_answer']}"
            if 'reasoning' in self.content:
                text += f"\nReasoning: {self.content['reasoning']}"
        elif self.node_type == NodeType.SUB_QA_NODE:
            text = f"Sub Question: {self.content['sub_question']}\nSub Answer: {self.content['sub_answer']}"
            if 'reasoning' in self.content:
                text += f"\nReasoning: {self.content['reasoning']}"
        elif self.node_type == NodeType.REPHRASED_QUESTION_NODE:
            text = f"Rephrased Question: {self.content['sub_question']}"
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            text = f"Sub Question: {self.content['sub_question']}\nSub Answer: {self.content['sub_answer']}"
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            text = f"{self.content['synthesized_reasoning']}"
        return text


class BaseReasoningNode(ABC, NodeMixin):
    def __init__(
            self, 
            node_state: NodeState,
            parent: Optional['BaseReasoningNode'] = None,
            children: Optional[List['BaseReasoningNode']] = None,
            max_depth: int = 10,
    ):  
        self.node_state = node_state
        self.node_type = node_state.node_type
        self.parent: Optional['BaseReasoningNode'] = parent
        if children:
            self.children: List['BaseReasoningNode'] = children
        content = node_state.content

        self.user_question = content['user_question']
        self.golden_answer = content.get('golden_answer', None)
        self.max_depth = max_depth

        if self.node_type == NodeType.REPHRASED_QUESTION_NODE:
            assert parent.node_type in [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE], \
                "REPHRASED_QUESTION_NODE must be generated from USER_QUESTION or SUB_QA_NODE."
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            assert parent.node_type == NodeType.SUB_QA_NODE, \
                "SELF_CORRECTED_NODE must be generated from SUB_QA_NODE."
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            assert parent.node_type in [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE], \
                "SYNTHESIS_NODE must be generated from SUB_QA_NODE or SELF_CORRECTED_NODE."
        else:
            raise ValueError(f"Unsupported node type: {self.node_type}")
        
    def __repr__(self):
        return self.node_state.model_dump_json(indent=2)
    
    def get_trajectory(self) -> List[NodeState]:
        """Get the trajectory from the root to this node."""
        trajectory = []
        for node in self.path:
            if node: 
                trajectory.append(node.node_state)
        return trajectory
    
    def get_reasoning_trace(self) -> str:
        """Get the reasoning trace from the root to this node."""
        trajectory = self.get_trajectory()
        reasoning_trace: List[str] = []
        for node in trajectory:
            if node.node_type in [NodeType.SUB_QA_NODE, NodeType.SYNTHESIS_NODE]:
                reasoning_trace.append(str(node))
            elif node.node_type == NodeType.SELF_CORRECTED_NODE:
                reasoning_trace[-1] += f"\nSelf Corrected: {str(node)}"
            elif node.node_type == NodeType.FINAL_ANSWER:
                reasoning_trace.append(str(node))
        reasoning_trace = [f"Step {i+1}:\n {step.strip()}" for i, step in enumerate(reasoning_trace)]
        return "\n".join(reasoning_trace)

    def is_terminal(self) -> bool:
        """Check if the node is a terminal node."""
        return self.node_type == NodeType.FINAL_ANSWER or self.depth >= self.max_depth
    
    def is_valid_leaf(self) -> bool:
        """Check if the node is a valid leaf node."""
        return self.node_type == NodeType.FINAL_ANSWER
    
    def is_root(self) -> bool:
        """Check if the node is a root node."""
        return self.parent is None
    
    @abstractmethod
    def generate_children(self) -> List['BaseReasoningNode']:
        """Generate children nodes for the current node."""

    