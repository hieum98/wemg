"""Base reasoning node and state definitions.

This module defines the core node types and base class for reasoning nodes
used in both MCTS and CoT reasoning strategies.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import pydantic
from anytree import NodeMixin


class NodeType(Enum):
    """Types of reasoning nodes in the search tree."""
    USER_QUESTION = "USER_QUESTION"          # Root node with user's question
    FINAL_ANSWER = "FINAL_ANSWER"            # Terminal node with final answer
    SUB_QA_NODE = "SUBQUESTION"              # Intermediate node with sub-question and answer
    REPHRASED_QUESTION_NODE = "REPHRASE_QUESTION"  # Rephrased question node
    SELF_CORRECTED_NODE = "SELF_CORRECT"     # Self-corrected answer node
    SYNTHESIS_NODE = "SYNTHESIS"             # Reasoning synthesis node


class NodeState(pydantic.BaseModel):
    """State of a reasoning node containing its type and content."""
    node_type: NodeType = pydantic.Field(..., description="The type of the node.")
    content: Dict[str, str] = pydantic.Field(
        ..., 
        description="The content of the node, which varies based on the node type."
    )

    @pydantic.model_validator(mode='after')
    def check_content(self) -> 'NodeState':
        """Validate that content has required fields based on node type."""
        assert 'user_question' in self.content, "All nodes must have 'user_question' in content."
        
        validators = {
            NodeType.FINAL_ANSWER: lambda c: 'final_answer' in c,
            NodeType.SUB_QA_NODE: lambda c: 'sub_question' in c and 'sub_answer' in c,
            NodeType.REPHRASED_QUESTION_NODE: lambda c: 'sub_question' in c,
            NodeType.SELF_CORRECTED_NODE: lambda c: 'sub_question' in c and 'sub_answer' in c,
            NodeType.SYNTHESIS_NODE: lambda c: 'synthesized_reasoning' in c,
        }
        
        validator = validators.get(self.node_type)
        if validator and not validator(self.content):
            raise ValueError(f"Missing required content fields for {self.node_type}")
        
        return self

    def __str__(self) -> str:
        """Format node state as readable string."""
        c = self.content
        
        formatters = {
            NodeType.FINAL_ANSWER: lambda: f"Final Answer: {c['final_answer']}" + 
                (f"\nReasoning: {c['reasoning']}" if 'reasoning' in c else ""),
            NodeType.SUB_QA_NODE: lambda: f"Sub Question: {c['sub_question']}\nSub Answer: {c['sub_answer']}" +
                (f"\nReasoning: {c['reasoning']}" if 'reasoning' in c else ""),
            NodeType.REPHRASED_QUESTION_NODE: lambda: f"Rephrased Question: {c['sub_question']}",
            NodeType.SELF_CORRECTED_NODE: lambda: f"Sub Question: {c['sub_question']}\nSub Answer: {c['sub_answer']}",
            NodeType.SYNTHESIS_NODE: lambda: c['synthesized_reasoning'],
            NodeType.USER_QUESTION: lambda: f"User Question: {c['user_question']}",
        }
        
        formatter = formatters.get(self.node_type, lambda: str(c))
        return formatter()


# Parent node type constraints for validation
PARENT_TYPE_CONSTRAINTS = {
    NodeType.REPHRASED_QUESTION_NODE: [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE],
    NodeType.SELF_CORRECTED_NODE: [NodeType.SUB_QA_NODE],
    NodeType.SYNTHESIS_NODE: [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE],
}


class BaseReasoningNode(ABC, NodeMixin):
    """Abstract base class for reasoning tree nodes.
    
    Provides common functionality for tree traversal, state management,
    and trajectory tracking. Subclasses must implement generate_children().
    """
    
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
        self.golden_answer = content.get('golden_answer')
        self.max_depth = max_depth
        
        # Validate parent type constraints
        self._validate_parent_constraints()
    
    def _validate_parent_constraints(self) -> None:
        """Validate that parent node type is valid for this node type."""
        constraints = PARENT_TYPE_CONSTRAINTS.get(self.node_type)
        if constraints and self.parent:
            if self.parent.node_type not in constraints:
                raise ValueError(
                    f"{self.node_type} must be generated from one of {constraints}, "
                    f"got {self.parent.node_type}"
                )

    def __repr__(self) -> str:
        return self.node_state.model_dump_json(indent=2)
    
    def get_trajectory(self) -> List[NodeState]:
        """Get the trajectory of node states from root to this node."""
        return [node.node_state for node in self.path if node]
    
    def get_reasoning_trace(self) -> str:
        """Get formatted reasoning trace from root to this node."""
        trajectory = self.get_trajectory()
        trace_parts: List[str] = []
        
        for node in trajectory:
            if node.node_type in [NodeType.SUB_QA_NODE, NodeType.SYNTHESIS_NODE]:
                trace_parts.append(str(node))
            elif node.node_type == NodeType.SELF_CORRECTED_NODE and trace_parts:
                trace_parts[-1] += f"\nSelf Corrected: {str(node)}"
            elif node.node_type == NodeType.FINAL_ANSWER:
                trace_parts.append(str(node))
        
        return "\n".join(f"Step {i+1}:\n {step.strip()}" for i, step in enumerate(trace_parts))

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (final answer or max depth reached)."""
        return self.node_type == NodeType.FINAL_ANSWER or self.depth >= self.max_depth
    
    def is_valid_leaf(self) -> bool:
        """Check if this is a valid leaf node (has final answer)."""
        return self.node_type == NodeType.FINAL_ANSWER
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    @abstractmethod
    def generate_children(self) -> List['BaseReasoningNode']:
        """Generate children nodes for the current node.
        
        Must be implemented by subclasses to define how children are generated
        based on the node type and search strategy.
        """
        pass
