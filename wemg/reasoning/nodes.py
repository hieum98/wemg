"""Reasoning tree nodes for MCTS and CoT search strategies."""

import math
from enum import Enum
from typing import Dict, List, Optional
import pydantic
from anytree import NodeMixin, RenderTree

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = YELLOW = BLUE = MAGENTA = CYAN = RED = ""
    class Style:
        RESET_ALL = ""


class NodeType(Enum):
    USER_QUESTION = "USER_QUESTION"
    FINAL_ANSWER = "FINAL_ANSWER"
    SUB_QA_NODE = "SUBQUESTION"
    REPHRASED_QUESTION_NODE = "REPHRASE_QUESTION"
    SELF_CORRECTED_NODE = "SELF_CORRECT"
    SYNTHESIS_NODE = "SYNTHESIS"


class NodeState(pydantic.BaseModel):
    node_type: NodeType
    content: Dict[str, str]
    
    @pydantic.model_validator(mode='after')
    def check_content(self) -> 'NodeState':
        assert 'user_question' in self.content, "All nodes must have 'user_question'"
        validators = {
            NodeType.FINAL_ANSWER: lambda c: 'final_answer' in c,
            NodeType.SUB_QA_NODE: lambda c: 'sub_question' in c and 'sub_answer' in c,
            NodeType.REPHRASED_QUESTION_NODE: lambda c: 'sub_question' in c,
            NodeType.SELF_CORRECTED_NODE: lambda c: 'sub_question' in c and 'sub_answer' in c,
            NodeType.SYNTHESIS_NODE: lambda c: 'synthesized_reasoning' in c,
        }
        validator = validators.get(self.node_type)
        if validator and not validator(self.content):
            raise ValueError(f"Missing required content fields for {self.node_type}, content: {self.content}")
        return self
    
    def __str__(self) -> str:
        c = self.content
        formatters = {
            NodeType.FINAL_ANSWER: lambda: f"Final Answer: {c['final_answer']}" + (f"\nReasoning: {c['reasoning']}" if 'reasoning' in c else ""),
            NodeType.SUB_QA_NODE: lambda: f"Sub Question: {c['sub_question']}\nSub Answer: {c['sub_answer']}" + (f"\nReasoning: {c['reasoning']}" if 'reasoning' in c else ""),
            NodeType.REPHRASED_QUESTION_NODE: lambda: f"Rephrased Question: {c['sub_question']}",
            NodeType.SELF_CORRECTED_NODE: lambda: f"Sub Question: {c['sub_question']}\nSub Answer: {c['sub_answer']}" + (f"\nReasoning: {c['reasoning']}" if 'reasoning' in c else ""),
            NodeType.SYNTHESIS_NODE: lambda: c['synthesized_reasoning'],
            NodeType.USER_QUESTION: lambda: f"User Question: {c['user_question']}",
        }
        return formatters.get(self.node_type, lambda: str(c))()


PARENT_TYPE_CONSTRAINTS = {
    NodeType.REPHRASED_QUESTION_NODE: [NodeType.USER_QUESTION, NodeType.SUB_QA_NODE],
    NodeType.SELF_CORRECTED_NODE: [NodeType.SUB_QA_NODE],
    NodeType.SYNTHESIS_NODE: [NodeType.SUB_QA_NODE, NodeType.SELF_CORRECTED_NODE],
}


class ReasoningNode(NodeMixin):
    """Reasoning tree node built on anytree."""

    def __init__(
        self,
        node_state: NodeState,
        parent: Optional["ReasoningNode"] = None,
        max_depth: int = 10,
    ):
        # Core state
        self.node_state: NodeState = node_state
        self.node_type: NodeType = node_state.node_type
        self.max_depth: int = max_depth

        # Convenience attributes
        self.user_question: str = node_state.content["user_question"]
        self.golden_answer: Optional[str] = node_state.content.get("golden_answer")

        # anytree relationship
        self.parent = parent

        # Enforce parent-type constraints if a parent exists
        if self.parent is not None:
            constraints = PARENT_TYPE_CONSTRAINTS.get(self.node_type)
            if constraints and self.parent.node_type not in constraints:
                raise ValueError(
                    f"{self.node_type} must have parent of type {constraints}, "
                    f"got {self.parent.node_type}"
                )
    
    @property
    def path(self) -> List["ReasoningNode"]:
        """Path from root to this node."""
        nodes: List["ReasoningNode"] = []
        node: Optional["ReasoningNode"] = self
        while node is not None:
            nodes.append(node)
            node = node.parent  # type: ignore[assignment]
        return list(reversed(nodes))

    def get_trajectory(self) -> List[NodeState]:
        return [node.node_state for node in self.path]

    def get_reasoning_trace(self) -> str:
        trajectory = self.get_trajectory()
        parts = []
        for node in trajectory:
            if node.node_type in [NodeType.SUB_QA_NODE, NodeType.SYNTHESIS_NODE]:
                parts.append(str(node))
            elif node.node_type == NodeType.SELF_CORRECTED_NODE and parts:
                parts[-1] += f"\nSelf Corrected: {str(node)}"
            elif node.node_type == NodeType.FINAL_ANSWER:
                parts.append(str(node))
        return "\n".join(f"Step {i+1}:\n {step.strip()}" for i, step in enumerate(parts))

    def is_terminal(self) -> bool:
        return self.node_type == NodeType.FINAL_ANSWER or self.depth >= self.max_depth

    def is_valid_leaf(self) -> bool:
        return self.node_type == NodeType.FINAL_ANSWER

    def is_root(self) -> bool:
        return self.parent is None

    def _summary(self) -> str:
        """Compact, single-line summary string for this node."""
        data = self.node_state.content
        if self.node_type == NodeType.USER_QUESTION:
            gt = data.get("golden_answer", "N/A")
            detail = f"User: {data['user_question'][:80]} - GT: {gt}"
        elif self.node_type == NodeType.FINAL_ANSWER:
            detail = f"Final: {data.get('final_answer', '')[:80]}"
        elif self.node_type == NodeType.SUB_QA_NODE:
            detail = (
                f"Sub_Q: {data.get('sub_question', '')[:60]} - "
                f"Sub_A: {data.get('sub_answer', '')[:60]}"
            )
        elif self.node_type == NodeType.REPHRASED_QUESTION_NODE:
            detail = f"Rephrase: {data.get('sub_question', '')[:80]}"
        elif self.node_type == NodeType.SELF_CORRECTED_NODE:
            detail = f"Self_corrected: {data.get('sub_answer', '')[:80]}"
        elif self.node_type == NodeType.SYNTHESIS_NODE:
            detail = f"Synthesis: {data.get('synthesized_reasoning', '')[:80]}"
        else:
            detail = str(data)[:80]

        return detail.replace("\n", " ").replace("\r", " ")

    def _stable_id(self) -> str:
        """Stable identifier based on node type and content, depth-agnostic."""
        key = (self.node_type.value, tuple(sorted(self.node_state.content.items())))
        return f"{hash(key)}-{self.node_type.value}"

    def print_tree(self) -> None:
        """Print tree with indentation and color coding using anytree.RenderTree."""
        # Determine root for rendering
        root: "ReasoningNode" = self
        while root.parent is not None:
            root = root.parent  # type: ignore[assignment]

        color_map = {
            NodeType.USER_QUESTION: Fore.GREEN,
            NodeType.REPHRASED_QUESTION_NODE: Fore.YELLOW,
            NodeType.FINAL_ANSWER: Fore.BLUE,
            NodeType.SELF_CORRECTED_NODE: Fore.MAGENTA,
            NodeType.SUB_QA_NODE: Fore.CYAN,
            NodeType.SYNTHESIS_NODE: Fore.RED,
        }

        for pre, _fill, node in RenderTree(root):
            assert isinstance(node, ReasoningNode)
            color = color_map.get(node.node_type, "")
            node_id = node._stable_id()
            detail = node._summary()
            print(f"{pre}{color}{node_id}{Style.RESET_ALL} {detail}")

    def __repr__(self) -> str:
        return self.node_state.model_dump_json(indent=2)


class MCTSNode(ReasoningNode):
    """Reasoning node with MCTS-specific value/visits and UCT."""
    
    def __init__(self, node_state: NodeState, parent: Optional['MCTSNode'] = None, max_depth: int = 10):
        super().__init__(node_state=node_state, parent=parent, max_depth=max_depth)
        self.value: float = 0.0
        self.visits: int = 0
    
    def upper_confidence_bound(self, exploration_weight: float = 2.0) -> float:
        if self.parent is None:
            raise ValueError("Cannot compute UCT for root node")
        if self.visits == 0:
            return float('inf')
        average_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term
    
    def backpropagate(self, reward: float) -> None:
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class CoTNode(ReasoningNode):
    """Reasoning node for Chain-of-Thought (no MCTS-specific fields needed)."""
    pass
