import math
import random
from typing import Optional
from typing_extensions import TypedDict
import pydantic

from wemg.runners.base_reasoning_node import BaseReasoningNode, NodeType, NodeState


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


class MCTSSearchTree(TypedDict):
    root: MCTSReasoningNode
    explored_nodes: set[MCTSReasoningNode]


def select(node: MCTSReasoningNode, exploration_weight=1.0) -> MCTSReasoningNode:
    """Select the child with the highest UCT score."""
    path = []
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


def expand(node: MCTSReasoningNode) -> MCTSReasoningNode:
    pass


