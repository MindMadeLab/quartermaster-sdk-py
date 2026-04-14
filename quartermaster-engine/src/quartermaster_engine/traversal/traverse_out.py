"""TraverseOut — branching gate that decides which successor nodes to trigger.

After a node finishes execution, the TraverseOut strategy determines which
outgoing edges to follow: all of them, none, a specific picked node, or
loop back to the start.
"""

from __future__ import annotations

from uuid import UUID

from quartermaster_engine.nodes import NodeResult
from quartermaster_engine.types import GraphSpec, GraphNode, TraverseOut


class TraverseOutGate:
    """Determines which successor nodes to dispatch after a node completes."""

    def get_next_nodes(
        self,
        node_id: UUID,
        graph: GraphSpec,
        strategy: TraverseOut,
        result: NodeResult,
    ) -> list[GraphNode]:
        """Determine which successor nodes to execute next.

        Args:
            node_id: The node that just finished.
            graph: The agent graph definition.
            strategy: The branching strategy for this node.
            result: The execution result from this node.

        Returns:
            A list of GraphNode objects to dispatch next.
        """
        if strategy == TraverseOut.SPAWN_ALL:
            return self._spawn_all(node_id, graph)
        elif strategy == TraverseOut.SPAWN_NONE:
            return self._spawn_none()
        elif strategy == TraverseOut.SPAWN_PICKED:
            return self._spawn_picked(node_id, graph, result)
        elif strategy == TraverseOut.SPAWN_START:
            return self._spawn_start(graph)
        else:
            return self._spawn_all(node_id, graph)

    def _spawn_all(self, node_id: UUID, graph: GraphSpec) -> list[GraphNode]:
        """Trigger ALL successor nodes (parallel execution)."""
        return graph.get_successors(node_id)

    def _spawn_none(self) -> list[GraphNode]:
        """Stop execution — no successors triggered (dead end / End node)."""
        return []

    def _spawn_picked(
        self, node_id: UUID, graph: GraphSpec, result: NodeResult
    ) -> list[GraphNode]:
        """Trigger ONE specific successor based on the node's decision.

        The node's result must include a `picked_node` field containing
        either the target node's name or UUID string.
        """
        if not result.picked_node:
            return self._spawn_all(node_id, graph)

        successors = graph.get_successors(node_id)
        picked = result.picked_node.strip()

        # Try matching by name first
        for successor in successors:
            if successor.name == picked:
                return [successor]

        # Try matching by UUID string
        for successor in successors:
            if str(successor.id) == picked:
                return [successor]

        # Try matching by edge label
        edges = graph.get_edges_from(node_id)
        for edge in edges:
            if edge.label == picked:
                node = graph.get_node(edge.target_id)
                if node:
                    return [node]

        # Fallback: spawn all (better to continue than silently stop)
        return self._spawn_all(node_id, graph)

    def _spawn_start(self, graph: GraphSpec) -> list[GraphNode]:
        """Loop back to the start node (for iterative flows)."""
        start = graph.get_start_node()
        if start:
            return [start]
        return []
