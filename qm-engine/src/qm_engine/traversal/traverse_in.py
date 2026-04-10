"""TraverseIn — synchronization gate that decides when a node should execute.

A node may have multiple incoming edges (predecessors). The TraverseIn strategy
determines whether the node should wait for ALL predecessors or just the FIRST one.
"""

from __future__ import annotations

from uuid import UUID

from qm_engine.context.node_execution import NodeExecution
from qm_engine.types import AgentVersion, GraphNode, TraverseIn


class TraverseInGate:
    """Evaluates whether a node is ready to execute based on predecessor states."""

    def should_execute(
        self,
        node_id: UUID,
        graph: AgentVersion,
        node_executions: dict[UUID, NodeExecution],
        strategy: TraverseIn,
    ) -> bool:
        """Determine if a node should execute now.

        Args:
            node_id: The node to check.
            graph: The agent graph definition.
            node_executions: Current execution states of all nodes.
            strategy: The synchronization strategy (AwaitAll or AwaitFirst).

        Returns:
            True if the node is ready to execute.
        """
        predecessors = graph.get_predecessors(node_id)

        # No predecessors (start node or orphan) — always execute
        if not predecessors:
            return True

        if strategy == TraverseIn.AWAIT_ALL:
            return self._await_all(predecessors, node_executions)
        elif strategy == TraverseIn.AWAIT_FIRST:
            return self._await_first(predecessors, node_executions)
        else:
            return True

    def _await_all(
        self,
        predecessors: list[GraphNode],
        node_executions: dict[UUID, NodeExecution],
    ) -> bool:
        """Wait for ALL predecessors to reach a terminal state."""
        for pred in predecessors:
            execution = node_executions.get(pred.id)
            if execution is None:
                return False
            if not execution.status.is_terminal:
                return False
        return True

    def _await_first(
        self,
        predecessors: list[GraphNode],
        node_executions: dict[UUID, NodeExecution],
    ) -> bool:
        """Execute as soon as ANY predecessor reaches a terminal state."""
        for pred in predecessors:
            execution = node_executions.get(pred.id)
            if execution is not None and execution.status.is_terminal:
                return True
        return False
