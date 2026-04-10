"""TaskDispatcher protocol — pluggable execution strategy for successor nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import UUID


class TaskDispatcher(Protocol):
    """Protocol for dispatching node execution tasks.

    Implementations control how successor nodes are executed:
    synchronously (in-process), via threads, asyncio, or external task queues.
    """

    def dispatch(
        self,
        flow_id: UUID,
        node_id: UUID,
        execute_fn: Callable[[UUID, UUID], None],
    ) -> None:
        """Dispatch a node for execution.

        Args:
            flow_id: The flow this node belongs to.
            node_id: The node to execute.
            execute_fn: A callable that executes the node — signature (flow_id, node_id).
        """
        ...

    def wait_all(self) -> None:
        """Block until all dispatched tasks have completed.

        Called at synchronization points (e.g., merge nodes) and at flow end.
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources. Called when the flow runner is done."""
        ...
