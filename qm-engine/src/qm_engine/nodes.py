"""Node protocol — the interface that node implementations must satisfy.

When qm-nodes is available, this can be replaced with imports from that package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qm_engine.context.execution_context import ExecutionContext


@dataclass
class NodeResult:
    """The result returned by a node after execution."""

    success: bool
    data: dict[str, Any]
    error: str | None = None
    picked_node: str | None = None  # For decision nodes: which successor to trigger
    output_text: str | None = None  # The main text output of the node
    wait_for_user: bool = False  # If True, flow pauses for user input
    user_prompt: str | None = None  # Prompt to show the user
    user_options: list[str] | None = None  # Options for user selection


class NodeExecutor(Protocol):
    """Protocol that node implementations must satisfy.

    This is the contract between the engine and the node implementations
    (from qm-nodes). The engine calls `execute()` and receives a `NodeResult`.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute this node with the given context.

        Args:
            context: The execution context containing messages, memory, metadata, etc.

        Returns:
            A NodeResult indicating success/failure and any output data.
        """
        ...


class NodeRegistry(Protocol):
    """Protocol for looking up node implementations by type.

    The engine uses this to resolve a GraphNode's type string into an
    actual executable node implementation.
    """

    def get_executor(self, node_type: str) -> NodeExecutor | None:
        """Retrieve the executor for a given node type string.

        Args:
            node_type: The node type identifier (e.g., "Instruction1").

        Returns:
            A NodeExecutor if one is registered for this type, otherwise None.
        """
        ...


class SimpleNodeRegistry:
    """A basic in-memory node registry.

    Maps node type strings to NodeExecutor instances.
    """

    def __init__(self) -> None:
        self._executors: dict[str, NodeExecutor] = {}

    def register(self, node_type: str, executor: NodeExecutor) -> None:
        """Register an executor for a node type."""
        self._executors[node_type] = executor

    def get_executor(self, node_type: str) -> NodeExecutor | None:
        """Retrieve the executor for a node type."""
        return self._executors.get(node_type)

    def list_types(self) -> list[str]:
        """List all registered node type strings."""
        return list(self._executors.keys())
