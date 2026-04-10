"""ExecutionStore protocol — pluggable storage for flow execution state."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from quartermaster_engine.context.node_execution import NodeExecution
from quartermaster_engine.types import Message


class ExecutionStore(Protocol):
    """Protocol for storing flow execution state.

    Implementations might use in-memory dicts, SQLite, Redis, PostgreSQL, etc.
    The engine is agnostic — it only talks to this interface.
    """

    def save_node_execution(self, flow_id: UUID, node_id: UUID, execution: NodeExecution) -> None:
        """Persist the execution state for a node."""
        ...

    def get_node_execution(self, flow_id: UUID, node_id: UUID) -> NodeExecution | None:
        """Retrieve the execution state for a node, or None if not tracked."""
        ...

    def get_all_node_executions(self, flow_id: UUID) -> dict[UUID, NodeExecution]:
        """Get all node executions for a flow."""
        ...

    def save_memory(self, flow_id: UUID, key: str, value: Any) -> None:
        """Store a flow-scoped variable."""
        ...

    def get_memory(self, flow_id: UUID, key: str) -> Any:
        """Retrieve a flow-scoped variable. Returns None if not found."""
        ...

    def get_all_memory(self, flow_id: UUID) -> dict[str, Any]:
        """Get all flow-scoped variables."""
        ...

    def delete_memory(self, flow_id: UUID, key: str) -> None:
        """Delete a flow-scoped variable."""
        ...

    def save_messages(self, flow_id: UUID, node_id: UUID, messages: list[Message]) -> None:
        """Store conversation history for a node."""
        ...

    def get_messages(self, flow_id: UUID, node_id: UUID) -> list[Message]:
        """Retrieve conversation history for a node."""
        ...

    def append_message(self, flow_id: UUID, node_id: UUID, message: Message) -> None:
        """Append a single message to a node's conversation history."""
        ...

    def clear_flow(self, flow_id: UUID) -> None:
        """Remove all state for a flow (cleanup after completion)."""
        ...
