"""Execution context — the runtime state passed to each node during flow execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qm_engine.context.node_execution import NodeStatus
from qm_engine.types import AgentVersion, GraphNode, Message


@dataclass
class ExecutionContext:
    """Runtime context for node execution.

    Carries the full state needed by a node to execute: the graph definition,
    current node reference, conversation history, flow-scoped memory, and
    callbacks for streaming and status updates.
    """

    flow_id: UUID
    node_id: UUID
    graph: AgentVersion
    current_node: GraphNode
    messages: list[Message] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    parent_context: ExecutionContext | None = None

    # Callbacks for real-time streaming
    on_message: Callable[[str], None] | None = None
    on_status_change: Callable[[NodeStatus], None] | None = None
    on_token: Callable[[str], None] | None = None

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get a value from the node's metadata, falling back to graph metadata."""
        if key in self.current_node.metadata:
            return self.current_node.metadata[key]
        return self.metadata.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        """Set a metadata value on this context."""
        self.metadata[key] = value

    def emit_token(self, token: str) -> None:
        """Emit a streaming token if a callback is registered."""
        if self.on_token:
            self.on_token(token)

    def emit_message(self, content: str) -> None:
        """Emit a complete message if a callback is registered."""
        if self.on_message:
            self.on_message(content)

    def update_status(self, status: NodeStatus) -> None:
        """Update this context's status and fire the callback."""
        self.status = status
        if self.on_status_change:
            self.on_status_change(status)
