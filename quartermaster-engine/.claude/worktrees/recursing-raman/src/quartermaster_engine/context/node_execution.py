"""Node execution state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID


class NodeStatus(str, Enum):
    """Lifecycle status of a node during flow execution."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    WAITING_TOOL = "waiting_tool"
    FINISHED = "finished"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        """Whether this status represents a completed state."""
        return self in (NodeStatus.FINISHED, NodeStatus.FAILED, NodeStatus.SKIPPED)

    @property
    def is_active(self) -> bool:
        """Whether this status represents an active (non-terminal) state."""
        return not self.is_terminal and self != NodeStatus.PENDING


@dataclass
class NodeExecution:
    """Tracks the execution state of a single node within a flow."""

    node_id: UUID
    status: NodeStatus = NodeStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    retry_count: int = 0
    output_data: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark this node as running."""
        self.status = NodeStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def finish(self, result: str | None = None, output_data: dict[str, Any] | None = None) -> None:
        """Mark this node as successfully finished."""
        self.status = NodeStatus.FINISHED
        self.finished_at = datetime.now(UTC)
        self.result = result
        if output_data:
            self.output_data.update(output_data)

    def fail(self, error: str) -> None:
        """Mark this node as failed."""
        self.status = NodeStatus.FAILED
        self.finished_at = datetime.now(UTC)
        self.error = error

    def skip(self) -> None:
        """Mark this node as skipped."""
        self.status = NodeStatus.SKIPPED
        self.finished_at = datetime.now(UTC)

    def wait_for_user(self) -> None:
        """Mark this node as waiting for user input."""
        self.status = NodeStatus.WAITING_USER

    def wait_for_tool(self) -> None:
        """Mark this node as waiting for tool execution."""
        self.status = NodeStatus.WAITING_TOOL

    @property
    def duration_seconds(self) -> float | None:
        """Execution duration in seconds, or None if not yet finished."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
