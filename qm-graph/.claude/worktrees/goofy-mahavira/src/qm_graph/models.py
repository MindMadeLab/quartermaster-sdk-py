"""Core Pydantic models for the agent graph schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from qm_graph.enums import (
    ErrorStrategy,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NodePosition(BaseModel):
    """Visual position of a node in the editor."""

    x: int = 0
    y: int = 0
    icon: str | None = None


class GraphNode(BaseModel):
    """A node in the agent graph."""

    id: UUID = Field(default_factory=uuid4)
    type: NodeType
    name: str = ""
    traverse_in: TraverseIn = TraverseIn.AWAIT_ALL
    traverse_out: TraverseOut = TraverseOut.SPAWN_ALL
    thought_type: ThoughtType = ThoughtType.NEW
    message_type: MessageType = MessageType.AUTOMATIC
    error_handling: ErrorStrategy = ErrorStrategy.STOP
    metadata: dict[str, Any] = Field(default_factory=dict)
    position: NodePosition | None = None


class GraphEdge(BaseModel):
    """A directed connection between two nodes."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    label: str = ""
    is_main: bool = True
    points: list[tuple[float, float]] = Field(default_factory=list)


class AgentVersion(BaseModel):
    """A versioned snapshot of an agent's graph."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    version: str = "0.1.0"
    start_node_id: UUID
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    features: str = ""
    is_published: bool = False
    forked_from: UUID | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Agent(BaseModel):
    """Top-level agent definition."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    current_version: str | None = None


class NodeDiff(BaseModel):
    """Difference for a single node between two versions."""

    node_id: UUID
    change: str  # "added", "removed", "modified"
    old: GraphNode | None = None
    new: GraphNode | None = None


class EdgeDiff(BaseModel):
    """Difference for a single edge between two versions."""

    edge_id: UUID
    change: str  # "added", "removed", "modified"
    old: GraphEdge | None = None
    new: GraphEdge | None = None


class GraphDiff(BaseModel):
    """Difference between two graph versions."""

    version_from: str
    version_to: str
    node_diffs: list[NodeDiff] = Field(default_factory=list)
    edge_diffs: list[EdgeDiff] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return len(self.node_diffs) > 0 or len(self.edge_diffs) > 0
