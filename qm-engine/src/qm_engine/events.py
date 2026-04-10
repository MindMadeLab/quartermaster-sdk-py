"""Flow execution events for real-time streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qm_engine.types import NodeType


@dataclass
class FlowEvent:
    """Base class for all flow execution events."""

    flow_id: UUID


@dataclass
class NodeStarted(FlowEvent):
    """Emitted when a node begins execution."""

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    node_type: NodeType = NodeType.START
    node_name: str = ""


@dataclass
class TokenGenerated(FlowEvent):
    """Emitted for each streaming token from an LLM node."""

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    token: str = ""


@dataclass
class NodeFinished(FlowEvent):
    """Emitted when a node completes execution."""

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    result: str = ""
    output_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowFinished(FlowEvent):
    """Emitted when the entire flow completes."""

    final_output: str = ""
    output_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserInputRequired(FlowEvent):
    """Emitted when a node is waiting for user input."""

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    prompt: str = ""
    options: list[str] = field(default_factory=list)


@dataclass
class FlowError(FlowEvent):
    """Emitted when a node fails."""

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    error: str = ""
    recoverable: bool = False
