"""Core type definitions for qm-engine.

Graph-related types are imported from qm-graph; engine-specific types
(Message, MessageRole) are defined locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Re-export graph types from qm-graph so that the rest of qm-engine
# can continue to import them from ``qm_engine.types``.
from qm_graph.enums import (  # noqa: F401
    ErrorStrategy,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)
from qm_graph.models import (  # noqa: F401
    AgentVersion,
    GraphEdge,
    GraphNode,
    NodePosition,
)


# ── Engine-specific types ───────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Role of a message in conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """A message in the conversation history."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d
