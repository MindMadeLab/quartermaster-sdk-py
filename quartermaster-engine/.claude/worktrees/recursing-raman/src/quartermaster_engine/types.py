"""Core type definitions for quartermaster-engine.

These types define the graph structure and enumerations used throughout the engine.
When quartermaster-graph is available, these can be replaced with imports from that package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class NodeType(str, Enum):
    """All supported node types in the agent graph."""

    # Core LLM nodes
    INSTRUCTION = "Instruction1"
    DECISION = "Decision1"
    REASONING = "Reasoning1"
    AGENT = "Agent1"

    # Control flow nodes
    START = "Start1"
    END = "End1"
    MERGE = "Merge1"
    IF = "If1"
    SWITCH = "Switch1"
    BREAK = "Break1"

    # User interaction nodes
    USER = "User1"
    USER_DECISION = "UserDecision1"
    USER_FORM = "UserForm1"

    # Data / utility nodes
    STATIC = "Static1"
    VAR = "Var1"
    TEXT = "Text1"
    CODE = "Code1"
    PROGRAM_RUNNER = "ProgramRunner1"

    # Memory nodes
    FLOW_MEMORY = "FlowMemory1"
    READ_MEMORY = "ReadMemory1"
    WRITE_MEMORY = "WriteMemory1"


class TraverseIn(str, Enum):
    """Synchronization strategy for incoming edges."""

    AWAIT_ALL = "AwaitAll"
    AWAIT_FIRST = "AwaitFirst"


class TraverseOut(str, Enum):
    """Branching strategy for outgoing edges."""

    SPAWN_ALL = "SpawnAll"
    SPAWN_NONE = "SpawnNone"
    SPAWN_START = "SpawnStart"
    SPAWN_PICKED = "SpawnPickedNode"


class ThoughtType(str, Enum):
    """How a node creates its thought/execution context."""

    SKIP = "SkipThought1"
    NEW = "NewThought1"
    NEW_HIDDEN = "NewHiddenThought1"
    INHERIT = "InheritThought1"
    CONTINUE = "ContinueThought1"


class MessageType(str, Enum):
    """How a node receives its input message."""

    AUTOMATIC = "Automatic"
    USER = "User"
    VARIABLE = "Variable"
    ASSISTANT = "Assistant"


class MessageRole(str, Enum):
    """Role of a message in conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ErrorStrategy(str, Enum):
    """How to handle errors during node execution."""

    STOP = "Stop"
    RETRY = "Retry"
    SKIP = "Skip"
    CUSTOM = "Custom"


@dataclass
class NodePosition:
    """Visual position of a node in the graph editor."""

    x: int = 0
    y: int = 0
    icon: str | None = None


@dataclass
class GraphNode:
    """A node in the agent graph."""

    id: UUID
    type: NodeType
    name: str = ""
    traverse_in: TraverseIn = TraverseIn.AWAIT_ALL
    traverse_out: TraverseOut = TraverseOut.SPAWN_ALL
    thought_type: ThoughtType = ThoughtType.NEW
    message_type: MessageType = MessageType.AUTOMATIC
    error_handling: ErrorStrategy = ErrorStrategy.STOP
    metadata: dict[str, Any] = field(default_factory=dict)
    position: NodePosition | None = None

    # Error handling configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None


@dataclass
class GraphEdge:
    """A directed edge connecting two nodes."""

    id: UUID
    source_id: UUID
    target_id: UUID
    label: str = ""
    is_main: bool = True


@dataclass
class AgentVersion:
    """A versioned snapshot of an agent's graph."""

    id: UUID
    agent_id: UUID
    version: str
    start_node_id: UUID
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    features: str = ""
    is_published: bool = False

    def get_node(self, node_id: UUID) -> GraphNode | None:
        """Find a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_start_node(self) -> GraphNode | None:
        """Get the start node of the graph."""
        return self.get_node(self.start_node_id)

    def get_successors(self, node_id: UUID) -> list[GraphNode]:
        """Get all successor nodes of a given node."""
        successor_ids = [e.target_id for e in self.edges if e.source_id == node_id]
        return [n for n in self.nodes if n.id in successor_ids]

    def get_predecessors(self, node_id: UUID) -> list[GraphNode]:
        """Get all predecessor nodes of a given node."""
        predecessor_ids = [e.source_id for e in self.edges if e.target_id == node_id]
        return [n for n in self.nodes if n.id in predecessor_ids]

    def get_edges_from(self, node_id: UUID) -> list[GraphEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to(self, node_id: UUID) -> list[GraphEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target_id == node_id]


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
