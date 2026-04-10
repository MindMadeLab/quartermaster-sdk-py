"""Protocols defining interfaces for framework-agnostic node execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import UUID

# Re-export LLMConfig from qm-providers (canonical definition)
from qm_providers.config import LLMConfig


@runtime_checkable
class ThoughtHandle(Protocol):
    """Interface for manipulating thought state during node execution."""

    def append_text(self, text: str) -> None:
        """Append text content to the thought."""
        ...

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update thought metadata with key-value pairs."""
        ...


@runtime_checkable
class Thought(Protocol):
    """Interface for accessing thought data."""

    @property
    def text(self) -> str:
        """The text content of this thought."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dictionary for this thought."""
        ...

    def get_previous_child_thoughts(self) -> list[Thought]:
        """Get thoughts from previous child branches (for merge nodes)."""
        ...


@runtime_checkable
class Edge(Protocol):
    """Interface for a graph edge connecting nodes."""

    @property
    def tail_id(self) -> Any:
        """The target node ID of this edge."""
        ...

    @property
    def main_direction(self) -> bool:
        """Whether this is the main (true) direction edge."""
        ...

    @property
    def direction_text(self) -> str:
        """Human-readable description of this edge direction."""
        ...


@runtime_checkable
class AssistantNode(Protocol):
    """Interface for accessing the assistant node definition."""

    @property
    def predecessor_edges(self) -> EdgeQuerySet:
        """Edges coming from this node."""
        ...


@runtime_checkable
class EdgeQuerySet(Protocol):
    """Interface for querying edges."""

    def all(self) -> list[Edge]:
        """Return all edges."""
        ...


@runtime_checkable
class NodeContext(Protocol):
    """Interface for the context passed to node.think().

    This is the main abstraction that decouples nodes from any specific
    framework (Django, FastAPI, etc.). Implement this protocol to integrate
    nodes with your runtime.
    """

    @property
    def node_metadata(self) -> dict[str, Any]:
        """Node configuration metadata."""
        ...

    @property
    def flow_node_id(self) -> UUID:
        """The ID of the current flow node instance."""
        ...

    @property
    def thought_id(self) -> Optional[UUID]:
        """The ID of the current thought (memory)."""
        ...

    @property
    def thought(self) -> Optional[Thought]:
        """The current thought object."""
        ...

    @property
    def handle(self) -> Optional[ThoughtHandle]:
        """Handle for manipulating the current thought."""
        ...

    @property
    def assistant_node(self) -> AssistantNode:
        """The assistant node definition."""
        ...

    @property
    def chat_id(self) -> Optional[UUID]:
        """The chat session ID."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM service providers."""

    def generate_stream(
        self, messages: list[dict], config: Any, **kwargs: Any
    ) -> Any:
        """Generate a streaming response."""
        ...

    def generate_structured(
        self, messages: list[dict], tools: list[Any], config: Any, **kwargs: Any
    ) -> Any:
        """Generate a structured response with tool calls."""
        ...


@runtime_checkable
class ExpressionEvaluator(Protocol):
    """Interface for evaluating Python expressions safely."""

    def eval_expression(
        self, node_id: Any, expression: str, context: dict[str, Any]
    ) -> ExpressionResult:
        """Evaluate an expression in the given context."""
        ...


@dataclass
class ExpressionResult:
    """Result of an expression evaluation."""

    result: Any = None
    error: Optional[str] = None
    success: bool = True


@dataclass
class ContextManagerConfig:
    """Configuration for context management in LLM handlers."""

    tool_clearing_trigger: Optional[int] = None
    tool_clearing_keep: Optional[int] = None
    exclude_tools: list[str] = field(default_factory=list)
    max_tool_result_tokens: Optional[int] = None
    image_clearing_trigger: Optional[int] = None
    image_clearing_keep: Optional[int] = None
    max_messages: int = 0
    max_input_tokens: int = 0


@dataclass
class ParameterContainer:
    """Container for tool parameter definitions."""

    name: str
    type: str
    description: str
    is_required: bool = False
    enum: Optional[list[str]] = None


@dataclass
class ProgramContainer:
    """Container for tool/program definitions."""

    name: str
    description: str
    parameters: list[ParameterContainer] = field(default_factory=list)

    def add_parameter(self, parameter: ParameterContainer) -> None:
        self.parameters.append(parameter)

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        properties = {}
        required = []
        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.is_required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
