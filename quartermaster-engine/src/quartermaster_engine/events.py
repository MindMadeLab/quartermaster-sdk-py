"""Flow execution events for real-time streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from quartermaster_engine.types import NodeType


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


@dataclass
class ToolCallStarted(FlowEvent):
    """Emitted when an agent node is about to invoke a tool.

    Downstream SSE handlers translate this into a
    ``ToolCallChunk`` so chat UIs can render "calling tool X..."
    cards as they happen, rather than batching them at the end of
    the turn.  The ``iteration`` field corresponds to the agent's
    tool-loop round the call belongs to.
    """

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


@dataclass
class ToolCallFinished(FlowEvent):
    """Emitted after a tool call resolves (success or error).

    ``result`` is the string surface the LLM sees next turn; the
    original structured payload — if any — lives under ``raw``.
    ``error`` is populated for tool-registry lookups that fail or
    for tools that raised during execution; in both cases ``result``
    also carries an ``[ERROR: ...]`` sentinel string so the model
    can react to the failure and retry or apologise.
    """

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    raw: Any = None
    error: str | None = None
    iteration: int = 0


@dataclass
class ProgressEvent(FlowEvent):
    """User-emitted progress signal from inside a node or tool.

    Fires when application code calls
    ``ExecutionContext.emit_progress(...)`` — typically from a long-
    running tool that wants to report status to the stream (e.g. a web
    search saying "query 3 of 5") without interrupting the LLM token
    channel. ``percent`` is optional: set to a 0.0–1.0 float for
    determinate progress bars, leave as ``None`` for spinner-style
    indeterminate progress. ``data`` is a free-form dict for payload
    fields the consumer wants to render (search query, step index,
    etc.) without bloating the message string.
    """

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    message: str = ""
    percent: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomEvent(FlowEvent):
    """Application-defined structured event.

    Fires when application code calls
    ``ExecutionContext.emit_custom(name, payload)``. The ``name``
    field is a caller-chosen discriminator that downstream consumers
    filter on (``stream.custom(name="retrieved_docs")``); ``payload``
    carries the structured data. Use this instead of
    :class:`ProgressEvent` when the event is a discrete milestone
    (document retrieved, quota updated, cache hit/miss, …) rather
    than a percent-along-a-task signal.
    """

    node_id: UUID = field(default_factory=lambda: UUID(int=0))
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
