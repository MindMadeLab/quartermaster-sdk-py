"""Typed stream-chunk classes for :func:`quartermaster_sdk.run.stream`.

v0.2.0 wraps :class:`quartermaster_engine.FlowEvent` into a small typed
discriminated union so callers can pattern-match on ``chunk.type`` without
needing to know about the underlying engine event hierarchy.  The mapping:

``FlowEvent``                           → ``Chunk``
``TokenGenerated``                      → :class:`TokenChunk`
``NodeStarted``                         → :class:`NodeStartChunk`
``NodeFinished``                        → :class:`NodeFinishChunk`
``FlowFinished``                        → :class:`DoneChunk`
``FlowError``                           → :class:`ErrorChunk`
``UserInputRequired``                   → :class:`AwaitInputChunk`
tool-call inside an ``AgentExecutor``   → :class:`ToolCallChunk` + :class:`ToolResultChunk`
                                          (future; currently surface via
                                          ``NodeFinishChunk.node_name``.)

The classes are intentionally flat dataclasses — callers can compare
``chunk.type`` against literal strings (``"token"``, ``"done"``, etc.) or do
``isinstance(chunk, TokenChunk)``; both styles work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from ._result import Result


@dataclass
class TokenChunk:
    """Streaming text token from an LLM node."""

    content: str
    node_name: str | None = None
    type: Literal["token"] = "token"


@dataclass
class NodeStartChunk:
    """A node has begun executing."""

    node_name: str
    node_type: str
    type: Literal["node_start"] = "node_start"


@dataclass
class NodeFinishChunk:
    """A node has finished executing.

    ``output`` is the string output of the node (may be empty).  For
    LLM nodes this is the full assembled text; for tool or decision
    nodes it's whatever the executor wrote to ``NodeResult.output_text``.
    """

    node_name: str
    output: str = ""
    type: Literal["node_finish"] = "node_finish"


@dataclass
class ToolCallChunk:
    """The agent executor requested a tool call."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResultChunk:
    """A tool call completed with a result (or an error payload)."""

    tool: str
    result: Any = None
    error: str | None = None
    type: Literal["tool_result"] = "tool_result"


@dataclass
class AwaitInputChunk:
    """A ``User`` node is waiting for a ``runner.resume`` call."""

    prompt: str
    options: list[str] = field(default_factory=list)
    type: Literal["await_input"] = "await_input"


@dataclass
class DoneChunk:
    """The flow finished — carries the final :class:`Result`."""

    result: Result
    type: Literal["done"] = "done"


@dataclass
class ErrorChunk:
    """The flow failed with an unrecoverable error."""

    error: str
    node_name: str | None = None
    type: Literal["error"] = "error"


Chunk = Union[
    TokenChunk,
    NodeStartChunk,
    NodeFinishChunk,
    ToolCallChunk,
    ToolResultChunk,
    AwaitInputChunk,
    DoneChunk,
    ErrorChunk,
]
"""Discriminated union of every chunk a stream can yield.

Callers usually pattern-match on ``chunk.type`` — the literal strings are
``"token"``, ``"node_start"``, ``"node_finish"``, ``"tool_call"``,
``"tool_result"``, ``"await_input"``, ``"done"``, ``"error"``.
"""


__all__ = [
    "Chunk",
    "TokenChunk",
    "NodeStartChunk",
    "NodeFinishChunk",
    "ToolCallChunk",
    "ToolResultChunk",
    "AwaitInputChunk",
    "DoneChunk",
    "ErrorChunk",
]
