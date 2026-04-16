"""Execution context — the runtime state passed to each node during flow execution."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from quartermaster_engine.context.node_execution import NodeStatus
from quartermaster_engine.types import GraphSpec, GraphNode, Message


@dataclass
class ExecutionContext:
    """Runtime context for node execution.

    Carries the full state needed by a node to execute: the graph definition,
    current node reference, conversation history, flow-scoped memory, and
    callbacks for streaming and status updates.
    """

    flow_id: UUID
    node_id: UUID
    graph: GraphSpec
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
    # Tool-call streaming — fires once per agent tool invocation so
    # downstream chat UIs can render live "calling tool X..." cards
    # instead of waiting for the full NodeResult. Payload shape matches
    # ``events.ToolCallStarted`` / ``ToolCallFinished``.
    on_tool_start: Callable[[str, dict, int], None] | None = None
    on_tool_finish: Callable[[str, dict, str, Any, str | None, int], None] | None = None
    # Application-emitted event hooks — populated by the runner so
    # user code calling ``emit_progress`` / ``emit_custom`` from
    # inside a tool interleaves with engine-emitted tokens on the
    # consumer's side. Both stay None when no streaming consumer is
    # attached (e.g. a unit test invoking a tool directly).
    on_progress: Callable[[str, float | None, dict], None] | None = None
    on_custom: Callable[[str, dict], None] | None = None

    # v0.4.0 cooperative cancellation.
    #
    # Populated by :class:`FlowRunner` with the per-flow cancellation
    # :class:`threading.Event`; every :class:`ExecutionContext` the
    # runner builds for a given flow shares the same event so when the
    # runner is asked to :meth:`FlowRunner.stop` (from the SDK's stream
    # context-manager exit) every tool in that flow observes
    # ``ctx.cancelled`` flipping to ``True`` on the next check.
    #
    # Remains ``None`` outside a running flow (unit tests, out-of-flow
    # helpers); the :attr:`cancelled` property returns ``False`` in
    # that case so tool code stays safe to exercise without a runner.
    _cancelled_event: threading.Event | None = None

    @property
    def cancelled(self) -> bool:
        """Best-effort "the flow has been asked to stop" flag.

        Tools that want to bail out cooperatively mid-work can poll this
        and either ``raise qm.Cancelled(...)`` or early-return. The flag
        flips to ``True`` when the SDK's stream context-manager exits
        (break / return / exception) and calls ``runner.stop(flow_id)``,
        which sets the per-flow event. Reading outside an active flow
        returns ``False`` — the check is always safe.
        """
        event = self._cancelled_event
        return event is not None and event.is_set()

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

    def emit_tool_start(self, tool: str, arguments: dict[str, Any], iteration: int) -> None:
        """Fire ``on_tool_start`` for the agent-executor tool loop, if wired."""
        if self.on_tool_start:
            self.on_tool_start(tool, arguments, iteration)

    def emit_tool_finish(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: str,
        raw: Any,
        error: str | None,
        iteration: int,
    ) -> None:
        """Fire ``on_tool_finish`` for the agent-executor tool loop, if wired."""
        if self.on_tool_finish:
            self.on_tool_finish(tool, arguments, result, raw, error, iteration)

    def emit_progress(
        self,
        message: str,
        percent: float | None = None,
        **data: Any,
    ) -> None:
        """Emit a user-facing progress signal.

        Called by application code — typically from a long-running
        tool — to report progress to the stream alongside model tokens.
        Safe to call when no runner is attached (``on_progress`` is
        ``None`` outside a flow): becomes a silent no-op so tools stay
        unit-testable.

        Args:
            message: Short human-readable status line. Rendered in UIs.
            percent: 0.0–1.0 for determinate progress, ``None`` (the
                default) for indeterminate / spinner-style. Values are
                passed through verbatim — clamp in the caller if needed.
            **data: Free-form structured fields (query strings, step
                indices, …) packed into a dict for the consumer.
        """
        if self.on_progress:
            self.on_progress(message, percent, dict(data))

    def emit_custom(
        self,
        name_or_event: str | Any,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an application-defined structured event.

        Prefer this over :meth:`emit_progress` when the event marks a
        discrete milestone (document retrieved, quota updated, cache
        hit, …) rather than a percent-along-a-task signal. Consumers
        filter on ``name`` via ``stream.custom(name=...)``.

        No-op when no runner is attached.

        **v0.4.0:** Accepts a :class:`TypedEvent` instance in addition
        to the original ``(name: str, payload: dict)`` form. When a
        ``TypedEvent`` is passed, the ``name`` and ``payload`` are
        extracted from the model automatically::

            ctx.emit_custom(SearchResultsEvent(count=5, query=q))

        Args:
            name_or_event: Either a string discriminator or a
                :class:`TypedEvent` instance. When a ``TypedEvent`` is
                passed, ``payload`` must be ``None``.
            payload: Free-form dict carried with the event. Defaults
                to an empty dict when ``None``. Ignored when
                ``name_or_event`` is a ``TypedEvent``.
        """
        if self.on_custom:
            # v0.4.0: duck-type check for TypedEvent — avoid importing
            # pydantic in the engine package. TypedEvent instances have
            # ``model_dump`` (Pydantic v2) and a ``name`` attribute.
            if hasattr(name_or_event, "model_dump") and not isinstance(name_or_event, str):
                event = name_or_event
                name = event.name
                data = event.model_dump(exclude={"name"})
            else:
                name = name_or_event
                data = dict(payload or {})
            self.on_custom(name, data)

    def emit_message(self, content: str) -> None:
        """Emit a complete message if a callback is registered."""
        if self.on_message:
            self.on_message(content)

    def update_status(self, status: NodeStatus) -> None:
        """Update this context's status and fire the callback."""
        self.status = status
        if self.on_status_change:
            self.on_status_change(status)
