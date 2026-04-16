"""Structured post-mortem trace for a Quartermaster flow run.

Every :class:`quartermaster_sdk.Result` returned from :func:`run` or the
terminal :class:`DoneChunk.result` of :func:`run.stream` carries a
:class:`Trace` built from the full :class:`FlowEvent` stream the runner
observed.  The trace gives the caller post-mortem access to everything
that happened during the run — tokens, tool calls, progress, custom
events — without having to re-iterate the event stream themselves::

    result = qm.run(graph, "hi")
    result.trace.text                       # full model output
    result.trace.tool_calls                 # list[dict] tool-call records
    result.trace.progress                   # list[ProgressEvent]
    result.trace.custom(name="docs")        # filtered custom events
    result.trace.by_node["research"].text   # tokens for a single node
    result.trace.as_jsonl()                 # JSONL export for logs / fixtures

The same shape is populated for streaming runs — the runner installs an
``on_event`` collector that appends every :class:`FlowEvent` to a list,
then builds the trace from that list once the run completes.

Per-node bucketing
------------------
:attr:`Trace.by_node` is keyed by node *name* (the ``node_name`` string
on :class:`NodeStarted` / :class:`NodeFinished`), not by UUID.  Events
that carry a ``node_name`` (``NodeStarted``, ``NodeFinished``,
``TokenGenerated``, ``ToolCallStarted``, ``ToolCallFinished``,
``ProgressEvent``, ``CustomEvent``) are attributed to the node that was
active when they fired.  Flow-scoped events (``FlowFinished``,
``FlowError``) don't belong to any single node and land in a synthetic
``"_flow"`` bucket so they're still reachable without polluting the
per-node views.

The "active node" is tracked by walking events in arrival order and
latching ``NodeStarted.node_name`` until the matching
``NodeFinished.node_name`` closes it — events that fire between those
bookends (tokens, tool calls, progress, custom) inherit that node's
name.  Events emitted *before* any ``NodeStarted`` (rare, but possible
for flow-level setup) land in ``"_flow"`` as well.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from uuid import UUID

from quartermaster_engine import (
    CustomEvent,
    FlowError,
    FlowEvent,
    FlowFinished,
    NodeFinished,
    NodeStarted,
    ProgressEvent,
    TokenGenerated,
    ToolCallFinished,
    ToolCallStarted,
    UserInputRequired,
)


# ── Event (de)serialisation helpers ────────────────────────────────────

#: Registry mapping ``_event_type`` discriminator strings to their
#: :class:`FlowEvent` subclasses.  Used by :func:`_event_from_dict` to
#: reconstruct the correct subclass when loading JSONL.
_EVENT_CLASSES: dict[str, type[FlowEvent]] = {
    cls.__name__: cls
    for cls in (
        NodeStarted,
        NodeFinished,
        TokenGenerated,
        FlowFinished,
        FlowError,
        UserInputRequired,
        ToolCallStarted,
        ToolCallFinished,
        ProgressEvent,
        CustomEvent,
    )
}


def _coerce_field(cls: type, name: str, value: Any) -> Any:
    """Best-effort coercion of a JSON-deserialised *value* back to the
    type expected by the dataclass *cls* for field *name*.

    Handles the most common cases — ``UUID`` from string, enum from
    value — and returns *value* unchanged when no coercion is needed.
    """
    field_map = {f.name: f for f in fields(cls)}
    if name not in field_map:
        return value

    ftype = field_map[name].type
    # Resolve string annotations produced by ``from __future__ import annotations``.
    if isinstance(ftype, str):
        # Simple resolution for the types we actually encounter.
        if ftype == "UUID" or ftype == "uuid.UUID":
            ftype = UUID
        else:
            return value

    if ftype is UUID and isinstance(value, str):
        return UUID(value)
    # Enum fields (e.g. NodeType) — attempt by-value reconstruction.
    if isinstance(ftype, type) and issubclass(ftype, __import__("enum").Enum):
        try:
            return ftype(value)
        except (ValueError, KeyError):
            return value
    return value


def _event_from_dict(d: dict[str, Any]) -> FlowEvent:
    """Reconstruct a :class:`FlowEvent` subclass instance from a dict
    produced by :meth:`Trace.as_jsonl`.

    The dict must carry an ``"_event_type"`` key whose value is the
    class name of the original event (e.g. ``"NodeStarted"``).

    Raises:
        ValueError: if ``_event_type`` is missing or unknown.
    """
    d = dict(d)  # shallow copy — we pop from it
    event_type = d.pop("_event_type", None)
    if event_type is None:
        raise ValueError("Cannot reconstruct FlowEvent: dict has no '_event_type' key")
    cls = _EVENT_CLASSES.get(event_type)
    if cls is None:
        raise ValueError(
            f"Unknown _event_type {event_type!r}; known types: {sorted(_EVENT_CLASSES)}"
        )
    # Only pass keys that the dataclass actually declares — extra keys
    # (e.g. from future schema evolution) are silently dropped.
    valid_names = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k in valid_names:
            kwargs[k] = _coerce_field(cls, k, v)
    return cls(**kwargs)


#: Synthetic bucket key for events that don't belong to any single node
#: (flow-level events like :class:`FlowFinished` / :class:`FlowError`,
#: and any event fired before the first :class:`NodeStarted`).
_FLOW_BUCKET = "_flow"


def _extract_tool_call(event: ToolCallFinished) -> dict:
    """Translate a :class:`ToolCallFinished` event into the public dict
    shape used by :attr:`Trace.tool_calls` and
    :attr:`NodeTrace.tool_calls`.

    Mirrors the ``tool_calls`` entry already written to
    :attr:`NodeResult.data` by ``AgentExecutor`` so that both surfaces
    (post-hoc trace inspection and in-flight capture introspection)
    carry identical keys.
    """
    return {
        "tool": event.tool,
        "arguments": dict(event.arguments),
        "result": event.result,
        "raw": event.raw,
        "error": event.error,
        "iteration": event.iteration,
    }


@dataclass
class NodeTrace:
    """Per-node slice of a :class:`Trace`.

    Scoped to a single node (the one whose name keys it in
    :attr:`Trace.by_node`).  Same accessor shape as :class:`Trace` so
    callers can write ``result.trace.by_node["research"].text`` the same
    way they write ``result.trace.text``.

    Attributes:
        node_name: The ``node_name`` string this trace belongs to.
        events: Every :class:`FlowEvent` attributed to this node, in
            arrival order.
    """

    node_name: str
    events: list[FlowEvent] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenation of every :class:`TokenGenerated.token` for this node."""
        return "".join(
            event.token for event in self.events if isinstance(event, TokenGenerated)
        )

    @property
    def tool_calls(self) -> list[dict]:
        """Tool-call records derived from this node's
        :class:`ToolCallFinished` events.

        Each dict has keys ``tool``, ``arguments``, ``result``, ``raw``,
        ``error``, ``iteration`` — identical shape to the entries in
        ``result["<agent_node>"].data["tool_calls"]`` so downstream code
        can move between the trace and the capture without remapping.
        """
        return [
            _extract_tool_call(event)
            for event in self.events
            if isinstance(event, ToolCallFinished)
        ]

    @property
    def progress(self) -> list[ProgressEvent]:
        """Every :class:`ProgressEvent` emitted from inside this node."""
        return [event for event in self.events if isinstance(event, ProgressEvent)]

    def custom(self, name: str | None = None) -> list[CustomEvent]:
        """Return every :class:`CustomEvent` for this node, optionally
        filtered by ``name``.

        ``name=None`` (the default) returns every custom event; a
        specific ``name`` returns only the matching ones so callers can
        subscribe to a single milestone stream without inspecting every
        payload.
        """
        return [
            event
            for event in self.events
            if isinstance(event, CustomEvent) and (name is None or event.name == name)
        ]


@dataclass
class Trace:
    """Structured view of everything that happened during a flow run.

    Populated automatically on every :class:`Result` — both sync
    (:func:`run`) and streaming (:class:`DoneChunk.result`) surfaces
    attach the same object so downstream code doesn't branch on the
    run mode.  Events are captured via the runner's ``on_event``
    callback, which fires for every :class:`FlowEvent` the engine
    emits.

    Attributes:
        events: Every :class:`FlowEvent` in arrival order.  This is the
            source of truth — all aggregate views derive from it.
        by_node: Per-node slices of :attr:`events`, keyed by
            ``node_name``.  Flow-scoped events (no ``node_name``) are
            collected into a synthetic ``"_flow"`` bucket so the caller
            can still reach them without them clogging the per-node
            views.
        duration_seconds: Wall-clock duration of the run, mirroring
            :attr:`Result.duration_seconds`.
    """

    events: list[FlowEvent] = field(default_factory=list)
    by_node: dict[str, NodeTrace] = field(default_factory=dict)
    duration_seconds: float = 0.0
    user_input: str | None = None

    @property
    def text(self) -> str:
        """Concatenation of every :class:`TokenGenerated.token`, in order.

        Equivalent to walking the event stream and gluing ``token``
        strings together — the "full model output" view regardless of
        how many nodes contributed tokens.
        """
        return "".join(
            event.token for event in self.events if isinstance(event, TokenGenerated)
        )

    @property
    def tool_calls(self) -> list[dict]:
        """Every tool-call record across every node.

        Each dict has keys ``tool``, ``arguments``, ``result``, ``raw``,
        ``error``, ``iteration``.  Use :attr:`NodeTrace.tool_calls`
        via :attr:`by_node` if you need to scope to a single agent.
        """
        return [
            _extract_tool_call(event)
            for event in self.events
            if isinstance(event, ToolCallFinished)
        ]

    @property
    def progress(self) -> list[ProgressEvent]:
        """Every :class:`ProgressEvent` across the whole run."""
        return [event for event in self.events if isinstance(event, ProgressEvent)]

    def custom(self, name: str | None = None) -> list[CustomEvent]:
        """Return every :class:`CustomEvent`, optionally filtered by ``name``.

        ``name=None`` yields every custom event; a specific ``name``
        yields only the matching ones so UIs can subscribe to a single
        milestone stream without inspecting every payload.
        """
        return [
            event
            for event in self.events
            if isinstance(event, CustomEvent) and (name is None or event.name == name)
        ]

    def as_jsonl(self, *, user_input: str | None = None) -> str:
        """Return the trace serialised as JSONL (one event per line).

        Uses ``dataclasses.asdict`` to flatten each event, then
        ``json.dumps(..., default=str)`` to coerce ``UUID`` /
        ``datetime`` / enum values to strings.  Each event dict
        carries an ``"_event_type"`` discriminator key (the class
        name) so that :meth:`from_jsonl` can reconstruct the correct
        :class:`FlowEvent` subclass.

        When *user_input* is given (or was stashed on the trace via
        :attr:`user_input`), a synthetic header line is prepended::

            {"_meta": "trace_header", "user_input": "..."}

        The output is suitable for log shipping, test fixtures, and
        replay workflows via :meth:`from_jsonl`.
        """
        lines: list[str] = []
        # Resolve user_input: explicit kwarg > stashed attribute.
        ui = user_input if user_input is not None else self.user_input
        if ui is not None:
            lines.append(json.dumps({"_meta": "trace_header", "user_input": ui}))
        for event in self.events:
            d = asdict(event)
            d["_event_type"] = type(event).__name__
            lines.append(json.dumps(d, default=str))
        return "\n".join(lines)

    @classmethod
    def from_events(
        cls, events: list[FlowEvent], duration_seconds: float = 0.0
    ) -> Trace:
        """Build a :class:`Trace` from a list of :class:`FlowEvent`.

        Walks ``events`` in arrival order, latching the currently-active
        node name from :class:`NodeStarted` / :class:`NodeFinished` so
        events fired between those bookends (tokens, tool calls,
        progress, custom) are attributed to the right
        :class:`NodeTrace` in :attr:`by_node`.

        Flow-scoped events (:class:`FlowFinished`, :class:`FlowError`,
        :class:`UserInputRequired` — they have no ``node_name``) land
        in the synthetic ``"_flow"`` bucket alongside any events fired
        before the first :class:`NodeStarted`.
        """
        by_node: dict[str, NodeTrace] = {}
        current_node: str | None = None

        def bucket(name: str) -> NodeTrace:
            """Fetch-or-create the :class:`NodeTrace` for ``name``."""
            if name not in by_node:
                by_node[name] = NodeTrace(node_name=name)
            return by_node[name]

        for event in events:
            # Decide which bucket this event belongs to.
            #
            # NodeStarted sets the "active" node for subsequent events
            # (tokens, tool calls, progress, custom) and is itself
            # stored under that node.  NodeFinished closes the bucket
            # and lands in it too.  Anything that carries an explicit
            # node_name (rare — most events use the latched value) wins
            # over the latched state.
            if isinstance(event, NodeStarted):
                current_node = event.node_name
                bucket(current_node).events.append(event)
            elif isinstance(event, NodeFinished):
                # NodeFinished has ``node_name`` on the engine side —
                # but defensively fall back to the latched value if an
                # older engine event is missing it.
                name = getattr(event, "node_name", None) or current_node
                if name:
                    bucket(name).events.append(event)
                else:
                    bucket(_FLOW_BUCKET).events.append(event)
                # Close the active node — subsequent events (until the
                # next NodeStarted) are flow-scoped.
                current_node = None
            elif isinstance(
                event,
                (
                    TokenGenerated,
                    ToolCallStarted,
                    ToolCallFinished,
                    ProgressEvent,
                    CustomEvent,
                ),
            ):
                # Node-scoped events.  Use the latched node name; fall
                # back to the flow bucket for the edge case where the
                # event fires outside any NodeStarted/NodeFinished pair
                # (e.g. a misconfigured node, or a caller emitting from
                # flow-level setup code).
                name = current_node or _FLOW_BUCKET
                bucket(name).events.append(event)
            else:
                # FlowFinished, FlowError, UserInputRequired, or any
                # future FlowEvent subtype that isn't node-scoped.
                bucket(_FLOW_BUCKET).events.append(event)

        return cls(
            events=list(events),
            by_node=by_node,
            duration_seconds=duration_seconds,
        )


__all__ = ["Trace", "NodeTrace"]
