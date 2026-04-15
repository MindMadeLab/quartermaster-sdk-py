"""OpenTelemetry instrumentation for Quartermaster flows.

Bolt-on tracing — call :func:`instrument` once at app boot and every
:func:`quartermaster_sdk.run` / :func:`quartermaster_sdk.arun` /
``run.stream`` will emit GenAI-conventioned OpenTelemetry spans without
any further callsite changes.

    import quartermaster_sdk as qm
    from quartermaster_sdk import telemetry

    qm.configure(provider="ollama", default_model="gemma4:26b")
    telemetry.instrument()                     # uses global tracer provider
    # or with explicit provider:
    # telemetry.instrument(tracer_provider=my_provider)

    result = qm.run(graph, "hi")               # spans flow to your exporter

The implementation hooks into the SDK's :mod:`._listeners` registry,
which the runners already invoke from inside their ``on_event=`` callback
to :class:`FlowRunner`.  We never touch the engine itself.

Span model
----------

* One root ``qm.flow`` span per flow_id, opened on the first event we
  see for that flow and closed on :class:`~quartermaster_engine.FlowFinished`
  / :class:`~quartermaster_engine.FlowError`.
* One ``qm.node.<node_name>`` span per node, opened on
  :class:`~quartermaster_engine.NodeStarted`, closed on
  :class:`~quartermaster_engine.NodeFinished`.  Parented to the flow span.
* One ``qm.tool.<tool>`` span per tool call (matched by ``iteration``),
  opened on :class:`~quartermaster_engine.ToolCallStarted`, closed on
  :class:`~quartermaster_engine.ToolCallFinished`.  Parented to the
  enclosing node span.
* :class:`~quartermaster_engine.TokenGenerated` /
  :class:`~quartermaster_engine.ProgressEvent` /
  :class:`~quartermaster_engine.CustomEvent` are recorded as span events
  on the active node span — never as their own spans (would be far too
  noisy at LLM token granularity).

Attributes follow the OpenTelemetry GenAI semantic conventions:
``gen_ai.system``, ``gen_ai.operation.name``, ``gen_ai.tool.name``,
``gen_ai.tool.call.arguments``, ``gen_ai.usage.input_tokens``,
``gen_ai.usage.output_tokens``.  See
https://opentelemetry.io/docs/specs/semconv/gen-ai/ .

Scope & limitations
-------------------

* Single-process only.  The active-span dict is module-level state
  guarded by a :class:`threading.Lock`; cross-process distribution is
  the responsibility of whatever exporter the caller wires up
  (e.g. OTLP → Jaeger / Tempo / Honeycomb).
* The OpenTelemetry SDK is an *optional* dependency.  Install with
  ``pip install 'quartermaster-sdk[telemetry]'``.  Importing this module
  works without OTEL installed; only :func:`instrument` requires it.
"""

from __future__ import annotations

import json
import logging
import threading
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
)

from . import _listeners

logger = logging.getLogger(__name__)

# ── Module-level state ───────────────────────────────────────────────
#
# Active spans are tracked here; we cannot rely on
# ``opentelemetry.trace.use_span`` context-manager scoping because the
# FlowEvent stream is delivered out-of-band from a worker thread, after
# the event-source code has already returned.  All access goes through
# ``_state_lock`` so concurrent flows on different threads don't trample
# each other.
_state_lock = threading.Lock()

# Set when ``instrument()`` succeeds; tested by ``uninstrument()`` and
# by ``instrument()`` itself to avoid double-registration.
_listener: Any = None  # callable[[FlowEvent], None]
_tracer: Any = None  # opentelemetry.trace.Tracer

# Open spans, keyed by stable identifiers we synthesise from event fields.
# - flow_id (UUID)            → root flow span
# - (flow_id, node_id) (UUIDs) → per-node span
# - (flow_id, node_id, "tool", iteration: int) → per-tool-call span
_active_spans: dict[Any, Any] = {}

# OTEL semantic-convention constants.  We hard-code rather than import
# from ``opentelemetry.semconv`` because the GenAI SemConv module moved
# packages a few times in the 1.2x → 1.4x window — string literals are
# simpler and don't pin us to a specific SDK version.
_GEN_AI_SYSTEM = "gen_ai.system"
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
_GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Quartermaster-specific attributes (custom prefix ``qm.*`` so they
# don't collide with future SemConv reservations).
_QM_FLOW_ID = "qm.flow.id"
_QM_NODE_ID = "qm.node.id"
_QM_NODE_NAME = "qm.node.name"
_QM_NODE_TYPE = "qm.node.type"


def _safe_json(value: Any) -> str:
    """Serialise *value* to JSON, falling back to ``repr`` on failure.

    Tool-call arguments can contain anything the LLM dreamt up — bytes,
    custom objects, NaN floats — and ``json.dumps`` choking would leak
    the exception into the engine thread and abort the listener.  This
    helper keeps the listener bullet-proof while still getting the
    common case (plain dict of primitives) right.
    """
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover — defensive
        return repr(value)


def _flow_key(flow_id: UUID) -> UUID:
    """Stable key for the root flow span — just the flow_id."""
    return flow_id


def _node_key(flow_id: UUID, node_id: UUID) -> tuple[UUID, UUID]:
    """Stable key for a per-node span."""
    return (flow_id, node_id)


def _tool_key(flow_id: UUID, node_id: UUID, iteration: int) -> tuple:
    """Stable key for a per-tool-call span (one per agent iteration)."""
    return (flow_id, node_id, "tool", iteration)


def _ensure_flow_span(event: FlowEvent) -> Any:
    """Open the root ``qm.flow`` span on first sight of *event.flow_id*.

    Idempotent — returns the existing span if we've already seen this
    flow.  Caller holds ``_state_lock``.
    """
    key = _flow_key(event.flow_id)
    span = _active_spans.get(key)
    if span is not None:
        return span
    span = _tracer.start_span("qm.flow")
    span.set_attribute(_GEN_AI_SYSTEM, "quartermaster")
    span.set_attribute(_GEN_AI_OPERATION_NAME, "flow")
    span.set_attribute(_QM_FLOW_ID, str(event.flow_id))
    _active_spans[key] = span
    return span


def _start_node_span(event: NodeStarted) -> None:
    """Open a child span for a node beginning execution."""
    # OTEL needs explicit parent context for cross-thread parent linkage
    # (the runner's worker thread doesn't share the OTEL context-vars
    # with whoever called ``instrument()``).  We attach a context whose
    # active span is the flow span so the new node span gets parented
    # correctly even though we're outside the original flow context.
    from opentelemetry import trace

    with _state_lock:
        flow_span = _ensure_flow_span(event)
        ctx = trace.set_span_in_context(flow_span)
        span = _tracer.start_span(
            f"qm.node.{event.node_name or event.node_id}",
            context=ctx,
        )
        span.set_attribute(_GEN_AI_SYSTEM, "quartermaster")
        # The engine's NodeType is a string-enum; .value gives the
        # serialised form integrators see in their graph definitions.
        span.set_attribute(
            _GEN_AI_OPERATION_NAME,
            event.node_type.value if event.node_type is not None else "unknown",
        )
        span.set_attribute(_QM_NODE_NAME, event.node_name)
        span.set_attribute(_QM_NODE_ID, str(event.node_id))
        span.set_attribute(_QM_NODE_TYPE, event.node_type.value)
        _active_spans[_node_key(event.flow_id, event.node_id)] = span


def _finish_node_span(event: NodeFinished) -> None:
    """Close the matching node span and surface usage attributes."""
    with _state_lock:
        span = _active_spans.pop(_node_key(event.flow_id, event.node_id), None)
    if span is None:
        # Missed the start (instrument() called mid-flight, or duplicate
        # finish) — nothing to close.
        return
    # Pull token-usage hints from output_data when the node populated
    # them.  Different node types report different shapes; we look for
    # the common keys without mandating any particular schema.
    output = event.output_data or {}
    in_tok = (
        output.get("input_tokens")
        or output.get("prompt_tokens")
        or output.get("usage", {}).get("input_tokens")
        if isinstance(output.get("usage"), dict)
        else None
    )
    out_tok = (
        output.get("output_tokens")
        or output.get("completion_tokens")
        or output.get("usage", {}).get("output_tokens")
        if isinstance(output.get("usage"), dict)
        else None
    )
    if isinstance(in_tok, int):
        span.set_attribute(_GEN_AI_USAGE_INPUT_TOKENS, in_tok)
    if isinstance(out_tok, int):
        span.set_attribute(_GEN_AI_USAGE_OUTPUT_TOKENS, out_tok)
    span.end()


def _start_tool_span(event: ToolCallStarted) -> None:
    """Open a tool span nested under the agent-node span that issued it."""
    from opentelemetry import trace

    with _state_lock:
        # Parent the tool span on the enclosing node span when we have
        # one; fall back to the flow span otherwise.  Direct reach into
        # ``_active_spans`` is fine — we're under ``_state_lock``.
        parent = _active_spans.get(_node_key(event.flow_id, event.node_id))
        if parent is None:
            parent = _active_spans.get(_flow_key(event.flow_id))
        ctx = trace.set_span_in_context(parent) if parent is not None else None
        span = _tracer.start_span(
            f"qm.tool.{event.tool}",
            context=ctx,
        )
        span.set_attribute(_GEN_AI_SYSTEM, "quartermaster")
        span.set_attribute(_GEN_AI_OPERATION_NAME, "execute_tool")
        span.set_attribute(_GEN_AI_TOOL_NAME, event.tool)
        span.set_attribute(
            _GEN_AI_TOOL_CALL_ARGUMENTS,
            _safe_json(dict(event.arguments or {})),
        )
        _active_spans[_tool_key(event.flow_id, event.node_id, event.iteration)] = span


def _finish_tool_span(event: ToolCallFinished) -> None:
    """Close the tool span and mark ERROR status if the tool raised."""
    from opentelemetry.trace import Status, StatusCode

    with _state_lock:
        span = _active_spans.pop(
            _tool_key(event.flow_id, event.node_id, event.iteration),
            None,
        )
    if span is None:
        return
    if event.error:
        span.set_status(Status(StatusCode.ERROR, event.error))
        span.set_attribute("error.message", event.error)
    span.end()


def _record_event_on_active_span(event: FlowEvent, name: str, attrs: dict[str, Any]) -> None:
    """Add an OTEL event to the active node span (preferred) or flow span."""
    with _state_lock:
        node_id = getattr(event, "node_id", None)
        span = None
        if node_id is not None:
            span = _active_spans.get(_node_key(event.flow_id, node_id))
        if span is None:
            span = _active_spans.get(_flow_key(event.flow_id))
    if span is None:
        return
    # OTEL attributes must be primitives or sequences of primitives —
    # stringify anything dict-shaped via _safe_json.
    safe_attrs: dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, (str, bool, int, float)):
            safe_attrs[k] = v
        elif v is None:
            continue
        else:
            safe_attrs[k] = _safe_json(v)
    try:
        span.add_event(name, attributes=safe_attrs)
    except Exception:  # pragma: no cover — defensive
        logger.exception("telemetry: add_event(%r) failed", name)


def _finish_flow_span(event: FlowEvent, *, error: str | None = None) -> None:
    """Close the root flow span (and any orphan child spans for that flow)."""
    from opentelemetry.trace import Status, StatusCode

    with _state_lock:
        flow_span = _active_spans.pop(_flow_key(event.flow_id), None)
        # Mop up any node / tool spans we never saw a finish for.
        # They'd otherwise leak forever — bad for memory and for
        # exporters that batch on span end.
        orphans: list[Any] = []
        for key in list(_active_spans):
            if isinstance(key, tuple) and key and key[0] == event.flow_id:
                orphans.append(_active_spans.pop(key))
    for orphan in orphans:
        try:
            orphan.end()
        except Exception:  # pragma: no cover — defensive
            logger.exception("telemetry: failed to end orphan span")
    if flow_span is None:
        return
    if error:
        flow_span.set_status(Status(StatusCode.ERROR, error))
        flow_span.set_attribute("error.message", error)
    flow_span.end()


def _on_event(event: FlowEvent) -> None:
    """The single listener registered on :mod:`_listeners`."""
    try:
        # Lazy: even FlowFinished should ensure a flow span exists in
        # the pathological case of an empty flow that emits only the
        # terminal event — without this, _finish_flow_span pops nothing
        # and we lose the trace.
        if not isinstance(event, (FlowFinished, FlowError)):
            with _state_lock:
                _ensure_flow_span(event)

        if isinstance(event, NodeStarted):
            _start_node_span(event)
        elif isinstance(event, NodeFinished):
            _finish_node_span(event)
        elif isinstance(event, ToolCallStarted):
            _start_tool_span(event)
        elif isinstance(event, ToolCallFinished):
            _finish_tool_span(event)
        elif isinstance(event, TokenGenerated):
            # Recorded as a span event on the node — one-per-token would
            # blow out exporters at LLM throughput, so we attach the
            # token to the active node span as an event with the token
            # text as an attribute.  Most exporters dedupe by event name
            # and surface counts in their UI.
            _record_event_on_active_span(
                event, "token", {"gen_ai.token.value": event.token}
            )
        elif isinstance(event, ProgressEvent):
            _record_event_on_active_span(
                event,
                "progress",
                {
                    "message": event.message,
                    "percent": event.percent,
                    **(event.data or {}),
                },
            )
        elif isinstance(event, CustomEvent):
            _record_event_on_active_span(
                event,
                event.name or "custom",
                dict(event.payload or {}),
            )
        elif isinstance(event, FlowError):
            _finish_flow_span(event, error=event.error)
        elif isinstance(event, FlowFinished):
            _finish_flow_span(event)
    except Exception:  # pragma: no cover — defensive
        logger.exception("telemetry: listener crashed on %r", type(event).__name__)


def instrument(
    *,
    tracer_provider: Any | None = None,
    instrumenting_module_name: str = "quartermaster_sdk",
) -> None:
    """Register the OTEL listener; every subsequent flow emits spans.

    Args:
        tracer_provider: An :class:`opentelemetry.sdk.trace.TracerProvider`
            (or compatible).  Defaults to the global provider returned
            by :func:`opentelemetry.trace.get_tracer_provider`, so most
            integrators don't need to pass anything.
        instrumenting_module_name: Library name reported on each span;
            shows up as ``otel.library.name`` in many exporters.

    Idempotent: calling :func:`instrument` more than once is a no-op
    until :func:`uninstrument` is called.
    """
    global _listener, _tracer

    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover — depends on extras
        raise ImportError(
            "OpenTelemetry not installed. "
            "Run: pip install 'quartermaster-sdk[telemetry]'"
        ) from exc

    with _state_lock:
        if _listener is not None:
            # Already instrumented — caller probably double-bootstrapped
            # something.  No harm done; just bail.
            return
        provider = tracer_provider or trace.get_tracer_provider()
        _tracer = provider.get_tracer(instrumenting_module_name)
        _listener = _on_event

    _listeners.register(_on_event)


def uninstrument() -> None:
    """Reverse of :func:`instrument` — remove the OTEL listener.

    Any in-flight spans are left as they are; they'll close naturally
    when their flow finishes.  Calling :func:`uninstrument` before
    :func:`instrument` (or twice in a row) is a no-op.
    """
    global _listener, _tracer
    with _state_lock:
        if _listener is None:
            return
        listener = _listener
        _listener = None
        _tracer = None
    _listeners.unregister(listener)


__all__ = ["instrument", "uninstrument"]
