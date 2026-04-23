"""Cooperative cancellation primitives.

The engine-side home of the :class:`Cancelled` exception. Tools inside
an :class:`AgentExecutor` loop that want to bail out mid-work raise
this to signal the flow has been cancelled — the executor treats it
like a node failure but with a distinct ``error="cancelled"`` string so
SDK consumers can tell it apart from an ordinary crash.

The SDK re-exports this same class as ``quartermaster_sdk.Cancelled``;
identity is preserved (``is`` checks across the boundary hold) because
the SDK imports this module directly.

Example — a tool that aborts as soon as the stream consumer goes away::

    from quartermaster_sdk import current_context, Cancelled

    @tool()
    def slow_search(query: str) -> dict:
        ctx = current_context()
        if ctx and ctx.cancelled:
            raise Cancelled("flow aborted by consumer")
        # ... real work ...

The flag that drives ``ctx.cancelled`` is a per-flow
:class:`threading.Event` the runner populates on every
:class:`ExecutionContext` it builds for the flow; see
:class:`ExecutionContext._cancelled_event`.
"""

from __future__ import annotations


class Cancelled(Exception):
    """Raised by application code to abort the enclosing flow.

    Treated as a node failure by :class:`AgentExecutor`, surfaced to
    SDK consumers as ``ErrorChunk(error="cancelled", ...)`` so they can
    distinguish a cooperative abort from a genuine crash.

    Tools don't *have* to raise this — the SDK's stream context-manager
    already tells the runner to stop on exit, which short-circuits
    ``_execute_node`` dispatch — but raising it lets a tool deep in a
    long-running loop unwind immediately instead of waiting for the
    next node dispatch to short-circuit.
    """


__all__ = ["Cancelled"]
