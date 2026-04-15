"""Process-global ``FlowEvent`` listener registry.

Lets the SDK fan out a single :class:`FlowEvent` to many subscribers
without each subscriber needing to thread its own ``on_event=`` callback
through every ``run`` / ``arun`` / ``run.stream`` call site.

This is the seam :mod:`quartermaster_sdk.telemetry` (and any other
bolt-on instrumentation) hooks into via :func:`register` / :func:`unregister`.
The two SDK runners (:mod:`._runner` and :mod:`._async_runner`) call
:func:`dispatch` from inside the existing ``on_event`` callback they
already pass to :class:`FlowRunner`, so we never have to reach into the
engine itself to install a global handler.

**Scope:** single-process. The list lives in module-level state guarded
by a :class:`threading.Lock`; cross-process fan-out (e.g. distributed
tracing across worker pools) is the responsibility of whatever exporter
the listener wires up to.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from quartermaster_engine import FlowEvent

logger = logging.getLogger(__name__)

#: Registered listeners. Mutated under :data:`_lock`; read under the
#: lock too, then iterated outside the critical section so a slow
#: handler can't stall a concurrent ``register`` / ``unregister`` call.
_global_listeners: list[Callable[[FlowEvent], None]] = []
_lock = threading.Lock()


def register(fn: Callable[[FlowEvent], None]) -> None:
    """Add *fn* to the process-global listener list.

    Idempotent: registering the same callable twice is a no-op so
    callers don't accidentally double-subscribe (which would emit
    duplicate spans on every event).
    """
    with _lock:
        if fn not in _global_listeners:
            _global_listeners.append(fn)


def unregister(fn: Callable[[FlowEvent], None]) -> None:
    """Remove *fn* from the listener list. Silent if not registered."""
    with _lock:
        try:
            _global_listeners.remove(fn)
        except ValueError:
            pass


def dispatch(event: FlowEvent) -> None:
    """Fan out *event* to every registered listener.

    Exceptions raised by a listener are logged and swallowed — one
    misbehaving subscriber must not break the SDK runner or starve
    other subscribers of the same event.
    """
    # Snapshot under the lock so we never iterate a list someone is
    # mutating; emit outside the lock so a slow listener can't block
    # ``register`` / ``unregister`` from the instrumenting code.
    with _lock:
        listeners = list(_global_listeners)
    for fn in listeners:
        try:
            fn(event)
        except Exception:  # pragma: no cover — defensive
            logger.exception("Listener %r raised on %r", fn, type(event).__name__)


def clear() -> None:
    """Drop every listener. Test-only helper; not part of the public API."""
    with _lock:
        _global_listeners.clear()


__all__ = ["register", "unregister", "dispatch", "clear"]
