"""Per-flow ``ExecutionContext`` reachable from inside tool calls.

This module exposes the ``qm.current_context()`` helper that application
code — typically ``@tool()``-decorated functions — uses to emit
progress / custom events back onto the stream::

    from quartermaster_tools import tool
    from quartermaster_sdk import current_context

    @tool()
    def slow_search(query: str) -> dict:
        ctx = current_context()
        if ctx is not None:
            ctx.emit_progress(f"searching '{query}'", percent=0.25)
        # ... do real work ...
        if ctx is not None:
            ctx.emit_progress("parsing results", percent=0.75)
        return {"results": [...]}

The contextvar is bound by :class:`FlowRunner._run_executor` around the
call to ``executor.execute(context)``; it stays ``None`` when no flow
is active, so tools remain safe to call from unit tests without a
runner.

Threading note
--------------
``FlowRunner._run_executor`` wraps ``executor.execute(context)`` in a
``ThreadPoolExecutor.submit(...)`` when the caller is inside a running
asyncio event loop. Python's ``contextvars.ContextVar`` values do NOT
automatically propagate into pool-worker threads — you must invoke
``contextvars.copy_context().run(...)`` from the worker. The runner
does this for us; the :func:`bind` helper here just sets the var in
the current context, trusting the runner to copy it across.

The contextvar lives in the engine (not the SDK) so the runner can
bind it without the engine importing from the SDK (which would flip
the dependency direction).
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quartermaster_engine.context.execution_context import ExecutionContext


# Module-level contextvar. ``None`` outside any flow — read via
# :func:`current_context` which returns ``None`` cleanly for unit tests
# and other out-of-flow callers.
_current_ctx: contextvars.ContextVar["ExecutionContext | None"] = contextvars.ContextVar(
    "quartermaster_current_context",
    default=None,
)


def current_context() -> "ExecutionContext | None":
    """Return the currently-executing flow's :class:`ExecutionContext`.

    Returns ``None`` when called outside of an active flow, including
    from unit tests and any code path that hasn't gone through
    :class:`FlowRunner`. Tools that emit progress should always
    null-check the return — the no-runner case is fully supported.
    """
    return _current_ctx.get()


@contextlib.contextmanager
def bind(context: "ExecutionContext") -> Iterator["ExecutionContext"]:
    """Bind *context* as the current contextvar for the duration of the
    ``with``-block.

    Called by :meth:`FlowRunner._run_executor` around
    ``executor.execute(context)``. The contextvar reset in the
    ``finally`` clause guarantees nested or concurrent flow runs don't
    leak state across each other — each flow sees the context bound by
    its own runner frame, regardless of thread hopping in-between.

    Yields the same context so callers can do
    ``with bind(ctx) as c: ...`` if they find the binding symmetric.
    """
    token = _current_ctx.set(context)
    try:
        yield context
    finally:
        _current_ctx.reset(token)


__all__ = ["current_context", "bind"]
