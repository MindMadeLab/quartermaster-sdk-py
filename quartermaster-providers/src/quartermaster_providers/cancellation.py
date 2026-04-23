"""Provider-level cooperative cancellation for streaming LLM calls.

v0.7.0. Motivation:

``qm.run.stream(graph, user_input)`` returns a generator. When the HTTP
client disconnects (browser AbortController, SSE client close), the
outer caller breaks out of the generator. The SDK propagates that via
``FlowRunner.stop(flow_id)`` which flips an asyncio event the running
executor polls via ``ctx.cancelled``.

Pre-v0.7.0 that flag only stopped the AGENT LOOP from scheduling the
*next* LLM iteration. The in-flight ``/v1/chat/completions`` stream
kept draining tokens from the server until the model finished
naturally — vLLM kept its slot occupied, the token bill kept ticking.

This module exposes a contextvar-based cancellation hook that the
provider's streaming path polls between chunks. When the engine sets
``set_cancel_check(lambda: ctx.cancelled)`` around the provider call,
the provider calls ``should_cancel()`` after each yielded chunk and
closes the openai ``AsyncStream`` (which in turn closes the underlying
``httpx.AsyncClient`` response) when cancellation is requested.

Contextvar rather than a kwarg on ``LLMConfig``:
- ``LLMConfig`` is serialisable (survives to_json / from_json); a
  callable wouldn't round-trip.
- ``ContextVar`` carries through asyncio.to_thread and copy_context()
  already (v0.5.0's parallel-tool-dispatch path relies on this), so
  cancellation propagates into worker threads without extra plumbing.
- Scoped push/pop means nested ``qm.run`` calls each see their own
  cancel check rather than the outermost one leaking through.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Callable, Iterator

#: The contextvar itself. ``None`` means "no cancel check installed" — the
#: default case for calls outside a ``qm.run.stream`` flow (one-shot
#: ``qm.instruction``, direct provider use in tests, etc.).
_cancel_check: ContextVar[Callable[[], bool] | None] = ContextVar("qm_cancel_check", default=None)


@contextlib.contextmanager
def set_cancel_check(check: Callable[[], bool] | None) -> Iterator[None]:
    """Install *check* for the duration of a ``with`` block.

    The engine wraps each streaming provider call with this manager,
    passing ``lambda: ctx.cancelled``. Inside the block, any code that
    calls :func:`should_cancel` observes the installed check; on exit
    we reset the previous value so nested flows behave correctly.

    ``check`` may be ``None`` — useful when a caller wants to
    explicitly strip cancellation for a sub-call (e.g. a cleanup path
    that must complete even after the outer flow was cancelled).
    """
    token = _cancel_check.set(check)
    try:
        yield
    finally:
        _cancel_check.reset(token)


def should_cancel() -> bool:
    """Return ``True`` iff a cancel check is installed AND reports True.

    Safe to call from any code path — returns ``False`` when no check
    is installed, so one-shot ``qm.instruction`` callers never trip it.
    Providers poll this between streaming chunks and close the response
    when it flips.
    """
    check = _cancel_check.get()
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        # A misbehaving predicate must not crash the streaming path —
        # treat a raising check as "don't cancel" so the stream keeps
        # flowing and the caller observes the exception on the next
        # natural check (e.g. a per-node error).
        return False
