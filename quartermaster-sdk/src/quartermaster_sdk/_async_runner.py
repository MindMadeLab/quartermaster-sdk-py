"""Async variant of the v0.2.0 graph runner.

Mirror of :mod:`quartermaster_sdk._runner` for ``async def`` call sites.
Django views, FastAPI endpoints, and other asyncio codebases can drop
``asgiref.sync.sync_to_async(qm.run)(...)`` in favour of the first-class
coroutine form:

    result = await qm.arun(graph, "Hello!")
    async for chunk in qm.arun.stream(graph, "Hello!"):
        if chunk.type == "token":
            print(chunk.content, end="")

No engine logic is duplicated — under the hood this is a thin adapter
over :class:`FlowRunner` that dispatches the synchronous execution onto
a worker thread (via :func:`asyncio.to_thread`) while the event loop
keeps servicing other tasks.

**Cancellation:** when the consumer's :class:`asyncio.Task` is cancelled,
``arun`` and ``arun.stream`` both reach into the still-running
:class:`FlowRunner` and call :meth:`FlowRunner.stop` with the pre-
generated ``flow_id``.  The engine checks ``flow_id in self._stopped``
in ``_execute_node`` before dispatching further work, so nodes already
mid-LLM-call finish their current request (no hard kill) but no new
nodes are scheduled — long agent loops unwind within a bounded time
instead of leaking API costs after the HTTP client disconnects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator
from uuid import uuid4

from quartermaster_engine import FlowEvent, FlowRunner, ImageInput, prepare_images

from . import _listeners
from ._chunks import Chunk, DoneChunk, ErrorChunk
from ._config import get_default_registry
from ._result import Result
from ._runner import _event_to_chunk, _resolve_graph
from ._stream_filters import _AsyncStream
from ._trace import Trace

if TYPE_CHECKING:
    from quartermaster_graph import GraphBuilder, GraphSpec
    from quartermaster_providers import ProviderRegistry

logger = logging.getLogger(__name__)


class _ARunCallable:
    """Callable + ``.stream`` method exposed as ``qm.arun``.

    Async analogue of :class:`_RunCallable` — same signature, same
    semantics, but every public method is a coroutine (or async
    iterator).  Shares :func:`_resolve_graph` and :func:`_event_to_chunk`
    with the sync path so event mapping stays in one place.
    """

    async def __call__(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> Result:
        """Execute *graph* against *user_input* and return a :class:`Result`.

        Args:
            graph: Either a :class:`GraphBuilder` (auto-finalised) or
                a pre-built :class:`GraphSpec`.
            user_input: Primary user message injected into the graph.
            image: Optional single image input for vision-capable graphs.
                Accepts raw ``bytes``, a :class:`pathlib.Path`, or a
                filesystem path string. Mutually exclusive with *images*.
            images: Optional list of image inputs. Mutually exclusive
                with *image*.
            provider_registry: Override the configured default registry.
            tool_registry: Optional :class:`quartermaster_tools.ToolRegistry`
                made available to ``agent()``-type nodes that specify
                ``tools=[...]``.

        **Cancellation:** if the awaiting task is cancelled, the running
        flow is stopped via :meth:`FlowRunner.stop` before the
        ``CancelledError`` propagates — nodes already dispatched finish
        their current LLM/tool call but no new work is scheduled.
        """
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()
        prepared_images = prepare_images(image=image, images=images)

        # v0.3.0 trace: accumulate every FlowEvent on the worker
        # thread so we can build a ``Trace`` and attach it to the
        # returned ``Result``. The list is populated from inside the
        # engine's worker thread and only READ after ``await
        # asyncio.to_thread`` resolves, so no cross-thread sync is
        # required.
        trace_events: list[FlowEvent] = []

        def _collect(event: FlowEvent) -> None:
            trace_events.append(event)
            _listeners.dispatch(event)

        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=tool_registry,
            # Forward every event to the global listener registry so
            # bolt-on instrumentation (e.g. ``qm.telemetry.instrument()``)
            # observes the same FlowEvent stream the streaming runner
            # gets — even though there's no consumer queue here.
            on_event=_collect,
        )
        # Pre-generate so the async cancel handler has something to stop,
        # even if the to_thread() call hasn't entered ``runner.run`` yet.
        flow_id = uuid4()

        started = time.perf_counter()
        try:
            # ``asyncio.to_thread`` forwards *args and **kwargs — keep
            # the image payload as a kwarg so the engine picks it up
            # via its named ``images=`` parameter.
            fr = await asyncio.to_thread(
                runner.run,
                user_input,
                images=prepared_images or None,
                flow_id=flow_id,
            )
        except asyncio.CancelledError:
            # ``asyncio.to_thread`` can't actually interrupt the worker
            # thread — the thread keeps executing until it voluntarily
            # returns — but calling ``runner.stop(flow_id)`` flips the
            # engine's ``_stopped`` gate so it won't dispatch any more
            # nodes.  The current node completes, the flow wraps up,
            # and the CancelledError re-raises to the caller.  Net
            # effect: bounded unwind instead of runaway API calls.
            try:
                runner.stop(flow_id)
            except Exception:  # pragma: no cover — defensive
                logger.exception("arun: runner.stop(%s) raised", flow_id)
            raise
        elapsed = time.perf_counter() - started

        result = Result.from_flow_result(fr)
        result.trace = Trace.from_events(
            trace_events,
            duration_seconds=fr.duration_seconds or elapsed,
        )
        return result

    def stream(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> _AsyncStream:
        """Run the graph and yield typed :class:`Chunk` events as they arrive.

        Returns an :class:`_AsyncStream` wrapper that is itself
        async-iterable (so existing ``async for chunk in
        qm.arun.stream(...)`` loops work unchanged) and exposes filter
        helpers — ``.tokens()``, ``.tool_calls()``, ``.progress()``,
        ``.custom(name=...)`` — for the common "pluck one chunk type
        out of the stream" patterns.

        Terminates with a :class:`DoneChunk` (on success) or
        :class:`ErrorChunk` (on unrecoverable failure).  The graph
        executes on a background thread; events flow back through an
        :class:`asyncio.Queue` so the event loop keeps servicing other
        tasks between chunks — no blocking ``next()`` inside the coro.

        **Cancellation:** if the async iterator is abandoned early
        (``break`` out of ``async for``, enclosing task cancelled, etc.)
        :meth:`FlowRunner.stop` is called on the pre-generated
        ``flow_id``.  The engine short-circuits on its next
        ``_execute_node`` dispatch — nodes mid-LLM-call finish their
        current request but no new work is scheduled, so long agent
        loops unwind within a bounded time instead of leaking API costs
        after the consumer goes away.
        """
        return _AsyncStream(self._aiter_chunks(
            graph=graph,
            user_input=user_input,
            image=image,
            images=images,
            provider_registry=provider_registry,
            tool_registry=tool_registry,
        ))

    async def _aiter_chunks(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        image: ImageInput | None = None,
        images: list[ImageInput] | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> AsyncIterator[Chunk]:
        """Inner async generator that powers :meth:`stream`.

        Kept private: callers always go through ``stream()`` which
        wraps the result in :class:`_AsyncStream` for filter support.
        """
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()
        prepared_images = prepare_images(image=image, images=images)

        # Capture the consumer's loop so the thread can marshal events
        # back to it via ``call_soon_threadsafe``.  ``run_coroutine_threadsafe``
        # would also work, but we want fire-and-forget ``put`` without
        # blocking the engine thread on asyncio internals.
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[FlowEvent | None] = asyncio.Queue()
        holder: dict[str, Any] = {}
        flow_id = uuid4()

        # v0.3.0 trace: accumulate events as they fire (on the engine
        # worker thread) so we can build a ``Trace`` before yielding
        # the terminal ``DoneChunk``. Append-only from the engine
        # thread, read-only once ``run_task`` has finished.
        trace_events: list[FlowEvent] = []

        def on_event(event: FlowEvent) -> None:
            # Fan out to bolt-on instrumentation (telemetry, custom
            # listeners, etc.) on the engine's worker thread BEFORE
            # marshalling the event back to the consumer's loop.
            _listeners.dispatch(event)
            trace_events.append(event)
            # Called from the engine's worker threads.  Hop back to the
            # consumer's loop so Queue.put() touches only loop-owned
            # state — ``asyncio.Queue`` is not thread-safe.
            loop.call_soon_threadsafe(q.put_nowait, event)

        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=tool_registry,
            on_event=on_event,
        )

        def _run_sync() -> None:
            """Runs on the ``to_thread`` worker — synchronous engine loop."""
            try:
                fr = runner.run(
                    user_input,
                    images=prepared_images or None,
                    flow_id=flow_id,
                )
                holder["result"] = fr
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("arun.stream: runner.run raised")
                holder["exception"] = exc
            finally:
                # Sentinel — unblocks the async iterator even on crash.
                loop.call_soon_threadsafe(q.put_nowait, None)

        started = time.perf_counter()
        run_task = asyncio.create_task(
            asyncio.to_thread(_run_sync), name="qm-arun-stream"
        )

        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                chunk = _event_to_chunk(event)
                if chunk is not None:
                    yield chunk
        finally:
            # Either the stream completed naturally, the caller
            # ``break``-ed early, or our task was cancelled.  In the
            # latter two cases the engine is still running — tell it to
            # stop so nodes stop getting dispatched.  ``run_task`` may
            # take up to one node's worth of runtime to actually finish,
            # which is the same bound the sync variant documents.
            if not run_task.done():
                try:
                    runner.stop(flow_id)
                except Exception:  # pragma: no cover — defensive
                    logger.exception(
                        "arun.stream: runner.stop(%s) raised", flow_id
                    )
                # Wait (bounded) for the worker thread to observe the
                # ``_stopped`` flag and return.  If it overshoots we let
                # the task leak rather than blocking the event loop
                # indefinitely — matches the 5s join() on the sync side.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(run_task), timeout=5.0
                    )
                except asyncio.TimeoutError:  # pragma: no cover — defensive
                    logger.warning(
                        "arun.stream: runner thread did not exit within 5s"
                    )
                except asyncio.CancelledError:
                    # Expected when the outer task was cancelled — let
                    # the shielded run_task keep going in the background
                    # (it will notice ``_stopped`` and bail on its next
                    # node).  Re-raise so the cancellation semantics
                    # propagate to the caller.
                    raise

        exception = holder.get("exception")
        fr = holder.get("result")

        if exception is not None:
            yield ErrorChunk(error=str(exception))
            return

        if fr is None:
            yield ErrorChunk(error="arun.stream: runner produced no result")
            return

        elapsed = time.perf_counter() - started
        result = Result.from_flow_result(fr)
        # v0.3.0: populate the structured trace from every FlowEvent
        # the engine dispatched. The ``on_event`` hook on the engine
        # thread appended each event as it fired; we build the
        # bucketed view once the run has finished and hand it to the
        # caller as part of the DoneChunk's Result.
        result.trace = Trace.from_events(
            trace_events,
            duration_seconds=fr.duration_seconds or elapsed,
        )
        yield DoneChunk(result=result)


#: Publicly exported async runner.  ``await qm.arun(graph, input)`` runs
#: the flow on a worker thread and awaits its result; ``async for
#: chunk in qm.arun.stream(graph, input)`` yields typed Chunk objects.
arun = _ARunCallable()


__all__ = ["arun"]
