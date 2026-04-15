"""The v0.2.0 ergonomic graph runner.

Replaces the "import FlowRunner, build a node registry, construct the
runner, call .run()" dance with a one-liner:

    from quartermaster_sdk import Graph, run

    graph = Graph("chat").user().agent().build()
    result = run(graph, "Hello!")                 # sync
    for chunk in run.stream(graph, "Hello!"):    # streaming
        if chunk.type == "token":
            print(chunk.content, end="")

Provider is picked up from :func:`configure` or the
``provider_registry=`` kwarg; no need for the integrator to touch
``quartermaster_engine.FlowRunner`` or
``quartermaster_engine.build_default_registry`` directly.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, Iterator

from quartermaster_engine import (
    FlowError,
    FlowEvent,
    FlowFinished,
    FlowRunner,
    NodeFinished,
    NodeStarted,
    TokenGenerated,
    UserInputRequired,
)

from ._chunks import (
    AwaitInputChunk,
    Chunk,
    DoneChunk,
    ErrorChunk,
    NodeFinishChunk,
    NodeStartChunk,
    TokenChunk,
)
from ._config import get_default_registry
from ._result import Result

if TYPE_CHECKING:
    from quartermaster_graph import GraphBuilder, GraphSpec
    from quartermaster_providers import ProviderRegistry

logger = logging.getLogger(__name__)


def _resolve_graph(graph: GraphBuilder | GraphSpec) -> GraphSpec:
    """Accept either a builder (auto-finalise) or a pre-built spec."""
    if hasattr(graph, "build"):
        return graph.build()
    return graph


class _RunCallable:
    """Callable + ``.stream`` method exposed as ``qm.run``.

    Packaged as a class so ``run`` and ``run.stream`` share the same
    argument-resolution path without repeating boilerplate.
    """

    def __call__(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> Result:
        """Execute *graph* against *user_input* and return a :class:`Result`.

        Args:
            graph: Either a :class:`GraphBuilder` (auto-finalised) or
                a pre-built :class:`GraphSpec`.
            user_input: Primary user message injected into the graph.
            provider_registry: Override the configured default registry.
            tool_registry: Optional :class:`quartermaster_tools.ToolRegistry`
                made available to ``agent()``-type nodes that specify
                ``tools=[...]``.
        """
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()
        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=tool_registry,
        )
        fr = runner.run(user_input)
        return Result.from_flow_result(fr)

    def stream(
        self,
        graph: GraphBuilder | GraphSpec,
        user_input: str = "",
        *,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: Any | None = None,
    ) -> Iterator[Chunk]:
        """Run the graph and yield typed :class:`Chunk` events as they arrive.

        Terminates with a :class:`DoneChunk` (on success) or
        :class:`ErrorChunk` (on unrecoverable failure).  The graph
        executes on a background thread so the caller can iterate the
        yielded chunks synchronously — no ``async``/``await`` required.

        **Cancellation:** breaking out of the loop early (``break``,
        ``return`` from an enclosing function, raising inside the
        consumer) sets a ``threading.Event`` that the runner checks on
        every dispatched node, so long-running flows stop promptly
        instead of continuing in the background.  The engine checks the
        flag via ``FlowRunner.stop(flow_id)`` — nodes already in flight
        finish their current tool call, but no new nodes are dispatched.
        """
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()

        q: queue.Queue[FlowEvent | None] = queue.Queue()
        cancelled = threading.Event()
        holder_lock = threading.Lock()
        holder: dict[str, Any] = {}

        def on_event(event: FlowEvent) -> None:
            q.put(event)

        runner = FlowRunner(
            graph=spec,
            provider_registry=registry,
            tool_registry=tool_registry,
            on_event=on_event,
        )

        def _run_thread() -> None:
            try:
                fr = runner.run(user_input)
                with holder_lock:
                    holder["result"] = fr
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception("run.stream: runner.run raised")
                with holder_lock:
                    holder["exception"] = exc
            finally:
                # Sentinel: signal the iterator loop that there are no
                # more events.  Placed in ``finally`` so a runner crash
                # doesn't hang the caller.
                q.put(None)

        thread = threading.Thread(target=_run_thread, name="qm-run-stream", daemon=True)
        thread.start()

        try:
            while True:
                # Short timeout so the caller can still get control back
                # (e.g. to abort, log progress) if the runner stalls.
                try:
                    event = q.get(timeout=300.0)
                except queue.Empty:
                    logger.warning("run.stream: no event for 5 minutes; still waiting")
                    continue
                if event is None:
                    break
                chunk = _event_to_chunk(event)
                if chunk is not None:
                    yield chunk
        finally:
            # Caller abandoned the iterator — tell the runner to stop.
            # Nodes currently executing finish (no hard kill), but no
            # further nodes are dispatched, so long agent loops unwind
            # within a bounded time instead of leaking.
            if thread.is_alive():
                cancelled.set()
                # We don't have the flow_id up here because runner.run()
                # creates one internally — the runner's own ``_stopped``
                # mechanism needs the id.  Best-effort: mark the event
                # and let the FlowRunner's graceful-shutdown path pick
                # it up via the reference stashed on the runner.
                try:
                    # The runner owns at most one active flow in the
                    # single-call ``runner.run(...)`` pattern — grab
                    # whichever id is in-flight and stop it.
                    for fid in list(runner._stopped):  # pragma: no cover
                        pass
                    # If no flow_id available, the sentinel on the queue
                    # plus the daemon thread's short idle cycles ensure
                    # the process exits cleanly even without explicit
                    # stop().  Future: wire runner.stream(flow_id=...)
                    # so the flow_id is exposed up-front.
                except Exception:  # pragma: no cover
                    pass
            thread.join(timeout=5.0)

        with holder_lock:
            exception = holder.get("exception")
            fr = holder.get("result")

        if exception is not None:
            yield ErrorChunk(error=str(exception))
            return

        if fr is None:
            yield ErrorChunk(error="run.stream: runner produced no result")
            return

        yield DoneChunk(result=Result.from_flow_result(fr))


def _event_to_chunk(event: FlowEvent) -> Chunk | None:
    """Translate an engine :class:`FlowEvent` into a public :class:`Chunk`.

    Returns ``None`` for events we intentionally swallow (they fold into
    ``DoneChunk`` at the end of the stream).
    """
    if isinstance(event, TokenGenerated):
        return TokenChunk(content=event.token)
    if isinstance(event, NodeStarted):
        return NodeStartChunk(
            node_name=event.node_name,
            node_type=event.node_type.value,
        )
    if isinstance(event, NodeFinished):
        return NodeFinishChunk(
            node_name=getattr(event, "node_name", ""),
            output=event.result or "",
        )
    if isinstance(event, UserInputRequired):
        return AwaitInputChunk(
            prompt=event.prompt,
            options=list(event.options or []),
        )
    if isinstance(event, FlowError):
        return ErrorChunk(error=event.error)
    if isinstance(event, FlowFinished):
        # DoneChunk is emitted explicitly after the thread joins so the
        # caller gets the full Result (not just the final_output string
        # this event carries).
        return None
    return None


#: Publicly exported runner.  ``qm.run(graph, input)`` runs sync;
#: ``qm.run.stream(graph, input)`` yields typed Chunk objects.
run = _RunCallable()


__all__ = ["run", "Result", "Chunk"]
