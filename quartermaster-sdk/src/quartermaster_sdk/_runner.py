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
from uuid import uuid4

from quartermaster_engine import (
    FlowError,
    FlowEvent,
    FlowFinished,
    FlowRunner,
    NodeFinished,
    NodeStarted,
    TokenGenerated,
    ToolCallFinished,
    ToolCallStarted,
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
    ToolCallChunk,
    ToolResultChunk,
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
        consumer) calls :meth:`FlowRunner.stop` on the in-flight
        ``flow_id``, which the engine checks in ``_execute_node``
        before dispatching any further work.  Nodes already mid-flight
        finish their current LLM/tool call — no hard kill — but no new
        nodes are dispatched, so long agent loops unwind within a
        bounded time instead of leaking API costs in the background.
        """
        spec = _resolve_graph(graph)
        registry = provider_registry or get_default_registry()

        q: queue.Queue[FlowEvent | None] = queue.Queue()
        holder_lock = threading.Lock()
        holder: dict[str, Any] = {}
        # Pre-generate the flow_id so cancellation can call
        # ``runner.stop(flow_id)`` even if the caller breaks out of the
        # iterator before the first event fires.  ``FlowRunner.run``
        # accepts an optional ``flow_id=`` and will use ours instead of
        # minting its own — this is the documented public API, not
        # reaching into a private attribute.
        flow_id = uuid4()

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
                fr = runner.run(user_input, flow_id=flow_id)
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
            # Caller abandoned the iterator — tell the runner to stop
            # the flow by its pre-generated id.  ``_execute_node``
            # short-circuits on the next dispatch, so nodes mid-LLM-call
            # finish (no hard kill) but no new work is scheduled.
            if thread.is_alive():
                try:
                    runner.stop(flow_id)
                except Exception:  # pragma: no cover — defensive
                    logger.exception("run.stream: runner.stop(%s) raised", flow_id)
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
    if isinstance(event, ToolCallStarted):
        # Live "tool is being called now" card — fires before the tool
        # actually runs.  Arguments are passed straight through so UIs
        # can render them the moment the call is dispatched, instead of
        # waiting for the final NodeFinish / Done event.
        return ToolCallChunk(tool=event.tool, args=dict(event.arguments))
    if isinstance(event, ToolCallFinished):
        # Paired with ToolCallStarted.  ``result`` is the string the LLM
        # sees next turn (``"[ERROR: ...]"`` sentinel on failure);
        # ``raw`` is the structured payload when the tool returned one,
        # ``None`` otherwise.  ``error`` is non-None only on failure —
        # UIs can use it as the "red card" trigger.
        return ToolResultChunk(
            tool=event.tool,
            result=event.raw if event.raw is not None else event.result,
            error=event.error,
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
