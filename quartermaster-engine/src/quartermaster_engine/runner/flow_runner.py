"""FlowRunner — the core orchestration engine that executes agent graphs.

This is the heart of quartermaster-engine. It takes a graph definition, resolves node
implementations, and orchestrates the execution: traversal, branching, merging,
memory, message passing, and error handling.

The execution loop:
  1. Start at the start node
  2. Check traverse_in — should this node execute now?
  3. Resolve node implementation from the registry
  4. Build ExecutionContext for the node
  5. Execute the node
  6. Process result (store messages, update memory)
  7. Check traverse_out — which successors to trigger?
  8. Dispatch successor nodes
  9. Repeat until all branches reach End or stop
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from quartermaster_engine.context.current_context import bind as bind_current_context
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.context.node_execution import NodeExecution, NodeStatus
from quartermaster_engine.dispatchers.sync_dispatcher import SyncDispatcher
from quartermaster_engine.events import (
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
from quartermaster_engine.messaging.context_manager import ContextManager
from quartermaster_engine.messaging.message_router import MessageRouter
from quartermaster_engine.nodes import NodeExecutor, NodeRegistry, NodeResult
from quartermaster_engine.stores.base import ExecutionStore
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.traversal.traverse_in import TraverseInGate
from quartermaster_engine.traversal.traverse_out import TraverseOutGate
from quartermaster_engine.types import (
    ErrorStrategy,
    GraphNode,
    GraphSpec,
    Message,
    MessageRole,
    NodeType,
    TraverseOut,
)

logger = logging.getLogger(__name__)


@dataclass
class FlowResult:
    """The final result of a flow execution.

    The UUID-keyed ``node_results`` is the authoritative per-node record
    and is mostly useful for debugging tooling (visualisers, trace
    dumps).  The name-keyed ``captures`` dict (v0.2.0+) is what
    application code should reach for — populated from the
    ``capture_as="..."`` kwarg users pass on node builder methods.
    """

    flow_id: UUID
    success: bool
    final_output: str = ""
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    node_results: dict[UUID, NodeResult] = field(default_factory=dict)
    #: Name → NodeResult mapping for nodes that set ``capture_as="..."``.
    #: Lets callers write ``result.captures["research"].output_text`` instead
    #: of fishing through ``node_results`` with UUID keys and node-type
    #: pattern matching.
    captures: dict[str, NodeResult] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def __getitem__(self, name: str) -> NodeResult:
        """Syntactic sugar: ``result["research"]`` → the captured node result.

        Raises ``KeyError`` with a helpful message listing available
        capture names when *name* is unknown.  The SDK's ``Result`` type
        uses the same message format (see ``quartermaster_sdk._result.
        format_missing_capture_error``) so switching between the two is
        seamless.
        """
        try:
            return self.captures[name]
        except KeyError:
            raise KeyError(_format_missing_capture(name, self.captures)) from None


def _format_missing_capture(name: str, captures: dict[str, NodeResult]) -> str:
    """Shared format for "no capture named X" errors — matches the SDK's
    ``format_missing_capture_error``.  Kept as a module-private helper here
    so the engine doesn't depend on the SDK package."""
    available = ", ".join(sorted(captures)) or "(no captures registered)"
    return f"No capture named {name!r}. Available captures: {available}"


class FlowRunner:
    """Main orchestration class for executing agent graphs.

    Coordinates graph traversal, node execution, message routing, memory management,
    and error handling. Fully pluggable: storage, dispatching, and node implementations
    are all injected at construction time.
    """

    def __init__(
        self,
        graph: GraphSpec,
        node_registry: NodeRegistry | None = None,
        store: ExecutionStore | None = None,
        dispatcher: Any | None = None,
        context_manager: ContextManager | None = None,
        on_event: Callable[[FlowEvent], None] | None = None,
        *,
        provider_registry: Any | None = None,
        tool_registry: Any | None = None,
    ) -> None:
        """Construct a runner.

        Either *node_registry* (the low-level node-type → executor map) or
        *provider_registry* (a :class:`quartermaster_providers.ProviderRegistry`)
        must be supplied.  When only *provider_registry* is given the runner
        builds a default node registry via
        :func:`quartermaster_engine.build_default_registry` — which covers
        every node type the bundled DSL can emit.

        Pass *tool_registry* (e.g. a ``quartermaster_tools.ToolRegistry``)
        to enable real tool execution for ``agent()`` nodes; without it
        the agent loop still runs but treats tool-less graphs as plain
        text completion.

        ``self.provider_registry`` and ``self.tool_registry`` are stored
        for introspection but are only meaningful when supplied through
        this constructor.  When you build the node registry yourself the
        runner has no way to know which provider/tool registry sits
        behind it, so they will be ``None`` — that's expected, not a bug.
        """
        if node_registry is None:
            if provider_registry is None:
                raise TypeError(
                    "FlowRunner requires either node_registry or "
                    "provider_registry. Pass a ProviderRegistry to use the "
                    "default node registry, or build a SimpleNodeRegistry "
                    "directly via quartermaster_engine.build_default_registry()."
                )
            # Imported here to avoid the example_runner ↔ flow_runner cycle
            # at module import time.
            from quartermaster_engine.example_runner import build_default_registry

            node_registry = build_default_registry(
                provider_registry,
                interactive=False,
                tool_registry=tool_registry,
            )

        self.graph = graph
        self.node_registry = node_registry
        # NOTE: these are introspection-only references; they're ``None``
        # when callers wire their own ``node_registry`` because we have no
        # safe way to recover them from an arbitrary executor map.
        self.provider_registry = provider_registry
        self.tool_registry = tool_registry
        self.store: ExecutionStore = store or InMemoryStore()
        self.dispatcher = dispatcher or SyncDispatcher()
        self.context_manager = context_manager or ContextManager()
        self.on_event = on_event

        self._traverse_in = TraverseInGate()
        self._traverse_out = TraverseOutGate()
        self._message_router = MessageRouter(self.store)
        self._stopped: set[UUID] = set()

    def run(
        self,
        input_message: str,
        *,
        images: list[tuple[str, str]] | None = None,
        flow_id: UUID | None = None,
    ) -> FlowResult:
        """Execute the graph synchronously.

        Args:
            input_message: The user's input message.
            images: Optional list of ``(base64_ascii, mime_type)`` pairs
                attached to the initial user turn. Vision-capable
                nodes (``Graph().vision(...)``) read this list from
                flow memory via the ``__user_images__`` key and forward
                it to the provider alongside the text prompt. Pass raw
                image bytes in the SDK (``qm.run(..., image=bytes)``);
                the SDK normalises them to the internal base64 tuple
                shape before calling this method.
            flow_id: Optional flow ID (auto-generated if not provided).

        Returns:
            A FlowResult with the final output and metadata.
        """
        fid = flow_id or uuid4()
        start_time = time.monotonic()

        try:
            start_node = self.graph.get_start_node()
            if not start_node:
                return FlowResult(flow_id=fid, success=False, error="No start node found in graph")

            # Store the initial user input in flow memory
            self.store.save_memory(fid, "__user_input__", input_message)
            # Store any attached images so vision-capable nodes can pick
            # them up via ``context.memory["__user_images__"]`` without
            # the caller having to touch the store directly. Stored as
            # a plain list[tuple[str, str]] — base64 ASCII + MIME type.
            if images:
                self.store.save_memory(fid, "__user_images__", list(images))

            # Execute the start node, which kicks off the traversal
            self._execute_node(fid, start_node.id, input_message)

            # Wait for any parallel branches to finish
            self.dispatcher.wait_all()

            # Collect results
            result = self._collect_result(fid)
            result.duration_seconds = time.monotonic() - start_time
            return result

        except Exception as e:
            logger.exception("Flow %s failed with error: %s", fid, e)
            return FlowResult(
                flow_id=fid,
                success=False,
                error=str(e),
                duration_seconds=time.monotonic() - start_time,
            )

    async def run_async(
        self, input_message: str, flow_id: UUID | None = None
    ) -> AsyncIterator[FlowEvent]:
        """Execute the graph asynchronously, yielding events as they occur.

        Args:
            input_message: The user's input message.
            flow_id: Optional flow ID.

        Yields:
            FlowEvent instances as the execution progresses.
        """
        fid = flow_id or uuid4()
        event_queue: asyncio.Queue[FlowEvent | None] = asyncio.Queue()

        original_on_event = self.on_event

        def queue_event(event: FlowEvent) -> None:
            event_queue.put_nowait(event)
            if original_on_event:
                original_on_event(event)

        self.on_event = queue_event

        async def _run() -> None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.run, input_message, fid)
            finally:
                event_queue.put_nowait(None)  # Sentinel to stop iteration

        task = asyncio.create_task(_run())

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

        await task
        self.on_event = original_on_event

    def resume(self, flow_id: UUID, user_input: str) -> FlowResult:
        """Resume a flow that is waiting for user input.

        Finds the node waiting for user input, provides the input, and
        continues execution.
        """
        start_time = time.monotonic()
        executions = self.store.get_all_node_executions(flow_id)

        waiting_node_id: UUID | None = None
        for nid, execution in executions.items():
            if execution.status == NodeStatus.WAITING_USER:
                waiting_node_id = nid
                break

        if waiting_node_id is None:
            return FlowResult(
                flow_id=flow_id,
                success=False,
                error="No node is waiting for user input",
            )

        # Add the user's response as a message
        self._message_router.append_to_node(
            flow_id, waiting_node_id, Message(role=MessageRole.USER, content=user_input)
        )

        # Re-execute the waiting node
        self._execute_node(flow_id, waiting_node_id, user_input)
        self.dispatcher.wait_all()

        result = self._collect_result(flow_id)
        result.duration_seconds = time.monotonic() - start_time
        return result

    def stop(self, flow_id: UUID) -> None:
        """Stop a running flow. Marks all active nodes as failed."""
        self._stopped.add(flow_id)
        executions = self.store.get_all_node_executions(flow_id)
        for nid, execution in executions.items():
            if execution.status.is_active:
                execution.fail("Flow stopped by user")
                self.store.save_node_execution(flow_id, nid, execution)

    def _execute_node(self, flow_id: UUID, node_id: UUID, user_input: str) -> None:
        """Execute a single node: check gate, run, handle result, dispatch successors."""
        if flow_id in self._stopped:
            return

        node = self.graph.get_node(node_id)
        if node is None:
            logger.error("Node %s not found in graph", node_id)
            return

        # Check traverse_in gate
        executions = self.store.get_all_node_executions(flow_id)
        if not self._traverse_in.should_execute(node_id, self.graph, executions, node.traverse_in):
            logger.debug("Node %s not ready (waiting for predecessors)", node_id)
            return

        # Skip if already executed (avoid re-execution in loops unless SpawnStart)
        existing = self.store.get_node_execution(flow_id, node_id)
        if existing and existing.status.is_terminal and not self._is_loop_target(flow_id, node_id):
            return

        # Create execution record
        execution = NodeExecution(node_id=node_id)
        execution.start()
        self.store.save_node_execution(flow_id, node_id, execution)

        # Emit event
        self._emit(
            NodeStarted(
                flow_id=flow_id,
                node_id=node_id,
                node_type=node.type,
                node_name=node.name,
            )
        )

        # Handle built-in control flow nodes
        result: NodeResult | None
        if node.type in (NodeType.START, NodeType.END, NodeType.MERGE):
            result = self._execute_control_node(flow_id, node, user_input, execution)
        else:
            result = self._execute_logic_node(flow_id, node, user_input, execution)

        if result is None:
            return

        # Check for user input wait
        if result.wait_for_user:
            execution.wait_for_user()
            self.store.save_node_execution(flow_id, node_id, execution)
            self._emit(
                UserInputRequired(
                    flow_id=flow_id,
                    node_id=node_id,
                    prompt=result.user_prompt or "",
                    options=result.user_options or [],
                )
            )
            return

        # Surface explicit failures from the executor.  Without this branch
        # a node that returns ``NodeResult(success=False, error=...)`` would
        # be marked FINISHED and the error would silently drop on the floor
        # — making FlowResult.success=True even though no output was
        # produced (e.g. provider connection refused).  Route through the
        # standard error-handling pipeline so RETRY / SKIP / STOP semantics
        # apply uniformly whether the executor raised or just returned a
        # failure result.
        if not result.success:
            handled = self._handle_node_error(
                flow_id, node, execution, result.error or "Node reported failure"
            )
            if handled is None:
                # RETRY in flight — _handle_node_error has already
                # re-dispatched; nothing further to do here.
                return
            # Replace with the handled result so _dispatch_successors sees
            # the failure status and applies STOP-vs-SKIP correctly.
            result = handled
        else:
            execution.finish(result=result.output_text, output_data=result.data)
            self.store.save_node_execution(flow_id, node_id, execution)
            self._emit(
                NodeFinished(
                    flow_id=flow_id,
                    node_id=node_id,
                    result=result.output_text or "",
                    output_data=result.data,
                )
            )

        # Determine and dispatch successors (no-op when STOP + failure).
        self._dispatch_successors(flow_id, node, result, user_input)

    def _execute_control_node(
        self,
        flow_id: UUID,
        node: GraphNode,
        user_input: str,
        execution: NodeExecution,
    ) -> NodeResult:
        """Handle built-in control flow nodes (Start, End, Merge)."""
        if node.type == NodeType.START:
            # Start node: pass the user input through
            messages = [Message(role=MessageRole.USER, content=user_input)]
            self._message_router.save_node_output(flow_id, node.id, messages)
            return NodeResult(success=True, data={}, output_text=user_input)

        if node.type == NodeType.END:
            # End node: collect the final output from predecessors
            messages = self._message_router.get_messages_for_node(flow_id, node, self.graph)
            final_output = messages[-1].content if messages else ""
            self._message_router.save_node_output(flow_id, node.id, messages)
            return NodeResult(success=True, data={}, output_text=final_output)

        if node.type == NodeType.MERGE:
            # Merge node: combine outputs from all predecessors
            messages = self._message_router.get_messages_for_node(flow_id, node, self.graph)
            self._message_router.save_node_output(flow_id, node.id, messages)
            combined = "\n".join(m.content for m in messages if m.content)
            return NodeResult(success=True, data={}, output_text=combined)

        return NodeResult(success=True, data={})

    def _execute_logic_node(
        self,
        flow_id: UUID,
        node: GraphNode,
        user_input: str,
        execution: NodeExecution,
    ) -> NodeResult | None:
        """Execute a logic node (Instruction, Decision, etc.) via the registry."""
        executor = self.node_registry.get_executor(node.type.value)
        if executor is None:
            error_msg = f"No executor registered for node type: {node.type.value}"
            logger.error(error_msg)
            return self._handle_node_error(flow_id, node, execution, error_msg)

        # Build execution context
        messages = self._message_router.get_messages_for_node(flow_id, node, self.graph)

        # Add input message based on MessageType
        flow_memory = self.store.get_all_memory(flow_id)
        input_msg = self._message_router.build_input_message(node, user_input, flow_memory)
        if input_msg:
            messages.append(input_msg)

        # Apply context window truncation
        messages = self.context_manager.truncate(messages)

        context = ExecutionContext(
            flow_id=flow_id,
            node_id=node.id,
            graph=self.graph,
            current_node=node,
            messages=messages,
            memory=flow_memory,
            metadata=dict(node.metadata),
            on_token=lambda t: self._emit(
                TokenGenerated(flow_id=flow_id, node_id=node.id, token=t)
            ),
            on_tool_start=lambda tool, args, it: self._emit(
                ToolCallStarted(
                    flow_id=flow_id,
                    node_id=node.id,
                    tool=tool,
                    arguments=args,
                    iteration=it,
                )
            ),
            on_tool_finish=lambda tool, args, result, raw, err, it: self._emit(
                ToolCallFinished(
                    flow_id=flow_id,
                    node_id=node.id,
                    tool=tool,
                    arguments=args,
                    result=result,
                    raw=raw,
                    error=err,
                    iteration=it,
                )
            ),
            on_progress=lambda msg, percent, data: self._emit(
                ProgressEvent(
                    flow_id=flow_id,
                    node_id=node.id,
                    message=msg,
                    percent=percent,
                    data=data,
                )
            ),
            on_custom=lambda name, payload: self._emit(
                CustomEvent(
                    flow_id=flow_id,
                    node_id=node.id,
                    name=name,
                    payload=payload,
                )
            ),
        )

        # Execute with error handling
        try:
            result = self._run_executor(executor, context, node)
        except Exception as e:
            return self._handle_node_error(flow_id, node, execution, str(e))

        # Save output messages
        output_messages = list(messages)
        if result.output_text:
            output_messages.append(Message(role=MessageRole.ASSISTANT, content=result.output_text))
        self._message_router.save_node_output(flow_id, node.id, output_messages)

        # Update flow memory if node produced memory updates
        if "memory_updates" in result.data:
            for key, value in result.data["memory_updates"].items():
                self.store.save_memory(flow_id, key, value)

        return result

    def _run_executor(
        self, executor: NodeExecutor, context: ExecutionContext, node: GraphNode
    ) -> NodeResult:
        """Run a node executor, handling sync/async transparently.

        Wraps the executor invocation in ``current_context.bind(context)``
        so application code inside the executor (most importantly
        ``@tool()`` callables) can reach the live :class:`ExecutionContext`
        via :func:`quartermaster_sdk.current_context` and emit progress
        or custom events back onto the stream.

        Threading: when we're inside a running asyncio event loop we
        dispatch to a ``ThreadPoolExecutor``. Python's ``contextvars``
        do NOT propagate into pool workers automatically, so we submit
        ``contextvars.copy_context().run(target)`` — the worker thread
        inherits a snapshot of the caller's contextvars (including the
        freshly-bound ``_current_ctx``) and tools running inside the
        coroutine see ``current_context()`` return a non-None value.

        If the node has a ``timeout`` set (in seconds), execution is wrapped
        in ``asyncio.wait_for`` so that a ``TimeoutError`` is raised when the
        node exceeds its time budget.
        """
        coro = executor.execute(context)

        # Wrap with timeout if configured
        if node.timeout is not None and node.timeout > 0:
            coro = asyncio.wait_for(coro, timeout=node.timeout)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Bind the contextvar for the duration of this executor call so
        # tools invoked during execution can emit progress/custom events.
        # The ``with`` scope unwinds the var after the executor returns.
        with bind_current_context(context):
            if loop and loop.is_running():
                # Inside an async context — spawn a worker thread for the
                # coroutine. ``copy_context().run(...)`` carries the
                # freshly-bound contextvar into the worker's stack so
                # ``current_context()`` resolves correctly there too.
                import concurrent.futures

                ctx_snapshot = contextvars.copy_context()

                def _worker() -> NodeResult:
                    return asyncio.run(coro)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(ctx_snapshot.run, _worker)
                    return future.result(timeout=node.timeout)
            return asyncio.run(coro)

    def _handle_node_error(
        self,
        flow_id: UUID,
        node: GraphNode,
        execution: NodeExecution,
        error: str,
    ) -> NodeResult | None:
        """Apply the node's error handling strategy."""
        strategy = node.error_handling

        if strategy == ErrorStrategy.RETRY and execution.retry_count < node.max_retries:
            execution.retry_count += 1
            execution.status = NodeStatus.PENDING
            self.store.save_node_execution(flow_id, node.id, execution)
            logger.info(
                "Retrying node %s (attempt %d/%d)",
                node.id,
                execution.retry_count,
                node.max_retries,
            )
            # Retry by re-dispatching
            user_input = self.store.get_memory(flow_id, "__user_input__") or ""
            self._execute_node(flow_id, node.id, str(user_input))
            return None

        if strategy == ErrorStrategy.SKIP:
            execution.skip()
            self.store.save_node_execution(flow_id, node.id, execution)
            self._emit(FlowError(flow_id=flow_id, node_id=node.id, error=error, recoverable=True))
            return NodeResult(success=False, data={}, error=error)

        # STOP or CUSTOM — halt execution
        execution.fail(error)
        self.store.save_node_execution(flow_id, node.id, execution)
        self._emit(FlowError(flow_id=flow_id, node_id=node.id, error=error, recoverable=False))
        return NodeResult(success=False, data={}, error=error)

    def _dispatch_successors(
        self,
        flow_id: UUID,
        node: GraphNode,
        result: NodeResult,
        user_input: str,
    ) -> None:
        """Determine and dispatch successor nodes based on traverse_out strategy."""
        # End nodes and failed nodes with Stop strategy don't dispatch
        if node.type == NodeType.END:
            return
        if not result.success and node.error_handling == ErrorStrategy.STOP:
            return

        next_nodes = self._traverse_out.get_next_nodes(
            node.id, self.graph, node.traverse_out, result
        )

        for next_node in next_nodes:
            self.dispatcher.dispatch(
                flow_id,
                next_node.id,
                lambda fid, nid: self._execute_node(fid, nid, user_input),
            )

    def _collect_result(self, flow_id: UUID) -> FlowResult:
        """Collect the final result after all nodes have completed."""
        executions = self.store.get_all_node_executions(flow_id)
        node_results: dict[UUID, NodeResult] = {}
        captures: dict[str, NodeResult] = {}
        final_output = ""
        all_success = True
        errors: list[str] = []

        for nid, execution in executions.items():
            node = self.graph.get_node(nid)

            if execution.status == NodeStatus.FAILED:
                all_success = False
                if execution.error:
                    errors.append(execution.error)

            if execution.result:
                nr = NodeResult(
                    success=execution.status == NodeStatus.FINISHED,
                    data=execution.output_data,
                    output_text=execution.result,
                    error=execution.error,
                )
                node_results[nid] = nr
                # Populate name-keyed captures if the builder set
                # ``capture_as="..."`` on this node.  We look up via the
                # shared constant from ``quartermaster_graph`` so rename
                # drift is caught at import time.
                if node is not None:
                    capture_name = node.metadata.get("capture_as")
                    if isinstance(capture_name, str) and capture_name:
                        captures[capture_name] = nr

            # The final output comes from End nodes
            if node and node.type == NodeType.END and execution.result:
                final_output = execution.result

        # If no End node result, use the last finished node's result
        if not final_output:
            finished = [
                e for e in executions.values() if e.status == NodeStatus.FINISHED and e.result
            ]
            if finished:
                finished.sort(key=lambda e: e.finished_at or e.started_at or 0)
                final_output = finished[-1].result or ""

        result = FlowResult(
            flow_id=flow_id,
            success=all_success,
            final_output=final_output,
            node_results=node_results,
            captures=captures,
            error="; ".join(errors) if errors else None,
        )

        self._emit(FlowFinished(flow_id=flow_id, final_output=final_output))

        return result

    def _is_loop_target(self, flow_id: UUID, node_id: UUID) -> bool:
        """Check if a node is being re-executed as part of a loop.

        Returns True when *any* already-finished predecessor dispatches back
        to this node — covers both ``SPAWN_START`` (explicit loop-to-start)
        and conditional loops where ``SPAWN_PICKED`` selects a back-edge.
        """
        executions = self.store.get_all_node_executions(flow_id)

        # SpawnStart → start node
        for node in self.graph.nodes:
            if node.traverse_out == TraverseOut.SPAWN_START:
                start = self.graph.get_start_node()
                if start and start.id == node_id:
                    return True

        # Any predecessor that has already completed and has an edge to this
        # node is a back-edge (the node finished, but it dispatched us again).
        predecessors = self.graph.get_predecessors(node_id)
        for pred in predecessors:
            pred_exec = executions.get(pred.id)
            if pred_exec and pred_exec.status.is_terminal:
                return True

        return False

    def _emit(self, event: FlowEvent) -> None:
        """Emit a flow event to the registered callback."""
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("Error in event handler")
