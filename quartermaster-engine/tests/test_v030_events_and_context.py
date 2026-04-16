"""Regression tests for v0.3.0 ``ProgressEvent`` / ``CustomEvent`` and
the ``current_context()`` contextvar plumbing.

Surface under test (engine-side):

* ``quartermaster_engine.events.ProgressEvent`` /
  ``quartermaster_engine.events.CustomEvent`` — the two new typed
  flow events fired when application code calls
  ``ExecutionContext.emit_progress(...)`` / ``.emit_custom(...)``.
* ``ExecutionContext.emit_progress`` / ``.emit_custom`` —
  including the no-callback no-op contract that lets tools stay
  unit-testable without a runner.
* ``quartermaster_engine.context.current_context.current_context()``
  / ``bind(...)`` — the contextvar that tools reach for to grab the
  live :class:`ExecutionContext`.
* ``FlowRunner._run_executor`` plumbing — both the synchronous path
  and the ``ThreadPoolExecutor`` path used when the runner is
  invoked from inside a running asyncio event loop. The latter is
  the **regression guard** for the
  ``contextvars.copy_context().run(...)`` snapshot — without it
  ``current_context()`` would resolve to ``None`` in the worker
  thread.
* End-to-end via the public SDK surface: a ``@tool()``-decorated
  function calling ``qm.current_context().emit_progress(...)``
  surfaces a ``ProgressChunk`` from ``qm.run.stream(...)`` interleaved
  between ``ToolCallChunk`` and ``ToolResultChunk``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from quartermaster_engine.context.current_context import current_context
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import (
    CustomEvent,
    FlowEvent,
    ProgressEvent,
)
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import (
    GraphSpec,
    GraphNode,
    NodeType,
    TraverseOut,
)

from tests.conftest import make_edge, make_graph, make_node


# ── Helpers ──────────────────────────────────────────────────────────


def _build_one_node_graph(
    executor: Any,
) -> tuple[GraphSpec, GraphNode, SimpleNodeRegistry]:
    """Build a tiny ``Start → Inst → End`` graph wired to *executor*.

    Returns the graph spec, the instruction node (so tests can assert
    on its ``id``), and the registry.
    """
    start = make_node(NodeType.START, name="Start")
    inst = make_node(NodeType.INSTRUCTION, name="Worker")
    end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

    graph = make_graph(
        [start, inst, end],
        [make_edge(start, inst), make_edge(inst, end)],
        start,
    )

    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, executor)
    return graph, inst, registry


# ── 1. ProgressEvent ─────────────────────────────────────────────────


class TestProgressEvent:
    """``ExecutionContext.emit_progress`` fires a ``ProgressEvent``
    on the runner's ``on_event`` hook with the right shape, and is a
    safe no-op when no callback is wired."""

    def test_progress_event_emitted_when_node_emits(self):
        """One ``emit_progress`` call → exactly one ``ProgressEvent`` with
        matching message / percent / data / node_id."""

        class _ProgressExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                context.emit_progress("loading", percent=0.5, query="hello")
                return NodeResult(success=True, data={}, output_text="ok")

        graph, inst, registry = _build_one_node_graph(_ProgressExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("kickoff")
        assert result.success, result.error

        progress = [e for e in events if isinstance(e, ProgressEvent)]
        assert len(progress) == 1, f"expected exactly 1 ProgressEvent, got {progress}"

        ev = progress[0]
        assert ev.message == "loading"
        assert ev.percent == 0.5
        assert ev.data == {"query": "hello"}
        assert ev.node_id == inst.id

    def test_progress_event_no_op_when_no_callback(self):
        """Calling ``emit_progress`` on a freshly-built ``ExecutionContext``
        with ``on_progress=None`` must not raise and must not produce any
        observable side effect — tools stay unit-testable without a runner."""
        node = make_node(NodeType.INSTRUCTION, name="Standalone")
        graph = make_graph([node], [], node)
        ctx = ExecutionContext(
            flow_id=uuid4(),
            node_id=node.id,
            graph=graph,
            current_node=node,
        )
        assert ctx.on_progress is None

        # Must not raise — and the callback is None so there's nothing
        # to invoke.  Verify by re-asserting the callback is still None
        # afterwards (no mutation as a side effect).
        ctx.emit_progress("x")
        ctx.emit_progress("y", percent=0.25, foo="bar")
        assert ctx.on_progress is None


# ── 2. CustomEvent ───────────────────────────────────────────────────


class TestCustomEvent:
    """``ExecutionContext.emit_custom`` fires a ``CustomEvent`` with the
    caller-supplied name and payload; payload defaults to an empty dict."""

    def test_custom_event_emitted_when_node_emits(self):
        class _CustomExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                context.emit_custom("retrieved_docs", {"count": 3})
                return NodeResult(success=True, data={}, output_text="ok")

        graph, inst, registry = _build_one_node_graph(_CustomExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("kickoff")
        assert result.success, result.error

        customs = [e for e in events if isinstance(e, CustomEvent)]
        assert len(customs) == 1, f"expected exactly 1 CustomEvent, got {customs}"

        ev = customs[0]
        assert ev.name == "retrieved_docs"
        assert ev.payload == {"count": 3}
        assert ev.node_id == inst.id

    def test_custom_event_payload_defaults_to_empty_dict(self):
        """``emit_custom("ping")`` (no payload arg) → ``CustomEvent`` with
        ``payload={}``, not ``None``."""

        class _PingExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                context.emit_custom("ping")
                return NodeResult(success=True, data={}, output_text="ok")

        graph, _inst, registry = _build_one_node_graph(_PingExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("kickoff")
        assert result.success, result.error

        customs = [e for e in events if isinstance(e, CustomEvent)]
        assert len(customs) == 1
        assert customs[0].name == "ping"
        assert customs[0].payload == {}


# ── 3. current_context contextvar ────────────────────────────────────


class TestCurrentContext:
    """The ``_current_ctx`` contextvar is bound by ``FlowRunner._run_executor``
    around ``executor.execute(context)`` and reset on the way out — so
    out-of-flow callers see ``None`` and in-flow tools see the live context.

    The ``test_current_context_propagates_into_thread_pool_executor`` case
    is the regression guard for the
    ``contextvars.copy_context().run(...)`` snapshot used when the runner
    dispatches into a worker thread from inside a running asyncio loop;
    without it the contextvar wouldn't propagate and tools would see
    ``None`` mid-flow.
    """

    def test_current_context_is_none_outside_flow(self):
        """No runner active → ``current_context()`` returns ``None``."""
        assert current_context() is None

    def test_current_context_is_set_inside_executor(self):
        """Inside the executor body, ``current_context()`` returns the
        same ``ExecutionContext`` instance the runner passed in."""
        captured: list[Any] = []
        seen_context: list[Any] = []

        class _CapturingExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                seen_context.append(context)
                captured.append(current_context())
                return NodeResult(success=True, data={}, output_text="ok")

        graph, _inst, registry = _build_one_node_graph(_CapturingExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("kickoff")
        assert result.success, result.error

        assert len(captured) == 1
        assert captured[0] is not None, "current_context() returned None inside flow"
        # Identity check — must be the very same object the runner passed.
        assert captured[0] is seen_context[0]

    def test_current_context_unset_after_run_completes(self):
        """After the run finishes, ``current_context()`` is back to ``None``
        at the top level — the contextvar was reset by ``bind``'s
        ``finally`` clause."""

        class _NoopExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                # Just confirm we have a context in-flight.
                assert current_context() is context
                return NodeResult(success=True, data={}, output_text="ok")

        graph, _inst, registry = _build_one_node_graph(_NoopExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("kickoff")
        assert result.success, result.error

        # Top-level contextvar state must be clean.
        assert current_context() is None

    def test_current_context_propagates_into_thread_pool_executor(self):
        """REGRESSION GUARD: when the runner dispatches via the
        ``ThreadPoolExecutor`` path (running asyncio loop active), the
        worker thread must see the bound contextvar via
        ``contextvars.copy_context().run(...)`` — otherwise
        ``current_context()`` would return ``None`` inside the executor.

        We force the thread-pool path by running the runner from inside
        a coroutine: ``asyncio.get_running_loop()`` succeeds and
        ``loop.is_running()`` is ``True``, so ``_run_executor`` takes
        the worker-thread branch.
        """
        captured: dict[str, Any] = {}

        class _ThreadPoolCapturingExecutor:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                # This runs inside the worker thread spawned by
                # _run_executor's asyncio.run(coro) — without
                # contextvars.copy_context().run(...) the contextvar
                # would be unset here and current_context() would be
                # None.
                seen = current_context()
                captured["seen"] = seen
                captured["is_same"] = seen is context
                return NodeResult(success=True, data={}, output_text="ok")

        graph, _inst, registry = _build_one_node_graph(_ThreadPoolCapturingExecutor())
        runner = FlowRunner(graph=graph, node_registry=registry)

        async def _make_runner_call_from_async() -> Any:
            # Calling runner.run from inside a coroutine forces
            # _run_executor to take the ThreadPoolExecutor branch
            # because ``asyncio.get_running_loop()`` resolves and
            # ``loop.is_running()`` is True.
            return runner.run("kickoff")

        result = asyncio.run(_make_runner_call_from_async())
        assert result.success, result.error

        assert captured.get("seen") is not None, (
            "current_context() returned None in the ThreadPoolExecutor "
            "worker — contextvars.copy_context().run(...) plumbing in "
            "FlowRunner._run_executor regressed."
        )
        assert captured["is_same"] is True, (
            "current_context() in the worker did not match the "
            "ExecutionContext the runner passed in."
        )

        # And cleanup still holds at the top level.
        assert current_context() is None


# ── 4. End-to-end via the public SDK surface ─────────────────────────


class TestEmitFromTool:
    """End-to-end: a ``@tool()`` function calling
    ``qm.current_context().emit_progress(...)`` from inside an agent's
    tool loop surfaces a ``ProgressChunk`` on ``qm.run.stream(...)``,
    interleaved between the matching ``ToolCallChunk`` and
    ``ToolResultChunk``.

    Lifts the ``MockProvider`` / ``_OkTool`` / ``_ToolRegistry`` shim
    pattern from ``quartermaster-sdk/tests/test_v022_tool_streaming.py``
    so the agent loop terminates cleanly after one tool call.
    """

    def test_tool_inside_agent_can_emit_progress(self):
        # Imported here so the rest of this engine-side test module
        # stays importable even if the SDK happens to be missing in some
        # downstream environment. With the workspace layout in this
        # repo, both packages are always present.
        import quartermaster_sdk as qm
        from quartermaster_providers import ProviderRegistry
        from quartermaster_providers.testing import MockProvider
        from quartermaster_providers.types import (
            NativeResponse,
            TokenResponse,
            ToolCall,
        )
        from quartermaster_tools import tool

        # ── Define a tool that emits a progress event mid-execution.
        # The decorator builds a ``FunctionTool``; the agent executor
        # will call ``.safe_run(**args)`` which delegates to ``run``
        # which calls our wrapped function.  Inside the function the
        # contextvar is set (the agent executor runs inside the runner's
        # ``bind_current_context(...)`` scope).
        @tool()
        def slow_search(q: str) -> dict:
            """Pretend to do a slow search and emit a progress signal."""
            ctx = qm.current_context()
            assert ctx is not None, (
                "current_context() returned None inside the @tool body — "
                "the contextvar was not propagated into the tool call."
            )
            ctx.emit_progress("searching", percent=0.5)
            return {"ok": True, "q": q}

        # ── Mock provider: turn 1 returns a tool call, turn 2 ends.
        mock = MockProvider(
            responses=[TokenResponse(content="Done.", stop_reason="stop")],
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="slow_search",
                            tool_id="call_1",
                            parameters={"q": "hello"},
                        )
                    ],
                    stop_reason="tool_calls",
                ),
                NativeResponse(
                    text_content="Done.",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ],
        )
        provider_reg = ProviderRegistry(auto_configure=False)
        provider_reg.register_instance("mock", mock)
        provider_reg.set_default_provider("mock")
        provider_reg.set_default_model("mock", "test-model")

        # ── ToolRegistry shim matching the agent executor's interface
        # (``.get(name)`` + ``.to_openai_tools()``). We bypass
        # ``quartermaster_tools.ToolRegistry`` to keep the test focused
        # on the contextvar + event plumbing — the registry shape mirrors
        # the one in test_v022_tool_streaming.py.
        class _ToolRegistry:
            def __init__(self, t: Any):
                self._t = t

            def get(self, name: str) -> Any:
                if name != "slow_search":
                    raise KeyError(name)
                return self._t

            def to_openai_tools(self) -> list[dict]:
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "slow_search",
                            "description": "stub",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        },
                    }
                ]

        # The agent executor calls ``.safe_run(**args)`` on the tool —
        # ``FunctionTool`` inherits this from ``AbstractTool``. The
        # tool's body runs synchronously inside the agent's tool loop,
        # which itself runs inside ``bind_current_context(...)``, so
        # ``current_context()`` resolves correctly.
        tool_reg = _ToolRegistry(slow_search)

        qm.reset_config()
        try:
            qm.configure(registry=provider_reg)

            graph = (
                qm.Graph("chat")
                .user()
                .agent("Tooled", tools=["slow_search"], capture_as="agent")
                .build()
            )

            chunks = list(qm.run.stream(graph, "hi", tool_registry=tool_reg))
        finally:
            qm.reset_config()

        types_seq = [c.type for c in chunks]

        # The progress chunk must arrive AND must sit between the
        # matching tool_call and tool_result chunks — that's the whole
        # point of the contextvar plumbing: live status from inside
        # tools, interleaved with the agent loop.
        assert "tool_call" in types_seq, f"no tool_call in {types_seq}"
        assert "tool_result" in types_seq, f"no tool_result in {types_seq}"
        assert "progress" in types_seq, f"no progress chunk in {types_seq}"

        tool_call_idx = types_seq.index("tool_call")
        progress_idx = types_seq.index("progress")
        tool_result_idx = types_seq.index("tool_result")
        assert tool_call_idx < progress_idx < tool_result_idx, (
            f"progress chunk must sit between tool_call and tool_result; got order {types_seq}"
        )

        progress_chunk = next(c for c in chunks if c.type == "progress")
        assert isinstance(progress_chunk, qm.ProgressChunk)
        assert progress_chunk.message == "searching"
        assert progress_chunk.percent == 0.5
