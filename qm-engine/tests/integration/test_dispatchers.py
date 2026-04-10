"""Integration tests for dispatcher variations — Sync, Thread, and Async."""

from __future__ import annotations

import time
from uuid import uuid4

from qm_engine.dispatchers.sync_dispatcher import SyncDispatcher
from qm_engine.dispatchers.thread_dispatcher import ThreadDispatcher
from qm_engine.events import FlowError, FlowEvent, NodeFinished, NodeStarted
from qm_engine.nodes import SimpleNodeRegistry
from qm_engine.runner.flow_runner import FlowRunner
from qm_engine.types import (
    ErrorStrategy,
    NodeType,
    TraverseIn,
    TraverseOut,
)
from tests.conftest import (
    EchoExecutor,
    FailingExecutor,
    SlowExecutor,
    make_edge,
    make_graph,
    make_node,
)


class TestSyncDispatcher:
    """SyncDispatcher: sequential execution in the current thread."""

    def test_sync_dispatcher_basic_flow(self):
        start = make_node(NodeType.START, name="Start")
        inst = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, inst, end],
            [make_edge(start, inst), make_edge(inst, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry, dispatcher=SyncDispatcher())
        result = runner.run("sync test")

        assert result.success
        assert "sync test" in result.final_output

    def test_sync_dispatcher_sequential_execution_order(self):
        """With SyncDispatcher, parallel branches are executed sequentially."""
        start = make_node(NodeType.START, name="Start")
        a = make_node(NodeType.INSTRUCTION, name="A")
        b = make_node(NodeType.STATIC, name="B")
        merge = make_node(NodeType.MERGE, name="Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, a, b, merge, end],
            [
                make_edge(start, a),
                make_edge(start, b),
                make_edge(a, merge),
                make_edge(b, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            dispatcher=SyncDispatcher(),
            on_event=events.append,
        )
        result = runner.run("parallel with sync")

        assert result.success
        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "A" in started_names
        assert "B" in started_names

    def test_sync_dispatcher_wait_all_is_noop(self):
        """SyncDispatcher.wait_all() is a no-op and should not raise."""
        d = SyncDispatcher()
        d.wait_all()  # should not raise

    def test_sync_dispatcher_shutdown_is_noop(self):
        d = SyncDispatcher()
        d.shutdown()  # should not raise


class TestThreadDispatcher:
    """ThreadDispatcher: parallel execution using thread pool."""

    def test_thread_dispatcher_basic_flow(self):
        start = make_node(NodeType.START, name="Start")
        inst = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, inst, end],
            [make_edge(start, inst), make_edge(inst, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        dispatcher = ThreadDispatcher(max_workers=2)
        try:
            runner = FlowRunner(graph=graph, node_registry=registry, dispatcher=dispatcher)
            result = runner.run("thread test")
            assert result.success
        finally:
            dispatcher.shutdown()

    def test_thread_dispatcher_parallel_branches(self):
        """Parallel branches should both execute with ThreadDispatcher."""
        start = make_node(NodeType.START, name="Start")
        a = make_node(NodeType.INSTRUCTION, name="A")
        b = make_node(NodeType.STATIC, name="B")
        merge = make_node(NodeType.MERGE, name="Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, a, b, merge, end],
            [
                make_edge(start, a),
                make_edge(start, b),
                make_edge(a, merge),
                make_edge(b, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        dispatcher = ThreadDispatcher(max_workers=4)
        try:
            runner = FlowRunner(
                graph=graph,
                node_registry=registry,
                dispatcher=dispatcher,
                on_event=events.append,
            )
            result = runner.run("thread parallel")
            assert result.success
            started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
            assert "A" in started_names
            assert "B" in started_names
        finally:
            dispatcher.shutdown()

    def test_thread_dispatcher_shutdown(self):
        """Shutdown should clean up the thread pool without errors."""
        dispatcher = ThreadDispatcher(max_workers=2)
        dispatcher.shutdown()
        # Should be safe to call twice
        dispatcher.shutdown()


class TestDispatcherErrorHandling:
    """Error handling with different dispatchers."""

    def test_sync_dispatcher_error_captured_in_events(self):
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.STOP,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, end],
            [make_edge(start, failing), make_edge(failing, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("sync error"))

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            dispatcher=SyncDispatcher(),
            on_event=events.append,
        )
        result = runner.run("error test")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert "sync error" in error_events[0].error

    def test_thread_dispatcher_error_captured_in_events(self):
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.STOP,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, end],
            [make_edge(start, failing), make_edge(failing, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("thread error"))

        events: list[FlowEvent] = []
        dispatcher = ThreadDispatcher(max_workers=2)
        try:
            runner = FlowRunner(
                graph=graph,
                node_registry=registry,
                dispatcher=dispatcher,
                on_event=events.append,
            )
            result = runner.run("error test")

            error_events = [e for e in events if isinstance(e, FlowError)]
            assert len(error_events) >= 1
            assert "thread error" in error_events[0].error
        finally:
            dispatcher.shutdown()

    def test_skip_error_continues_to_next_node(self):
        """With SKIP strategy and SyncDispatcher, flow continues past error."""
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.SKIP,
        )
        recovery = make_node(NodeType.STATIC, name="Recovery")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, recovery, end],
            [
                make_edge(start, failing),
                make_edge(failing, recovery),
                make_edge(recovery, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("skip me"))
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            dispatcher=SyncDispatcher(),
            on_event=events.append,
        )
        result = runner.run("skip error test")

        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "Recovery" in started_names


class TestDispatcherDirect:
    """Direct unit-level tests on dispatcher dispatch/wait_all."""

    def test_sync_dispatch_executes_immediately(self):
        """SyncDispatcher.dispatch() calls execute_fn inline."""
        calls: list[tuple] = []

        def execute_fn(fid, nid):
            calls.append((fid, nid))

        d = SyncDispatcher()
        fid, nid = uuid4(), uuid4()
        d.dispatch(fid, nid, execute_fn)

        assert len(calls) == 1
        assert calls[0] == (fid, nid)

    def test_thread_dispatch_executes_in_pool(self):
        """ThreadDispatcher.dispatch() runs in a separate thread."""
        import threading

        thread_ids: list[int] = []
        main_thread = threading.current_thread().ident

        def execute_fn(fid, nid):
            thread_ids.append(threading.current_thread().ident)

        d = ThreadDispatcher(max_workers=2)
        try:
            fid, nid = uuid4(), uuid4()
            d.dispatch(fid, nid, execute_fn)
            d.wait_all()

            assert len(thread_ids) == 1
            assert thread_ids[0] != main_thread
        finally:
            d.shutdown()
