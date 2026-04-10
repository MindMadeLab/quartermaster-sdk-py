"""Integration tests for error handling strategies in flow execution."""

from __future__ import annotations

from uuid import uuid4

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import FlowError, FlowEvent, FlowFinished, NodeFinished, NodeStarted
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.types import (
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


class _FlakyExecutor:
    """Fails the first N calls, then succeeds."""

    def __init__(self, fail_count: int = 2) -> None:
        self._fail_count = fail_count
        self.call_count = 0

    async def execute(self, context: ExecutionContext) -> NodeResult:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError(f"Flaky failure #{self.call_count}")
        return NodeResult(success=True, data={}, output_text="recovered")


class TestErrorStrategyStop:
    """ErrorStrategy.STOP: flow stops on first error."""

    def test_stop_emits_error_and_does_not_dispatch_after(self):
        """STOP error emits a FlowError event. The failed node does not dispatch successors."""
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.STOP,
        )
        after = make_node(NodeType.STATIC, name="After")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, after, end],
            [
                make_edge(start, failing),
                make_edge(failing, after),
                make_edge(after, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("fatal"))
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("stop test")

        # A FlowError event should have been emitted
        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert "fatal" in error_events[0].error

    def test_stop_emits_error_event(self):
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
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("stop error"))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("error emit test")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert error_events[0].recoverable is False
        assert "stop error" in error_events[0].error


class TestErrorStrategySkip:
    """ErrorStrategy.SKIP: flow continues past errors."""

    def test_skip_continues_to_downstream(self):
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.SKIP,
        )
        next_node = make_node(NodeType.STATIC, name="Next")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, next_node, end],
            [
                make_edge(start, failing),
                make_edge(failing, next_node),
                make_edge(next_node, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("skip me"))
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("skip test")

        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "Next" in started_names

    def test_skip_error_is_marked_recoverable(self):
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.SKIP,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, end],
            [make_edge(start, failing), make_edge(failing, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("recoverable"))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("recoverable test")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert error_events[0].recoverable is True


class TestErrorStrategyRetry:
    """ErrorStrategy.RETRY: node retries N times then fails."""

    def test_retry_exhaustion_then_stop(self):
        start = make_node(NodeType.START, name="Start")
        flaky = make_node(
            NodeType.INSTRUCTION,
            name="Flaky",
            error_handling=ErrorStrategy.RETRY,
            max_retries=2,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, flaky, end],
            [make_edge(start, flaky), make_edge(flaky, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("always"))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("retry exhaust")

        assert not result.success

    def test_retry_recovers_after_transient_failure(self):
        """Node fails twice then succeeds on third attempt."""
        start = make_node(NodeType.START, name="Start")
        flaky = make_node(
            NodeType.INSTRUCTION,
            name="Flaky",
            error_handling=ErrorStrategy.RETRY,
            max_retries=3,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, flaky, end],
            [make_edge(start, flaky), make_edge(flaky, end)],
            start,
        )

        flaky_exec = _FlakyExecutor(fail_count=2)
        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, flaky_exec)

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("flaky recovery")

        assert result.success
        assert flaky_exec.call_count == 3  # 2 failures + 1 success
        assert "recovered" in result.final_output


class TestErrorPropagation:
    """Error effects on downstream nodes and parallel branches."""

    def test_stop_error_emits_error_for_failing_node(self):
        """With STOP, a FlowError is emitted for the failing node."""
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Failing",
            error_handling=ErrorStrategy.STOP,
        )
        successor = make_node(NodeType.STATIC, name="Successor")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, successor, end],
            [
                make_edge(start, failing),
                make_edge(failing, successor),
                make_edge(successor, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("blocked"))
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("propagation")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert "blocked" in error_events[0].error
        assert error_events[0].recoverable is False

    def test_error_in_one_parallel_branch_stop(self):
        """With STOP on a parallel branch, the failing branch stops."""
        start = make_node(NodeType.START, name="Start")
        good = make_node(NodeType.INSTRUCTION, name="Good")
        bad = make_node(
            NodeType.STATIC,
            name="Bad",
            error_handling=ErrorStrategy.STOP,
        )
        merge = make_node(NodeType.MERGE, name="Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, good, bad, merge, end],
            [
                make_edge(start, good),
                make_edge(start, bad),
                make_edge(good, merge),
                make_edge(bad, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.STATIC.value, FailingExecutor("branch fail"))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("parallel error")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1


class TestTimeoutErrors:
    """Per-node timeout enforcement."""

    def test_timeout_triggers_error_event(self):
        start = make_node(NodeType.START, name="Start")
        slow = make_node(
            NodeType.INSTRUCTION,
            name="Slow",
            error_handling=ErrorStrategy.STOP,
        )
        slow.timeout = 0.05  # 50ms
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, slow, end],
            [make_edge(start, slow), make_edge(slow, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, SlowExecutor(delay=0.5))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("timeout test")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1

    def test_no_timeout_completes_normally(self):
        start = make_node(NodeType.START, name="Start")
        slow = make_node(NodeType.INSTRUCTION, name="Slow")
        # No timeout set
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, slow, end],
            [make_edge(start, slow), make_edge(slow, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, SlowExecutor(delay=0.05))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("no timeout")

        assert result.success

    def test_timeout_with_retry_retries_then_fails(self):
        """Timeout on a node with RETRY strategy should retry then ultimately fail."""
        start = make_node(NodeType.START, name="Start")
        slow = make_node(
            NodeType.INSTRUCTION,
            name="Slow",
            error_handling=ErrorStrategy.RETRY,
            max_retries=2,
        )
        slow.timeout = 0.05
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, slow, end],
            [make_edge(start, slow), make_edge(slow, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, SlowExecutor(delay=0.5))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("timeout retry")

        assert not result.success
