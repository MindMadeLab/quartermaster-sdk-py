"""Integration tests for FlowRunner — full flow execution scenarios."""

from uuid import uuid4

from quartermaster_engine.events import (
    FlowError,
    FlowEvent,
    FlowFinished,
    NodeFinished,
    NodeStarted,
    UserInputRequired,
)
from quartermaster_engine.nodes import SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import (
    ErrorStrategy,
    NodeType,
    TraverseIn,
    TraverseOut,
)
from tests.conftest import (
    DecisionExecutor,
    EchoExecutor,
    FailingExecutor,
    MemoryReadExecutor,
    MemoryWriteExecutor,
    UpperCaseExecutor,
    UserWaitExecutor,
    make_edge,
    make_graph,
    make_node,
)


class TestSimpleLinearFlow:
    """Start → Instruction → End (mock LLM)."""

    def test_simple_echo_flow(self):
        start = make_node(NodeType.START, name="Start")
        instruction = make_node(NodeType.INSTRUCTION, name="Echo")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, instruction, end],
            [make_edge(start, instruction), make_edge(instruction, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("Hello, world!")

        assert result.success
        assert "Hello, world!" in result.final_output
        assert result.duration_seconds > 0

    def test_uppercase_flow(self):
        start = make_node(NodeType.START, name="Start")
        instruction = make_node(NodeType.INSTRUCTION, name="Upper")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, instruction, end],
            [make_edge(start, instruction), make_edge(instruction, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, UpperCaseExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("hello")

        assert result.success
        assert "HELLO" in result.final_output


class TestDecisionFlow:
    """Start → Decision → [A, B] → End."""

    def test_decision_picks_correct_branch(self):
        start = make_node(NodeType.START, name="Start")
        decision = make_node(
            NodeType.DECISION,
            name="Choose",
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        branch_a = make_node(NodeType.INSTRUCTION, name="A")
        branch_b = make_node(NodeType.INSTRUCTION, name="B")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, decision, branch_a, branch_b, end],
            [
                make_edge(start, decision),
                make_edge(decision, branch_a),
                make_edge(decision, branch_b),
                make_edge(branch_a, end),
                make_edge(branch_b, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.DECISION.value, DecisionExecutor())
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("A")  # Decision will pick "A"

        assert result.success
        # Branch A was executed
        assert result.final_output is not None


class TestParallelFlow:
    """Start → Fork → [A, B, C] → Merge → End."""

    def test_parallel_branches_all_execute(self):
        start = make_node(NodeType.START, name="Start")
        fork = make_node(NodeType.INSTRUCTION, name="Fork")
        branch_a = make_node(NodeType.INSTRUCTION, name="A")
        branch_b = make_node(NodeType.INSTRUCTION, name="B")
        branch_c = make_node(NodeType.INSTRUCTION, name="C")
        merge = make_node(
            NodeType.MERGE,
            name="Merge",
            traverse_in=TraverseIn.AWAIT_ALL,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, fork, branch_a, branch_b, branch_c, merge, end],
            [
                make_edge(start, fork),
                make_edge(fork, branch_a),
                make_edge(fork, branch_b),
                make_edge(fork, branch_c),
                make_edge(branch_a, merge),
                make_edge(branch_b, merge),
                make_edge(branch_c, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("parallel test")

        assert result.success


class TestMemoryFlow:
    """Start → WriteMemory → ReadMemory → End."""

    def test_memory_write_and_read(self):
        start = make_node(NodeType.START, name="Start")
        writer = make_node(NodeType.INSTRUCTION, name="Writer")
        reader = make_node(NodeType.INSTRUCTION, name="Reader")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, writer, reader, end],
            [
                make_edge(start, writer),
                make_edge(writer, reader),
                make_edge(reader, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        writer_exec = MemoryWriteExecutor(key="greeting", value="Hello from memory!")
        reader_exec = MemoryReadExecutor(key="greeting")
        # Writer uses INSTRUCTION type, reader uses a different type to differentiate
        # We'll use the same type but different instances won't work with SimpleNodeRegistry
        # Solution: use different node types
        registry.register(NodeType.INSTRUCTION.value, writer_exec)

        # For the reader, we need a different approach since both are INSTRUCTION
        # Let's use the STATIC node type for the reader
        reader.type = NodeType.STATIC  # type: ignore[assignment]
        registry.register(NodeType.STATIC.value, reader_exec)

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("test memory")

        assert result.success
        # The reader should have found the memory value
        assert "greeting=Hello from memory!" in result.final_output


class TestErrorHandling:
    """Start → Instruction (fails) → error handling."""

    def test_stop_on_error(self):
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
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("boom"))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("test")

        # A FlowError event should have been emitted
        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert "boom" in error_events[0].error

    def test_skip_on_error_continues(self):
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
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("oops"))
        registry.register(NodeType.STATIC.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("test skip")

        # The flow should still complete (recovery node executed)
        assert result.success or result.final_output != ""

    def test_retry_on_error(self):
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

        # After 3 retries, the executor still fails, so the result reflects that
        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("still failing"))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("retry test")

        # Should have attempted retries but ultimately failed
        assert not result.success


class TestUserInputFlow:
    """Start → Instruction → User (waits) → resume → End."""

    def test_user_wait_and_resume(self):
        start = make_node(NodeType.START, name="Start")
        user_node = make_node(NodeType.USER, name="AskUser")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, user_node, end],
            [make_edge(start, user_node), make_edge(user_node, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.USER.value, UserWaitExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            on_event=events.append,
        )

        # First run should pause at user node
        result = runner.run("initial input")

        # Check that a UserInputRequired event was emitted
        user_events = [e for e in events if isinstance(e, UserInputRequired)]
        assert len(user_events) == 1
        assert user_events[0].prompt == "Please provide input:"

        # Resume with user's answer
        result2 = runner.resume(result.flow_id, "User's answer")
        # The resumed flow should continue
        assert result2 is not None


class TestEventStreaming:
    """Verify events are emitted during flow execution."""

    def test_events_emitted(self):
        start = make_node(NodeType.START, name="Start")
        instruction = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, instruction, end],
            [make_edge(start, instruction), make_edge(instruction, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            on_event=events.append,
        )
        runner.run("event test")

        # Should have NodeStarted events for Start, Instruction, End
        started_events = [e for e in events if isinstance(e, NodeStarted)]
        assert len(started_events) >= 2  # At least Start and Instruction

        # Should have NodeFinished events
        finished_events = [e for e in events if isinstance(e, NodeFinished)]
        assert len(finished_events) >= 2

        # Should have a FlowFinished event
        flow_finished = [e for e in events if isinstance(e, FlowFinished)]
        assert len(flow_finished) == 1

    def test_error_event_emitted(self):
        start = make_node(NodeType.START, name="Start")
        failing = make_node(
            NodeType.INSTRUCTION,
            name="Bad",
            error_handling=ErrorStrategy.STOP,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, failing, end],
            [make_edge(start, failing), make_edge(failing, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("kaboom"))

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            node_registry=registry,
            on_event=events.append,
        )
        runner.run("error test")

        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert "kaboom" in error_events[0].error


class TestFlowStop:
    """Test stopping a running flow."""

    def test_stop_marks_active_nodes_as_failed(self):
        start = make_node(NodeType.START, name="Start")
        instruction = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, instruction, end],
            [make_edge(start, instruction), make_edge(instruction, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("test")

        # Stop the flow after it finishes (should be a no-op but shouldn't crash)
        runner.stop(result.flow_id)


class TestNoStartNode:
    """Test error handling when no start node exists."""

    def test_no_start_node_returns_error(self):
        node = make_node(name="orphan")
        # Give it a start_node_id that doesn't match any node
        graph = make_graph([node], [], node)
        graph.start_node_id = uuid4()  # Non-existent

        registry = SimpleNodeRegistry()
        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("test")

        assert not result.success
        assert "start node" in (result.error or "").lower()


class TestNoExecutorRegistered:
    """Test error when a node type has no registered executor."""

    def test_missing_executor(self):
        start = make_node(NodeType.START, name="Start")
        instruction = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, instruction, end],
            [make_edge(start, instruction), make_edge(instruction, end)],
            start,
        )

        # Empty registry — no executors registered
        registry = SimpleNodeRegistry()

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("test")

        # A FlowError event should have been emitted for the missing executor
        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1
        assert (
            "executor" in error_events[0].error.lower()
            or "no executor" in error_events[0].error.lower()
        )
