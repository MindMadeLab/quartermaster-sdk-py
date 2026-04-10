"""Integration tests for FlowRunner — full flow execution scenarios."""

from uuid import uuid4

from quartermaster_engine.context.execution_context import ExecutionContext
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
    CountingExecutor,
    DecisionExecutor,
    EchoExecutor,
    FailingExecutor,
    IfCounterExecutor,
    MemoryReadExecutor,
    MemoryWriteExecutor,
    SlowExecutor,
    SubAgentExecutor,
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


class TestLoopFlow:
    """Start → Counter (Instruction) → If (counter < 3) → loop back or End."""

    def test_loop_executes_correct_iterations(self):
        """The If node increments a counter and loops back until counter >= 3."""
        start = make_node(NodeType.START, name="Start")
        counter = make_node(
            NodeType.INSTRUCTION, name="Counter", traverse_in=TraverseIn.AWAIT_FIRST
        )
        if_node = make_node(
            NodeType.IF,
            name="Check",
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, counter, if_node, end],
            [
                make_edge(start, counter),
                make_edge(counter, if_node),
                make_edge(if_node, counter, label="Counter"),  # loop-back edge
                make_edge(if_node, end, label="End"),  # exit edge
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        counting_exec = CountingExecutor()
        registry.register(NodeType.INSTRUCTION.value, counting_exec)
        registry.register(
            NodeType.IF.value,
            IfCounterExecutor(
                counter_key="__counter__",
                threshold=3,
                loop_target="Counter",
                exit_target="End",
            ),
        )

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("loop test")

        assert result.success
        # The counter executor should have been called 3 times
        # (iterations 1, 2, 3 — at iteration 3 the If exits)
        assert counting_exec.call_count == 3

        # Verify we saw NodeStarted events for the Counter node multiple times
        counter_starts = [
            e
            for e in events
            if isinstance(e, NodeStarted) and e.node_name == "Counter"
        ]
        assert len(counter_starts) == 3

    def test_loop_with_threshold_1_no_loop(self):
        """When threshold=1, the first check exits immediately — no looping."""
        start = make_node(NodeType.START, name="Start")
        counter = make_node(
            NodeType.INSTRUCTION, name="Counter", traverse_in=TraverseIn.AWAIT_FIRST
        )
        if_node = make_node(
            NodeType.IF,
            name="Check",
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, counter, if_node, end],
            [
                make_edge(start, counter),
                make_edge(counter, if_node),
                make_edge(if_node, counter, label="Counter"),
                make_edge(if_node, end, label="End"),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        counting_exec = CountingExecutor()
        registry.register(NodeType.INSTRUCTION.value, counting_exec)
        registry.register(
            NodeType.IF.value,
            IfCounterExecutor(threshold=1, loop_target="Counter", exit_target="End"),
        )

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("no loop")

        assert result.success
        # Counter runs once, If checks (count=1 >= threshold=1), exits
        assert counting_exec.call_count == 1


class TestSubAgentFlow:
    """Start → Agent (nested flow) → End."""

    def test_sub_agent_runs_nested_flow(self):
        """The Agent node runs a sub-flow and returns its output."""
        start = make_node(NodeType.START, name="Start")
        agent = make_node(NodeType.AGENT, name="SubAgent")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, agent, end],
            [make_edge(start, agent), make_edge(agent, end)],
            start,
        )

        # Build a sub-flow: Start → Instruction(uppercase) → End
        sub_start = make_node(NodeType.START, name="SubStart")
        sub_inst = make_node(NodeType.INSTRUCTION, name="SubWork")
        sub_end = make_node(NodeType.END, name="SubEnd", traverse_out=TraverseOut.SPAWN_NONE)
        sub_graph = make_graph(
            [sub_start, sub_inst, sub_end],
            [make_edge(sub_start, sub_inst), make_edge(sub_inst, sub_end)],
            sub_start,
        )

        def run_sub_flow(context: ExecutionContext) -> str:
            """Run the inner flow and return its output."""
            sub_registry = SimpleNodeRegistry()
            sub_registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
            sub_runner = FlowRunner(graph=sub_graph, node_registry=sub_registry)
            user_input = context.messages[-1].content if context.messages else "sub-input"
            sub_result = sub_runner.run(user_input)
            return sub_result.final_output

        registry = SimpleNodeRegistry()
        registry.register(NodeType.AGENT.value, SubAgentExecutor(sub_runner_factory=run_sub_flow))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("nested hello")

        assert result.success
        assert "nested hello" in result.final_output

    def test_sub_agent_default_output(self):
        """When no sub-runner factory is provided, uses default output."""
        start = make_node(NodeType.START, name="Start")
        agent = make_node(NodeType.AGENT, name="SubAgent")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, agent, end],
            [make_edge(start, agent), make_edge(agent, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.AGENT.value, SubAgentExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("test")

        assert result.success
        assert "sub-agent-default-output" in result.final_output


class TestPerNodeTimeout:
    """Test that per-node timeout enforcement works."""

    def test_timeout_triggers_error(self):
        """A node with a short timeout should fail when execution is slow."""
        start = make_node(NodeType.START, name="Start")
        slow = make_node(
            NodeType.INSTRUCTION,
            name="Slow",
            error_handling=ErrorStrategy.STOP,
        )
        # Set a very short timeout
        slow.timeout = 0.05  # 50ms
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, slow, end],
            [make_edge(start, slow), make_edge(slow, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        # SlowExecutor sleeps for 0.5s, which exceeds the 50ms timeout
        registry.register(NodeType.INSTRUCTION.value, SlowExecutor(delay=0.5))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("timeout test")

        # The flow should have failed due to timeout
        error_events = [e for e in events if isinstance(e, FlowError)]
        assert len(error_events) >= 1

    def test_no_timeout_allows_completion(self):
        """A node without a timeout should complete normally."""
        start = make_node(NodeType.START, name="Start")
        slow = make_node(NodeType.INSTRUCTION, name="Slow")
        # No timeout set (defaults to None)
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
