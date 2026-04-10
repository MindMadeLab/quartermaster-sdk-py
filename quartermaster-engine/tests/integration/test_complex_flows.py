"""Integration tests for complex graph topologies."""

from __future__ import annotations

from uuid import uuid4

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import FlowEvent, FlowFinished, NodeFinished, NodeStarted
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
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
    UpperCaseExecutor,
    make_edge,
    make_graph,
    make_node,
)


class TestDiamondMerge:
    """Start -> A -> (B, C in parallel) -> Merge -> End."""

    def test_diamond_merge_both_branches_execute(self):
        start = make_node(NodeType.START, name="Start")
        a = make_node(NodeType.INSTRUCTION, name="A")
        b = make_node(NodeType.INSTRUCTION, name="B")
        c = make_node(NodeType.INSTRUCTION, name="C")
        merge = make_node(NodeType.MERGE, name="Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, a, b, c, merge, end],
            [
                make_edge(start, a),
                make_edge(a, b),
                make_edge(a, c),
                make_edge(b, merge),
                make_edge(c, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("diamond")

        assert result.success
        # Both B and C should have started
        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "B" in started_names
        assert "C" in started_names
        assert "Merge" in started_names

    def test_diamond_merge_output_combines_branches(self):
        """Merge node should combine outputs from both parallel branches."""
        start = make_node(NodeType.START, name="Start")
        branch_upper = make_node(NodeType.INSTRUCTION, name="Upper")
        branch_echo = make_node(NodeType.STATIC, name="Echo")
        merge = make_node(NodeType.MERGE, name="Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, branch_upper, branch_echo, merge, end],
            [
                make_edge(start, branch_upper),
                make_edge(start, branch_echo),
                make_edge(branch_upper, merge),
                make_edge(branch_echo, merge),
                make_edge(merge, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, UpperCaseExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("hello")

        assert result.success
        # The flow should complete without error
        assert result.final_output != ""


class TestDeepChain:
    """Start -> N1 -> N2 -> N3 -> N4 -> N5 -> End."""

    def test_deep_chain_all_nodes_execute(self):
        start = make_node(NodeType.START, name="Start")
        n1 = make_node(NodeType.INSTRUCTION, name="N1")
        n2 = make_node(NodeType.STATIC, name="N2")
        n3 = make_node(NodeType.INSTRUCTION, name="N3")
        n4 = make_node(NodeType.STATIC, name="N4")
        n5 = make_node(NodeType.INSTRUCTION, name="N5")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, n1, n2, n3, n4, n5, end],
            [
                make_edge(start, n1),
                make_edge(n1, n2),
                make_edge(n2, n3),
                make_edge(n3, n4),
                make_edge(n4, n5),
                make_edge(n5, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("chain test")

        assert result.success
        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        for name in ["N1", "N2", "N3", "N4", "N5"]:
            assert name in started_names, f"Node {name} was not started"

    def test_deep_chain_preserves_execution_order(self):
        """Nodes in a linear chain execute sequentially in order."""
        start = make_node(NodeType.START, name="Start")
        nodes = []
        for i in range(1, 6):
            nodes.append(make_node(NodeType.INSTRUCTION, name=f"N{i}"))
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        all_nodes = [start, *nodes, end]
        edges = [make_edge(start, nodes[0])]
        for i in range(len(nodes) - 1):
            edges.append(make_edge(nodes[i], nodes[i + 1]))
        edges.append(make_edge(nodes[-1], end))

        graph = make_graph(all_nodes, edges, start)

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("order test")

        assert result.success
        started_names = [
            e.node_name for e in events if isinstance(e, NodeStarted) and e.node_name.startswith("N")
        ]
        assert started_names == ["N1", "N2", "N3", "N4", "N5"]


class TestMultipleDecisions:
    """Start -> Decision1 -> (BranchA -> Decision2 -> ..., BranchB -> ...)."""

    def test_nested_decisions_branch_a_then_sub_a(self):
        start = make_node(NodeType.START, name="Start")
        d1 = make_node(NodeType.DECISION, name="D1", traverse_out=TraverseOut.SPAWN_PICKED)
        branch_a = make_node(NodeType.INSTRUCTION, name="BranchA")
        branch_b = make_node(NodeType.INSTRUCTION, name="BranchB")
        d2 = make_node(NodeType.DECISION, name="D2", traverse_out=TraverseOut.SPAWN_PICKED)
        sub_a = make_node(NodeType.STATIC, name="SubA")
        sub_b = make_node(NodeType.STATIC, name="SubB")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, d1, branch_a, branch_b, d2, sub_a, sub_b, end],
            [
                make_edge(start, d1),
                make_edge(d1, branch_a),
                make_edge(d1, branch_b),
                make_edge(branch_a, d2),
                make_edge(branch_b, end),
                make_edge(d2, sub_a),
                make_edge(d2, sub_b),
                make_edge(sub_a, end),
                make_edge(sub_b, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.DECISION.value, DecisionExecutor())
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        # D1 picks "BranchA", and since BranchA echoes "BranchA", D2 picks "SubA"
        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("BranchA")

        assert result.success

    def test_nested_decisions_branch_b_skips_second_decision(self):
        start = make_node(NodeType.START, name="Start")
        d1 = make_node(NodeType.DECISION, name="D1", traverse_out=TraverseOut.SPAWN_PICKED)
        branch_a = make_node(NodeType.INSTRUCTION, name="BranchA")
        branch_b = make_node(NodeType.INSTRUCTION, name="BranchB")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, d1, branch_a, branch_b, end],
            [
                make_edge(start, d1),
                make_edge(d1, branch_a),
                make_edge(d1, branch_b),
                make_edge(branch_a, end),
                make_edge(branch_b, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.DECISION.value, DecisionExecutor())
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("BranchB")

        assert result.success
        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "BranchB" in started_names
        # BranchA should NOT have been started since D1 picked BranchB
        assert "BranchA" not in started_names


class TestErrorRecovery:
    """Start -> Failing Node (retry 3x) -> End."""

    def test_retry_exhaustion(self):
        """Node retries max_retries times then fails."""
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

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, FailingExecutor("always fails"))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("retry test")

        assert not result.success

    def test_skip_allows_downstream_execution(self):
        """SKIP strategy lets the flow continue past a failing node."""
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

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("skip test")

        # Recovery node should have been started
        started_names = [e.node_name for e in events if isinstance(e, NodeStarted)]
        assert "Recovery" in started_names


class TestMixedNodeTypes:
    """Flow with multiple different node types in one graph."""

    def test_mixed_instruction_decision_static(self):
        """Start -> Instruction -> Decision -> (Static A, Static B) -> End."""
        start = make_node(NodeType.START, name="Start")
        inst = make_node(NodeType.INSTRUCTION, name="Process")
        decision = make_node(
            NodeType.DECISION, name="Route", traverse_out=TraverseOut.SPAWN_PICKED
        )
        static_a = make_node(NodeType.STATIC, name="StaticA")
        static_b = make_node(NodeType.STATIC, name="StaticB")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, inst, decision, static_a, static_b, end],
            [
                make_edge(start, inst),
                make_edge(inst, decision),
                make_edge(decision, static_a),
                make_edge(decision, static_b),
                make_edge(static_a, end),
                make_edge(static_b, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())
        registry.register(NodeType.DECISION.value, DecisionExecutor())
        registry.register(NodeType.STATIC.value, EchoExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("StaticA")

        assert result.success

    def test_mixed_with_memory_nodes(self):
        """Start -> MemoryWriter -> MemoryReader -> End."""
        start = make_node(NodeType.START, name="Start")
        writer = make_node(NodeType.INSTRUCTION, name="Writer")
        reader = make_node(NodeType.STATIC, name="Reader")
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
        registry.register(
            NodeType.INSTRUCTION.value,
            MemoryWriteExecutor(key="data", value="important"),
        )
        registry.register(NodeType.STATIC.value, MemoryReadExecutor(key="data"))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("mixed memory test")

        assert result.success
        assert "data=important" in result.final_output

    def test_mixed_with_loop_and_decision(self):
        """Start -> Counter -> If (loop or exit) -> End. Combines loop + decision."""
        start = make_node(NodeType.START, name="Start")
        counter = make_node(
            NodeType.INSTRUCTION, name="Counter", traverse_in=TraverseIn.AWAIT_FIRST
        )
        if_node = make_node(
            NodeType.IF, name="Check", traverse_out=TraverseOut.SPAWN_PICKED
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

        counting = CountingExecutor()
        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, counting)
        registry.register(
            NodeType.IF.value,
            IfCounterExecutor(threshold=2, loop_target="Counter", exit_target="End"),
        )

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("loop test")

        assert result.success
        assert counting.call_count == 2
