"""Benchmark tests for flow execution performance."""

import time
from uuid import uuid4

import pytest

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.types import (
    Message,
    MessageRole,
    NodeType,
    TraverseOut,
)
from tests.conftest import make_edge, make_graph, make_node


class NoOpExecutor:
    """Minimal executor for benchmarking — does nothing."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(success=True, data={}, output_text="ok")


class TestLinearGraphBenchmark:
    """Benchmark: linear chain of N nodes."""

    @pytest.mark.parametrize("num_nodes", [10, 50, 100])
    def test_linear_chain(self, num_nodes: int):
        # Build: Start → N1 → N2 → ... → Nn → End
        nodes = [make_node(NodeType.START, name="Start")]
        for i in range(num_nodes):
            nodes.append(make_node(NodeType.INSTRUCTION, name=f"N{i}"))
        nodes.append(make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE))

        edges = [make_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        graph = make_graph(nodes, edges, nodes[0])

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, NoOpExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)

        start_time = time.monotonic()
        result = runner.run("benchmark")
        elapsed = time.monotonic() - start_time

        assert result.success, f"Flow failed: {result.error}"
        print(f"\n  Linear chain ({num_nodes} nodes): {elapsed:.4f}s")


class TestParallelBranchBenchmark:
    """Benchmark: fan-out to N parallel branches, then merge."""

    @pytest.mark.parametrize("num_branches", [5, 10, 20])
    def test_parallel_fan_out(self, num_branches: int):
        start = make_node(NodeType.START, name="Start")
        fork = make_node(NodeType.INSTRUCTION, name="Fork")
        merge = make_node(NodeType.MERGE, name="Merge")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        branches = [make_node(NodeType.INSTRUCTION, name=f"Branch{i}") for i in range(num_branches)]

        nodes = [start, fork] + branches + [merge, end]
        edges = [make_edge(start, fork)]
        edges.extend(make_edge(fork, b) for b in branches)
        edges.extend(make_edge(b, merge) for b in branches)
        edges.append(make_edge(merge, end))

        graph = make_graph(nodes, edges, start)

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, NoOpExecutor())

        runner = FlowRunner(graph=graph, node_registry=registry)

        start_time = time.monotonic()
        result = runner.run("parallel benchmark")
        elapsed = time.monotonic() - start_time

        assert result.success, f"Flow failed: {result.error}"
        print(f"\n  Parallel fan-out ({num_branches} branches): {elapsed:.4f}s")


class TestMemoryBenchmark:
    """Benchmark: memory operations with many keys."""

    def test_memory_throughput(self):
        store = InMemoryStore()
        flow_id = uuid4()

        num_ops = 1000

        start = time.monotonic()
        for i in range(num_ops):
            store.save_memory(flow_id, f"key_{i}", f"value_{i}")
        write_time = time.monotonic() - start

        start = time.monotonic()
        for i in range(num_ops):
            store.get_memory(flow_id, f"key_{i}")
        read_time = time.monotonic() - start

        print(
            f"\n  Memory: {num_ops} writes in {write_time:.4f}s, {num_ops} reads in {read_time:.4f}s"
        )

        assert write_time < 1.0  # Should be very fast for in-memory
        assert read_time < 1.0


class TestMessageHistoryBenchmark:
    """Benchmark: message accumulation with large histories."""

    def test_large_message_history(self):
        store = InMemoryStore()
        flow_id, node_id = uuid4(), uuid4()

        num_messages = 500
        messages = [
            Message(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"Message number {i} with some content to simulate real usage. " * 5,
            )
            for i in range(num_messages)
        ]

        start = time.monotonic()
        store.save_messages(flow_id, node_id, messages)
        save_time = time.monotonic() - start

        start = time.monotonic()
        retrieved = store.get_messages(flow_id, node_id)
        read_time = time.monotonic() - start

        assert len(retrieved) == num_messages
        print(f"\n  Messages: save {num_messages} in {save_time:.4f}s, read in {read_time:.4f}s")
