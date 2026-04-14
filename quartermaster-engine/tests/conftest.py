"""Shared test fixtures and mock node executors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import pytest

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.types import (
    GraphSpec,
    ErrorStrategy,
    GraphEdge,
    GraphNode,
    MessageRole,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)

# ── Mock Node Executors ──────────────────────────────────────────────────────


class EchoExecutor:
    """Returns the last user message as output. Simplest possible executor."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        user_msgs = [m for m in context.messages if m.role == MessageRole.USER]
        text = user_msgs[-1].content if user_msgs else "echo"
        return NodeResult(success=True, data={}, output_text=text)


class UpperCaseExecutor:
    """Returns the last user message uppercased."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        user_msgs = [m for m in context.messages if m.role == MessageRole.USER]
        text = user_msgs[-1].content.upper() if user_msgs else "ECHO"
        return NodeResult(success=True, data={}, output_text=text)


class DecisionExecutor:
    """Picks a successor based on metadata config or message content."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        # Check metadata for a hardcoded decision
        pick = context.get_meta("pick")
        if not pick:
            # Use the last user message content as the decision
            user_msgs = [m for m in context.messages if m.role == MessageRole.USER]
            pick = user_msgs[-1].content if user_msgs else None
        return NodeResult(
            success=True,
            data={},
            output_text=f"Decided: {pick}",
            picked_node=pick,
        )


class FailingExecutor:
    """Always fails with a configurable error message."""

    def __init__(self, error: str = "Node execution failed") -> None:
        self._error = error

    async def execute(self, context: ExecutionContext) -> NodeResult:
        raise RuntimeError(self._error)


class CountingExecutor:
    """Counts how many times it has been called. Good for loop testing."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, context: ExecutionContext) -> NodeResult:
        self.call_count += 1
        return NodeResult(
            success=True,
            data={"call_count": self.call_count},
            output_text=f"Call #{self.call_count}",
        )


class MemoryWriteExecutor:
    """Writes a value to flow memory."""

    def __init__(self, key: str = "test_key", value: str = "test_value") -> None:
        self._key = key
        self._value = value

    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(
            success=True,
            data={"memory_updates": {self._key: self._value}},
            output_text=f"Wrote {self._key}={self._value}",
        )


class MemoryReadExecutor:
    """Reads a value from flow memory and returns it."""

    def __init__(self, key: str = "test_key") -> None:
        self._key = key

    async def execute(self, context: ExecutionContext) -> NodeResult:
        value = context.memory.get(self._key, "NOT_FOUND")
        return NodeResult(
            success=True,
            data={},
            output_text=f"{self._key}={value}",
        )


class UserWaitExecutor:
    """Pauses for user input."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(
            success=True,
            data={},
            wait_for_user=True,
            user_prompt="Please provide input:",
            user_options=["Option A", "Option B"],
        )


class IfCounterExecutor:
    """Simulates an If node that checks a counter in flow memory.

    Each call increments the counter. If the counter is below *threshold*,
    it picks *loop_target* (the node name to loop back to). Otherwise it
    picks *exit_target* (usually the End node).
    """

    def __init__(
        self,
        counter_key: str = "__counter__",
        threshold: int = 3,
        loop_target: str = "Counter",
        exit_target: str = "End",
    ) -> None:
        self._counter_key = counter_key
        self._threshold = threshold
        self._loop_target = loop_target
        self._exit_target = exit_target

    async def execute(self, context: ExecutionContext) -> NodeResult:
        count = int(context.memory.get(self._counter_key, 0))
        count += 1
        if count < self._threshold:
            picked = self._loop_target
        else:
            picked = self._exit_target
        return NodeResult(
            success=True,
            data={"memory_updates": {self._counter_key: str(count)}},
            output_text=f"counter={count}, picked={picked}",
            picked_node=picked,
        )


class SubAgentExecutor:
    """Simulates an Agent node that runs a nested sub-flow.

    Accepts a *sub_runner_factory* callable that builds and runs the
    inner flow, returning the sub-flow's final output.
    """

    def __init__(self, sub_runner_factory: Callable[..., str] | None = None) -> None:
        self._factory = sub_runner_factory

    async def execute(self, context: ExecutionContext) -> NodeResult:
        if self._factory:
            sub_output = self._factory(context)
        else:
            sub_output = "sub-agent-default-output"
        return NodeResult(
            success=True,
            data={"sub_agent_output": sub_output},
            output_text=sub_output,
        )


class SlowExecutor:
    """Simulates a slow node (for parallel testing)."""

    def __init__(self, delay: float = 0.1, label: str = "slow") -> None:
        self._delay = delay
        self._label = label

    async def execute(self, context: ExecutionContext) -> NodeResult:
        import asyncio

        await asyncio.sleep(self._delay)
        return NodeResult(
            success=True,
            data={"label": self._label},
            output_text=f"{self._label} done",
        )


# ── Graph Builder Helpers ────────────────────────────────────────────────────


def make_node(
    node_type: NodeType = NodeType.INSTRUCTION,
    name: str = "",
    traverse_in: TraverseIn = TraverseIn.AWAIT_ALL,
    traverse_out: TraverseOut = TraverseOut.SPAWN_ALL,
    thought_type: ThoughtType = ThoughtType.CONTINUE,
    message_type: MessageType = MessageType.AUTOMATIC,
    error_handling: ErrorStrategy = ErrorStrategy.STOP,
    metadata: dict | None = None,
    node_id: UUID | None = None,
    max_retries: int = 3,
) -> GraphNode:
    """Create a GraphNode with sensible defaults."""
    return GraphNode(
        id=node_id or uuid4(),
        type=node_type,
        name=name,
        traverse_in=traverse_in,
        traverse_out=traverse_out,
        thought_type=thought_type,
        message_type=message_type,
        error_handling=error_handling,
        metadata=metadata or {},
        max_retries=max_retries,
    )


def make_edge(source: GraphNode, target: GraphNode, label: str = "") -> GraphEdge:
    """Create a GraphEdge between two nodes."""
    return GraphEdge(id=uuid4(), source_id=source.id, target_id=target.id, label=label)


def make_graph(
    nodes: list[GraphNode], edges: list[GraphEdge], start_node: GraphNode
) -> GraphSpec:
    """Create an GraphSpec from nodes and edges."""
    return GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=start_node.id,
        nodes=nodes,
        edges=edges,
    )


def make_linear_graph(
    node_types: list[NodeType], names: list[str] | None = None
) -> tuple[GraphSpec, list[GraphNode]]:
    """Create a simple linear graph: node0 → node1 → node2 → ...

    First node always Start, last always End.
    """
    if names is None:
        names = [f"Node{i}" for i in range(len(node_types))]

    nodes = []
    for i, (ntype, name) in enumerate(zip(node_types, names)):
        tout = TraverseOut.SPAWN_NONE if ntype == NodeType.END else TraverseOut.SPAWN_ALL
        nodes.append(make_node(node_type=ntype, name=name, traverse_out=tout))

    edges = [make_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    graph = make_graph(nodes, edges, nodes[0])
    return graph, nodes


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def registry() -> SimpleNodeRegistry:
    reg = SimpleNodeRegistry()
    reg.register(NodeType.INSTRUCTION.value, EchoExecutor())
    reg.register(NodeType.DECISION.value, DecisionExecutor())
    return reg
