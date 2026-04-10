"""Test fixtures and mock implementations for qm-nodes tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest

from qm_nodes.protocols import ExpressionResult


@dataclass
class MockThought:
    """Mock implementation of the Thought protocol."""

    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _child_thoughts: list = field(default_factory=list)

    def get_previous_child_thoughts(self) -> list:
        return self._child_thoughts


@dataclass
class MockHandle:
    """Mock implementation of the ThoughtHandle protocol."""

    texts: list[str] = field(default_factory=list)
    metadata_updates: list[dict[str, Any]] = field(default_factory=list)

    def append_text(self, text: str) -> None:
        self.texts.append(text)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata_updates.append(metadata)

    @property
    def last_text(self) -> str:
        return self.texts[-1] if self.texts else ""

    @property
    def all_text(self) -> str:
        return "".join(self.texts)

    @property
    def last_metadata_update(self) -> dict:
        return self.metadata_updates[-1] if self.metadata_updates else {}


@dataclass
class MockEdge:
    """Mock implementation of the Edge protocol."""

    tail_id: Any = None
    main_direction: bool = True
    direction_text: str = ""


class MockEdgeQuerySet:
    """Mock implementation of the EdgeQuerySet protocol."""

    def __init__(self, edges: list[MockEdge] | None = None):
        self._edges = edges or []

    def all(self) -> list[MockEdge]:
        return self._edges


@dataclass
class MockAssistantNode:
    """Mock implementation of the AssistantNode protocol."""

    predecessor_edges: MockEdgeQuerySet = field(default_factory=MockEdgeQuerySet)


@dataclass
class MockExpressionEvaluator:
    """Mock implementation of the ExpressionEvaluator protocol."""

    results: dict[str, Any] = field(default_factory=dict)

    def eval_expression(
        self, node_id: Any, expression: str, context: dict[str, Any]
    ) -> ExpressionResult:
        if expression in self.results:
            return ExpressionResult(result=self.results[expression])
        # Fall back to eval for simple expressions
        try:
            result = eval(expression, {"__builtins__": {}}, context)
            return ExpressionResult(result=result)
        except Exception as e:
            return ExpressionResult(result=None, error=str(e), success=False)


@dataclass
class MockNodeContext:
    """Mock implementation of the NodeContext protocol.

    Use this in tests to create a fully controllable context
    for testing nodes in isolation.
    """

    node_metadata: dict[str, Any] = field(default_factory=dict)
    flow_node_id: UUID = field(default_factory=uuid4)
    thought_id: Optional[UUID] = field(default_factory=uuid4)
    thought: Optional[MockThought] = field(default_factory=MockThought)
    handle: Optional[MockHandle] = field(default_factory=MockHandle)
    assistant_node: MockAssistantNode = field(default_factory=MockAssistantNode)
    chat_id: Optional[UUID] = field(default_factory=uuid4)
    user: Any = None
    flow_node: Any = None


@pytest.fixture
def mock_context():
    """Create a fresh mock context for each test."""
    return MockNodeContext()


@pytest.fixture
def mock_context_with_metadata():
    """Create a context factory that accepts metadata."""

    def _make(metadata: dict[str, Any] | None = None, **kwargs) -> MockNodeContext:
        ctx = MockNodeContext(node_metadata=metadata or {}, **kwargs)
        return ctx

    return _make


@pytest.fixture
def mock_evaluator():
    """Create a mock expression evaluator."""
    return MockExpressionEvaluator()


@pytest.fixture
def mock_edges():
    """Create a factory for mock edges."""

    def _make(*edges: tuple[Any, bool, str]) -> MockEdgeQuerySet:
        return MockEdgeQuerySet([
            MockEdge(tail_id=tid, main_direction=md, direction_text=dt)
            for tid, md, dt in edges
        ])

    return _make
