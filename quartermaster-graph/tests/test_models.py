"""Tests for core Pydantic models."""

from datetime import datetime
from uuid import UUID, uuid4

from quartermaster_graph.enums import (
    ErrorStrategy,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)
from quartermaster_graph.models import (
    Agent,
    AgentGraph,
    GraphEdge,
    GraphNode,
    NodePosition,
)


class TestNodePosition:
    def test_defaults(self):
        pos = NodePosition()
        assert pos.x == 0
        assert pos.y == 0
        assert pos.icon is None

    def test_custom_values(self):
        pos = NodePosition(x=100, y=200, icon="star")
        assert pos.x == 100
        assert pos.y == 200
        assert pos.icon == "star"

    def test_serialization(self):
        pos = NodePosition(x=50, y=75, icon="cog")
        data = pos.model_dump()
        restored = NodePosition.model_validate(data)
        assert restored == pos


class TestGraphNode:
    def test_defaults(self):
        node = GraphNode(type=NodeType.INSTRUCTION)
        assert isinstance(node.id, UUID)
        assert node.type == NodeType.INSTRUCTION
        assert node.name == ""
        assert node.traverse_in == TraverseIn.AWAIT_ALL
        assert node.traverse_out == TraverseOut.SPAWN_ALL
        assert node.thought_type == ThoughtType.NEW
        assert node.message_type == MessageType.AUTOMATIC
        assert node.error_handling == ErrorStrategy.STOP
        assert node.metadata == {}
        assert node.position is None

    def test_custom_values(self):
        node = GraphNode(
            type=NodeType.DECISION,
            name="My Decision",
            traverse_in=TraverseIn.AWAIT_FIRST,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"decision_prompt": "Choose wisely"},
            position=NodePosition(x=10, y=20),
        )
        assert node.name == "My Decision"
        assert node.traverse_out == TraverseOut.SPAWN_PICKED
        assert node.metadata["decision_prompt"] == "Choose wisely"

    def test_serialization_roundtrip(self):
        node = GraphNode(
            type=NodeType.CODE,
            name="Run Code",
            metadata={"language": "python", "code": "print('hello')"},
        )
        data = node.model_dump(mode="json")
        restored = GraphNode.model_validate(data)
        assert restored.type == node.type
        assert restored.name == node.name
        assert restored.metadata == node.metadata

    def test_unique_ids(self):
        n1 = GraphNode(type=NodeType.START)
        n2 = GraphNode(type=NodeType.START)
        assert n1.id != n2.id


class TestGraphEdge:
    def test_creation(self):
        src = uuid4()
        tgt = uuid4()
        edge = GraphEdge(source_id=src, target_id=tgt, label="Yes")
        assert edge.source_id == src
        assert edge.target_id == tgt
        assert edge.label == "Yes"
        assert edge.is_main is True
        assert edge.points == []

    def test_with_points(self):
        edge = GraphEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            points=[(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)],
        )
        assert len(edge.points) == 3

    def test_serialization_roundtrip(self):
        edge = GraphEdge(
            source_id=uuid4(),
            target_id=uuid4(),
            label="No",
            is_main=False,
            points=[(1.0, 2.0)],
        )
        data = edge.model_dump(mode="json")
        restored = GraphEdge.model_validate(data)
        assert restored.label == edge.label
        assert restored.is_main == edge.is_main


class TestAgent:
    def test_defaults(self):
        agent = Agent(name="Test")
        assert agent.name == "Test"
        assert agent.description == ""
        assert agent.tags == []
        assert isinstance(agent.created_at, datetime)

    def test_with_tags(self):
        agent = Agent(name="Tagged", tags=["ai", "workflow"])
        assert "ai" in agent.tags

    def test_serialization(self):
        agent = Agent(name="Ser", description="test", tags=["a"])
        data = agent.model_dump(mode="json")
        restored = Agent.model_validate(data)
        assert restored.name == agent.name


class TestAgentGraph:
    def test_creation(self, simple_graph):
        assert len(simple_graph.nodes) == 3
        assert len(simple_graph.edges) == 2

    def test_serialization_roundtrip(self, simple_graph):
        data = simple_graph.model_dump(mode="json")
        restored = AgentGraph.model_validate(data)
        assert len(restored.nodes) == len(simple_graph.nodes)
        assert len(restored.edges) == len(simple_graph.edges)
