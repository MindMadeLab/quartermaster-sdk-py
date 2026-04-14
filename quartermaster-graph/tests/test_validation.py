"""Tests for graph validation."""

from uuid import uuid4

import pytest

from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import Agent, GraphSpec, GraphEdge, GraphNode
from quartermaster_graph.validation import validate_graph


@pytest.fixture
def agent():
    return Agent(name="Test")


class TestValidGraph:
    def test_simple_graph_passes(self, simple_graph):
        errors = validate_graph(simple_graph)
        assert len(errors) == 0

    def test_decision_graph_passes(self, decision_graph):
        errors = validate_graph(decision_graph)
        assert len(errors) == 0


class TestStartNodeValidation:
    def test_no_start_node(self, agent):
        end = GraphNode(type=NodeType.END, name="End")
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=end.id,
            nodes=[end],
            edges=[],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "no_start" in codes

    def test_multiple_start_nodes(self, agent):
        s1 = GraphNode(type=NodeType.START, name="S1")
        s2 = GraphNode(type=NodeType.START, name="S2")
        end = GraphNode(type=NodeType.END, name="End")
        edges = [
            GraphEdge(source_id=s1.id, target_id=end.id),
            GraphEdge(source_id=s2.id, target_id=end.id),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=s1.id,
            nodes=[s1, s2, end],
            edges=edges,
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "multiple_starts" in codes

    def test_invalid_start_id(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        fake_id = uuid4()
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=fake_id,
            nodes=[start, end],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "invalid_start_id" in codes

    def test_start_id_wrong_type(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=end.id,  # points to End, not Start
            nodes=[start, end],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "start_id_not_start_type" in codes


class TestEndNodeValidation:
    def test_no_end_node(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        inst = GraphNode(type=NodeType.INSTRUCTION, name="Inst")
        edge = GraphEdge(source_id=start.id, target_id=inst.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, inst],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "no_end" in codes


class TestEdgeValidation:
    def test_invalid_source(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=uuid4(), target_id=end.id)
        good_edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end],
            edges=[edge, good_edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "invalid_edge_source" in codes

    def test_invalid_target(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        bad_edge = GraphEdge(source_id=start.id, target_id=uuid4())
        good_edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end],
            edges=[bad_edge, good_edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "invalid_edge_target" in codes


class TestOrphanDetection:
    def test_orphan_node(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        orphan = GraphNode(type=NodeType.INSTRUCTION, name="Orphan")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end, orphan],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "orphan_node" in codes

    def test_comment_nodes_not_flagged(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        comment = GraphNode(type=NodeType.COMMENT, name="A comment")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end, comment],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "orphan_node" not in codes


class TestCycleDetection:
    def test_cycle_detected(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        a = GraphNode(type=NodeType.INSTRUCTION, name="A")
        b = GraphNode(type=NodeType.INSTRUCTION, name="B")
        end = GraphNode(type=NodeType.END, name="End")
        edges = [
            GraphEdge(source_id=start.id, target_id=a.id),
            GraphEdge(source_id=a.id, target_id=b.id),
            GraphEdge(source_id=b.id, target_id=a.id),  # cycle!
            GraphEdge(source_id=b.id, target_id=end.id),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, a, b, end],
            edges=edges,
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "cycle_detected" in codes


class TestDecisionEdgeLabels:
    def test_unlabeled_decision_edges(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        decision = GraphNode(type=NodeType.DECISION, name="Choose")
        a = GraphNode(type=NodeType.END, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        edges = [
            GraphEdge(source_id=start.id, target_id=decision.id),
            GraphEdge(source_id=decision.id, target_id=a.id),  # no label!
            GraphEdge(source_id=decision.id, target_id=b.id, label="No"),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, decision, a, b],
            edges=edges,
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "decision_unlabeled_edges" in codes

    def test_if_node_missing_labels_warning(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        if_node = GraphNode(type=NodeType.IF, name="Check")
        a = GraphNode(type=NodeType.END, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        edges = [
            GraphEdge(source_id=start.id, target_id=if_node.id),
            GraphEdge(source_id=if_node.id, target_id=a.id, label="maybe"),
            GraphEdge(source_id=if_node.id, target_id=b.id, label="possibly"),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, if_node, a, b],
            edges=edges,
        )
        errors = validate_graph(version)
        warnings = [e for e in errors if e.severity == "warning"]
        assert any(e.code == "if_missing_labels" for e in warnings)

    def test_switch_unlabeled_edges(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        switch = GraphNode(type=NodeType.SWITCH, name="Route")
        a = GraphNode(type=NodeType.END, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        edges = [
            GraphEdge(source_id=start.id, target_id=switch.id),
            GraphEdge(source_id=switch.id, target_id=a.id),  # no label
            GraphEdge(source_id=switch.id, target_id=b.id, label="case2"),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, switch, a, b],
            edges=edges,
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "switch_unlabeled_edges" in codes
