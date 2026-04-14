"""Tests for graph traversal utilities."""

from uuid import uuid4

import pytest

from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import GraphSpec, GraphEdge, GraphNode
from quartermaster_graph.traversal import (
    find_decision_points,
    find_merge_points,
    get_path,
    get_predecessors,
    get_start_node,
    get_successors,
    topological_sort,
)


class TestGetStartNode:
    def test_finds_start(self, simple_graph):
        start = get_start_node(simple_graph)
        assert start.type == NodeType.START
        assert start.name == "Start"

    def test_raises_if_no_start(self, agent):
        end = GraphNode(type=NodeType.END, name="End")
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=uuid4(),
            nodes=[end],
            edges=[],
        )
        with pytest.raises(ValueError, match="No start node"):
            get_start_node(version)


class TestGetSuccessors:
    def test_simple(self, simple_graph):
        start = simple_graph.nodes[0]
        succs = get_successors(simple_graph, start.id)
        assert len(succs) == 1
        assert succs[0].type == NodeType.INSTRUCTION

    def test_decision_has_multiple(self, decision_graph):
        decision = [n for n in decision_graph.nodes if n.type == NodeType.DECISION][0]
        succs = get_successors(decision_graph, decision.id)
        assert len(succs) == 2

    def test_end_has_none(self, simple_graph):
        end = [n for n in simple_graph.nodes if n.type == NodeType.END][0]
        succs = get_successors(simple_graph, end.id)
        assert len(succs) == 0


class TestGetPredecessors:
    def test_simple(self, simple_graph):
        end = [n for n in simple_graph.nodes if n.type == NodeType.END][0]
        preds = get_predecessors(simple_graph, end.id)
        assert len(preds) == 1
        assert preds[0].type == NodeType.INSTRUCTION

    def test_start_has_none(self, simple_graph):
        start = simple_graph.nodes[0]
        preds = get_predecessors(simple_graph, start.id)
        assert len(preds) == 0


class TestGetPath:
    def test_direct_path(self, simple_graph):
        start = simple_graph.nodes[0]
        end = [n for n in simple_graph.nodes if n.type == NodeType.END][0]
        path = get_path(simple_graph, start.id, end.id)
        assert len(path) == 3
        assert path[0].type == NodeType.START
        assert path[-1].type == NodeType.END

    def test_no_path(self, agent):
        a = GraphNode(type=NodeType.START, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=a.id,
            nodes=[a, b],
            edges=[],  # no edges
        )
        path = get_path(version, a.id, b.id)
        assert path == []

    def test_self_path(self, simple_graph):
        start = simple_graph.nodes[0]
        path = get_path(simple_graph, start.id, start.id)
        assert len(path) == 1
        assert path[0].id == start.id

    def test_nonexistent_node(self, simple_graph):
        path = get_path(simple_graph, uuid4(), uuid4())
        assert path == []


class TestTopologicalSort:
    def test_simple_graph(self, simple_graph):
        sorted_nodes = topological_sort(simple_graph)
        assert len(sorted_nodes) == 3
        assert sorted_nodes[0].type == NodeType.START
        assert sorted_nodes[-1].type == NodeType.END

    def test_decision_graph(self, decision_graph):
        sorted_nodes = topological_sort(decision_graph)
        assert len(sorted_nodes) == 6
        # Start should be first
        assert sorted_nodes[0].type == NodeType.START

    def test_cycle_raises(self, agent):
        a = GraphNode(type=NodeType.START, name="A")
        b = GraphNode(type=NodeType.INSTRUCTION, name="B")
        c = GraphNode(type=NodeType.END, name="C")
        edges = [
            GraphEdge(source_id=a.id, target_id=b.id),
            GraphEdge(source_id=b.id, target_id=c.id),
            GraphEdge(source_id=c.id, target_id=b.id),  # cycle
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=a.id,
            nodes=[a, b, c],
            edges=edges,
        )
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(version)


class TestFindMergePoints:
    def test_no_merge_in_simple(self, simple_graph):
        merges = find_merge_points(simple_graph)
        assert len(merges) == 0

    def test_finds_merge(self, agent):
        start = GraphNode(type=NodeType.START, name="Start")
        a = GraphNode(type=NodeType.INSTRUCTION, name="A")
        b = GraphNode(type=NodeType.INSTRUCTION, name="B")
        merge = GraphNode(type=NodeType.MERGE, name="Merge")
        end = GraphNode(type=NodeType.END, name="End")
        edges = [
            GraphEdge(source_id=start.id, target_id=a.id),
            GraphEdge(source_id=start.id, target_id=b.id),
            GraphEdge(source_id=a.id, target_id=merge.id),
            GraphEdge(source_id=b.id, target_id=merge.id),
            GraphEdge(source_id=merge.id, target_id=end.id),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, a, b, merge, end],
            edges=edges,
        )
        merges = find_merge_points(version)
        assert len(merges) == 1
        assert merges[0].name == "Merge"


class TestFindDecisionPoints:
    def test_no_decisions_in_simple(self, simple_graph):
        decisions = find_decision_points(simple_graph)
        assert len(decisions) == 0

    def test_finds_decision(self, decision_graph):
        decisions = find_decision_points(decision_graph)
        assert len(decisions) >= 1
        types = [d.type for d in decisions]
        assert NodeType.DECISION in types or NodeType.START in types
