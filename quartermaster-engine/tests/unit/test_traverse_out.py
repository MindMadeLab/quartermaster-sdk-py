"""Tests for TraverseOut branching gate."""

from quartermaster_engine.nodes import NodeResult
from quartermaster_engine.traversal.traverse_out import TraverseOutGate
from quartermaster_engine.types import NodeType, TraverseOut
from tests.conftest import make_edge, make_graph, make_node


class TestSpawnAll:
    def setup_method(self):
        self.gate = TraverseOutGate()

    def test_spawn_all_with_successors(self):
        """SpawnAll should return all successor nodes."""
        start = make_node(NodeType.START, name="Start")
        a = make_node(name="A")
        b = make_node(name="B")
        c = make_node(name="C")
        graph = make_graph(
            [start, a, b, c],
            [make_edge(start, a), make_edge(start, b), make_edge(start, c)],
            start,
        )

        result = NodeResult(success=True, data={})
        next_nodes = self.gate.get_next_nodes(start.id, graph, TraverseOut.SPAWN_ALL, result)
        assert len(next_nodes) == 3
        assert {n.name for n in next_nodes} == {"A", "B", "C"}

    def test_spawn_all_no_successors(self):
        """SpawnAll with no successors returns empty list."""
        node = make_node(name="lonely")
        graph = make_graph([node], [], node)

        result = NodeResult(success=True, data={})
        next_nodes = self.gate.get_next_nodes(node.id, graph, TraverseOut.SPAWN_ALL, result)
        assert next_nodes == []


class TestSpawnNone:
    def setup_method(self):
        self.gate = TraverseOutGate()

    def test_spawn_none(self):
        """SpawnNone should always return empty list."""
        node = make_node(name="end")
        other = make_node(name="other")
        graph = make_graph([node, other], [make_edge(node, other)], node)

        result = NodeResult(success=True, data={})
        next_nodes = self.gate.get_next_nodes(node.id, graph, TraverseOut.SPAWN_NONE, result)
        assert next_nodes == []


class TestSpawnPicked:
    def setup_method(self):
        self.gate = TraverseOutGate()

    def test_pick_by_name(self):
        """SpawnPicked should match successor by name."""
        decision = make_node(NodeType.DECISION, name="Choose")
        yes = make_node(name="Yes")
        no = make_node(name="No")
        graph = make_graph(
            [decision, yes, no],
            [make_edge(decision, yes), make_edge(decision, no)],
            decision,
        )

        result = NodeResult(success=True, data={}, picked_node="Yes")
        next_nodes = self.gate.get_next_nodes(decision.id, graph, TraverseOut.SPAWN_PICKED, result)
        assert len(next_nodes) == 1
        assert next_nodes[0].name == "Yes"

    def test_pick_by_uuid(self):
        """SpawnPicked should match successor by UUID string."""
        decision = make_node(NodeType.DECISION, name="Choose")
        target = make_node(name="Target")
        other = make_node(name="Other")
        graph = make_graph(
            [decision, target, other],
            [make_edge(decision, target), make_edge(decision, other)],
            decision,
        )

        result = NodeResult(success=True, data={}, picked_node=str(target.id))
        next_nodes = self.gate.get_next_nodes(decision.id, graph, TraverseOut.SPAWN_PICKED, result)
        assert len(next_nodes) == 1
        assert next_nodes[0].id == target.id

    def test_pick_by_edge_label(self):
        """SpawnPicked should match successor by edge label."""
        decision = make_node(NodeType.DECISION, name="Choose")
        a = make_node(name="A")
        b = make_node(name="B")
        graph = make_graph(
            [decision, a, b],
            [make_edge(decision, a, label="path_a"), make_edge(decision, b, label="path_b")],
            decision,
        )

        result = NodeResult(success=True, data={}, picked_node="path_b")
        next_nodes = self.gate.get_next_nodes(decision.id, graph, TraverseOut.SPAWN_PICKED, result)
        assert len(next_nodes) == 1
        assert next_nodes[0].name == "B"

    def test_pick_fallback_to_spawn_all(self):
        """When no match found, fallback to SpawnAll."""
        decision = make_node(NodeType.DECISION, name="Choose")
        a = make_node(name="A")
        b = make_node(name="B")
        graph = make_graph(
            [decision, a, b],
            [make_edge(decision, a), make_edge(decision, b)],
            decision,
        )

        result = NodeResult(success=True, data={}, picked_node="NonExistent")
        next_nodes = self.gate.get_next_nodes(decision.id, graph, TraverseOut.SPAWN_PICKED, result)
        assert len(next_nodes) == 2

    def test_pick_no_picked_node_falls_back(self):
        """When picked_node is None, fallback to SpawnAll."""
        decision = make_node(NodeType.DECISION, name="Choose")
        a = make_node(name="A")
        graph = make_graph([decision, a], [make_edge(decision, a)], decision)

        result = NodeResult(success=True, data={}, picked_node=None)
        next_nodes = self.gate.get_next_nodes(decision.id, graph, TraverseOut.SPAWN_PICKED, result)
        assert len(next_nodes) == 1


class TestSpawnStart:
    def setup_method(self):
        self.gate = TraverseOutGate()

    def test_spawn_start_returns_start_node(self):
        """SpawnStart should return the graph's start node."""
        start = make_node(NodeType.START, name="Start")
        middle = make_node(name="Middle")
        graph = make_graph([start, middle], [make_edge(start, middle)], start)

        result = NodeResult(success=True, data={})
        next_nodes = self.gate.get_next_nodes(middle.id, graph, TraverseOut.SPAWN_START, result)
        assert len(next_nodes) == 1
        assert next_nodes[0].id == start.id
