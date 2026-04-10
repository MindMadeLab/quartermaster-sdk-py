"""Tests for TraverseIn synchronization gate."""

from qm_engine.context.node_execution import NodeExecution
from qm_engine.traversal.traverse_in import TraverseInGate
from qm_engine.types import TraverseIn
from tests.conftest import make_edge, make_graph, make_node


class TestAwaitAll:
    def setup_method(self):
        self.gate = TraverseInGate()

    def test_no_predecessors_always_executes(self):
        """A node with no incoming edges should always execute."""
        node = make_node(name="solo")
        graph = make_graph([node], [], node)
        assert self.gate.should_execute(node.id, graph, {}, TraverseIn.AWAIT_ALL)

    def test_single_predecessor_finished(self):
        """With one finished predecessor, node should execute."""
        pred = make_node(name="pred")
        target = make_node(name="target")
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        executions = {pred.id: _finished_execution(pred.id)}
        assert self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)

    def test_single_predecessor_not_finished(self):
        """With one running predecessor, node should NOT execute."""
        pred = make_node(name="pred")
        target = make_node(name="target")
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        executions = {pred.id: _running_execution(pred.id)}
        assert not self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)

    def test_two_predecessors_both_finished(self):
        """With two finished predecessors, node should execute."""
        p1 = make_node(name="p1")
        p2 = make_node(name="p2")
        target = make_node(name="target")
        graph = make_graph(
            [p1, p2, target],
            [make_edge(p1, target), make_edge(p2, target)],
            p1,
        )

        executions = {
            p1.id: _finished_execution(p1.id),
            p2.id: _finished_execution(p2.id),
        }
        assert self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)

    def test_two_predecessors_one_finished(self):
        """With only one of two predecessors finished, node should NOT execute."""
        p1 = make_node(name="p1")
        p2 = make_node(name="p2")
        target = make_node(name="target")
        graph = make_graph(
            [p1, p2, target],
            [make_edge(p1, target), make_edge(p2, target)],
            p1,
        )

        executions = {
            p1.id: _finished_execution(p1.id),
            p2.id: _running_execution(p2.id),
        }
        assert not self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)

    def test_predecessor_not_tracked(self):
        """If a predecessor has no execution record at all, node should NOT execute."""
        p1 = make_node(name="p1")
        target = make_node(name="target")
        graph = make_graph([p1, target], [make_edge(p1, target)], p1)

        assert not self.gate.should_execute(target.id, graph, {}, TraverseIn.AWAIT_ALL)

    def test_failed_predecessor_counts_as_terminal(self):
        """A failed predecessor should count as terminal (allow gate to open)."""
        pred = make_node(name="pred")
        target = make_node(name="target")
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        executions = {pred.id: _failed_execution(pred.id)}
        assert self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)

    def test_skipped_predecessor_counts_as_terminal(self):
        """A skipped predecessor should count as terminal."""
        pred = make_node(name="pred")
        target = make_node(name="target")
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        execution = NodeExecution(node_id=pred.id)
        execution.skip()
        executions = {pred.id: execution}
        assert self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_ALL)


class TestAwaitFirst:
    def setup_method(self):
        self.gate = TraverseInGate()

    def test_first_predecessor_finished(self):
        """With one of two predecessors finished, node should execute."""
        p1 = make_node(name="p1")
        p2 = make_node(name="p2")
        target = make_node(name="target")
        graph = make_graph(
            [p1, p2, target],
            [make_edge(p1, target), make_edge(p2, target)],
            p1,
        )

        executions = {
            p1.id: _finished_execution(p1.id),
            p2.id: _running_execution(p2.id),
        }
        assert self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_FIRST)

    def test_no_predecessors_finished(self):
        """With no finished predecessors, node should NOT execute."""
        p1 = make_node(name="p1")
        p2 = make_node(name="p2")
        target = make_node(name="target")
        graph = make_graph(
            [p1, p2, target],
            [make_edge(p1, target), make_edge(p2, target)],
            p1,
        )

        executions = {
            p1.id: _running_execution(p1.id),
            p2.id: _running_execution(p2.id),
        }
        assert not self.gate.should_execute(target.id, graph, executions, TraverseIn.AWAIT_FIRST)

    def test_no_predecessors(self):
        """Node with no predecessors should always execute."""
        node = make_node(name="solo")
        graph = make_graph([node], [], node)
        assert self.gate.should_execute(node.id, graph, {}, TraverseIn.AWAIT_FIRST)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _finished_execution(node_id) -> NodeExecution:
    e = NodeExecution(node_id=node_id)
    e.start()
    e.finish(result="done")
    return e


def _running_execution(node_id) -> NodeExecution:
    e = NodeExecution(node_id=node_id)
    e.start()
    return e


def _failed_execution(node_id) -> NodeExecution:
    e = NodeExecution(node_id=node_id)
    e.start()
    e.fail("error")
    return e
