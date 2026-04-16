"""Integration tests for graph validation rules.

Tests all validation rules using the GraphBuilder and manual graph construction,
covering: missing start, orphan nodes, cycles, invalid edges, decision label
requirements, and multi-error scenarios.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from quartermaster_graph.builder import GraphBuilder
from quartermaster_graph.enums import NodeType, TraverseOut
from quartermaster_graph.models import Agent, GraphSpec, GraphEdge, GraphNode
from quartermaster_graph.validation import ValidationError, validate_graph


@pytest.fixture
def agent() -> Agent:
    """A test agent for manual graph construction."""
    return Agent(name="Validation Test Agent")


class TestMissingStartNode:
    """Validation must reject graphs without a Start node.

    Post-v0.2.0 this is only reachable when the caller opts out of the
    auto-start convenience via ``auto_start=False`` and then forgets to
    call ``.start()`` themselves — still a valid error path, just a
    harder one to hit by accident.
    """

    def test_builder_requires_start(self) -> None:
        """GraphBuilder(auto_start=False).build() raises if .start() was never called."""
        builder = GraphBuilder("No Start", auto_start=False)
        builder._nodes.append(GraphNode(type=NodeType.END, name="End"))
        with pytest.raises(ValueError, match="start node"):
            builder.build()

    def test_manual_graph_no_start(self, agent: Agent) -> None:
        """Manual graph with no Start node produces no_start error."""
        inst = GraphNode(type=NodeType.INSTRUCTION, name="Inst")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=inst.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=inst.id,
            nodes=[inst, end],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors if e.severity == "error"}
        assert "no_start" in codes

    def test_start_node_id_points_to_nonexistent_node(self, agent: Agent) -> None:
        """start_node_id referencing a UUID not in the graph produces invalid_start_id."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=uuid4(),  # does not exist
            nodes=[start, end],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors if e.severity == "error"}
        assert "invalid_start_id" in codes

    def test_start_node_id_points_to_non_start_type(self, agent: Agent) -> None:
        """start_node_id pointing to an End node produces start_id_not_start_type."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=end.id,  # wrong type
            nodes=[start, end],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors if e.severity == "error"}
        assert "start_id_not_start_type" in codes


class TestMissingEndNode:
    """Graphs without an End node are allowed since v0.2.0."""

    def test_no_end_node_validates_clean(self, agent: Agent) -> None:
        """Pre-0.2.0 the validator emitted ``no_end``.  Now it's legal —
        the runner falls back to the last finished node's output so
        single-node flows don't need trailing ``.end()`` boilerplate."""
        start = GraphNode(type=NodeType.START, name="Start")
        inst = GraphNode(type=NodeType.INSTRUCTION, name="Process")
        edge = GraphEdge(source_id=start.id, target_id=inst.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, inst],
            edges=[edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors if e.severity == "error"}
        assert "no_end" not in codes


class TestOrphanNodes:
    """Validation must detect nodes not reachable from Start."""

    def test_single_orphan(self, agent: Agent) -> None:
        """One disconnected node is flagged as orphan."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        orphan = GraphNode(type=NodeType.INSTRUCTION, name="Lonely")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end, orphan],
            edges=[edge],
        )
        errors = validate_graph(version)
        orphan_errors = [e for e in errors if e.code == "orphan_node"]
        assert len(orphan_errors) == 1
        assert orphan_errors[0].node_id == orphan.id

    def test_multiple_orphans(self, agent: Agent) -> None:
        """Multiple disconnected nodes each produce an orphan_node error."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        orphan_a = GraphNode(type=NodeType.INSTRUCTION, name="Orphan A")
        orphan_b = GraphNode(type=NodeType.INSTRUCTION, name="Orphan B")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end, orphan_a, orphan_b],
            edges=[edge],
        )
        errors = validate_graph(version)
        orphan_ids = {e.node_id for e in errors if e.code == "orphan_node"}
        assert orphan_a.id in orphan_ids
        assert orphan_b.id in orphan_ids

    def test_comment_nodes_excluded(self, agent: Agent) -> None:
        """Comment nodes are never flagged as orphans."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        comment = GraphNode(type=NodeType.COMMENT, name="Note")
        edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end, comment],
            edges=[edge],
        )
        errors = validate_graph(version)
        orphan_errors = [e for e in errors if e.code == "orphan_node"]
        assert len(orphan_errors) == 0


class TestCycleDetection:
    """Validation must detect cycles in the graph."""

    def test_simple_cycle(self, agent: Agent) -> None:
        """A -> B -> A cycle produces a cycle_detected warning.

        v0.3.0: under Proposal A the default End-node semantics loop
        back to Start implicitly, so user-defined cycles via explicit
        edges are now a WARNING — the runtime has a loop guard
        (``FlowRunner.max_loop_iterations``) and intentional cycles
        are fully supported.
        """
        start = GraphNode(type=NodeType.START, name="Start")
        a = GraphNode(type=NodeType.INSTRUCTION, name="A")
        b = GraphNode(type=NodeType.INSTRUCTION, name="B")
        end = GraphNode(type=NodeType.END, name="End")
        edges = [
            GraphEdge(source_id=start.id, target_id=a.id),
            GraphEdge(source_id=a.id, target_id=b.id),
            GraphEdge(source_id=b.id, target_id=a.id),  # cycle
            GraphEdge(source_id=b.id, target_id=end.id),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, a, b, end],
            edges=edges,
        )
        errors = validate_graph(version)
        warning_codes = {e.code for e in errors if e.severity == "warning"}
        error_codes = {e.code for e in errors if e.severity == "error"}
        assert "cycle_detected" in warning_codes
        assert "cycle_detected" not in error_codes

    def test_self_loop(self, agent: Agent) -> None:
        """A node pointing to itself creates a cycle (warning in v0.3.0)."""
        start = GraphNode(type=NodeType.START, name="Start")
        inst = GraphNode(type=NodeType.INSTRUCTION, name="Self-Loop")
        end = GraphNode(type=NodeType.END, name="End")
        edges = [
            GraphEdge(source_id=start.id, target_id=inst.id),
            GraphEdge(source_id=inst.id, target_id=inst.id),  # self-loop
            GraphEdge(source_id=inst.id, target_id=end.id),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, inst, end],
            edges=edges,
        )
        errors = validate_graph(version)
        warning_codes = {e.code for e in errors if e.severity == "warning"}
        error_codes = {e.code for e in errors if e.severity == "error"}
        assert "cycle_detected" in warning_codes
        assert "cycle_detected" not in error_codes


class TestInvalidEdges:
    """Validation must detect edges referencing nonexistent nodes."""

    def test_edge_source_missing(self, agent: Agent) -> None:
        """Edge with source_id not in the node list produces invalid_edge_source."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        bad_edge = GraphEdge(source_id=uuid4(), target_id=end.id)
        good_edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end],
            edges=[bad_edge, good_edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors}
        assert "invalid_edge_source" in codes

    def test_edge_target_missing(self, agent: Agent) -> None:
        """Edge with target_id not in the node list produces invalid_edge_target."""
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
        codes = {e.code for e in errors}
        assert "invalid_edge_target" in codes

    def test_both_source_and_target_missing(self, agent: Agent) -> None:
        """Edge with both source and target missing produces two errors."""
        start = GraphNode(type=NodeType.START, name="Start")
        end = GraphNode(type=NodeType.END, name="End")
        bad_edge = GraphEdge(source_id=uuid4(), target_id=uuid4())
        good_edge = GraphEdge(source_id=start.id, target_id=end.id)
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, end],
            edges=[bad_edge, good_edge],
        )
        errors = validate_graph(version)
        codes = [e.code for e in errors]
        assert "invalid_edge_source" in codes
        assert "invalid_edge_target" in codes


class TestDecisionEdgeLabels:
    """Decision, If, and Switch nodes require labeled edges."""

    def test_decision_unlabeled_edges(self, agent: Agent) -> None:
        """Decision node with unlabeled outgoing edges produces error."""
        start = GraphNode(type=NodeType.START, name="Start")
        decision = GraphNode(
            type=NodeType.DECISION,
            name="Choose",
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        a = GraphNode(type=NodeType.END, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        edges = [
            GraphEdge(source_id=start.id, target_id=decision.id),
            GraphEdge(source_id=decision.id, target_id=a.id),  # no label
            GraphEdge(source_id=decision.id, target_id=b.id, label="No"),
        ]
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=start.id,
            nodes=[start, decision, a, b],
            edges=edges,
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors}
        assert "decision_unlabeled_edges" in codes

    def test_if_node_wrong_labels_warning(self, agent: Agent) -> None:
        """If node with non-boolean labels produces a warning."""
        start = GraphNode(type=NodeType.START, name="Start")
        if_node = GraphNode(type=NodeType.IF, name="Check")
        a = GraphNode(type=NodeType.END, name="A")
        b = GraphNode(type=NodeType.END, name="B")
        edges = [
            GraphEdge(source_id=start.id, target_id=if_node.id),
            GraphEdge(source_id=if_node.id, target_id=a.id, label="maybe"),
            GraphEdge(source_id=if_node.id, target_id=b.id, label="perhaps"),
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

    def test_switch_unlabeled_edges(self, agent: Agent) -> None:
        """Switch node with unlabeled edges produces error."""
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
        codes = {e.code for e in errors}
        assert "switch_unlabeled_edges" in codes


class TestMultipleValidationErrors:
    """Validation returns all errors at once, not just the first."""

    def test_multiple_errors_reported(self, agent: Agent) -> None:
        """A graph with multiple issues reports all of them.

        Note: ``no_end`` was removed from the validator in v0.2.0 (End
        nodes are optional now), so only ``no_start`` and
        ``invalid_edge_target`` survive in this fixture.
        """
        # No start node, no end node, invalid edge
        inst = GraphNode(type=NodeType.INSTRUCTION, name="Lonely")
        bad_edge = GraphEdge(source_id=inst.id, target_id=uuid4())
        version = GraphSpec(
            agent_id=agent.id,
            start_node_id=inst.id,
            nodes=[inst],
            edges=[bad_edge],
        )
        errors = validate_graph(version)
        codes = {e.code for e in errors}
        assert "no_start" in codes
        assert "invalid_edge_target" in codes


class TestBuilderValidation:
    """The builder's validate=True flag rejects invalid graphs."""

    def test_builder_rejects_invalid_graph(self) -> None:
        """Builder still surfaces genuinely invalid graphs (dangling edge target)."""
        builder = GraphBuilder("Invalid")  # auto-start creates a Start node
        builder.instruction("Process")
        # Dangling edge to a uuid that doesn't resolve — still an error.
        builder._edges.append(GraphEdge(source_id=uuid4(), target_id=uuid4()))
        version = builder.build(validate=False)  # bypass builder-side raise
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) > 0

    def test_builder_skips_validation(self) -> None:
        """Builder with validate=False returns the graph even if invalid."""
        builder = GraphBuilder("Skip Validation")
        builder.start()
        builder.instruction("Process")
        # No .end() but validate=False
        version = builder.build(validate=False)
        assert version is not None
        assert len(version.nodes) == 2
