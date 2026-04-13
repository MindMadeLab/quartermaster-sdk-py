"""Tests for nested control flow inside _BranchBuilder.

Covers: if_node, decision, parallel, merge inside branches;
deep nesting; edge labels; auto-merge in nested contexts.
"""

import pytest

from quartermaster_graph import Graph
from quartermaster_graph.builder import GraphBuilder, _BranchBuilder
from quartermaster_graph.enums import NodeType, TraverseIn
from quartermaster_graph.models import AgentVersion
from quartermaster_graph.validation import validate_graph


def _no_errors(version: AgentVersion) -> list:
    """Return only real errors (not warnings) from validation."""
    return [e for e in validate_graph(version) if e.severity == "error"]


def _edges_from(graph, node_id):
    """Return edges originating from a given node."""
    return [e for e in graph.edges if e.source_id == node_id]


def _edges_to(graph, node_id):
    """Return edges targeting a given node."""
    return [e for e in graph.edges if e.target_id == node_id]


def _node_by_name(graph, name):
    """Find a single node by name."""
    matches = [n for n in graph.nodes if n.name == name]
    assert len(matches) == 1, f"Expected 1 node named '{name}', found {len(matches)}"
    return matches[0]


def _nodes_by_type(graph, node_type):
    """Return all nodes of a given type."""
    return [n for n in graph.nodes if n.type == node_type]


# ---------------------------------------------------------------------------
# 1. TestIfInsideBranch
# ---------------------------------------------------------------------------
class TestIfInsideBranch:
    def test_if_inside_parallel_branch(self):
        """Parallel with one branch containing an if_node with true/false sub-branches."""
        graph = (
            Graph("IfInParallel")
            .start()
            .parallel()
            .branch().instruction("Path A").end()
            .branch()
                .if_node("Quality?", expression="quality > 0.8")
                .on("true").instruction("Good").end()
                .on("false").instruction("Bad").end()
                .merge("IF merge")
            .end()
            .merge("M1")
            .end()
            .build()
        )
        # Nodes: Start, Path A, IF(Quality?), Good, Bad, IF merge, M1, End = 8
        assert len(graph.nodes) == 8
        # Edges: Start->PathA, Start->IF, IF->Good(true), IF->Bad(false),
        #        Good->IFmerge, Bad->IFmerge, PathA->M1, IFmerge->M1, M1->End = 9
        assert len(graph.edges) == 9

        # Verify true/false labels
        if_node = _node_by_name(graph, "Quality?")
        if_edges = _edges_from(graph, if_node.id)
        labels = sorted(e.label for e in if_edges)
        assert labels == ["false", "true"]

        assert _no_errors(graph) == []

    def test_if_with_merge_inside_branch(self):
        """If_node with explicit merge inside a decision branch."""
        graph = (
            Graph("IfMergeInBranch")
            .start()
            .decision("Route", options=["x", "y"])
            .on("x")
                .if_node("Check", expression="a > 1")
                .on("true").instruction("Yes").end()
                .on("false").instruction("No").end()
                .merge("Inner merge")
                .instruction("After IF")
            .end()
            .on("y").instruction("Y path").end()
            .merge("Outer merge")
            .end()
            .build()
        )
        # Verify merge nodes
        merges = _nodes_by_type(graph, NodeType.MERGE)
        merge_names = sorted(m.name for m in merges)
        assert "Inner merge" in merge_names
        assert "Outer merge" in merge_names

        # Verify IF node edges have labels
        if_node = _node_by_name(graph, "Check")
        if_edges = _edges_from(graph, if_node.id)
        labels = sorted(e.label for e in if_edges)
        assert labels == ["false", "true"]

        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 2. TestDecisionInsideBranch
# ---------------------------------------------------------------------------
class TestDecisionInsideBranch:
    def test_decision_inside_parallel_branch(self):
        """Parallel with one branch containing a decision node."""
        graph = (
            Graph("DecisionInParallel")
            .start()
            .parallel()
            .branch().instruction("Simple").end()
            .branch()
                .decision("Inner D", options=["a", "b"])
                .on("a").instruction("Handle A").end()
                .on("b").instruction("Handle B").end()
                .merge("D merge")
            .end()
            .merge("P merge")
            .end()
            .build()
        )
        decisions = _nodes_by_type(graph, NodeType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].name == "Inner D"

        # Labeled edges from decision
        d_edges = _edges_from(graph, decisions[0].id)
        labels = sorted(e.label for e in d_edges)
        assert labels == ["a", "b"]

        # Both merges exist
        merges = _nodes_by_type(graph, NodeType.MERGE)
        merge_names = sorted(m.name for m in merges)
        assert "D merge" in merge_names
        assert "P merge" in merge_names

        assert _no_errors(graph) == []

    def test_decision_three_options_inside_branch(self):
        """3-way decision nested in a parallel branch."""
        graph = (
            Graph("ThreeWay")
            .start()
            .parallel()
            .branch().instruction("Fast track").end()
            .branch()
                .decision("Triage", options=["low", "medium", "high"])
                .on("low").instruction("Low handler").end()
                .on("medium").instruction("Medium handler").end()
                .on("high").instruction("High handler").end()
                .merge("Triage merge")
            .end()
            .merge("Fan-in")
            .end()
            .build()
        )
        decisions = _nodes_by_type(graph, NodeType.DECISION)
        assert len(decisions) == 1

        d_edges = _edges_from(graph, decisions[0].id)
        labels = sorted(e.label for e in d_edges)
        assert labels == ["high", "low", "medium"]

        # Verify triage merge receives 3 edges
        triage_merge = _node_by_name(graph, "Triage merge")
        edges_in = _edges_to(graph, triage_merge.id)
        assert len(edges_in) == 3

        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 3. TestParallelInsideBranch
# ---------------------------------------------------------------------------
class TestParallelInsideBranch:
    def test_nested_parallel(self):
        """Parallel inside a parallel branch -- double fan-out/fan-in."""
        graph = (
            Graph("NestedParallel")
            .start()
            .parallel()
            .branch().instruction("Outer A").end()
            .branch()
                .parallel()
                .branch().instruction("Inner X").end()
                .branch().instruction("Inner Y").end()
                .merge("Inner merge")
            .end()
            .merge("Outer merge")
            .end()
            .build()
        )
        merges = _nodes_by_type(graph, NodeType.MERGE)
        merge_names = sorted(m.name for m in merges)
        assert "Inner merge" in merge_names
        assert "Outer merge" in merge_names

        # Inner merge should have 2 incoming edges
        inner_merge = _node_by_name(graph, "Inner merge")
        edges_in = _edges_to(graph, inner_merge.id)
        assert len(edges_in) == 2

        # Outer merge should have 2 incoming edges (Outer A + Inner merge)
        outer_merge = _node_by_name(graph, "Outer merge")
        edges_in_outer = _edges_to(graph, outer_merge.id)
        assert len(edges_in_outer) == 2

        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 4. TestWhiteboardPattern
# ---------------------------------------------------------------------------
class TestWhiteboardPattern:
    def test_whiteboard_diagram(self):
        """Full whiteboard pattern:
        S -> U -> parallel(3 paths) -> M1 -> parallel(2 paths, one with IF) -> M2 -> E
        """
        graph = (
            Graph("Whiteboard")
            .start().user("U")
            .parallel()
            .branch().text("T1").instruction("I1").end()
            .branch().end()
            .branch().instruction("I2").end()
            .merge("M1")
            .parallel()
            .branch().text("T2").instruction("I3").end()
            .branch()
                .if_node("Check?", expression="quality > 0.8")
                .on("true").text("Yes").end()
                .on("false").text("No").node(NodeType.BLANK, "B").end()
                .merge("IF merge")
            .end()
            .merge("M2")
            .end()
            .build()
        )
        # Count nodes:
        # Start, U, T1, I1, I2, M1, T2, I3, Check?, Yes, No, B, IF merge, M2, End = 15
        assert len(graph.nodes) == 15

        # Count edges:
        # Start->U (1)
        # U->T1, U->I2 (2) -- fan-out from U
        # T1->I1 (1)
        # I1->M1, U->M1, I2->M1 (3) -- fan-in to M1 (U->M1 is the empty branch)
        # M1->T2, M1->Check? (2) -- fan-out from M1
        # T2->I3 (1)
        # Check?->Yes(true), Check?->No(false) (2)
        # No->B (1)
        # Yes->IFmerge, B->IFmerge (2)
        # I3->M2, IFmerge->M2 (2)
        # M2->End (1)
        # Total = 1+2+1+3+2+1+2+1+2+2+1 = 18
        assert len(graph.edges) == 18

        # Verify IF edges have labels
        check_node = _node_by_name(graph, "Check?")
        check_edges = _edges_from(graph, check_node.id)
        labels = sorted(e.label for e in check_edges)
        assert labels == ["false", "true"]

        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 5. TestDeepNesting
# ---------------------------------------------------------------------------
class TestDeepNesting:
    def test_three_levels_deep(self):
        """parallel > branch > parallel > branch > instruction."""
        graph = (
            Graph("DeepNest")
            .start()
            .parallel()
            .branch()
                .parallel()
                .branch()
                    .parallel()
                    .branch().instruction("Level 3").end()
                    .branch().instruction("Level 3b").end()
                    .merge("L3 merge")
                .end()
                .branch().instruction("Level 2b").end()
                .merge("L2 merge")
            .end()
            .branch().instruction("Level 1b").end()
            .merge("L1 merge")
            .end()
            .build()
        )
        merges = _nodes_by_type(graph, NodeType.MERGE)
        assert len(merges) == 3
        merge_names = sorted(m.name for m in merges)
        assert merge_names == ["L1 merge", "L2 merge", "L3 merge"]

        # L3 merge gets 2 inputs
        l3_merge = _node_by_name(graph, "L3 merge")
        assert len(_edges_to(graph, l3_merge.id)) == 2

        assert _no_errors(graph) == []

    def test_decision_in_decision_branch(self):
        """Decision inside an on() branch of another decision."""
        graph = (
            Graph("DecInDec")
            .start()
            .decision("Outer", options=["p", "q"])
            .on("p")
                .decision("Inner", options=["r", "s"])
                .on("r").instruction("R handler").end()
                .on("s").instruction("S handler").end()
                .merge("Inner merge")
            .end()
            .on("q").instruction("Q handler").end()
            .merge("Outer merge")
            .end()
            .build()
        )
        decisions = _nodes_by_type(graph, NodeType.DECISION)
        assert len(decisions) == 2
        decision_names = sorted(d.name for d in decisions)
        assert decision_names == ["Inner", "Outer"]

        # Outer decision should have edges labeled p and q
        outer = _node_by_name(graph, "Outer")
        outer_edges = _edges_from(graph, outer.id)
        outer_labels = sorted(e.label for e in outer_edges)
        assert outer_labels == ["p", "q"]

        # Inner decision should have edges labeled r and s
        inner = _node_by_name(graph, "Inner")
        inner_edges = _edges_from(graph, inner.id)
        inner_labels = sorted(e.label for e in inner_edges)
        assert inner_labels == ["r", "s"]

        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 6. TestEdgeLabelsInNestedBranches
# ---------------------------------------------------------------------------
class TestEdgeLabelsInNestedBranches:
    def test_labels_correct_on_nested_if(self):
        """Edge labels (true/false) are set correctly on nested IF branches."""
        graph = (
            Graph("Labels")
            .start()
            .parallel()
            .branch()
                .if_node("Cond", expression="x > 0")
                .on("true").instruction("T").end()
                .on("false").instruction("F").end()
                .merge("IF merge")
            .end()
            .branch().instruction("Other").end()
            .merge("P merge")
            .end()
            .build()
        )
        cond = _node_by_name(graph, "Cond")
        cond_edges = _edges_from(graph, cond.id)
        labels = sorted(e.label for e in cond_edges)
        assert labels == ["false", "true"]

    def test_parallel_branches_have_no_labels(self):
        """Edges from a parallel fan-out node should NOT have labels."""
        graph = (
            Graph("NoLabels")
            .start()
            .parallel()
            .branch().instruction("A").end()
            .branch().instruction("B").end()
            .merge("M")
            .end()
            .build()
        )
        # Find start node (which is the fan-out node)
        start = _nodes_by_type(graph, NodeType.START)[0]
        start_edges = _edges_from(graph, start.id)
        for e in start_edges:
            assert e.label is None or e.label == "", (
                f"Parallel branch edge should have no label, got '{e.label}'"
            )

    def test_decision_labels_nested_in_parallel(self):
        """Decision option labels are preserved when nested inside parallel."""
        graph = (
            Graph("DecLabels")
            .start()
            .parallel()
            .branch()
                .decision("Pick", options=["alpha", "beta"])
                .on("alpha").instruction("Alpha handler").end()
                .on("beta").instruction("Beta handler").end()
                .merge("D merge")
            .end()
            .branch().instruction("Side").end()
            .merge("P merge")
            .end()
            .build()
        )
        pick = _node_by_name(graph, "Pick")
        pick_edges = _edges_from(graph, pick.id)
        labels = sorted(e.label for e in pick_edges)
        assert labels == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# 7. TestAutoMergeInNestedBranches
# ---------------------------------------------------------------------------
class TestConvergenceInNestedBranches:
    """After decision/if only ONE branch fires — no merge needed.
    Branches converge directly on the next node."""

    def test_if_branches_converge_in_parallel_branch(self):
        """IF branches converge directly on 'After IF' inside a parallel branch."""
        graph = (
            Graph("ConvergeNested")
            .start()
            .parallel()
            .branch()
                .if_node("Test?", expression="flag")
                .on("true").instruction("T path").end()
                .on("false").instruction("F path").end()
                .instruction("After IF")
            .end()
            .branch().instruction("Other").end()
            .merge("P merge")
            .end()
            .build()
        )
        # Only the explicit parallel merge — no auto-merge for IF
        merges = [n for n in graph.nodes if n.type in (NodeType.MERGE, NodeType.STATIC_MERGE)]
        merge_names = [m.name for m in merges]
        assert "P merge" in merge_names
        assert len(merges) == 1

        # "After IF" should be reachable from BOTH IF branches directly
        after_if = _node_by_name(graph, "After IF")
        edges_in = _edges_to(graph, after_if.id)
        assert len(edges_in) == 2  # T path + F path converge here
        source_names = sorted(
            _node_by_name(graph, "").name  # get names via id lookup
            if False else
            [n for n in graph.nodes if n.id == e.source_id][0].name
            for e in edges_in
        )
        assert "F path" in source_names
        assert "T path" in source_names

        assert _no_errors(graph) == []

    def test_decision_branches_converge_in_parallel_branch(self):
        """Decision branches converge directly on 'Continue' inside a parallel branch."""
        graph = (
            Graph("ConvergeDec")
            .start()
            .parallel()
            .branch()
                .decision("Pick", options=["a", "b"])
                .on("a").instruction("A").end()
                .on("b").instruction("B").end()
                .instruction("Continue")
            .end()
            .branch().instruction("Side").end()
            .merge("P merge")
            .end()
            .build()
        )
        # Only the explicit parallel merge
        merges = [n for n in graph.nodes if n.type in (NodeType.MERGE, NodeType.STATIC_MERGE)]
        assert len(merges) == 1

        # Both decision branches converge directly on "Continue"
        cont = _node_by_name(graph, "Continue")
        edges_in = _edges_to(graph, cont.id)
        assert len(edges_in) == 2  # A + B converge here

        assert _no_errors(graph) == []
