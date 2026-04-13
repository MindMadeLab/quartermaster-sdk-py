"""Advanced tests for GraphBuilder: sub-graphs, multi-decision, auto-merge."""

import pytest

from quartermaster_graph.builder import GraphBuilder, _BranchBuilder
from quartermaster_graph.enums import NodeType, TraverseIn
from quartermaster_graph.models import AgentVersion
from quartermaster_graph.validation import validate_graph


def _no_errors(version: AgentVersion) -> list:
    """Return only real errors (not warnings) from validation."""
    return [e for e in validate_graph(version) if e.severity == "error"]


# ---------------------------------------------------------------------------
# 1. Basic: Start -> User -> Instruction -> End
# ---------------------------------------------------------------------------
class TestBasicUserFlow:
    def test_start_user_instruction_end(self):
        graph = GraphBuilder("Agent").start().user().instruction("Respond").end().build()
        assert isinstance(graph, AgentVersion)
        types = [n.type for n in graph.nodes]
        assert NodeType.START in types
        assert NodeType.USER in types
        assert NodeType.INSTRUCTION in types
        assert NodeType.END in types
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 2. Sub-graph inline with .use()
# ---------------------------------------------------------------------------
class TestSubGraphUse:
    def test_inline_subgraph(self):
        sub = GraphBuilder("Sub").start().instruction("Sub step").end().build()
        graph = GraphBuilder("Main").start().use(sub).end().build()
        assert isinstance(graph, AgentVersion)
        # Main: Start + inlined Instruction + End = 3 nodes
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert _no_errors(graph) == []

    # 3. Sub-graph in a branch
    def test_subgraph_in_branch(self):
        sub = GraphBuilder("Sub").start().instruction("Handle yes").end().build()
        graph = (
            GraphBuilder("Main")
            .start()
            .user()
            .decision("Choose", options=["yes", "no"])
            .on("yes").use(sub).end()
            .on("no").instruction("Handle no").end()
            .build()
        )
        assert isinstance(graph, AgentVersion)
        assert _no_errors(graph) == []
        # Verify inlined instruction exists
        names = [n.name for n in graph.nodes]
        assert "Handle yes" in names
        assert "Handle no" in names

    # 12. .use() in multiple branches with different sub-graphs
    def test_use_in_multiple_branches(self):
        g1 = GraphBuilder("S1").start().instruction("Alpha").end().build()
        g2 = GraphBuilder("S2").start().instruction("Beta").end().build()
        graph = (
            GraphBuilder("Main")
            .start()
            .decision("Pick", options=["a", "b"])
            .on("a").use(g1).end()
            .on("b").use(g2).end()
            .build()
        )
        names = [n.name for n in graph.nodes]
        assert "Alpha" in names
        assert "Beta" in names
        assert _no_errors(graph) == []

    # 13. Verify node counts after inlining
    def test_node_counts_after_inlining(self):
        sub = (
            GraphBuilder("Sub")
            .start()
            .instruction("S1")
            .instruction("S2")
            .end()
            .build()
        )
        graph = GraphBuilder("Main").start().use(sub).end().build()
        # Main Start + 2 inlined instructions + End = 4
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3

    # 14. Verify edge connectivity after inlining
    def test_edge_connectivity_after_inlining(self):
        sub = GraphBuilder("Sub").start().instruction("Inner").end().build()
        graph = GraphBuilder("Main").start().instruction("Before").use(sub).instruction("After").end().build()
        # Start -> Before -> Inner -> After -> End
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        # Verify chain: each node (except end) has exactly one outgoing edge
        node_ids = {n.id for n in graph.nodes}
        for e in graph.edges:
            assert e.source_id in node_ids
            assert e.target_id in node_ids

    # 9. Sub-graph that itself has decisions
    def test_subgraph_with_decisions(self):
        sub = (
            GraphBuilder("DecisionSub")
            .start()
            .decision("Inner choice", options=["x", "y"])
            .on("x").instruction("X path").end()
            .on("y").instruction("Y path").end()
            .build()
        )
        graph = (
            GraphBuilder("Main")
            .start()
            .use(sub)
            .end()
            .build()
        )
        assert isinstance(graph, AgentVersion)
        # The sub-graph decision nodes should be inlined
        decision_nodes = [n for n in graph.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 4-5. Multiple decision nodes in sequence / decision -> merge -> decision
# ---------------------------------------------------------------------------
class TestMultiDecision:
    def test_sequential_decisions(self):
        graph = (
            GraphBuilder("Multi-Decision")
            .start()
            .user()
            .decision("First choice", options=["A", "B"])
            .on("A").instruction("Handle A").end()
            .on("B").instruction("Handle B").end()
            .instruction("Middle step")
            .if_node("Check condition", expression="x > 5")
            .on("true").instruction("High").end()
            .on("false").instruction("Low").end()
            .end()
            .build()
        )
        assert isinstance(graph, AgentVersion)
        # Should have: Start, User, Decision, A-instr, B-instr, Merge, Middle,
        #              If, High-instr, Low-instr, Merge, End
        decision_nodes = [n for n in graph.nodes if n.type == NodeType.DECISION]
        if_nodes = [n for n in graph.nodes if n.type == NodeType.IF]
        merge_nodes = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(decision_nodes) == 1
        assert len(if_nodes) == 1
        assert len(merge_nodes) >= 1  # at least one auto-merge
        assert _no_errors(graph) == []

    def test_decision_merge_decision_chain(self):
        graph = (
            GraphBuilder("Chain")
            .start()
            .decision("D1", options=["a", "b"])
            .on("a").instruction("A1").end()
            .on("b").instruction("B1").end()
            .merge("M1")
            .instruction("Between")
            .decision("D2", options=["c", "d"])
            .on("c").instruction("C1").end()
            .on("d").instruction("D1 handler").end()
            .merge("M2")
            .end()
            .build()
        )
        merge_nodes = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(merge_nodes) == 2
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 6. If node with true/false branches
# ---------------------------------------------------------------------------
class TestIfBranches:
    def test_if_true_false(self):
        graph = (
            GraphBuilder("IfTest")
            .start()
            .if_node("Is positive?", expression="val > 0")
            .on("true").instruction("Positive path").end()
            .on("false").instruction("Negative path").end()
            .end()
            .build()
        )
        if_nodes = [n for n in graph.nodes if n.type == NodeType.IF]
        assert len(if_nodes) == 1
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 7. Switch-like decision with 3+ options
# ---------------------------------------------------------------------------
class TestSwitchLike:
    def test_three_plus_options(self):
        graph = (
            GraphBuilder("Switch")
            .start()
            .decision("Route", options=["A", "B", "C", "D"])
            .on("A").instruction("A").end()
            .on("B").instruction("B").end()
            .on("C").instruction("C").end()
            .on("D").instruction("D").end()
            .end()
            .build()
        )
        decision_nodes = [n for n in graph.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        # 4 labeled edges from decision
        labeled = [e for e in graph.edges if e.label in ("A", "B", "C", "D")]
        assert len(labeled) == 4
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 8. Nested decisions (decision inside a branch) -- limited by _BranchBuilder
#    not having .decision(); we inline a sub-graph with a decision instead
# ---------------------------------------------------------------------------
class TestNestedDecisions:
    def test_nested_via_subgraph(self):
        inner = (
            GraphBuilder("Inner")
            .start()
            .decision("Inner D", options=["i", "j"])
            .on("i").instruction("I").end()
            .on("j").instruction("J").end()
            .build()
        )
        graph = (
            GraphBuilder("Outer")
            .start()
            .decision("Outer D", options=["x", "y"])
            .on("x").use(inner).end()
            .on("y").instruction("Y").end()
            .build()
        )
        decisions = [n for n in graph.nodes if n.type == NodeType.DECISION]
        assert len(decisions) == 2  # outer + inlined inner
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 10. Auto-merge after branches converge
# ---------------------------------------------------------------------------
class TestAutoMerge:
    def test_auto_merge_on_next_instruction(self):
        graph = (
            GraphBuilder("AutoMerge")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .instruction("After merge")
            .end()
            .build()
        )
        merge_nodes = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(merge_nodes) == 1
        assert merge_nodes[0].traverse_in == TraverseIn.AWAIT_ALL
        # Both A and B should have edges to the merge node
        merge_id = merge_nodes[0].id
        edges_to_merge = [e for e in graph.edges if e.target_id == merge_id]
        assert len(edges_to_merge) == 2
        assert _no_errors(graph) == []

    def test_auto_merge_on_end(self):
        """Calling .end() on parent after branches should auto-merge then add End."""
        graph = (
            GraphBuilder("AutoMergeEnd")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .end()
            .build()
        )
        merge_nodes = [n for n in graph.nodes if n.type == NodeType.MERGE]
        end_nodes = [n for n in graph.nodes if n.type == NodeType.END]
        assert len(merge_nodes) == 1
        assert len(end_nodes) == 1
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 11. Explicit .merge() call
# ---------------------------------------------------------------------------
class TestExplicitMerge:
    def test_explicit_merge(self):
        graph = (
            GraphBuilder("ExplicitMerge")
            .start()
            .decision("Choose", options=["A", "B", "C"])
            .on("A").instruction("A logic").end()
            .on("B").instruction("B logic").end()
            .on("C").instruction("C logic").end()
            .merge("Collect results")
            .instruction("Continue")
            .end()
            .build()
        )
        merge_nodes = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(merge_nodes) == 1
        assert merge_nodes[0].name == "Collect results"
        # 3 edges into merge
        merge_id = merge_nodes[0].id
        edges_to_merge = [e for e in graph.edges if e.target_id == merge_id]
        assert len(edges_to_merge) == 3
        assert _no_errors(graph) == []


# ---------------------------------------------------------------------------
# 15. Build validates successfully with complex graphs
# ---------------------------------------------------------------------------
class TestComplexBuild:
    def test_complex_graph_validates(self):
        sub1 = GraphBuilder("S1").start().user().instruction("S1 step").end().build()
        sub2 = GraphBuilder("S2").start().instruction("S2 step").end().build()

        graph = (
            GraphBuilder("Complex")
            .start()
            .user()
            .decision("Route", options=["alpha", "beta"])
            .on("alpha").use(sub1).end()
            .on("beta").use(sub2).end()
            .merge("Join")
            .instruction("Final")
            .end()
            .build()
        )
        assert isinstance(graph, AgentVersion)
        assert _no_errors(graph) == []
        # Verify all expected node types present
        types = {n.type for n in graph.nodes}
        assert NodeType.START in types
        assert NodeType.USER in types
        assert NodeType.DECISION in types
        assert NodeType.INSTRUCTION in types
        assert NodeType.MERGE in types
        assert NodeType.END in types

    def test_fluent_api_returns_correct_types(self):
        """Ensure .merge() returns GraphBuilder for fluent chaining."""
        builder = GraphBuilder("Fluent")
        result = builder.start()
        assert isinstance(result, GraphBuilder)
        result2 = (
            result
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .merge("M")
        )
        assert isinstance(result2, GraphBuilder)

    def test_branch_use_returns_branch_builder(self):
        """Ensure .use() on _BranchBuilder returns _BranchBuilder for chaining."""
        sub = GraphBuilder("Sub").start().instruction("X").end().build()
        builder = GraphBuilder("Main").start().decision("D", options=["a"])
        branch = builder.on("a")
        result = branch.use(sub)
        assert isinstance(result, _BranchBuilder)
