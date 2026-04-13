"""Advanced tests for GraphBuilder: sub-graphs, multi-decision, auto-merge, new node types."""

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

    def test_use_accepts_graph_builder_directly(self):
        """use() should accept a GraphBuilder without needing .build()."""
        sub = GraphBuilder("Sub").start().instruction("Inlined step").end()
        graph = GraphBuilder("Main").start().use(sub).end()
        assert len(graph.nodes) == 3  # Start + Inlined + End
        assert len(graph.edges) == 2
        names = [n.name for n in graph.nodes]
        assert "Inlined step" in names

    def test_use_graph_builder_in_branch(self):
        """use() on a _BranchBuilder should also accept GraphBuilder."""
        sub = GraphBuilder("Sub").start().instruction("Branch inline").end()
        graph = (
            GraphBuilder("Main")
            .start()
            .decision("Pick", options=["a", "b"])
            .on("a").use(sub).end()
            .on("b").instruction("Other").end()
            .end()
            .build()
        )
        names = [n.name for n in graph.nodes]
        assert "Branch inline" in names


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
        # Decision picks one path — no merge needed.  Branches converge
        # directly on the next node ("Middle step" / End).
        decision_nodes = [n for n in graph.nodes if n.type == NodeType.DECISION]
        if_nodes = [n for n in graph.nodes if n.type == NodeType.IF]
        assert len(decision_nodes) == 1
        assert len(if_nodes) == 1
        # No auto-merge nodes (decision/if only pick one branch)
        merge_nodes = [n for n in graph.nodes if n.type in (NodeType.MERGE, NodeType.STATIC_MERGE)]
        assert len(merge_nodes) == 0
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
class TestDecisionConvergence:
    """Decision picks ONE branch — branches converge directly on the next
    node without any merge node."""

    def test_branches_converge_on_next_instruction(self):
        graph = (
            GraphBuilder("Converge")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .instruction("After decision")
            .end()
            .build()
        )
        # No merge node — branches wire directly to "After decision"
        merge_nodes = [n for n in graph.nodes if n.type in (NodeType.MERGE, NodeType.STATIC_MERGE)]
        assert len(merge_nodes) == 0
        # Both A and B should have edges to "After decision"
        after = [n for n in graph.nodes if n.name == "After decision"][0]
        edges_to_after = [e for e in graph.edges if e.target_id == after.id]
        assert len(edges_to_after) == 2
        assert _no_errors(graph) == []

    def test_branches_converge_on_end(self):
        """Calling .end() on parent after branches wires them to End directly."""
        graph = (
            GraphBuilder("ConvergeEnd")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .end()
            .build()
        )
        merge_nodes = [n for n in graph.nodes if n.type in (NodeType.MERGE, NodeType.STATIC_MERGE)]
        end_nodes = [n for n in graph.nodes if n.type == NodeType.END]
        assert len(merge_nodes) == 0
        assert len(end_nodes) == 1
        # Both branches connect to End
        end_id = end_nodes[0].id
        edges_to_end = [e for e in graph.edges if e.target_id == end_id]
        assert len(edges_to_end) == 2
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


# ===========================================================================
# NEW TESTS -- GraphBuilder as graph, new node types, Graph alias
# ===========================================================================


class TestGraphWithoutBuild:
    """Graph usable without .build() -- access .nodes directly."""

    def test_access_nodes_directly(self):
        graph = GraphBuilder("Direct").start().instruction("Step").end()
        # No .build() call -- access .nodes property directly
        assert len(graph.nodes) == 3
        types = [n.type for n in graph.nodes]
        assert NodeType.START in types
        assert NodeType.INSTRUCTION in types
        assert NodeType.END in types

    def test_access_edges_directly(self):
        graph = GraphBuilder("Direct").start().instruction("A").instruction("B").end()
        assert len(graph.edges) == 3  # Start->A, A->B, B->End

    def test_start_node_id(self):
        graph = GraphBuilder("Direct").start().instruction("X").end()
        sid = graph.start_node_id
        start = graph.get_node(sid)
        assert start is not None
        assert start.type == NodeType.START

    def test_start_node_id_raises_without_start(self):
        graph = GraphBuilder("Empty")
        with pytest.raises(ValueError, match="No start node"):
            _ = graph.start_node_id

    def test_get_node(self):
        graph = GraphBuilder("G").start().instruction("Find me").end()
        instr = [n for n in graph.nodes if n.name == "Find me"][0]
        assert graph.get_node(instr.id) is instr

    def test_get_successors(self):
        graph = GraphBuilder("G").start().instruction("A").instruction("B").end()
        a_node = [n for n in graph.nodes if n.name == "A"][0]
        succs = graph.get_successors(a_node.id)
        assert len(succs) == 1
        assert succs[0].name == "B"

    def test_get_edges_from(self):
        graph = GraphBuilder("G").start().instruction("A").end()
        a_node = [n for n in graph.nodes if n.name == "A"][0]
        edges = graph.get_edges_from(a_node.id)
        assert len(edges) == 1

    def test_finalize_adds_end_for_dangling_branches(self):
        """Accessing .nodes on a graph with dangling branches auto-adds END nodes."""
        graph = (
            GraphBuilder("Dangling")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
        )
        # No explicit .end() or .merge() on the parent -- branches are dangling
        nodes = graph.nodes  # triggers _finalize
        end_nodes = [n for n in nodes if n.type == NodeType.END]
        assert len(end_nodes) == 2  # one per dangling branch

    def test_to_version(self):
        graph = GraphBuilder("V").start().instruction("X").end()
        version = graph.to_version(version="1.0.0")
        assert isinstance(version, AgentVersion)
        assert version.version == "1.0.0"


class TestSwitchNode:
    """switch() with 3+ branches."""

    def test_switch_three_branches(self):
        graph = (
            GraphBuilder("SwitchTest")
            .start()
            .switch("Route", cases=[
                {"expression": "speed == 'fast'", "edge_id": "fast"},
                {"expression": "speed == 'medium'", "edge_id": "medium"},
                {"expression": "speed == 'slow'", "edge_id": "slow"},
            ])
            .on("fast").instruction("Fast path").end()
            .on("medium").instruction("Medium path").end()
            .on("slow").instruction("Slow path").end()
            .end()
            .build()
        )
        switch_nodes = [n for n in graph.nodes if n.type == NodeType.SWITCH]
        assert len(switch_nodes) == 1
        labeled = [e for e in graph.edges if e.label in ("fast", "medium", "slow")]
        assert len(labeled) == 3
        assert _no_errors(graph) == []


class TestReasoningNode:
    def test_reasoning(self):
        graph = (
            GraphBuilder("R")
            .start()
            .reasoning("Think hard", model="o1-preview")
            .end()
            .build()
        )
        r_nodes = [n for n in graph.nodes if n.type == NodeType.REASONING]
        assert len(r_nodes) == 1
        assert r_nodes[0].metadata["llm_model"] == "o1-preview"
        assert _no_errors(graph) == []


class TestSummarizeNode:
    def test_summarize(self):
        graph = (
            GraphBuilder("S")
            .start()
            .summarize("Condense", model="gpt-4o-mini")
            .end()
            .build()
        )
        s_nodes = [n for n in graph.nodes if n.type == NodeType.SUMMARIZE]
        assert len(s_nodes) == 1
        assert s_nodes[0].metadata["llm_model"] == "gpt-4o-mini"
        assert _no_errors(graph) == []


class TestVarAndTextNodes:
    def test_var_node(self):
        graph = (
            GraphBuilder("Var")
            .start()
            .var("Set X", variable="x", expression="42")
            .end()
            .build()
        )
        var_nodes = [n for n in graph.nodes if n.type == NodeType.VAR]
        assert len(var_nodes) == 1
        assert var_nodes[0].metadata["name"] == "x"
        assert var_nodes[0].metadata["expression"] == "42"
        assert _no_errors(graph) == []

    def test_text_node(self):
        graph = (
            GraphBuilder("Txt")
            .start()
            .text("Template", template="Hello {{name}}")
            .end()
            .build()
        )
        txt_nodes = [n for n in graph.nodes if n.type == NodeType.TEXT]
        assert len(txt_nodes) == 1
        assert txt_nodes[0].metadata["text"] == "Hello {{name}}"
        assert _no_errors(graph) == []


class TestMemoryNodes:
    def test_read_memory(self):
        graph = (
            GraphBuilder("RM")
            .start()
            .read_memory("Load context", memory_name="user_prefs")
            .end()
            .build()
        )
        rm_nodes = [n for n in graph.nodes if n.type == NodeType.READ_MEMORY]
        assert len(rm_nodes) == 1
        assert rm_nodes[0].metadata["memory_name"] == "user_prefs"
        assert _no_errors(graph) == []

    def test_write_memory(self):
        graph = (
            GraphBuilder("WM")
            .start()
            .write_memory("Save result", memory_name="output", variables=[{"name": "result", "expression": "done"}])
            .end()
            .build()
        )
        wm_nodes = [n for n in graph.nodes if n.type == NodeType.WRITE_MEMORY]
        assert len(wm_nodes) == 1
        assert wm_nodes[0].metadata["memory_name"] == "output"
        assert _no_errors(graph) == []

    def test_flow_memory(self):
        graph = GraphBuilder("FM").start().flow_memory().end()
        fm_nodes = [n for n in graph.nodes if n.type == NodeType.FLOW_MEMORY]
        assert len(fm_nodes) == 1


class TestUserDecisionNode:
    def test_user_decision(self):
        graph = (
            GraphBuilder("UD")
            .start()
            .user_decision("Pick one")
            .on("opt1").instruction("Handle opt1").end()
            .on("opt2").instruction("Handle opt2").end()
            .end()
            .build()
        )
        ud_nodes = [n for n in graph.nodes if n.type == NodeType.USER_DECISION]
        assert len(ud_nodes) == 1
        assert _no_errors(graph) == []


class TestGraphAlias:
    def test_graph_alias_import(self):
        from quartermaster_graph import Graph
        assert Graph is GraphBuilder

    def test_graph_alias_works(self):
        from quartermaster_graph import Graph
        g = Graph("Aliased").start().instruction("Step").end()
        assert len(g.nodes) == 3


class TestMultipleMergePoints:
    def test_two_merge_points(self):
        graph = (
            GraphBuilder("MultiMerge")
            .start()
            .decision("D1", options=["a", "b"])
            .on("a").instruction("A1").end()
            .on("b").instruction("B1").end()
            .merge("M1")
            .instruction("Mid")
            .decision("D2", options=["c", "d"])
            .on("c").instruction("C1").end()
            .on("d").instruction("D1x").end()
            .merge("M2")
            .end()
            .build()
        )
        merges = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(merges) == 2
        assert merges[0].name == "M1"
        assert merges[1].name == "M2"
        assert _no_errors(graph) == []


class TestComplexDecisionChain:
    def test_decision_merge_if_merge_decision(self):
        """decision -> merge -> if -> merge -> decision chain."""
        graph = (
            GraphBuilder("Chain")
            .start()
            # First decision
            .decision("D1", options=["x", "y"])
            .on("x").instruction("X handler").end()
            .on("y").instruction("Y handler").end()
            .merge("M1")
            .instruction("Between 1-2")
            # If node
            .if_node("Check", expression="val > 0")
            .on("true").instruction("Positive").end()
            .on("false").instruction("Negative").end()
            .merge("M2")
            .instruction("Between 2-3")
            # Second decision
            .decision("D2", options=["a", "b", "c"])
            .on("a").instruction("A").end()
            .on("b").instruction("B").end()
            .on("c").instruction("C").end()
            .merge("M3")
            .end()
            .build()
        )
        decisions = [n for n in graph.nodes if n.type == NodeType.DECISION]
        ifs = [n for n in graph.nodes if n.type == NodeType.IF]
        merges = [n for n in graph.nodes if n.type == NodeType.MERGE]
        assert len(decisions) == 2
        assert len(ifs) == 1
        assert len(merges) == 3
        assert _no_errors(graph) == []


class TestBranchBuilderNewNodes:
    """Verify that _BranchBuilder has all the new node methods."""

    def test_reasoning_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a", "b"])
            .on("a").reasoning("Think", model="o1").end()
            .on("b").instruction("Skip").end()
            .end()
            .build()
        )
        r_nodes = [n for n in graph.nodes if n.type == NodeType.REASONING]
        assert len(r_nodes) == 1

    def test_var_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").var("Set", variable="x", expression="1").end()
            .end()
            .build()
        )
        var_nodes = [n for n in graph.nodes if n.type == NodeType.VAR]
        assert len(var_nodes) == 1

    def test_read_write_memory_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").read_memory("Load", memory_name="k").write_memory("Save", memory_name="k", variables=[{"name": "k", "expression": "v"}]).end()
            .end()
            .build()
        )
        rm = [n for n in graph.nodes if n.type == NodeType.READ_MEMORY]
        wm = [n for n in graph.nodes if n.type == NodeType.WRITE_MEMORY]
        assert len(rm) == 1
        assert len(wm) == 1

    def test_user_decision_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").user_decision("Pick").end()
            .end()
            .build()
        )
        ud = [n for n in graph.nodes if n.type == NodeType.USER_DECISION]
        assert len(ud) == 1

    def test_summarize_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").summarize("Sum up").end()
            .end()
            .build()
        )
        s_nodes = [n for n in graph.nodes if n.type == NodeType.SUMMARIZE]
        assert len(s_nodes) == 1

    def test_text_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").text("T", template="hello").end()
            .end()
            .build()
        )
        t_nodes = [n for n in graph.nodes if n.type == NodeType.TEXT]
        assert len(t_nodes) == 1

    def test_flow_memory_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").flow_memory().end()
            .end()
            .build()
        )
        fm = [n for n in graph.nodes if n.type == NodeType.FLOW_MEMORY]
        assert len(fm) == 1

    def test_switch_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").switch("Inner SW", cases=["x", "y"]).end()
            .end()
            .build()
        )
        sw = [n for n in graph.nodes if n.type == NodeType.SWITCH]
        assert len(sw) == 1

    def test_break_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").break_node("Stop").end()
            .end()
            .build()
        )
        brk = [n for n in graph.nodes if n.type == NodeType.BREAK]
        assert len(brk) == 1

    def test_user_form_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").user_form("Form", parameters=[{"name": "email"}]).end()
            .end()
            .build()
        )
        uf = [n for n in graph.nodes if n.type == NodeType.USER_FORM]
        assert len(uf) == 1

    def test_program_runner_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").program_runner("Run", program="ls").end()
            .end()
            .build()
        )
        pr = [n for n in graph.nodes if n.type == NodeType.PROGRAM_RUNNER]
        assert len(pr) == 1

    def test_agent_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").agent("Auto", model="gpt-4o").end()
            .end()
            .build()
        )
        a_nodes = [n for n in graph.nodes if n.type == NodeType.AGENT]
        assert len(a_nodes) == 1

    def test_vision_on_branch(self):
        graph = (
            GraphBuilder("B")
            .start()
            .decision("D", options=["a"])
            .on("a").vision("See", model="gpt-4o").end()
            .end()
            .build()
        )
        v_nodes = [n for n in graph.nodes if n.type == NodeType.INSTRUCTION_IMAGE_VISION]
        assert len(v_nodes) == 1
