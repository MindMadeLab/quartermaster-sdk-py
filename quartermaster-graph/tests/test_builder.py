"""Tests for the fluent GraphBuilder API."""

import pytest

from quartermaster_graph.builder import GraphBuilder
from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import GraphSpec
from quartermaster_graph.validation import validate_graph


class TestBasicBuilder:
    def test_simple_chain(self):
        version = (
            GraphBuilder("Test")
            .start()
            .instruction("Process")
            .end()
            .build()
        )
        assert isinstance(version, GraphSpec)
        assert len(version.nodes) == 3
        assert len(version.edges) == 2

    def test_multi_step(self):
        version = (
            GraphBuilder("Multi")
            .start()
            .instruction("Step 1")
            .instruction("Step 2")
            .instruction("Step 3")
            .end()
            .build()
        )
        assert len(version.nodes) == 5
        assert len(version.edges) == 4

    def test_no_start_raises(self):
        with pytest.raises(ValueError, match="start node"):
            GraphBuilder("Test").instruction("X").end().build()

    def test_validation_on_build(self):
        # No end node should produce validation warnings/errors (but build still returns)
        version = GraphBuilder("Test").start().instruction("X").build()
        from quartermaster_graph.validation import validate_graph
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) > 0

    def test_skip_validation(self):
        version = (
            GraphBuilder("Test")
            .start()
            .instruction("X")
            .build(validate=False)
        )
        assert len(version.nodes) == 2

    def test_to_graph(self):
        graph = (
            GraphBuilder("Test")
            .start()
            .instruction("X")
            .end()
            .to_graph()
        )
        assert isinstance(graph, GraphSpec)


class TestDecisionBuilder:
    def test_simple_decision(self):
        version = (
            GraphBuilder("Decision Test")
            .start()
            .decision("Choose?", options=["Yes", "No"])
            .on("Yes").instruction("Yes handler").end()
            .on("No").instruction("No handler").end()
            .build()
        )
        assert len(version.nodes) >= 5  # start, decision, 2 instructions, 2 ends
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_three_way_decision(self):
        version = (
            GraphBuilder("Three Way")
            .start()
            .decision("Route?", options=["A", "B", "C"])
            .on("A").instruction("Route A").end()
            .on("B").instruction("Route B").end()
            .on("C").instruction("Route C").end()
            .build()
        )
        assert len(version.nodes) >= 8


class TestIfBuilder:
    def test_if_node(self):
        version = (
            GraphBuilder("If Test")
            .start()
            .if_node("Check", expression="x > 0")
            .on("true").instruction("Positive").end()
            .on("false").instruction("Negative").end()
            .build()
        )
        if_nodes = [n for n in version.nodes if n.type == NodeType.IF]
        assert len(if_nodes) == 1


class TestNodeTypes:
    def test_static_node(self):
        version = (
            GraphBuilder("Static")
            .start()
            .static("Content", text="Hello World")
            .end()
            .build()
        )
        static_nodes = [n for n in version.nodes if n.type == NodeType.STATIC]
        assert len(static_nodes) == 1
        assert static_nodes[0].metadata["static_text"] == "Hello World"

    def test_code_node(self):
        version = (
            GraphBuilder("Code")
            .start()
            .code("Script", code="print('hi')", filename="script.py")
            .end()
            .build()
        )
        code_nodes = [n for n in version.nodes if n.type == NodeType.CODE]
        assert len(code_nodes) == 1
        assert code_nodes[0].metadata["code"] == "print('hi')"

    def test_user_node(self):
        version = (
            GraphBuilder("User")
            .start()
            .user("Get Input")
            .end()
            .build()
        )
        user_nodes = [n for n in version.nodes if n.type == NodeType.USER]
        assert len(user_nodes) == 1

    def test_sub_agent_node(self):
        version = (
            GraphBuilder("SubAgent")
            .start()
            .sub_agent("Delegate", graph_id="agent-123")
            .end()
            .build()
        )
        sub_nodes = [n for n in version.nodes if n.type == NodeType.SUB_ASSISTANT]
        assert len(sub_nodes) == 1

class TestBuilderValidation:
    def test_valid_graph_no_errors(self):
        version = (
            GraphBuilder("Valid")
            .start()
            .instruction("Process")
            .end()
            .build()
        )
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_built_graph_passes_full_validation(self):
        version = (
            GraphBuilder("Full Test")
            .start()
            .instruction("Step 1", model="gpt-4o", temperature=0.5)
            .code("Execute", code="result = process()", filename="execute.py")
            .instruction("Step 2", model="claude-3")
            .end()
            .build()
        )
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0
