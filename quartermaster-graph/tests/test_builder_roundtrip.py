"""Integration tests: Build graph with GraphBuilder -> validate -> serialize -> deserialize -> compare."""

from __future__ import annotations

import json

from quartermaster_graph.builder import GraphBuilder
from quartermaster_graph.enums import NodeType, TraverseOut
from quartermaster_graph.serialization import from_json, to_json
from quartermaster_graph.validation import validate_graph


class TestSimpleLinearRoundtrip:
    """Round-trip tests for simple linear graphs."""

    def test_start_instruction_end(self) -> None:
        """Build Start -> Instruction -> End, serialize to JSON, deserialize, compare."""
        version = (
            GraphBuilder("Simple Agent")
            .start()
            .instruction("Analyze", model="gpt-4o", temperature=0.5)
            .end()
            .build()
        )

        errors = validate_graph(version)
        assert not [e for e in errors if e.severity == "error"]

        data = to_json(version)
        restored = from_json(data)

        assert restored.version == version.version
        assert restored.agent_id == version.agent_id
        assert restored.start_node_id == version.start_node_id
        assert len(restored.nodes) == len(version.nodes)
        assert len(restored.edges) == len(version.edges)

        for orig, rest in zip(version.nodes, restored.nodes):
            assert orig.id == rest.id
            assert orig.type == rest.type
            assert orig.name == rest.name
            assert orig.metadata == rest.metadata

    def test_multi_step_pipeline(self) -> None:
        """Build a multi-step pipeline and verify round-trip preserves all nodes."""
        version = (
            GraphBuilder("Pipeline", description="Multi-step pipeline")
            .start()
            .instruction("Step 1", model="claude-3-opus")
            .code("Transform", code="x = x.upper()", language="python")
            .instruction("Step 2", model="gpt-4o")
            .static("Report", content="Done processing.")
            .end()
            .build()
        )

        assert len(version.nodes) == 6  # start + 4 steps + end
        assert len(version.edges) == 5

        data = to_json(version)
        json_str = json.dumps(data)
        data_back = json.loads(json_str)
        restored = from_json(data_back)

        assert len(restored.nodes) == 6
        assert len(restored.edges) == 5

        node_types = [n.type for n in restored.nodes]
        assert node_types[0] == NodeType.START
        assert node_types[-1] == NodeType.END
        assert NodeType.CODE in node_types
        assert NodeType.STATIC in node_types


class TestDecisionBranchRoundtrip:
    """Round-trip tests for graphs with decision branches."""

    def test_two_branch_decision(self) -> None:
        """Build a decision with Yes/No branches and verify round-trip."""
        version = (
            GraphBuilder("Classifier")
            .start()
            .instruction("Classify input")
            .decision("Is positive?", options=["Yes", "No"])
            .on("Yes").instruction("Positive response").end()
            .on("No").instruction("Negative response").end()
            .build()
        )

        errors = validate_graph(version)
        assert not [e for e in errors if e.severity == "error"]

        data = to_json(version)
        restored = from_json(data)

        assert len(restored.nodes) == len(version.nodes)
        assert len(restored.edges) == len(version.edges)

        # Verify the decision node is preserved
        decision_nodes = [n for n in restored.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        assert decision_nodes[0].traverse_out == TraverseOut.SPAWN_PICKED

        # Verify labeled edges survive round-trip
        decision_id = decision_nodes[0].id
        decision_edges = [e for e in restored.edges if e.source_id == decision_id]
        labels = {e.label for e in decision_edges}
        assert "Yes" in labels
        assert "No" in labels

    def test_three_branch_decision(self) -> None:
        """Build a decision with three branches and verify all labels preserved."""
        version = (
            GraphBuilder("Router")
            .start()
            .decision("Route?", options=["A", "B", "C"])
            .on("A").instruction("Path A").end()
            .on("B").instruction("Path B").end()
            .on("C").instruction("Path C").end()
            .build()
        )

        data = to_json(version)
        restored = from_json(data)

        decision_nodes = [n for n in restored.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        decision_edges = [e for e in restored.edges if e.source_id == decision_nodes[0].id]
        labels = {e.label for e in decision_edges}
        assert labels == {"A", "B", "C"}

        end_nodes = [n for n in restored.nodes if n.type == NodeType.END]
        assert len(end_nodes) == 3


class TestToolAndSubAgentRoundtrip:
    """Round-trip tests for tool and sub-agent nodes."""

    def test_tool_node_metadata(self) -> None:
        """Tool node metadata survives JSON round-trip."""
        version = (
            GraphBuilder("Tool Agent")
            .start()
            .tool("Search", tool_name="web_search")
            .instruction("Summarize results")
            .end()
            .build()
        )

        data = to_json(version)
        restored = from_json(data)

        tool_nodes = [n for n in restored.nodes if n.type == NodeType.TOOL]
        assert len(tool_nodes) == 1
        assert tool_nodes[0].metadata["tool_name"] == "web_search"

    def test_sub_agent_metadata(self) -> None:
        """Sub-agent node metadata survives JSON round-trip."""
        version = (
            GraphBuilder("Orchestrator")
            .start()
            .sub_agent("Worker", agent_id="agent-123")
            .end()
            .build()
        )

        data = to_json(version)
        restored = from_json(data)

        sub_nodes = [n for n in restored.nodes if n.type == NodeType.SUB_AGENT]
        assert len(sub_nodes) == 1
        assert sub_nodes[0].metadata["agent_id"] == "agent-123"


class TestEdgeAttributeRoundtrip:
    """Verify edge attributes survive serialization."""

    def test_edge_is_main_flag(self) -> None:
        """The is_main flag on edges is preserved."""
        version = (
            GraphBuilder("Agent")
            .start()
            .instruction("Process")
            .end()
            .build()
        )

        for edge in version.edges:
            assert edge.is_main is True

        data = to_json(version)
        restored = from_json(data)

        for edge in restored.edges:
            assert edge.is_main is True

    def test_node_positions_preserved(self) -> None:
        """Node positions assigned by the builder survive round-trip."""
        version = (
            GraphBuilder("Agent")
            .start()
            .instruction("Do something")
            .end()
            .build()
        )

        # Builder assigns positions
        for node in version.nodes:
            assert node.position is not None

        data = to_json(version)
        restored = from_json(data)

        for orig, rest in zip(version.nodes, restored.nodes):
            assert orig.position is not None
            assert rest.position is not None
            assert orig.position.x == rest.position.x
            assert orig.position.y == rest.position.y


class TestBuildValidateSerializeCycle:
    """End-to-end: build -> validate -> serialize -> deserialize -> re-validate."""

    def test_full_cycle_simple(self) -> None:
        """A simple graph passes validation both before and after round-trip."""
        version = (
            GraphBuilder("E2E Agent")
            .start()
            .instruction("Analyze")
            .end()
            .build()
        )

        errors_before = validate_graph(version)
        assert not [e for e in errors_before if e.severity == "error"]

        data = to_json(version)
        restored = from_json(data)

        errors_after = validate_graph(restored)
        assert not [e for e in errors_after if e.severity == "error"]

    def test_full_cycle_complex(self) -> None:
        """A complex decision graph passes validation after round-trip."""
        version = (
            GraphBuilder("Complex")
            .start()
            .instruction("Prepare")
            .decision("Which path?", options=["Alpha", "Beta"])
            .on("Alpha")
                .instruction("Alpha work")
                .code("Alpha transform", code="pass")
                .end()
            .on("Beta")
                .instruction("Beta work")
                .end()
            .build()
        )

        data = to_json(version)
        restored = from_json(data)

        errors = validate_graph(restored)
        assert not [e for e in errors if e.severity == "error"]

        # Verify structural integrity
        assert len(restored.nodes) == len(version.nodes)
        assert len(restored.edges) == len(version.edges)
