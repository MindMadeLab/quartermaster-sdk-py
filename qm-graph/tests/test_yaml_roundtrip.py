"""Integration tests: Build graph -> to_yaml -> from_yaml -> compare."""

from __future__ import annotations

from qm_graph.builder import GraphBuilder
from qm_graph.enums import NodeType, TraverseOut
from qm_graph.serialization import from_yaml, to_yaml
from qm_graph.validation import validate_graph


class TestSimpleYamlRoundtrip:
    """YAML round-trip tests for linear graphs."""

    def test_start_instruction_end(self) -> None:
        """Build a simple graph, convert to YAML, parse back, and compare."""
        version = (
            GraphBuilder("YAML Agent")
            .start()
            .instruction("Process", model="gpt-4o", temperature=0.3)
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 0

        restored = from_yaml(yaml_str)

        assert restored.agent_id == version.agent_id
        assert restored.version == version.version
        assert restored.start_node_id == version.start_node_id
        assert len(restored.nodes) == len(version.nodes)
        assert len(restored.edges) == len(version.edges)

    def test_node_metadata_preserved(self) -> None:
        """Node metadata (model, temperature, etc.) survives YAML round-trip."""
        version = (
            GraphBuilder("Meta Agent")
            .start()
            .instruction(
                "Generate",
                model="claude-3-opus",
                provider="anthropic",
                temperature=0.9,
                system_instruction="Be creative.",
            )
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        orig_inst = [n for n in version.nodes if n.type == NodeType.INSTRUCTION][0]
        rest_inst = [n for n in restored.nodes if n.type == NodeType.INSTRUCTION][0]

        assert rest_inst.metadata["model"] == orig_inst.metadata["model"]
        assert rest_inst.metadata["provider"] == orig_inst.metadata["provider"]
        assert rest_inst.metadata["temperature"] == orig_inst.metadata["temperature"]
        assert rest_inst.metadata["system_instruction"] == orig_inst.metadata["system_instruction"]

    def test_multi_step_yaml(self) -> None:
        """Multi-step pipeline survives YAML serialization."""
        version = (
            GraphBuilder("Multi-Step")
            .start()
            .instruction("Step 1")
            .code("Run code", code="result = 42", language="python")
            .static("Output", content="Final answer.")
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        assert len(restored.nodes) == 5
        code_nodes = [n for n in restored.nodes if n.type == NodeType.CODE]
        assert len(code_nodes) == 1
        assert code_nodes[0].metadata["code"] == "result = 42"
        assert code_nodes[0].metadata["language"] == "python"


class TestDecisionYamlRoundtrip:
    """YAML round-trip tests for decision graphs."""

    def test_decision_labels_preserved(self) -> None:
        """Decision branch labels survive YAML round-trip."""
        version = (
            GraphBuilder("Decision YAML")
            .start()
            .decision("Route?", options=["Left", "Right"])
            .on("Left").instruction("Go left").end()
            .on("Right").instruction("Go right").end()
            .build()
        )

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        decision_nodes = [n for n in restored.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        assert decision_nodes[0].traverse_out == TraverseOut.SPAWN_PICKED

        decision_edges = [e for e in restored.edges if e.source_id == decision_nodes[0].id]
        labels = {e.label for e in decision_edges}
        assert "Left" in labels
        assert "Right" in labels

    def test_complex_branching(self) -> None:
        """Graph with decision + multiple branches survives YAML round-trip."""
        version = (
            GraphBuilder("Complex YAML")
            .start()
            .instruction("Analyze")
            .decision("Category?", options=["Urgent", "Normal", "Low"])
            .on("Urgent").instruction("Handle urgent").end()
            .on("Normal").instruction("Handle normal").end()
            .on("Low").static("Auto-reply", content="We will get back to you.").end()
            .build()
        )

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        errors = validate_graph(restored)
        assert not [e for e in errors if e.severity == "error"]

        end_nodes = [n for n in restored.nodes if n.type == NodeType.END]
        assert len(end_nodes) == 3


class TestYamlContainsReadableContent:
    """Verify YAML output is human-readable."""

    def test_yaml_contains_node_names(self) -> None:
        """YAML output includes node names as readable strings."""
        version = (
            GraphBuilder("Readable Agent")
            .start()
            .instruction("Summarize the document")
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        assert "Summarize the document" in yaml_str
        assert "Start" in yaml_str
        assert "End" in yaml_str

    def test_yaml_contains_enum_values(self) -> None:
        """YAML output contains enum string values, not Python repr."""
        version = (
            GraphBuilder("Enum Agent")
            .start()
            .instruction("Work")
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        assert "Start1" in yaml_str  # NodeType.START.value
        assert "Instruction1" in yaml_str  # NodeType.INSTRUCTION.value
        assert "End1" in yaml_str  # NodeType.END.value


class TestYamlValidationAfterRoundtrip:
    """Ensure graphs stay valid after YAML round-trip."""

    def test_simple_graph_valid_after_yaml(self) -> None:
        """Simple graph passes validation after YAML round-trip."""
        version = (
            GraphBuilder("Valid YAML")
            .start()
            .instruction("Do work")
            .end()
            .build()
        )

        errors_before = validate_graph(version)
        assert not [e for e in errors_before if e.severity == "error"]

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        errors_after = validate_graph(restored)
        assert not [e for e in errors_after if e.severity == "error"]

    def test_tool_graph_valid_after_yaml(self) -> None:
        """Graph with tool nodes passes validation after YAML round-trip."""
        version = (
            GraphBuilder("Tool YAML")
            .start()
            .tool("Search", tool_name="vector_search")
            .instruction("Generate answer")
            .end()
            .build()
        )

        yaml_str = to_yaml(version)
        restored = from_yaml(yaml_str)

        errors = validate_graph(restored)
        assert not [e for e in errors if e.severity == "error"]

        tool_nodes = [n for n in restored.nodes if n.type == NodeType.TOOL]
        assert len(tool_nodes) == 1
        assert tool_nodes[0].metadata["tool_name"] == "vector_search"
