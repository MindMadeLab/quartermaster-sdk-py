"""Tests for pre-built graph templates."""


from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import AgentVersion
from quartermaster_graph.templates import Templates
from quartermaster_graph.validation import validate_graph


class TestSimpleChat:
    def test_creates_valid_graph(self):
        version = Templates.simple_chat()
        assert isinstance(version, AgentVersion)
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_has_start_and_end(self):
        version = Templates.simple_chat()
        types = [n.type for n in version.nodes]
        assert NodeType.START in types
        assert NodeType.END in types

    def test_custom_model(self):
        version = Templates.simple_chat(model="claude-3")
        inst_nodes = [n for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert any(n.metadata.get("model") == "claude-3" for n in inst_nodes)


class TestDecisionTree:
    def test_creates_valid_graph(self):
        version = Templates.decision_tree()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_custom_options(self):
        version = Templates.decision_tree(options=["A", "B", "C"])
        assert len(version.nodes) >= 5  # start + decision + 3 branches + 3 ends

    def test_has_decision_node(self):
        version = Templates.decision_tree()
        types = [n.type for n in version.nodes]
        assert NodeType.DECISION in types


class TestRagPipeline:
    def test_creates_valid_graph(self):
        version = Templates.rag_pipeline()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_has_tool_node(self):
        version = Templates.rag_pipeline()
        types = [n.type for n in version.nodes]
        assert NodeType.TOOL in types


class TestMultiStep:
    def test_creates_valid_graph(self):
        version = Templates.multi_step()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_custom_steps(self):
        version = Templates.multi_step(steps=["A", "B", "C", "D"])
        inst_nodes = [n for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert len(inst_nodes) == 4


class TestParallelProcessing:
    def test_creates_valid_graph(self):
        version = Templates.parallel_processing()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_custom_branches(self):
        version = Templates.parallel_processing(branches=["X", "Y"])
        assert len(version.nodes) >= 4
