"""Tests for pre-built graph templates."""


from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import GraphSpec
from quartermaster_graph.templates import Templates
from quartermaster_graph.validation import validate_graph


class TestSimpleChat:
    def test_creates_valid_graph(self):
        version = Templates.simple_chat()
        assert isinstance(version, GraphSpec)
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
        assert any(n.metadata.get("llm_model") == "claude-3" for n in inst_nodes)


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


class TestMultiAgentSupervisor:
    """Tests for the multi-agent supervisor template."""

    def test_creates_valid_graph(self):
        """Template produces a graph that passes validation."""
        version = Templates.multi_agent_supervisor()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_has_decision_node(self):
        """Template includes a Decision node for routing."""
        version = Templates.multi_agent_supervisor()
        types = [n.type for n in version.nodes]
        assert NodeType.DECISION in types

    def test_has_supervisor_instruction(self):
        """Template has a Supervisor instruction node."""
        version = Templates.multi_agent_supervisor()
        inst_names = [n.name for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert "Supervisor" in inst_names

    def test_default_workers(self):
        """Default template has Researcher, Writer, Coder workers."""
        version = Templates.multi_agent_supervisor()
        inst_names = [n.name for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert "Researcher" in inst_names
        assert "Writer" in inst_names
        assert "Coder" in inst_names

    def test_custom_workers(self):
        """Custom worker names create corresponding instruction nodes."""
        version = Templates.multi_agent_supervisor(worker_names=["Analyst", "Reporter"])
        inst_names = [n.name for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert "Analyst" in inst_names
        assert "Reporter" in inst_names

    def test_labeled_edges(self):
        """Decision edges are labeled with worker names."""
        version = Templates.multi_agent_supervisor()
        decision_nodes = [n for n in version.nodes if n.type == NodeType.DECISION]
        assert len(decision_nodes) == 1
        decision_edges = [e for e in version.edges if e.source_id == decision_nodes[0].id]
        labels = {e.label for e in decision_edges}
        assert "Researcher" in labels
        assert "Writer" in labels
        assert "Coder" in labels


