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


class TestToolUsingAgent:
    """Tests for the tool-using agent template."""

    def test_creates_valid_graph(self):
        """Template produces a graph that passes validation."""
        version = Templates.tool_using_agent()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_has_tool_node(self):
        """Template includes at least one Tool node."""
        version = Templates.tool_using_agent()
        types = [n.type for n in version.nodes]
        assert NodeType.TOOL in types

    def test_has_plan_and_synthesize(self):
        """Template has Plan and Synthesize instruction nodes."""
        version = Templates.tool_using_agent()
        inst_names = [n.name for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert "Plan" in inst_names
        assert "Synthesize" in inst_names

    def test_custom_tools(self):
        """Multiple tools create multiple Tool nodes."""
        version = Templates.tool_using_agent(tools=["search", "calculator"])
        tool_nodes = [n for n in version.nodes if n.type == NodeType.TOOL]
        assert len(tool_nodes) == 2

    def test_custom_model(self):
        """Custom model is propagated to instruction nodes."""
        version = Templates.tool_using_agent(model="claude-3-opus")
        inst_nodes = [n for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert all(n.metadata.get("model") == "claude-3-opus" for n in inst_nodes)


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


class TestAdvancedRag:
    """Tests for the advanced RAG pipeline template."""

    def test_creates_valid_graph(self):
        """Template produces a graph that passes validation."""
        version = Templates.advanced_rag()
        errors = validate_graph(version)
        real_errors = [e for e in errors if e.severity == "error"]
        assert len(real_errors) == 0

    def test_has_two_tool_nodes(self):
        """Template has retrieval and reranking tool nodes."""
        version = Templates.advanced_rag()
        tool_nodes = [n for n in version.nodes if n.type == NodeType.TOOL]
        assert len(tool_nodes) == 2
        tool_names = {n.metadata.get("tool_name") for n in tool_nodes}
        assert "vector_search" in tool_names
        assert "reranker" in tool_names

    def test_has_rewrite_and_generate(self):
        """Template has query rewrite and answer generation steps."""
        version = Templates.advanced_rag()
        inst_names = [n.name for n in version.nodes if n.type == NodeType.INSTRUCTION]
        assert "Rewrite Query" in inst_names
        assert "Generate Answer" in inst_names

    def test_custom_tools(self):
        """Custom tool names are used in the pipeline."""
        version = Templates.advanced_rag(
            retrieval_tool="elastic_search",
            rerank_tool="cross_encoder",
        )
        tool_nodes = [n for n in version.nodes if n.type == NodeType.TOOL]
        tool_names = {n.metadata.get("tool_name") for n in tool_nodes}
        assert "elastic_search" in tool_names
        assert "cross_encoder" in tool_names

    def test_node_count(self):
        """Template has expected number of nodes: start + rewrite + retrieve + rerank + generate + end."""
        version = Templates.advanced_rag()
        assert len(version.nodes) == 6
