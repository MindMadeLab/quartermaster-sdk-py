"""Tests for metadata schemas and validation."""


from quartermaster_graph.enums import NodeType
from quartermaster_graph.metadata import (
    CodeMetadata,
    DecisionMetadata,
    IfMetadata,
    InstructionMetadata,
    StaticMetadata,
    SwitchMetadata,
    UserFormMetadata,
    VarMetadata,
    get_metadata_class,
    validate_metadata,
)


class TestGetMetadataClass:
    def test_instruction_type(self):
        assert get_metadata_class(NodeType.INSTRUCTION) is InstructionMetadata

    def test_decision_type(self):
        assert get_metadata_class(NodeType.DECISION) is DecisionMetadata

    def test_if_type(self):
        assert get_metadata_class(NodeType.IF) is IfMetadata

    def test_switch_type(self):
        assert get_metadata_class(NodeType.SWITCH) is SwitchMetadata

    def test_code_type(self):
        assert get_metadata_class(NodeType.CODE) is CodeMetadata

    def test_static_type(self):
        assert get_metadata_class(NodeType.STATIC) is StaticMetadata

    def test_var_type(self):
        assert get_metadata_class(NodeType.VAR) is VarMetadata

    def test_untyped_returns_none(self):
        assert get_metadata_class(NodeType.START) is None
        assert get_metadata_class(NodeType.END) is None
        assert get_metadata_class(NodeType.BLANK) is None


class TestValidateMetadata:
    def test_valid_instruction(self):
        result = validate_metadata(NodeType.INSTRUCTION, {
            "llm_system_instruction": "Be helpful",
            "llm_model": "gpt-4o",
        })
        assert isinstance(result, InstructionMetadata)
        assert result.llm_system_instruction == "Be helpful"

    def test_valid_code(self):
        result = validate_metadata(NodeType.CODE, {
            "filename": "script.py",
            "code": "x = 1",
        })
        assert isinstance(result, CodeMetadata)
        assert result.code == "x = 1"

    def test_untyped_returns_none(self):
        assert validate_metadata(NodeType.START, {}) is None

    def test_defaults_applied(self):
        result = validate_metadata(NodeType.INSTRUCTION, {})
        assert isinstance(result, InstructionMetadata)
        assert result.llm_model == "gpt-4o"
        assert result.llm_temperature == 0.5

    def test_extra_fields_ignored(self):
        result = validate_metadata(NodeType.STATIC, {
            "static_text": "hello",
            "extra_field": "ignored",
        })
        assert isinstance(result, StaticMetadata)
        assert result.static_text == "hello"


class TestIndividualMetadata:
    def test_instruction_metadata(self):
        m = InstructionMetadata(
            llm_system_instruction="test",
            llm_model="claude-3",
            llm_provider="anthropic",
            llm_temperature=0.5,
        )
        assert m.llm_model == "claude-3"
        assert m.llm_system_instruction == "test"

    def test_decision_inherits_instruction(self):
        m = DecisionMetadata(prefix_message="Choose", llm_model="gpt-4o")
        assert m.prefix_message == "Choose"
        assert m.llm_model == "gpt-4o"  # inherited

    def test_if_metadata(self):
        m = IfMetadata(if_expression="x > 0")
        assert m.if_expression == "x > 0"

    def test_switch_metadata(self):
        m = SwitchMetadata(cases=[{"expression": "active", "edge_id": "go"}, {"expression": "inactive", "edge_id": "stop"}])
        assert len(m.cases) == 2

    def test_code_metadata_defaults(self):
        m = CodeMetadata()
        assert m.filename == ""
        assert m.code == ""

    def test_var_metadata(self):
        m = VarMetadata(name="count", expression="0")
        assert m.name == "count"

    def test_user_form_metadata(self):
        m = UserFormMetadata(
            parameters=[{"name": "email", "type": "text"}],
        )
        assert len(m.parameters) == 1
