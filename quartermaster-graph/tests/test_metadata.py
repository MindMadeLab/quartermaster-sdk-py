"""Tests for metadata schemas and validation."""


from quartermaster_graph.enums import NodeType
from quartermaster_graph.metadata import (
    AggregatorMetadata,
    ApiCallMetadata,
    CodeMetadata,
    DecisionMetadata,
    FilterMetadata,
    IfMetadata,
    InstructionMetadata,
    LogMetadata,
    LoopMetadata,
    NotificationMetadata,
    RouterMetadata,
    StaticMetadata,
    SwitchMetadata,
    TimerMetadata,
    ToolMetadata,
    TransformerMetadata,
    UserFormMetadata,
    ValidatorMetadata,
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

    def test_tool_type(self):
        assert get_metadata_class(NodeType.TOOL) is ToolMetadata

    def test_loop_type(self):
        assert get_metadata_class(NodeType.LOOP) is LoopMetadata

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

    def test_tool_metadata(self):
        m = ToolMetadata(tool_name="web_search", tool_args={"query": "test"})
        assert m.tool_name == "web_search"

    def test_api_call_metadata(self):
        m = ApiCallMetadata(url="https://api.example.com", method="POST")
        assert m.method == "POST"

    def test_timer_metadata(self):
        m = TimerMetadata(delay_seconds=5.0)
        assert m.delay_seconds == 5.0

    def test_loop_metadata(self):
        m = LoopMetadata(max_iterations=20, break_condition="done == True")
        assert m.max_iterations == 20

    def test_validator_metadata(self):
        m = ValidatorMetadata(validation_schema='{"type": "object"}')
        assert "object" in m.validation_schema

    def test_transformer_metadata(self):
        m = TransformerMetadata(
            transform_expression="upper()",
            input_variable="text",
            output_variable="result",
        )
        assert m.output_variable == "result"

    def test_filter_metadata(self):
        m = FilterMetadata(filter_expression="len(x) > 0", input_variable="items")
        assert m.input_variable == "items"

    def test_aggregator_metadata(self):
        m = AggregatorMetadata(strategy="merge", input_variable="results")
        assert m.strategy == "merge"

    def test_router_metadata(self):
        m = RouterMetadata(
            route_expression="category",
            routes={"A": "path_a", "B": "path_b"},
        )
        assert len(m.routes) == 2

    def test_log_metadata(self):
        m = LogMetadata(level="warning", message_template="Issue: {msg}")
        assert m.level == "warning"

    def test_notification_metadata(self):
        m = NotificationMetadata(channel="slack", message_template="Alert!")
        assert m.channel == "slack"
