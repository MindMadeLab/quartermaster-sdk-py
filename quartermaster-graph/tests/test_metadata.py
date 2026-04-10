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
        assert get_metadata_class(NodeType.MERGE) is None
        assert get_metadata_class(NodeType.COMMENT) is None


class TestValidateMetadata:
    def test_valid_instruction(self):
        result = validate_metadata(NodeType.INSTRUCTION, {
            "system_instruction": "Be helpful",
            "model": "gpt-4o",
        })
        assert isinstance(result, InstructionMetadata)
        assert result.system_instruction == "Be helpful"

    def test_valid_code(self):
        result = validate_metadata(NodeType.CODE, {
            "language": "python",
            "code": "x = 1",
        })
        assert isinstance(result, CodeMetadata)
        assert result.code == "x = 1"

    def test_untyped_returns_none(self):
        assert validate_metadata(NodeType.START, {}) is None

    def test_defaults_applied(self):
        result = validate_metadata(NodeType.INSTRUCTION, {})
        assert isinstance(result, InstructionMetadata)
        assert result.model == "gpt-4o"
        assert result.temperature == 0.7

    def test_extra_fields_ignored(self):
        result = validate_metadata(NodeType.STATIC, {
            "content": "hello",
            "extra_field": "ignored",
        })
        assert isinstance(result, StaticMetadata)
        assert result.content == "hello"


class TestIndividualMetadata:
    def test_instruction_metadata(self):
        m = InstructionMetadata(
            system_instruction="test",
            model="claude-3",
            provider="anthropic",
            temperature=0.5,
            max_tokens=100,
            tools=["search"],
        )
        assert m.model == "claude-3"
        assert m.tools == ["search"]

    def test_decision_inherits_instruction(self):
        m = DecisionMetadata(decision_prompt="Choose", model="gpt-4o")
        assert m.decision_prompt == "Choose"
        assert m.model == "gpt-4o"  # inherited

    def test_if_metadata(self):
        m = IfMetadata(expression="x > 0", variable="x")
        assert m.expression == "x > 0"

    def test_switch_metadata(self):
        m = SwitchMetadata(variable="status", cases={"active": "go", "inactive": "stop"})
        assert len(m.cases) == 2

    def test_code_metadata_defaults(self):
        m = CodeMetadata()
        assert m.language == "python"
        assert m.timeout_seconds == 30

    def test_var_metadata(self):
        m = VarMetadata(variable_name="count", default_value="0")
        assert m.variable_name == "count"

    def test_user_form_metadata(self):
        m = UserFormMetadata(
            fields=[{"name": "email", "type": "text"}],
            submit_label="Go",
        )
        assert len(m.fields) == 1
        assert m.submit_label == "Go"

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
