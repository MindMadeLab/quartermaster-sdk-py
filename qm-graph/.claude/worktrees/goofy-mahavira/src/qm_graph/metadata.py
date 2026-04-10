"""Typed metadata schemas per node type."""

from __future__ import annotations

from pydantic import BaseModel, Field

from qm_graph.enums import NodeType


class InstructionMetadata(BaseModel):
    """Metadata for instruction-type nodes (LLM calls)."""

    system_instruction: str = ""
    model: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[str] = Field(default_factory=list)
    response_format: str | None = None


class DecisionMetadata(InstructionMetadata):
    """Metadata for decision nodes — an instruction that picks a branch."""

    decision_prompt: str = ""


class IfMetadata(BaseModel):
    """Metadata for conditional if-nodes."""

    expression: str = ""
    variable: str = ""


class SwitchMetadata(BaseModel):
    """Metadata for switch/case nodes."""

    variable: str = ""
    cases: dict[str, str] = Field(default_factory=dict)
    default_label: str = "default"


class StaticMetadata(BaseModel):
    """Metadata for static content nodes."""

    content: str = ""


class CodeMetadata(BaseModel):
    """Metadata for code execution nodes."""

    language: str = "python"
    code: str = ""
    timeout_seconds: int = 30


class VarMetadata(BaseModel):
    """Metadata for variable nodes."""

    variable_name: str = ""
    default_value: str | None = None


class UserFormMetadata(BaseModel):
    """Metadata for user form nodes."""

    fields: list[dict[str, str]] = Field(default_factory=list)
    submit_label: str = "Submit"


class ToolMetadata(BaseModel):
    """Metadata for tool invocation nodes."""

    tool_name: str = ""
    tool_args: dict[str, str] = Field(default_factory=dict)


class ApiCallMetadata(BaseModel):
    """Metadata for API call nodes."""

    url: str = ""
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str = ""


class TimerMetadata(BaseModel):
    """Metadata for timer/delay nodes."""

    delay_seconds: float = 0.0


class LoopMetadata(BaseModel):
    """Metadata for loop nodes."""

    max_iterations: int = 10
    break_condition: str = ""


class ValidatorMetadata(BaseModel):
    """Metadata for validator nodes."""

    validation_schema: str = ""
    error_message: str = "Validation failed"


class TransformerMetadata(BaseModel):
    """Metadata for transformer nodes."""

    transform_expression: str = ""
    input_variable: str = ""
    output_variable: str = ""


class FilterMetadata(BaseModel):
    """Metadata for filter nodes."""

    filter_expression: str = ""
    input_variable: str = ""


class AggregatorMetadata(BaseModel):
    """Metadata for aggregator nodes."""

    strategy: str = "concat"  # concat, merge, first, last
    input_variable: str = ""


class RouterMetadata(BaseModel):
    """Metadata for router nodes."""

    route_expression: str = ""
    routes: dict[str, str] = Field(default_factory=dict)


class LogMetadata(BaseModel):
    """Metadata for log nodes."""

    level: str = "info"
    message_template: str = ""


class NotificationMetadata(BaseModel):
    """Metadata for notification nodes."""

    channel: str = ""
    message_template: str = ""


_NODE_TYPE_METADATA: dict[NodeType, type[BaseModel]] = {
    NodeType.INSTRUCTION: InstructionMetadata,
    NodeType.DECISION: DecisionMetadata,
    NodeType.REASONING: InstructionMetadata,
    NodeType.IF: IfMetadata,
    NodeType.SWITCH: SwitchMetadata,
    NodeType.STATIC: StaticMetadata,
    NodeType.TEXT: StaticMetadata,
    NodeType.CODE: CodeMetadata,
    NodeType.PROGRAM_RUNNER: CodeMetadata,
    NodeType.VAR: VarMetadata,
    NodeType.USER_FORM: UserFormMetadata,
    NodeType.TOOL: ToolMetadata,
    NodeType.API_CALL: ApiCallMetadata,
    NodeType.WEBHOOK: ApiCallMetadata,
    NodeType.TIMER: TimerMetadata,
    NodeType.LOOP: LoopMetadata,
    NodeType.VALIDATOR: ValidatorMetadata,
    NodeType.TRANSFORMER: TransformerMetadata,
    NodeType.FILTER: FilterMetadata,
    NodeType.AGGREGATOR: AggregatorMetadata,
    NodeType.ROUTER: RouterMetadata,
    NodeType.LOG: LogMetadata,
    NodeType.NOTIFICATION: NotificationMetadata,
}


def get_metadata_class(node_type: NodeType) -> type[BaseModel] | None:
    """Get the metadata schema class for a node type, or None if untyped."""
    return _NODE_TYPE_METADATA.get(node_type)


def validate_metadata(node_type: NodeType, metadata: dict) -> BaseModel | None:  # type: ignore[type-arg]
    """Validate and parse metadata for a given node type.

    Returns a typed metadata object, or None if the node type has no schema.
    Raises ValidationError if the metadata doesn't match the schema.
    """
    cls = get_metadata_class(node_type)
    if cls is None:
        return None
    return cls.model_validate(metadata)
