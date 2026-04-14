"""Typed metadata schemas per node type.

Keys match the actual quartermaster-nodes implementations exactly so the
engine can read metadata without mapping.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quartermaster_graph.enums import NodeType


# ── Base LLM configuration ───────────────────────────────────────────


class LLMMetadata(BaseModel):
    """Common LLM configuration shared by all nodes that call an LLM.

    Key names match ``AbstractLLMAssistantNode.get_metadata_key_value()``
    calls in quartermaster-nodes.
    """

    llm_model: str = "gpt-4o"
    llm_provider: str = "openai"
    llm_temperature: float = 0.5
    llm_max_input_tokens: int = 16385
    llm_max_output_tokens: int = 2048
    llm_max_messages: int | None = None
    llm_stream: bool = True
    llm_vision: bool = False
    llm_system_instruction: str = "You are helpful agent, try being precise and helpful."
    llm_thinking_level: str = "off"  # off, low, medium, high


# ── LLM nodes ────────────────────────────────────────────────────────


class InstructionMetadata(LLMMetadata):
    """Metadata for Instruction nodes — pure LLM text generation, no tools.

    Handler chain: ValidateMemoryID → PrepareMessages → ContextManager →
    TransformToProvider → GenerateStreamResponse → ProcessStreamResponse
    """

    pass


class DecisionMetadata(LLMMetadata):
    """Metadata for Decision nodes — LLM picks ONE path via ``pick_path`` tool.

    The LLM sees the available edges and calls ``pick_path(next_assistant_node_id=...)``
    to select which branch to follow.  Does NOT stream (forces ``llm_stream=False``).

    Handler chain: ValidateMemoryID → PrepareMessages → ContextManager →
    TransformToProvider → GenerateToolCall → ProcessStreamResponse
    """

    prefix_message: str = ""
    suffix_message: str = ""
    llm_stream: bool = False  # override — Decision never streams


class MergeMetadata(LLMMetadata):
    """Metadata for Merge nodes — LLM combines parallel branch outputs.

    Waits for ALL incoming branches (traverse_in=AwaitAll), then sends
    the combined child-thought content to the LLM for compression into
    a single coherent message.
    """

    prefix_message: str = "Compress following conversations into one"
    suffix_message: str = ""


class AgentMetadata(LLMMetadata):
    """Metadata for Agent nodes — autonomous agentic loop with tools.

    Iterates up to ``max_iterations`` times. Each iteration:
    1. Generate native response (with tool definitions)
    2. If tool calls present → execute via ``_tool_executor`` → add results → loop
    3. If no tool calls → break (agent is done)
    """

    program_version_ids: list[str] = Field(default_factory=list)
    max_iterations: int = 25
    context_tool_clearing_trigger: int = 10000
    context_tool_clearing_keep: int = 3
    context_max_tool_result_tokens: int = 2000


class SummarizeMetadata(LLMMetadata):
    """Metadata for Summarize nodes — LLM condenses conversation history."""

    llm_system_instruction: str = "Summarize the given conversation concisely."


# ── Control flow nodes ───────────────────────────────────────────────


class IfMetadata(BaseModel):
    """Metadata for If nodes — evaluates Python expression, picks true/false branch."""

    if_expression: str = ""


class SwitchMetadata(BaseModel):
    """Metadata for Switch nodes — evaluates multiple cases, first match wins."""

    cases: list[dict[str, str]] = Field(default_factory=list)  # [{expression, edge_id}, ...]
    default_edge_id: str = ""


class SubAssistantMetadata(BaseModel):
    """Metadata for SubAssistant nodes — calls another graph synchronously."""

    sub_assistant_id: str = ""


class BreakMetadata(BaseModel):
    """Metadata for Break nodes — stops backward message collection."""

    break_targets: list[str] = Field(default_factory=list)  # [], ['tools'], ['thinking']


# ── Data nodes ───────────────────────────────────────────────────────


class StaticMetadata(BaseModel):
    """Metadata for Static nodes — outputs fixed text content, no LLM."""

    static_text: str = ""


class StaticDecisionMetadata(BaseModel):
    """Metadata for StaticDecision — expression-based branching, no LLM."""

    expression: str = ""


class StaticMergeMetadata(BaseModel):
    """Metadata for StaticMerge — combines branches with static text, no LLM."""

    static_text: str = ""


class TextMetadata(BaseModel):
    """Metadata for Text nodes — Jinja2 template rendering."""

    text: str = ""


class VarMetadata(BaseModel):
    """Metadata for Var nodes — evaluates expression, stores in metadata."""

    name: str = ""
    expression: str = ""


class CodeMetadata(BaseModel):
    """Metadata for Code nodes — code execution (handled by runtime)."""

    code: str = ""
    filename: str = ""


# ── Memory nodes ─────────────────────────────────────────────────────


class ReadMemoryMetadata(BaseModel):
    """Metadata for ReadMemory nodes."""

    memory_name: str = ""
    memory_type: str = "flow"  # "flow" or "user"
    variable_names: list[str] = Field(default_factory=list)


class WriteMemoryMetadata(BaseModel):
    """Metadata for WriteMemory nodes."""

    memory_name: str = ""
    memory_type: str = "flow"
    variables: list[dict[str, str]] = Field(default_factory=list)  # [{name, expression}, ...]


class UpdateMemoryMetadata(BaseModel):
    """Metadata for UpdateMemory nodes."""

    memory_name: str = ""
    memory_type: str = "flow"
    variables: list[dict[str, str]] = Field(default_factory=list)


class FlowMemoryMetadata(BaseModel):
    """Metadata for FlowMemory definition nodes (not connected to flow)."""

    memory_name: str = ""
    initial_data: list[dict[str, str]] = Field(default_factory=list)


class UserMemoryMetadata(BaseModel):
    """Metadata for UserMemory definition nodes (not connected to flow)."""

    memory_name: str = ""
    initial_data: list[dict[str, str]] = Field(default_factory=list)


# ── User interaction nodes ───────────────────────────────────────────


class UserMetadata(BaseModel):
    """Metadata for User input nodes."""

    text_snippets: list[str] = Field(default_factory=list)


class UserDecisionMetadata(BaseModel):
    """Metadata for UserDecision nodes — user picks a path."""

    pass  # No extra metadata; choices come from incoming edges


class UserFormMetadata(BaseModel):
    """Metadata for UserForm nodes — structured form input."""

    parameters: list[dict[str, str]] = Field(default_factory=list)


# ── Utility nodes ────────────────────────────────────────────────────


class CommentMetadata(BaseModel):
    """Metadata for Comment nodes — documentation only."""

    comment: str = ""


class UseEnvironmentMetadata(BaseModel):
    """Metadata for UseEnvironment nodes."""

    environment_id: str = ""


# ── Lookup ───────────────────────────────────────────────────────────

_NODE_TYPE_METADATA: dict[NodeType, type[BaseModel]] = {
    # LLM nodes
    NodeType.INSTRUCTION: InstructionMetadata,
    NodeType.DECISION: DecisionMetadata,
    NodeType.MERGE: MergeMetadata,
    NodeType.AGENT: AgentMetadata,
    NodeType.SUMMARIZE: SummarizeMetadata,
    NodeType.INSTRUCTION_IMAGE_VISION: InstructionMetadata,
    # Control flow
    NodeType.IF: IfMetadata,
    NodeType.SWITCH: SwitchMetadata,
    NodeType.SUB_ASSISTANT: SubAssistantMetadata,
    NodeType.BREAK: BreakMetadata,
    # Data
    NodeType.STATIC: StaticMetadata,
    NodeType.STATIC_DECISION: StaticDecisionMetadata,
    NodeType.STATIC_MERGE: StaticMergeMetadata,
    NodeType.TEXT: TextMetadata,
    NodeType.VAR: VarMetadata,
    NodeType.CODE: CodeMetadata,
    NodeType.PROGRAM_RUNNER: CodeMetadata,
    # Memory
    NodeType.READ_MEMORY: ReadMemoryMetadata,
    NodeType.WRITE_MEMORY: WriteMemoryMetadata,
    NodeType.UPDATE_MEMORY: UpdateMemoryMetadata,
    NodeType.FLOW_MEMORY: FlowMemoryMetadata,
    NodeType.USER_MEMORY: UserMemoryMetadata,
    # User interaction
    NodeType.USER: UserMetadata,
    NodeType.USER_DECISION: UserDecisionMetadata,
    NodeType.USER_FORM: UserFormMetadata,
    # Utility
    NodeType.COMMENT: CommentMetadata,
    NodeType.USE_ENVIRONMENT: UseEnvironmentMetadata,
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
