"""Response types and data structures for LLM providers.

This module defines the unified response types returned by all LLM providers,
along with supporting types for tools, structured output, and message history.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict


@dataclass
class TokenResponse:
    """A single token or chunk of response content.

    Attributes:
        content: The text content of this response chunk.
        stop_reason: Why the response stopped ('end_turn', 'max_tokens', 'tool_use', etc.).
    """

    content: str
    stop_reason: str | None = None


@dataclass
class ThinkingResponse:
    """Extended thinking/reasoning content from models like Claude.

    Attributes:
        thinking: The internal reasoning/thinking text.
        type: Type of thinking block ('thinking', 'planning', etc.).
    """

    thinking: str
    type: str = "thinking"


@dataclass
class TokenUsage:
    """Token usage statistics for a response.

    Attributes:
        input_tokens: Number of tokens in the input prompt.
        output_tokens: Number of tokens in the response.
        cache_creation_input_tokens: Tokens used to create cache (Claude).
        cache_read_input_tokens: Tokens read from cache (Claude).
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output, excluding cache reads)."""
        return self.input_tokens + self.output_tokens

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache creation."""
        return self.input_tokens + self.cache_creation_input_tokens


class ToolParameter(TypedDict, total=False):
    """Type hint for tool call parameters."""

    pass


@dataclass
class ToolCall:
    """A single tool/function call in a response.

    Attributes:
        tool_name: Name of the tool being called.
        tool_id: Unique identifier for this tool call.
        parameters: Parameters passed to the tool as a dict.
    """

    tool_name: str
    tool_id: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResponse:
    """Response containing tool calls and text.

    Attributes:
        text_content: Any text content before/after tool calls.
        tool_calls: List of tool calls made by the model.
        stop_reason: Why the response stopped.
        usage: Token usage information.
    """

    text_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass
class StructuredResponse:
    """Response with validated structured (JSON) output.

    Attributes:
        structured_output: The parsed JSON/structured data.
        raw_output: The raw model output before parsing.
        stop_reason: Why the response stopped.
        usage: Token usage information.
    """

    structured_output: dict[str, Any]
    raw_output: str = ""
    stop_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass
class NativeResponse:
    """Hybrid response containing text, tool calls, and thinking.

    This is the most complete response type, combining all possible
    model output elements.

    Attributes:
        text_content: The main text response.
        thinking: Extended thinking/reasoning blocks.
        tool_calls: Function/tool calls requested by the model.
        stop_reason: Why the response stopped.
        usage: Token usage information.
    """

    text_content: str = ""
    thinking: list[ThinkingResponse] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: TokenUsage | None = None


class ToolDefinition(TypedDict, total=False):
    """Tool/function definition passed to models.

    Attributes:
        name: Tool name.
        description: Human-readable description of the tool.
        input_schema: JSON Schema describing tool input parameters.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class Message(TypedDict, total=False):
    """A single message in a conversation.

    Attributes:
        role: 'user', 'assistant', or 'system'.
        content: Message text content.
        tool_calls: Tool calls made in this message (assistant only).
        tool_results: Results from tool execution (user role for tool responses).
    """

    role: str
    content: str
    tool_calls: list[ToolCall]
    tool_results: list[dict[str, Any]]


class MessageHistory(Protocol):
    """Protocol for message/conversation history storage.

    Implementations can use in-memory dicts, databases, or file-based storage.
    This protocol allows providers to work with different history backends.
    """

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history.

        Args:
            role: 'user', 'assistant', or 'system'.
            content: Message text.
        """
        ...

    def add_tool_call(self, tool_name: str, tool_id: str, parameters: dict) -> None:
        """Add a tool call to the latest assistant message.

        Args:
            tool_name: Name of the tool called.
            tool_id: Unique identifier for this call.
            parameters: Parameters passed to the tool.
        """
        ...

    def add_tool_result(self, tool_id: str, result: str | dict) -> None:
        """Add a tool execution result.

        Args:
            tool_id: ID of the tool call this result is for.
            result: Tool execution result.
        """
        ...

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get conversation messages.

        Args:
            limit: Maximum number of recent messages to return.

        Returns:
            List of messages in conversation order.
        """
        ...

    def clear(self) -> None:
        """Clear all messages from history."""
        ...

    def __len__(self) -> int:
        """Return number of messages in history."""
        ...
