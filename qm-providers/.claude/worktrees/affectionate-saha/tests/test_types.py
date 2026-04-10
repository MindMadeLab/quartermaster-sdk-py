"""Tests for response types and data structures."""

from qm_providers.types import (
    Message,
    NativeResponse,
    StructuredResponse,
    ThinkingResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)


class TestTokenResponse:
    def test_basic(self):
        r = TokenResponse(content="Hello")
        assert r.content == "Hello"
        assert r.stop_reason is None

    def test_with_stop_reason(self):
        r = TokenResponse(content="Done", stop_reason="end_turn")
        assert r.stop_reason == "end_turn"

    def test_empty_content(self):
        r = TokenResponse(content="")
        assert r.content == ""


class TestThinkingResponse:
    def test_basic(self):
        r = ThinkingResponse(thinking="Let me analyze...")
        assert r.thinking == "Let me analyze..."
        assert r.type == "thinking"

    def test_custom_type(self):
        r = ThinkingResponse(thinking="Planning...", type="planning")
        assert r.type == "planning"


class TestTokenUsage:
    def test_basic(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_total_input_tokens(self):
        u = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
        )
        assert u.total_input_tokens == 120

    def test_with_cache(self):
        u = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=10,
        )
        assert u.total_tokens == 150  # cache_read not counted
        assert u.total_input_tokens == 130


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(
            tool_name="get_weather",
            tool_id="call_1",
            parameters={"location": "SF"},
        )
        assert tc.tool_name == "get_weather"
        assert tc.tool_id == "call_1"
        assert tc.parameters == {"location": "SF"}

    def test_empty_parameters(self):
        tc = ToolCall(tool_name="no_args", tool_id="call_2")
        assert tc.parameters == {}


class TestToolCallResponse:
    def test_basic(self):
        r = ToolCallResponse(
            text_content="Calling weather...",
            tool_calls=[
                ToolCall(
                    tool_name="weather",
                    tool_id="call_1",
                    parameters={"city": "NYC"},
                )
            ],
            stop_reason="tool_use",
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].tool_name == "weather"
        assert r.text_content == "Calling weather..."

    def test_empty(self):
        r = ToolCallResponse()
        assert r.text_content == ""
        assert r.tool_calls == []
        assert r.stop_reason is None
        assert r.usage is None

    def test_with_usage(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        r = ToolCallResponse(usage=usage)
        assert r.usage.total_tokens == 15


class TestStructuredResponse:
    def test_basic(self):
        r = StructuredResponse(
            structured_output={"key": "value"},
            raw_output='{"key": "value"}',
        )
        assert r.structured_output == {"key": "value"}

    def test_with_usage(self):
        r = StructuredResponse(
            structured_output={},
            usage=TokenUsage(input_tokens=20, output_tokens=10),
        )
        assert r.usage.total_tokens == 30


class TestNativeResponse:
    def test_basic(self):
        r = NativeResponse(
            text_content="Response text",
            thinking=[ThinkingResponse(thinking="Hmm...")],
            tool_calls=[ToolCall(tool_name="fn", tool_id="id1", parameters={})],
            stop_reason="end_turn",
        )
        assert r.text_content == "Response text"
        assert len(r.thinking) == 1
        assert len(r.tool_calls) == 1

    def test_empty(self):
        r = NativeResponse()
        assert r.text_content == ""
        assert r.thinking == []
        assert r.tool_calls == []


class TestToolDefinition:
    def test_creation(self):
        td: ToolDefinition = {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        }
        assert td["name"] == "search"
        assert td["input_schema"]["type"] == "object"


class TestMessage:
    def test_basic(self):
        msg: Message = {"role": "user", "content": "Hello"}
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
