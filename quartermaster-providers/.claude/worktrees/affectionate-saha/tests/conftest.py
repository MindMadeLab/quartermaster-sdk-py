"""Shared test fixtures."""

import pytest

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.testing import MockProvider, InMemoryHistory
from quartermaster_providers.types import (
    NativeResponse,
    StructuredResponse,
    ThinkingResponse,
    ToolCall,
    ToolCallResponse,
    TokenResponse,
    TokenUsage,
)


@pytest.fixture
def basic_config():
    return LLMConfig(model="gpt-4o", provider="openai")


@pytest.fixture
def streaming_config():
    return LLMConfig(model="gpt-4o", provider="openai", stream=True)


@pytest.fixture
def anthropic_config():
    return LLMConfig(model="claude-sonnet-4-20250514", provider="anthropic")


@pytest.fixture
def thinking_config():
    return LLMConfig(
        model="claude-3-7-sonnet-20250219",
        provider="anthropic",
        thinking_enabled=True,
        thinking_budget=10000,
    )


@pytest.fixture
def google_config():
    return LLMConfig(model="gemini-2.0-flash", provider="google")


@pytest.fixture
def mock_provider():
    return MockProvider(
        responses=[
            TokenResponse(content="Hello!", stop_reason="end_turn"),
            TokenResponse(content="World!", stop_reason="end_turn"),
        ]
    )


@pytest.fixture
def mock_tool_provider():
    return MockProvider(
        tool_responses=[
            ToolCallResponse(
                text_content="I'll check the weather.",
                tool_calls=[
                    ToolCall(
                        tool_name="get_weather",
                        tool_id="call_123",
                        parameters={"location": "San Francisco"},
                    )
                ],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=50, output_tokens=20),
            )
        ]
    )


@pytest.fixture
def mock_native_provider():
    return MockProvider(
        native_responses=[
            NativeResponse(
                text_content="Let me think about that.",
                thinking=[
                    ThinkingResponse(thinking="Analyzing the question...", type="thinking"),
                ],
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
        ]
    )


@pytest.fixture
def mock_structured_provider():
    return MockProvider(
        structured_responses=[
            StructuredResponse(
                structured_output={"title": "Test", "summary": "A test article"},
                raw_output='{"title": "Test", "summary": "A test article"}',
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=30, output_tokens=15),
            )
        ]
    )


@pytest.fixture
def sample_tool():
    return {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }


@pytest.fixture
def history():
    return InMemoryHistory()
