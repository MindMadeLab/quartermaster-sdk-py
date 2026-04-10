"""Testing utilities for qm-providers.

Provides MockProvider and InMemoryHistory for unit testing applications
that use qm-providers without making real API calls.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from qm_providers.base import AbstractLLMProvider
from qm_providers.config import LLMConfig
from qm_providers.types import (
    Message,
    NativeResponse,
    StructuredResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
    TokenResponse,
    TokenUsage,
)


class MockProvider(AbstractLLMProvider):
    """Mock provider for testing.

    Returns pre-configured responses in order. Tracks all calls
    for assertion in tests.

    Example:
        mock = MockProvider(responses=[
            TokenResponse(content="Hello!"),
            TokenResponse(content="World!"),
        ])
        resp = await mock.generate_text_response("hi", config)
        assert resp.content == "Hello!"
        assert mock.call_count == 1
        assert mock.last_prompt == "hi"
    """

    PROVIDER_NAME = "mock"

    def __init__(
        self,
        responses: list[TokenResponse] | None = None,
        tool_responses: list[ToolCallResponse] | None = None,
        structured_responses: list[StructuredResponse] | None = None,
        native_responses: list[NativeResponse] | None = None,
        models: list[str] | None = None,
        transcription_text: str = "Mock transcription",
    ):
        self._responses = list(responses or [])
        self._tool_responses = list(tool_responses or [])
        self._structured_responses = list(structured_responses or [])
        self._native_responses = list(native_responses or [])
        self._models = models or ["mock-model-1", "mock-model-2"]
        self._transcription_text = transcription_text

        self._response_index = 0
        self._tool_response_index = 0
        self._structured_response_index = 0
        self._native_response_index = 0

        # Tracking
        self.calls: list[dict[str, Any]] = []
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_config: LLMConfig | None = None
        self.last_tools: list[ToolDefinition] | None = None

    def _track_call(
        self,
        method: str,
        prompt: str,
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> None:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_config = config
        self.last_tools = tools
        self.calls.append(
            {
                "method": method,
                "prompt": prompt,
                "config": config,
                "tools": tools,
            }
        )

    def _next_response(self) -> TokenResponse:
        if not self._responses:
            return TokenResponse(content="Mock response", stop_reason="end_turn")
        idx = self._response_index % len(self._responses)
        self._response_index += 1
        return self._responses[idx]

    def _next_tool_response(self) -> ToolCallResponse:
        if not self._tool_responses:
            return ToolCallResponse(
                text_content="",
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        idx = self._tool_response_index % len(self._tool_responses)
        self._tool_response_index += 1
        return self._tool_responses[idx]

    def _next_structured_response(self) -> StructuredResponse:
        if not self._structured_responses:
            return StructuredResponse(
                structured_output={"mock": True},
                raw_output='{"mock": true}',
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        idx = self._structured_response_index % len(self._structured_responses)
        self._structured_response_index += 1
        return self._structured_responses[idx]

    def _next_native_response(self) -> NativeResponse:
        if not self._native_responses:
            return NativeResponse(
                text_content="Mock native response",
                thinking=[],
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        idx = self._native_response_index % len(self._native_responses)
        self._native_response_index += 1
        return self._native_responses[idx]

    async def list_models(self) -> list[str]:
        return list(self._models)

    def estimate_token_count(self, text: str, model: str) -> int:
        return len(text.split())

    def prepare_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        return dict(tool)

    async def generate_text_response(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> TokenResponse | AsyncIterator[TokenResponse]:
        self._track_call("generate_text_response", prompt, config)

        if config.stream:
            return self._stream_response(prompt, config)

        return self._next_response()

    async def _stream_response(
        self, prompt: str, config: LLMConfig
    ) -> AsyncIterator[TokenResponse]:
        resp = self._next_response()
        # Split content into chunks for streaming simulation
        words = resp.content.split()
        for i, word in enumerate(words):
            prefix = " " if i > 0 else ""
            yield TokenResponse(content=prefix + word)
        yield TokenResponse(content="", stop_reason=resp.stop_reason or "end_turn")

    async def generate_tool_parameters(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        config: LLMConfig,
    ) -> ToolCallResponse:
        self._track_call("generate_tool_parameters", prompt, config, tools)
        return self._next_tool_response()

    async def generate_native_response(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        config: LLMConfig | None = None,
    ) -> NativeResponse:
        self._track_call("generate_native_response", prompt, config, tools)
        return self._next_native_response()

    async def generate_structured_response(
        self,
        prompt: str,
        response_schema: dict[str, Any] | type,
        config: LLMConfig,
    ) -> StructuredResponse:
        self._track_call("generate_structured_response", prompt, config)
        return self._next_structured_response()

    async def transcribe(self, audio_path: str) -> str:
        self.calls.append({"method": "transcribe", "audio_path": audio_path})
        self.call_count += 1
        return self._transcription_text

    def reset(self) -> None:
        """Reset all tracking state and response indices."""
        self.calls.clear()
        self.call_count = 0
        self.last_prompt = None
        self.last_config = None
        self.last_tools = None
        self._response_index = 0
        self._tool_response_index = 0
        self._structured_response_index = 0
        self._native_response_index = 0


class InMemoryHistory:
    """In-memory implementation of the MessageHistory protocol.

    Stores messages in a plain list. Useful for testing and simple
    applications that don't need persistence.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))

    def add_tool_call(self, tool_name: str, tool_id: str, parameters: dict) -> None:
        if self._messages and self._messages[-1].get("role") == "assistant":
            calls = self._messages[-1].get("tool_calls", [])
            calls.append(ToolCall(tool_name=tool_name, tool_id=tool_id, parameters=parameters))
            self._messages[-1]["tool_calls"] = calls

    def add_tool_result(self, tool_id: str, result: str | dict) -> None:
        self._messages.append(
            Message(
                role="tool",
                content=str(result) if isinstance(result, str) else str(result),
                tool_results=[{"tool_id": tool_id, "result": result}],
            )
        )

    def get_messages(self, limit: int | None = None) -> list[Message]:
        if limit is not None:
            return list(self._messages[-limit:])
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
