"""Tests for MockProvider and InMemoryHistory testing utilities."""

import pytest

from qm_providers.config import LLMConfig
from qm_providers.testing import MockProvider
from qm_providers.types import (
    NativeResponse,
    StructuredResponse,
    ThinkingResponse,
    ToolCall,
    ToolCallResponse,
    TokenResponse,
)


class TestMockProvider:
    @pytest.fixture
    def config(self):
        return LLMConfig(model="mock", provider="mock")

    @pytest.mark.asyncio
    async def test_text_response(self, config):
        mock = MockProvider(responses=[TokenResponse(content="Hi!", stop_reason="end_turn")])
        resp = await mock.generate_text_response("Hello", config)
        assert resp.content == "Hi!"
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_text_response_cycling(self, config):
        mock = MockProvider(
            responses=[
                TokenResponse(content="First"),
                TokenResponse(content="Second"),
            ]
        )
        r1 = await mock.generate_text_response("a", config)
        r2 = await mock.generate_text_response("b", config)
        r3 = await mock.generate_text_response("c", config)
        assert r1.content == "First"
        assert r2.content == "Second"
        assert r3.content == "First"  # cycles back

    @pytest.mark.asyncio
    async def test_default_response(self, config):
        mock = MockProvider()
        resp = await mock.generate_text_response("test", config)
        assert resp.content == "Mock response"

    @pytest.mark.asyncio
    async def test_streaming(self, config):
        config.stream = True
        mock = MockProvider(
            responses=[TokenResponse(content="Hello World", stop_reason="end_turn")]
        )
        chunks = []
        async for chunk in await mock.generate_text_response("test", config):
            chunks.append(chunk)
        text = "".join(c.content for c in chunks)
        assert "Hello" in text
        assert "World" in text

    @pytest.mark.asyncio
    async def test_tool_response(self, config):
        tool_resp = ToolCallResponse(
            text_content="Let me check.",
            tool_calls=[
                ToolCall(
                    tool_name="search",
                    tool_id="c1",
                    parameters={"q": "python"},
                )
            ],
            stop_reason="tool_use",
        )
        mock = MockProvider(tool_responses=[tool_resp])
        resp = await mock.generate_tool_parameters(
            "search for python", [{"name": "search"}], config
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_native_response(self, config):
        native = NativeResponse(
            text_content="Analysis",
            thinking=[ThinkingResponse(thinking="Deep thought")],
            stop_reason="end_turn",
        )
        mock = MockProvider(native_responses=[native])
        resp = await mock.generate_native_response("analyze", config=config)
        assert resp.text_content == "Analysis"
        assert len(resp.thinking) == 1

    @pytest.mark.asyncio
    async def test_structured_response(self, config):
        struct = StructuredResponse(
            structured_output={"name": "Test"},
            raw_output='{"name": "Test"}',
        )
        mock = MockProvider(structured_responses=[struct])
        resp = await mock.generate_structured_response("extract", {}, config)
        assert resp.structured_output == {"name": "Test"}

    @pytest.mark.asyncio
    async def test_transcribe(self, config):
        mock = MockProvider(transcription_text="Transcribed audio")
        text = await mock.transcribe("/path/to/audio.wav")
        assert text == "Transcribed audio"

    @pytest.mark.asyncio
    async def test_list_models(self):
        mock = MockProvider(models=["m1", "m2"])
        models = await mock.list_models()
        assert models == ["m1", "m2"]

    def test_estimate_token_count(self):
        mock = MockProvider()
        count = mock.estimate_token_count("hello world foo", "mock")
        assert count == 3

    def test_prepare_tool(self):
        mock = MockProvider()
        tool = {"name": "fn", "description": "desc", "input_schema": {}}
        prepared = mock.prepare_tool(tool)
        assert prepared["name"] == "fn"

    @pytest.mark.asyncio
    async def test_call_tracking(self, config):
        mock = MockProvider()
        await mock.generate_text_response("prompt1", config)
        await mock.generate_text_response("prompt2", config)
        assert mock.call_count == 2
        assert mock.last_prompt == "prompt2"
        assert len(mock.calls) == 2
        assert mock.calls[0]["prompt"] == "prompt1"
        assert mock.calls[1]["method"] == "generate_text_response"

    @pytest.mark.asyncio
    async def test_reset(self, config):
        mock = MockProvider()
        await mock.generate_text_response("test", config)
        assert mock.call_count == 1
        mock.reset()
        assert mock.call_count == 0
        assert mock.last_prompt is None
        assert mock.calls == []


class TestInMemoryHistory:
    def test_add_and_get_messages(self, history):
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        msgs = history.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "Hi there!"

    def test_get_messages_with_limit(self, history):
        history.add_message("user", "msg1")
        history.add_message("assistant", "msg2")
        history.add_message("user", "msg3")
        msgs = history.get_messages(limit=2)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "msg2"
        assert msgs[1]["content"] == "msg3"

    def test_clear(self, history):
        history.add_message("user", "Hello")
        assert len(history) == 1
        history.clear()
        assert len(history) == 0

    def test_len(self, history):
        assert len(history) == 0
        history.add_message("user", "a")
        history.add_message("user", "b")
        assert len(history) == 2

    def test_add_tool_call(self, history):
        history.add_message("assistant", "Let me search.")
        history.add_tool_call("search", "call_1", {"q": "test"})
        msgs = history.get_messages()
        assert len(msgs) == 1
        assert len(msgs[0].get("tool_calls", [])) == 1

    def test_add_tool_result(self, history):
        history.add_tool_result("call_1", {"result": "found"})
        msgs = history.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"

    def test_get_messages_returns_copy(self, history):
        history.add_message("user", "Hello")
        msgs1 = history.get_messages()
        msgs2 = history.get_messages()
        assert msgs1 is not msgs2
