"""v0.7.1 — ``stream_native_response`` on the OpenAI provider.

The agent loop needs the full response (text + tool_calls) atomically to
decide whether to dispatch tools or finalise. The pre-v0.7.1 path used
``generate_native_response`` (non-streaming), which meant the entire
visible-text reply landed as ONE big ``TokenGenerated`` event at end of
turn — chat UIs saw a long pause then a wall of text instead of
incremental tokens.

``stream_native_response`` solves it: visible-text chunks flow to
``on_token`` as they arrive, while tool_call deltas (which OpenAI splits
across many chunks) accumulate per ``index`` and surface only on the
returned :class:`NativeResponse`. The agent loop gets atomic dispatch
AND the UI gets streaming tokens.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import OpenAIProvider

# ── Helpers: build fake openai SSE chunks ────────────────────────────


def _content_delta(content: str, finish_reason: str | None = None) -> MagicMock:
    """Build a chunk that carries a visible-text fragment."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    # Reasoning fields default to falsy; provider's _extract_reasoning_text
    # returns "" so they don't accidentally double-fire.
    delta.reasoning = ""
    delta.reasoning_content = ""
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _tool_call_delta(
    index: int,
    *,
    tc_id: str | None = None,
    name: str | None = None,
    args_fragment: str | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Build a chunk that carries a partial tool_call delta."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = args_fragment
    tc_delta = MagicMock()
    tc_delta.index = index
    tc_delta.id = tc_id
    tc_delta.function = fn
    delta = MagicMock()
    delta.content = ""
    delta.tool_calls = [tc_delta]
    delta.reasoning = ""
    delta.reasoning_content = ""
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _usage_chunk(input_tokens: int, output_tokens: int) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = []
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    chunk.usage = usage
    return chunk


class _AsyncIter:
    """Minimal async-iterator over a fixed list of chunks. Exposes
    ``aclose`` so the provider's ``_aclose_stream`` no-op cleanup runs
    without warnings."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "_AsyncIter":
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self) -> None:
        pass


def _provider_with_chunks(chunks: list[Any]) -> tuple[OpenAIProvider, MagicMock]:
    provider = OpenAIProvider(api_key="sk-test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_AsyncIter(chunks))
    provider._client = client
    return provider, client


# ── Tests ────────────────────────────────────────────────────────────


class TestVisibleTextStreaming:
    """Visible-text deltas must reach ``on_token`` in order, and the
    final ``NativeResponse.text_content`` must be the full concatenation."""

    def test_text_chunks_flow_to_on_token_in_order(self) -> None:
        chunks = [
            _content_delta("Hello"),
            _content_delta(", "),
            _content_delta("world"),
            _content_delta("!", finish_reason="stop"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        seen: list[str] = []
        result = asyncio.run(
            provider.stream_native_response("hi", tools=None, config=config, on_token=seen.append)
        )

        assert seen == ["Hello", ", ", "world", "!"]
        assert result.text_content == "Hello, world!"
        assert result.stop_reason == "stop"
        assert result.tool_calls == []

    def test_streaming_request_has_stream_true(self) -> None:
        """The provider must force ``stream=True`` regardless of
        ``config.stream`` (which the agent executor leaves False)."""
        chunks = [_content_delta("x", finish_reason="stop")]
        provider, client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai", stream=False)

        asyncio.run(provider.stream_native_response("hi", None, config, on_token=None))

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs.get("stream_options") == {"include_usage": True}

    def test_on_token_optional(self) -> None:
        """``on_token=None`` must still produce a valid response — the
        executor falls back to this path when nothing wants the
        stream."""
        chunks = [
            _content_delta("ok"),
            _content_delta(".", finish_reason="stop"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", None, config, on_token=None))

        assert result.text_content == "ok."


class TestToolCallAssembly:
    """OpenAI splits large ``arguments`` strings across many deltas. We
    accumulate per-``index`` and ``json.loads`` once at end-of-stream."""

    def test_single_tool_call_assembled_from_many_arg_fragments(self) -> None:
        chunks = [
            _tool_call_delta(0, tc_id="call_abc", name="get_weather"),
            _tool_call_delta(0, args_fragment='{"city"'),
            _tool_call_delta(0, args_fragment=': "Zagreb"'),
            _tool_call_delta(0, args_fragment=', "units": "C"}'),
            _content_delta("", finish_reason="tool_calls"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", tools=None, config=config))

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_id == "call_abc"
        assert tc.tool_name == "get_weather"
        assert tc.parameters == {"city": "Zagreb", "units": "C"}

    def test_multiple_tool_calls_kept_separate_by_index(self) -> None:
        """Two parallel tool calls in one turn — assembled independently
        even though their deltas interleave."""
        chunks = [
            _tool_call_delta(0, tc_id="call_a", name="search"),
            _tool_call_delta(1, tc_id="call_b", name="fetch"),
            _tool_call_delta(0, args_fragment='{"q": "abc"}'),
            _tool_call_delta(1, args_fragment='{"url"'),
            _tool_call_delta(1, args_fragment=': "x"}'),
            _content_delta("", finish_reason="tool_calls"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", None, config))

        assert [tc.tool_name for tc in result.tool_calls] == ["search", "fetch"]
        assert [tc.parameters for tc in result.tool_calls] == [
            {"q": "abc"},
            {"url": "x"},
        ]

    def test_text_and_tool_call_in_same_response(self) -> None:
        """Text deltas and tool_call deltas can be interleaved — both
        surface correctly in the final response."""
        chunks = [
            _content_delta("Calling search"),
            _tool_call_delta(0, tc_id="call_x", name="search"),
            _tool_call_delta(0, args_fragment='{"q":"foo"}'),
            _content_delta("", finish_reason="tool_calls"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        seen: list[str] = []
        result = asyncio.run(
            provider.stream_native_response("hi", None, config, on_token=seen.append)
        )

        assert seen == ["Calling search"]
        assert result.text_content == "Calling search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].parameters == {"q": "foo"}

    def test_malformed_arguments_stashed_under_raw(self) -> None:
        """If the assembled ``arguments`` string isn't valid JSON, we
        stash it under ``parameters["raw"]`` instead of dropping the
        tool call silently — matches generate_native_response's
        behaviour."""
        chunks = [
            _tool_call_delta(0, tc_id="call_x", name="bad", args_fragment="{not json"),
            _content_delta("", finish_reason="tool_calls"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", None, config))

        assert result.tool_calls[0].parameters == {"raw": "{not json"}


class TestUsageCapture:
    def test_usage_chunk_populates_response(self) -> None:
        """The trailing ``include_usage`` chunk arrives after content;
        we capture it onto the final NativeResponse."""
        chunks = [
            _content_delta("hi", finish_reason="stop"),
            _usage_chunk(input_tokens=42, output_tokens=7),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", None, config))

        assert result.usage is not None
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 7


class TestTextFormToolCallSalvage:
    """When a mis-configured vLLM / Ollama emits ``<|tool_call|>``
    markers in plain text instead of structured tool_calls, we still
    recover them post-stream — same salvage path as
    ``generate_native_response``."""

    def test_text_form_block_promoted_to_tool_calls(self) -> None:
        chunks = [
            _content_delta('<|tool_call|>{"name": "ping", "arguments": {}}<|tool_call|>'),
            _content_delta("", finish_reason="stop"),
        ]
        provider, _client = _provider_with_chunks(chunks)
        config = LLMConfig(model="gpt-4o", provider="openai")

        result = asyncio.run(provider.stream_native_response("hi", None, config))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "ping"
        # Text content is the residual after the markers are stripped.
        assert "<|tool_call|>" not in result.text_content


class TestDefaultBaseImplementation:
    """``AbstractLLMProvider.stream_native_response`` provides a
    no-op streaming shim — it just delegates to
    ``generate_native_response`` and emits the whole text at the end.
    Used by every provider that hasn't been migrated yet."""

    def test_base_impl_emits_full_text_via_on_token(self) -> None:
        from quartermaster_providers.testing import MockProvider
        from quartermaster_providers.types import NativeResponse, ToolCall

        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="hello world",
                    thinking=[],
                    tool_calls=[ToolCall(tool_name="t", tool_id="x", parameters={"a": 1})],
                    stop_reason="stop",
                )
            ]
        )
        config = LLMConfig(model="m", provider="mock")

        seen: list[str] = []
        result = asyncio.run(mock.stream_native_response("hi", None, config, on_token=seen.append))

        # Base shim emits the full text exactly once — UI experience
        # matches the pre-v0.7.1 behaviour for providers that haven't
        # overridden the streaming path.
        assert seen == ["hello world"]
        assert result.text_content == "hello world"
        assert len(result.tool_calls) == 1
