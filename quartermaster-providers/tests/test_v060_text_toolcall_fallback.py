"""v0.6.0 — client-side salvage of text-form ``<|tool_call|>`` blocks.

A vLLM / Ollama server launched WITHOUT ``--tool-call-parser <flavour>``
leaks the chat-template's literal ``<|tool_call|>...`` sentinels into
``message.content`` instead of converting them into structured
``tool_calls``. Before v0.6.0 the agent silently lost those calls and
exited thinking it had a final answer.

The fallback only fires when ``message.tool_calls`` is empty AND the
content holds a marker-delimited block. Servers that DO have the right
parser (every correctly-configured deployment) keep hitting the normal
structured path and don't touch this code.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import (
    OpenAIProvider,
    _coerce_text_tool_call_payload,
    _parse_text_form_tool_calls,
)


# ── Unit: _coerce_text_tool_call_payload shapes ──────────────────────


class TestCoercePayload:
    def test_name_arguments_shape(self) -> None:
        out = _coerce_text_tool_call_payload(
            '{"name": "list_orders", "arguments": {"status": "active"}}'
        )
        assert out == ("list_orders", {"status": "active"})

    def test_tool_name_parameters_shape(self) -> None:
        out = _coerce_text_tool_call_payload(
            '{"tool_name": "list_orders", "parameters": {"limit": 5}}'
        )
        assert out == ("list_orders", {"limit": 5})

    def test_nested_function_shape(self) -> None:
        out = _coerce_text_tool_call_payload('{"function": {"name": "f", "arguments": {"x": 1}}}')
        assert out == ("f", {"x": 1})

    def test_double_encoded_arguments_string(self) -> None:
        out = _coerce_text_tool_call_payload('{"name": "f", "arguments": "{\\"x\\": 1}"}')
        assert out == ("f", {"x": 1})

    def test_empty_arguments_defaults_to_empty_dict(self) -> None:
        out = _coerce_text_tool_call_payload('{"name": "f"}')
        assert out == ("f", {})

    def test_malformed_json_returns_none(self) -> None:
        assert _coerce_text_tool_call_payload("{not json") is None

    def test_missing_name_returns_none(self) -> None:
        assert _coerce_text_tool_call_payload('{"arguments": {}}') is None

    def test_non_dict_top_level_returns_none(self) -> None:
        assert _coerce_text_tool_call_payload('"just a string"') is None
        assert _coerce_text_tool_call_payload("[1, 2, 3]") is None


# ── Unit: _parse_text_form_tool_calls marker flavours ────────────────


class TestParseTextForm:
    def test_gemma_style_single_call(self) -> None:
        content = (
            "Here you go.\n"
            '<|tool_call|>{"name": "list_orders", "arguments": {"status": "active"}}<|tool_call|>\n'
            "Working on it."
        )
        calls, residual = _parse_text_form_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].tool_name == "list_orders"
        assert calls[0].parameters == {"status": "active"}
        # Residual must NOT carry the literal marker block — callers
        # stream the residual as the assistant's visible answer.
        assert "<|tool_call|>" not in residual
        assert "Here you go." in residual
        assert "Working on it." in residual

    def test_qwen_style_single_call(self) -> None:
        content = 'thinking...\n<tool_call>{"name": "f", "arguments": {"y": 2}}</tool_call>\n'
        calls, residual = _parse_text_form_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].tool_name == "f"
        assert "<tool_call>" not in residual

    def test_fireworks_style(self) -> None:
        content = '[TOOL_CALLS]{"name": "g", "arguments": {}}[/TOOL_CALLS]'
        calls, _ = _parse_text_form_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].tool_name == "g"

    def test_multiple_calls_same_marker(self) -> None:
        content = (
            '<|tool_call|>{"name": "a", "arguments": {}}<|tool_call|>'
            "some text between\n"
            '<|tool_call|>{"name": "b", "arguments": {"x": 1}}<|tool_call|>'
        )
        calls, residual = _parse_text_form_tool_calls(content)
        assert [c.tool_name for c in calls] == ["a", "b"]
        assert "some text between" in residual

    def test_unique_tool_ids_per_call(self) -> None:
        """Agent dispatch loops key by tool_id — if two salvaged calls
        in one turn share an id the runner would collapse them."""
        content = (
            '<|tool_call|>{"name": "a", "arguments": {}}<|tool_call|>'
            '<|tool_call|>{"name": "a", "arguments": {}}<|tool_call|>'
        )
        calls, _ = _parse_text_form_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].tool_id != calls[1].tool_id

    def test_only_first_matching_flavour_consumed(self) -> None:
        """If Gemma markers match, we don't also chop Qwen markers from
        the same string — one template at a time."""
        content = (
            '<|tool_call|>{"name": "g", "arguments": {}}<|tool_call|>'
            '<tool_call>{"name": "q", "arguments": {}}</tool_call>'
        )
        calls, residual = _parse_text_form_tool_calls(content)
        assert [c.tool_name for c in calls] == ["g"]
        # The Qwen-form block survives in the residual text — we DON'T
        # double-consume with different marker regex on the same pass.
        assert "<tool_call>" in residual

    def test_no_markers_returns_empty(self) -> None:
        calls, residual = _parse_text_form_tool_calls("plain old answer")
        assert calls == []
        assert residual == "plain old answer"

    def test_empty_input_returns_empty(self) -> None:
        calls, residual = _parse_text_form_tool_calls("")
        assert calls == []
        assert residual == ""

    def test_malformed_block_skipped(self) -> None:
        """A block whose JSON doesn't parse is silently dropped; the
        rest of the content stays intact."""
        content = (
            "<|tool_call|>not valid json at all<|tool_call|>"
            '<|tool_call|>{"name": "good", "arguments": {}}<|tool_call|>'
        )
        calls, _ = _parse_text_form_tool_calls(content)
        # Exactly the ONE valid block is salvaged.
        assert len(calls) == 1
        assert calls[0].tool_name == "good"


# ── Integration: plumbed into generate_native_response ───────────────


def _fake_response(content: str, tool_calls=None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _provider_with(client_mock: MagicMock) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-test")
    provider._client = client_mock
    return provider


class TestNativeResponseIntegration:
    def test_text_form_salvaged_when_structured_empty(self) -> None:
        """Gemma-style content + empty structured tool_calls → fallback
        fires, NativeResponse carries structured tool_calls."""
        content = (
            "I'll check.\n"
            '<|tool_call|>{"name": "list_orders", "arguments": {"status": "active"}}<|tool_call|>'
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response(content=content, tool_calls=None)
        )
        provider = _provider_with(client)

        config = LLMConfig(model="gemma-4-26b", provider="openai")
        resp = asyncio.run(provider.generate_native_response("hi", tools=None, config=config))

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "list_orders"
        assert resp.tool_calls[0].parameters == {"status": "active"}
        assert "<|tool_call|>" not in resp.text_content
        assert "I'll check." in resp.text_content

    def test_structured_tool_calls_take_precedence(self) -> None:
        """When the server DID emit structured tool_calls, we don't
        also scan content — no risk of duplicate calls."""
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "real_call"
        tc.function.arguments = '{"foo": "bar"}'
        content_with_junk = '<|tool_call|>{"name": "ghost", "arguments": {}}<|tool_call|>'
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response(content=content_with_junk, tool_calls=[tc])
        )
        provider = _provider_with(client)

        config = LLMConfig(model="gemma-4-26b", provider="openai")
        resp = asyncio.run(provider.generate_native_response("hi", tools=None, config=config))

        assert [c.tool_name for c in resp.tool_calls] == ["real_call"]
        # Text content is untouched (we only scan when structured was empty)
        assert resp.text_content == content_with_junk

    def test_no_markers_no_op(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response(content="plain answer", tool_calls=None)
        )
        provider = _provider_with(client)

        config = LLMConfig(model="gpt-4o", provider="openai")
        resp = asyncio.run(provider.generate_native_response("hi", tools=None, config=config))

        assert resp.tool_calls == []
        assert resp.text_content == "plain answer"


class TestToolParametersIntegration:
    """generate_tool_parameters (the generate_tool_calls path) gets the
    same salvage. Needed for legacy agent loops that use the single-shot
    tool-picking entry point."""

    def test_text_form_salvaged(self) -> None:
        content = '<|tool_call|>{"name": "f", "arguments": {}}<|tool_call|>'
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response(content=content, tool_calls=None)
        )
        provider = _provider_with(client)

        from quartermaster_providers.types import ToolDefinition

        tools: list[ToolDefinition] = [
            {"name": "f", "description": "", "input_schema": {"type": "object"}}
        ]
        config = LLMConfig(model="gemma-4-26b", provider="openai")
        resp = asyncio.run(provider.generate_tool_parameters("hi", tools, config))

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "f"
        assert "<|tool_call|>" not in resp.text_content
