"""v0.8.0 — ``history`` kwarg on every provider method splices into
the outbound ``messages`` array as proper multi-turn turns.

Pre-v0.8.0 wire format collapsed every prior turn into a single
``role="user"`` blob. This file locks down the new shape:

    [system, *history (one msg each), {role:user, content:prompt}]

— so the LLM can distinguish "this came from Agent1" (a separate
``user`` turn) from "this is the current question to act on" (the
trailing ``user`` turn).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import OpenAIProvider


def _fake_response(content: str = "ok") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _provider() -> tuple[OpenAIProvider, MagicMock]:
    p = OpenAIProvider(api_key="sk-test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response())
    p._client = client
    return p, client


class TestHistoryAbsentMatchesPreV080:
    """When ``history=None`` (the default), the wire format is identical
    to pre-v0.8.0 — single user message after the system prompt. No
    silent break for callers that haven't started using history yet."""

    def test_no_history_kwarg_yields_single_user_message(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai", system_message="be brief")
        asyncio.run(provider.generate_text_response("hello", config))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ]


class TestHistorySplicedBetweenSystemAndUser:
    """``history`` entries appear between the system message and the
    trailing user message, in order, each as its own ``{role, content}``
    object — never concatenated into one bigger user message."""

    def test_two_history_turns_become_two_messages(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai", system_message="sys")
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        asyncio.run(provider.generate_text_response("next question", config, history=history))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "next question"},
        ]

    def test_history_order_preserved(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [
            {"role": "user", "content": "msg-1"},
            {"role": "assistant", "content": "msg-2"},
            {"role": "user", "content": "msg-3"},
            {"role": "assistant", "content": "msg-4"},
        ]
        asyncio.run(provider.generate_text_response("msg-5", config, history=history))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        # Strip system message (none configured); compare the rest.
        contents = [m["content"] for m in messages]
        assert contents == ["msg-1", "msg-2", "msg-3", "msg-4", "msg-5"]

    def test_history_role_alternation_preserved(self) -> None:
        """Sora chat pattern: real user/assistant alternation through
        the history, not flattened to ``[role-name]: text`` strings."""
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [
            {"role": "user", "content": "/stranka PIGO"},
            {"role": "assistant", "content": "PIGO d.o.o."},
            {"role": "user", "content": "in status?"},
            {"role": "assistant", "content": "Naročilo SO-123 je..."},
        ]
        asyncio.run(provider.generate_text_response("hvala", config, history=history))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        # No system message → straight alternation + the trailing user.
        assert roles == ["user", "assistant", "user", "assistant", "user"]


class TestHistoryDefensiveAgainstMalformedEntries:
    """The provider's history splicer is defensive — bad entries don't
    crash the request, they just get dropped. The engine's history
    builder is the canonical source, but a typo from a hand-rolled
    caller shouldn't take the whole turn down."""

    def test_missing_role_skipped(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [
            {"content": "no role here"},
            {"role": "user", "content": "valid"},
        ]
        asyncio.run(provider.generate_text_response("now", config, history=history))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert "no role here" not in contents
        assert "valid" in contents

    def test_unknown_role_skipped(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [
            {"role": "tool", "content": "tool message — not user/assistant"},
            {"role": "user", "content": "valid"},
        ]
        asyncio.run(provider.generate_text_response("now", config, history=history))

        contents = [
            m["content"] for m in client.chat.completions.create.call_args.kwargs["messages"]
        ]
        assert "tool message — not user/assistant" not in contents
        assert "valid" in contents

    def test_missing_content_skipped(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [
            {"role": "user"},
            {"role": "user", "content": "valid"},
        ]
        asyncio.run(provider.generate_text_response("now", config, history=history))

        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        # Only the valid history entry + the trailing prompt should be there.
        assert [m["content"] for m in msgs if m["role"] == "user"] == ["valid", "now"]


class TestHistoryAcrossEveryGenerateMethod:
    """Every ``generate_*`` / ``stream_native_response`` accepts and
    forwards the ``history`` kwarg. Locked down here so a future provider
    refactor can't silently regress one of them."""

    def test_generate_native_response_forwards_history(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [{"role": "user", "content": "h1"}]
        asyncio.run(provider.generate_native_response("now", None, config, history=history))
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert {"role": "user", "content": "h1"} in msgs

    def test_generate_tool_parameters_forwards_history(self) -> None:
        provider, client = _provider()
        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [{"role": "user", "content": "h1"}]
        tools = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
        asyncio.run(provider.generate_tool_parameters("now", tools, config, history=history))
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert {"role": "user", "content": "h1"} in msgs

    def test_stream_native_response_forwards_history(self) -> None:
        """The streaming path also splices history. Use a fake stream
        that yields one chunk so the wire-format assertion stands."""
        provider = OpenAIProvider(api_key="sk-test")
        delta = MagicMock()
        delta.content = "ok"
        delta.tool_calls = None
        delta.reasoning = ""
        delta.reasoning_content = ""
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = "stop"
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = None

        class _Iter:
            def __init__(self) -> None:
                self._done = False

            def __aiter__(self) -> "_Iter":
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                return chunk

            async def aclose(self) -> None:
                pass

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_Iter())
        provider._client = client

        config = LLMConfig(model="gpt-4o", provider="openai")
        history = [{"role": "user", "content": "history-turn"}]
        asyncio.run(
            provider.stream_native_response("now", None, config, on_token=None, history=history)
        )

        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert {"role": "user", "content": "history-turn"} in msgs
