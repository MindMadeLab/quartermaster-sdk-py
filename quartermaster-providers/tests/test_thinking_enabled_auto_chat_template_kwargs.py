"""``thinking_enabled=True`` auto-merges into Gemma-4 / vLLM's
``chat_template_kwargs.enable_thinking`` so users don't have to hand-
splice ``extra_body`` for the common case.

Rules:
- When ``thinking_enabled=True`` and no ``chat_template_kwargs.enable_thinking``
  override is present, we inject ``{"enable_thinking": True}``.
- When the caller has already set ``enable_thinking`` explicitly in
  ``extra_body.chat_template_kwargs``, we never overwrite.
- When ``thinking_enabled`` is False / unset, we never touch ``extra_body``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import OpenAIProvider


def _fake_response() -> MagicMock:
    msg = MagicMock()
    msg.content = "ok"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _provider_with_mock_client() -> tuple[OpenAIProvider, MagicMock]:
    provider = OpenAIProvider(api_key="sk-test")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_fake_response())
    provider._client = mock_client
    return provider, mock_client


def test_thinking_enabled_injects_chat_template_kwargs() -> None:
    """thinking_enabled=True with no extra_body → vLLM sees
    ``chat_template_kwargs.enable_thinking=True``."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": True}}


def test_thinking_enabled_merges_with_existing_extra_body() -> None:
    """Unrelated extra_body keys (like repetition_penalty) are preserved;
    chat_template_kwargs gains ``enable_thinking=True``."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
        extra_body={"repetition_penalty": 1.1},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_thinking_enabled_preserves_sibling_chat_template_kwargs() -> None:
    """Other chat_template_kwargs (like a custom system prompt toggle) are
    kept alongside the injected enable_thinking."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
        extra_body={"chat_template_kwargs": {"add_generation_prompt": True}},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
    }


def test_explicit_enable_thinking_override_wins() -> None:
    """Caller-set ``enable_thinking=False`` in extra_body is respected even
    if thinking_enabled=True — the explicit override wins."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_disabled_leaves_extra_body_untouched() -> None:
    """thinking_enabled defaults to False → the auto-merge never fires."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        extra_body={"repetition_penalty": 1.1},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"repetition_penalty": 1.1}


def test_thinking_enabled_does_not_mutate_caller_dict() -> None:
    """The auto-merge must not mutate the caller's extra_body dict — we
    copy before splicing."""
    provider, mock_client = _provider_with_mock_client()
    original = {"repetition_penalty": 1.1}
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
        extra_body=original,
    )

    asyncio.run(provider.generate_text_response("hi", config))

    assert original == {"repetition_penalty": 1.1}, (
        "Provider mutated caller's extra_body dict — the auto-merge must build a fresh dict."
    )


def test_thinking_enabled_roundtrips_through_native_tool_path() -> None:
    """generate_native_response (agent tool-calling path) applies the
    same auto-merge."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        thinking_enabled=True,
    )

    asyncio.run(provider.generate_native_response(prompt="hi", tools=None, config=config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": True}}
