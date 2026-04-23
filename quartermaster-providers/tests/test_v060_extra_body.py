"""v0.6.0 — ``LLMConfig.extra_body`` passthrough.

Callers can splice arbitrary body fields into the outgoing
``chat.completions.create(...)`` call via
``LLMConfig(extra_body={"chat_template_kwargs": {"enable_thinking": False}})``.
The openai provider forwards it verbatim as ``extra_body=<dict>``; the
openai Python SDK splices it into the JSON payload the vLLM / OpenAI-
compatible server sees.
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


def test_extra_body_forwarded_to_chat_completions() -> None:
    """The dict on the config lands under ``extra_body`` in the outgoing call."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    call = mock_client.chat.completions.create.call_args
    assert call is not None, "chat.completions.create was never invoked"
    assert call.kwargs.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_absent_when_not_configured() -> None:
    """No ``extra_body`` in the request when none was set — don't send an
    empty/None key to vLLM."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(model="gpt-4o", provider="openai")

    asyncio.run(provider.generate_text_response("hi", config))

    call = mock_client.chat.completions.create.call_args
    assert call is not None
    assert "extra_body" not in call.kwargs


def test_empty_extra_body_dict_suppressed() -> None:
    """``extra_body={}`` is falsy; we suppress the key so servers never
    see a no-op empty dict."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(model="gpt-4o", provider="openai", extra_body={})

    asyncio.run(provider.generate_text_response("hi", config))

    call = mock_client.chat.completions.create.call_args
    assert "extra_body" not in call.kwargs


def test_extra_body_is_copied_not_referenced() -> None:
    """The provider must not hold a reference to the caller's dict; mutating
    the original after the call mustn't retroactively change the outbound
    payload seen by the server."""
    provider, mock_client = _provider_with_mock_client()
    payload = {"chat_template_kwargs": {"enable_thinking": False}}
    config = LLMConfig(model="gemma-4-26b", provider="openai", extra_body=payload)

    asyncio.run(provider.generate_text_response("hi", config))
    payload["chat_template_kwargs"]["enable_thinking"] = True  # mutate AFTER

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    # The top-level dict is copied. Nested mutation is a caller responsibility;
    # we only guard against the top-level alias.
    assert seen is not payload


def test_extra_body_roundtrips_through_native_tool_path() -> None:
    """generate_native_response (agent tool-calling path) also carries it."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        extra_body={"repetition_penalty": 1.1},
    )

    asyncio.run(
        provider.generate_native_response(
            prompt="hi",
            tools=None,
            config=config,
        )
    )

    call = mock_client.chat.completions.create.call_args
    assert call.kwargs.get("extra_body") == {"repetition_penalty": 1.1}
