"""``thinking_enabled`` auto-merges into Gemma-4 / vLLM's
``chat_template_kwargs.enable_thinking`` so users don't have to hand-
splice ``extra_body`` for the common case.

Rules:
- When ``thinking_enabled=True`` and no ``chat_template_kwargs.enable_thinking``
  override is present, both OpenAI and OpenAI-compat inject
  ``{"enable_thinking": True}``.
- When ``thinking_enabled=False``, **OpenAI-compat** (vLLM / Ollama /
  LM Studio) injects ``{"enable_thinking": False}`` because those chat
  templates default thinking ON. Official ``OpenAIProvider`` does **not**
  inject on False — that would attach extra_body to every ChatGPT call.
- When the caller has already set ``enable_thinking`` explicitly in
  ``extra_body.chat_template_kwargs``, we never overwrite.
- The auto-merge must not mutate the caller's ``extra_body`` dict.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider


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


def _compat_provider_with_mock_client() -> tuple[OpenAICompatibleProvider, MagicMock]:
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",
        api_key="sk-test",
        provider_name="vllm",
    )
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
    """Official OpenAI: thinking_enabled defaults to False → no auto-merge.
    Caller extra_body is forwarded as-is; we do not inject enable_thinking.
    """
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(
        model="gemma-4-26b",
        provider="openai",
        extra_body={"repetition_penalty": 1.1},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"repetition_penalty": 1.1}


def test_openai_thinking_disabled_does_not_inject_extra_body() -> None:
    """Official OpenAI must not attach extra_body to every default
    ``thinking_level=off`` / ``thinking_enabled=False`` ChatGPT request."""
    provider, mock_client = _provider_with_mock_client()
    config = LLMConfig(model="gpt-4o", provider="openai", thinking_enabled=False)

    asyncio.run(provider.generate_text_response("hi", config))

    call = mock_client.chat.completions.create.call_args
    assert call is not None
    assert "extra_body" not in call.kwargs


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


def test_compat_thinking_disabled_injects_enable_thinking_false() -> None:
    """vLLM / OpenAI-compat: thinking_enabled=False → enable_thinking=false
    so Qwen3.6-style templates actually turn thinking off."""
    provider, mock_client = _compat_provider_with_mock_client()
    config = LLMConfig(
        model="Qwen3.6-27B",
        provider="vllm",
        thinking_enabled=False,
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": False}}


def test_compat_thinking_disabled_merges_with_existing_extra_body() -> None:
    """Unrelated extra_body keys survive the off → enable_thinking=false merge."""
    provider, mock_client = _compat_provider_with_mock_client()
    config = LLMConfig(
        model="Qwen3.6-27B",
        provider="vllm",
        thinking_enabled=False,
        extra_body={"repetition_penalty": 1.1},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_compat_explicit_enable_thinking_override_wins_when_off() -> None:
    """Caller extra_body.enable_thinking is never overwritten, even when
    thinking_enabled=False would otherwise inject false."""
    provider, mock_client = _compat_provider_with_mock_client()
    config = LLMConfig(
        model="Qwen3.6-27B",
        provider="vllm",
        thinking_enabled=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    asyncio.run(provider.generate_text_response("hi", config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": True}}


def test_compat_thinking_disabled_does_not_mutate_caller_dict() -> None:
    """The off-mapping must not mutate the caller's extra_body dict."""
    provider, mock_client = _compat_provider_with_mock_client()
    original = {"chat_template_kwargs": {"add_generation_prompt": True}}
    nested = original["chat_template_kwargs"]
    config = LLMConfig(
        model="Qwen3.6-27B",
        provider="vllm",
        thinking_enabled=False,
        extra_body=original,
    )

    asyncio.run(provider.generate_text_response("hi", config))

    assert original == {"chat_template_kwargs": {"add_generation_prompt": True}}
    assert nested == {"add_generation_prompt": True}
    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
    }


def test_compat_thinking_disabled_roundtrips_through_native_tool_path() -> None:
    """Agent tool-calling path on OpenAI-compat also injects enable_thinking=false."""
    provider, mock_client = _compat_provider_with_mock_client()
    config = LLMConfig(
        model="Qwen3.6-27B",
        provider="vllm",
        thinking_enabled=False,
    )

    asyncio.run(provider.generate_native_response(prompt="hi", tools=None, config=config))

    seen = mock_client.chat.completions.create.call_args.kwargs.get("extra_body")
    assert seen == {"chat_template_kwargs": {"enable_thinking": False}}
