"""v0.4.0 — native ``/api/chat`` transport for OllamaProvider.

These tests guard the v0.4.0 regression surface:

* Capability detection via ``/api/tags`` must route tool-calling
  requests through ``/api/chat`` (NOT ``/v1/chat/completions``) when
  the model supports native tools.
* Models without native tool support must keep working through the
  OpenAI-compat shim (the pre-v0.4.0 path).
* Tool-call responses from the native path must parse into clean
  ``ToolCallResponse`` / ``NativeResponse`` objects with no
  ``default_api:`` prefix on the tool name — that prefix was the
  whole reason the native path got built in the first place.
* Streaming NDJSON from ``/api/chat`` emits ``TokenResponse`` deltas
  that callers can iterate.
* ``LLMConfig.images`` flows into Ollama's native ``images: [base64]``
  sibling on the user message (not OpenAI's ``image_url`` content-part
  wrapping).
* The explicit ``tool_protocol="openai_compat"`` escape hatch forces
  the compat path even when the model would otherwise take native.

``openai`` is imported eagerly (same reason as ``test_ollama_chat.py``)
to get ``openai._DefaultHttpxClient`` materialised before we start
monkeypatching ``httpx.AsyncClient`` / ``httpx.Client``.
"""

from __future__ import annotations

import json

import openai  # noqa: F401  — eager import; see module docstring

import httpx
import pytest

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.ollama import (
    VALID_TOOL_PROTOCOLS,
    OllamaNativeProvider,
    model_supports_native_tools,
)
from quartermaster_providers.types import TokenResponse


# ── httpx fixtures ─────────────────────────────────────────────────────


class _RequestRecorder:
    """Captures outbound httpx requests so tests can assert routing."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.payloads: list[dict] = []

    def record(self, request: httpx.Request) -> None:
        self.requests.append(request)
        try:
            self.payloads.append(json.loads(request.content) if request.content else {})
        except json.JSONDecodeError:
            self.payloads.append({})

    def last_url(self) -> str:
        return str(self.requests[-1].url)

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


@pytest.fixture
def recorder() -> _RequestRecorder:
    return _RequestRecorder()


@pytest.fixture
def patch_async_httpx(monkeypatch, recorder):
    """Install a mock transport on both ``httpx.AsyncClient`` and ``httpx.Client``.

    ``OllamaNativeProvider._fetch_tags`` uses a sync ``httpx.Client``
    (the capability probe can safely block for a few hundred ms) while
    ``_post_native`` / ``_stream_native`` use ``httpx.AsyncClient``,
    so the fixture patches both.

    Takes a dict ``responses`` keyed on URL suffix (e.g. ``"/api/tags"``,
    ``"/api/chat"``) returning httpx.Response objects or callables that
    produce them.  Falls back to 404 for unmapped URLs.
    """

    def install(responses: dict[str, object]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            """Sync handler — httpx.MockTransport calls it from both sync and async paths."""
            recorder.record(request)
            path = request.url.path
            for suffix, responder in responses.items():
                if path.endswith(suffix):
                    if callable(responder):
                        return responder(request)
                    return responder
            return httpx.Response(404, text=f"no mock for {path}")

        transport = httpx.MockTransport(handler)
        original_async = httpx.AsyncClient

        def _async_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original_async(*args, **kwargs)

        original_sync = httpx.Client

        def _sync_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", _async_factory)
        monkeypatch.setattr(httpx, "Client", _sync_factory)

    return install


# ── tag fixtures ──────────────────────────────────────────────────────

TAGS_WITH_TOOL_SUPPORT = {
    "models": [
        {
            "name": "gemma4:26b",
            "model": "gemma4:26b",
            "capabilities": ["completion", "tools"],
            "details": {"family": "gemma"},
        }
    ]
}

TAGS_WITHOUT_TOOL_SUPPORT = {
    "models": [
        {
            "name": "llama2:7b",
            "model": "llama2:7b",
            "capabilities": ["completion"],
            "details": {"family": "llama"},
        }
    ]
}

TAGS_FAMILY_ONLY = {
    "models": [
        {
            "name": "qwen2.5:7b",
            "model": "qwen2.5:7b",
            # No ``capabilities`` list — older Ollama — so we fall
            # through to the family-name heuristic.
            "details": {"family": "qwen"},
        }
    ]
}


# ── Test 1: native path used when model supports tools ────────────────


class TestNativePathUsedWhenModelSupportsTools:
    """Capability probe says tools supported → /api/chat, not /v1/."""

    def test_native_path_hit_for_tool_capable_model(self, patch_async_httpx, recorder) -> None:
        chat_body = {
            "model": "gemma4:26b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_orders",
                            "arguments": {"status": "open"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 8,
        }
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
                "/api/chat": httpx.Response(200, json=chat_body),
            }
        )

        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        config = LLMConfig(
            model="gemma4:26b",
            provider="ollama",
            temperature=0.1,
            max_output_tokens=64,
        )
        tools = [
            {
                "name": "list_orders",
                "description": "List recent orders",
                "input_schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                },
            }
        ]

        import asyncio

        result = asyncio.run(provider.generate_tool_parameters("show orders", tools, config))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "list_orders"
        assert result.tool_calls[0].parameters == {"status": "open"}

        # At least one request must have gone to /api/chat — the native
        # endpoint.  None may target /v1/chat/completions.
        urls = recorder.urls()
        assert any(u.endswith("/api/chat") for u in urls), urls
        assert not any("/v1/chat/completions" in u for u in urls), urls


# ── Test 2: fallback to OpenAI-compat when model lacks native tools ────


class TestFallbackToOpenAICompat:
    """Capability probe says NO tool support → /v1/chat/completions."""

    def test_fallback_path_for_old_model(self, patch_async_httpx, recorder, monkeypatch) -> None:
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITHOUT_TOOL_SUPPORT),
            }
        )

        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")

        # We patch the inherited super().generate_tool_parameters so we
        # can detect which path the provider took without actually
        # firing through the OpenAI SDK (which needs a full mock of
        # its own).
        fallback_called = False

        async def _fake_super_tool(*args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            from quartermaster_providers.types import ToolCallResponse

            return ToolCallResponse(text_content="compat-fallback", tool_calls=[])

        from quartermaster_providers.providers.local import (
            OllamaProvider as _CompatOllama,
        )

        monkeypatch.setattr(
            _CompatOllama,
            "generate_tool_parameters",
            _fake_super_tool,
        )

        config = LLMConfig(model="llama2:7b", provider="ollama", temperature=0.1)
        tools = [
            {
                "name": "noop",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        import asyncio

        result = asyncio.run(provider.generate_tool_parameters("hi", tools, config))

        assert fallback_called is True
        assert result.text_content == "compat-fallback"
        # /api/chat must NOT have been hit on the auto fallback.
        assert not any(str(r.url).endswith("/api/chat") for r in recorder.requests)


# ── Test 3: tool-call response parsed cleanly (no default_api: prefix) ─


class TestNativeToolCallParsing:
    """Native responses yield bare tool names — no ``default_api:`` prefix."""

    def test_tool_name_has_no_default_api_prefix(self, patch_async_httpx, recorder) -> None:
        chat_body = {
            "model": "gemma4:26b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_customer_summary",
                            "arguments": {"customer_id": "C-42"},
                        }
                    },
                    {
                        "id": "call_abc",
                        "function": {
                            "name": "list_orders",
                            "arguments": {"limit": 10},
                        },
                    },
                ],
            },
            "done": True,
            "prompt_eval_count": 12,
            "eval_count": 6,
        }
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
                "/api/chat": httpx.Response(200, json=chat_body),
            }
        )

        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="native",  # force native for this assertion
        )
        config = LLMConfig(model="gemma4:26b", provider="ollama")
        tools = [
            {
                "name": "get_customer_summary",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "list_orders",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

        import asyncio

        result = asyncio.run(provider.generate_tool_parameters("show customer", tools, config))

        # No ``default_api:`` prefix on any returned tool name — the
        # whole reason this module exists.
        for call in result.tool_calls:
            assert not call.tool_name.startswith("default_api"), call.tool_name
            assert not call.tool_name.startswith("functions"), call.tool_name
        assert [c.tool_name for c in result.tool_calls] == [
            "get_customer_summary",
            "list_orders",
        ]
        # Usage / stop reason must be populated.
        assert result.usage is not None
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 6


# ── Test 4: streaming yields TokenResponse deltas ─────────────────────


class TestNativeStreaming:
    """NDJSON streaming from /api/chat → TokenResponse chunks."""

    def test_streaming_yields_token_response_deltas(self, patch_async_httpx, recorder) -> None:
        ndjson_lines = [
            json.dumps({"message": {"role": "assistant", "content": "Hel"}}),
            json.dumps({"message": {"role": "assistant", "content": "lo "}}),
            json.dumps({"message": {"role": "assistant", "content": "world"}}),
            json.dumps(
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                }
            ),
        ]
        stream_body = "\n".join(ndjson_lines)

        patch_async_httpx(
            {
                "/api/chat": httpx.Response(
                    200,
                    text=stream_body,
                    headers={"content-type": "application/x-ndjson"},
                ),
            }
        )

        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="native",
        )
        config = LLMConfig(
            model="gemma4:26b",
            provider="ollama",
            stream=True,
        )

        import asyncio

        async def collect():
            gen = await provider.generate_text_response("hi", config)
            out: list[TokenResponse] = []
            async for chunk in gen:
                out.append(chunk)
            return out

        chunks = asyncio.run(collect())

        # We must see "Hel", "lo ", "world", then a final stop chunk.
        content_chunks = [c for c in chunks if c.content]
        assert [c.content for c in content_chunks] == ["Hel", "lo ", "world"]
        # The final chunk carries the done_reason.
        assert any(c.stop_reason == "stop" for c in chunks)


# ── Test 5: force protocol via config ─────────────────────────────────


class TestForceProtocolViaConfig:
    """``tool_protocol="openai_compat"`` MUST bypass the native path."""

    def test_openai_compat_forces_compat_even_when_native_supported(
        self, patch_async_httpx, recorder, monkeypatch
    ) -> None:
        patch_async_httpx(
            {
                # Even though /api/tags says native is supported, the
                # explicit override must win.
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
            }
        )

        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="openai_compat",
        )

        # The native path must not be used — we prove it by replacing
        # the super() hook and asserting it got called.
        compat_called = False

        async def _fake_super_tool(*args, **kwargs):
            nonlocal compat_called
            compat_called = True
            from quartermaster_providers.types import ToolCallResponse

            return ToolCallResponse(text_content="forced compat", tool_calls=[])

        from quartermaster_providers.providers.local import (
            OllamaProvider as _CompatOllama,
        )

        monkeypatch.setattr(
            _CompatOllama,
            "generate_tool_parameters",
            _fake_super_tool,
        )

        config = LLMConfig(model="gemma4:26b", provider="ollama")
        tools = [
            {
                "name": "anything",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        import asyncio

        asyncio.run(provider.generate_tool_parameters("hi", tools, config))
        assert compat_called is True
        # /api/chat must NOT have been hit.
        assert not any("/api/chat" in str(r.url) for r in recorder.requests)

    def test_invalid_tool_protocol_rejected(self):
        with pytest.raises(ValueError, match="Unknown tool_protocol"):
            OllamaNativeProvider(
                base_url="http://localhost:11434/v1",
                tool_protocol="whatever",
            )

    def test_valid_protocols_enumerated(self):
        assert {"auto", "native", "openai_compat"} == VALID_TOOL_PROTOCOLS


# ── Test 6: native vision uses images: [base64] field ─────────────────


class TestNativeVision:
    """``LLMConfig.images`` must ride as Ollama's top-level ``images`` list."""

    def test_images_flow_into_native_payload(self, patch_async_httpx, recorder) -> None:
        chat_body = {
            "model": "gemma4:26b",
            "message": {"role": "assistant", "content": "I see a cat."},
            "done": True,
        }
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
                "/api/chat": httpx.Response(200, json=chat_body),
            }
        )

        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="native",
        )
        config = LLMConfig(
            model="gemma4:26b",
            provider="ollama",
            images=[("FAKE_B64_PAYLOAD", "image/png")],
        )

        import asyncio

        asyncio.run(provider.generate_native_response("describe", config=config))

        # Find the /api/chat request.
        chat_requests = [
            (req, pl)
            for req, pl in zip(recorder.requests, recorder.payloads)
            if req.url.path.endswith("/api/chat")
        ]
        assert chat_requests, "expected at least one /api/chat request"
        _req, payload = chat_requests[-1]

        user_message = next(m for m in payload["messages"] if m["role"] == "user")
        # Native shape: ``images`` is a top-level list of bare base64
        # strings, NOT an OpenAI-style ``content: [{type: image_url...}]``.
        assert user_message["content"] == "describe"
        assert user_message["images"] == ["FAKE_B64_PAYLOAD"]
        # Ensure we did NOT wrap in OpenAI content-parts.
        assert not isinstance(user_message["content"], list)


# ── Additional guards ────────────────────────────────────────────────


class TestCapabilityDetection:
    """Family-name heuristic fallback when /api/tags lacks capabilities."""

    def test_family_heuristic_used_when_capabilities_absent(
        self, patch_async_httpx, recorder
    ) -> None:
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_FAMILY_ONLY),
            }
        )
        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        # qwen family is in the known-tool-capable set.
        assert provider._supports_native_tools("qwen2.5:7b") is True

    def test_llama2_rejected_even_with_llama_family(self, patch_async_httpx, recorder) -> None:
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITHOUT_TOOL_SUPPORT),
            }
        )
        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        # llama2:7b must be rejected even though the family string is "llama".
        assert provider._supports_native_tools("llama2:7b") is False

    def test_tags_endpoint_failure_falls_back_to_heuristic(
        self, patch_async_httpx, recorder
    ) -> None:
        """If /api/tags errors out we should still make a best-effort
        capability decision based on the model name itself."""
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(500, text="server error"),
            }
        )
        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        # mistral family is tool-capable.
        assert provider._supports_native_tools("mistral-nemo") is True
        # codellama is explicitly blacklisted.
        assert provider._supports_native_tools("codellama:13b") is False

    def test_cache_prevents_repeat_tags_fetches(self, patch_async_httpx, recorder) -> None:
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
            }
        )
        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        _ = provider._supports_native_tools("gemma4:26b")
        _ = provider._supports_native_tools("gemma4:26b")
        _ = provider._supports_native_tools("gemma4:26b")
        tag_hits = [r for r in recorder.requests if r.url.path.endswith("/api/tags")]
        assert len(tag_hits) == 1, "tags endpoint should only be hit once"

    def test_reset_cache_forces_refetch(self, patch_async_httpx, recorder) -> None:
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
            }
        )
        provider = OllamaNativeProvider(base_url="http://localhost:11434/v1")
        _ = provider._supports_native_tools("gemma4:26b")
        provider.reset_capability_cache()
        _ = provider._supports_native_tools("gemma4:26b")
        tag_hits = [r for r in recorder.requests if r.url.path.endswith("/api/tags")]
        assert len(tag_hits) == 2


class TestRegistryDefault:
    """``register_local("ollama")`` must wire up the native provider by default."""

    def test_register_local_returns_native_provider(self) -> None:
        from quartermaster_providers import ProviderRegistry

        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama")
        provider = reg.get("ollama")
        # New default is the native subclass.
        assert isinstance(provider, OllamaNativeProvider)

    def test_register_local_forwards_tool_protocol(self) -> None:
        from quartermaster_providers import ProviderRegistry

        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", tool_protocol="native")
        provider = reg.get("ollama")
        assert isinstance(provider, OllamaNativeProvider)
        assert provider.tool_protocol == "native"


class TestModuleLevelHelper:
    """``model_supports_native_tools`` convenience helper."""

    def test_helper_returns_bool(self, patch_async_httpx, recorder) -> None:
        # Clear the module-level lru_cache so the helper actually
        # fires a probe this test run.
        from quartermaster_providers.providers import ollama as _om

        _om._cached_model_supports_native_tools.cache_clear()

        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
            }
        )
        result = model_supports_native_tools("gemma4:26b", base_url="http://localhost:11434/v1")
        assert result is True


class TestThinkField:
    """``LLMConfig.thinking_enabled`` must forward as Ollama's ``think`` flag."""

    def test_thinking_enabled_emits_think_true(self, patch_async_httpx, recorder) -> None:
        chat_body = {
            "message": {"role": "assistant", "content": "done"},
            "done": True,
        }
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
                "/api/chat": httpx.Response(200, json=chat_body),
            }
        )
        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="native",
        )
        config = LLMConfig(
            model="gemma4:26b",
            provider="ollama",
            thinking_enabled=True,
        )

        import asyncio

        asyncio.run(provider.generate_native_response("hi", config=config))

        chat_reqs = [
            pl
            for req, pl in zip(recorder.requests, recorder.payloads)
            if req.url.path.endswith("/api/chat")
        ]
        assert chat_reqs
        assert chat_reqs[-1].get("think") is True

    def test_thinking_disabled_omits_think_field(self, patch_async_httpx, recorder) -> None:
        chat_body = {"message": {"content": "ok"}, "done": True}
        patch_async_httpx(
            {
                "/api/tags": httpx.Response(200, json=TAGS_WITH_TOOL_SUPPORT),
                "/api/chat": httpx.Response(200, json=chat_body),
            }
        )
        provider = OllamaNativeProvider(
            base_url="http://localhost:11434/v1",
            tool_protocol="native",
        )
        config = LLMConfig(
            model="gemma4:26b",
            provider="ollama",
            thinking_enabled=False,
        )

        import asyncio

        asyncio.run(provider.generate_native_response("hi", config=config))

        chat_reqs = [
            pl
            for req, pl in zip(recorder.requests, recorder.payloads)
            if req.url.path.endswith("/api/chat")
        ]
        assert chat_reqs
        # think=False intentionally NOT sent (stay forward/backward compatible).
        assert "think" not in chat_reqs[-1]


# ── SDK config integration ────────────────────────────────────────────


class TestSDKConfigPassThrough:
    """``qm.configure(ollama_tool_protocol=...)`` threads through the registry."""

    def test_valid_protocol_reaches_provider(self, monkeypatch) -> None:
        # Reset module state so the test is hermetic.
        from quartermaster_sdk import _config

        monkeypatch.setattr(_config, "_default_registry", None)
        monkeypatch.setattr(_config, "_default_model", None)

        reg = _config.configure(
            provider="ollama",
            base_url="http://localhost:11434",
            default_model="gemma4:26b",
            ollama_tool_protocol="native",
        )
        provider = reg.get("ollama")
        assert isinstance(provider, OllamaNativeProvider)
        assert provider.tool_protocol == "native"

    def test_invalid_protocol_rejected_in_configure(self, monkeypatch) -> None:
        from quartermaster_sdk import _config

        monkeypatch.setattr(_config, "_default_registry", None)
        monkeypatch.setattr(_config, "_default_model", None)

        with pytest.raises(ValueError, match="ollama_tool_protocol"):
            _config.configure(
                provider="ollama",
                base_url="http://localhost:11434",
                default_model="gemma4:26b",
                ollama_tool_protocol="definitely_not_valid",
            )
