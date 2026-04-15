"""Tests for ``OllamaProvider.chat`` — the v0.1.3 sync native-``/api/chat`` shim.

We don't depend on a running Ollama daemon: we monkeypatch ``httpx.Client`` to
return canned responses and assert the request shape + result parsing.

NOTE: ``openai`` is imported eagerly below so its ``_DefaultHttpxClient`` (a
``httpx.Client`` subclass) is built before our fixture monkeypatches
``httpx.Client``.  Without this, patching ``httpx.Client`` in a fixture and
then constructing an ``OllamaProvider`` (which lazy-imports openai) hits a
``TypeError: function() argument 'code' must be code, not str`` from openai
trying to subclass our patched function.
"""

from __future__ import annotations

import openai  # noqa: F401  — eager import; see module docstring above

import httpx
import pytest

from quartermaster_providers import ChatResult
from quartermaster_providers.exceptions import ProviderError, ServiceUnavailableError
from quartermaster_providers.providers.local import OllamaProvider, _strip_v1


class _MockTransport(httpx.MockTransport):
    """Tiny httpx mock that records the last request for assertion."""

    def __init__(self, responder):
        self.last_request = None
        self.last_payload: dict = {}
        super().__init__(self._wrap(responder))

    def _wrap(self, responder):
        def handle(request: httpx.Request) -> httpx.Response:
            self.last_request = request
            import json as _json

            self.last_payload = _json.loads(request.content) if request.content else {}
            return responder(request)

        return handle


@pytest.fixture
def patch_httpx_client(monkeypatch):
    """Replace ``httpx.Client`` with one wired to a mock transport."""

    transport_holder: dict[str, _MockTransport] = {}

    def install(responder):
        mock = _MockTransport(responder)
        transport_holder["transport"] = mock
        original = httpx.Client

        def _factory(*args, **kwargs):
            kwargs["transport"] = mock
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", _factory)
        return mock

    return install


# ── URL routing ────────────────────────────────────────────────────────


class TestStripV1:
    def test_strips_trailing_v1(self):
        assert _strip_v1("http://localhost:11434/v1") == "http://localhost:11434"

    def test_strips_v1_with_trailing_slash(self):
        assert _strip_v1("http://localhost:11434/v1/") == "http://localhost:11434"

    def test_leaves_bare_host_alone(self):
        assert _strip_v1("http://localhost:11434") == "http://localhost:11434"

    def test_preserves_corporate_gateway_path(self):
        """Regression: ``http://gateway/api/v1`` is a real-world prefix
        for corporate Ollama proxies.  We must NOT strip ``/v1`` and
        produce ``http://gateway/api`` (which would route ``/api/chat``
        requests to ``/api/api/chat`` and 404)."""
        assert _strip_v1("http://gateway.corp/api/v1") == "http://gateway.corp/api/v1"

    def test_preserves_v1beta(self):
        assert _strip_v1("http://api.example.com/v1beta") == "http://api.example.com/v1beta"

    def test_preserves_path_segment_named_v1(self):
        assert _strip_v1("http://h/v1/inner") == "http://h/v1/inner"


# ── chat() happy path ──────────────────────────────────────────────────


class TestChatHappyPath:
    def test_basic_text_response(self, patch_httpx_client):
        def responder(_req):
            return httpx.Response(
                200,
                json={
                    "model": "gemma4:26b",
                    "message": {"role": "assistant", "content": "Pozdravljen!"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                },
            )

        mock = patch_httpx_client(responder)
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        result = provider.chat(
            messages=[{"role": "user", "content": "Pozdravljen, koliko je ura?"}],
            model="gemma4:26b",
            temperature=0.3,
            max_output_tokens=128,
        )

        assert isinstance(result, ChatResult)
        assert result.content == "Pozdravljen!"
        assert result.tool_calls == []
        assert result.usage == {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        }
        assert result.stop_reason == "stop"

        # URL must be /api/chat (NOT /v1/chat/completions).
        assert str(mock.last_request.url).endswith("/api/chat")
        assert "/v1" not in str(mock.last_request.url)
        # Payload must carry options the user asked for.
        assert mock.last_payload["model"] == "gemma4:26b"
        assert mock.last_payload["stream"] is False
        assert mock.last_payload["options"] == {
            "temperature": 0.3,
            "num_predict": 128,
        }


class TestReasoningFallback:
    """gemma4:26b empties ``content`` on short prompts and stuffs the
    answer into ``thinking`` / ``reasoning``.  ``chat()`` must promote
    the first non-empty fallback so callers don't see an empty string."""

    def test_promotes_thinking_when_content_empty(self, patch_httpx_client):
        patch_httpx_client(
            lambda _r: httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "Trenutno je deset.",
                    },
                    "done": True,
                },
            )
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        result = provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")
        assert result.content == "Trenutno je deset."

    def test_promotes_reasoning_when_content_and_thinking_empty(self, patch_httpx_client):
        patch_httpx_client(
            lambda _r: httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "Reasoning text here.",
                    },
                    "done": True,
                },
            )
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        result = provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")
        assert result.content == "Reasoning text here."

    def test_empty_when_nothing_present(self, patch_httpx_client):
        patch_httpx_client(
            lambda _r: httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": ""}, "done": True},
            )
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        result = provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")
        assert result.content == ""


class TestToolCalls:
    def test_tool_calls_normalized(self, patch_httpx_client):
        patch_httpx_client(
            lambda _r: httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Ljubljana"},
                                }
                            }
                        ],
                    },
                    "done": True,
                },
            )
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        result = provider.chat(
            messages=[{"role": "user", "content": "weather?"}],
            model="gemma4:26b",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_weather"
        assert result.tool_calls[0].parameters == {"city": "Ljubljana"}


# ── error bubbling ────────────────────────────────────────────────────


class TestErrorBubbling:
    """Connection failures must raise (not return success=True with empty content)."""

    def test_connect_error_raises_service_unavailable(self, monkeypatch):
        def explode(*_a, **_k):
            raise httpx.ConnectError("connection refused")

        class _BoomClient:
            def __init__(self, *_a, **_k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def post(self, *_a, **_k):
                explode()

        monkeypatch.setattr(httpx, "Client", _BoomClient)
        provider = OllamaProvider(base_url="http://wrong:11434/v1")
        with pytest.raises(ServiceUnavailableError, match="Could not reach Ollama"):
            provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")

    def test_http_status_error_raises_provider_error(self, patch_httpx_client):
        patch_httpx_client(lambda _r: httpx.Response(404, text='{"error":"model not found"}'))
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        with pytest.raises(ProviderError, match="HTTP 404"):
            provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")

    @pytest.mark.parametrize(
        "exc_cls",
        [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout],
        ids=["connect", "read", "write", "pool"],
    )
    def test_all_timeout_subclasses_raise_service_unavailable(self, monkeypatch, exc_cls):
        """Regression for the v0.1.3 review: pre-fix only ``ReadTimeout``
        was caught; ``ConnectTimeout`` (server up but slow to accept the
        socket — common while a model is loading) escaped and surfaced as
        a generic ``ProviderError`` instead.  All timeout subclasses must
        now route through ``ServiceUnavailableError`` so callers can
        distinguish "Ollama unreachable" from other failures."""

        class _BoomClient:
            def __init__(self, *_a, **_k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def post(self, *_a, **_k):
                raise exc_cls("simulated timeout")

        monkeypatch.setattr(httpx, "Client", _BoomClient)
        provider = OllamaProvider(base_url="http://slow:11434/v1")
        with pytest.raises(ServiceUnavailableError, match="Could not reach Ollama"):
            provider.chat(messages=[{"role": "user", "content": "?"}], model="gemma4:26b")


# ── argument validation ───────────────────────────────────────────────


class TestArgValidation:
    def test_no_model_raises(self, patch_httpx_client):
        patch_httpx_client(lambda _r: httpx.Response(200, json={"message": {"content": "x"}}))
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        with pytest.raises(ProviderError, match="requires a model"):
            provider.chat(messages=[{"role": "user", "content": "?"}])

    def test_thinking_level_off_emits_false(self, patch_httpx_client):
        mock = patch_httpx_client(
            lambda _r: httpx.Response(200, json={"message": {"content": "x"}, "done": True})
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        provider.chat(
            messages=[{"role": "user", "content": "?"}],
            model="gemma4:26b",
            thinking_level="off",
        )
        assert mock.last_payload["think"] is False

    def test_thinking_level_high_emits_true(self, patch_httpx_client):
        mock = patch_httpx_client(
            lambda _r: httpx.Response(200, json={"message": {"content": "x"}, "done": True})
        )
        provider = OllamaProvider(base_url="http://localhost:11434/v1")
        provider.chat(
            messages=[{"role": "user", "content": "?"}],
            model="gemma4:26b",
            thinking_level="high",
        )
        assert mock.last_payload["think"] is True
