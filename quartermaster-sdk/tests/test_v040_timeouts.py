"""Tests for v0.4.0 application-level LLM timeouts + stream deadlines.

Sorex round-2 P1.1: today ``qm.configure(timeout=...)`` raises
TypeError and there's no application-level deadline — an Ollama
instance wedging mid-stream hangs the worker until Celery's blunt
``CELERY_TASK_TIME_LIMIT`` kills it. v0.4.0 adds:

* ``qm.configure(timeout=..., connect_timeout=..., read_timeout=...)``
* Per-call overrides on ``qm.run`` / ``qm.arun`` / their stream variants.
* ``deadline_seconds=`` on ``run.stream`` / ``arun.stream`` for a
  total wall-clock ceiling (independent of ``read_timeout``).

The tests mock the provider with :class:`MockProvider` so no network
calls fire. ``mock.last_config.connect_timeout`` / ``.read_timeout``
is the single source of truth we assert against — confirming the
SDK actually reaches the provider SDK layer.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai  # noqa: F401 — eager import, see test_v020_surface.py for context

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


# ── Helpers ───────────────────────────────────────────────────────────


def _mock_registry(
    text: str = "canned",
) -> tuple[ProviderRegistry, MockProvider]:
    """Build a ProviderRegistry with a MockProvider registered as ``ollama``."""
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=text,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            )
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg, mock


def _plain_graph() -> qm.GraphSpec:
    """Simple single-node instruction graph exercising LLMExecutor."""
    return (
        qm.Graph("plain").instruction("reply", system_instruction="Be brief.").build()
    )


@pytest.fixture(autouse=True)
def _reset_config():
    qm.reset_config()
    yield
    qm.reset_config()


# ── 1. Regression: configure(timeout=...) no longer TypeErrors ────────


class TestConfigureTimeoutKwarg:
    def test_configure_timeout_kwarg_no_longer_raises(self):
        """Sorex round-2 regression — pre-0.4.0 this raised TypeError."""
        # No exception should fire.
        qm.configure(provider="ollama", default_model="x", timeout=30)
        # And the resolved defaults make it through to the accessor.
        assert qm.get_default_timeouts() == {
            "connect_timeout": 30.0,
            "read_timeout": 30.0,
        }

    def test_configure_separate_connect_read_timeouts(self):
        """The split form sets both phases independently."""
        qm.configure(
            provider="ollama",
            default_model="x",
            connect_timeout=10,
            read_timeout=120,
        )
        assert qm.get_default_timeouts() == {
            "connect_timeout": 10.0,
            "read_timeout": 120.0,
        }

    def test_configure_timeout_and_split_form_mutually_exclusive(self):
        """Passing ``timeout=`` AND ``connect_timeout=`` is ambiguous."""
        with pytest.raises(ValueError, match="timeout= OR connect_timeout"):
            qm.configure(
                provider="ollama",
                default_model="x",
                timeout=30,
                connect_timeout=10,
            )

    def test_configure_rejects_non_positive_timeouts(self):
        for bad in (0, -1, -0.5):
            with pytest.raises(ValueError, match="must be > 0"):
                qm.configure(
                    provider="ollama",
                    default_model="x",
                    timeout=bad,
                )

    def test_timeout_is_keyword_only(self):
        """``configure`` keeps a keyword-only signature to protect
        v0.3.x call sites — positional ``qm.configure("ollama", 60)``
        must still fail."""
        sig = inspect.signature(qm.configure)
        for name in ("timeout", "connect_timeout", "read_timeout"):
            assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            assert sig.parameters[name].default is None

        # Positional attempt should raise — all of ``configure``'s
        # params are keyword-only. We don't pass a registry here, just
        # confirm the signature rejects positional args.
        with pytest.raises(TypeError):
            qm.configure("ollama", 60)  # type: ignore[misc]

    def test_reset_config_clears_timeouts(self):
        qm.configure(provider="ollama", default_model="x", timeout=30)
        qm.reset_config()
        assert qm.get_default_timeouts() == {
            "connect_timeout": None,
            "read_timeout": None,
        }


# ── 2. configure() default threads through to the provider ────────────


class TestConfigureDefaultReachesProvider:
    def test_configure_timeout_reaches_llm_config_on_run(self):
        reg, mock = _mock_registry()
        qm.configure(registry=reg, timeout=45)

        qm.run(_plain_graph(), "hi")

        assert mock.last_config is not None
        assert mock.last_config.connect_timeout == 45.0
        assert mock.last_config.read_timeout == 45.0

    def test_configure_split_timeouts_reach_llm_config(self):
        reg, mock = _mock_registry()
        qm.configure(registry=reg, connect_timeout=10, read_timeout=120)

        qm.run(_plain_graph(), "hi")

        assert mock.last_config is not None
        assert mock.last_config.connect_timeout == 10.0
        assert mock.last_config.read_timeout == 120.0

    def test_no_timeouts_configured_leaves_llm_config_at_none(self):
        """Backwards-compat: callers who never opt in see ``None``
        so the provider SDK's own default kicks in."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        qm.run(_plain_graph(), "hi")

        assert mock.last_config is not None
        assert mock.last_config.connect_timeout is None
        assert mock.last_config.read_timeout is None


# ── 3. Per-call overrides win over configure() defaults ──────────────


class TestPerCallOverrides:
    def test_per_call_read_timeout_overrides_default(self):
        """Sorex's 5-min research loop use case: ``qm.configure(timeout=60)``
        applies everywhere EXCEPT one long-running call that passes
        ``read_timeout=300``."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg, timeout=60)

        qm.run(_plain_graph(), "hi", read_timeout=300)

        assert mock.last_config is not None
        assert mock.last_config.read_timeout == 300.0
        # connect_timeout falls back to the configured default
        # (60) since the caller only overrode read_timeout.
        assert mock.last_config.connect_timeout == 60.0

    def test_per_call_timeout_shortcut_overrides_both(self):
        reg, mock = _mock_registry()
        qm.configure(registry=reg, connect_timeout=5, read_timeout=60)

        qm.run(_plain_graph(), "hi", timeout=120)

        assert mock.last_config is not None
        assert mock.last_config.connect_timeout == 120.0
        assert mock.last_config.read_timeout == 120.0

    def test_per_call_connect_and_read_timeout_win_independently(self):
        reg, mock = _mock_registry()
        qm.configure(registry=reg, connect_timeout=5, read_timeout=60)

        qm.run(_plain_graph(), "hi", connect_timeout=2, read_timeout=300)

        assert mock.last_config is not None
        assert mock.last_config.connect_timeout == 2.0
        assert mock.last_config.read_timeout == 300.0

    def test_per_call_timeout_and_split_mutually_exclusive(self):
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="timeout= OR connect_timeout"):
            qm.run(_plain_graph(), "hi", timeout=30, read_timeout=60)

    def test_per_call_rejects_non_positive(self):
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="must be > 0"):
            qm.run(_plain_graph(), "hi", timeout=0)


# ── 4. arun mirrors run for timeouts ──────────────────────────────────


class TestArunTimeouts:
    def test_arun_per_call_read_timeout_overrides_default(self):
        reg, mock = _mock_registry()
        qm.configure(registry=reg, timeout=60)

        async def _main() -> Any:
            return await qm.arun(_plain_graph(), "hi", read_timeout=240)

        asyncio.run(_main())

        assert mock.last_config is not None
        assert mock.last_config.read_timeout == 240.0
        assert mock.last_config.connect_timeout == 60.0


# ── 5. Provider actually honors the timeout (Anthropic) ───────────────


class TestProviderHonorsTimeout:
    def _make_fake_anthropic_response(self) -> MagicMock:
        """Build a minimal response stub for ``client.messages.create``."""
        resp = MagicMock()
        resp.content = []
        resp.stop_reason = "end_turn"
        # ``_parse_usage`` probes a few attrs — give them defaults.
        resp.usage = MagicMock(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return resp

    def test_anthropic_threads_timeout_into_request_params(self):
        """End-to-end: the anthropic SDK's ``messages.create`` receives
        a ``timeout=`` kwarg derived from ``LLMConfig.read_timeout``."""
        from quartermaster_providers.config import LLMConfig
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(
            return_value=self._make_fake_anthropic_response()
        )
        provider._client = fake_client

        config = LLMConfig(
            model="claude-3-5-haiku-20241022",
            provider="anthropic",
            stream=False,
            read_timeout=42.0,
        )

        async def _call() -> None:
            await provider.generate_text_response("hi", config)

        asyncio.run(_call())

        call = fake_client.messages.create.call_args
        assert call is not None
        assert "timeout" in call.kwargs
        # Scalar float when only read_timeout is set — simplest form.
        assert call.kwargs["timeout"] == 42.0

    def test_anthropic_connect_plus_read_yields_httpx_timeout(self):
        """When both phases are set, the provider passes an httpx.Timeout."""
        import httpx

        from quartermaster_providers.config import LLMConfig
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")
        fake_client = MagicMock()
        fake_client.messages.create = AsyncMock(
            return_value=self._make_fake_anthropic_response()
        )
        provider._client = fake_client

        config = LLMConfig(
            model="claude-3-5-haiku-20241022",
            provider="anthropic",
            stream=False,
            connect_timeout=5.0,
            read_timeout=60.0,
        )

        async def _call() -> None:
            await provider.generate_text_response("hi", config)

        asyncio.run(_call())

        call = fake_client.messages.create.call_args
        assert "timeout" in call.kwargs
        timeout_obj = call.kwargs["timeout"]
        assert isinstance(timeout_obj, httpx.Timeout)
        assert timeout_obj.connect == 5.0
        assert timeout_obj.read == 60.0

    def test_openai_provider_also_threads_timeout(self):
        from quartermaster_providers.config import LLMConfig
        from quartermaster_providers.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test")

        class _FakeMessage:
            content = ""
            tool_calls = None

        class _FakeChoice:
            message = _FakeMessage()
            finish_reason = "stop"

        class _FakeResponse:
            choices = [_FakeChoice()]
            usage = None

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=_FakeResponse())
        provider._client = fake_client

        config = LLMConfig(
            model="gpt-4o-mini",
            provider="openai",
            stream=False,
            read_timeout=17.0,
        )

        async def _call() -> None:
            await provider.generate_text_response("hi", config)

        asyncio.run(_call())

        call = fake_client.chat.completions.create.call_args
        assert call is not None
        assert call.kwargs.get("timeout") == 17.0


# ── 6. Stream deadline_seconds fires on a slow stub ───────────────────


class TestStreamDeadlineSeconds:
    def test_stream_deadline_seconds_raises_when_exceeded(self):
        """``deadline_seconds`` is independent of ``read_timeout`` —
        it caps the TOTAL wall-clock of the whole stream.

        We stub ``FlowRunner.run`` to block past the deadline so the
        iterator keeps waiting on an empty queue; once ``deadline_at``
        passes the loop must raise :class:`StreamDeadlineExceeded`.
        """
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        from quartermaster_engine import FlowRunner

        original_run = FlowRunner.run

        def slow_run(
            self, input_message, *, images=None, flow_id=None, llm_timeouts=None
        ):
            # Block for well past the 0.3s deadline — lets the
            # iterator's deadline check fire before any chunk arrives.
            time.sleep(2.0)
            return original_run(
                self,
                input_message,
                images=images,
                flow_id=flow_id,
                llm_timeouts=llm_timeouts,
            )

        with patch.object(FlowRunner, "run", slow_run):
            stream = qm.run.stream(_plain_graph(), "hi", deadline_seconds=0.3)
            with pytest.raises(qm.StreamDeadlineExceeded):
                for _ in stream:
                    pass

    def test_stream_deadline_exceeded_is_timeout_subclass(self):
        """Callers that already ``except TimeoutError`` keep catching it."""
        assert issubclass(qm.StreamDeadlineExceeded, TimeoutError)

    def test_stream_rejects_non_positive_deadline(self):
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="deadline_seconds must be > 0"):
            # Even constructing the iterator must reject the bad value.
            list(qm.run.stream(_plain_graph(), "hi", deadline_seconds=0))

    def test_arun_stream_deadline_seconds_raises_when_exceeded(self):
        """Async variant of the deadline test."""
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        from quartermaster_engine import FlowRunner

        original_run = FlowRunner.run

        def slow_run(
            self, input_message, *, images=None, flow_id=None, llm_timeouts=None
        ):
            time.sleep(2.0)
            return original_run(
                self,
                input_message,
                images=images,
                flow_id=flow_id,
                llm_timeouts=llm_timeouts,
            )

        async def _main() -> None:
            with patch.object(FlowRunner, "run", slow_run):
                stream = qm.arun.stream(_plain_graph(), "hi", deadline_seconds=0.3)
                async for _ in stream:
                    pass

        with pytest.raises(qm.StreamDeadlineExceeded):
            asyncio.run(_main())
