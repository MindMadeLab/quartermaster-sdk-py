"""v0.9.0 — surface provider token usage on Result and stream (#100).

Providers already parse OpenAI ``usage.prompt_tokens`` /
``usage.completion_tokens`` into :class:`TokenUsage`. These tests pin
the leak that used to drop that data above the provider: engine
``NodeResult.data["usage"]``, SDK ``Result.usage``, and the terminal
:class:`DoneChunk`.

``qm.instruction()`` stays a plain ``str``; Graph ``qm.run`` is the
public surface for usage.
"""

from __future__ import annotations

import openai  # noqa: F401 — eager import; see test_ollama_chat.py

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse, TokenUsage


def _registry(
    *,
    text: str = "canned",
    usage: TokenUsage | None = None,
) -> tuple[ProviderRegistry, MockProvider]:
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop", usage=usage)],
        native_responses=[
            NativeResponse(
                text_content=text,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
                usage=usage,
            )
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg, mock


@pytest.fixture(autouse=True)
def _reset_config():
    qm.reset_config()
    yield
    qm.reset_config()


class TestRunResultUsage:
    def test_instruction_graph_result_usage_has_input_output_tokens(self):
        usage = TokenUsage(input_tokens=12, output_tokens=4)
        reg, _ = _registry(text="hello", usage=usage)
        qm.configure(registry=reg)

        result = qm.run(qm.Graph("x").instruction("One", capture_as="reply"), "hi")

        assert result.success
        assert result.text == "hello"
        assert result.usage is not None
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 4
        assert result["reply"].data["usage"] == {
            "input_tokens": 12,
            "output_tokens": 4,
        }

    def test_missing_usage_stays_none(self):
        """Providers that omit usage must not invent 0-token counts."""
        reg, _ = _registry(text="hello", usage=None)
        qm.configure(registry=reg)

        result = qm.run(qm.Graph("x").instruction("One", capture_as="reply"), "hi")

        assert result.success
        assert result.usage is None
        assert "usage" not in result["reply"].data

    def test_instruction_helper_still_returns_str(self):
        usage = TokenUsage(input_tokens=9, output_tokens=3)
        reg, _ = _registry(text="gotcha", usage=usage)
        qm.configure(registry=reg)

        reply = qm.instruction(system="sys", user="usr")

        assert isinstance(reply, str)
        assert reply == "gotcha"


class TestStreamUsage:
    def test_done_chunk_includes_usage_when_provider_sends_it(self):
        usage = TokenUsage(input_tokens=42, output_tokens=7)
        reg, _ = _registry(text="streamed", usage=usage)
        qm.configure(registry=reg)

        chunks = list(qm.run.stream(qm.Graph("x").instruction("One"), "hi"))

        assert chunks[-1].type == "done"
        done = chunks[-1]
        assert done.usage is not None
        assert done.usage.input_tokens == 42
        assert done.usage.output_tokens == 7
        assert done.result.usage is not None
        assert done.result.usage.input_tokens == 42

    def test_done_chunk_usage_absent_when_provider_omits_it(self):
        reg, _ = _registry(text="streamed", usage=None)
        qm.configure(registry=reg)

        chunks = list(qm.run.stream(qm.Graph("x").instruction("One"), "hi"))

        assert chunks[-1].type == "done"
        assert chunks[-1].usage is None
        assert chunks[-1].result.usage is None
        token_text = "".join(c.content for c in chunks if c.type == "token")
        assert "streamed" in token_text
