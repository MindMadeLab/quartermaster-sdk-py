"""v0.3.0 vision regression: every provider that claims vision support
must actually forward ``LLMConfig.images`` into its API request.

Anthropic and OpenAI got vision wired in the v0.3.0 vision commit.
Groq + xAI inherit ``OpenAIProvider._build_user_content`` unchanged,
so they get vision via inheritance. Google has its own implementation
and needed explicit wiring (see ``GoogleProvider._build_content_parts``).

These tests don't hit real APIs; they patch each provider's underlying
client and assert the request payload carries the image data in the
shape the upstream API expects. This is the regression guard that
prevents a refactor from silently dropping image parts again.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quartermaster_providers.config import LLMConfig

# Tiny payload (10 bytes). We don't need it to be a valid PNG — providers
# don't decode it, they just forward the base64 (or raw bytes for Google)
# to the upstream API.
_FAKE_IMG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
_FAKE_IMG_BYTES = base64.b64decode(_FAKE_IMG_B64)


def _cfg(
    provider: str,
    model: str,
    images: list[tuple[str, str]] | None = None,
) -> LLMConfig:
    """Build a minimal LLMConfig with the v0.3.0 ``images`` field set."""
    return LLMConfig(
        provider=provider,
        model=model,
        temperature=0.1,
        max_output_tokens=64,
        images=images or [],
    )


# ── Anthropic ───────────────────────────────────────────────────────────


class TestAnthropicVision:
    def test_image_block_emitted_in_user_content(self) -> None:
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")
        config = _cfg(
            "anthropic",
            "claude-haiku-4-5-20251001",
            [(_FAKE_IMG_B64, "image/png")],
        )

        content = provider._build_user_content("describe this", config)

        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[0]["source"]["data"] == _FAKE_IMG_B64
        assert content[-1]["type"] == "text"
        assert content[-1]["text"] == "describe this"

    def test_no_image_returns_plain_string(self) -> None:
        from quartermaster_providers.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-test")
        config = _cfg("anthropic", "claude-haiku-4-5-20251001")

        content = provider._build_user_content("plain text only", config)

        assert content == "plain text only"


# ── OpenAI ──────────────────────────────────────────────────────────────


class TestOpenAIVision:
    def test_image_url_part_emitted_in_user_content(self) -> None:
        from quartermaster_providers.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test")
        config = _cfg(
            "openai",
            "gpt-4o-mini",
            [(_FAKE_IMG_B64, "image/png")],
        )

        content = provider._build_user_content("describe this", config)

        assert isinstance(content, list)
        # OpenAI uses a data: URI under image_url.url
        image_part = next(p for p in content if p.get("type") == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert _FAKE_IMG_B64 in image_part["image_url"]["url"]
        text_part = next(p for p in content if p.get("type") == "text")
        assert text_part["text"] == "describe this"


# ── Groq (inherits OpenAI; verify no override broke it) ─────────────────


class TestGroqVisionInheritance:
    def test_groq_inherits_openai_vision_path(self) -> None:
        from quartermaster_providers.providers.groq import GroqProvider

        provider = GroqProvider(api_key="gsk-test")
        config = _cfg(
            "groq",
            "llama-3.2-90b-vision-preview",
            [(_FAKE_IMG_B64, "image/jpeg")],
        )

        content = provider._build_user_content("what is in this picture", config)

        # Groq doesn't override _build_user_content — should match OpenAI shape.
        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content)


# ── xAI (inherits OpenAI; verify no override broke it) ──────────────────


class TestXAIVisionInheritance:
    def test_xai_inherits_openai_vision_path(self) -> None:
        from quartermaster_providers.providers.xai import XAIProvider

        provider = XAIProvider(api_key="xai-test")
        config = _cfg(
            "xai",
            "grok-2-vision-1212",
            [(_FAKE_IMG_B64, "image/jpeg")],
        )

        content = provider._build_user_content("what is in this picture", config)

        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content)


# ── Google ──────────────────────────────────────────────────────────────


class TestGoogleVision:
    """Google takes a different shape than Anthropic/OpenAI — the
    ``generate_content_async`` call accepts a list of parts where each
    image is a ``{mime_type, data}`` dict and the text is a plain str.
    Our ``_build_content_parts`` produces this list; these tests pin
    the conversion (base64 -> raw bytes) and ordering (images first,
    text last)."""

    def _provider(self):
        # Google provider validates the SDK is installed at __init__.
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            pytest.skip("google-generativeai not installed in this env")
        from quartermaster_providers.providers.google import GoogleProvider

        return GoogleProvider(api_key="AIza-test")

    def test_image_part_emitted_with_decoded_bytes(self) -> None:
        provider = self._provider()
        config = _cfg(
            "google",
            "gemini-1.5-flash",
            [(_FAKE_IMG_B64, "image/png")],
        )

        parts = provider._build_content_parts("describe this", config)

        assert isinstance(parts, list)
        assert len(parts) == 2
        # First part: image dict with raw decoded bytes (NOT base64).
        assert parts[0]["mime_type"] == "image/png"
        assert parts[0]["data"] == _FAKE_IMG_BYTES
        # Last part: prompt string.
        assert parts[1] == "describe this"

    def test_no_image_returns_plain_string(self) -> None:
        provider = self._provider()
        config = _cfg("google", "gemini-1.5-flash")

        parts = provider._build_content_parts("plain text only", config)

        assert parts == "plain text only"

    def test_default_mime_type_when_none(self) -> None:
        """An image with no mime_type defaults to image/jpeg."""
        provider = self._provider()
        config = _cfg(
            "google",
            "gemini-1.5-flash",
            [(_FAKE_IMG_B64, "")],
        )

        parts = provider._build_content_parts("describe this", config)

        assert isinstance(parts, list)
        assert parts[0]["mime_type"] == "image/jpeg"

    def test_multiple_images_preserved_in_order(self) -> None:
        provider = self._provider()
        img2_b64 = base64.b64encode(b"second image bytes").decode("ascii")
        config = _cfg(
            "google",
            "gemini-1.5-flash",
            [
                (_FAKE_IMG_B64, "image/png"),
                (img2_b64, "image/webp"),
            ],
        )

        parts = provider._build_content_parts("compare these two", config)

        assert isinstance(parts, list)
        assert len(parts) == 3
        assert parts[0]["mime_type"] == "image/png"
        assert parts[1]["mime_type"] == "image/webp"
        assert parts[2] == "compare these two"

    @pytest.mark.asyncio
    async def test_generate_text_response_passes_content_parts(self) -> None:
        """Integration-style: patch the model and assert the call site
        forwards the structured parts (not the raw prompt string)."""
        provider = self._provider()
        config = _cfg(
            "google",
            "gemini-1.5-flash",
            [(_FAKE_IMG_B64, "image/png")],
        )

        fake_response = MagicMock()
        fake_response.text = "I see a tiny image"
        fake_response.candidates = []
        fake_response.usage_metadata = None

        with patch.object(provider, "_get_model") as get_model:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=fake_response)
            get_model.return_value = mock_model

            await provider.generate_text_response("describe", config)

        # The arg passed to generate_content_async should be the list of parts
        # we'd get from _build_content_parts (not the bare prompt string).
        called_with = mock_model.generate_content_async.await_args.args[0]
        assert isinstance(called_with, list)
        assert called_with[0]["mime_type"] == "image/png"
        assert called_with[1] == "describe"
