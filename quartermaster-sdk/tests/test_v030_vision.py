"""Tests for the v0.3.0 vision image input surface.

Covers ``qm.run(graph, user_input, image=...)`` / ``images=[...]``,
``qm.arun(...)`` equivalents, and the :func:`qm.instruction` helper's
image forwarding. Replaces the pre-0.3.0 ``[IMAGE_BASE64::...]`` shim
that downstream Sorex ERP used to prepend to user input.

The tests mock the provider via :class:`MockProvider` so nothing hits
the network — each assertion reaches into ``mock.last_config.images``
to verify that the engine forwarded the base64-encoded tuples through
``flow_memory["__user_images__"]`` into ``LLMConfig.images`` when the
node declared ``vision=True``, and that plain ``.instruction()`` nodes
receive an empty list (the image kwarg is a no-op on non-vision graphs).
"""

from __future__ import annotations

import asyncio
import base64
import pathlib
from typing import Any

import openai  # noqa: F401 — eager import, matches test_v020_surface.py
import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


# ── Helpers ───────────────────────────────────────────────────────────


def _mock_registry(text: str = "ocr-output") -> tuple[ProviderRegistry, MockProvider]:
    """Build a ProviderRegistry with a MockProvider registered as ``ollama``.

    Both text (streamed) and native-response channels are primed so the
    vision LLMExecutor — which takes the streamed text path — sees a
    consistent reply. Matches the helper used by ``test_v020_surface``.
    """
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


def _vision_graph() -> qm.GraphSpec:
    """Single-node vision graph for most tests."""
    return (
        qm.Graph("ocr")
        .vision(
            "extract",
            system_instruction="You are an OCR system.",
        )
        .build()
    )


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """A minimal, valid 1x1 PNG. Lets us test real bytes without fixtures on disk."""
    # Smallest syntactically valid PNG — 67 bytes. Good enough to
    # exercise bytes → base64 conversion without pulling a real image
    # into the test tree.
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63f8cfc0500f0000040001212c2e4e0000000049454e44ae426082"
    )


@pytest.fixture
def tiny_png_file(tmp_path: pathlib.Path, tiny_png_bytes: bytes) -> pathlib.Path:
    """Writes the tiny PNG to a temp file so tests can exercise the Path code path."""
    p = tmp_path / "test.png"
    p.write_bytes(tiny_png_bytes)
    return p


# ── 1. qm.run accepts image= bytes ────────────────────────────────────


class TestQmRunImageBytes:
    def test_qm_run_accepts_image_bytes(self, tiny_png_bytes):
        """Passing ``image=bytes`` lands a base64 tuple on ``LLMConfig.images``.

        The vision node's LLMExecutor reads ``__user_images__`` from
        flow memory; ``_build_llm_config`` forwards it to the provider
        via ``LLMConfig.images`` only when ``vision=True`` is set on
        the node (the builder handles that).
        """
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        result = qm.run(_vision_graph(), "extract the invoice", image=tiny_png_bytes)

        assert result.success, f"run failed: {result.error}"
        # The provider received the image through LLMConfig.
        assert mock.last_config is not None
        assert mock.last_config.vision is True
        assert len(mock.last_config.images) == 1
        b64, mime = mock.last_config.images[0]
        assert b64 == base64.b64encode(tiny_png_bytes).decode("ascii")
        # Raw bytes default to image/jpeg — we didn't give an extension.
        assert mime == "image/jpeg"


# ── 2. qm.run accepts pathlib.Path ────────────────────────────────────


class TestQmRunImagePath:
    def test_qm_run_accepts_path(self, tiny_png_file, tiny_png_bytes):
        """Passing a ``pathlib.Path`` reads the file and detects MIME from ext."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        qm.run(_vision_graph(), "extract", image=tiny_png_file)

        assert mock.last_config is not None
        assert len(mock.last_config.images) == 1
        b64, mime = mock.last_config.images[0]
        assert b64 == base64.b64encode(tiny_png_bytes).decode("ascii")
        assert mime == "image/png"

    def test_qm_run_accepts_str_path(self, tiny_png_file, tiny_png_bytes):
        """A filesystem path given as ``str`` works the same as Path."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        qm.run(_vision_graph(), "extract", image=str(tiny_png_file))

        assert mock.last_config is not None
        b64, mime = mock.last_config.images[0]
        assert b64 == base64.b64encode(tiny_png_bytes).decode("ascii")
        assert mime == "image/png"


# ── 3. qm.run accepts images= list ────────────────────────────────────


class TestQmRunImagesList:
    def test_qm_run_accepts_images_list(self, tiny_png_bytes):
        """Multiple images forward as a list of ``(b64, mime)`` tuples."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        # Two copies of the same bytes is enough to exercise the list path.
        qm.run(
            _vision_graph(),
            "compare these",
            images=[tiny_png_bytes, tiny_png_bytes],
        )

        assert mock.last_config is not None
        assert len(mock.last_config.images) == 2
        expected_b64 = base64.b64encode(tiny_png_bytes).decode("ascii")
        assert all(b64 == expected_b64 for b64, _ in mock.last_config.images)

    def test_qm_run_rejects_both_image_and_images(self, tiny_png_bytes):
        """Passing both at once is ambiguous — fail loudly."""
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="either image= .* or images="):
            qm.run(
                _vision_graph(),
                "x",
                image=tiny_png_bytes,
                images=[tiny_png_bytes],
            )


# ── 4. qm.arun equivalent ─────────────────────────────────────────────


class TestQmArunImage:
    def test_qm_arun_accepts_image(self, tiny_png_bytes):
        """``await qm.arun(..., image=bytes)`` forwards the same way qm.run does."""
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        async def _main() -> Any:
            return await qm.arun(_vision_graph(), "extract", image=tiny_png_bytes)

        result = asyncio.run(_main())
        assert result.success, f"arun failed: {result.error}"
        assert mock.last_config is not None
        assert len(mock.last_config.images) == 1
        b64, _ = mock.last_config.images[0]
        assert b64 == base64.b64encode(tiny_png_bytes).decode("ascii")


# ── 5. instruction() helper ───────────────────────────────────────────


class TestInstructionImage:
    def test_instruction_helper_accepts_image(self, tiny_png_bytes):
        """``qm.instruction(..., image=bytes)`` routes through a .vision() node.

        The helper builds a one-node graph under the hood; passing
        ``image=`` flips it from an instruction node to a vision node so
        the engine actually forwards the image to the provider config.
        """
        reg, mock = _mock_registry(text="extracted")
        qm.configure(registry=reg)

        out = qm.instruction(
            system="Extract",
            user="read the image",
            image=tiny_png_bytes,
        )

        assert out == "extracted"
        assert mock.last_config is not None
        assert mock.last_config.vision is True
        assert len(mock.last_config.images) == 1
        b64, _ = mock.last_config.images[0]
        assert b64 == base64.b64encode(tiny_png_bytes).decode("ascii")


# ── 6. Non-vision graph silently drops the image ──────────────────────


class TestNoOpOnNonVisionGraph:
    def test_image_kwarg_works_without_vision_node(self, tiny_png_bytes):
        """Passing ``image=`` to a plain instruction graph is a no-op.

        Rationale: callers shouldn't have to branch on "is this graph
        vision-enabled?" — the image is stored in flow memory, but
        ``_build_llm_config`` only reads it when the node declared
        ``vision=True``. Plain instruction nodes see an empty list.
        """
        reg, mock = _mock_registry()
        qm.configure(registry=reg)

        # Note: plain .instruction(), not .vision()
        graph = qm.Graph("plain").instruction("one").build()

        result = qm.run(graph, "hi", image=tiny_png_bytes)

        assert result.success, f"run failed: {result.error}"
        assert mock.last_config is not None
        # vision=False on the config — no image parts were forwarded.
        assert mock.last_config.vision is False
        assert mock.last_config.images == []


# ── 7. URL / data: URI rejection ──────────────────────────────────────


class TestUriRejection:
    def test_data_uri_raises_clear_error(self):
        """``data:image/...;base64,...`` URIs are out of scope."""
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="data: URIs"):
            qm.run(
                _vision_graph(),
                "x",
                image="data:image/jpeg;base64,/9j/4AAQ",
            )

    def test_http_url_raises_clear_error(self):
        """``http://...`` URLs are out of scope — callers fetch bytes themselves."""
        reg, _ = _mock_registry()
        qm.configure(registry=reg)

        with pytest.raises(ValueError, match="http\\(s\\) URLs"):
            qm.run(
                _vision_graph(),
                "x",
                image="https://example.com/image.jpg",
            )
