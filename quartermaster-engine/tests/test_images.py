"""Tests for the :mod:`quartermaster_engine.images` helper module.

Covers the normalisation path that the SDK's ``image=`` / ``images=``
kwargs flow through before reaching ``FlowRunner.run(images=...)``.
"""

from __future__ import annotations

import base64
import pathlib

import pytest

from quartermaster_engine.images import prepare_images


TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63f8cfc0500f0000040001212c2e4e0000000049454e44ae426082"
)


class TestPrepareImagesBytes:
    def test_bytes_input_becomes_base64_default_mime(self):
        out = prepare_images(image=TINY_PNG)
        assert len(out) == 1
        b64, mime = out[0]
        assert b64 == base64.b64encode(TINY_PNG).decode("ascii")
        assert mime == "image/jpeg"  # default when no extension hint


class TestPrepareImagesPath:
    def test_path_input_reads_file_detects_mime(self, tmp_path):
        png = tmp_path / "t.png"
        png.write_bytes(TINY_PNG)
        out = prepare_images(image=png)
        b64, mime = out[0]
        assert b64 == base64.b64encode(TINY_PNG).decode("ascii")
        assert mime == "image/png"

    def test_str_path_works_like_pathlib(self, tmp_path):
        jpg = tmp_path / "t.jpg"
        jpg.write_bytes(TINY_PNG)
        out = prepare_images(image=str(jpg))
        _, mime = out[0]
        assert mime == "image/jpeg"

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path):
        weird = tmp_path / "t.xyz"
        weird.write_bytes(TINY_PNG)
        _, mime = prepare_images(image=weird)[0]
        assert mime == "image/jpeg"

    def test_webp_detected(self, tmp_path):
        p = tmp_path / "t.webp"
        p.write_bytes(TINY_PNG)
        _, mime = prepare_images(image=p)[0]
        assert mime == "image/webp"


class TestPrepareImagesList:
    def test_images_list_preserves_order(self, tmp_path):
        png = tmp_path / "a.png"
        jpg = tmp_path / "b.jpg"
        png.write_bytes(TINY_PNG)
        jpg.write_bytes(TINY_PNG)
        out = prepare_images(images=[png, jpg])
        assert [m for _, m in out] == ["image/png", "image/jpeg"]

    def test_empty_when_both_none(self):
        assert prepare_images() == []

    def test_rejects_both_image_and_images(self):
        with pytest.raises(ValueError, match="either image="):
            prepare_images(image=TINY_PNG, images=[TINY_PNG])

    def test_rejects_non_list_images_kwarg(self):
        with pytest.raises(TypeError, match="images= must be a list"):
            # A single bytes value for ``images=`` is ambiguous — force
            # the user to pick ``image=`` for singular, ``images=`` for
            # plural. This avoids the "forgot the list brackets and got
            # unexpected behaviour" footgun.
            prepare_images(images=TINY_PNG)  # type: ignore[arg-type]


class TestPrepareImagesRejection:
    def test_data_uri_rejected(self):
        with pytest.raises(ValueError, match="data: URIs"):
            prepare_images(image="data:image/jpeg;base64,/9j/4AAQ")

    def test_http_url_rejected(self):
        with pytest.raises(ValueError, match="http\\(s\\) URLs"):
            prepare_images(image="http://example.com/img.png")

    def test_https_url_rejected(self):
        with pytest.raises(ValueError, match="http\\(s\\) URLs"):
            prepare_images(image="https://example.com/img.png")

    def test_unsupported_type_rejected(self):
        with pytest.raises(TypeError, match="accepts bytes, pathlib.Path"):
            prepare_images(image=123)  # type: ignore[arg-type]
