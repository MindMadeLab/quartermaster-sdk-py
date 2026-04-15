"""Image-input normalisation for vision-capable nodes.

Public helper :func:`prepare_images` accepts the shapes the SDK forwards
from ``qm.run(graph, user_input, image=...)`` / ``images=[...]`` and
returns the internal ``list[(base64_ascii, mime_type)]`` representation
that's stored in flow memory under the ``__user_images__`` key and then
forwarded into :class:`quartermaster_providers.LLMConfig.images` for the
active node.

Accepts:

* ``bytes``                     — raw image bytes, MIME defaulted to
                                  ``image/jpeg`` (most common; overridable
                                  by providers that sniff headers).
* ``pathlib.Path``              — read from disk, MIME detected from
                                  extension (``.png`` / ``.jpg`` /
                                  ``.jpeg`` / ``.gif`` / ``.webp``).
* ``str`` filesystem path       — same as ``pathlib.Path``.

Explicitly rejects (raises :class:`ValueError` with a helpful message):

* ``data:image/...;base64,...`` URIs — out of scope; caller should
                                       fetch + pass bytes directly.
* ``http://...`` / ``https://...`` URLs — same rationale.

The helper is intentionally synchronous — image IO is small enough that
blocking on disk reads in the hot path is fine, and staying sync keeps
the engine's (sync) ``flow_runner.run`` contract clean.
"""

from __future__ import annotations

import base64
import logging
import pathlib
from typing import Union

logger = logging.getLogger(__name__)

#: Types accepted by :func:`prepare_images` for each individual image.
ImageInput = Union[bytes, pathlib.Path, str]

_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

_DEFAULT_MIME = "image/jpeg"


def _mime_from_extension(path: pathlib.Path) -> str:
    """Best-effort MIME detection from the path's suffix.

    Falls back to :data:`_DEFAULT_MIME` when the extension is missing
    or unknown — providers that inspect raw bytes can still override
    this downstream, but keeping the default off ``image/jpeg`` matches
    what most vision APIs accept when the MIME is ambiguous.
    """
    suffix = path.suffix.lower()
    return _EXT_TO_MIME.get(suffix, _DEFAULT_MIME)


def _reject_uri_like(value: str) -> None:
    """Raise :class:`ValueError` if *value* looks like a URI, not a path.

    ``data:image/...;base64,...`` and ``http(s)://...`` strings are
    intentionally out of scope for the v0.3.0 image-input API — callers
    should either decode the base64 or fetch the URL themselves and
    forward the raw bytes.  Raising here gives a much clearer error
    than letting ``open()`` fail with a confusing path-doesn't-exist
    traceback three layers down.
    """
    lowered = value.strip().lower()
    if lowered.startswith("data:"):
        raise ValueError(
            "image= does not accept data: URIs. Decode the base64 "
            "payload yourself (``base64.b64decode(...)``) and pass the "
            "raw bytes instead."
        )
    if lowered.startswith("http://") or lowered.startswith("https://"):
        raise ValueError(
            "image= does not accept http(s) URLs. Fetch the image "
            "yourself (e.g. ``requests.get(url).content``) and pass the "
            "raw bytes instead."
        )


def _prepare_one(image: ImageInput) -> tuple[str, str]:
    """Convert a single input into ``(base64_ascii, mime_type)``.

    See the module docstring for the accepted shapes.  Raises
    :class:`ValueError` on URIs/URLs and :class:`TypeError` on other
    unsupported types so callers get an immediate, descriptive error
    rather than a downstream provider failure.
    """
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii"), _DEFAULT_MIME

    if isinstance(image, pathlib.Path):
        data = image.read_bytes()
        return base64.b64encode(data).decode("ascii"), _mime_from_extension(image)

    if isinstance(image, str):
        _reject_uri_like(image)
        path = pathlib.Path(image)
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii"), _mime_from_extension(path)

    raise TypeError(
        "image= / images= accepts bytes, pathlib.Path, or a filesystem "
        f"path (str). Got {type(image).__name__!r}."
    )


def prepare_images(
    image: ImageInput | None = None,
    images: list[ImageInput] | tuple[ImageInput, ...] | None = None,
) -> list[tuple[str, str]]:
    """Normalise single/multi image inputs into ``[(b64, mime), ...]``.

    * Exactly one of *image* / *images* may be non-``None``; passing
      both raises :class:`ValueError`.
    * ``image=X`` is equivalent to ``images=[X]`` — the singular kwarg
      is a convenience for the common one-image case.
    * When both are ``None`` returns an empty list — downstream code
      treats that as a text-only request.

    Returns a plain ``list[tuple[str, str]]`` so providers can iterate
    without caring about the original input shape.  The base64 strings
    are ASCII (never bytes) so JSON serialisation is trivial.
    """
    if image is not None and images is not None:
        raise ValueError("Pass either image= (single) or images= (list), not both.")

    if image is not None:
        return [_prepare_one(image)]

    if images is None:
        return []

    if not isinstance(images, (list, tuple)):
        raise TypeError(
            "images= must be a list or tuple; got "
            f"{type(images).__name__!r}. For a single image, use "
            "image= instead."
        )

    return [_prepare_one(item) for item in images]


__all__ = ["ImageInput", "prepare_images"]
