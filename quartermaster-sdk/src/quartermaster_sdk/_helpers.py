"""Top-level single-shot helpers: :func:`instruction`, :func:`instruction_form`.

These wrap a single-node graph so callers never touch ``Graph``,
``FlowRunner``, or ``register_local`` for the 90% of use-cases that are
just ``prompt → text`` or ``prompt → typed JSON``.

The docs / Sorex feedback called this "the thing we'd actually write";
v0.2.0 ships it as the primary recommended API for non-agentic calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, TypeVar

from quartermaster_engine import ImageInput
from quartermaster_graph import Graph

from ._config import get_default_model
from ._runner import run

if TYPE_CHECKING:
    from pydantic import BaseModel
    from quartermaster_providers import ProviderRegistry

logger = logging.getLogger(__name__)


T = TypeVar("T", bound="BaseModel")


# Match an opening ```lang\n (or ``` alone) at the start of a response and
# a closing ``` at the end, leaving any triple-backtick-looking characters
# INSIDE the JSON alone.  ``str.strip("`")`` would eat backticks out of
# string values (e.g. SQL or regex examples quoted in the JSON body);
# this regex pair targets only the fence itself.
_MD_FENCE_OPEN_RE = re.compile(r"\A\s*```[A-Za-z0-9_-]*\s*\n?")
_MD_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*\Z")


def _strip_markdown_fence(raw: str) -> str:
    """Remove a leading ``` fence (with optional language tag) and a trailing ``` fence.

    Preserves every backtick that appears inside the fenced content —
    ``re.sub(r'^```[lang]\\n?', '')`` + ``re.sub(r'\\n?```$', '')`` rather
    than the pre-0.2.0 naive ``str.strip("`")`` which corrupted JSON
    strings containing literal backticks.
    """
    text = _MD_FENCE_OPEN_RE.sub("", raw)
    text = _MD_FENCE_CLOSE_RE.sub("", text)
    return text.strip()


def instruction(
    *,
    user: str,
    system: str = "",
    model: str | None = None,
    provider: str = "",
    temperature: float = 0.7,
    max_output_tokens: int | None = None,
    thinking_level: str = "off",
    image: ImageInput | None = None,
    images: list[ImageInput] | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> str:
    """Single-shot prompt → text.

    Builds a one-node ``Instruction`` graph under the hood, runs it,
    and returns the final assistant text.  The ``user`` kwarg is the
    primary user message — no ``.user()`` node boilerplate.

    Args:
        user: The user message / prompt.  **Required.**
        system: System instruction steering the model.
        model: Model identifier.  Falls back to the ``default_model`` set
            via :func:`configure` or ``$QM_DEFAULT_MODEL``.
        provider: Optional explicit provider name.  Leave blank for
            auto-resolution from the registry.
        temperature: Sampling temperature.
        max_output_tokens: Hard cap on output tokens.
        thinking_level: ``off``/``low``/``medium``/``high`` — forwarded
            to reasoning-capable models.
        image: Optional single image input (``bytes``,
            :class:`pathlib.Path`, or path string). When set, the
            one-node graph is built as a ``.vision()`` node so the
            image is forwarded to the model alongside the text prompt.
            Mutually exclusive with *images*.
        images: Optional list of image inputs (same per-item types as
            *image*). Mutually exclusive with *image*.
        provider_registry: Override the module-level default registry.

    Returns:
        The assistant's reply as a plain ``str``.  Raises
        ``RuntimeError`` when the underlying flow fails.
    """
    resolved_model = model or get_default_model()
    if not resolved_model:
        raise ValueError(
            "instruction(): no model resolved. Pass model=... or call "
            "quartermaster_sdk.configure(default_model=...) at app boot."
        )

    # Pick the right single-node graph based on whether the caller
    # supplied any images. A ``.vision()`` node is identical to an
    # ``.instruction()`` node except it sets ``vision=True`` in metadata,
    # which is what the engine uses to decide whether to forward the
    # flow-memory ``__user_images__`` list into the provider config.
    has_image = image is not None or images is not None

    builder = Graph("instruction")
    if has_image:
        builder = builder.vision(
            "Vision",
            model=resolved_model,
            provider=provider,
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
    else:
        builder = builder.instruction(
            "Instruction",
            model=resolved_model,
            provider=provider,
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
    graph = builder.build()

    result = run(
        graph,
        user,
        image=image,
        images=images,
        provider_registry=provider_registry,
    )
    if not result.success:
        raise RuntimeError(f"instruction() failed: {result.error}")
    return result.text


def instruction_form(
    schema: type[T],
    *,
    user: str,
    system: str = "",
    model: str | None = None,
    provider: str = "",
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
    image: ImageInput | None = None,
    images: list[ImageInput] | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> T:
    """Single-shot prompt → Pydantic model.

    Runs the prompt through a one-node instruction graph, then parses
    the LLM's JSON output into *schema* via ``model_validate_json``.
    Default temperature is 0.1 — structured extraction benefits from
    the more-deterministic end of the range.

    The helper injects the schema into the system prompt so the LLM
    knows what JSON shape to return; callers don't need to describe
    the format themselves.

    Args:
        schema: A Pydantic v2 ``BaseModel`` subclass describing the
            output shape.  **Required.**
        user: The user message / prompt.  **Required.**
        system: Instruction steering the extraction (e.g. "Classify
            this email...").  The schema description is appended
            automatically.
        model: Model identifier (falls back to the configured default).
        temperature: Sampling temperature — default 0.1 for
            determinism.

    Returns:
        An instance of *schema* populated from the LLM's output.

    Raises:
        ValueError: ``schema`` isn't a Pydantic ``BaseModel`` subclass.
        RuntimeError: the underlying flow failed, or the model's output
            couldn't be parsed into *schema* (raises a
            ``ValidationError``-wrapping ``RuntimeError`` with the raw
            text attached for debugging).

    .. warning::
        The schema's JSON representation is injected **verbatim** into
        the system prompt so the LLM knows the target shape.  That
        includes every field's ``description=`` and ``default=``
        metadata.  If your Pydantic model derives descriptions from
        external / attacker-influenced sources (rare but not unheard of
        in code-generation pipelines), an attacker could use them as an
        indirect-prompt-injection channel.  Keep ``Field(description=)``
        values static / developer-controlled.
    """
    try:
        from pydantic import BaseModel, ValidationError
    except (
        ImportError
    ) as exc:  # pragma: no cover — pydantic is a hard dep of many ecosystems
        raise ImportError(
            "instruction_form() requires pydantic. Install with `pip install pydantic`."
        ) from exc

    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise ValueError(
            f"instruction_form(schema=...) must be a pydantic.BaseModel "
            f"subclass; got {type(schema).__name__}"
        )

    # Inject a JSON-schema hint into the system prompt so the model
    # knows what fields are expected.  Keep this short — bloating the
    # system prompt with a giant schema eats context.
    try:
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    except Exception:
        schema_json = str(schema.__name__)

    full_system = (
        f"{system}\n\n"
        "Respond with a single JSON object matching this schema. "
        "Do not wrap the JSON in markdown code fences. Do not emit any "
        "text outside the JSON object.\n\n"
        f"Schema: {schema_json}"
    ).strip()

    raw = instruction(
        user=user,
        system=full_system,
        model=model,
        provider=provider,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level="off",
        image=image,
        images=images,
        provider_registry=provider_registry,
    )

    # Strip fences in case the model insists on markdown despite instructions.
    # Uses a regex-anchored pattern — the pre-0.2.0 ``str.strip("`")`` ate
    # backticks out of string values (code snippets, regex examples) and
    # corrupted the JSON before parsing could see it.
    cleaned = _strip_markdown_fence(raw)

    try:
        return schema.model_validate_json(cleaned)
    except ValidationError as exc:
        raise RuntimeError(
            f"instruction_form(): model output did not match schema "
            f"{schema.__name__}. Raw output:\n{raw}\n\nValidation errors:\n{exc}"
        ) from exc


__all__ = ["instruction", "instruction_form"]
