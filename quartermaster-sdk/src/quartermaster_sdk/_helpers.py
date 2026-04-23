"""Top-level single-shot helpers: :func:`instruction`, :func:`instruction_form`.

These wrap a single-node graph so callers never touch ``Graph``,
``FlowRunner``, or ``register_local`` for the 90% of use-cases that are
just ``prompt → text`` or ``prompt → typed JSON``.

This is the recommended API for non-agentic calls.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

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


# Reusable decoder — cheaper than newing one up per call-site, and
# ``raw_decode`` is thread-safe for read-only usage.
_JSON_DECODER = json.JSONDecoder()


def _extract_last_json_object(raw: str) -> Any:
    """Extract the *last* parseable JSON object (or array) from ``raw``.

    Strategy:
    1. Strip a wrapping ``` markdown fence (fast path — most models obey
       the system prompt and wrap their JSON or return it bare).
    2. If the stripped text is itself valid JSON, return it.
    3. Otherwise walk the text looking for ``{`` / ``[`` positions and
       attempt :meth:`json.JSONDecoder.raw_decode` at each candidate.
       Keep the **last** successful decode — the downstream ``extract_json``
       uses the same heuristic: reasoning models emit a bullet preamble
       then the final JSON, so the trailing object is the answer.

    Using the stdlib :class:`json.JSONDecoder` (rather than a hand-rolled
    brace counter) means we get string awareness for free — escaped
    quotes, ``}`` inside string values, nested objects, arrays inside
    arrays, control-char handling, etc. all behave the same way as
    :func:`json.loads`.

    Raises:
        :class:`json.JSONDecodeError` if no valid JSON can be recovered
        from the text. Callers wrap this in a domain error (e.g.
        ``RuntimeError`` with the raw text attached) so end-users see a
        debuggable message.
    """
    cleaned = _strip_markdown_fence(raw)

    # Fast path — the stripped text *is* the JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Walk every candidate start position. Objects start with ``{`` and
    # arrays with ``[`` — we scan the original ``raw`` (not ``cleaned``)
    # because the fence stripper can change offsets but the underlying
    # JSON content is the same. Using ``raw`` directly also means we
    # tolerate stray fences embedded in the text (bullet preamble +
    # fence + JSON is the Gemma case).
    last_decoded: Any = None
    last_found = False
    last_end = -1

    for idx, ch in enumerate(raw):
        if ch not in ("{", "["):
            continue
        try:
            value, end = _JSON_DECODER.raw_decode(raw, idx)
        except json.JSONDecodeError:
            continue
        # Keep the decode that ends LATEST in the text — that's the
        # model's final answer when a reasoning preamble came first.
        # Ties broken by later start (which can't happen with
        # raw_decode since earlier starts have earlier ends for the
        # same match), but the ``>`` guard keeps behaviour predictable.
        if end > last_end:
            last_decoded = value
            last_end = end
            last_found = True

    if last_found:
        return last_decoded

    # No luck — re-raise the first parse error so the caller can surface
    # the canonical stdlib message alongside the raw text.
    raise json.JSONDecodeError(
        "No valid JSON object or array could be extracted from the text",
        raw,
        0,
    )


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
    schema: type[T] | dict[str, Any],
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
) -> T | dict[str, Any]:
    """Single-shot prompt → Pydantic model *or* JSON-schema-validated dict.

    Runs the prompt through a one-node instruction graph, then parses
    the LLM's JSON output into *schema*.  Default temperature is 0.1 —
    structured extraction benefits from the more-deterministic end of
    the range.

    The helper injects the schema into the system prompt so the LLM
    knows what JSON shape to return; callers don't need to describe
    the format themselves.

    Schema forms (v0.4.0):

    * A Pydantic v2 ``BaseModel`` subclass → returns an instance,
      validated via ``model_validate`` on the parsed JSON.
    * A :class:`dict` (literal JSON Schema) → returns the parsed dict.
      Validation runs via :mod:`jsonschema` if the package is installed
      (a soft/optional dep); otherwise the raw parsed dict is returned
      after emitting a single :class:`UserWarning`.

    Args:
        schema: Either a Pydantic ``BaseModel`` subclass **or** a JSON
            Schema ``dict``.  **Required.**
        user: The user message / prompt.  **Required.**
        system: Instruction steering the extraction (e.g. "Classify
            this email...").  The schema description is appended
            automatically.
        model: Model identifier (falls back to the configured default).
        temperature: Sampling temperature — default 0.1 for
            determinism.

    Returns:
        An instance of *schema* (Pydantic path) or a validated dict
        (JSON Schema path) populated from the LLM's output.

    Raises:
        TypeError: ``schema`` is neither a Pydantic ``BaseModel``
            subclass nor a ``dict``.
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

    is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
    is_dict_schema = isinstance(schema, dict)

    if not (is_pydantic_schema or is_dict_schema):
        # TypeError (not ValueError): this is a programmer error about the
        # *type* of the argument, not its value. Matches the pattern used
        # elsewhere in the SDK.
        raise TypeError(
            f"instruction_form(schema=...) must be a pydantic.BaseModel "
            f"subclass or a dict (JSON Schema); got {type(schema).__name__}"
        )

    # Inject a JSON-schema hint into the system prompt so the model
    # knows what fields are expected.  Keep this short — bloating the
    # system prompt with a giant schema eats context.
    if is_pydantic_schema:
        try:
            schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        except Exception:
            schema_json = str(schema.__name__)
        schema_label = schema.__name__
    else:
        # Dict-schema path: encode the literal JSON Schema. We keep the
        # separators tight (matches the Pydantic path) so we don't blow
        # the system prompt up more than necessary.
        try:
            schema_json = json.dumps(schema, separators=(",", ":"))
        except (TypeError, ValueError):
            # Dict isn't JSON-encodable — fall back to ``str`` so we at
            # least tell the model *something*. The caller will see the
            # failure downstream when validation tries to apply the
            # unusable schema.
            schema_json = str(schema)
        schema_label = "<dict schema>"

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

    # v0.4.0 (T4): tolerate reasoning-model preambles by walking the
    # text for the LAST valid JSON object. Pre-0.4.0 code only stripped
    # a wrapping fence — Gemma-family models often emit a bullet list
    # before the fenced JSON, which broke ``model_validate_json``.
    try:
        parsed = _extract_last_json_object(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"instruction_form(): model output did not contain parseable "
            f"JSON for schema {schema_label}. Raw output:\n{raw}\n\n"
            f"Parse error:\n{exc}"
        ) from exc

    if is_pydantic_schema:
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise RuntimeError(
                f"instruction_form(): model output did not match schema "
                f"{schema.__name__}. Raw output:\n{raw}\n\nValidation errors:\n{exc}"
            ) from exc

    # Dict-schema path: try to validate with jsonschema (soft dep) and
    # fall back to returning the raw parsed dict with a warning if the
    # package isn't installed.
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        warnings.warn(
            "instruction_form(schema=<dict>): jsonschema is not installed, "
            "so the returned dict is not validated against the provided "
            "schema. Install with `pip install jsonschema` to enable "
            "validation.",
            UserWarning,
            stacklevel=2,
        )
        return parsed

    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as exc:
        raise RuntimeError(
            f"instruction_form(): model output did not match the provided "
            f"JSON Schema. Raw output:\n{raw}\n\nValidation error:\n{exc}"
        ) from exc
    except jsonschema.SchemaError as exc:
        raise RuntimeError(
            f"instruction_form(): the provided JSON Schema is invalid.\n"
            f"Schema error:\n{exc}"
        ) from exc

    return parsed


__all__ = ["instruction", "instruction_form"]
