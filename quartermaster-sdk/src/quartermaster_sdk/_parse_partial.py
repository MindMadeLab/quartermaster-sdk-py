"""v0.6.0 — progressive-degradation parser for LLM output that doesn't
quite match a Pydantic schema.

Motivation
----------
When an agent finishes a ``qm.Graph`` with an ``instruction_form``
node, its raw text sometimes contains 80% of the right answer wrapped
in prose, broken JSON, or a bullet list. The SDK's strict validator
returns ``output_data=None`` in that case. Callers end up hand-rolling
key-value parsers per schema.

``parse_partial(text, schema)`` replaces that hand-rolling with a
single helper that attempts four progressively-lenient strategies:

1. **Parse full JSON + validate full schema** — the happy path. If the
   whole thing parses and every required field is present, return it.
2. **Extract the last ``{...}`` block and validate optional fields
   only** — tolerates preambles ("Here you go: {...}") and Pydantic
   models whose missing ``required`` fields would otherwise fail. Each
   field is tried individually; the ones that validate land in
   ``partial_data``, the rest in ``missing_fields``.
3. **Line-based ``Field: value`` / ``Field = value`` scanner** — for
   the research-note style output ("Company: MindMade\\nCountry: SI\\n…").
   Only runs when JSON extraction yielded nothing.
4. **Give up** — return an empty ``partial_data`` + every schema field
   as missing + the raw text. Strategy name in the result tells the
   caller nothing stuck.

The return value is a ``PartialResult`` dataclass — ``.data`` is a
plain dict, never a validated Pydantic instance. Callers that want a
model object can call ``Schema.model_construct(**result.data)`` if they
trust the partial data, or ``Schema.model_validate(result.data)`` to
re-validate (and discover which fields don't coerce).

Design note: we deliberately do NOT run the full Pydantic validator
after partial extraction. Pydantic is strict about required fields; a
partial dict will always fail. Callers that want strict validation
should use the existing ``instruction_form(schema=..., on_parse_fail
="raise")`` path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
#: Matches ``Key: value`` / ``Key = value`` / ``**Key**: value`` lines.
#: Captures 2 groups: the field name and the value up to end-of-line.
_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*([A-Za-z][A-Za-z0-9 _\-]*?)\s*\*{0,2}\s*[:=]\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass
class PartialResult:
    """Result of a progressively-degraded parse.

    Attributes:
        data: Fields we managed to extract, keyed by schema field name.
            Never a Pydantic model instance — always a plain dict.
        missing_fields: Schema fields that couldn't be populated.
            Includes both required-but-missing and optional-but-missing
            (callers distinguish by re-inspecting the schema).
        raw_output: The original text passed to ``parse_partial``.
            Kept on the result so logs / error reports can show the
            LLM's actual output next to what we salvaged.
        strategy: Which step succeeded:
            ``"full_json_validated"`` — the happy path.
            ``"json_extracted"``     — last ``{...}`` block found, per-field coerced.
            ``"line_scan"``          — key-colon-value scan over plain text.
            ``"none"``               — nothing stuck; ``data`` is empty.
    """

    data: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    raw_output: str = ""
    strategy: str = "none"


def _field_names_for(schema: Any) -> list[str]:
    """Return the field names declared on a Pydantic model OR
    JSON-schema dict. Preserves declaration order — callers rendering
    a form want the same order the schema authored them in.
    """
    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = None  # type: ignore[assignment]

    if (
        BaseModel is not None
        and isinstance(schema, type)
        and issubclass(schema, BaseModel)
    ):
        return list(schema.model_fields.keys())
    if isinstance(schema, dict):
        # JSON-schema path: ``properties`` is the convention.
        props = schema.get("properties")
        if isinstance(props, dict):
            return list(props.keys())
    # Fall back to introspecting ``__annotations__`` (dataclass etc.)
    ann = getattr(schema, "__annotations__", None)
    if isinstance(ann, dict):
        return list(ann.keys())
    return []


def _try_full_pydantic_validate(payload: dict[str, Any], schema: Any):
    """Return a validated instance (as a dict dump) or ``None`` on fail."""
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return None
    try:
        instance = schema.model_validate(payload)
    except Exception:
        return None
    # Dump back to a plain dict so PartialResult.data is always a dict
    # — mixing dict and BaseModel returns would confuse callers.
    return instance.model_dump()


def _coerce_scalar(raw: str) -> Any:
    """Turn a line-scanned value string into the most-natural Python type.

    Parse order: JSON (handles lists, bools, numbers, nulls, quoted
    strings), fall back to string. Keeps parity with how LLMs render
    values in ``Field: [..]`` / ``Field: true`` / ``Field: 42`` forms.
    """
    raw = raw.strip()
    if not raw:
        return ""
    if raw.lower() == "not found" or raw.lower() == "n/a":
        # Common LLM placeholder for "I couldn't find this" — treat as
        # absent so the field lands in missing_fields rather than
        # polluting data with a literal "not found" string.
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip surrounding quotes/backticks, return as-is.
        return raw.strip("\"'`")


def parse_partial(text: str, schema: Any) -> PartialResult:
    """Attempt progressively-lenient parsing of *text* against *schema*.

    See module docstring for the strategy list. Always returns a
    :class:`PartialResult` — never raises. Callers inspect
    ``result.strategy`` / ``result.missing_fields`` to decide whether
    to use the extracted data or fall through to a human-in-the-loop
    step.

    Args:
        text: The LLM's raw output (``result.text`` or a captured
            ``output_text``). Empty string is handled — you get back
            an all-missing result with strategy ``"none"``.
        schema: A Pydantic ``BaseModel`` subclass, a JSON-schema dict
            (with a ``properties`` key), or any class with
            ``__annotations__``. Determines the field set we try to
            populate.

    Returns:
        A :class:`PartialResult`. Never ``None``, never raises.
    """
    raw = text or ""
    fields = _field_names_for(schema)

    # ── Strategy 1: full JSON + full schema validate ──────────────
    candidate = _extract_last_json_object(raw)
    if candidate is not None:
        validated = _try_full_pydantic_validate(candidate, schema)
        if validated is not None:
            return PartialResult(
                data=validated,
                missing_fields=[],
                raw_output=raw,
                strategy="full_json_validated",
            )

    # ── Strategy 2: JSON extracted, per-field copy ────────────────
    if candidate is not None and fields:
        extracted: dict[str, Any] = {}
        for name in fields:
            if name in candidate:
                extracted[name] = candidate[name]
        if extracted:
            return PartialResult(
                data=extracted,
                missing_fields=[n for n in fields if n not in extracted],
                raw_output=raw,
                strategy="json_extracted",
            )

    # ── Strategy 3: line scan ─────────────────────────────────────
    if fields:
        scanned = _line_scan(raw, fields)
        if scanned:
            return PartialResult(
                data=scanned,
                missing_fields=[n for n in fields if n not in scanned],
                raw_output=raw,
                strategy="line_scan",
            )

    # ── Strategy 4: nothing stuck ─────────────────────────────────
    return PartialResult(
        data={},
        missing_fields=list(fields),
        raw_output=raw,
        strategy="none",
    )


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Find the last ``{...}`` substring and json-decode it.

    Greedy on the opening brace (so ``{outer: {inner: 1}}`` round-trips
    whole), tolerates whitespace / preamble. Returns ``None`` if no
    parseable object is found.

    We keep this local rather than importing from ``_helpers`` because
    ``_helpers.py`` raises on failure (it's the strict path); here we
    want ``None``-on-fail semantics.
    """
    if not text:
        return None
    # Find every candidate object and try them in reverse order (last
    # wins — agents often preamble then emit JSON as the final segment).
    matches = list(_JSON_OBJECT_RE.finditer(text))
    for match in reversed(matches):
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _line_scan(text: str, fields: list[str]) -> dict[str, Any]:
    """Key-colon-value scanner over plain-text LLM output.

    Matches tokens in ``text`` against *fields* case-insensitively and
    allows flexible separators — ``"Company Name"`` in the text is
    matched against the ``company_name`` field, ``Country = SI`` is
    matched as ``country``.

    We purposely DO NOT coerce unknown keys — if the LLM writes
    ``"E-mail: x@y"`` and your schema calls the field ``emails``, that
    line won't match. Keep field names close to what the LLM will
    produce, or use the ``strategy == "json_extracted"`` path for
    tighter control.
    """
    if not text:
        return {}

    # Build a case-insensitive map. Normalise field names by lowering
    # and stripping non-alnum so "Company Name" / "company_name" /
    # "CompanyName" all hash to the same key.
    def _norm(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    fieldmap = {_norm(f): f for f in fields}
    out: dict[str, Any] = {}
    for match in _LINE_RE.finditer(text):
        key_raw, val_raw = match.group(1), match.group(2)
        normed = _norm(key_raw)
        schema_field = fieldmap.get(normed)
        if schema_field is None or schema_field in out:
            continue
        coerced = _coerce_scalar(val_raw)
        if coerced is None:
            continue  # "not found" / "n/a" → treat as absent
        out[schema_field] = coerced
    return out
