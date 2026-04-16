"""v0.4.0 regressions for :func:`qm.instruction_form` robustness.

Covers two v0.4.0 items:

* **P1.3 (T4)** — Gemma-family reasoning models emit a bullet preamble
  before the fenced JSON. The pre-0.4.0 helper stripped the fence but
  then choked because the stripped text still started with the bullets.
  v0.4.0 introduces ``_extract_last_json_object`` which walks every
  ``{`` / ``[`` position via :func:`json.JSONDecoder.raw_decode` and
  keeps the LAST valid decode — matching the heuristic that downstream
  integrators already ship in ``services.extract_json``.

* **P3.2 (T5)** — Accept ``schema`` as a dict (JSON Schema literal),
  not just a Pydantic ``BaseModel`` subclass. Dict-schema validation
  uses :mod:`jsonschema` as a soft optional dep; when the package isn't
  installed the helper warns once and returns the raw parsed dict.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import openai  # noqa: F401 — eager import, mirrors test_v020_surface.py

import pytest
from pydantic import BaseModel

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


# ── Helpers (mirror of test_v020_surface._mock_registry) ─────────────


def _mock_registry(
    text: str = "canned",
    native_text: str | None = None,
) -> tuple[ProviderRegistry, MockProvider]:
    """Wire up a ``ProviderRegistry`` returning canned text."""
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=native_text if native_text is not None else text,
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


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


class _Classification(BaseModel):
    category: str
    priority: str


# ── T4: Gemma preamble parser ────────────────────────────────────────


class TestGemmaPreambleParser:
    """Reasoning models often emit bullet reasoning before fenced JSON.
    The SDK must tolerate this and still return a valid Pydantic
    instance."""

    def test_strips_fence_only_still_works(self):
        """Regression: the pre-existing ```json ... ``` fence case still
        parses — we haven't regressed the fast path."""
        canned = '```json\n{"category":"order","priority":"urgent"}\n```'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(_Classification, system="c", user="u")
        assert out.category == "order"
        assert out.priority == "urgent"

    def test_strips_preamble_bullets_then_parses_json(self):
        """The T4 headline case: Gemma-style bullets + fenced JSON. The
        walker finds the object after the preamble and returns it."""
        canned = (
            '*   Input: "classify this email"\n'
            "*   Constraint: return JSON with category + priority\n"
            "\n"
            "```json\n"
            '{"category":"order","priority":"normal"}\n'
            "```\n"
        )
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(_Classification, system="c", user="u")
        assert out.category == "order"
        assert out.priority == "normal"

    def test_picks_last_json_when_multiple(self):
        """Multiple JSON objects in the text → the LAST one wins. Matches
        The downstream ``services.extract_json`` heuristic: when a
        model thinks aloud with interim JSON sketches, the final object
        is the answer."""
        canned = (
            "First attempt:\n"
            '{"category":"spam","priority":"low"}\n'
            "\n"
            "Wait, re-reading the email, the correct answer is:\n"
            '{"category":"order","priority":"high"}\n'
        )
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(_Classification, system="c", user="u")
        assert out.category == "order"
        assert out.priority == "high"

    def test_invalid_json_raises_clear_error_with_raw_text(self):
        """Regression: malformed output still raises ``RuntimeError``
        and the raw text is included in the message so end-users can
        debug."""
        canned = "this is definitely not json at all"
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        with pytest.raises(RuntimeError) as excinfo:
            qm.instruction_form(_Classification, system="c", user="u")
        # Raw text is surfaced for debugging.
        assert "this is definitely not json at all" in str(excinfo.value)

    def test_balanced_brace_walker_handles_nested(self):
        """Nested objects and arrays surrounded by garbage still parse."""

        class Nested(BaseModel):
            user: dict
            tags: list[str]

        canned = (
            "Here is my analysis, which I'm presenting inline:\n"
            "Some random text { not actually json\n"
            '{"user":{"name":"Alice","roles":["admin","editor"]},'
            '"tags":["urgent","vip"]}\n'
            "trailing noise } more braces"
        )
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(Nested, system="c", user="u")
        assert out.user == {"name": "Alice", "roles": ["admin", "editor"]}
        assert out.tags == ["urgent", "vip"]

    def test_balanced_brace_walker_skips_braces_in_strings(self):
        """JSON strings containing literal ``{`` / ``}`` / ``[`` / ``]``
        don't confuse the walker.  This is the whole reason we use the
        stdlib :class:`json.JSONDecoder` instead of a hand-rolled brace
        counter — the decoder is string-aware."""

        class Message(BaseModel):
            text: str
            code: str

        canned = (
            "Reasoning: the user wants a templated response.\n"
            '{"text":"text with } brace and { opener","code":"if (x) { return y; }"}'
        )
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(Message, system="c", user="u")
        assert out.text == "text with } brace and { opener"
        assert out.code == "if (x) { return y; }"


# ── T5: dict schema support ──────────────────────────────────────────


class TestDictSchemaSupport:
    """``qm.instruction_form(schema={"type":"object",...})`` returns a
    dict validated against the provided JSON Schema (soft dep on
    ``jsonschema``)."""

    _DICT_SCHEMA = {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["category", "priority"],
        "additionalProperties": False,
    }

    def test_dict_schema_returns_dict(self):
        """Dict schema → dict return (no Pydantic instance)."""
        canned = '{"category":"order","priority":"high"}'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(self._DICT_SCHEMA, system="c", user="u")
        assert isinstance(out, dict)
        assert not isinstance(out, BaseModel)
        assert out == {"category": "order", "priority": "high"}

    def test_dict_schema_validates_via_jsonschema_if_installed(self):
        """Missing required field → ``RuntimeError`` whose message names
        the offending field.  ``pytest.importorskip`` guards the case
        where ``jsonschema`` isn't in the test venv (the verification
        step installs it explicitly)."""
        pytest.importorskip("jsonschema")
        # LLM "forgot" the ``priority`` required field — jsonschema
        # must catch this at validate() time.
        canned = '{"category":"order"}'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        with pytest.raises(RuntimeError) as excinfo:
            qm.instruction_form(self._DICT_SCHEMA, system="c", user="u")
        assert "priority" in str(excinfo.value)

    def test_dict_schema_without_jsonschema_returns_raw_with_warning(self):
        """When ``jsonschema`` isn't importable, the helper still returns
        the parsed dict but emits a ``UserWarning`` so callers know
        validation was silently skipped."""
        canned = '{"category":"order","priority":"high"}'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)

        # Simulate ``jsonschema`` not being installed. We blow away both
        # a cached import and the importlib loader for the name — the
        # helper does a fresh ``import jsonschema`` inside the dict path
        # so we only need to make that import fail.
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _raising_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("simulated: jsonschema not installed")
            return original_import(name, *args, **kwargs)

        # Also clear the sys.modules cache so the import really re-runs.
        saved = sys.modules.pop("jsonschema", None)
        try:
            with patch("builtins.__import__", side_effect=_raising_import):
                with pytest.warns(UserWarning, match="jsonschema"):
                    out = qm.instruction_form(self._DICT_SCHEMA, system="c", user="u")
        finally:
            if saved is not None:
                sys.modules["jsonschema"] = saved

        # The parsed dict still comes back — the helper doesn't refuse
        # just because validation is unavailable.
        assert out == {"category": "order", "priority": "high"}

    def test_invalid_schema_type_raises_typeerror(self):
        """Non-Pydantic, non-dict schemas raise a clear ``TypeError``.
        We cover two common footguns — a bare int and a bare string."""
        reg, _ = _mock_registry("{}")
        qm.configure(registry=reg)
        with pytest.raises(TypeError, match="BaseModel.*dict"):
            qm.instruction_form(42, system="c", user="u")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BaseModel.*dict"):
            qm.instruction_form("not a schema", system="c", user="u")  # type: ignore[arg-type]
