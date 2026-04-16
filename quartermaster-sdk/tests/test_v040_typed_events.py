"""Tests for v0.4.0 TypedEvent — Pydantic-backed typed custom events.

TypedEvent is a thin BaseModel subclass with ``extra="forbid"`` that
lets integrators define typed payloads for ``ctx.emit_custom(...)`` and
filter them back out on the stream side with ``stream.custom(MyEvent)``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quartermaster_sdk._typed_events import TypedEvent


# ── 1. Subclass carries name + payload fields ──────────────────────────


class SearchEvent(TypedEvent):
    name: str = "search"
    count: int
    query: str


def test_typed_event_subclass_has_name_field() -> None:
    """Instantiate a TypedEvent subclass and verify that ``name`` and
    the declared payload fields are accessible with expected values.
    """
    event = SearchEvent(count=5, query="hello")

    assert event.name == "search"
    assert event.count == 5
    assert event.query == "hello"


# ── 2. Pydantic validation fires on wrong types ───────────────────────


def test_typed_event_validates_via_pydantic() -> None:
    """Passing an incompatible type (e.g. a string where ``int`` is
    expected) must raise ``ValidationError`` at construction time —
    the ``extra="forbid"`` config also catches unknown fields.
    """
    with pytest.raises(ValidationError):
        SearchEvent(count="not-an-int", query="ok")  # type: ignore[arg-type]


# ── 3. Top-level import ────────────────────────────────────────────────


def test_typed_event_importable_from_top_level() -> None:
    """``from quartermaster_sdk import TypedEvent`` resolves and is
    the same class as the private-module original.
    """
    from quartermaster_sdk import TypedEvent as TopLevel

    assert TopLevel is TypedEvent
