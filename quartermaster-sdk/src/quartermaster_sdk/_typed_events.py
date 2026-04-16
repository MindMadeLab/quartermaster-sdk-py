"""Typed CustomEvent schemas — Pydantic-backed type-safe event classes.

v0.4.0 adds :class:`TypedEvent`, a thin Pydantic ``BaseModel`` subclass
that doubles as both the emit payload and the stream-side filter key.

Subclasses define a ``name`` field whose default value is used as the
``CustomEvent`` discriminator::

    class SearchResultsEvent(qm.TypedEvent):
        name: str = "search_results"
        count: int
        query: str

    # emit
    ctx.emit_custom(SearchResultsEvent(count=5, query=q))

    # consume — typed instances, not raw CustomChunks
    for ev in stream.custom(SearchResultsEvent):
        print(f"Found {ev.count} results for '{ev.query}'")

``extra="forbid"`` in the model config catches typos in field names at
construction time so they don't silently vanish into the wire format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TypedEvent(BaseModel):
    """Base class for typed custom events.

    Subclass this with a ``name: str = "your_event_name"`` field and any
    additional typed payload fields. The ``name`` default is used as the
    ``CustomEvent`` discriminator when emitting and filtering.

    ``extra="forbid"`` ensures that typos in field names raise a
    ``ValidationError`` at construction time rather than being silently
    dropped.
    """

    model_config = ConfigDict(extra="forbid")

    name: str


__all__ = ["TypedEvent"]
