"""Filtered stream iterators for :func:`run.stream` / :func:`arun.stream`.

v0.2.x callers wrote the same boilerplate over and over to pull specific
chunk types out of the raw :class:`Chunk` stream::

    for chunk in qm.run.stream(graph, "hi"):
        if chunk.type == "token":
            print(chunk.content, end="")

v0.3.0 wraps the underlying iterator in a small object that exposes
filter methods so consumers write::

    for token in qm.run.stream(graph, "hi").tokens():
        print(token, end="")

    for call in qm.run.stream(graph, "hi").tool_calls():
        ui.show_tool_card(call)

    for prog in qm.run.stream(graph, "hi").progress():
        ui.update_status(prog.message, prog.percent)

    for c in qm.run.stream(graph, "hi").custom(name="docs"):
        ui.add_doc(c.payload)

Raw iteration continues to work unchanged — the wrapper is itself an
iterator / async-iterator, so every existing ``for chunk in
run.stream(...)`` call site keeps its exact semantics::

    for chunk in qm.run.stream(graph, "hi"):   # still fine
        ...

Single-pass semantics
---------------------
The wrappers own the underlying generator. Calling a filter method
(or raw iteration) drains it — there is no replay. Calling a second
filter after the first, or raw-iterating after a filter, raises
:class:`RuntimeError` so the mistake surfaces loudly instead of
silently yielding zero chunks.

Design decision: ``tokens()`` yields ``str``
---------------------------------------------
All other filters yield the full chunk object because they carry
fields the consumer needs (``tool``, ``args``, ``message``,
``percent``, ``payload``, ``name``, …). ``tokens()`` is the one case
where the consumer almost always wants a bare string to concatenate
into a UI — the whole point of "30 chars to get streamed text" is to
cut out the ``.content`` hop. Callers who want the chunk itself can
still pattern-match on the raw stream.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from ._chunks import (
    Chunk,
    CustomChunk,
    ProgressChunk,
    TokenChunk,
    ToolCallChunk,
)


_ALREADY_CONSUMED_MSG = "stream already consumed"


class _Stream:
    """Synchronous wrapper around the raw :class:`Chunk` generator.

    Exposes ``.tokens()``, ``.tool_calls()``, ``.progress()``,
    ``.custom(name=...)`` helpers plus a default ``__iter__`` that
    yields every chunk unmodified.

    The wrapper is single-pass: whichever consumer starts reading
    first wins, and any second consumer raises :class:`RuntimeError`.
    """

    __slots__ = ("_source", "_consumed")

    def __init__(self, source: Iterator[Chunk]) -> None:
        self._source = source
        self._consumed = False

    def _claim(self) -> None:
        """Mark the stream as claimed by the first consumer; raise on re-use.

        Must be called before any consumer starts reading. Keeps every
        entry point (``__iter__``, ``tokens``, ``tool_calls``,
        ``progress``, ``custom``) in sync on a single flag.
        """
        if self._consumed:
            raise RuntimeError(_ALREADY_CONSUMED_MSG)
        self._consumed = True

    # ── Raw iteration (preserves pre-v0.3.0 behaviour) ────────────────
    def __iter__(self) -> Iterator[Chunk]:
        self._claim()
        return self._source

    # ── Filter methods ────────────────────────────────────────────────
    def tokens(self) -> Iterator[str]:
        """Yield ``chunk.content`` for every :class:`TokenChunk`.

        Yields ``str`` (not the chunk itself) because concatenation is
        the overwhelming common case.
        """
        self._claim()
        return self._yield_tokens()

    def _yield_tokens(self) -> Iterator[str]:
        for chunk in self._source:
            if isinstance(chunk, TokenChunk):
                yield chunk.content

    def tool_calls(self) -> Iterator[ToolCallChunk]:
        """Yield every :class:`ToolCallChunk` as it fires."""
        self._claim()
        return self._yield_type(ToolCallChunk)

    def progress(self) -> Iterator[ProgressChunk]:
        """Yield every :class:`ProgressChunk` as it fires."""
        self._claim()
        return self._yield_type(ProgressChunk)

    def custom(self, name: str | None = None) -> Iterator[CustomChunk]:
        """Yield :class:`CustomChunk` events, optionally filtered by ``name``.

        ``name=None`` (the default) yields every custom chunk; passing
        a specific ``name`` yields only the matching ones so UIs can
        subscribe to a single milestone stream without inspecting
        every payload.
        """
        self._claim()
        return self._yield_custom(name)

    # ── Internal helpers ──────────────────────────────────────────────
    def _yield_type(self, chunk_cls: type) -> Iterator:
        for chunk in self._source:
            if isinstance(chunk, chunk_cls):
                yield chunk

    def _yield_custom(self, name: str | None) -> Iterator[CustomChunk]:
        for chunk in self._source:
            if isinstance(chunk, CustomChunk) and (name is None or chunk.name == name):
                yield chunk


class _AsyncStream:
    """Async analogue of :class:`_Stream`.

    Filter methods return async generators so consumers can write
    ``async for token in qm.arun.stream(...).tokens(): ...`` without
    materialising the whole stream first. Same single-pass guarantee
    as the sync variant — :class:`RuntimeError` on double consumption.
    """

    __slots__ = ("_source", "_consumed")

    def __init__(self, source: AsyncIterator[Chunk]) -> None:
        self._source = source
        self._consumed = False

    def _claim(self) -> None:
        if self._consumed:
            raise RuntimeError(_ALREADY_CONSUMED_MSG)
        self._consumed = True

    # ── Raw async iteration ──────────────────────────────────────────
    def __aiter__(self) -> AsyncIterator[Chunk]:
        self._claim()
        return self._source

    # ── Filter methods ────────────────────────────────────────────────
    def tokens(self) -> AsyncIterator[str]:
        self._claim()
        return self._yield_tokens()

    async def _yield_tokens(self) -> AsyncIterator[str]:
        async for chunk in self._source:
            if isinstance(chunk, TokenChunk):
                yield chunk.content

    def tool_calls(self) -> AsyncIterator[ToolCallChunk]:
        self._claim()
        return self._yield_type(ToolCallChunk)

    def progress(self) -> AsyncIterator[ProgressChunk]:
        self._claim()
        return self._yield_type(ProgressChunk)

    def custom(self, name: str | None = None) -> AsyncIterator[CustomChunk]:
        self._claim()
        return self._yield_custom(name)

    # ── Internal helpers ──────────────────────────────────────────────
    async def _yield_type(self, chunk_cls: type) -> AsyncIterator:
        async for chunk in self._source:
            if isinstance(chunk, chunk_cls):
                yield chunk

    async def _yield_custom(self, name: str | None) -> AsyncIterator[CustomChunk]:
        async for chunk in self._source:
            if isinstance(chunk, CustomChunk) and (name is None or chunk.name == name):
                yield chunk


__all__ = ["_Stream", "_AsyncStream"]
