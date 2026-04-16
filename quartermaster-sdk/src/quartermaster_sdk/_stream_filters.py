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

import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, overload

from ._chunks import (
    Chunk,
    CustomChunk,
    ProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from ._typed_events import TypedEvent


logger = logging.getLogger(__name__)

_ALREADY_CONSUMED_MSG = "stream already consumed"


class _Stream:
    """Synchronous wrapper around the raw :class:`Chunk` generator.

    Exposes ``.tokens()``, ``.tool_calls()``, ``.progress()``,
    ``.custom(name=...)`` helpers plus a default ``__iter__`` that
    yields every chunk unmodified.

    The wrapper is single-pass: whichever consumer starts reading
    first wins, and any second consumer raises :class:`RuntimeError`.

    **v0.4.0 context-manager protocol (Sorex round-2 P1.2).** The
    wrapper implements ``__enter__`` / ``__exit__`` so consumers can
    write ``with qm.run.stream(...) as s: ...`` — on exit (normal,
    ``break``, ``return``, exception) the ``_on_exit`` callback fires,
    which the runner wires to :meth:`FlowRunner.stop`. That makes the
    "SSE tab-close keeps the agent burning Ollama tokens" bug
    impossible to trigger by accident: breaking out of the ``for``
    loop inside the ``with`` unwinds the context manager, which
    cancels the flow. The pre-v0.4.0 iterator-abandon path (on
    generator close / GC) still fires the same stop — the ``with``
    just makes the intent explicit and deterministic.
    """

    __slots__ = ("_source", "_consumed", "_on_exit", "_exit_called")

    def __init__(
        self,
        source: Iterator[Chunk],
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        self._source = source
        self._consumed = False
        # v0.4.0: optional callback fired on context-manager exit.
        # Runner wires this to ``FlowRunner.stop(flow_id)`` so the
        # engine short-circuits on the next ``_execute_node`` dispatch.
        # ``None`` keeps the wrapper usable as a plain iterator
        # without runner wiring (unit-test ergonomics).
        self._on_exit = on_exit
        self._exit_called = False

    def _claim(self) -> None:
        """Mark the stream as claimed by the first consumer; raise on re-use.

        Must be called before any consumer starts reading. Keeps every
        entry point (``__iter__``, ``tokens``, ``tool_calls``,
        ``progress``, ``custom``) in sync on a single flag.
        """
        if self._consumed:
            raise RuntimeError(_ALREADY_CONSUMED_MSG)
        self._consumed = True

    # ── Context-manager protocol (v0.4.0) ────────────────────────────
    def __enter__(self) -> "_Stream":
        """Enter the context manager — returns ``self`` so callers can
        iterate or call a filter directly::

            with qm.run.stream(graph, "hi") as s:
                for tok in s.tokens():
                    ...
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Fire the cancellation callback on every exit path.

        Normal completion, ``break``/``return``, or a raising exception
        — all three route here and all three call ``_on_exit`` exactly
        once. That's the behaviour integrators rely on: once control
        leaves the ``with`` block, the engine stops dispatching new
        nodes for this flow. The callback itself must be idempotent
        (calling ``runner.stop`` on an already-stopped flow is a no-op).
        """
        self._fire_on_exit()

    def _fire_on_exit(self) -> None:
        """Call the on-exit callback exactly once (guards double-fire)."""
        if self._exit_called:
            return
        self._exit_called = True
        if self._on_exit is None:
            return
        try:
            self._on_exit()
        except Exception:
            # Cancellation is best-effort — a failure to stop must not
            # replace the caller's original exception (if any) with a
            # secondary one from the callback. Log and move on.
            logger.exception("_Stream: on_exit callback raised")

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

    @overload
    def custom(self, name_or_schema: type[TypedEvent]) -> Iterator[Any]: ...

    @overload
    def custom(self, name_or_schema: str | None = None) -> Iterator[CustomChunk]: ...

    def custom(
        self,
        name_or_schema: str | type[TypedEvent] | None = None,
        *,
        name: str | None = None,
    ) -> Iterator[CustomChunk] | Iterator[Any]:
        """Yield :class:`CustomChunk` events, optionally filtered.

        **v0.4.0:** Accepts a :class:`TypedEvent` subclass in addition
        to the original ``name: str`` filter. When a ``TypedEvent``
        subclass is passed, the stream filters by the schema's
        ``name`` default and yields validated typed instances instead
        of raw :class:`CustomChunk` objects::

            for ev in stream.custom(SearchResultsEvent):
                print(ev.count, ev.query)   # typed fields

        ``name=None`` (the default) yields every custom chunk; passing
        a specific ``name`` string yields only the matching ones so UIs
        can subscribe to a single milestone stream without inspecting
        every payload.

        The ``name=`` keyword arg is a backwards-compatible alias for
        the positional *name_or_schema* when passed as a string.
        """
        # Backwards compat: ``stream.custom(name="x")`` from v0.3.0.
        resolved = name_or_schema if name_or_schema is not None else name
        self._claim()
        if isinstance(resolved, type) and issubclass(resolved, TypedEvent):
            return self._yield_typed(resolved)
        return self._yield_custom(resolved)

    # ── Internal helpers ──────────────────────────────────────────────
    def _yield_type(self, chunk_cls: type) -> Iterator:
        for chunk in self._source:
            if isinstance(chunk, chunk_cls):
                yield chunk

    def _yield_custom(self, name: str | None) -> Iterator[CustomChunk]:
        for chunk in self._source:
            if isinstance(chunk, CustomChunk) and (name is None or chunk.name == name):
                yield chunk

    def _yield_typed(self, schema: type[TypedEvent]) -> Iterator[Any]:
        """Yield validated typed instances for chunks matching a TypedEvent schema."""
        expected_name = schema.model_fields["name"].default
        for chunk in self._source:
            if isinstance(chunk, CustomChunk) and chunk.name == expected_name:
                yield schema(name=chunk.name, **chunk.payload)


class _AsyncStream:
    """Async analogue of :class:`_Stream`.

    Filter methods return async generators so consumers can write
    ``async for token in qm.arun.stream(...).tokens(): ...`` without
    materialising the whole stream first. Same single-pass guarantee
    as the sync variant — :class:`RuntimeError` on double consumption.

    **v0.4.0 async context-manager protocol (Sorex round-2 P1.2).**
    Implements ``__aenter__`` / ``__aexit__`` so ``async with
    qm.arun.stream(graph, user_input) as s: ...`` cancels the flow on
    any exit path. The on-exit callback is ``async`` because the
    async-runner may want to await a bounded join on the worker thread
    after calling ``runner.stop``; doing that from inside ``__aexit__``
    keeps the semantics identical to the sync variant.
    """

    __slots__ = ("_source", "_consumed", "_on_exit", "_exit_called")

    def __init__(
        self,
        source: AsyncIterator[Chunk],
        on_exit: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._source = source
        self._consumed = False
        # v0.4.0: optional async callback fired on ``__aexit__``.
        # Wired by ``qm.arun.stream`` to run the same
        # ``FlowRunner.stop(flow_id)`` dance the iterator ``finally``
        # block performs, so breaking out of the ``async for`` inside
        # the ``async with`` cancels the flow deterministically.
        self._on_exit = on_exit
        self._exit_called = False

    def _claim(self) -> None:
        if self._consumed:
            raise RuntimeError(_ALREADY_CONSUMED_MSG)
        self._consumed = True

    # ── Async context-manager protocol (v0.4.0) ──────────────────────
    async def __aenter__(self) -> "_AsyncStream":
        """Enter the async context manager — returns ``self``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Fire the cancellation callback on every exit path.

        Same contract as the sync variant: normal completion, ``break``
        out of ``async for``, and exceptions all call ``_on_exit``
        exactly once. The runner's implementation of the callback
        already handles the "already-stopped" case idempotently.
        """
        await self._fire_on_exit()

    async def _fire_on_exit(self) -> None:
        """Await the on-exit callback exactly once (guards double-fire)."""
        if self._exit_called:
            return
        self._exit_called = True
        if self._on_exit is None:
            return
        try:
            await self._on_exit()
        except Exception:
            logger.exception("_AsyncStream: on_exit callback raised")

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
