"""Tests for v0.3.0 filtered stream iterators.

v0.3.0 wraps the raw ``Iterator[Chunk]`` / ``AsyncIterator[Chunk]``
returned from ``qm.run.stream(...)`` / ``qm.arun.stream(...)`` in a
small object exposing filter methods — ``.tokens()``, ``.tool_calls()``,
``.progress()``, ``.custom(name=...)`` — so callers don't write the
``if chunk.type == "token": print(chunk.content)`` boilerplate.

The wrapper is still iterable / async-iterable, so every pre-0.3 code
path that did ``for chunk in qm.run.stream(...)`` keeps working
unchanged. This file pins both sides of the guarantee: filters yield
the right subset, and raw iteration still yields the full stream.

Single-pass contract: the wrapper owns the underlying generator. First
consumer drains it; a second filter (or raw iter after a filter)
raises ``RuntimeError("stream already consumed")``.
"""

from __future__ import annotations

from typing import Any

import openai  # noqa: F401 — eager import, see test_ollama_chat.py for context

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse, ToolCall


# ── Fixtures / helpers ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


def _mock_registry(
    text: str = "canned",
    native_text: str | None = None,
) -> tuple[ProviderRegistry, MockProvider]:
    """Minimal mock registry — copy of the helper from test_v020_surface.py.

    Token responses default to a single-chunk streamed reply and the
    native channel is primed with the same text so either path the
    engine picks yields a consistent string.
    """
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


def _multi_token_registry(tokens: list[str]) -> tuple[ProviderRegistry, MockProvider]:
    """Primed mock that streams ``tokens`` as one TokenResponse per token.

    ``MockProvider`` consumes its ``responses`` queue one entry per
    streamed call, so each entry becomes one ``TokenGenerated`` event
    which maps 1:1 to a :class:`TokenChunk`. That lets
    ``test_tokens_filter_yields_strings_only`` assert the exact string
    sequence produced by ``.tokens()``.
    """
    mock = MockProvider(
        responses=[TokenResponse(content=t, stop_reason="stop") for t in tokens],
        native_responses=[
            NativeResponse(
                text_content="".join(tokens),
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


# Tool-aware helpers — lifted from test_v022_tool_streaming.py so these
# tests stay self-contained (they're the lowest-friction way to
# provoke ToolCall / Progress / Custom chunks).


def _tool_aware_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> tuple[ProviderRegistry, MockProvider]:
    mock = MockProvider(
        responses=[TokenResponse(content=final_text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content="",
                thinking=[],
                tool_calls=[
                    ToolCall(
                        tool_name=tool_name,
                        tool_id="call_1",
                        parameters=tool_args or {"q": "hello"},
                    )
                ],
                stop_reason="tool_calls",
            ),
            NativeResponse(
                text_content=final_text,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            ),
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("mock", mock)
    reg.set_default_provider("mock")
    reg.set_default_model("mock", "test-model")
    return reg, mock


class _OkTool:
    """Plain tool that returns a structured payload. See the v0.2.2 suite."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {"ok": True}

    def safe_run(self, **kwargs: Any):
        class _R:
            success = True

        r = _R()
        r.data = dict(self._data)
        return r


class _ProgressEmittingTool:
    """Tool whose ``safe_run`` body emits progress + custom events.

    Mirrors the documented ``@tool()`` pattern: look up the current
    :class:`ExecutionContext` via :func:`current_context` and fire
    ``emit_progress`` / ``emit_custom``. The agent-executor binds the
    context before dispatching ``safe_run``, so these become real
    :class:`ProgressChunk` / :class:`CustomChunk` entries in the stream.
    """

    def __init__(
        self,
        progress_message: str | None = None,
        progress_percent: float | None = None,
        customs: list[tuple[str, dict[str, Any]]] | None = None,
    ):
        self._progress_message = progress_message
        self._progress_percent = progress_percent
        self._customs = list(customs or [])

    def safe_run(self, **kwargs: Any):
        ctx = qm.current_context()
        if ctx is not None:
            if self._progress_message is not None:
                ctx.emit_progress(
                    self._progress_message, percent=self._progress_percent
                )
            for name, payload in self._customs:
                ctx.emit_custom(name, payload)

        class _R:
            success = True

        r = _R()
        r.data = {"emitted": True}
        return r


class _ToolRegistry:
    """Minimal shim matching the agent executor's interface."""

    def __init__(self, tools: dict[str, Any], schema_name: str = "x"):
        self._tools = tools
        self._schema_name = schema_name

    def get(self, name: str):
        return self._tools[name]

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": self._schema_name,
                    "description": "stub",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]


def _graph_with_agent() -> Any:
    return (
        qm.Graph("chat").user().agent("Tooled", tools=["x"], capture_as="agent").build()
    )


# ── 1. .tokens() yields strings ──────────────────────────────────────


def test_tokens_filter_yields_strings_only():
    """A stream primed with three one-token responses yields exactly
    ``["a", "b", "c"]`` through ``.tokens()``.

    Proves two things at once:

    * filter returns ``str`` (not ``TokenChunk``) — the documented
      shortcut so callers don't write ``chunk.content`` in the hot path.
    * non-token chunks (node_start, node_finish, done) are filtered
      out silently.
    """
    reg, _ = _multi_token_registry(["a", "b", "c"])
    qm.configure(registry=reg)
    # Three instruction nodes — one streamed TokenResponse per node.
    graph = qm.Graph("x").instruction("A").instruction("B").instruction("C").build()

    stream = qm.run.stream(graph, "hi")
    tokens = list(stream.tokens())

    assert tokens == ["a", "b", "c"], (
        f"expected ['a', 'b', 'c'] from .tokens(), got {tokens!r}"
    )


# ── 2. .tool_calls() yields ToolCallChunks ───────────────────────────


def test_tool_calls_filter_yields_tool_call_chunks():
    """When a tool fires during the stream, ``.tool_calls()`` yields
    the typed :class:`ToolCallChunk` — not the paired ``ToolResultChunk``,
    not the surrounding node lifecycle events.
    """
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hello"})
    qm.configure(registry=reg)
    tool_reg = _ToolRegistry({"x": _OkTool({"ok": True})})
    graph = _graph_with_agent()

    stream = qm.run.stream(graph, "hi", tool_registry=tool_reg)
    calls = list(stream.tool_calls())

    assert len(calls) == 1, f"expected exactly 1 tool_call, got {calls}"
    assert all(isinstance(c, qm.ToolCallChunk) for c in calls), (
        f"all entries must be ToolCallChunk, got {[type(c) for c in calls]}"
    )
    assert calls[0].tool == "x"
    assert calls[0].args == {"q": "hello"}


# ── 3. .progress() yields ProgressChunks ─────────────────────────────


def test_progress_filter():
    """A tool that calls ``current_context().emit_progress(...)`` inside
    ``safe_run`` surfaces as a :class:`ProgressChunk` on the stream —
    ``.progress()`` pulls it out verbatim.
    """
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
    qm.configure(registry=reg)
    tool = _ProgressEmittingTool(progress_message="searching", progress_percent=0.5)
    tool_reg = _ToolRegistry({"x": tool})
    graph = _graph_with_agent()

    stream = qm.run.stream(graph, "hi", tool_registry=tool_reg)
    progress_chunks = list(stream.progress())

    assert len(progress_chunks) == 1, (
        f"expected 1 progress chunk, got {progress_chunks}"
    )
    assert isinstance(progress_chunks[0], qm.ProgressChunk)
    assert progress_chunks[0].message == "searching"
    assert progress_chunks[0].percent == 0.5


# ── 4. .custom(name=...) filters by name ─────────────────────────────


def test_custom_filter_by_name():
    """Two ``emit_custom`` calls ("a", "b") surface as two
    :class:`CustomChunk` entries; ``.custom(name="a")`` yields only the
    matching one.
    """
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
    qm.configure(registry=reg)
    tool = _ProgressEmittingTool(customs=[("a", {"v": 1}), ("b", {"v": 2})])
    tool_reg = _ToolRegistry({"x": tool})
    graph = _graph_with_agent()

    stream = qm.run.stream(graph, "hi", tool_registry=tool_reg)
    a_only = list(stream.custom(name="a"))

    assert len(a_only) == 1, f"expected 1 chunk named 'a', got {a_only}"
    assert isinstance(a_only[0], qm.CustomChunk)
    assert a_only[0].name == "a"
    assert a_only[0].payload == {"v": 1}


def test_custom_filter_without_name_yields_all():
    """``.custom()`` with no ``name=`` yields every custom chunk."""
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
    qm.configure(registry=reg)
    tool = _ProgressEmittingTool(customs=[("a", {"v": 1}), ("b", {"v": 2})])
    tool_reg = _ToolRegistry({"x": tool})
    graph = _graph_with_agent()

    stream = qm.run.stream(graph, "hi", tool_registry=tool_reg)
    all_customs = list(stream.custom())

    assert len(all_customs) == 2
    names = {c.name for c in all_customs}
    assert names == {"a", "b"}


# ── 5. Raw iteration still works (regression guard) ──────────────────


def test_raw_iteration_still_works():
    """Pre-v0.3 code paths that did ``for chunk in qm.run.stream(...)``
    must keep yielding the full chunk sequence unchanged — the wrapper
    object is itself iterable and delegates to the underlying generator.
    """
    reg, _ = _mock_registry("streaming words")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.run.stream(graph, "hi")
    chunks = [c for c in stream]
    types = [c.type for c in chunks]

    # Same terminal guarantee as test_v020_surface — DoneChunk last,
    # populated Result, at least one node lifecycle pair.
    assert types[-1] == "done"
    last = chunks[-1]
    assert isinstance(last, qm.DoneChunk)
    assert last.result.success
    assert "node_start" in types
    assert "node_finish" in types


# ── 6. Single-pass contract ──────────────────────────────────────────


def test_double_consumption_raises():
    """Calling a filter twice on the same wrapper raises ``RuntimeError``
    with the documented ``"stream already consumed"`` message — so the
    mistake shows up immediately instead of silently yielding zero
    chunks from a drained generator.
    """
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.run.stream(graph, "hi")
    # First consumer drains the generator — we don't care what it yields.
    list(stream.tokens())

    with pytest.raises(RuntimeError, match="stream already consumed"):
        list(stream.tokens())


def test_filter_then_raw_iter_raises():
    """Any two entry points on the same wrapper share the single-pass
    flag — filter-then-raw-iter trips the same guard."""
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.run.stream(graph, "hi")
    list(stream.tokens())

    with pytest.raises(RuntimeError, match="stream already consumed"):
        for _ in stream:
            pass


def test_raw_iter_then_filter_raises():
    """Inverse of the previous — raw iteration first, filter second."""
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.run.stream(graph, "hi")
    list(stream)

    with pytest.raises(RuntimeError, match="stream already consumed"):
        list(stream.tokens())


# ── 7. Async equivalents ─────────────────────────────────────────────


async def test_async_tokens_filter():
    """Async analogue of test_tokens_filter_yields_strings_only —
    ``async for token in qm.arun.stream(...).tokens()`` yields ``str``."""
    reg, _ = _multi_token_registry(["a", "b", "c"])
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("A").instruction("B").instruction("C").build()

    stream = qm.arun.stream(graph, "hi")
    tokens = [t async for t in stream.tokens()]

    assert tokens == ["a", "b", "c"], (
        f"expected ['a', 'b', 'c'] from async .tokens(), got {tokens!r}"
    )


async def test_async_raw_iteration():
    """Regression: ``async for chunk in qm.arun.stream(...)`` continues
    to yield the raw full chunk sequence — the async wrapper is itself
    async-iterable and delegates to the underlying async generator.
    """
    reg, _ = _mock_registry("streaming async words")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.arun.stream(graph, "hi")
    chunks = [c async for c in stream]
    types = [c.type for c in chunks]

    assert types[-1] == "done"
    last = chunks[-1]
    assert isinstance(last, qm.DoneChunk)
    assert last.result.success
    assert "node_start" in types
    assert "node_finish" in types


async def test_async_double_consumption_raises():
    """The single-pass guarantee applies to the async wrapper too."""
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stream = qm.arun.stream(graph, "hi")
    _ = [t async for t in stream.tokens()]

    with pytest.raises(RuntimeError, match="stream already consumed"):
        _ = [t async for t in stream.tokens()]
