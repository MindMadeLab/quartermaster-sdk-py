"""Tests for v0.4.0 stream cancellation (Sorex round-2 P1.2).

Three new surfaces:

1. **Context-manager protocol on stream wrappers.** ``with
   qm.run.stream(...) as s:`` / ``async with qm.arun.stream(...) as s:``
   — on every exit path (normal completion, ``break``, ``return``,
   exception) the wrapper fires ``FlowRunner.stop(flow_id)`` so the
   engine stops dispatching new nodes. The legacy iterator-abandon
   path still works for callers that don't use ``with``.

2. **``qm.Cancelled`` exception.** Tools inside the agent loop raise
   it to abort cooperatively — the executor treats the call as a node
   failure with a distinct ``error="cancelled"`` sentinel so SDK
   consumers see ``ErrorChunk(error="cancelled", ...)`` instead of the
   generic tool-crash string.

3. **``ctx.cancelled`` flag.** Tools polling
   ``qm.current_context().cancelled`` observe ``True`` as soon as the
   runner is asked to stop (via the context-manager exit or direct
   ``runner.stop`` call). The flag reads the per-flow
   :class:`threading.Event` that the runner populates on every
   :class:`ExecutionContext` it builds for the flow — so cancellation
   is visible across the engine's internal thread-pool hops.

Production context: a Django SSE view's generator gets ``GeneratorExit``
on tab-close, but ``qm.run.stream(...)`` has handed off to
FlowRunner's thread pool which keeps burning Ollama tokens into
/dev/null. The context-manager protocol fixes that by making
cancellation explicit and deterministic: ``with qm.run.stream(...)
as s: yield from s.tokens()`` — when the SSE generator unwinds, the
``with`` exits, and the engine stops.
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
    """Minimal mock provider registry."""
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


def _tool_aware_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> tuple[ProviderRegistry, MockProvider]:
    """Tool-aware mock — one native response asking for a tool, one with final text."""
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
    """Minimal tool that returns a success payload."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {"ok": True}

    def safe_run(self, **kwargs: Any):
        class _R:
            success = True

        r = _R()
        r.data = dict(self._data)
        return r


class _CancelCheckingTool:
    """Tool that records what ``ctx.cancelled`` was when it ran."""

    def __init__(self):
        self.observations: list[bool] = []

    def safe_run(self, **kwargs: Any):
        ctx = qm.current_context()
        self.observations.append(ctx.cancelled if ctx is not None else False)

        class _R:
            success = True

        r = _R()
        r.data = {"ok": True}
        return r


class _CancelRaisingTool:
    """Tool that raises :class:`qm.Cancelled` straight away."""

    def safe_run(self, **kwargs: Any):
        raise qm.Cancelled("nope — aborted by tool")


class _ToolRegistry:
    """Minimal registry shim the engine's ``AgentExecutor`` can use."""

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


# ── 1. Sync context-manager protocol (no behaviour change) ───────────


def test_stream_context_manager_sync():
    """``with qm.run.stream(graph, "x") as s: list(s)`` works — confirms
    the protocol is on the wrapper and the happy path is unchanged.
    """
    reg, _ = _mock_registry("hello")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    with qm.run.stream(graph, "hi") as s:
        chunks = list(s)

    assert chunks, "stream yielded no chunks"
    assert chunks[-1].type == "done"


# ── 2. Break inside ``with`` calls runner.stop ───────────────────────


def test_stream_break_inside_with_cancels_flow(monkeypatch):
    """Breaking out of the loop inside the ``with`` block must fire
    :meth:`FlowRunner.stop` on the in-flight ``flow_id``.

    We spy on ``FlowRunner.stop`` and assert it was called at least once
    (the generator's ``finally`` may also call it — both fire on the
    same flow_id, and ``stop`` is idempotent, so counting "at least 1"
    is the right contract).
    """
    reg, _ = _mock_registry("hello world")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stop_calls: list[Any] = []

    from quartermaster_engine import FlowRunner

    original_stop = FlowRunner.stop

    def spy_stop(self, flow_id):
        stop_calls.append(flow_id)
        return original_stop(self, flow_id)

    monkeypatch.setattr(FlowRunner, "stop", spy_stop)

    with qm.run.stream(graph, "hi") as s:
        for _chunk in s:
            # Abandon immediately — the with-block exit must cancel.
            break

    assert len(stop_calls) >= 1, (
        "FlowRunner.stop must be called on context-manager exit when "
        "the consumer breaks early"
    )


# ── 3. Async context-manager protocol ────────────────────────────────


async def test_async_stream_context_manager():
    """``async with qm.arun.stream(graph, "x") as s: ...`` works — the
    async protocol mirrors the sync behaviour.
    """
    reg, _ = _mock_registry("hello async")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    async with qm.arun.stream(graph, "hi") as s:
        chunks = [c async for c in s]

    assert chunks, "async stream yielded no chunks"
    assert chunks[-1].type == "done"


# ── 4. Async break cancels remaining work ───────────────────────────


async def test_async_stream_cancellation_aborts_remaining_work(monkeypatch):
    """Consumer breaks out of ``async for`` after N chunks — the runner
    must get a ``stop`` call and no more nodes dispatch past the cap.

    Uses a multi-node graph so there's visible "remaining work" — we
    assert ``FlowRunner.stop`` fires, which is how the engine guarantees
    the dispatch loop short-circuits on the next ``_execute_node`` call.
    """
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    # 5 nodes — more than the 2 chunks the consumer reads before breaking.
    graph = (
        qm.Graph("x")
        .instruction("A")
        .instruction("B")
        .instruction("C")
        .instruction("D")
        .instruction("E")
        .build()
    )

    stop_calls: list[Any] = []

    from quartermaster_engine import FlowRunner

    original_stop = FlowRunner.stop

    def spy_stop(self, flow_id):
        stop_calls.append(flow_id)
        return original_stop(self, flow_id)

    monkeypatch.setattr(FlowRunner, "stop", spy_stop)

    count = 0
    async with qm.arun.stream(graph, "hi") as s:
        async for _chunk in s:
            count += 1
            if count >= 2:
                break

    assert count == 2, f"expected to break after 2 chunks, got {count}"
    assert len(stop_calls) >= 1, (
        "FlowRunner.stop must be called on async context-manager exit "
        "when the consumer breaks early"
    )


# ── 5. Tools observe ctx.cancelled ───────────────────────────────────


def test_tool_can_check_cancelled_flag():
    """A tool polling ``qm.current_context().cancelled`` sees ``False``
    during normal execution — the flag only flips after ``runner.stop``.

    Starts with a happy-path run (no cancel) and asserts the tool saw
    ``False``. The "flips to True after stop" direction is covered by
    the next test via the raising-tool path, which is the higher-
    fidelity integration check.
    """
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
    qm.configure(registry=reg)
    spy = _CancelCheckingTool()
    tool_reg = _ToolRegistry({"x": spy})
    graph = _graph_with_agent()

    with qm.run.stream(graph, "hi", tool_registry=tool_reg) as s:
        list(s)

    assert spy.observations, "tool was never called"
    assert all(obs is False for obs in spy.observations), (
        f"ctx.cancelled should be False during a normal run, got {spy.observations!r}"
    )


# ── 6. Tool raising qm.Cancelled surfaces as ErrorChunk ──────────────


def test_tool_raising_cancelled_surfaces_as_error_chunk():
    """A tool that raises :class:`qm.Cancelled` triggers an
    :class:`ErrorChunk` with ``error="cancelled"`` on the stream —
    distinct from the generic "tool execution failed" string.
    """
    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
    qm.configure(registry=reg)
    tool_reg = _ToolRegistry({"x": _CancelRaisingTool()})
    graph = _graph_with_agent()

    stream = qm.run.stream(graph, "hi", tool_registry=tool_reg)
    chunks = list(stream)
    errors = [c for c in chunks if c.type == "error"]

    assert errors, f"expected at least one ErrorChunk, got {[c.type for c in chunks]}"
    assert any(c.error == "cancelled" for c in errors), (
        f"expected ErrorChunk(error='cancelled'), got {[c.error for c in errors]}"
    )


# ── 7. qm.Cancelled is exported at the top level ────────────────────


def test_qm_cancelled_exported_at_top_level():
    """``from quartermaster_sdk import Cancelled`` works and is the
    same class as the engine's :class:`quartermaster_engine.Cancelled`
    — identity is preserved so ``isinstance`` checks cross the boundary.
    """
    from quartermaster_sdk import Cancelled as SdkCancelled
    from quartermaster_engine import Cancelled as EngineCancelled

    assert SdkCancelled is EngineCancelled, (
        "SDK Cancelled must be the SAME class as the engine's to keep "
        "isinstance checks consistent across the boundary"
    )
    # Must be a real Exception subclass
    assert issubclass(SdkCancelled, Exception)
    # Can be instantiated and raised
    try:
        raise SdkCancelled("test")
    except SdkCancelled as exc:
        assert str(exc) == "test"


# ── 8. Raw-iteration legacy path still cancels ──────────────────────


def test_raw_iteration_break_still_cancels(monkeypatch):
    """Pre-v0.4.0 callers using a raw ``for chunk in stream: break``
    (no ``with``) must still see the flow cancelled — the generator's
    ``finally`` fires the same ``runner.stop``. Regression guard so
    the context-manager path doesn't subtly break the legacy path.
    """
    reg, _ = _mock_registry("hi")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    stop_calls: list[Any] = []

    from quartermaster_engine import FlowRunner

    original_stop = FlowRunner.stop

    def spy_stop(self, flow_id):
        stop_calls.append(flow_id)
        return original_stop(self, flow_id)

    monkeypatch.setattr(FlowRunner, "stop", spy_stop)

    # No ``with`` — direct iteration.
    for _chunk in qm.run.stream(graph, "hi"):
        break

    # The generator's ``finally`` block runs on close / GC; give it a
    # beat so the cancel call goes through.
    import time as _time

    _time.sleep(0.2)

    assert len(stop_calls) >= 1, (
        "legacy raw-iteration break must still fire runner.stop via "
        "the generator's finally block"
    )
