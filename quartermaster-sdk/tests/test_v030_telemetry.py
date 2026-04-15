"""Tests for the v0.3.0 OpenTelemetry instrumentation module.

Covers four guarantees of :mod:`quartermaster_sdk.telemetry`:

1. After :func:`telemetry.instrument`, every node executed by
   :func:`qm.run` produces a ``qm.node.<name>`` OpenTelemetry span.
2. Tool calls during an agent node produce a ``qm.tool.<name>`` span
   parented to the agent's ``qm.node.<name>`` span.
3. :func:`telemetry.uninstrument` removes the listener so subsequent
   flows emit zero spans.
4. ``ProgressEvent`` from inside a tool surfaces as an OTEL span event
   with name ``progress`` on the active node span.

All tests run only when the optional ``opentelemetry`` extra is
installed — skipped cleanly otherwise so the bare-bones SDK test suite
stays portable.
"""

from __future__ import annotations

from typing import Any

import openai  # noqa: F401 — eager import; see test_ollama_chat.py for context

import pytest

# Skip the entire module if OTEL isn't installed — keeps the bare SDK
# test run green for users who haven't pulled the [telemetry] extra.
pytest.importorskip("opentelemetry")

# Imported here so missed imports surface as ImportError, not skip.
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import quartermaster_sdk as qm
from quartermaster_engine.context.current_context import current_context
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import (
    NativeResponse,
    TokenResponse,
    ToolCall,
)
from quartermaster_sdk import telemetry
from quartermaster_sdk import _listeners as _listeners_mod


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Fresh in-memory span exporter wired to a private TracerProvider.

    A private provider per test prevents cross-test span pollution and
    avoids stomping on whatever the global provider is configured for.
    """
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    yield exp, provider
    exp.clear()


@pytest.fixture(autouse=True)
def _reset_state(exporter):
    """Reset SDK config + telemetry state between tests."""
    qm.reset_config()
    # Make sure a stray instrument() from a previous test doesn't leak.
    telemetry.uninstrument()
    _listeners_mod.clear()
    yield
    telemetry.uninstrument()
    _listeners_mod.clear()
    qm.reset_config()


def _mock_registry(text: str = "ok") -> ProviderRegistry:
    """Single-shot mock provider — same shape as test_v022_async_runner."""
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=text,
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
    return reg


def _agent_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> ProviderRegistry:
    """Mock provider primed for a 2-turn agent loop (one tool call)."""
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
    return reg


class _OkTool:
    """Tool that always succeeds with a fixed payload."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {"ok": True}

    def safe_run(self, **kwargs: Any):
        class _R:
            success = True

        r = _R()
        r.data = dict(self._data)
        return r


class _ProgressTool:
    """Tool that emits a ``ProgressEvent`` while running."""

    def safe_run(self, **kwargs: Any):
        ctx = current_context()
        if ctx is not None:
            ctx.emit_progress("loading", percent=0.5, data={"step": "fetch"})

        class _R:
            success = True

        r = _R()
        r.data = {"ok": True}
        return r


class _ToolRegistry:
    """Minimal shim matching the agent executor's expected interface."""

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


# ── 1. Node spans ────────────────────────────────────────────────────


def test_instrument_creates_span_per_node(exporter):
    """Each executed node yields a ``qm.node.<name>`` OTEL span.

    A two-instruction graph executes Start → Instruction → Instruction →
    End plus the auto-injected user node, so we expect at least the
    two named instruction spans plus a root flow span.
    """
    exp, provider = exporter
    qm.configure(registry=_mock_registry("ok"))
    telemetry.instrument(tracer_provider=provider)

    graph = qm.Graph("x").instruction("first").instruction("second").build()
    result = qm.run(graph, "hi")
    assert result.success, result.error

    spans = exp.get_finished_spans()
    names = [s.name for s in spans]
    # Root flow span always present once a single event fired.
    assert "qm.flow" in names, f"missing qm.flow span; got {names}"
    # Both named instruction nodes produced their own spans.
    assert "qm.node.first" in names, f"missing first node span; got {names}"
    assert "qm.node.second" in names, f"missing second node span; got {names}"

    # Each node span carries the GenAI semconv attributes.
    for span in spans:
        if span.name.startswith("qm.node."):
            assert span.attributes.get("gen_ai.system") == "quartermaster"
            assert "gen_ai.operation.name" in span.attributes


# ── 2. Tool spans nested under agent ─────────────────────────────────


def test_instrument_creates_tool_spans_under_agent_node(exporter):
    """Tool spans are parented to the enclosing agent node span."""
    exp, provider = exporter
    qm.configure(registry=_agent_registry(tool_name="x", final_text="done"))
    telemetry.instrument(tracer_provider=provider)

    tool_reg = _ToolRegistry({"x": _OkTool({"echo": "hi"})})
    graph = (
        qm.Graph("chat").user().agent("Tooled", tools=["x"], capture_as="agent").build()
    )

    result = qm.run(graph, "hi", tool_registry=tool_reg)
    assert result.success, result.error

    spans = exp.get_finished_spans()
    names = [s.name for s in spans]

    # Tool span exists.
    tool_spans = [s for s in spans if s.name == "qm.tool.x"]
    assert tool_spans, f"no qm.tool.x span; got {names}"
    tool_span = tool_spans[0]

    # Tool span carries gen_ai.tool.* attributes.
    assert tool_span.attributes.get("gen_ai.tool.name") == "x"
    assert "gen_ai.tool.call.arguments" in tool_span.attributes

    # The agent node span exists and is the tool span's parent.
    agent_spans = [s for s in spans if s.name == "qm.node.Tooled"]
    assert agent_spans, f"no qm.node.Tooled span; got {names}"
    agent_span = agent_spans[0]

    assert tool_span.parent is not None, "tool span has no parent context"
    assert tool_span.parent.span_id == agent_span.context.span_id, (
        "tool span parent must be the agent node span; "
        f"tool.parent.span_id={tool_span.parent.span_id} "
        f"agent.span_id={agent_span.context.span_id}"
    )


# ── 3. Uninstrument removes the listener ─────────────────────────────


def test_uninstrument_removes_listener(exporter):
    """After uninstrument(), no spans flow to the exporter."""
    exp, provider = exporter
    qm.configure(registry=_mock_registry("ok"))

    telemetry.instrument(tracer_provider=provider)
    telemetry.uninstrument()

    graph = qm.Graph("x").instruction("One").build()
    result = qm.run(graph, "hi")
    assert result.success

    spans = exp.get_finished_spans()
    assert len(spans) == 0, (
        f"expected zero spans after uninstrument, got {[s.name for s in spans]}"
    )


# ── 4. Progress events become span events ────────────────────────────


def test_progress_event_becomes_span_event(exporter):
    """Tool-emitted ProgressEvent appears as a span event named ``progress``."""
    exp, provider = exporter
    qm.configure(registry=_agent_registry(tool_name="p", final_text="done"))
    telemetry.instrument(tracer_provider=provider)

    tool_reg = _ToolRegistry({"p": _ProgressTool()}, schema_name="p")
    graph = (
        qm.Graph("chat").user().agent("Loader", tools=["p"], capture_as="agent").build()
    )
    result = qm.run(graph, "hi", tool_registry=tool_reg)
    assert result.success, result.error

    spans = exp.get_finished_spans()

    # Search every node-or-tool span for a "progress" event — it lives
    # on whichever span was active when ``emit_progress`` fired.
    progress_events: list[Any] = []
    for span in spans:
        for evt in span.events:
            if evt.name == "progress":
                progress_events.append((span.name, evt))

    assert progress_events, (
        f"no 'progress' span event found; spans={[(s.name, [e.name for e in s.events]) for s in spans]}"
    )
    # Payload made it through the OTEL boundary.
    span_name, evt = progress_events[0]
    assert evt.attributes.get("message") == "loading"
    assert evt.attributes.get("percent") == 0.5
