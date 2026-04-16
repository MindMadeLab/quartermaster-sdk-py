"""Tests for v0.3.0 structured trace.

Covers the :class:`quartermaster_sdk.Trace` / :class:`NodeTrace` surface
that every :class:`Result` now carries:

* ``result.trace.text`` — concatenated :class:`TokenGenerated` tokens
* ``result.trace.tool_calls`` — dicts extracted from
  :class:`ToolCallFinished` events
* ``result.trace.by_node["name"]`` — per-node :class:`NodeTrace` buckets
* ``result.trace.progress`` / ``result.trace.custom(name=...)`` — event
  filter shortcuts
* ``result.trace.as_jsonl()`` — JSONL export
* Populated for both streaming (``DoneChunk.result``) and non-streaming
  runs
* ``result.trace.duration_seconds`` — wall-clock

The :class:`MockProvider` / ``_OkTool`` / ``_ToolRegistry`` shim pattern
is lifted wholesale from :mod:`tests.test_v022_tool_streaming`.
"""

from __future__ import annotations

import json
from typing import Any

import openai  # noqa: F401 — eager import, see test_ollama_chat.py for context

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse, ToolCall


# ── Test fixtures ────────────────────────────────────────────────────


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
    """Mirror of :func:`test_v020_surface._mock_registry`.

    Primes both the streamed-token and the native-response channels so
    a graph built without ``agent()`` gets the same string from either
    path the engine might take.
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


def _tool_aware_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> tuple[ProviderRegistry, MockProvider]:
    """Mirror of :func:`test_v022_tool_streaming._tool_aware_registry`."""
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
    """Same success-envelope tool used by v0.2.2 tests."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {"ok": True, "echo": "hello"}

    def safe_run(self, **kwargs: Any):
        class _R:
            success = True

        r = _R()
        r.data = dict(self._data)
        return r


class _ToolRegistry:
    """Minimal shim matching the agent executor's expected interface —
    ``.get(name)`` plus ``.to_openai_tools()``.

    Keeps the same contract as ``test_v022_tool_streaming._ToolRegistry``
    so a tool registered under its bare public name can be invoked by
    name from a :class:`MockProvider`-driven agent loop.
    """

    def __init__(self, tools: dict[str, Any], schema_name: str = "x"):
        self._tools = tools
        self._schema_name = schema_name

    def get(self, name: str):
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(name) from exc

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


# ── 1. ``trace.text`` aggregates all tokens ──────────────────────────


def test_trace_text_aggregates_all_tokens():
    """The mock provider streams ``"hello world"`` — the trace's
    :attr:`Trace.text` must be exactly that string, concatenated from
    every :class:`TokenGenerated` event regardless of which node emitted
    it."""
    reg, _ = _mock_registry("hello world")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    result = qm.run(graph, "hi")

    assert isinstance(result.trace, qm.Trace)
    # The mock's streamed response is a single TokenResponse — the
    # engine delivers it as one token, so ``trace.text`` equals the
    # whole canned reply. Using ``result.text`` as the oracle keeps
    # this robust to how the mock is tokenised internally.
    assert result.trace.text == result.text == "hello world"


# ── 2. Tool-call records extracted from ToolCallFinished events ──────


def test_trace_tool_calls_extracted_from_events():
    """One tool call → exactly one :class:`ToolCallFinished` event →
    exactly one dict in :attr:`Trace.tool_calls` carrying every field
    the spec requires (``tool``, ``arguments``, ``result``, ``raw``,
    ``error``, ``iteration``)."""
    reg, _ = _tool_aware_registry(
        tool_name="x", tool_args={"q": "world"}, final_text="ack"
    )
    qm.configure(registry=reg)

    tool_reg = _ToolRegistry({"x": _OkTool({"value": 42})})
    graph = (
        qm.Graph("chat").user().agent("Tooled", tools=["x"], capture_as="agent").build()
    )

    result = qm.run(graph, "hi", tool_registry=tool_reg)
    assert result.success, result.error

    tool_calls = result.trace.tool_calls
    assert isinstance(tool_calls, list), f"expected list, got {type(tool_calls)}"
    assert len(tool_calls) == 1, f"expected exactly 1 tool call, got {tool_calls}"

    entry = tool_calls[0]
    expected_keys = {"tool", "arguments", "result", "raw", "error", "iteration"}
    assert set(entry) == expected_keys, (
        f"keys mismatch — missing {expected_keys - set(entry)}, "
        f"extra {set(entry) - expected_keys}"
    )
    assert entry["tool"] == "x"
    assert entry["arguments"] == {"q": "world"}
    assert entry["error"] is None
    assert entry["raw"] == {"value": 42}
    assert isinstance(entry["iteration"], int)


# ── 3. by_node buckets events by node name ──────────────────────────


def test_trace_by_node_buckets_events_by_name():
    """Two named instruction nodes; each :class:`NodeTrace.events` must
    contain only the events from its own node (the ``NodeStarted`` /
    ``NodeFinished`` pair plus any tokens scoped to it).

    The :class:`MockProvider` is primed with two responses so the engine
    can serve each instruction node independently — the test asserts
    that the two per-node buckets in :attr:`Trace.by_node` carry
    disjoint event lists keyed by the two node names.
    """
    mock = MockProvider(
        responses=[
            TokenResponse(content="first-out", stop_reason="stop"),
            TokenResponse(content="second-out", stop_reason="stop"),
        ],
        native_responses=[
            NativeResponse(
                text_content="first-out",
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            ),
            NativeResponse(
                text_content="second-out",
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            ),
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    qm.configure(registry=reg)

    # The ``instruction(name, ...)`` first arg IS the node's name — no
    # second ``name=`` kwarg. The trace buckets on that same string
    # via :class:`NodeStarted.node_name` / :class:`NodeFinished.node_name`.
    graph = qm.Graph("x").instruction("node_a").instruction("node_b").build()

    result = qm.run(graph, "hi")
    assert result.success, result.error

    assert "node_a" in result.trace.by_node, (
        f"node_a missing from by_node keys: {list(result.trace.by_node)}"
    )
    assert "node_b" in result.trace.by_node, (
        f"node_b missing from by_node keys: {list(result.trace.by_node)}"
    )

    node_a_trace = result.trace.by_node["node_a"]
    node_b_trace = result.trace.by_node["node_b"]

    # Each per-node bucket carries at least the NodeStarted and
    # NodeFinished events for that node plus its own tokens — we
    # don't hard-code counts (the engine may emit other events too)
    # but the two buckets must be disjoint by identity.
    assert isinstance(node_a_trace, qm.NodeTrace)
    assert node_a_trace.node_name == "node_a"
    assert isinstance(node_b_trace, qm.NodeTrace)
    assert node_b_trace.node_name == "node_b"

    # No event in node_a's bucket may appear in node_b's bucket.
    a_ids = {id(e) for e in node_a_trace.events}
    b_ids = {id(e) for e in node_b_trace.events}
    assert a_ids.isdisjoint(b_ids), (
        "by_node buckets must be disjoint — event overlap indicates "
        "the latch logic leaked events across nodes"
    )

    # Per-node text accessor must scope correctly.  Tokens emitted for
    # node_a are the ones the engine generated for that node; combining
    # them must equal that node's share of ``result.trace.text``.
    combined = node_a_trace.text + node_b_trace.text
    assert combined == result.trace.text, (
        f"per-node text should concat to trace.text: "
        f"got {combined!r} vs {result.trace.text!r}"
    )


# ── 4. Progress + custom filters ─────────────────────────────────────


def test_trace_progress_and_custom_filters():
    """A tool emits one :class:`ProgressEvent` and one
    :class:`CustomEvent`.  After the run:

    * ``result.trace.progress`` must carry the progress event
    * ``result.trace.custom(name="x")`` must carry only the matching
      custom event (here named ``"x"``) — zero matches for a different
      name.
    """

    # Tool that emits progress + custom via the contextvar on each call.
    class _EmittingTool:
        """Emit progress/custom when invoked by the agent loop."""

        def safe_run(self, **kwargs: Any):
            ctx = qm.current_context()
            # Contract: the agent executor runs inside
            # ``bind_current_context(...)`` so ctx is non-None here.
            assert ctx is not None, (
                "current_context() returned None inside tool — "
                "contextvar propagation is broken"
            )
            ctx.emit_progress("emitting", percent=0.5)
            ctx.emit_custom("x", {"hello": "world"})

            class _R:
                success = True

            r = _R()
            r.data = {"ok": True}
            return r

    reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "q"}, final_text="ok")
    qm.configure(registry=reg)
    tool_reg = _ToolRegistry({"x": _EmittingTool()})
    graph = (
        qm.Graph("chat").user().agent("Tooled", tools=["x"], capture_as="agent").build()
    )

    result = qm.run(graph, "hi", tool_registry=tool_reg)
    assert result.success, result.error

    # Progress filter: one event, with the message & percent we emitted.
    progress = result.trace.progress
    assert len(progress) == 1, f"expected 1 progress event, got {progress}"
    assert progress[0].message == "emitting"
    assert progress[0].percent == 0.5

    # Custom filter by name: only matching events come back.
    custom_x = result.trace.custom(name="x")
    assert len(custom_x) == 1, f"expected 1 custom event named 'x', got {custom_x}"
    assert custom_x[0].name == "x"
    assert custom_x[0].payload == {"hello": "world"}

    # Different name filter → zero matches.
    assert result.trace.custom(name="other") == [], (
        "filter by non-matching name must return []"
    )

    # No-arg filter returns every custom event (just the one here).
    assert len(result.trace.custom()) == 1


# ── 5. JSONL round-trip ──────────────────────────────────────────────


def test_trace_as_jsonl_roundtrips():
    """:meth:`Trace.as_jsonl` produces one valid JSON object per line,
    and the number of lines equals the number of events in the trace.
    """
    reg, _ = _mock_registry("abc")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    result = qm.run(graph, "hi")
    jsonl = result.trace.as_jsonl()

    # Every line must parse as JSON.
    lines = jsonl.splitlines()
    # v0.4.0: as_jsonl() may prepend a header line when user_input is set
    # (e.g. {"_meta": "trace_header", "user_input": "hi"}). Filter it out
    # for the event-count assertion.
    event_lines = [line for line in lines if not json.loads(line).get("_meta")]
    assert len(event_lines) == len(result.trace.events), (
        f"event line count {len(event_lines)} != event count {len(result.trace.events)}"
    )
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict), f"line not a JSON object: {line}"
        if parsed.get("_meta"):
            continue  # header line — no flow_id expected
        # ``flow_id`` is required on every FlowEvent (it's the only
        # base-class field) — its presence is the cheapest sanity check
        # that ``asdict`` serialised the dataclass correctly.
        assert "flow_id" in parsed, f"line missing flow_id: {line}"


# ── 6. Streaming run populates trace on DoneChunk.result ────────────


def test_trace_populated_for_streaming_run():
    """Drain :func:`run.stream` to the terminal :class:`DoneChunk` and
    assert its :attr:`Result.trace.text` matches the streamed canned
    reply — the streaming path installs the same collector as the
    sync path so both surfaces expose the trace.
    """
    reg, _ = _mock_registry("streamed reply")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    chunks = list(qm.run.stream(graph, "hi"))

    done = chunks[-1]
    assert isinstance(done, qm.DoneChunk)
    assert isinstance(done.result.trace, qm.Trace)
    assert done.result.trace.text == "streamed reply"
    # The trace must also carry events (i.e. the collector actually ran
    # — a default-constructed Trace() has an empty event list, which
    # would pass the .text check above trivially).
    assert len(done.result.trace.events) > 0, (
        "streaming path produced a Trace with zero events — "
        "on_event collector didn't run"
    )


# ── 7. Duration recorded ─────────────────────────────────────────────


def test_trace_duration_seconds_recorded():
    """:attr:`Trace.duration_seconds` is populated to a positive value
    after a run — either mirroring :attr:`Result.duration_seconds` or
    falling back to the runner's wall-clock measurement.
    """
    reg, _ = _mock_registry("x")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    result = qm.run(graph, "hi")

    assert result.trace.duration_seconds > 0, (
        f"expected positive duration, got {result.trace.duration_seconds!r}"
    )
