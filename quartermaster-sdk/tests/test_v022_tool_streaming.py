"""Tests for v0.2.2 live tool-call streaming.

Covers the four new surface guarantees introduced in v0.2.2:

1. The engine's new ``ToolCallStarted`` / ``ToolCallFinished`` events
   arrive during ``qm.run.stream(...)`` as typed
   :class:`ToolCallChunk` / :class:`ToolResultChunk` chunks, in the
   correct ordering relative to ``NodeStartChunk`` / ``NodeFinishChunk``
   / ``DoneChunk``.
2. A tool that raises surfaces via ``ToolResultChunk.error`` (non-None)
   while ``.result`` carries the ``[ERROR: ...]`` sentinel string the
   model sees on the next turn.
3. The non-streaming ``qm.run(...)`` path populates the agent node's
   ``NodeResult.data["tool_calls"]`` with the same shape as the streaming
   events — so callers reading ``result["agent"].data["tool_calls"]``
   after a sync run get the exact same record the stream surfaced.
4. Provider-side tool-name prefixes (``default_api:``, ``functions:``,
   ``mcp:``) are stripped in BOTH the streamed ``ToolCallChunk.tool`` AND
   the sync ``data["tool_calls"][0]["tool"]`` — so UIs never show the
   internal namespacing.

A :class:`MockProvider` primed with two ``NativeResponse`` entries
simulates an AgentExecutor loop: first turn requests a tool, second
turn returns plain text and terminates the loop.
"""

from __future__ import annotations

from typing import Any

import openai  # noqa: F401 — eager import, see test_ollama_chat.py for context

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, ToolCall, TokenResponse


# ── Test fixtures ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


def _tool_aware_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> tuple[ProviderRegistry, MockProvider]:
    """Build a :class:`ProviderRegistry` whose :class:`MockProvider`
    replays a two-turn agent loop:

    * turn 1 — model returns one ``ToolCall(tool_name, parameters)``
    * turn 2 — model returns final text with ``tool_calls=[]`` so the
      agent executor terminates.

    ``final_text`` lands on ``NodeResult.output_text`` and on
    ``Result.text`` so the test can assert the flow completed.
    """
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
    """A tool that returns a structured success payload the agent
    executor's ``_execute_tool_call`` will surface as both the LLM-facing
    prompt string AND the ``raw`` dict on ``ToolCallFinished``."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {"ok": True, "echo": "hello"}

    def safe_run(self, **kwargs: Any):
        class _R:
            success = True

        r = _R()
        # Agent executor reads ``result.data`` for structured payload.
        r.data = dict(self._data)
        return r


class _RaisingTool:
    """A tool whose ``safe_run`` raises — exercises the
    ``[ERROR: ...]`` + ``error`` branch of ``_execute_tool_call``."""

    def safe_run(self, **kwargs: Any):
        raise RuntimeError("boom")


class _ToolRegistry:
    """Minimal shim matching the agent executor's expected interface:
    ``.get(name) → tool`` and ``.to_openai_tools() → list[dict]``.

    Registered names here are the *stripped* public names (``"x"``),
    mirroring how integrators register tools in production — the prefix
    normalisation happens inside the agent executor before lookup.
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


def _graph_with_agent() -> Any:
    """Build a ``start → user → agent(tools=["x"]) → end`` graph with a
    captured agent output so non-streaming tests can read it back via
    ``result["agent"]``."""
    return (
        qm.Graph("chat").user().agent("Tooled", tools=["x"], capture_as="agent").build()
    )


# ── 1. Streaming event ordering ──────────────────────────────────────


class TestToolCallStreaming:
    """The engine's new ``ToolCallStarted`` / ``ToolCallFinished`` events
    must reach the SDK stream as typed :class:`ToolCallChunk` /
    :class:`ToolResultChunk` in the expected ordering."""

    def test_tool_call_events_fire_during_stream(self):
        reg, _ = _tool_aware_registry(
            tool_name="x",
            tool_args={"q": "hello"},
            final_text="Done.",
        )
        qm.configure(registry=reg)

        tool_reg = _ToolRegistry({"x": _OkTool({"ok": True, "echo": "hi"})})
        graph = _graph_with_agent()

        chunks = list(qm.run.stream(graph, "hi", tool_registry=tool_reg))
        types = [c.type for c in chunks]

        # We expect, in order:
        #   node_start (agent) → tool_call (x) → tool_result (x) →
        #   node_finish (agent) → done
        # Other node_start/finish pairs (e.g. user input passthrough)
        # may appear but the agent-scoped subsequence must be intact.
        assert "node_start" in types, f"no node_start in {types}"
        assert "tool_call" in types, f"no tool_call in {types}"
        assert "tool_result" in types, f"no tool_result in {types}"
        assert "node_finish" in types, f"no node_finish in {types}"
        assert types[-1] == "done", f"expected done last, got {types}"

        # Ordering: the tool_call chunk must come AFTER at least one
        # node_start (the agent node was started) and BEFORE the done.
        tool_call_idx = types.index("tool_call")
        tool_result_idx = types.index("tool_result")
        first_node_start_idx = types.index("node_start")
        done_idx = types.index("done")
        assert first_node_start_idx < tool_call_idx < tool_result_idx < done_idx, (
            f"out-of-order chunks: {types}"
        )
        # tool_result immediately pairs with tool_call — nothing else
        # may slip between them for a synchronous tool.
        assert tool_result_idx == tool_call_idx + 1, (
            f"tool_result must immediately follow tool_call, got: {types}"
        )

        # Payload shape on the two new chunk types.
        tool_call_chunk = next(c for c in chunks if c.type == "tool_call")
        assert isinstance(tool_call_chunk, qm.ToolCallChunk)
        assert tool_call_chunk.tool == "x"
        assert tool_call_chunk.args == {"q": "hello"}

        tool_result_chunk = next(c for c in chunks if c.type == "tool_result")
        assert isinstance(tool_result_chunk, qm.ToolResultChunk)
        assert tool_result_chunk.tool == "x"
        assert tool_result_chunk.error is None
        # ``result`` carries the structured ``raw`` payload when the
        # tool returned a success envelope — mirrors ``_event_to_chunk``.
        assert tool_result_chunk.result == {"ok": True, "echo": "hi"}


# ── 2. Error surfacing ───────────────────────────────────────────────


class TestToolErrorChunk:
    """When the tool raises, ``ToolResultChunk.error`` carries the error
    message and ``.result`` carries the ``[ERROR: ...]`` sentinel the
    LLM sees on the next turn."""

    def test_tool_call_error_surfaces_via_result_chunk_error(self):
        reg, _ = _tool_aware_registry(tool_name="x", tool_args={"q": "hi"})
        qm.configure(registry=reg)

        tool_reg = _ToolRegistry({"x": _RaisingTool()})
        graph = _graph_with_agent()

        chunks = list(qm.run.stream(graph, "hi", tool_registry=tool_reg))
        tool_result = next((c for c in chunks if c.type == "tool_result"), None)
        assert tool_result is not None, "no tool_result chunk emitted"
        assert isinstance(tool_result, qm.ToolResultChunk)

        # The raising tool path sets ``error`` to the exception message,
        # and ``result`` to the ``[ERROR: ...]`` sentinel fed back to the
        # LLM (since ``raw`` is None on failure, the chunk's ``result``
        # falls back to the prompt-facing sentinel).
        assert tool_result.error is not None, "error must be non-None on failure"
        assert "boom" in tool_result.error
        assert isinstance(tool_result.result, str)
        assert tool_result.result.startswith("[ERROR:"), (
            f"expected [ERROR: ...] sentinel, got {tool_result.result!r}"
        )


# ── 3. Non-streaming run populates data["tool_calls"] ────────────────


class TestNonStreamingToolCallCapture:
    """``qm.run(graph, ...)`` (sync) populates
    ``result["agent"].data["tool_calls"]`` with the same shape the
    streaming events carry — so callers don't need to choose between
    streaming vs introspection."""

    def test_nonstreaming_run_populates_tool_calls_on_capture(self):
        reg, _ = _tool_aware_registry(
            tool_name="x",
            tool_args={"q": "world"},
            final_text="wrapped",
        )
        qm.configure(registry=reg)

        tool_reg = _ToolRegistry({"x": _OkTool({"value": 42})})
        graph = _graph_with_agent()

        result = qm.run(graph, "hi", tool_registry=tool_reg)
        assert result.success, result.error
        assert result.text == "wrapped"

        agent_capture = result["agent"]
        tool_calls = agent_capture.data.get("tool_calls")
        assert isinstance(tool_calls, list), (
            f"expected data['tool_calls'] to be a list, got {type(tool_calls)}"
        )
        assert len(tool_calls) == 1, f"expected exactly 1 tool call, got {tool_calls}"

        entry = tool_calls[0]
        # Keys exactly match the ToolCallFinished event fields so both
        # surfaces (streaming chunks + NodeResult data) stay in sync.
        expected_keys = {"tool", "arguments", "result", "raw", "error", "iteration"}
        assert set(entry) == expected_keys, (
            f"keys mismatch — missing {expected_keys - set(entry)}, "
            f"extra {set(entry) - expected_keys}"
        )
        assert entry["tool"] == "x"
        assert entry["arguments"] == {"q": "world"}
        assert entry["error"] is None
        assert entry["raw"] == {"value": 42}
        assert entry["iteration"] == 1


# ── 4. Prefix normalisation on BOTH surfaces ─────────────────────────


class TestToolPrefixNormalisation:
    """The provider sometimes emits ``default_api:list_orders`` (Gemma
    family) or ``functions:foo`` (OpenAI native) or ``mcp:foo`` (MCP
    bridge).  The engine strips the prefix before dispatching the tool
    AND before emitting ``ToolCallStarted`` / ``ToolCallFinished``, so
    BOTH the streamed ``ToolCallChunk.tool`` and the sync
    ``data['tool_calls'][0]['tool']`` show the bare public name."""

    def test_tool_prefix_normalised_in_events(self):
        reg, _ = _tool_aware_registry(
            tool_name="default_api:list_orders",
            tool_args={"status": "open"},
            final_text="ack",
        )
        qm.configure(registry=reg)

        # Tool is registered under the BARE name "list_orders" — the
        # prefix is a provider-side quirk, not something integrators
        # choose when they register tools.
        tool_reg = _ToolRegistry(
            {"list_orders": _OkTool({"orders": [1, 2, 3]})},
            schema_name="list_orders",
        )
        graph = (
            qm.Graph("chat")
            .user()
            .agent("Tooled", tools=["list_orders"], capture_as="agent")
            .build()
        )

        # --- Streaming surface: ToolCallChunk.tool is stripped ---
        chunks = list(qm.run.stream(graph, "hi", tool_registry=tool_reg))
        tool_call_chunk = next((c for c in chunks if c.type == "tool_call"), None)
        assert tool_call_chunk is not None, "no tool_call chunk emitted"
        assert isinstance(tool_call_chunk, qm.ToolCallChunk)
        assert tool_call_chunk.tool == "list_orders", (
            f"prefix should be stripped in streaming event, got {tool_call_chunk.tool!r}"
        )

        # --- Non-streaming surface: data['tool_calls'][0]['tool'] ---
        # Re-run for the sync path (the streamed run already drained the
        # mock's queued native responses).
        reg2, _ = _tool_aware_registry(
            tool_name="default_api:list_orders",
            tool_args={"status": "open"},
            final_text="ack",
        )
        qm.configure(registry=reg2)
        tool_reg2 = _ToolRegistry(
            {"list_orders": _OkTool({"orders": [1, 2, 3]})},
            schema_name="list_orders",
        )
        result = qm.run(graph, "hi", tool_registry=tool_reg2)
        assert result.success, result.error
        tool_calls = result["agent"].data["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "list_orders", (
            "prefix should be stripped in NodeResult.data['tool_calls'], "
            f"got {tool_calls[0]['tool']!r}"
        )
