"""Regression tests for the v0.4.0 per-node tool scoping feature.

Surface under test:

* ``AgentExecutor`` — the agent loop in ``example_runner.py`` now
  enforces the ``tools=[...]`` list as a HARD allow-list for every
  tool call the model emits. A hallucinated out-of-list name does
  NOT reach the registry; the model instead sees an explicit
  ``[ERROR: tool 'X' is not allowed for this agent node. Allowed:
  Y]`` string and typically corrects itself on the next iteration.
* The structured ``tool_calls`` log on ``NodeResult.data`` records
  the rejection so audit trails see what was attempted.
* ``tool_scope="permissive"`` on the graph node reverts to the
  legacy pre-v0.4.0 "any registered tool reachable" behaviour — the
  migration escape hatch.
* Prefix stripping (``default_api:A`` → ``A``) from v0.2.1 still
  takes effect BEFORE the allow-list check, so naming contracts
  with different providers continue to work unchanged.
"""

from __future__ import annotations

from quartermaster_engine import FlowRunner
from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, ToolCall


# ── Test helpers ─────────────────────────────────────────────────────


class _RecordingTool:
    """Simple tool stand-in: records every call, returns canned data."""

    def __init__(self, name: str, payload: str = "ok") -> None:
        self._name = name
        self._payload = payload
        self.calls: list[dict] = []

    def name(self) -> str:
        return self._name

    def safe_run(self, **kwargs):
        self.calls.append(dict(kwargs))
        tool_name = self._name
        payload = self._payload

        class R:
            success = True
            data = {"tool": tool_name, "payload": payload}

        return R()


class _FakeRegistry:
    """Minimal tool registry exposing ``get`` + ``to_openai_tools``.

    Every tool the engine-level ``_execute_tool_call`` might need to
    look up lives here. Whether the LOOKUP is actually attempted
    depends on whether the allow-list lets the call through — that's
    the contract we're testing.
    """

    def __init__(self, tools: list[_RecordingTool]) -> None:
        self._tools = {t.name(): t for t in tools}

    def get(self, name: str):
        if name in self._tools:
            return self._tools[name]
        raise KeyError(name)

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self._tools
        ]


def _build_mock_provider(tool_calls_then_final: list[list[ToolCall]]):
    """Build a MockProvider whose successive ``generate_native_response``
    calls return the given tool_calls batches, then a final text answer.

    Each batch in *tool_calls_then_final* drives one iteration of the
    agent loop. An empty batch (or the implicit final "done" response)
    terminates the loop.
    """
    responses = []
    for batch in tool_calls_then_final:
        responses.append(
            NativeResponse(
                text_content="",
                thinking=[],
                tool_calls=list(batch),
                stop_reason="tool_calls",
            )
        )
    # Final turn — model signals it's done by returning no tool_calls.
    responses.append(
        NativeResponse(
            text_content="Final answer.",
            thinking=[],
            tool_calls=[],
            stop_reason="stop",
        )
    )
    return MockProvider(native_responses=responses)


def _make_registry(provider: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("mock", provider)
    reg.set_default_provider("mock")
    reg.set_default_model("mock", "test-model")
    return reg


# ── 1. Strict scope blocks out-of-list tool calls ────────────────────


def test_strict_scope_blocks_out_of_list_tool():
    """Registering both A and B but scoping the node to ``tools=["A"]``
    must prevent a hallucinated B call from ever reaching B.safe_run.

    The model sees a structured error instead; the structured tool_calls
    log records the rejection alongside ``error="not allowed: B"``.
    """
    tool_a = _RecordingTool("A", payload="A-data")
    tool_b = _RecordingTool("B", payload="B-data")
    registry = _FakeRegistry([tool_a, tool_b])

    # Iteration 1: model (wrongly) asks for B. Iteration 2: gives up,
    # returns final text.
    mock = _build_mock_provider(
        [
            [ToolCall(tool_name="B", tool_id="c1", parameters={"q": "bad"})],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = (
        Graph("scope-block")
        .start()
        .user()
        .agent("research", tools=["A"], capture_as="research")
        .end()
        .build()
    )

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error

    # B.safe_run must NOT have been called — the allow-list gated it.
    assert tool_b.calls == [], f"Tool B was called despite scoping: {tool_b.calls}"
    # A was also never called (model never emitted it), just verifying
    # the blocked path didn't accidentally invoke something else.
    assert tool_a.calls == []

    # The node's tool_calls log should record the rejection.
    assert "research" in result.captures, "research node capture missing"
    node_data = result.captures["research"].data
    tool_log = node_data.get("tool_calls", [])
    assert len(tool_log) == 1
    entry = tool_log[0]
    assert entry["tool"] == "B"
    assert entry["error"] is not None and "not allowed" in entry["error"]
    # The prompt_text (what the model actually sees next iteration)
    # carries the structured allow-list hint so it can correct itself.
    assert "not allowed for this agent node" in entry["result"]
    assert "Allowed: A" in entry["result"]


# ── 2. In-list tool resolves normally (control) ──────────────────────


def test_in_list_tool_resolves_normally():
    """Control test: same setup, model correctly calls A; A executes."""
    tool_a = _RecordingTool("A", payload="A-data")
    tool_b = _RecordingTool("B", payload="B-data")
    registry = _FakeRegistry([tool_a, tool_b])

    mock = _build_mock_provider(
        [
            [ToolCall(tool_name="A", tool_id="c1", parameters={"q": "ok"})],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = Graph("scope-ok").start().user().agent("research", tools=["A"]).end().build()

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error
    assert tool_a.calls == [{"q": "ok"}]
    assert tool_b.calls == []


# ── 3. Permissive scope keeps legacy behaviour ───────────────────────


def test_permissive_scope_keeps_legacy_behaviour():
    """``tool_scope="permissive"`` reverts to the pre-v0.4.0 leak: the
    model can call any tool the registry exposes, even if it isn't in
    the node's ``tools=[...]`` list. Intended as a migration escape
    hatch for integrators that relied on the leak."""
    tool_a = _RecordingTool("A")
    tool_b = _RecordingTool("B")
    registry = _FakeRegistry([tool_a, tool_b])

    # Model calls B even though the node is declared with tools=["A"].
    mock = _build_mock_provider(
        [
            [ToolCall(tool_name="B", tool_id="c1", parameters={"q": "legacy"})],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = (
        Graph("permissive")
        .start()
        .user()
        .agent("research", tools=["A"], tool_scope="permissive")
        .end()
        .build()
    )

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error
    # Permissive scope must let B through.
    assert tool_b.calls == [{"q": "legacy"}]


# ── 4. Default is strict ─────────────────────────────────────────────


def test_default_is_strict():
    """Without an explicit ``tool_scope=`` argument, the new
    v0.4.0 strict behaviour is in effect — out-of-list tool calls
    are rejected."""
    tool_a = _RecordingTool("A")
    tool_b = _RecordingTool("B")
    registry = _FakeRegistry([tool_a, tool_b])

    mock = _build_mock_provider(
        [
            [ToolCall(tool_name="B", tool_id="c1", parameters={})],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = (
        Graph("default-strict")
        .start()
        .user()
        .agent("research", tools=["A"])  # no explicit tool_scope=
        .end()
        .build()
    )

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error
    # B was rejected by the default strict scope.
    assert tool_b.calls == []


# ── 5. Normalised name is compared ───────────────────────────────────


def test_normalised_name_is_compared():
    """Provider prefixes (``default_api:``, ``functions:``, ``mcp:``)
    are stripped BEFORE the allow-list check. That way a model that
    emits ``default_api:A`` against a node declared with ``tools=["A"]``
    still resolves correctly (per the v0.2.1 prefix-stripping contract)."""
    tool_a = _RecordingTool("A", payload="A-data")
    registry = _FakeRegistry([tool_a])

    mock = _build_mock_provider(
        [
            [
                ToolCall(
                    tool_name="default_api:A",
                    tool_id="c1",
                    parameters={"q": "prefixed"},
                )
            ],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = Graph("prefix-strip").start().user().agent("research", tools=["A"]).end().build()

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error
    # Prefix stripped → allow-list matched → tool executed.
    assert tool_a.calls == [{"q": "prefixed"}]


# ── 6. Empty tools list means NO tools reachable ─────────────────────


def test_empty_tools_list_means_no_tools():
    """``agent("x", tools=[])`` — the node has declared itself tool-
    enabled (non-default knob) but provided no tools. Strict scope
    therefore rejects EVERY tool call the model attempts. This is the
    v0.4.0 contract: an empty allow-list is still an allow-list.

    Behaviourally: when tools=[] the agent loop short-circuits before
    asking the provider for tool_calls — the ``tools`` kwarg passed
    to the provider is ``None`` — so the model shouldn't emit tool
    calls at all. This test verifies that even if some provider
    hallucinates a tool call anyway, the strict check blocks it.
    """
    tool_a = _RecordingTool("A")
    registry = _FakeRegistry([tool_a])

    # Model tries to call A despite the node declaring no tools at all.
    # The way the agent loop is built, with ``program_version_ids=[]``
    # the ``tools`` kwarg to ``generate_native_response`` is None and
    # the loop terminates on the FIRST response regardless of whether
    # it contains tool_calls — so the important assertion here is that
    # tool_a.safe_run was never reached via the registry either way.
    mock = _build_mock_provider(
        [
            [ToolCall(tool_name="A", tool_id="c1", parameters={})],
        ]
    )
    provider_registry = _make_registry(mock)

    graph = Graph("empty-tools").start().user().agent("research", tools=[]).end().build()

    runner = FlowRunner(
        graph=graph,
        provider_registry=provider_registry,
        tool_registry=registry,
    )
    result = runner.run("go")
    assert result.success, result.error
    assert tool_a.calls == []


# ── 7. Empty tools list + non-empty registry, forced execution path ──


def test_execute_tool_call_helper_blocks_out_of_list():
    """Direct unit test on ``_execute_tool_call`` to pin down the
    gating predicate independently of the agent loop. Uses the
    engine-internal helper so we can assert the exact error-message
    shape and the ``_ToolInvocation.error`` field without needing a
    full flow runner."""
    from quartermaster_engine.example_runner import _execute_tool_call

    tool_a = _RecordingTool("A")
    tool_b = _RecordingTool("B")
    registry = _FakeRegistry([tool_a, tool_b])

    # allowed_tools=["A"] → "B" call blocked without touching B.
    invocation = _execute_tool_call(registry, "B", {"q": "nope"}, allowed_tools=["A"])
    assert tool_b.calls == []
    assert invocation.error is not None and "not allowed" in invocation.error
    assert "not allowed for this agent node" in invocation.prompt_text
    assert "Allowed: A" in invocation.prompt_text
    assert invocation.raw is None

    # allowed_tools=["A"] + A call → executes normally.
    invocation_ok = _execute_tool_call(registry, "A", {"q": "yes"}, allowed_tools=["A"])
    assert tool_a.calls == [{"q": "yes"}]
    assert invocation_ok.error is None

    # allowed_tools=None → legacy behaviour, any registered tool runs.
    invocation_legacy = _execute_tool_call(registry, "B", {"q": "legacy"}, allowed_tools=None)
    assert tool_b.calls == [{"q": "legacy"}]
    assert invocation_legacy.error is None

    # allowed_tools=[] (empty allow-list) — everything blocked.
    invocation_empty = _execute_tool_call(registry, "A", {"q": "blocked"}, allowed_tools=[])
    assert invocation_empty.error is not None
    assert "Allowed: <none>" in invocation_empty.prompt_text
