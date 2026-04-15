"""Smoke test for the 0.1.2 release-note snippet.

This is the minimum-viable graph the SDK needs to support: ``start → user →
agent → end`` running against a registry built with the one-liner
``register_local()`` helper.  We swap the real Ollama provider for a
:class:`MockProvider` so the test runs without a local LLM, but everything
else (the builder DSL, ``FlowRunner(provider_registry=...)``, default-model
resolution, end-to-end execution) is exercised exactly as a user would.
"""

from __future__ import annotations

import pytest

from quartermaster_engine import FlowRunner
from quartermaster_graph import Graph
from quartermaster_providers import register_local
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


@pytest.fixture
def provider_registry_with_mock():
    """Mirror ``register_local("ollama", default_model="gemma4:26b")`` but
    swap the real OllamaProvider out for a MockProvider that returns a
    canned Slovenian reply.  Lets us run the full 0.1.2 snippet in CI
    without depending on a running Ollama daemon.

    To guarantee no real network connection ever happens we drop the
    ``OllamaProvider`` factory entirely after ``register_local`` sets up
    the routing/default-model state, then ``register_instance`` the mock
    under the same name.  That way even if some future code path causes
    cache eviction the registry can only ever resolve to the mock.
    """
    registry = register_local(
        "ollama",
        base_url="http://host.docker.internal:11434",
        default_model="gemma4:26b",
    )
    answer = "Pozdravljen! Ura je deset."
    mock = MockProvider(
        responses=[TokenResponse(content=answer, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=answer,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            )
        ],
    )
    # Drop the real OllamaProvider factory and replace with the mock
    # instance — get("ollama") can never accidentally hit Docker.
    registry.unregister("ollama")
    registry.register_instance("ollama", mock)
    return registry, mock


class TestSimpleGraphSnippet:
    """Cover the full v0.1.2 release-note snippet end-to-end."""

    def test_user_snippet_runs(self, provider_registry_with_mock):
        registry, mock = provider_registry_with_mock

        graph = (
            Graph("chat")
            .start()
            .user()
            .agent()  # no tools, just text completion
            .end()
            .build()
        )

        runner = FlowRunner(graph=graph, provider_registry=registry)
        result = runner.run("Koliko je ura?")

        assert result.success, f"Flow failed: {result.error}"
        assert result.final_output, "Final output should be non-empty"
        assert "Pozdravljen" in result.final_output
        # And the mock saw the model the registry resolved
        assert mock.last_config is not None
        assert mock.last_config.model == "gemma4:26b"

    def test_flowrunner_requires_a_registry(self):
        """Constructing a runner without either kind of registry must error."""
        graph = Graph("chat").start().user().agent().end().build()
        with pytest.raises(TypeError, match="node_registry or provider_registry"):
            FlowRunner(graph=graph)

    def test_flowrunner_accepts_provider_registry_keyword(self, provider_registry_with_mock):
        """provider_registry must be keyword-only (we don't want a future
        positional swap with node_registry to bite anyone)."""
        registry, _ = provider_registry_with_mock
        graph = Graph("chat").start().user().agent().end().build()
        runner = FlowRunner(graph=graph, provider_registry=registry)
        assert runner.node_registry is not None
        assert runner.provider_registry is registry


class TestAgentExecutor:
    """The agent executor must mirror the canonical Quartermaster loop:
    iterate until the model returns no tool calls, executing each tool the
    model requests in between.  Tool-less graphs collapse to a one-shot
    text completion.
    """

    def test_no_tools_single_iteration(self, provider_registry_with_mock):
        """With no tools wired up, the loop runs exactly once."""
        registry, mock = provider_registry_with_mock
        graph = Graph("chat").start().user().agent().end().build()
        runner = FlowRunner(graph=graph, provider_registry=registry)
        result = runner.run("ping")
        assert result.success
        # Exactly one native-response call.
        assert mock.call_count == 1
        assert mock.calls[0]["method"] == "generate_native_response"
        assert mock.calls[0]["tools"] is None  # no tools requested

    def test_loop_executes_tool_then_terminates(self):
        """When the model returns tool_calls, the agent runs the tool and
        loops; when the next iteration returns no tool_calls, it stops."""
        from quartermaster_engine import AgentExecutor, FlowRunner
        from quartermaster_providers import ProviderRegistry
        from quartermaster_providers.testing import MockProvider
        from quartermaster_providers.types import NativeResponse, ToolCall

        # First call: model wants the weather tool.  Second call: model
        # gives a final text answer with no tool calls -> loop exits.
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="get_weather",
                            tool_id="call_1",
                            parameters={"city": "Ljubljana"},
                        )
                    ],
                    stop_reason="tool_calls",
                ),
                NativeResponse(
                    text_content="It is sunny in Ljubljana.",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                ),
            ],
        )
        registry = ProviderRegistry(auto_configure=False)
        registry.register_instance("mock", mock)
        registry.set_default_provider("mock")
        registry.set_default_model("mock", "test-model")

        # Inline tool registry — anything with .get(name) and to_openai_tools().
        class FakeTool:
            def name(self):
                return "get_weather"

            def safe_run(self, **kwargs):
                class R:
                    success = True
                    data = {"city": kwargs.get("city"), "weather": "sunny"}

                return R()

        class FakeToolRegistry:
            def __init__(self):
                self._tool = FakeTool()

            def get(self, name):
                if name == "get_weather":
                    return self._tool
                raise KeyError(name)

            def to_openai_tools(self):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ]

        # Build a graph with program_version_ids set so the agent treats
        # itself as tool-enabled.
        graph = (
            Graph("agent-with-tools")
            .start()
            .user()
            .agent("Tooled", tools=["get_weather"])
            .end()
            .build()
        )

        runner = FlowRunner(
            graph=graph,
            provider_registry=registry,
            tool_registry=FakeToolRegistry(),
        )
        result = runner.run("What is the weather?")
        assert result.success, result.error
        assert "sunny" in result.final_output
        # Two native-response calls: tool request, then final answer.
        native_calls = [c for c in mock.calls if c["method"] == "generate_native_response"]
        assert len(native_calls) == 2

    def test_max_iterations_surfaces_error(self):
        """If the model keeps requesting tools past max_iterations the
        flow must surface a failure rather than silently succeed."""
        from quartermaster_engine import FlowRunner
        from quartermaster_providers import ProviderRegistry
        from quartermaster_providers.testing import MockProvider
        from quartermaster_providers.types import NativeResponse, ToolCall

        # Always returns a tool call → never terminates on its own.
        always_tool = NativeResponse(
            text_content="",
            thinking=[],
            tool_calls=[
                ToolCall(tool_name="loopy", tool_id="x", parameters={})
            ],
            stop_reason="tool_calls",
        )
        mock = MockProvider(native_responses=[always_tool])

        registry = ProviderRegistry(auto_configure=False)
        registry.register_instance("mock", mock)
        registry.set_default_provider("mock")
        registry.set_default_model("mock", "test-model")

        class FakeTool:
            def safe_run(self, **kwargs):
                class R:
                    success = True
                    data = "done"

                return R()

        class FakeToolRegistry:
            def get(self, name):
                return FakeTool()

            def to_openai_tools(self):
                return [{"type": "function", "function": {"name": "loopy",
                                                           "description": "",
                                                           "parameters": {"type": "object", "properties": {}}}}]

        graph = (
            Graph("loop-cap")
            .start()
            .user()
            .agent("Looping", tools=["loopy"], max_iterations=3)
            .end()
            .build()
        )

        runner = FlowRunner(
            graph=graph,
            provider_registry=registry,
            tool_registry=FakeToolRegistry(),
        )
        result = runner.run("forever?")
        assert result.success is False
        assert result.error is not None
        assert "max_iterations" in result.error.lower()


class TestPropagatesNodeFailure:
    """Regression test for the silent-success bug.

    Before 0.1.2, when an executor returned ``NodeResult(success=False,
    error=...)``, the runner stored the execution as FINISHED (not FAILED)
    and ``FlowResult.success`` came back ``True`` with an empty
    ``final_output`` and ``error=None`` — masking real failures like an
    unreachable Ollama daemon.
    """

    def test_node_failure_surfaces_in_flowresult(self):
        from quartermaster_engine.context.execution_context import ExecutionContext
        from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
        from quartermaster_engine.types import NodeType

        class AlwaysFails:
            async def execute(self, context: ExecutionContext) -> NodeResult:
                return NodeResult(success=False, data={}, error="boom: connection refused")

        reg = SimpleNodeRegistry()
        # Cover every node type the simple chat graph emits.
        reg.register(NodeType.AGENT.value, AlwaysFails())

        # Provide passthrough executors for the other node types so the
        # runner doesn't bail on "no executor registered" before reaching
        # the failing node.
        from quartermaster_engine.example_runner import (
            PassthroughExecutor,
            UserExecutor,
        )

        reg.register(NodeType.USER.value, UserExecutor(interactive=False))
        reg.register(NodeType.STATIC.value, PassthroughExecutor())

        graph = Graph("chat").start().user().agent().end().build()
        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("hello")

        assert result.success is False
        assert result.error is not None
        assert "boom" in result.error
