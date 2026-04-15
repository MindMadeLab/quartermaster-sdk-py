"""Tests for v0.4.0 inline ``@tool`` callables in ``.agent(tools=[...])``.

Before v0.4.0 callers had to:

    @tool()
    def weather(city: str) -> dict: ...

    get_default_registry().register(weather)     # mandatory
    graph = qm.Graph("x").agent(tools=["weather"])

v0.4.0 lets them drop the ``register()`` line and pass the callable
directly:

    graph = qm.Graph("x").agent(tools=[weather])               # all inline
    graph = qm.Graph("x").agent(tools=["web_search", weather]) # mixed

The mechanism: ``GraphBuilder`` stashes callables in a side-channel
``_inline_tools`` dict; the SDK runner reads that dict at run time and
merges it into the run-scoped tool registry before handing off to
:class:`FlowRunner`. Callables never land in the serialisable
``GraphSpec`` — a spec round-tripped through JSON sees only the tool
NAMES, so inline callables do NOT survive a build-then-transport flow.
(See ``test_inline_tools_do_not_survive_graphspec_rebuild`` for the
explicit behaviour.)

Mirrors the ``MockProvider`` / ``_OkTool`` patterns from
``test_v022_tool_streaming.py`` for the LLM stub so the agent executor
actually calls the inline tool end-to-end.
"""

from __future__ import annotations

from typing import Any

import openai  # noqa: F401 — eager import keeps OpenAI sdk metadata sane in the test env

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, ToolCall, TokenResponse
from quartermaster_tools import (
    FunctionTool,
    get_default_registry,
    tool,
)


# ── Helpers ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config_and_registry():
    """Reset SDK config + the module-level default tool registry between
    tests so inline tools registered in one case don't leak to the next."""
    qm.reset_config()
    get_default_registry().clear()
    yield
    qm.reset_config()
    get_default_registry().clear()


def _tool_aware_registry(
    tool_name: str = "x",
    tool_args: dict[str, Any] | None = None,
    final_text: str = "Done.",
) -> ProviderRegistry:
    """Build a ``ProviderRegistry`` whose ``MockProvider`` replays a
    two-turn agent loop: turn 1 calls ``tool_name``, turn 2 returns
    final text and terminates the loop."""
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


# ── 1. Callable passed inline is auto-registered ─────────────────────


class TestCallablePassedInline:
    """Dropping a ``@tool()``-decorated function into ``tools=[...]``
    must transparently wire it through the agent executor."""

    def test_callable_passed_inline_is_auto_registered(self):
        calls: list[dict[str, Any]] = []

        @tool()
        def weather(city: str) -> dict:
            """Return a fake weather payload."""
            calls.append({"city": city})
            return {"city": city, "temp": 22}

        reg = _tool_aware_registry(
            tool_name="weather",
            tool_args={"city": "Berlin"},
            final_text="It's sunny.",
        )
        qm.configure(registry=reg)

        graph = (
            qm.Graph("chat")
            .user()
            .agent("Tooled", tools=[weather], capture_as="agent")
            .build()
        )

        # No ``tool_registry=`` kwarg on purpose — the inline callable
        # must flow through on its own.
        result = qm.run(graph, "weather in Berlin?")
        assert result.success, result.error
        assert result.text == "It's sunny."
        assert calls == [{"city": "Berlin"}], (
            f"inline tool was not invoked as expected: {calls}"
        )


# ── 2. Mixed callable + string tools ─────────────────────────────────


class TestMixedCallableAndStringTools:
    """Callers can mix pre-registered tool name strings with inline
    callables in a single ``tools=[...]`` list."""

    def test_mixed_callable_and_string_tools(self):
        calls: list[dict[str, Any]] = []

        # Pre-register one tool by name in the default registry.
        @tool()
        def search(query: str) -> dict:
            """Pretend-search."""
            calls.append({"search": query})
            return {"results": ["one", "two"]}

        get_default_registry().register(search)

        @tool()
        def weather(city: str) -> dict:
            """Pretend-weather."""
            calls.append({"weather": city})
            return {"city": city, "temp": 22}

        # Agent's FIRST turn will call ``search`` (the string-by-name
        # tool) — the mock only drives one tool call, but both tools
        # must be visible to the agent's schema catalog.
        reg = _tool_aware_registry(
            tool_name="search",
            tool_args={"query": "hi"},
            final_text="ok",
        )
        qm.configure(registry=reg)

        graph = (
            qm.Graph("chat")
            .user()
            .agent("Tooled", tools=["search", weather], capture_as="agent")
            .build()
        )

        # Pass the default tool registry so the string-named "search"
        # resolves; the inline ``weather`` merges on top via the runner.
        result = qm.run(
            graph,
            "hi",
            tool_registry=get_default_registry(),
        )
        assert result.success, result.error
        assert result.text == "ok"
        assert calls == [{"search": "hi"}], (
            f"expected only the search tool to fire, got {calls}"
        )

        # The agent's schema catalog must have seen BOTH tools —
        # ``program_version_ids`` on the agent node stores the normalised
        # names so we can assert the builder recorded them.
        agent_node = next(n for n in graph.nodes if n.name == "Tooled")
        assert agent_node.metadata["program_version_ids"] == ["search", "weather"]


# ── 3. Undecorated callables are auto-decorated ──────────────────────


class TestUndecoratedCallable:
    """Bare ``def``s without the ``@tool()`` decorator are still
    acceptable — the builder auto-decorates them on the fly."""

    def test_undecorated_callable_can_still_be_used(self):
        calls: list[dict[str, Any]] = []

        def lookup(name: str) -> dict:
            """Plain function — no @tool() decorator."""
            calls.append({"name": name})
            return {"name": name, "id": 42}

        reg = _tool_aware_registry(
            tool_name="lookup",
            tool_args={"name": "alice"},
            final_text="Got it.",
        )
        qm.configure(registry=reg)

        graph = qm.Graph("chat").user().agent("Tooled", tools=[lookup]).build()

        result = qm.run(graph, "who is alice?")
        assert result.success, result.error
        assert result.text == "Got it."
        assert calls == [{"name": "alice"}]


# ── 4. Lambda / unparseable callables raise clear errors ─────────────


class TestLambdaRaisesClearError:
    """Anonymous / unintrospectable callables must produce an
    actionable error message, not a cryptic AttributeError downstream."""

    def test_lambda_raises_clear_error(self):
        with pytest.raises(ValueError) as exc_info:
            qm.Graph("x").user().agent(tools=[lambda q: q])

        msg = str(exc_info.value)
        assert "not @tool()-decorated" in msg, msg
        assert "register" in msg, (
            f"error message must mention the manual register() fallback: {msg}"
        )

    def test_non_callable_non_string_raises_typeerror(self):
        """Passing something that's neither a string, a callable, nor a
        tool instance (e.g. an int) must raise ``TypeError``, not a
        misleading ``ValueError``."""
        with pytest.raises(TypeError) as exc_info:
            qm.Graph("x").user().agent(tools=[42])  # type: ignore[list-item]
        assert "unsupported item type" in str(exc_info.value)


# ── 5. Inline tools don't pollute the global registry ────────────────


class TestInlineToolDoesNotPolluteGlobalRegistry:
    """Running with ``tools=[foo]`` is a per-run concern — the inline
    callable must NOT be registered in the module-level default
    registry so unrelated tests / graphs don't see it."""

    def test_inline_callable_stays_out_of_default_registry(self):
        @tool()
        def secret(code: str) -> dict:
            """Local-only tool — must not leak into the default registry."""
            return {"code": code}

        reg = _tool_aware_registry(
            tool_name="secret",
            tool_args={"code": "X"},
            final_text="done",
        )
        qm.configure(registry=reg)

        graph = qm.Graph("chat").user().agent("Tooled", tools=[secret]).build()

        default_names_before = set(get_default_registry().list_names())
        result = qm.run(graph, "?")
        assert result.success, result.error

        default_names_after = set(get_default_registry().list_names())
        assert "secret" not in default_names_after, (
            "inline tools must not register themselves in the default registry"
        )
        assert default_names_before == default_names_after, (
            f"default registry was mutated during run: "
            f"added={default_names_after - default_names_before}, "
            f"removed={default_names_before - default_names_after}"
        )


# ── 6. Inline tools work in branch builders ──────────────────────────


class TestInlineToolInBranchBuilder:
    """``tools=[...]`` with callables must work identically inside
    ``if_node().on(...).agent(...)`` branches — the branch builder
    forwards to the root graph's ``_inline_tools`` dict."""

    def test_inline_tool_works_in_branch_builder(self):
        calls: list[dict[str, Any]] = []

        @tool()
        def dig(path: str) -> dict:
            """Fake file-digger."""
            calls.append({"path": path})
            return {"found": True}

        reg = _tool_aware_registry(
            tool_name="dig",
            tool_args={"path": "/tmp"},
            final_text="dug",
        )
        qm.configure(registry=reg)

        graph = (
            qm.Graph("x")
            .user()
            .if_node("root", expression="True")
            .on("true")
            .agent("Tooled", tools=[dig])
            .end()
            .build()
        )

        # The inline tool dict is on the builder; it gets picked up
        # because qm.run accepts either a builder or a spec and the
        # runner extracts the dict BEFORE calling .build().
        # Here we already called .build(), so to exercise the branch
        # path we need to re-use the builder directly.
        builder = (
            qm.Graph("x")
            .user()
            .if_node("root", expression="True")
            .on("true")
            .agent("Tooled", tools=[dig])
            .end()
        )

        result = qm.run(builder, "go")
        assert result.success, result.error
        assert calls == [{"path": "/tmp"}], f"branch-tool didn't fire: {calls}"


# ── 7. FunctionTool instance (already wrapped) works inline ──────────


class TestFunctionToolInstance:
    """Not just ``@tool()``-returned functions — an explicit
    :class:`FunctionTool` instance is a legal item too (covers the
    case where integrators build a tool imperatively and want to drop
    it into one specific agent)."""

    def test_function_tool_instance_passes_through(self):
        @tool(name="cached")
        def _impl(x: int) -> dict:
            return {"x": x}

        assert isinstance(_impl, FunctionTool)
        reg = _tool_aware_registry(
            tool_name="cached",
            tool_args={"x": 7},
            final_text="ok",
        )
        qm.configure(registry=reg)

        graph = qm.Graph("chat").user().agent("Tooled", tools=[_impl]).build()
        result = qm.run(graph, "?")
        assert result.success, result.error
        assert result.text == "ok"


# ── 8. GraphSpec round-trip loses inline tools (documented) ──────────


class TestInlineToolGraphSpecRoundTrip:
    """Inline callables are held on the builder's ``_inline_tools``
    side-channel, NOT on the built ``GraphSpec``.  Rebuilding the spec
    from its JSON round-trip therefore drops them — the agent still
    references the tool NAME (which is on the spec), but the callable
    needs to be supplied some other way (e.g. a
    ``tool_registry=`` kwarg).  This test pins the documented
    limitation so we don't silently change it."""

    def test_graphspec_rebuilt_from_dump_loses_callables(self):
        @tool()
        def weather(city: str) -> dict:
            return {"city": city, "temp": 22}

        builder = qm.Graph("chat").user().agent("Tooled", tools=[weather])
        spec = builder.build()
        # The name survives on the serialisable spec…
        agent_node = next(n for n in spec.nodes if n.name == "Tooled")
        assert agent_node.metadata["program_version_ids"] == ["weather"]
        # …but a freshly-rehydrated GraphSpec doesn't carry the inline
        # callable dict (``_inline_tools`` is a builder-side attribute).
        from quartermaster_graph import GraphSpec

        rehydrated = GraphSpec.model_validate(spec.model_dump(mode="json"))
        assert not hasattr(rehydrated, "_inline_tools") or not getattr(
            rehydrated, "_inline_tools", None
        )
