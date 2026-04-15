"""Tests for the v0.2.0 ergonomic surface.

Covers the four primary new-in-0.2.0 public entry points:

* :func:`quartermaster_sdk.configure` — module-level default registry
* :func:`quartermaster_sdk.run` — one-liner graph runner returning a
  :class:`Result`
* :func:`quartermaster_sdk.run.stream` — typed :class:`Chunk` stream
* :func:`quartermaster_sdk.instruction` / :func:`instruction_form` —
  single-shot prompt helpers with no ``Graph`` boilerplate

Plus the underlying primitives: auto-start, optional End, ``capture_as=``
threading through :attr:`Result.captures`.

We mock the provider with :class:`MockProvider` so the suite runs with
no real LLM.  ``instruction_form`` uses a Pydantic model; the mock is
primed to return canned JSON.
"""

from __future__ import annotations

import openai  # noqa: F401 — eager import, see test_ollama_chat.py for context

from typing import Any

import pytest
from pydantic import BaseModel

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


# ── Helpers ───────────────────────────────────────────────────────────


def _mock_registry(
    text: str = "canned",
    native_text: str | None = None,
) -> tuple[ProviderRegistry, MockProvider]:
    """Build a ProviderRegistry with a MockProvider registered as ``ollama``.

    Both text (streamed) and native-response channels are primed so
    tests that pick either path see a consistent reply.
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


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


# ── configure() ───────────────────────────────────────────────────────


class TestConfigure:
    def test_configure_with_registry(self):
        reg, _ = _mock_registry()
        returned = qm.configure(registry=reg)
        assert returned is reg
        assert qm.get_default_registry() is reg

    def test_configure_rejects_mixed_args(self):
        reg, _ = _mock_registry()
        with pytest.raises(ValueError, match="registry= OR base_url"):
            qm.configure(registry=reg, base_url="http://localhost:11434")

    def test_get_default_registry_raises_before_configure(self):
        with pytest.raises(RuntimeError, match="No default provider registry"):
            qm.get_default_registry()

    def test_env_default_model_picked_up(self, monkeypatch):
        monkeypatch.setenv("QM_DEFAULT_MODEL", "gemma4:26b")
        reg, _ = _mock_registry()
        qm.configure(registry=reg)
        # registry= path uses the env var for get_default_model()
        assert qm.get_default_model() == "gemma4:26b"


# ── Graph builder v0.2.0 ──────────────────────────────────────────────


class TestAutoStart:
    def test_graph_constructor_creates_start_node(self):
        graph = qm.Graph("x").instruction("Plain").build()
        starts = [n for n in graph.nodes if n.type.value.startswith("Start")]
        assert len(starts) == 1, "exactly one Start node should exist"

    def test_explicit_start_call_is_idempotent(self):
        """Calling .start() after auto-start must NOT create a second Start."""
        graph = qm.Graph("x").start().instruction("Plain").build()
        starts = [n for n in graph.nodes if n.type.value.startswith("Start")]
        assert len(starts) == 1

    def test_auto_start_opt_out(self):
        """``auto_start=False`` restores the pre-0.2.0 manual path."""
        with pytest.raises(ValueError, match="start node"):
            qm.Graph("x", auto_start=False).instruction("Plain").build()


class TestOptionalEnd:
    def test_graph_without_end_validates_clean(self):
        # Just builds — no .end() call.  Previously this raised "no_end".
        graph = qm.Graph("x").instruction("Plain").build()
        assert graph is not None


# ── run() + capture_as + Result ──────────────────────────────────────


class TestRunAndCaptures:
    def test_run_returns_result_with_text(self):
        reg, _ = _mock_registry("Pozdravljen!")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("One", capture_as="one").build()
        result = qm.run(graph, "hi")
        assert result.success
        assert result.text == "Pozdravljen!"
        assert isinstance(result, qm.Result)

    def test_capture_as_populates_captures_dict(self):
        reg, _ = _mock_registry("hello")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("Named", capture_as="reply").build()
        result = qm.run(graph, "?")
        assert "reply" in result.captures
        assert result.captures["reply"].output_text == "hello"

    def test_result_getitem_shorthand(self):
        reg, _ = _mock_registry("hello")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("Named", capture_as="reply").build()
        result = qm.run(graph, "?")
        # result["reply"] is the shorthand for result.captures["reply"]
        assert result["reply"].output_text == "hello"

    def test_missing_capture_key_raises_with_known_keys(self):
        reg, _ = _mock_registry("hello")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("Named", capture_as="reply").build()
        result = qm.run(graph, "?")
        with pytest.raises(KeyError, match="Available captures: reply"):
            _ = result["nope"]

    def test_run_stream_yields_typed_chunks_ending_with_done(self):
        reg, _ = _mock_registry("streaming words")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("One").build()

        chunks = list(qm.run.stream(graph, "hi"))
        types = [c.type for c in chunks]
        # Must end with a DoneChunk that carries a populated Result.
        assert types[-1] == "done"
        last = chunks[-1]
        assert isinstance(last, qm.DoneChunk)
        assert last.result.success
        # Intermediate events should include at least one node_start / node_finish.
        assert "node_start" in types
        assert "node_finish" in types

    def test_run_uses_configured_registry_when_none_passed(self):
        reg, mock = _mock_registry("configured")
        qm.configure(registry=reg)
        graph = qm.Graph("x").instruction("One").build()
        result = qm.run(graph, "?")
        assert result.success
        assert mock.call_count >= 1

    def test_run_override_registry_wins(self):
        reg_a, _ = _mock_registry("A")
        reg_b, _ = _mock_registry("B")
        qm.configure(registry=reg_a)
        graph = qm.Graph("x").instruction("One").build()
        result = qm.run(graph, "?", provider_registry=reg_b)
        assert result.text == "B"


# ── instruction() / instruction_form() ───────────────────────────────


class TestInstruction:
    def test_instruction_returns_str(self):
        reg, _ = _mock_registry("gotcha")
        qm.configure(registry=reg)
        reply = qm.instruction(system="sys", user="usr")
        assert reply == "gotcha"

    def test_instruction_requires_model(self):
        reg, _ = _mock_registry()
        # Configure without default_model — instruction() must complain
        reg_no_default = ProviderRegistry(auto_configure=False)
        reg_no_default.register_instance("ollama", MockProvider())
        reg_no_default.set_default_provider("ollama")
        qm.configure(registry=reg_no_default)
        with pytest.raises(ValueError, match="no model resolved"):
            qm.instruction(user="x")


class TestInstructionForm:
    class _Classification(BaseModel):
        category: str
        priority: str

    def test_instruction_form_returns_pydantic_model(self):
        canned = '{"category":"order","priority":"urgent"}'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(
            self._Classification, system="Classify.", user="urgent order 42"
        )
        assert isinstance(out, self._Classification)
        assert out.category == "order"
        assert out.priority == "urgent"

    def test_instruction_form_strips_markdown_fence(self):
        """Models sometimes insist on fencing JSON in ```json. Must be tolerated."""
        canned = '```json\n{"category":"ok","priority":"normal"}\n```'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        out = qm.instruction_form(self._Classification, system="c", user="u")
        assert out.category == "ok"

    def test_instruction_form_raises_on_schema_mismatch(self):
        canned = '{"not_the_right":"shape"}'
        reg, _ = _mock_registry(canned)
        qm.configure(registry=reg)
        with pytest.raises(RuntimeError, match="did not match schema"):
            qm.instruction_form(self._Classification, system="c", user="u")

    def test_instruction_form_rejects_non_pydantic_schema(self):
        reg, _ = _mock_registry("{}")
        qm.configure(registry=reg)
        with pytest.raises(ValueError, match="pydantic.BaseModel subclass"):
            qm.instruction_form(dict, system="c", user="u")  # type: ignore[arg-type]


# ── Tool-name prefix stripping ───────────────────────────────────────


class TestToolNamePrefixStripping:
    """Regression: Gemma-family models emit ``default_api:foo``; OpenAI
    native wire uses ``functions:foo``; MCP bridge uses ``mcp:foo``.  All
    three must resolve to the same registered tool."""

    def test_strip_helper(self):
        from quartermaster_engine.example_runner import _normalise_tool_name

        assert _normalise_tool_name("default_api:list_orders") == "list_orders"
        assert _normalise_tool_name("default_api.list_orders") == "list_orders"
        assert _normalise_tool_name("functions:list_orders") == "list_orders"
        assert _normalise_tool_name("functions.list_orders") == "list_orders"
        assert _normalise_tool_name("mcp:list_orders") == "list_orders"
        # Bare names pass through unchanged.
        assert _normalise_tool_name("list_orders") == "list_orders"
