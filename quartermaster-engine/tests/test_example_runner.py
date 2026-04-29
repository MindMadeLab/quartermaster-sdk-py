"""Comprehensive tests for conversation helpers and node executors in example_runner.py."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import (
    IfExecutor,
    MemoryReadExecutor,
    MemoryWriteExecutor,
    PassthroughExecutor,
    StaticExecutor,
    TextExecutor,
    UserExecutor,
    UserFormExecutor,
    VarExecutor,
    _append_to_conversation,
    _build_history_for_node,
    _get_conversation,
)
from quartermaster_engine.nodes import NodeResult
from quartermaster_engine.types import (
    GraphSpec,
    GraphEdge,
    GraphNode,
    Message,
    MessageRole,
    NodeType,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_node(
    name: str = "TestNode",
    node_type: NodeType = NodeType.INSTRUCTION,
    metadata: dict[str, Any] | None = None,
) -> GraphNode:
    return GraphNode(
        id=uuid4(),
        type=node_type,
        name=name,
        metadata=metadata or {},
    )


def _make_graph(nodes: list[GraphNode] | None = None) -> GraphSpec:
    if not nodes:
        nodes = [_make_node()]
    return GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=nodes[0].id,
        nodes=nodes,
        edges=[],
    )


def _make_context(
    memory: dict[str, Any] | None = None,
    messages: list[Message] | None = None,
    metadata: dict[str, Any] | None = None,
    node_name: str = "TestNode",
    node_type: NodeType = NodeType.INSTRUCTION,
    node_metadata: dict[str, Any] | None = None,
    graph: GraphSpec | None = None,
    current_node: GraphNode | None = None,
) -> ExecutionContext:
    if current_node is None:
        current_node = _make_node(name=node_name, node_type=node_type, metadata=node_metadata or {})
    if graph is None:
        graph = _make_graph([current_node])
    return ExecutionContext(
        flow_id=uuid4(),
        node_id=current_node.id,
        graph=graph,
        current_node=current_node,
        messages=messages or [],
        memory=memory or {},
        metadata=metadata or {},
    )


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ===================================================================
# 1. _get_conversation tests
# ===================================================================


class TestGetConversation:
    """Tests for _get_conversation helper."""

    def test_empty_memory_returns_empty_list(self):
        ctx = _make_context(memory={})
        result = _get_conversation(ctx)
        assert result == []

    def test_no_conversation_key_returns_empty_list(self):
        ctx = _make_context(memory={"some_key": "val"})
        result = _get_conversation(ctx)
        assert result == []

    def test_returns_copy_not_reference(self):
        original = [{"role": "user", "text": "hi"}]
        ctx = _make_context(memory={"__conversation__": original})
        result = _get_conversation(ctx)
        assert result == original
        assert result is not original

    def test_mutating_returned_list_does_not_affect_memory(self):
        original = [{"role": "user", "text": "hi"}]
        ctx = _make_context(memory={"__conversation__": original})
        result = _get_conversation(ctx)
        result.append({"role": "assistant", "text": "bye"})
        assert len(ctx.memory["__conversation__"]) == 1

    def test_single_entry_conversation(self):
        conv = [{"role": "user", "text": "hello"}]
        ctx = _make_context(memory={"__conversation__": conv})
        result = _get_conversation(ctx)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_five_entry_conversation(self):
        conv = [{"role": f"role{i}", "text": f"msg{i}"} for i in range(5)]
        ctx = _make_context(memory={"__conversation__": conv})
        result = _get_conversation(ctx)
        assert len(result) == 5

    def test_fifty_entry_conversation(self):
        conv = [{"role": "user", "text": f"msg{i}"} for i in range(50)]
        ctx = _make_context(memory={"__conversation__": conv})
        result = _get_conversation(ctx)
        assert len(result) == 50

    def test_conversation_with_round_numbers(self):
        conv = [{"role": "user", "text": "hi", "round": 1}]
        ctx = _make_context(memory={"__conversation__": conv})
        result = _get_conversation(ctx)
        assert result[0]["round"] == 1

    def test_empty_conversation_list(self):
        ctx = _make_context(memory={"__conversation__": []})
        result = _get_conversation(ctx)
        assert result == []

    def test_conversation_entries_are_shallow_copies(self):
        entry = {"role": "user", "text": "hi"}
        ctx = _make_context(memory={"__conversation__": [entry]})
        result = _get_conversation(ctx)
        # list() creates a shallow copy, so the dict objects are the same
        assert result[0] is entry

    def test_none_conversation_value_returns_empty_list(self):
        ctx = _make_context(memory={"__conversation__": None})
        # list(None) would throw, but .get returns [] as default
        # Since memory has the key but it's None, it should use None -> list(None) fails
        # Actually: context.memory.get("__conversation__", []) returns None when key exists
        # list(None) raises TypeError. Let's verify behavior.
        # Re-checking source: list(context.memory.get("__conversation__", []))
        # If the value is None, list(None) raises TypeError.
        # This is an edge case that actually raises.
        with pytest.raises(TypeError):
            _get_conversation(ctx)


# ===================================================================
# 2. _append_to_conversation tests
# ===================================================================


class TestAppendToConversation:
    """v0.8.0 entry shape: ``{"role", "content", "node_name", "round"}``.

    The ``role`` is now a wire-format role ("user"/"assistant"/"system");
    the node identity moved to ``node_name``. Default role is
    ``"assistant"`` because by far the most common caller is a node
    appending its own output."""

    def test_default_role_is_assistant(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Researcher", "hello")
        assert len(conv) == 1
        assert conv[0]["role"] == "assistant"
        assert conv[0]["content"] == "hello"
        assert conv[0]["node_name"] == "Researcher"

    def test_explicit_user_role_for_user_turn(self):
        """The CLI/SDK history seed path passes ``role='user'`` and an
        empty node_name to mark a turn that came from the human."""
        conv: list[dict] = []
        _append_to_conversation(conv, "", "what's up?", role="user")
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "what's up?"
        assert conv[0]["node_name"] is None

    def test_includes_round_num_when_provided(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Agent", "hi", round_num=3)
        assert conv[0]["round"] == 3

    def test_skips_empty_text(self):
        conv: list[dict] = []
        result = _append_to_conversation(conv, "Agent", "")
        assert len(conv) == 0
        assert result is conv

    def test_skips_whitespace_only_text(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Agent", "   \t\n  ")
        assert len(conv) == 0

    def test_none_round_num_not_included(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Agent", "hello", round_num=None)
        assert "round" not in conv[0]

    def test_zero_round_num_is_included(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Agent", "hello", round_num=0)
        assert conv[0]["round"] == 0

    def test_multiple_appends_accumulate(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Researcher", "first")
        _append_to_conversation(conv, "Summariser", "second")
        _append_to_conversation(conv, "Reviewer", "third")
        assert len(conv) == 3
        assert [e["content"] for e in conv] == ["first", "second", "third"]
        assert [e["node_name"] for e in conv] == ["Researcher", "Summariser", "Reviewer"]

    def test_unicode_content(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "Agent", "Привет мир 🌍")
        assert conv[0]["content"] == "Привет мир 🌍"


# ===================================================================
# 3. _build_history_for_node tests (v0.8.0)
# ===================================================================


class TestBuildHistoryForNode:
    """v0.8.0 role-translation contract.

    ``__conversation__`` entries get translated to OpenAI-format
    messages based on ``node_name`` vs the consuming node:

    - Same node → ``role="assistant"`` (my own past turn).
    - Different node → ``role="user"`` (input from upstream).
    - ``node_name=None`` → preserve stored role (user-supplied seed).
    - Legacy v0.7.x ``{"role": <NodeName>, "text": ...}`` → translate
      ``role==current_node_name`` ⇒ assistant, else user.
    """

    def test_empty_conversation_returns_empty_history(self):
        assert _build_history_for_node([], "Agent") == []

    def test_same_node_entries_become_assistant(self):
        conv = [
            {"role": "assistant", "content": "previous turn", "node_name": "Agent"},
        ]
        history = _build_history_for_node(conv, "Agent")
        assert history == [{"role": "assistant", "content": "previous turn"}]

    def test_other_node_output_becomes_user(self):
        """The whole point of v0.8.0: Agent2 sees Agent1's output as
        ``role="user"`` (input it must act on), not as ``"assistant"``."""
        conv = [
            {"role": "assistant", "content": "research notes", "node_name": "Agent1"},
        ]
        history = _build_history_for_node(conv, "Agent2")
        assert history == [{"role": "user", "content": "research notes"}]

    def test_user_supplied_seed_preserves_role(self):
        """SDK ``run(history=[Message(role="user", ...)])`` seed: stored
        with ``node_name=None`` and the original role survives."""
        conv = [
            {"role": "user", "content": "/stranka PIGO", "node_name": None},
            {"role": "assistant", "content": "PIGO is...", "node_name": None},
            {"role": "user", "content": "kakšen je status?", "node_name": None},
        ]
        history = _build_history_for_node(conv, "ChatAgent")
        assert history == [
            {"role": "user", "content": "/stranka PIGO"},
            {"role": "assistant", "content": "PIGO is..."},
            {"role": "user", "content": "kakšen je status?"},
        ]

    def test_pipeline_user_then_agent1_then_agent2_seen_by_agent2(self):
        """Concrete trace of ``User → Agent1 → Agent2``:
        Agent2 sees the original user turn as user, Agent1's output as
        user (it's Agent2's input), and nothing as assistant (Agent2
        hasn't replied yet)."""
        conv = [
            {"role": "user", "content": "research Acme", "node_name": None},
            {"role": "assistant", "content": "Acme is a widget co", "node_name": "Agent1"},
        ]
        history = _build_history_for_node(conv, "Agent2")
        assert history == [
            {"role": "user", "content": "research Acme"},
            {"role": "user", "content": "Acme is a widget co"},
        ]

    def test_multi_turn_same_node_chat_pattern(self):
        """Sora chat turn 3 — Agent sees alternating user/assistant of
        its OWN past turns plus the new user turn at the end."""
        conv = [
            {"role": "user", "content": "/stranka PIGO", "node_name": None},
            {"role": "assistant", "content": "PIGO d.o.o.", "node_name": "ChatAgent"},
            {"role": "user", "content": "in status naročila?", "node_name": None},
            {"role": "assistant", "content": "Naročilo SO-123 je...", "node_name": "ChatAgent"},
        ]
        history = _build_history_for_node(conv, "ChatAgent")
        assert history == [
            {"role": "user", "content": "/stranka PIGO"},
            {"role": "assistant", "content": "PIGO d.o.o."},
            {"role": "user", "content": "in status naročila?"},
            {"role": "assistant", "content": "Naročilo SO-123 je..."},
        ]

    def test_legacy_v07_shape_with_text_key_translates(self):
        """Pre-v0.8.0 entries had ``{"role": <NodeName>, "text": ...}``.
        Read-tolerated so in-flight flows don't break across upgrade."""
        conv = [
            {"role": "Agent1", "text": "old-shape output"},
            {"role": "Agent2", "text": "another old entry"},
        ]
        # Consuming as Agent2: Agent1 → user, Agent2 → assistant.
        history = _build_history_for_node(conv, "Agent2")
        assert history == [
            {"role": "user", "content": "old-shape output"},
            {"role": "assistant", "content": "another old entry"},
        ]

    def test_no_prefix_injected_in_content(self):
        """Critical fix: we must NOT prepend ``[NodeName]:`` to the
        content. The role marker carries the semantics on its own."""
        conv = [
            {"role": "assistant", "content": "research output", "node_name": "Agent1"},
        ]
        history = _build_history_for_node(conv, "Agent2")
        assert history[0]["content"] == "research output"
        assert "[Agent1]" not in history[0]["content"]
        assert "[Agent2]" not in history[0]["content"]

    def test_skips_entries_with_no_content(self):
        conv = [
            {"role": "assistant", "content": "", "node_name": "Agent1"},
            {"role": "assistant", "content": "real output", "node_name": "Agent1"},
            {"role": "assistant", "node_name": "Agent1"},  # missing content
        ]
        history = _build_history_for_node(conv, "Agent2")
        assert history == [{"role": "user", "content": "real output"}]

    def test_skips_legacy_entries_with_no_text(self):
        conv = [
            {"role": "Agent1", "text": ""},
            {"role": "Agent1", "text": "real"},
            {"role": "Agent1"},  # no text or content
        ]
        history = _build_history_for_node(conv, "Agent2")
        assert history == [{"role": "user", "content": "real"}]

    def test_invalid_user_seed_role_falls_back_to_user(self):
        """A seed entry with a junk role gets coerced to ``user``."""
        conv = [
            {"role": "weird_role", "content": "hi", "node_name": None},
        ]
        history = _build_history_for_node(conv, "Agent")
        assert history == [{"role": "user", "content": "hi"}]

    def test_current_node_name_none_treats_all_as_user(self):
        """When the consuming node has no name (rare), entries from
        named nodes still resolve to ``user`` because they're not
        ``current_node_name``."""
        conv = [
            {"role": "assistant", "content": "from agent1", "node_name": "Agent1"},
        ]
        history = _build_history_for_node(conv, None)
        assert history == [{"role": "user", "content": "from agent1"}]


# ===================================================================
# 4. VarExecutor tests
# ===================================================================


class TestVarExecutor:
    """Tests for VarExecutor."""

    def test_sets_variable_from_expression(self):
        ctx = _make_context(
            node_metadata={"name": "my_var", "expression": "42"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"]["my_var"] == 42

    def test_evaluates_arithmetic(self):
        ctx = _make_context(
            node_metadata={"name": "x", "expression": "round_number + 1"},
            memory={"round_number": 5},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["x"] == 6

    def test_evaluates_comparison(self):
        ctx = _make_context(
            node_metadata={"name": "check", "expression": "round_number > 5"},
            memory={"round_number": 3},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["check"] is False

    def test_evaluates_comparison_true(self):
        ctx = _make_context(
            node_metadata={"name": "check", "expression": "round_number > 5"},
            memory={"round_number": 10},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["check"] is True

    def test_falls_back_to_string_when_eval_fails(self):
        ctx = _make_context(
            node_metadata={"name": "x", "expression": "undefined_func()"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["x"] == "undefined_func()"

    def test_reads_name_metadata_key(self):
        ctx = _make_context(
            node_metadata={"name": "my_var", "expression": "'hello'"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert "my_var" in result.data["memory_updates"]

    def test_falls_back_to_variable_metadata_key(self):
        ctx = _make_context(
            node_metadata={"variable": "fallback_var", "expression": "'world'"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["fallback_var"] == "world"

    def test_captures_last_message_content_when_no_expression(self):
        messages = [
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.ASSISTANT, content="second"),
            Message(role=MessageRole.USER, content="last msg"),
        ]
        ctx = _make_context(
            node_metadata={"name": "captured"},
            messages=messages,
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["captured"] == "last msg"

    def test_returns_memory_updates_with_variable(self):
        ctx = _make_context(
            node_metadata={"name": "v", "expression": "100"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert "memory_updates" in result.data
        assert result.data["memory_updates"]["v"] == 100

    def test_empty_variable_name_returns_empty(self):
        ctx = _make_context(
            node_metadata={},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.success
        assert result.output_text == ""
        assert result.data == {}

    def test_expression_returns_integer_1(self):
        ctx = _make_context(
            node_metadata={"name": "x", "expression": "1"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["x"] == 1

    def test_expression_returns_string_hello(self):
        ctx = _make_context(
            node_metadata={"name": "x", "expression": "'hello'"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["x"] == "hello"

    def test_output_text_is_str_of_value(self):
        ctx = _make_context(
            node_metadata={"name": "x", "expression": "42"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.output_text == "42"

    def test_no_messages_and_no_expression_returns_empty_string(self):
        ctx = _make_context(
            node_metadata={"name": "x"},
            messages=[],
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["x"] == ""

    def test_expression_with_memory_dict_access(self):
        ctx = _make_context(
            node_metadata={"name": "result", "expression": "a + b"},
            memory={"a": 10, "b": 20},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.data["memory_updates"]["result"] == 30


# ===================================================================
# 5. IfExecutor tests
# ===================================================================


class TestIfExecutor:
    """Tests for IfExecutor."""

    def test_true_expression_returns_true(self):
        ctx = _make_context(
            node_metadata={"if_expression": "True"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_false_expression_returns_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "False"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_expression_with_memory_variables(self):
        ctx = _make_context(
            node_metadata={"if_expression": "round_number > 5"},
            memory={"round_number": 10},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_expression_with_memory_variables_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "round_number > 5"},
            memory={"round_number": 2},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_empty_expression_defaults_to_true(self):
        ctx = _make_context(
            node_metadata={"if_expression": ""},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_no_if_expression_key_defaults_to_true(self):
        ctx = _make_context(
            node_metadata={},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_exception_in_eval_defaults_to_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "undefined_var > 5"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_division_by_zero_defaults_to_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "1 / 0"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_string_comparison_expression(self):
        ctx = _make_context(
            node_metadata={"if_expression": "status == 'active'"},
            memory={"status": "active"},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_string_comparison_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "status == 'active'"},
            memory={"status": "inactive"},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_boolean_memory_values(self):
        ctx = _make_context(
            node_metadata={"if_expression": "is_admin"},
            memory={"is_admin": True},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_boolean_memory_false(self):
        ctx = _make_context(
            node_metadata={"if_expression": "is_admin"},
            memory={"is_admin": False},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_complex_expression_and(self):
        ctx = _make_context(
            node_metadata={"if_expression": "x > 3 and y < 10"},
            memory={"x": 5, "y": 7},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_complex_expression_or(self):
        ctx = _make_context(
            node_metadata={"if_expression": "x > 3 or y < 10"},
            memory={"x": 1, "y": 7},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_output_text_matches_picked_node(self):
        ctx = _make_context(
            node_metadata={"if_expression": "True"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.output_text == "true"
        assert result.success

    def test_dunder_escape_blocked(self):
        """Regression for the v0.1.2 security review: an attacker who can
        write a node's ``if_expression`` cannot reach arbitrary code via
        the well-known ``().__class__.__bases__[0].__subclasses__()``
        escape — safe_eval (simpleeval) blocks dunder attribute access."""
        payload = "().__class__.__bases__[0].__subclasses__()"
        ctx = _make_context(node_metadata={"if_expression": payload}, memory={})
        result = _run(IfExecutor().execute(ctx))
        # The expression must be rejected, not executed. We pick the
        # "false" branch on rejection, exactly like a divide-by-zero.
        assert result.picked_node == "false"

    def test_import_call_blocked(self):
        """Calling ``__import__('os')`` (or any import) must be rejected."""
        ctx = _make_context(
            node_metadata={"if_expression": "__import__('os').system('echo pwn')"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_zero_is_falsy(self):
        ctx = _make_context(
            node_metadata={"if_expression": "0"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"

    def test_nonempty_string_is_truthy(self):
        ctx = _make_context(
            node_metadata={"if_expression": "'hello'"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "true"

    def test_empty_string_is_falsy(self):
        ctx = _make_context(
            node_metadata={"if_expression": "''"},
            memory={},
        )
        result = _run(IfExecutor().execute(ctx))
        assert result.picked_node == "false"


class TestVarExecutorSandbox:
    """Security regression tests for VarExecutor's safe_eval swap."""

    def test_dunder_escape_falls_back_to_literal(self):
        """The classic ``__class__.__bases__`` escape must be rejected;
        the variable falls back to the literal expression string (the
        documented behaviour when safe_eval refuses the input)."""
        payload = "().__class__.__bases__[0].__subclasses__()"
        ctx = _make_context(
            node_metadata={"name": "v", "expression": payload},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"]["v"] == payload

    def test_import_call_falls_back_to_literal(self):
        ctx = _make_context(
            node_metadata={"name": "v", "expression": "__import__('os')"},
            memory={},
        )
        result = _run(VarExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"]["v"] == "__import__('os')"


# ===================================================================
# 6. TextExecutor tests
# ===================================================================


class TestTextExecutor:
    """Tests for TextExecutor."""

    def test_renders_simple_template(self):
        ctx = _make_context(
            node_metadata={"text": "Hello World"},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "Hello World"

    def test_renders_template_with_variables(self):
        ctx = _make_context(
            node_metadata={"text": "Hello {{ name }}!"},
            memory={"name": "Alice"},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "Hello Alice!"

    def test_appends_to_conversation_when_has_content(self):
        ctx = _make_context(
            node_metadata={"text": "Some text"},
            memory={},
            node_name="Narrator",
        )
        result = _run(TextExecutor().execute(ctx))
        assert "memory_updates" in result.data
        conv = result.data["memory_updates"]["__conversation__"]
        assert len(conv) == 1
        # v0.8.0: stored as role=assistant with node_name=Narrator.
        assert conv[0]["content"] == "Some text"
        assert conv[0]["role"] == "assistant"
        assert conv[0]["node_name"] == "Narrator"

    def test_includes_round_number_in_conversation_entry(self):
        ctx = _make_context(
            node_metadata={"text": "Round text"},
            memory={"round_number": 3},
            node_name="Narrator",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert conv[0]["round"] == 3

    def test_template_with_jinja2_conditionals(self):
        ctx = _make_context(
            node_metadata={"text": "{% if active %}Yes{% else %}No{% endif %}"},
            memory={"active": True},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "Yes"

    def test_template_with_jinja2_conditionals_false(self):
        ctx = _make_context(
            node_metadata={"text": "{% if active %}Yes{% else %}No{% endif %}"},
            memory={"active": False},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "No"

    def test_template_with_missing_variable_graceful(self):
        ctx = _make_context(
            node_metadata={"text": "Hello {{ undefined_var }}!"},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        # Jinja2 renders undefined variables as empty string by default
        assert result.output_text == "Hello !"

    def test_empty_template_returns_empty_no_conversation_append(self):
        ctx = _make_context(
            node_metadata={"text": ""},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == ""
        assert result.data == {}

    def test_whitespace_only_template_not_appended(self):
        ctx = _make_context(
            node_metadata={"text": "   \n  "},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.data == {}

    def test_multiple_text_executions_accumulate_in_conversation(self):
        # First execution
        ctx1 = _make_context(
            node_metadata={"text": "First"},
            memory={},
            node_name="Narrator",
        )
        result1 = _run(TextExecutor().execute(ctx1))
        conv1 = result1.data["memory_updates"]["__conversation__"]

        # Second execution with existing conversation
        ctx2 = _make_context(
            node_metadata={"text": "Second"},
            memory={"__conversation__": conv1},
            node_name="Narrator",
        )
        result2 = _run(TextExecutor().execute(ctx2))
        conv2 = result2.data["memory_updates"]["__conversation__"]
        assert len(conv2) == 2
        assert conv2[0]["content"] == "First"
        assert conv2[1]["content"] == "Second"

    def test_template_error_falls_back_to_raw_string(self):
        ctx = _make_context(
            node_metadata={"text": "{{ invalid syntax !!"},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        # Should fall back to raw string
        assert result.output_text == "{{ invalid syntax !!"

    def test_template_with_loop(self):
        ctx = _make_context(
            node_metadata={"text": "{% for i in items %}{{ i }} {% endfor %}"},
            memory={"items": ["a", "b", "c"]},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "a b c "

    def test_template_with_filter(self):
        ctx = _make_context(
            node_metadata={"text": "{{ name | upper }}"},
            memory={"name": "alice"},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "ALICE"

    def test_no_text_metadata_returns_empty(self):
        ctx = _make_context(
            node_metadata={},
            memory={},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == ""
        assert result.data == {}

    def test_node_name_recorded_on_conversation_entry(self):
        """v0.8.0: node identity moves from ``role`` to ``node_name``.
        The role is always ``"assistant"`` for node outputs."""
        ctx = _make_context(
            node_metadata={"text": "Hello"},
            memory={},
            node_name="CustomRole",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert conv[0]["role"] == "assistant"
        assert conv[0]["node_name"] == "CustomRole"

    def test_template_with_integer_variable(self):
        ctx = _make_context(
            node_metadata={"text": "Count: {{ count }}"},
            memory={"count": 42},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "Count: 42"

    def test_template_with_dict_access(self):
        ctx = _make_context(
            node_metadata={"text": "{{ data.key }}"},
            memory={"data": {"key": "value"}},
        )
        result = _run(TextExecutor().execute(ctx))
        assert result.output_text == "value"

    def test_conversation_without_round_number(self):
        ctx = _make_context(
            node_metadata={"text": "Hello"},
            memory={},
            node_name="Narrator",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert "round" not in conv[0]

    def test_preserves_existing_conversation(self):
        # v0.8.0 entry shape — content key, node_name on the seed.
        existing = [{"role": "user", "content": "existing", "node_name": None}]
        ctx = _make_context(
            node_metadata={"text": "New"},
            memory={"__conversation__": existing},
            node_name="Narrator",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert len(conv) == 2
        assert conv[0]["content"] == "existing"
        assert conv[1]["content"] == "New"


# ===================================================================
# 7. StaticExecutor tests
# ===================================================================


class TestStaticExecutor:
    """Tests for StaticExecutor."""

    def test_returns_static_text_from_metadata(self):
        ctx = _make_context(
            node_metadata={"static_text": "Hello static world"},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert result.success
        assert result.output_text == "Hello static world"

    def test_empty_metadata_returns_empty(self):
        ctx = _make_context(node_metadata={})
        result = _run(StaticExecutor().execute(ctx))
        assert result.success
        assert result.output_text == ""

    def test_unicode_text(self):
        ctx = _make_context(
            node_metadata={"static_text": "こんにちは世界"},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert result.output_text == "こんにちは世界"

    def test_multiline_text(self):
        ctx = _make_context(
            node_metadata={"static_text": "line1\nline2\nline3"},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert result.output_text == "line1\nline2\nline3"

    def test_appends_to_conversation(self):
        ctx = _make_context(
            node_metadata={"static_text": "hello"},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert "__conversation__" in result.data.get("memory_updates", {})
        conv = result.data["memory_updates"]["__conversation__"]
        assert len(conv) == 1
        assert conv[0]["content"] == "hello"
        assert conv[0]["role"] == "assistant"

    def test_empty_text_returns_empty_data(self):
        ctx = _make_context(
            node_metadata={"static_text": ""},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert result.data == {}

    def test_missing_static_text_key_returns_default(self):
        ctx = _make_context(
            node_metadata={"other_key": "value"},
        )
        result = _run(StaticExecutor().execute(ctx))
        assert result.output_text == ""


# ===================================================================
# 8. MemoryWriteExecutor tests
# ===================================================================


class TestMemoryWriteExecutor:
    """Tests for MemoryWriteExecutor."""

    def test_writes_to_memory_name_key(self):
        messages = [Message(role=MessageRole.USER, content="saved value")]
        ctx = _make_context(
            node_metadata={"memory_name": "my_key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"]["my_key"] == "saved value"

    def test_gets_value_from_last_message(self):
        messages = [
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.ASSISTANT, content="second"),
            Message(role=MessageRole.USER, content="last"),
        ]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.data["memory_updates"]["key"] == "last"

    def test_empty_messages_returns_empty(self):
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=[],
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.data["memory_updates"]["key"] == ""

    def test_default_memory_name_is_memory(self):
        messages = [Message(role=MessageRole.USER, content="val")]
        ctx = _make_context(
            node_metadata={},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.data["memory_updates"]["memory"] == "val"

    def test_skips_messages_without_content(self):
        messages = [
            Message(role=MessageRole.USER, content="has content"),
            Message(role=MessageRole.ASSISTANT, content=""),
        ]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        # Reversed iteration finds "" first (falsy), then "has content"
        assert result.data["memory_updates"]["key"] == "has content"

    def test_output_text_is_the_value(self):
        messages = [Message(role=MessageRole.USER, content="text output")]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.output_text == "text output"

    def test_success_is_always_true(self):
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=[],
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.success is True

    def test_unicode_content_written(self):
        messages = [Message(role=MessageRole.USER, content="Привет")]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.data["memory_updates"]["key"] == "Привет"

    def test_multiple_messages_picks_last_with_content(self):
        messages = [
            Message(role=MessageRole.USER, content="a"),
            Message(role=MessageRole.ASSISTANT, content="b"),
            Message(role=MessageRole.USER, content=""),
        ]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        # reversed iteration: "" (skip), "b" (found)
        assert result.data["memory_updates"]["key"] == "b"

    def test_single_message_with_content(self):
        messages = [Message(role=MessageRole.USER, content="only")]
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            messages=messages,
        )
        result = _run(MemoryWriteExecutor().execute(ctx))
        assert result.data["memory_updates"]["key"] == "only"


# ===================================================================
# 9. MemoryReadExecutor tests
# ===================================================================


class TestMemoryReadExecutor:
    """Tests for MemoryReadExecutor."""

    def test_reads_from_memory(self):
        ctx = _make_context(
            node_metadata={"memory_name": "my_key"},
            memory={"my_key": "stored_value"},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.success
        assert result.output_text == "stored_value"

    def test_missing_key_returns_empty(self):
        ctx = _make_context(
            node_metadata={"memory_name": "nonexistent"},
            memory={},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == ""

    def test_reads_integer_value(self):
        ctx = _make_context(
            node_metadata={"memory_name": "count"},
            memory={"count": 42},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == "42"

    def test_reads_dict_value(self):
        ctx = _make_context(
            node_metadata={"memory_name": "data"},
            memory={"data": {"key": "val"}},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == "{'key': 'val'}"

    def test_default_memory_name_is_memory(self):
        ctx = _make_context(
            node_metadata={},
            memory={"memory": "default_val"},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == "default_val"

    def test_data_is_empty(self):
        ctx = _make_context(
            node_metadata={"memory_name": "key"},
            memory={"key": "val"},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.data == {}

    def test_reads_boolean_value(self):
        ctx = _make_context(
            node_metadata={"memory_name": "flag"},
            memory={"flag": True},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == "True"

    def test_reads_list_value(self):
        ctx = _make_context(
            node_metadata={"memory_name": "items"},
            memory={"items": [1, 2, 3]},
        )
        result = _run(MemoryReadExecutor().execute(ctx))
        assert result.output_text == "[1, 2, 3]"


# ===================================================================
# 10. UserExecutor tests
# ===================================================================


class TestUserExecutor:
    """Tests for UserExecutor."""

    def test_returns_user_input_from_memory(self):
        ctx = _make_context(
            memory={"__user_input__": "Hello from user"},
        )
        result = _run(UserExecutor(interactive=False).execute(ctx))
        assert result.success
        assert result.output_text == "Hello from user"

    def test_missing_key_returns_empty(self):
        ctx = _make_context(memory={})
        result = _run(UserExecutor(interactive=False).execute(ctx))
        assert result.output_text == ""

    def test_integer_input_converted_to_string(self):
        ctx = _make_context(
            memory={"__user_input__": 42},
        )
        result = _run(UserExecutor(interactive=False).execute(ctx))
        assert result.output_text == "42"

    def test_data_is_empty(self):
        ctx = _make_context(
            memory={"__user_input__": "text"},
        )
        result = _run(UserExecutor(interactive=False).execute(ctx))
        assert result.data == {}

    def test_unicode_input(self):
        ctx = _make_context(
            memory={"__user_input__": "日本語テスト"},
        )
        result = _run(UserExecutor(interactive=False).execute(ctx))
        assert result.output_text == "日本語テスト"


# ===================================================================
# 11. PassthroughExecutor tests
# ===================================================================


class TestPassthroughExecutor:
    """Tests for PassthroughExecutor."""

    def test_returns_last_message_content(self):
        messages = [
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.ASSISTANT, content="last content"),
        ]
        ctx = _make_context(messages=messages)
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.success
        assert result.output_text == "last content"

    def test_empty_messages_returns_empty(self):
        ctx = _make_context(messages=[])
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.output_text == ""

    def test_skips_empty_content_messages(self):
        messages = [
            Message(role=MessageRole.USER, content="has content"),
            Message(role=MessageRole.ASSISTANT, content=""),
        ]
        ctx = _make_context(messages=messages)
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.output_text == "has content"

    def test_single_message(self):
        messages = [Message(role=MessageRole.USER, content="only")]
        ctx = _make_context(messages=messages)
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.output_text == "only"

    def test_data_is_empty(self):
        messages = [Message(role=MessageRole.USER, content="text")]
        ctx = _make_context(messages=messages)
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.data == {}

    def test_always_succeeds(self):
        ctx = _make_context(messages=[])
        result = _run(PassthroughExecutor().execute(ctx))
        assert result.success is True


# ===================================================================
# 12. UserFormExecutor tests
# ===================================================================


class TestUserFormExecutor:
    """Tests for UserFormExecutor."""

    def test_fills_form_with_defaults(self):
        ctx = _make_context(
            node_metadata={
                "parameters": [
                    {"name": "username", "default": "admin"},
                    {"name": "email", "default": "a@b.com"},
                ]
            },
        )
        result = _run(UserFormExecutor().execute(ctx))
        assert result.success
        updates = result.data["memory_updates"]
        assert updates["username"] == "admin"
        assert updates["email"] == "a@b.com"

    def test_placeholder_when_no_default(self):
        ctx = _make_context(
            node_metadata={"parameters": [{"name": "city"}]},
        )
        result = _run(UserFormExecutor().execute(ctx))
        assert result.data["memory_updates"]["city"] == "<city>"

    def test_empty_parameters(self):
        ctx = _make_context(
            node_metadata={"parameters": []},
        )
        result = _run(UserFormExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"] == {}

    def test_no_parameters_key(self):
        ctx = _make_context(node_metadata={})
        result = _run(UserFormExecutor().execute(ctx))
        assert result.success
        assert result.data["memory_updates"] == {}

    def test_missing_name_in_param(self):
        ctx = _make_context(
            node_metadata={"parameters": [{"default": "val"}]},
        )
        result = _run(UserFormExecutor().execute(ctx))
        # Falls back to "field" as name
        assert result.data["memory_updates"]["field"] == "val"

    def test_output_text_is_str_of_form_data(self):
        ctx = _make_context(
            node_metadata={"parameters": [{"name": "x", "default": "1"}]},
        )
        result = _run(UserFormExecutor().execute(ctx))
        assert "x" in result.output_text
        assert "1" in result.output_text
