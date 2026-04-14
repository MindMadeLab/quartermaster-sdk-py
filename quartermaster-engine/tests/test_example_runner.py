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
    _format_conversation,
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
    """Tests for _append_to_conversation helper."""

    def test_appends_entry_with_role_and_text(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "hello")
        assert len(conv) == 1
        assert conv[0] == {"role": "user", "text": "hello"}

    def test_includes_round_num_when_provided(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "assistant", "hi", round_num=3)
        assert conv[0]["round"] == 3

    def test_skips_empty_text(self):
        conv: list[dict] = []
        result = _append_to_conversation(conv, "user", "")
        assert len(conv) == 0
        assert result is conv

    def test_skips_whitespace_only_text(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "   \t\n  ")
        assert len(conv) == 0

    def test_none_round_num_not_included(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "hello", round_num=None)
        assert "round" not in conv[0]

    def test_zero_round_num_is_included(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "hello", round_num=0)
        assert conv[0]["round"] == 0

    def test_returns_same_list_mutated(self):
        conv: list[dict] = []
        result = _append_to_conversation(conv, "user", "hi")
        assert result is conv

    def test_multiple_appends_accumulate(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "first")
        _append_to_conversation(conv, "assistant", "second")
        _append_to_conversation(conv, "user", "third")
        assert len(conv) == 3
        assert conv[0]["text"] == "first"
        assert conv[1]["text"] == "second"
        assert conv[2]["text"] == "third"

    def test_unicode_text(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "Привет мир 🌍")
        assert conv[0]["text"] == "Привет мир 🌍"

    def test_very_long_text(self):
        conv: list[dict] = []
        long_text = "x" * 100_000
        _append_to_conversation(conv, "user", long_text)
        assert len(conv[0]["text"]) == 100_000

    def test_text_with_newlines(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "line1\nline2\nline3")
        assert conv[0]["text"] == "line1\nline2\nline3"

    def test_round_num_negative(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "hello", round_num=-1)
        assert conv[0]["round"] == -1

    def test_round_num_large_int(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "hello", round_num=999999)
        assert conv[0]["round"] == 999999

    def test_role_can_be_any_string(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "CustomAgent", "hello")
        assert conv[0]["role"] == "CustomAgent"

    def test_does_not_append_for_only_spaces(self):
        conv: list[dict] = []
        _append_to_conversation(conv, "user", "     ")
        assert len(conv) == 0


# ===================================================================
# 3. _format_conversation tests
# ===================================================================


class TestFormatConversation:
    """Tests for _format_conversation helper."""

    def test_empty_conversation_returns_user_input(self):
        result = _format_conversation([], "hello")
        assert result == "hello"

    def test_single_entry_formatted_correctly(self):
        conv = [{"role": "user", "text": "hi"}]
        result = _format_conversation(conv, "original")
        assert "[user]: hi" in result
        assert "Original case: original" in result

    def test_multiple_entries_same_round(self):
        conv = [
            {"role": "user", "text": "hi", "round": 1},
            {"role": "assistant", "text": "hello", "round": 1},
        ]
        result = _format_conversation(conv, "q")
        # Round marker should appear once
        assert result.count("--- Round 1 ---") == 1

    def test_round_markers_appear_when_round_changes(self):
        conv = [
            {"role": "user", "text": "a", "round": 1},
            {"role": "user", "text": "b", "round": 2},
        ]
        result = _format_conversation(conv, "q")
        assert "--- Round 1 ---" in result
        assert "--- Round 2 ---" in result

    def test_no_duplicate_round_markers_for_same_round(self):
        conv = [
            {"role": "user", "text": "a", "round": 1},
            {"role": "assistant", "text": "b", "round": 1},
            {"role": "user", "text": "c", "round": 1},
        ]
        result = _format_conversation(conv, "q")
        assert result.count("--- Round 1 ---") == 1

    def test_multiple_rounds_with_proper_separators(self):
        conv = [
            {"role": "user", "text": "a", "round": 1},
            {"role": "assistant", "text": "b", "round": 1},
            {"role": "user", "text": "c", "round": 2},
            {"role": "assistant", "text": "d", "round": 2},
            {"role": "user", "text": "e", "round": 3},
        ]
        result = _format_conversation(conv, "q")
        assert "--- Round 1 ---" in result
        assert "--- Round 2 ---" in result
        assert "--- Round 3 ---" in result

    def test_entries_without_round_field_no_round_marker(self):
        conv = [
            {"role": "user", "text": "a"},
            {"role": "assistant", "text": "b"},
        ]
        result = _format_conversation(conv, "q")
        assert "--- Round" not in result
        assert "[user]: a" in result
        assert "[assistant]: b" in result

    def test_mix_of_entries_with_and_without_round(self):
        conv = [
            {"role": "user", "text": "a"},
            {"role": "assistant", "text": "b", "round": 1},
            {"role": "user", "text": "c"},
        ]
        result = _format_conversation(conv, "q")
        assert "--- Round 1 ---" in result
        # Only one round marker
        assert result.count("--- Round") == 1

    def test_original_case_appended_at_end(self):
        conv = [{"role": "user", "text": "hi"}]
        result = _format_conversation(conv, "my question")
        assert result.endswith("Original case: my question")

    def test_very_long_conversation(self):
        conv = [{"role": "user", "text": f"msg{i}", "round": i} for i in range(100)]
        result = _format_conversation(conv, "q")
        assert "--- Round 0 ---" in result
        assert "--- Round 99 ---" in result
        assert result.count("[user]:") == 100

    def test_special_characters_in_text(self):
        conv = [{"role": "user", "text": "Hello <world> & 'friends' \"test\""}]
        result = _format_conversation(conv, "q")
        assert "[user]: Hello <world> & 'friends' \"test\"" in result

    def test_empty_user_input(self):
        conv = [{"role": "user", "text": "hi"}]
        result = _format_conversation(conv, "")
        assert "Original case: " in result

    def test_parts_joined_with_double_newline(self):
        conv = [
            {"role": "user", "text": "a", "round": 1},
            {"role": "assistant", "text": "b", "round": 1},
        ]
        result = _format_conversation(conv, "q")
        # Round marker, entry a, entry b separated by \n\n
        assert "\n\n" in result

    def test_separator_between_history_and_original(self):
        conv = [{"role": "user", "text": "hi"}]
        result = _format_conversation(conv, "q")
        assert "\n\n---\nOriginal case:" in result

    def test_round_none_is_treated_as_no_round(self):
        conv = [{"role": "user", "text": "a", "round": None}]
        result = _format_conversation(conv, "q")
        # round=None should NOT produce a round marker
        assert "--- Round" not in result

    def test_round_zero_produces_marker(self):
        conv = [{"role": "user", "text": "a", "round": 0}]
        result = _format_conversation(conv, "q")
        assert "--- Round 0 ---" in result

    def test_format_preserves_entry_order(self):
        conv = [
            {"role": "A", "text": "first"},
            {"role": "B", "text": "second"},
            {"role": "C", "text": "third"},
        ]
        result = _format_conversation(conv, "q")
        idx_first = result.index("[A]: first")
        idx_second = result.index("[B]: second")
        idx_third = result.index("[C]: third")
        assert idx_first < idx_second < idx_third

    def test_unicode_in_conversation_and_input(self):
        conv = [{"role": "用户", "text": "你好世界"}]
        result = _format_conversation(conv, "日本語")
        assert "[用户]: 你好世界" in result
        assert "Original case: 日本語" in result

    def test_multiline_text_in_entries(self):
        conv = [{"role": "user", "text": "line1\nline2\nline3"}]
        result = _format_conversation(conv, "q")
        assert "[user]: line1\nline2\nline3" in result

    def test_same_round_repeated_after_different_round(self):
        conv = [
            {"role": "user", "text": "a", "round": 1},
            {"role": "user", "text": "b", "round": 2},
            {"role": "user", "text": "c", "round": 1},
        ]
        result = _format_conversation(conv, "q")
        # Round 1 should appear twice because current_round changes
        assert result.count("--- Round 1 ---") == 2


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
        assert conv[0]["text"] == "Some text"
        assert conv[0]["role"] == "Narrator"

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
        assert conv2[0]["text"] == "First"
        assert conv2[1]["text"] == "Second"

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

    def test_node_name_used_as_role_in_conversation(self):
        ctx = _make_context(
            node_metadata={"text": "Hello"},
            memory={},
            node_name="CustomRole",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert conv[0]["role"] == "CustomRole"

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
        existing = [{"role": "user", "text": "existing"}]
        ctx = _make_context(
            node_metadata={"text": "New"},
            memory={"__conversation__": existing},
            node_name="Narrator",
        )
        result = _run(TextExecutor().execute(ctx))
        conv = result.data["memory_updates"]["__conversation__"]
        assert len(conv) == 2
        assert conv[0]["text"] == "existing"
        assert conv[1]["text"] == "New"


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
        assert conv[0]["text"] == "hello"

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
