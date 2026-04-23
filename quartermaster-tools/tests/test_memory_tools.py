"""Tests for the memory/variable tools."""

from __future__ import annotations

from quartermaster_tools.builtin.memory.tools import (
    get_variable,
    list_variables,
    set_variable,
    clear_store,
    get_store,
)

import pytest


@pytest.fixture(autouse=True)
def _isolate_store():
    """Clear the shared module-level store before and after each test."""
    clear_store()
    yield
    clear_store()


# --- set_variable ---


class TestSetVariableTool:
    def test_set_returns_success(self):
        result = set_variable.run(name="x", value="42")
        assert result.success is True
        assert result.data["name"] == "x"
        assert result.data["value"] == "42"

    def test_set_missing_name(self):
        result = set_variable.run(name="", value="v")
        assert result.success is False
        assert "name" in result.error.lower()

    def test_set_missing_value(self):
        result = set_variable.run(name="x")
        # value is a required param, so the underlying function is called
        # without the required arg — FunctionTool catches the TypeError
        assert result.success is False

    def test_set_overwrite(self):
        set_variable.run(name="x", value="first")
        set_variable.run(name="x", value="second")
        result = get_variable.run(name="x")
        assert result.data["value"] == "second"

    def test_set_none_value_explicit(self):
        """Setting value=None explicitly should work."""
        result = set_variable.run(name="x", value=None)
        assert result.success is True
        r2 = get_variable.run(name="x")
        assert r2.data["value"] is None
        assert r2.data["found"] is True


# --- get_variable ---


class TestGetVariableTool:
    def test_get_existing(self):
        set_variable.run(name="color", value="blue")
        result = get_variable.run(name="color")
        assert result.success is True
        assert result.data["value"] == "blue"
        assert result.data["found"] is True

    def test_get_missing_returns_default_none(self):
        result = get_variable.run(name="nonexistent")
        assert result.success is True
        assert result.data["value"] is None
        assert result.data["found"] is False

    def test_get_missing_with_custom_default(self):
        result = get_variable.run(name="nonexistent", default="fallback")
        assert result.success is True
        assert result.data["value"] == "fallback"
        assert result.data["found"] is False

    def test_get_missing_name(self):
        result = get_variable.run(name="")
        assert result.success is False

    def test_get_does_not_store_default(self):
        """Getting with a default should not persist the default."""
        get_variable.run(name="k", default="val")
        assert "k" not in get_store()


# --- list_variables ---


class TestListVariablesTool:
    def test_list_empty_store(self):
        result = list_variables.run()
        assert result.success is True
        assert result.data["names"] == []
        assert result.data["count"] == 0

    def test_list_all_variables(self):
        set_variable.run(name="b", value="2")
        set_variable.run(name="a", value="1")
        result = list_variables.run()
        assert result.data["names"] == ["a", "b"]
        assert result.data["count"] == 2

    def test_list_with_prefix(self):
        set_variable.run(name="user.name", value="Alice")
        set_variable.run(name="user.age", value="30")
        set_variable.run(name="system.version", value="1.0")
        result = list_variables.run(prefix="user.")
        assert result.data["names"] == ["user.age", "user.name"]
        assert result.data["count"] == 2

    def test_list_prefix_no_match(self):
        set_variable.run(name="foo", value="bar")
        result = list_variables.run(prefix="zzz")
        assert result.data["names"] == []
        assert result.data["count"] == 0


# --- Store isolation ---


class TestStoreIsolation:
    def test_shared_module_store(self):
        """All tools share the same module-level store."""
        set_variable.run(name="x", value="from_set")
        assert get_variable.run(name="x").data["value"] == "from_set"
        assert "x" in list_variables.run().data["names"]

    def test_clear_store_resets(self):
        """clear_store() empties the shared store."""
        set_variable.run(name="a", value="1")
        clear_store()
        result = get_variable.run(name="a")
        assert result.data["found"] is False

    def test_get_store_returns_internal_dict(self):
        """get_store() provides direct access to the underlying dict."""
        set_variable.run(name="key", value="val")
        store = get_store()
        assert store["key"] == "val"


# --- Tool metadata ---


class TestMemoryToolMetadata:
    def test_set_tool_info(self):
        assert set_variable.name() == "set_variable"
        assert set_variable.version() == "1.0.0"
        info = set_variable.info()
        assert info.name == "set_variable"

    def test_get_tool_info(self):
        assert get_variable.name() == "get_variable"

    def test_list_tool_info(self):
        assert list_variables.name() == "list_variables"
