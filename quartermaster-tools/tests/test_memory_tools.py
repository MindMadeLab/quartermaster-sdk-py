"""Tests for the memory/variable tools."""

from __future__ import annotations

from quartermaster_tools.builtin.memory.tools import (
    GetVariableTool,
    ListVariablesTool,
    SetVariableTool,
)


def _make_tools(store: dict | None = None) -> tuple[SetVariableTool, GetVariableTool, ListVariablesTool]:
    """Create a set of memory tools sharing an isolated store."""
    if store is None:
        store = {}
    return SetVariableTool(store=store), GetVariableTool(store=store), ListVariablesTool(store=store)


# --- SetVariableTool ---


class TestSetVariableTool:
    def test_set_returns_success(self):
        setter, _, _ = _make_tools()
        result = setter.run(name="x", value="42")
        assert result.success is True
        assert result.data["name"] == "x"
        assert result.data["value"] == "42"

    def test_set_missing_name(self):
        setter, _, _ = _make_tools()
        result = setter.run(name="", value="v")
        assert result.success is False
        assert "name" in result.error.lower()

    def test_set_missing_value(self):
        setter, _, _ = _make_tools()
        result = setter.run(name="x")
        assert result.success is False
        assert "value" in result.error.lower()

    def test_set_overwrite(self):
        store: dict = {}
        setter, getter, _ = _make_tools(store)
        setter.run(name="x", value="first")
        setter.run(name="x", value="second")
        result = getter.run(name="x")
        assert result.data["value"] == "second"

    def test_set_none_value_explicit(self):
        """Setting value=None explicitly should work."""
        store: dict = {}
        setter, getter, _ = _make_tools(store)
        result = setter.run(name="x", value=None)
        assert result.success is True
        r2 = getter.run(name="x")
        assert r2.data["value"] is None
        assert r2.data["found"] is True


# --- GetVariableTool ---


class TestGetVariableTool:
    def test_get_existing(self):
        store: dict = {}
        setter, getter, _ = _make_tools(store)
        setter.run(name="color", value="blue")
        result = getter.run(name="color")
        assert result.success is True
        assert result.data["value"] == "blue"
        assert result.data["found"] is True

    def test_get_missing_returns_default_none(self):
        _, getter, _ = _make_tools()
        result = getter.run(name="nonexistent")
        assert result.success is True
        assert result.data["value"] is None
        assert result.data["found"] is False

    def test_get_missing_with_custom_default(self):
        _, getter, _ = _make_tools()
        result = getter.run(name="nonexistent", default="fallback")
        assert result.success is True
        assert result.data["value"] == "fallback"
        assert result.data["found"] is False

    def test_get_missing_name(self):
        _, getter, _ = _make_tools()
        result = getter.run(name="")
        assert result.success is False

    def test_get_does_not_store_default(self):
        """Getting with a default should not persist the default."""
        store: dict = {}
        _, getter, _ = _make_tools(store)
        getter.run(name="k", default="val")
        assert "k" not in store


# --- ListVariablesTool ---


class TestListVariablesTool:
    def test_list_empty_store(self):
        _, _, lister = _make_tools()
        result = lister.run()
        assert result.success is True
        assert result.data["names"] == []
        assert result.data["count"] == 0

    def test_list_all_variables(self):
        store: dict = {}
        setter, _, lister = _make_tools(store)
        setter.run(name="b", value="2")
        setter.run(name="a", value="1")
        result = lister.run()
        assert result.data["names"] == ["a", "b"]
        assert result.data["count"] == 2

    def test_list_with_prefix(self):
        store: dict = {}
        setter, _, lister = _make_tools(store)
        setter.run(name="user.name", value="Alice")
        setter.run(name="user.age", value="30")
        setter.run(name="system.version", value="1.0")
        result = lister.run(prefix="user.")
        assert result.data["names"] == ["user.age", "user.name"]
        assert result.data["count"] == 2

    def test_list_prefix_no_match(self):
        store: dict = {}
        setter, _, lister = _make_tools(store)
        setter.run(name="foo", value="bar")
        result = lister.run(prefix="zzz")
        assert result.data["names"] == []
        assert result.data["count"] == 0


# --- Store isolation ---


class TestStoreIsolation:
    def test_injected_stores_are_independent(self):
        store_a: dict = {}
        store_b: dict = {}
        setter_a = SetVariableTool(store=store_a)
        setter_b = SetVariableTool(store=store_b)
        getter_a = GetVariableTool(store=store_a)
        getter_b = GetVariableTool(store=store_b)

        setter_a.run(name="x", value="from_a")
        setter_b.run(name="x", value="from_b")

        assert getter_a.run(name="x").data["value"] == "from_a"
        assert getter_b.run(name="x").data["value"] == "from_b"

    def test_shared_store_across_tools(self):
        store: dict = {}
        setter = SetVariableTool(store=store)
        getter = GetVariableTool(store=store)
        lister = ListVariablesTool(store=store)

        setter.run(name="key", value="val")
        assert getter.run(name="key").data["value"] == "val"
        assert "key" in lister.run().data["names"]


# --- Tool metadata ---


class TestMemoryToolMetadata:
    def test_set_tool_info(self):
        setter = SetVariableTool(store={})
        assert setter.name() == "set_variable"
        assert setter.version() == "1.0.0"
        info = setter.info()
        assert info.name == "set_variable"
        assert info.is_local is True

    def test_get_tool_info(self):
        getter = GetVariableTool(store={})
        assert getter.name() == "get_variable"

    def test_list_tool_info(self):
        lister = ListVariablesTool(store={})
        assert lister.name() == "list_variables"
