"""
Variable/memory tools: Set, Get, and List in-memory key-value pairs.

All three tools share a module-level ``_default_store`` dict that persists
across calls within the same process.  The ``create_memory_tools`` factory
returns tools bound to a custom store for testing or isolation.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult

# Sentinel for distinguishing "not provided" from None.
_MISSING = object()

# Default module-level store shared by the default tool instances.
_default_store: dict[str, Any] = {}


def _make_set_variable(store: dict[str, Any]) -> Any:
    """Create a set_variable tool bound to the given store."""

    @tool()
    def set_variable(name: str, value: str = _MISSING) -> ToolResult:
        """Store a key-value pair in memory.

        Stores a named variable in a shared in-memory dictionary.
        Overwrites any existing value for the same key.

        Args:
            name: The variable name (key) to store.
            value: The value to store.
        """
        if not name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        if value is _MISSING:
            return ToolResult(success=False, error="Parameter 'value' is required")

        store[name] = value
        return ToolResult(
            success=True,
            data={"name": name, "value": value, "message": f"Variable '{name}' set."},
        )

    return set_variable


def _make_get_variable(store: dict[str, Any]) -> Any:
    """Create a get_variable tool bound to the given store."""

    @tool()
    def get_variable(name: str, default: str = None) -> ToolResult:
        """Retrieve a variable from memory.

        Looks up a named variable in the shared in-memory store.
        Returns the stored value, or a default if the key is missing.

        Args:
            name: The variable name (key) to retrieve.
            default: Default value to return if the variable is not found.
        """
        if not name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        found = name in store
        value = store.get(name, default)
        return ToolResult(
            success=True,
            data={"name": name, "value": value, "found": found},
        )

    return get_variable


def _make_list_variables(store: dict[str, Any]) -> Any:
    """Create a list_variables tool bound to the given store."""

    @tool()
    def list_variables(prefix: str = None) -> ToolResult:
        """List stored variable names.

        Returns a list of all variable names in the shared store,
        optionally filtered by a key prefix.

        Args:
            prefix: Optional prefix to filter variable names.
        """
        names = sorted(store.keys())
        if prefix is not None:
            names = [n for n in names if n.startswith(prefix)]

        return ToolResult(
            success=True,
            data={"names": names, "count": len(names)},
        )

    return list_variables


def create_memory_tools(
    store: dict[str, Any] | None = None,
) -> tuple:
    """Create a set of memory tools sharing an isolated store.

    Args:
        store: Optional dict to use as backing store. If None, creates a new one.

    Returns:
        Tuple of (set_variable, get_variable, list_variables) function tools.
    """
    if store is None:
        store = {}
    return (
        _make_set_variable(store),
        _make_get_variable(store),
        _make_list_variables(store),
    )


# ---------------------------------------------------------------------------
# Helpers for test isolation
# ---------------------------------------------------------------------------


def get_store() -> dict[str, Any]:
    """Return the default module-level store (for test inspection)."""
    return _default_store


def clear_store() -> None:
    """Clear all entries from the default module-level store."""
    _default_store.clear()


# Default tool instances using the module-level store.
set_variable = _make_set_variable(_default_store)
get_variable = _make_get_variable(_default_store)
list_variables = _make_list_variables(_default_store)


# Backward-compatible class-like aliases.
