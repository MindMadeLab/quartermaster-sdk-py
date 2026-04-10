"""
Variable/memory tools: Set, Get, and List in-memory key-value pairs.

All three tools share a class-level ``_store`` dict that persists across
calls within the same process.  An optional ``store`` parameter in
``__init__`` allows dependency injection for testing or isolation.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


class _VariableStoreMixin:
    """Mixin providing a shared class-level variable store."""

    _store: dict[str, Any] = {}

    def __init__(self, store: dict[str, Any] | None = None) -> None:
        if store is not None:
            self._store = store


class SetVariableTool(_VariableStoreMixin, AbstractTool):
    """Store a key-value pair in the in-memory variable store."""

    def name(self) -> str:
        return "set_variable"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="The variable name (key) to store.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="value",
                description="The value to store.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Store a key-value pair in memory.",
            long_description=(
                "Stores a named variable in a shared in-memory dictionary. "
                "Overwrites any existing value for the same key."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Set a variable in the store.

        Args:
            name: Variable name.
            value: Value to store.

        Returns:
            ToolResult confirming the variable was stored.
        """
        var_name: str = kwargs.get("name", "")
        value: Any = kwargs.get("value")

        if not var_name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        if value is None and "value" not in kwargs:
            return ToolResult(success=False, error="Parameter 'value' is required")

        self._store[var_name] = value
        return ToolResult(
            success=True,
            data={"name": var_name, "value": value, "message": f"Variable '{var_name}' set."},
        )


class GetVariableTool(_VariableStoreMixin, AbstractTool):
    """Retrieve a variable from the in-memory store."""

    def name(self) -> str:
        return "get_variable"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="The variable name (key) to retrieve.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="default",
                description="Default value to return if the variable is not found.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Retrieve a variable from memory.",
            long_description=(
                "Looks up a named variable in the shared in-memory store. "
                "Returns the stored value, or a default if the key is missing."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Get a variable from the store.

        Args:
            name: Variable name.
            default: Fallback value (default: None).

        Returns:
            ToolResult with the variable's value.
        """
        var_name: str = kwargs.get("name", "")
        default: Any = kwargs.get("default", None)

        if not var_name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        found = var_name in self._store
        value = self._store.get(var_name, default)
        return ToolResult(
            success=True,
            data={"name": var_name, "value": value, "found": found},
        )


class ListVariablesTool(_VariableStoreMixin, AbstractTool):
    """List variable names in the in-memory store."""

    def name(self) -> str:
        return "list_variables"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="prefix",
                description="Optional prefix to filter variable names.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="List stored variable names.",
            long_description=(
                "Returns a list of all variable names in the shared store, "
                "optionally filtered by a key prefix."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """List variable names, optionally filtered by prefix.

        Args:
            prefix: If provided, only return names starting with this string.

        Returns:
            ToolResult with a list of variable names.
        """
        prefix: str | None = kwargs.get("prefix", None)

        names = sorted(self._store.keys())
        if prefix is not None:
            names = [n for n in names if n.startswith(prefix)]

        return ToolResult(
            success=True,
            data={"names": names, "count": len(names)},
        )
