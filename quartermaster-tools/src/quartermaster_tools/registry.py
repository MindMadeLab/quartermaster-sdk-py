"""
Tool registry with version-aware lookup, decorator registration, and plugin discovery.

Provides a central registry for managing tools with features:
- Register tools by instance or class
- Version-aware lookup: get a specific version or latest
- Decorator-based registration: @register_tool
- Plugin discovery via entry points
- JSON Schema export for LLM function calling
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, TypeVar

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor

T = TypeVar("T", bound=AbstractTool)

# Module-level default registry (populated by @register_tool decorator)
_default_registry: ToolRegistry | None = None


def _get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


class ToolRegistry:
    """Registry for managing tool instances with version-aware lookup.

    Tools are stored by name, with multiple versions supported per name.
    The registry supports lazy initialization, decorator registration,
    and plugin discovery via Python entry points.
    """

    ENTRY_POINT_GROUP = "quartermaster_tools"

    def __init__(self) -> None:
        # {tool_name: {version: tool_instance}}
        self._tools: dict[str, dict[str, AbstractTool]] = {}
        self._plugins_loaded = False

    def register(self, tool: AbstractTool) -> None:
        """Register a tool instance.

        Args:
            tool: An AbstractTool instance to register.

        Raises:
            ValueError: If a tool with the same name and version is already registered.
        """
        name = tool.name()
        version = tool.version()
        if name not in self._tools:
            self._tools[name] = {}
        if version in self._tools[name]:
            raise ValueError(f"Tool '{name}' version '{version}' is already registered")
        self._tools[name][version] = tool

    def get(self, name: str, version: str | None = None) -> AbstractTool:
        """Look up a tool by name and optional version.

        If no version is specified, returns the latest registered version
        (by insertion order — last registered wins).

        Args:
            name: Tool name.
            version: Optional specific version string.

        Returns:
            The matching AbstractTool instance.

        Raises:
            KeyError: If no tool is found with the given name/version.
        """
        self._ensure_plugins_loaded()
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        versions = self._tools[name]
        if version is not None:
            if version not in versions:
                available = ", ".join(versions.keys())
                raise KeyError(
                    f"Tool '{name}' version '{version}' not found. Available versions: {available}"
                )
            return versions[version]
        # Return latest (last inserted)
        return list(versions.values())[-1]

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for all registered tools."""
        self._ensure_plugins_loaded()
        descriptors: list[ToolDescriptor] = []
        for versions in self._tools.values():
            for tool in versions.values():
                descriptors.append(tool.info())
        return descriptors

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        self._ensure_plugins_loaded()
        return list(self._tools.keys())

    def unregister(self, name: str, version: str | None = None) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name.
            version: If given, only remove that version. Otherwise remove all versions.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        if version is not None:
            if version not in self._tools[name]:
                raise KeyError(f"Tool '{name}' version '{version}' not found")
            del self._tools[name][version]
            if not self._tools[name]:
                del self._tools[name]
        else:
            del self._tools[name]

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    def __len__(self) -> int:
        """Return total number of tool instances (across all versions)."""
        return sum(len(versions) for versions in self._tools.values())

    def __contains__(self, name: str) -> bool:
        """Check if a tool name is registered."""
        self._ensure_plugins_loaded()
        return name in self._tools

    # --- Plugin Discovery ---

    def load_plugins(self) -> None:
        """Discover and register tools from installed packages via entry points.

        Looks for entry points in the 'quartermaster_tools' group. Each entry point
        should resolve to an AbstractTool subclass (will be instantiated)
        or an AbstractTool instance.
        """
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.9+ supports .select()
            if hasattr(eps, "select"):
                tool_eps = eps.select(group=self.ENTRY_POINT_GROUP)
            else:
                tool_eps = getattr(eps, "get", lambda g, d: d)(self.ENTRY_POINT_GROUP, [])

            for ep in tool_eps:
                try:
                    obj = ep.load()
                    if isinstance(obj, AbstractTool):
                        self.register(obj)
                    elif isinstance(obj, type) and issubclass(obj, AbstractTool):
                        self.register(obj())
                except Exception:
                    # Skip broken plugins silently
                    pass
        except Exception:
            pass
        self._plugins_loaded = True

    def _ensure_plugins_loaded(self) -> None:
        """Load plugins on first access (lazy initialization)."""
        if not self._plugins_loaded:
            self.load_plugins()

    # --- JSON Schema Export ---

    def to_json_schema(self) -> list[dict[str, Any]]:
        """Export all tools as JSON Schema definitions for LLM function calling.

        Returns a list of tool schemas, each containing:
        - name, description, and parameters in JSON Schema format.
        """
        self._ensure_plugins_loaded()
        schemas: list[dict[str, Any]] = []
        for versions in self._tools.values():
            for tool in versions.values():
                schemas.append(_tool_to_json_schema(tool))
        return schemas

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Export tools in OpenAI function calling format."""
        self._ensure_plugins_loaded()
        tools: list[dict[str, Any]] = []
        for versions in self._tools.values():
            for tool in versions.values():
                schema = _tool_to_json_schema(tool)
                tools.append(
                    {
                        "type": "function",
                        "function": schema,
                    }
                )
        return tools

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Export tools in Anthropic tool use format."""
        self._ensure_plugins_loaded()
        tools: list[dict[str, Any]] = []
        for versions in self._tools.values():
            for tool in versions.values():
                schema = _tool_to_json_schema(tool)
                tools.append(
                    {
                        "name": schema["name"],
                        "description": schema["description"],
                        "input_schema": schema["parameters"],
                    }
                )
        return tools

    def to_mcp_tools(self) -> list[dict[str, Any]]:
        """Export tools in MCP (Model Context Protocol) format."""
        self._ensure_plugins_loaded()
        tools: list[dict[str, Any]] = []
        for versions in self._tools.values():
            for tool in versions.values():
                schema = _tool_to_json_schema(tool)
                tools.append(
                    {
                        "name": schema["name"],
                        "description": schema["description"],
                        "inputSchema": schema["parameters"],
                    }
                )
        return tools


# --- Decorator Registration ---


def register_tool(cls: type[T]) -> type[T]:
    """Class decorator that registers a tool with the default registry.

    Usage:
        @register_tool
        class MyTool(AbstractTool):
            ...
    """
    registry = _get_default_registry()
    registry.register(cls())
    return cls


def get_default_registry() -> ToolRegistry:
    """Return the module-level default registry used by @register_tool."""
    return _get_default_registry()


# --- Schema Helpers ---

_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "number": "number",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


def _param_to_json_schema(param: Any) -> dict[str, Any]:
    """Convert a ToolParameter to a JSON Schema property definition."""
    schema: dict[str, Any] = {
        "type": _TYPE_MAP.get(param.type, param.type),
        "description": param.description,
    }
    if param.default is not None:
        schema["default"] = param.default
    if param.options:
        schema["enum"] = [opt.value for opt in param.options]
    return schema


def _tool_to_json_schema(tool: AbstractTool) -> dict[str, Any]:
    """Convert a tool to a JSON Schema function definition."""
    info = tool.info()
    params = tool.parameters()

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in params:
        properties[param.name] = _param_to_json_schema(param)
        if param.required:
            required.append(param.name)

    schema: dict[str, Any] = {
        "name": info.name,
        "description": info.short_description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required
    return schema
