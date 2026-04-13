"""
Tool type definitions: parameters, results, and descriptors.

Source: quartermaster/be/programs/services/__init__.py (ParameterContainer, ParameterOptionContainer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameterOption:
    """A single option for a tool parameter (e.g., dropdown choice).

    Source: quartermaster/be/programs/services/ParameterOptionContainer
    """

    label: str
    value: str


@dataclass
class ToolParameter:
    """A parameter definition for a tool.

    Defines the inputs that a tool accepts, including name, type,
    description, and optional validation rules.

    Source: quartermaster/be/programs/services/ParameterContainer
    """

    name: str
    description: str
    type: str  # string, number, boolean, array, object, etc.
    required: bool = False
    default: Any = None
    options: list[ToolParameterOption] = field(default_factory=list)
    validation: Any = None  # Callable or validation rule

    def to_json_schema(self) -> dict[str, Any]:
        """Convert this parameter to a JSON Schema property definition."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.options:
            schema["enum"] = [opt.value for opt in self.options]
        return schema


@dataclass
class ToolDescriptor:
    """Metadata describing a tool.

    Contains the tool's identity, documentation, version, and parameters.
    Used for registration, discovery, and client-side UI generation.

    Source: quartermaster/be/programs/internal_programs/ProgramDescriber
    """

    name: str
    short_description: str
    long_description: str
    version: str = "1.0.0"
    parameters: list[ToolParameter] = field(default_factory=list)
    is_local: bool = False  # Is this tool executable locally?

    def to_input_schema(self) -> dict[str, Any]:
        """Build a JSON Schema object describing this tool's input parameters."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def to_openai_tools(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool format.

        Returns a dict matching the OpenAI ``tools`` array element schema::

            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        This method does **not** require ``quartermaster-providers`` to be installed.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.short_description,
                "parameters": self.to_input_schema(),
            },
        }

    def to_anthropic_tools(self) -> dict[str, Any]:
        """Convert to Anthropic tool-use format.

        Returns a dict matching the Anthropic ``tools`` array element schema::

            {"name": ..., "description": ..., "input_schema": ...}

        This method does **not** require ``quartermaster-providers`` to be installed.
        """
        return {
            "name": self.name,
            "description": self.short_description,
            "input_schema": self.to_input_schema(),
        }

    def to_tool_definition(self) -> Any:
        """Convert to ``quartermaster_providers.ToolDefinition``.

        Requires the ``quartermaster-providers`` package (install ``quartermaster-tools[llm]``).

        Returns:
            A ``ToolDefinition`` TypedDict instance.

        Raises:
            ImportError: If ``quartermaster-providers`` is not installed.
        """
        try:
            from quartermaster_providers.types import ToolDefinition
        except ImportError:
            raise ImportError(
                "quartermaster-providers is required for to_tool_definition(). "
                "Install it with: pip install quartermaster-tools[llm]"
            )
        return ToolDefinition(
            name=self.name,
            description=self.short_description,
            input_schema=self.to_input_schema(),
        )


@dataclass
class ToolResult:
    """The result of executing a tool.

    Provides a clean return type for tool execution, distinguishing
    between success/error and optionally carrying data or error details.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allows ToolResult to be used in boolean context."""
        return self.success
