"""Standalone type definitions for MCP client.

This module contains dataclasses replacing Django-coupled types from the
Quartermaster backend, ensuring zero framework dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolParameterOption:
    """Represents an enumerated option for a tool parameter.

    Attributes:
        label: Human-readable label for the option.
        value: The actual value to use when this option is selected.
    """

    label: str
    value: str


@dataclass
class ToolParameter:
    """Metadata about a parameter for an MCP tool.

    Attributes:
        name: Parameter name as expected by the tool.
        type: JSON Schema type (string, number, integer, boolean, array, object).
        description: Human-readable description of the parameter.
        required: Whether this parameter is required. Defaults to False.
        default: Default value if not provided. Defaults to None.
        enum: List of allowed values if this is a restricted enum. Defaults to None.
        options: Pre-defined options with labels for UI display. Defaults to empty.
        min_value: Minimum value for numeric types.
        max_value: Maximum value for numeric types.
        min_length: Minimum length for string types.
        max_length: Maximum length for string types.
        pattern: Regex pattern for string validation.
    """

    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[list[str]] = None
    options: list[ToolParameterOption] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


@dataclass
class McpTool:
    """Represents a tool exposed by an MCP server.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: List of parameter metadata.
        input_schema: Full JSON Schema for tool input validation.
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    input_schema: dict[str, Any]


@dataclass
class McpServerInfo:
    """Information about the connected MCP server.

    Attributes:
        name: Server name.
        version: Server version string.
        protocol_version: MCP protocol version (e.g., "1.0").
        capabilities: Dict of supported capabilities.
    """

    name: str
    version: str
    protocol_version: str
    capabilities: dict[str, Any]
