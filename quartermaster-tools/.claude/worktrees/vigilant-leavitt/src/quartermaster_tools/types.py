"""
Tool type definitions: parameters, results, and descriptors.

Source: quartermaster/be/programs/services/__init__.py (ParameterContainer, ParameterOptionContainer)
"""

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
    version: str
    parameters: list[ToolParameter] = field(default_factory=list)
    is_local: bool = False  # Is this tool executable locally?


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
