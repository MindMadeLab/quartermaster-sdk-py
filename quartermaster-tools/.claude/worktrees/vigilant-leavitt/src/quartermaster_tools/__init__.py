"""
quartermaster-tools: Tool abstraction framework and chain-of-responsibility pattern.

Export all public APIs.
"""

from quartermaster_tools.base import AbstractLocalTool, AbstractTool
from quartermaster_tools.chain import Chain, Handler
from quartermaster_tools.registry import ToolRegistry, get_default_registry, register_tool
from quartermaster_tools.types import (
    ToolDescriptor,
    ToolParameter,
    ToolParameterOption,
    ToolResult,
)

__version__ = "0.1.0"
__all__ = [
    "AbstractLocalTool",
    "AbstractTool",
    "Chain",
    "Handler",
    "ToolDescriptor",
    "ToolParameter",
    "ToolParameterOption",
    "ToolRegistry",
    "ToolResult",
    "get_default_registry",
    "register_tool",
]
