"""
qm-tools: Tool abstraction framework and chain-of-responsibility pattern.

Export all public APIs.
"""

from qm_tools.base import AbstractLocalTool, AbstractTool
from qm_tools.builtin.file_read import ReadFileTool
from qm_tools.builtin.file_write import WriteFileTool
from qm_tools.builtin.web_request import WebRequestTool
from qm_tools.chain import Chain, Handler
from qm_tools.registry import ToolRegistry, get_default_registry, register_tool
from qm_tools.types import (
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
    "ReadFileTool",
    "ToolDescriptor",
    "ToolParameter",
    "ToolParameterOption",
    "ToolRegistry",
    "ToolResult",
    "WebRequestTool",
    "WriteFileTool",
    "get_default_registry",
    "register_tool",
]
