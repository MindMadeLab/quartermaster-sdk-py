"""
quartermaster-tools: Tool abstraction framework and chain-of-responsibility pattern.

Export all public APIs.
"""

from quartermaster_tools.base import AbstractLocalTool, AbstractTool
from quartermaster_tools.builtin.data import (
    ConvertFormatTool,
    DataFilterTool,
    ParseCSVTool,
    ParseJSONTool,
    ParseXMLTool,
    ParseYAMLTool,
)
from quartermaster_tools.builtin.code import (
    EvalMathTool,
    JavaScriptExecutorTool,
    PythonExecutorTool,
    ShellExecutorTool,
)
from quartermaster_tools.builtin.file_read import ReadFileTool
from quartermaster_tools.builtin.file_write import WriteFileTool
from quartermaster_tools.builtin.filesystem import (
    CopyFileTool,
    CreateDirectoryTool,
    DeleteFileTool,
    FileInfoTool,
    FindFilesTool,
    GrepTool,
    ListDirectoryTool,
    MoveFileTool,
)
from quartermaster_tools.builtin.web_request import WebRequestTool
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
    "ConvertFormatTool",
    "CopyFileTool",
    "DataFilterTool",
    "CreateDirectoryTool",
    "EvalMathTool",
    "DeleteFileTool",
    "FileInfoTool",
    "FindFilesTool",
    "GrepTool",
    "Handler",
    "JavaScriptExecutorTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "PythonExecutorTool",
    "ReadFileTool",
    "ShellExecutorTool",
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
