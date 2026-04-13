"""
quartermaster-tools: Tool abstraction framework and chain-of-responsibility pattern.

Export all public APIs.
"""

from quartermaster_tools.base import AbstractLocalTool, AbstractTool
from quartermaster_tools.builtin.code import (
    EvalMathTool,
    JavaScriptExecutorTool,
    PythonExecutorTool,
    ShellExecutorTool,
)
from quartermaster_tools.builtin.data import (
    ConvertFormatTool,
    DataFilterTool,
    ParseCSVTool,
    ParseJSONTool,
    ParseXMLTool,
    ParseYAMLTool,
)
from quartermaster_tools.builtin.database import (
    SQLiteQueryTool,
    SQLiteSchemaTool,
    SQLiteWriteTool,
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
from quartermaster_tools.builtin.memory import (
    GetVariableTool,
    ListVariablesTool,
    SetVariableTool,
)
from quartermaster_tools.builtin.web_request import WebRequestTool
from quartermaster_tools.builtin.web_search import (
    DuckDuckGoSearchTool,
    JsonApiTool,
    WebScraperTool,
)
from quartermaster_tools.chain import Chain, Handler
from quartermaster_tools.decorator import FunctionTool, tool
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
    "CreateDirectoryTool",
    "DataFilterTool",
    "DeleteFileTool",
    "DuckDuckGoSearchTool",
    "EvalMathTool",
    "FileInfoTool",
    "FindFilesTool",
    "FunctionTool",
    "GetVariableTool",
    "GrepTool",
    "Handler",
    "JavaScriptExecutorTool",
    "JsonApiTool",
    "ListDirectoryTool",
    "ListVariablesTool",
    "MoveFileTool",
    "ParseCSVTool",
    "ParseJSONTool",
    "ParseXMLTool",
    "ParseYAMLTool",
    "PythonExecutorTool",
    "ReadFileTool",
    "SQLiteQueryTool",
    "SQLiteSchemaTool",
    "SQLiteWriteTool",
    "SetVariableTool",
    "ShellExecutorTool",
    "ToolDescriptor",
    "tool",
    "ToolParameter",
    "ToolParameterOption",
    "ToolRegistry",
    "ToolResult",
    "WebRequestTool",
    "WebScraperTool",
    "WriteFileTool",
    "get_default_registry",
    "register_tool",
]
