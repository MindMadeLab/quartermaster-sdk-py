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
    convert_format,
    data_filter,
    parse_csv,
    parse_json,
    parse_xml,
    parse_yaml,
)
from quartermaster_tools.builtin.database import (
    SQLiteQueryTool,
    SQLiteSchemaTool,
    SQLiteWriteTool,
)
from quartermaster_tools.builtin.file_read import ReadFileTool, read_file
from quartermaster_tools.builtin.file_write import WriteFileTool, write_file
from quartermaster_tools.builtin.filesystem import (
    CopyFileTool,
    CreateDirectoryTool,
    DeleteFileTool,
    FileInfoTool,
    FindFilesTool,
    GrepTool,
    ListDirectoryTool,
    MoveFileTool,
    copy_file,
    create_directory,
    delete_file,
    file_info,
    find_files,
    grep,
    list_directory,
    move_file,
)
from quartermaster_tools.builtin.memory import (
    GetVariableTool,
    ListVariablesTool,
    SetVariableTool,
)
from quartermaster_tools.builtin.web_request import WebRequestTool, web_request
from quartermaster_tools.builtin.web_search import (
    DuckDuckGoSearchTool,
    JsonApiTool,
    WebScraperTool,
)
from quartermaster_tools.chain import Chain, Handler
from quartermaster_tools.decorator import (
    FunctionTool,
    auto_decorate,
    is_quartermaster_tool,
    tool,
)
from quartermaster_tools.registry import ToolRegistry, get_default_registry, register_tool
from quartermaster_tools.types import (
    ToolDescriptor,
    ToolParameter,
    ToolParameterOption,
    ToolResult,
)

__version__ = "0.5.1"
__all__ = [
    "AbstractLocalTool",
    "AbstractTool",
    "auto_decorate",
    "Chain",
    "convert_format",
    "ConvertFormatTool",
    "copy_file",
    "CopyFileTool",
    "create_directory",
    "CreateDirectoryTool",
    "data_filter",
    "DataFilterTool",
    "delete_file",
    "DeleteFileTool",
    "DuckDuckGoSearchTool",
    "EvalMathTool",
    "file_info",
    "FileInfoTool",
    "find_files",
    "FindFilesTool",
    "FunctionTool",
    "GetVariableTool",
    "grep",
    "is_quartermaster_tool",
    "GrepTool",
    "Handler",
    "JavaScriptExecutorTool",
    "JsonApiTool",
    "list_directory",
    "ListDirectoryTool",
    "ListVariablesTool",
    "move_file",
    "MoveFileTool",
    "parse_csv",
    "ParseCSVTool",
    "parse_json",
    "ParseJSONTool",
    "parse_xml",
    "ParseXMLTool",
    "parse_yaml",
    "ParseYAMLTool",
    "PythonExecutorTool",
    "read_file",
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
    "web_request",
    "WebRequestTool",
    "WebScraperTool",
    "write_file",
    "WriteFileTool",
    "get_default_registry",
    "register_tool",
]
