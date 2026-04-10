"""
Built-in tools for quartermaster-tools.

Provides ready-to-use tool implementations:
- ReadFileTool: Read file content with path validation and size limits
- WriteFileTool: Write content to file with size limits
- WebRequestTool: HTTP requests (GET, POST, PUT, DELETE, PATCH; requires httpx)
- DuckDuckGoSearchTool: Zero-config web search via DuckDuckGo HTML
- WebScraperTool: Fetch and convert web pages to text/markdown/html
- JsonApiTool: JSON API caller with optional JMESPath filtering
- Filesystem tools: list, find, grep, info, move, delete, copy, mkdir
- Code execution tools: Python, Shell, JavaScript, Math evaluation
- Data tools: CSV, JSON, YAML, XML parsing, format conversion, filtering
- Memory tools: Set, Get, List in-memory variables
- Database tools: SQLite query, write, and schema introspection
"""

from quartermaster_tools.builtin.database import (
    SQLiteQueryTool,
    SQLiteSchemaTool,
    SQLiteWriteTool,
)
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

__all__ = [
    "ConvertFormatTool",
    "CopyFileTool",
    "CreateDirectoryTool",
    "DataFilterTool",
    "DeleteFileTool",
    "DuckDuckGoSearchTool",
    "EvalMathTool",
    "FileInfoTool",
    "FindFilesTool",
    "GetVariableTool",
    "GrepTool",
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
    "WebRequestTool",
    "WebScraperTool",
    "WriteFileTool",
]
