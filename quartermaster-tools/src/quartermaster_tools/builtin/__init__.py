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
"""

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
from quartermaster_tools.builtin.web_search import (
    DuckDuckGoSearchTool,
    JsonApiTool,
    WebScraperTool,
)

__all__ = [
    "ConvertFormatTool",
    "CopyFileTool",
    "DuckDuckGoSearchTool",
    "DataFilterTool",
    "EvalMathTool",
    "CreateDirectoryTool",
    "DeleteFileTool",
    "FileInfoTool",
    "FindFilesTool",
    "GrepTool",
    "JavaScriptExecutorTool",
    "JsonApiTool",
    "ListDirectoryTool",
    "MoveFileTool",
    "ParseCSVTool",
    "ParseJSONTool",
    "ParseXMLTool",
    "ParseYAMLTool",
    "PythonExecutorTool",
    "ReadFileTool",
    "ShellExecutorTool",
    "WebRequestTool",
    "WebScraperTool",
    "WriteFileTool",
]
