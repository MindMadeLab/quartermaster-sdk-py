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
- Vector/RAG tools: Embed, Store, Search, Index, Hybrid Search
- Email tools: Send, Read, Search via SMTP/IMAP
- Messaging tools: Slack, Discord, Webhooks
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
from quartermaster_tools.builtin.email import (
    ReadEmailTool,
    SearchEmailTool,
    SendEmailTool,
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
from quartermaster_tools.builtin.messaging import (
    DiscordMessageTool,
    SlackMessageTool,
    SlackReadTool,
    WebhookNotifyTool,
)
from quartermaster_tools.builtin.web_request import WebRequestTool
from quartermaster_tools.builtin.vector import (
    DocumentIndexTool,
    EmbedTextTool,
    HybridSearchTool,
    VectorSearchTool,
    VectorStoreTool,
)
from quartermaster_tools.builtin.web_search import (
    BraveSearchTool,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    JsonApiTool,
    WebScraperTool,
)

__all__ = [
    "BraveSearchTool",
    "ConvertFormatTool",
    "CopyFileTool",
    "CreateDirectoryTool",
    "DataFilterTool",
    "DeleteFileTool",
    "DiscordMessageTool",
    "DocumentIndexTool",
    "DuckDuckGoSearchTool",
    "EmbedTextTool",
    "EvalMathTool",
    "FileInfoTool",
    "FindFilesTool",
    "GetVariableTool",
    "GoogleSearchTool",
    "GrepTool",
    "HybridSearchTool",
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
    "ReadEmailTool",
    "ReadFileTool",
    "SQLiteQueryTool",
    "SQLiteSchemaTool",
    "SQLiteWriteTool",
    "SearchEmailTool",
    "SendEmailTool",
    "SetVariableTool",
    "ShellExecutorTool",
    "SlackMessageTool",
    "SlackReadTool",
    "VectorSearchTool",
    "VectorStoreTool",
    "WebRequestTool",
    "WebScraperTool",
    "WebhookNotifyTool",
    "WriteFileTool",
]
