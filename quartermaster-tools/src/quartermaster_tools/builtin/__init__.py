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
- A2A tools: Agent discovery, task send/status/collect, agent card registration
- Browser tools: Navigate, Click, Type, Extract, Screenshot, Wait, Eval (requires Playwright)
- Privacy tools: PII detection, redaction, and file scanning
- Compliance tools: EU AI Act risk classification, audit logging, checklists
- Agent tools: Parallel agent session creation, execution, monitoring, and result collection
"""

from quartermaster_tools.builtin.agents import (
    AddFinishHookTool,
    CancelSessionTool,
    CollectResultsTool,
    CreateSessionTool,
    GetSessionStatusTool,
    InjectMessageTool,
    ListSessionsTool,
    SessionManager,
    SessionStatus,
    StartSessionTool,
    WaitSessionTool,
)
from quartermaster_tools.builtin.compliance import (
    AuditLogTool,
    ComplianceChecklistTool,
    ReadAuditLogTool,
    RiskClassifierTool,
)
from quartermaster_tools.builtin.privacy import (
    DetectPIITool,
    RedactPIITool,
    ScanFilePIITool,
)
from quartermaster_tools.builtin.browser import (
    BrowserClickTool,
    BrowserEvalTool,
    BrowserExtractTool,
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserSessionManager,
    BrowserTypeTool,
    BrowserWaitTool,
)
from quartermaster_tools.builtin.a2a import (
    A2ACheckStatusTool,
    A2ACollectResultTool,
    A2ADiscoverTool,
    A2ARegisterTool,
    A2ASendTaskTool,
)
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
    convert_format,
    data_filter,
    parse_csv,
    parse_json,
    parse_xml,
    parse_yaml,
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
from quartermaster_tools.builtin.messaging import (
    DiscordMessageTool,
    SlackMessageTool,
    SlackReadTool,
    WebhookNotifyTool,
)
from quartermaster_tools.builtin.web_request import WebRequestTool, web_request
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
    "AddFinishHookTool",
    "AuditLogTool",
    "ComplianceChecklistTool",
    "DetectPIITool",
    "A2ACheckStatusTool",
    "A2ACollectResultTool",
    "A2ADiscoverTool",
    "A2ARegisterTool",
    "A2ASendTaskTool",
    "CancelSessionTool",
    "CollectResultsTool",
    "CreateSessionTool",
    "BrowserClickTool",
    "BrowserEvalTool",
    "BrowserExtractTool",
    "BrowserNavigateTool",
    "BrowserScreenshotTool",
    "BrowserSessionManager",
    "BrowserTypeTool",
    "BrowserWaitTool",
    "BraveSearchTool",
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
    "DiscordMessageTool",
    "DocumentIndexTool",
    "DuckDuckGoSearchTool",
    "EmbedTextTool",
    "EvalMathTool",
    "file_info",
    "FileInfoTool",
    "find_files",
    "FindFilesTool",
    "GetSessionStatusTool",
    "GetVariableTool",
    "GoogleSearchTool",
    "grep",
    "GrepTool",
    "HybridSearchTool",
    "InjectMessageTool",
    "JavaScriptExecutorTool",
    "JsonApiTool",
    "list_directory",
    "ListDirectoryTool",
    "ListSessionsTool",
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
    "ReadAuditLogTool",
    "ReadEmailTool",
    "read_file",
    "ReadFileTool",
    "RedactPIITool",
    "RiskClassifierTool",
    "SessionManager",
    "SessionStatus",
    "SQLiteQueryTool",
    "SQLiteSchemaTool",
    "SQLiteWriteTool",
    "ScanFilePIITool",
    "SearchEmailTool",
    "SendEmailTool",
    "SetVariableTool",
    "StartSessionTool",
    "ShellExecutorTool",
    "SlackMessageTool",
    "SlackReadTool",
    "WaitSessionTool",
    "VectorSearchTool",
    "VectorStoreTool",
    "web_request",
    "WebRequestTool",
    "WebScraperTool",
    "WebhookNotifyTool",
    "write_file",
    "WriteFileTool",
]
