"""
Built-in tools for quartermaster-tools.

Provides ready-to-use tool implementations:
- read_file: Read file content with path validation and size limits
- write_file: Write content to file with size limits
- web_request: HTTP requests (GET, POST, PUT, DELETE, PATCH; requires httpx)
- duckduckgo_search: Zero-config web search via DuckDuckGo HTML
- web_scraper: Fetch and convert web pages to text/markdown/html
- json_api: JSON API caller with optional JMESPath filtering
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

from quartermaster_tools.builtin.a2a import (
    a2a_check_status,
    a2a_collect_result,
    a2a_discover,
    a2a_register,
    a2a_send_task,
)
from quartermaster_tools.builtin.agents import (
    SessionManager,
    SessionStatus,
    add_agent_finish_hook,
    cancel_agent_session,
    collect_agent_results,
    create_agent_session,
    get_agent_session_status,
    inject_agent_message,
    list_agent_sessions,
    notify_parent,
    spawn_agent,
    start_agent_session,
    wait_agent_session,
)
from quartermaster_tools.builtin.browser import (
    BrowserSessionManager,
    browser_click,
    browser_eval,
    browser_extract,
    browser_navigate,
    browser_screenshot,
    browser_type,
    browser_wait,
)
from quartermaster_tools.builtin.code import (
    eval_math,
    javascript_executor,
    python_executor,
    shell_executor,
)
from quartermaster_tools.builtin.compliance import (
    audit_log,
    compliance_checklist,
    read_audit_log,
    risk_classifier,
)
from quartermaster_tools.builtin.data import (
    convert_format,
    data_filter,
    parse_csv,
    parse_json,
    parse_xml,
    parse_yaml,
)
from quartermaster_tools.builtin.database import (
    sqlite_query,
    sqlite_schema,
    sqlite_write,
)
from quartermaster_tools.builtin.email import (
    read_email,
    search_email,
    send_email,
)
from quartermaster_tools.builtin.file_read import read_file
from quartermaster_tools.builtin.file_write import write_file
from quartermaster_tools.builtin.filesystem import (
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
    get_variable,
    list_variables,
    set_variable,
)
from quartermaster_tools.builtin.messaging import (
    discord_message,
    slack_message,
    slack_read,
    webhook_notify,
)
from quartermaster_tools.builtin.observability import (
    cost_tracker,
    log,
    metric,
    performance_profile,
    trace,
)
from quartermaster_tools.builtin.privacy import (
    detect_pii_tool,
    redact_pii,
    scan_file_pii,
)
from quartermaster_tools.builtin.vector import (
    document_index,
    embed_text,
    hybrid_search,
    vector_search,
    vector_store,
)
from quartermaster_tools.builtin.web_request import web_request
from quartermaster_tools.builtin.web_search import (
    brave_search,
    duckduckgo_search,
    google_search,
    json_api,
    web_scraper,
)

__all__ = [
    "BrowserSessionManager",
    "SessionManager",
    "SessionStatus",
    "a2a_check_status",
    "a2a_collect_result",
    "a2a_discover",
    "a2a_register",
    "a2a_send_task",
    "add_agent_finish_hook",
    "audit_log",
    "brave_search",
    "browser_click",
    "browser_eval",
    "browser_extract",
    "browser_navigate",
    "browser_screenshot",
    "browser_type",
    "browser_wait",
    "cancel_agent_session",
    "collect_agent_results",
    "compliance_checklist",
    "convert_format",
    "copy_file",
    "cost_tracker",
    "create_agent_session",
    "create_directory",
    "data_filter",
    "delete_file",
    "detect_pii_tool",
    "discord_message",
    "document_index",
    "duckduckgo_search",
    "embed_text",
    "eval_math",
    "file_info",
    "find_files",
    "get_agent_session_status",
    "get_variable",
    "google_search",
    "grep",
    "hybrid_search",
    "inject_agent_message",
    "javascript_executor",
    "json_api",
    "list_agent_sessions",
    "list_directory",
    "list_variables",
    "log",
    "metric",
    "move_file",
    "notify_parent",
    "parse_csv",
    "parse_json",
    "parse_xml",
    "parse_yaml",
    "performance_profile",
    "python_executor",
    "read_audit_log",
    "read_email",
    "read_file",
    "redact_pii",
    "risk_classifier",
    "scan_file_pii",
    "search_email",
    "send_email",
    "set_variable",
    "shell_executor",
    "slack_message",
    "slack_read",
    "spawn_agent",
    "sqlite_query",
    "sqlite_schema",
    "sqlite_write",
    "start_agent_session",
    "trace",
    "vector_search",
    "vector_store",
    "wait_agent_session",
    "web_request",
    "web_scraper",
    "webhook_notify",
    "write_file",
]
