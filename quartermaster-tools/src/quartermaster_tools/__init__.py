"""
quartermaster-tools: Tool abstraction framework and chain-of-responsibility pattern.

Export all public APIs.
"""

from quartermaster_tools.base import AbstractLocalTool, AbstractTool
from quartermaster_tools.builtin.code import (
    eval_math,
    javascript_executor,
    python_executor,
    shell_executor,
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
from quartermaster_tools.builtin.web_request import web_request
from quartermaster_tools.builtin.web_search import (
    duckduckgo_search,
    json_api,
    web_scraper,
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

__version__ = "0.6.2"
__all__ = [
    "AbstractLocalTool",
    "AbstractTool",
    "auto_decorate",
    "Chain",
    "convert_format",
    "copy_file",
    "create_directory",
    "data_filter",
    "delete_file",
    "duckduckgo_search",
    "eval_math",
    "file_info",
    "find_files",
    "FunctionTool",
    "get_variable",
    "grep",
    "is_quartermaster_tool",
    "Handler",
    "javascript_executor",
    "json_api",
    "list_directory",
    "list_variables",
    "move_file",
    "parse_csv",
    "parse_json",
    "parse_xml",
    "parse_yaml",
    "python_executor",
    "read_file",
    "sqlite_query",
    "sqlite_schema",
    "sqlite_write",
    "set_variable",
    "shell_executor",
    "ToolDescriptor",
    "tool",
    "ToolParameter",
    "ToolParameterOption",
    "ToolRegistry",
    "ToolResult",
    "web_request",
    "web_scraper",
    "write_file",
    "get_default_registry",
    "register_tool",
]
