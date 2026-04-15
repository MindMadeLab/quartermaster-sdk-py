"""Lightweight MCP (Model Context Protocol) client.

This module provides a clean, Pythonic API for interacting with MCP-compliant
servers. Supports both async and sync interfaces with multiple transport layers.
"""

from quartermaster_mcp_client.client import (
    McpClient,
    parse_json_schema_type,
    parse_sse_response,
    parse_tool_parameters,
)
from quartermaster_mcp_client.errors import (
    McpAuthenticationError,
    McpConnectionError,
    McpError,
    McpProtocolError,
    McpServerError,
    McpTimeoutError,
    McpToolNotFoundError,
)
from quartermaster_mcp_client.transports import (
    SSETransport,
    StreamableTransport,
    Transport,
    create_transport,
)
from quartermaster_mcp_client.types import (
    McpServerInfo,
    McpTool,
    ToolParameter,
    ToolParameterOption,
)

__all__ = [
    "McpClient",
    "McpServerInfo",
    "McpTool",
    "ToolParameter",
    "ToolParameterOption",
    "Transport",
    "SSETransport",
    "StreamableTransport",
    "create_transport",
    "McpError",
    "McpConnectionError",
    "McpProtocolError",
    "McpToolNotFoundError",
    "McpTimeoutError",
    "McpAuthenticationError",
    "McpServerError",
    "parse_sse_response",
    "parse_json_schema_type",
    "parse_tool_parameters",
]

__version__ = "0.3.0"
