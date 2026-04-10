"""Shared test fixtures for qm-mcp-client tests."""

from __future__ import annotations

from typing import Any

import pytest


# --- Sample data fixtures ---


@pytest.fixture
def sample_server_info_response() -> dict[str, Any]:
    """JSON-RPC response for initialize (server info)."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "test-server",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False},
            },
        },
    }


@pytest.fixture
def sample_tools_response() -> dict[str, Any]:
    """JSON-RPC response for tools/list."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": "weather",
                    "description": "Get weather for a location",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
                {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            },
                        },
                        "required": ["expression"],
                    },
                },
            ]
        },
    }


@pytest.fixture
def sample_tool_call_response() -> dict[str, Any]:
    """JSON-RPC response for tools/call."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "Sunny, 22°C in San Francisco",
                }
            ]
        },
    }


@pytest.fixture
def sample_resources_response() -> dict[str, Any]:
    """JSON-RPC response for resources/list."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "resources": [
                {
                    "uri": "file:///data/config.json",
                    "name": "Configuration",
                    "mimeType": "application/json",
                },
                {
                    "uri": "file:///data/readme.md",
                    "name": "README",
                    "mimeType": "text/markdown",
                },
            ]
        },
    }


@pytest.fixture
def sample_resource_read_response() -> dict[str, Any]:
    """JSON-RPC response for resources/read."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "contents": [
                {
                    "uri": "file:///data/config.json",
                    "mimeType": "application/json",
                    "text": '{"key": "value"}',
                }
            ]
        },
    }


@pytest.fixture
def sample_input_schema() -> dict[str, Any]:
    """A realistic JSON Schema for tool input."""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "minLength": 1,
                "maxLength": 500,
            },
            "limit": {
                "type": "integer",
                "description": "Max results",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
            },
            "format": {
                "type": "string",
                "description": "Output format",
                "enum": ["json", "csv", "xml"],
            },
            "verbose": {
                "type": "boolean",
                "description": "Verbose output",
                "default": False,
            },
            "tags": {
                "type": "array",
                "description": "Filter by tags",
            },
            "options": {
                "type": "object",
                "description": "Extra options",
            },
        },
        "required": ["query"],
    }


@pytest.fixture
def jsonrpc_error_method_not_found() -> dict[str, Any]:
    """JSON-RPC error for method not found."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32601,
            "message": "Tool 'nonexistent' not found",
        },
    }


@pytest.fixture
def jsonrpc_error_internal() -> dict[str, Any]:
    """JSON-RPC internal error response."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32603,
            "message": "Internal server error",
        },
    }
