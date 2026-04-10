# qm-mcp-client

[![PyPI version](https://img.shields.io/pypi/v/qm-mcp-client.svg)](https://pypi.org/project/qm-mcp-client/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

Lightweight async/sync Python client for the Model Context Protocol (MCP), implementing JSON-RPC 2.0 over SSE or Streamable HTTP transports.

## Features

- **Dual API**: Full async (`async with`) and synchronous (`with`) interfaces
- **Two Transports**: Server-Sent Events (SSE) and Streamable HTTP
- **Auto-Retry**: Exponential backoff with jitter on transient failures
- **Type-Safe**: Dataclass responses with complete type hints
- **Zero Framework Dependencies**: Only requires `httpx`
- **Auth Support**: Bearer token and custom header authentication
- **Rich Error Hierarchy**: Typed exceptions for connection, protocol, timeout, and auth errors

## Installation

```bash
pip install qm-mcp-client
```

## Quick Start

### Async Usage

```python
import asyncio
from qm_mcp_client import McpClient

async def main():
    async with McpClient("http://localhost:8000/mcp") as client:
        # Get server info
        info = await client.server_info()
        print(f"Server: {info.name} v{info.version}")

        # Discover available tools
        tools = await client.list_tools()
        for tool in tools:
            print(f"  {tool.name}: {tool.description}")

        # Call a tool
        result = await client.call_tool("weather", {"location": "San Francisco"})
        print(f"Result: {result}")

asyncio.run(main())
```

### Sync Usage

Sync methods use the `_sync` suffix and work with a standard `with` block:

```python
from qm_mcp_client import McpClient

with McpClient("http://localhost:8000/mcp") as client:
    tools = client.list_tools_sync()
    print(f"Available tools: {[t.name for t in tools]}")

    result = client.call_tool_sync("weather", {"location": "London"})
    print(f"Result: {result}")
```

### Error Handling

```python
import asyncio
from qm_mcp_client import McpClient
from qm_mcp_client.errors import (
    McpConnectionError,
    McpTimeoutError,
    McpToolNotFoundError,
    McpAuthenticationError,
    McpServerError,
)

async def main():
    try:
        async with McpClient("http://localhost:8000/mcp", timeout=10.0) as client:
            result = await client.call_tool("unknown_tool", {})
    except McpToolNotFoundError as e:
        print(f"Tool not found: {e}")
    except McpTimeoutError as e:
        print(f"Request timed out: {e}")
    except McpConnectionError as e:
        print(f"Connection failed: {e}")
    except McpAuthenticationError as e:
        print(f"Auth failed: {e}")
    except McpServerError as e:
        print(f"Server error (code {e.code}): {e}")

asyncio.run(main())
```

## API Reference

### McpClient

The main client class. Supports both async and sync context managers.

```python
client = McpClient(
    url="http://localhost:8000/mcp",  # MCP server URL (http/https required)
    transport="sse",       # "sse" (default) or "streamable"
    timeout=30.0,          # Request timeout in seconds
    max_retries=3,         # Retries on transient failures
    auth_token="sk-...",   # Optional Bearer token
    headers={"X-Custom": "value"},  # Additional HTTP headers
)
```

#### Async Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `await client.server_info()` | `McpServerInfo` | Server name, version, protocol version, capabilities |
| `await client.list_tools()` | `list[McpTool]` | All tools with parameter metadata |
| `await client.call_tool(name, arguments)` | `Any` | Invoke a tool and return its result |
| `await client.list_resources()` | `list[dict]` | List available resources |
| `await client.read_resource(uri)` | `str` | Read resource content by URI |

#### Sync Methods

Every async method has a sync counterpart with `_sync` suffix:

| Method | Returns |
|--------|---------|
| `client.server_info_sync()` | `McpServerInfo` |
| `client.list_tools_sync()` | `list[McpTool]` |
| `client.call_tool_sync(name, arguments)` | `Any` |
| `client.list_resources_sync()` | `list[dict]` |
| `client.read_resource_sync(uri)` | `str` |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `client.is_connected` | `bool` | Whether the client has an active transport |

### Data Types

**McpTool** -- a tool exposed by the server:

```python
@dataclass
class McpTool:
    name: str                        # Tool identifier
    description: str                 # Human-readable description
    parameters: list[ToolParameter]  # Parameter metadata
    input_schema: dict[str, Any]     # Full JSON Schema
```

**McpServerInfo** -- server metadata returned by `server_info()`:

```python
@dataclass
class McpServerInfo:
    name: str                     # Server name
    version: str                  # Server version
    protocol_version: str         # MCP protocol version
    capabilities: dict[str, Any]  # Supported capabilities
```

**ToolParameter** -- metadata about a single tool parameter:

```python
@dataclass
class ToolParameter:
    name: str
    type: str             # JSON Schema type (string, number, integer, boolean, etc.)
    description: str
    required: bool = False
    default: Any = None
    enum: list[str] | None = None
    options: list[ToolParameterOption] = []
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
```

### Transports

**SSE (Server-Sent Events)** -- the default and recommended transport. Sends JSON-RPC as HTTP POST and reads the response as an SSE stream.

**Streamable HTTP** -- standard HTTP POST with JSON responses. Use when SSE is unavailable.

```python
# SSE transport (default)
client = McpClient("http://localhost:8000/mcp", transport="sse")

# Streamable HTTP transport
client = McpClient("http://localhost:8000/mcp", transport="streamable")
```

### Errors

All exceptions inherit from `McpError`:

| Exception | When |
|-----------|------|
| `McpConnectionError` | Connection to server fails |
| `McpTimeoutError` | Request exceeds timeout |
| `McpProtocolError` | Malformed JSON-RPC or SSE response |
| `McpServerError` | Server returns a JSON-RPC error (has `.code` attribute) |
| `McpToolNotFoundError` | Requested tool does not exist |
| `McpAuthenticationError` | Authentication fails |

Errors can be imported from `qm_mcp_client.errors` or directly from `qm_mcp_client`.

## Configuration

### Authentication

```python
# Bearer token
client = McpClient(
    "https://mcp.example.com/api",
    auth_token="your-api-key",
)

# Custom headers
client = McpClient(
    "https://mcp.example.com/api",
    headers={"X-API-Key": "your-key", "X-Org-Id": "org-123"},
)
```

### Retry and Timeout

The client retries on `McpConnectionError` and `McpTimeoutError` with exponential backoff. Non-retriable errors (`McpProtocolError`, `McpServerError`, `McpAuthenticationError`) are raised immediately.

```python
client = McpClient(
    "http://localhost:8000/mcp",
    timeout=60.0,      # 60-second timeout per request
    max_retries=5,     # Up to 5 attempts on transient failures
)
```

## Contributing

Contributions welcome. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
