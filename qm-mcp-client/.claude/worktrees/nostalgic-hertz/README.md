# qm-mcp-client

[![PyPI version](https://img.shields.io/pypi/v/qm-mcp-client.svg)](https://pypi.org/project/qm-mcp-client/)
[![Python Versions](https://img.shields.io/pypi/pyversions/qm-mcp-client.svg)](https://pypi.org/project/qm-mcp-client/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Status](https://github.com/quartermaster-ai/quartermaster/actions/workflows/test.yml/badge.svg)](https://github.com/quartermaster-ai/quartermaster/actions)

A lightweight, production-ready Python client for the **Model Context Protocol (MCP)** with full async/sync support, multiple transport layers, and zero framework dependencies.

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is a standardized protocol for connecting AI models to tools, data sources, and services. `qm-mcp-client` provides a clean, Pythonic API for discovering and invoking MCP-compliant servers from your Python applications.

## Features

- **Dual API**: Async (`async with`) and synchronous (`with`) interfaces
- **Multiple Transports**: Server-Sent Events (SSE) and Streamable HTTP
- **Type-Safe**: Full type hints, mypy strict mode compatible
- **Zero Framework Dependencies**: httpx is the only external dependency
- **Production-Ready**: Comprehensive error handling, timeout control, and retry logic
- **Extensible**: Custom transport implementations supported
- **Well-Tested**: Unit and integration test coverage with pytest

## Installation

Install from PyPI:

```bash
pip install qm-mcp-client
```

Or with development dependencies:

```bash
pip install qm-mcp-client[dev]
```

## Quick Start

### Async Usage

```python
import asyncio
from qm_mcp_client import McpClient

async def main():
    # Connect to an MCP server
    async with McpClient("http://localhost:8000/mcp") as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")
        
        # Call a tool
        result = await client.call_tool("weather", {"location": "San Francisco"})
        print(f"Result: {result}")

asyncio.run(main())
```

### Sync Usage

```python
from qm_mcp_client import McpClient

# Connect to an MCP server
with McpClient("http://localhost:8000/mcp") as client:
    # List available tools
    tools = client.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")
    
    # Call a tool
    result = client.call_tool("weather", {"location": "San Francisco"})
    print(f"Result: {result}")
```

### Server Information

```python
import asyncio
from qm_mcp_client import McpClient

async def main():
    async with McpClient("http://localhost:8000/mcp") as client:
        # Get server information
        info = await client.server_info()
        print(f"Server: {info.name} v{info.version}")
        print(f"Protocol version: {info.protocol_version}")

asyncio.run(main())
```

## API Reference

### McpClient

Main client class for interacting with MCP servers.

#### Initialization

```python
client = McpClient(
    url: str,
    transport: str = "sse",  # "sse" or "streamable"
    timeout: float = 30.0,
    max_retries: int = 3,
    auth_token: Optional[str] = None,
)
```

#### Methods

**Async Interface:**

```python
async with client:
    # List available tools
    tools: list[McpTool] = await client.list_tools()
    
    # Get server info
    info: McpServerInfo = await client.server_info()
    
    # Call a tool
    result: Any = await client.call_tool(name: str, arguments: dict[str, Any])
    
    # List available resources
    resources = await client.list_resources()
    
    # Read a resource
    content = await client.read_resource(uri: str)
```

**Sync Interface:**

Same methods available without `await`, using context manager.

### McpTool

Represents a tool exposed by the MCP server.

```python
@dataclass
class McpTool:
    name: str
    description: str
    parameters: list[ToolParameter]
    input_schema: dict[str, Any]
```

### McpServerInfo

Information about the connected MCP server.

```python
@dataclass
class McpServerInfo:
    name: str
    version: str
    protocol_version: str
    capabilities: dict[str, Any]
```

### ToolParameter

Metadata about a tool parameter.

```python
@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[list[str]] = None
```

## Advanced Usage

### Custom HTTP Headers

```python
client = McpClient(
    "http://localhost:8000/mcp",
    auth_token="your-api-key"
)
```

### Timeout Configuration

```python
client = McpClient(
    "http://localhost:8000/mcp",
    timeout=60.0  # 60 second timeout
)
```

### Retry Logic

```python
client = McpClient(
    "http://localhost:8000/mcp",
    max_retries=5  # Retry up to 5 times on transient failures
)
```

### Custom Transport

```python
from qm_mcp_client.transports import StreamableTransport

client = McpClient(
    "http://localhost:8000/mcp",
    transport="streamable"
)
```

## Error Handling

```python
import asyncio
from qm_mcp_client import McpClient
from qm_mcp_client.errors import (
    McpConnectionError,
    McpProtocolError,
    McpToolNotFoundError,
)

async def main():
    try:
        async with McpClient("http://localhost:8000/mcp") as client:
            result = await client.call_tool("unknown_tool", {})
    except McpToolNotFoundError as e:
        print(f"Tool not found: {e}")
    except McpProtocolError as e:
        print(f"Protocol error: {e}")
    except McpConnectionError as e:
        print(f"Connection failed: {e}")

asyncio.run(main())
```

## Development

### Setup

```bash
git clone https://github.com/quartermaster-ai/quartermaster.git
cd packages/qm-mcp-client
pip install -e ".[dev]"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=qm_mcp_client

# Specific test file
pytest tests/test_client.py -v
```

### Type Checking

```bash
mypy src/qm_mcp_client --strict
```

### Linting

```bash
ruff check src/qm_mcp_client
ruff format src/qm_mcp_client
```

## Protocol Compliance

This client fully implements the Model Context Protocol (MCP) v1.0 specification:

- JSON-RPC 2.0 message framing
- Server-Sent Events transport (recommended)
- Streamable HTTP transport (fallback)
- Full tool discovery and invocation
- Resource reading and listing
- Error handling and protocol validation

## Performance Notes

- **Connection pooling**: Uses httpx's built-in connection pooling (5 connections by default)
- **Streaming**: SSE transport supports server-sent streams for real-time responses
- **Retry strategy**: Exponential backoff with jitter for transient failures
- **Timeouts**: Configurable per request with safe defaults

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/quartermaster-ai/quartermaster/blob/main/CONTRIBUTING.md) for guidelines.

Areas for contribution:

- Additional transport implementations
- Performance optimizations
- Documentation improvements
- Integration tests with real MCP servers
- TypeScript/JavaScript client based on this implementation

## License

Apache License 2.0. See [LICENSE](./LICENSE) for details.

## Related Projects

- [Model Context Protocol](https://modelcontextprotocol.io/) - Official protocol specification
- [Quartermaster](https://github.com/quartermaster-ai/quartermaster) - AI agent platform that uses MCP
- [Anthropic MCP SDK](https://github.com/anthropics/mcp) - Official Python SDK (reference implementation)

## Support

- GitHub Issues: [Report a bug or request a feature](https://github.com/quartermaster-ai/quartermaster/issues)
- Documentation: [Read the full docs](https://quartermaster.dev/docs/mcp-client)
- Email: info@mindmade.io
