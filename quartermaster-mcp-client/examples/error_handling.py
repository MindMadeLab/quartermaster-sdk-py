#!/usr/bin/env python3
"""Error handling example: demonstrates how to catch MCP-specific errors.

Usage:
    python error_handling.py

This example shows handling of connection errors, tool-not-found errors,
and other failure modes.
"""

import asyncio

from quartermaster_mcp_client import McpClient
from quartermaster_mcp_client.errors import (
    McpConnectionError,
    McpProtocolError,
    McpTimeoutError,
    McpToolNotFoundError,
)


async def main() -> None:
    # --- Example 1: Connection error ---
    print("Example 1: Handling connection errors")
    try:
        async with McpClient(
            "http://localhost:9999/nonexistent",
            max_retries=1,
            timeout=5.0,
        ) as client:
            await client.list_tools()
    except McpConnectionError as e:
        print(f"  Caught connection error: {e}")
    print()

    # --- Example 2: Tool not found ---
    print("Example 2: Handling tool-not-found errors")
    try:
        async with McpClient("http://localhost:8000/mcp") as client:
            await client.call_tool("nonexistent_tool", {"arg": "value"})
    except McpToolNotFoundError as e:
        print(f"  Caught tool-not-found: {e}")
    except McpConnectionError as e:
        print(f"  (Server not running, skipping: {e})")
    print()

    # --- Example 3: Timeout ---
    print("Example 3: Handling timeouts")
    try:
        async with McpClient(
            "http://localhost:8000/mcp",
            timeout=0.001,  # Very short timeout
        ) as client:
            await client.list_tools()
    except McpTimeoutError as e:
        print(f"  Caught timeout: {e}")
    except McpConnectionError as e:
        print(f"  (Server not running, skipping: {e})")
    print()

    # --- Example 4: Protocol error ---
    print("Example 4: Handling protocol errors")
    try:
        async with McpClient("http://httpbin.org/html") as client:
            await client.list_tools()
    except McpProtocolError as e:
        print(f"  Caught protocol error: {e}")
    except McpConnectionError as e:
        print(f"  Caught connection error: {e}")
    print()

    # --- Example 5: Auth token usage ---
    print("Example 5: Using authentication")
    client = McpClient(
        "http://localhost:8000/mcp",
        auth_token="your-api-key-here",
        timeout=10.0,
        max_retries=2,
    )
    print(f"  Client configured with auth (connected: {client.is_connected})")
    print("  Use 'async with client:' to establish connection")


if __name__ == "__main__":
    asyncio.run(main())
