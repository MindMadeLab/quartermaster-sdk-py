#!/usr/bin/env python3
"""Basic sync example: connect to an MCP server using the synchronous API.

Usage:
    python basic_sync.py

Requires a running MCP server at http://localhost:8000/mcp.
"""

from quartermaster_mcp_client import McpClient


def main() -> None:
    # Connect using the synchronous context manager
    with McpClient("http://localhost:8000/mcp") as client:
        # 1. Get server info
        info = client.server_info_sync()
        print(f"Connected to: {info.name} v{info.version}")
        print()

        # 2. List tools
        tools = client.list_tools_sync()
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # 3. Call a tool
        if tools:
            tool = tools[0]
            print(f"Calling tool: {tool.name}")
            result = client.call_tool_sync(tool.name, {})
            print(f"Result: {result}")


if __name__ == "__main__":
    main()
