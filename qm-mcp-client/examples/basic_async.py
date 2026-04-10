#!/usr/bin/env python3
"""Basic async example: connect to an MCP server, list tools, and call one.

Usage:
    python basic_async.py

Requires a running MCP server at http://localhost:8000/mcp.
"""

import asyncio

from qm_mcp_client import McpClient


async def main() -> None:
    # Connect to an MCP server using SSE transport (default)
    async with McpClient("http://localhost:8000/mcp") as client:
        # 1. Get server information
        info = await client.server_info()
        print(f"Connected to: {info.name} v{info.version}")
        print(f"Protocol version: {info.protocol_version}")
        print(f"Capabilities: {info.capabilities}")
        print()

        # 2. List available tools
        tools = await client.list_tools()
        print(f"Available tools ({len(tools)}):")
        for tool in tools:
            params = ", ".join(
                f"{p.name}: {p.type}{'*' if p.required else ''}"
                for p in tool.parameters
            )
            print(f"  - {tool.name}({params})")
            print(f"    {tool.description}")
        print()

        # 3. Call a tool (replace with an actual tool name from your server)
        if tools:
            tool = tools[0]
            print(f"Calling tool: {tool.name}")
            result = await client.call_tool(tool.name, {})
            print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
