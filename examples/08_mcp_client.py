"""Connect to an MCP server, list tools, and call a tool.

Demonstrates the McpClient from qm-mcp-client: both the sync context
manager interface and the async interface.  Requires an MCP server to
be running (the example uses a placeholder URL).
"""

from __future__ import annotations

import asyncio

try:
    from qm_mcp_client.client import McpClient
    from qm_mcp_client.errors import McpConnectionError
except ImportError:
    raise SystemExit("Install qm-mcp-client first:  pip install -e qm-mcp-client")


MCP_SERVER_URL = "http://localhost:3000"


def demo_sync() -> None:
    """Demonstrate the synchronous client interface."""
    print("=== Sync MCP Client ===")
    try:
        with McpClient(MCP_SERVER_URL, transport="sse", timeout=5.0) as client:
            # Get server info
            info = client.server_info_sync()
            print(f"Server: {info.name} v{info.version}")
            print(f"Protocol: {info.protocol_version}")

            # List available tools
            tools = client.list_tools_sync()
            print(f"\nAvailable tools ({len(tools)}):")
            for tool in tools:
                params = ", ".join(p.name for p in tool.parameters)
                print(f"  {tool.name}: {tool.description}  params=({params})")

            # Call a tool (example -- adapt to your server)
            if tools:
                first_tool = tools[0]
                print(f"\nCalling '{first_tool.name}'...")
                result = client.call_tool_sync(first_tool.name, {})
                print(f"Result: {result}")
    except McpConnectionError as e:
        print(f"Could not connect to {MCP_SERVER_URL}: {e}")
        print("Start an MCP server and try again.")


async def demo_async() -> None:
    """Demonstrate the asynchronous client interface."""
    print("\n=== Async MCP Client ===")
    try:
        async with McpClient(MCP_SERVER_URL, transport="sse", timeout=5.0) as client:
            info = await client.server_info()
            print(f"Server: {info.name} v{info.version}")

            tools = await client.list_tools()
            print(f"Tools: {[t.name for t in tools]}")

            resources = await client.list_resources()
            print(f"Resources: {len(resources)}")
    except McpConnectionError as e:
        print(f"Could not connect: {e}")


def main() -> None:
    # Show client creation and validation
    print("McpClient supports 'sse' and 'streamable' transports.")
    print(f"Target server: {MCP_SERVER_URL}\n")

    demo_sync()
    asyncio.run(demo_async())


if __name__ == "__main__":
    main()
