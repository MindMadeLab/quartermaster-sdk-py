"""Example 19 -- MCP protocol client integration.

Demonstrates how to connect to MCP (Model Context Protocol) servers,
discover available tools, and integrate them with Quartermaster graphs.

MCP is the standard protocol for LLM tool integration. Quartermaster's
MCP client connects to any MCP-compatible server.

Usage:
    # Start an MCP server first, then:
    uv run examples/19_mcp_client.py
"""

from __future__ import annotations

import asyncio
import sys

# ── Part 0: Import guard ────────────────────────────────────────────────────
# The MCP client is an optional dependency. If it is not installed we
# print a helpful message and exit instead of crashing with a traceback.

try:
    from quartermaster_mcp_client import (
        McpClient,
        McpTool,
        McpServerInfo,
        ToolParameter,
    )
    from quartermaster_mcp_client.errors import (
        McpConnectionError,
        McpTimeoutError,
        McpToolNotFoundError,
    )
except ImportError:
    print(
        "quartermaster-mcp-client is not installed.\n"
        "Install it with:\n"
        "    pip install quartermaster-mcp-client\n"
        "or:\n"
        "    uv pip install quartermaster-mcp-client"
    )
    sys.exit(1)


# ── Part 1: MCP client basics ──────────────────────────────────────────────
# The McpClient connects to any MCP-compatible server over SSE or
# Streamable HTTP.  It can discover tools, call them, and list
# resources.  Both async and sync interfaces are available.

MCP_SERVER_URL = "http://localhost:8000/mcp"


async def demo_mcp_client() -> list[McpTool]:
    """Connect to an MCP server, print its info and tools.

    Returns the discovered tools list (empty if the server is unreachable).
    """
    print("=" * 60)
    print("PART 1 -- Connecting to an MCP server")
    print("=" * 60)
    print()

    # McpClient accepts several options:
    #   transport   -- "sse" (default) or "streamable"
    #   timeout     -- per-request timeout in seconds
    #   max_retries -- retries with exponential backoff
    #   auth_token  -- Bearer token for authenticated servers
    #   headers     -- any extra HTTP headers
    client = McpClient(
        MCP_SERVER_URL,
        transport="sse",
        timeout=10.0,
        max_retries=2,
    )

    try:
        async with client:
            # ---- Server info ------------------------------------------------
            info: McpServerInfo = await client.server_info()
            print(f"Connected to:      {info.name} v{info.version}")
            print(f"Protocol version:  {info.protocol_version}")
            print(f"Capabilities:      {info.capabilities}")
            print()

            # ---- Discover tools ---------------------------------------------
            tools: list[McpTool] = await client.list_tools()
            print(f"Available tools ({len(tools)}):")
            for tool in tools:
                params = ", ".join(
                    f"{p.name}: {p.type}{'*' if p.required else ''}"
                    for p in tool.parameters
                )
                print(f"  {tool.name}({params})")
                print(f"    {tool.description}")
            print()

            # ---- Call a tool ------------------------------------------------
            if tools:
                first = tools[0]
                print(f"Calling first tool: {first.name}")
                # Build minimal arguments from required parameters
                args = {
                    p.name: p.default or f"<{p.name}>"
                    for p in first.parameters
                    if p.required
                }
                try:
                    result = await client.call_tool(first.name, args)
                    print(f"Result: {result}")
                except McpToolNotFoundError as exc:
                    print(f"Tool not found: {exc}")
            else:
                print("No tools found on the server.")
            print()

            # ---- List resources (optional MCP feature) ----------------------
            try:
                resources = await client.list_resources()
                if resources:
                    print(f"Resources ({len(resources)}):")
                    for r in resources:
                        print(f"  {r.get('uri', '?')} -- {r.get('name', '')}")
                    print()
            except Exception:
                # Some servers do not implement resources; that is fine.
                pass

            return tools

    except McpConnectionError:
        print(f"Could not connect to {MCP_SERVER_URL}.")
        print("Start an MCP server and re-run this example.")
        print()
        return []
    except McpTimeoutError:
        print(f"Connection to {MCP_SERVER_URL} timed out.")
        print()
        return []


# ── Part 2: Converting MCP tools to Quartermaster tools ────────────────────
# MCP returns tool metadata as McpTool dataclasses.  To use them inside
# a Quartermaster ToolRegistry we create thin wrapper functions.  The
# @registry.tool() decorator extracts parameter metadata from type hints
# and docstrings -- but for MCP tools the schema already exists, so we
# build the wrapper programmatically.


def mcp_to_quartermaster_tools(
    mcp_tools: list[McpTool],
    server_url: str = MCP_SERVER_URL,
) -> None:
    """Show how to wrap MCP tools for use in a Quartermaster ToolRegistry.

    This is the bridge between the MCP world (remote JSON-RPC tools)
    and the Quartermaster world (local callable tools).
    """
    print("=" * 60)
    print("PART 2 -- Converting MCP tools for Quartermaster")
    print("=" * 60)
    print()

    # When no real MCP tools are available, use a synthetic example
    # to demonstrate the pattern.
    if not mcp_tools:
        print("No live MCP tools available -- using a synthetic example.\n")
        mcp_tools = [
            McpTool(
                name="weather_lookup",
                description="Get the current weather for a city.",
                parameters=[
                    ToolParameter(
                        name="city",
                        type="string",
                        description="City name to look up.",
                        required=True,
                    ),
                    ToolParameter(
                        name="units",
                        type="string",
                        description="Temperature units: celsius or fahrenheit.",
                        required=False,
                        default="celsius",
                    ),
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                        "units": {
                            "type": "string",
                            "description": "celsius or fahrenheit",
                            "default": "celsius",
                        },
                    },
                    "required": ["city"],
                },
            ),
            McpTool(
                name="web_search",
                description="Search the web for information.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query.",
                        required=True,
                    ),
                    ToolParameter(
                        name="max_results",
                        type="integer",
                        description="Maximum number of results.",
                        required=False,
                        default="5",
                    ),
                ],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
        ]

    try:
        from quartermaster_tools import ToolRegistry
    except ImportError:
        print("quartermaster-tools is not installed; showing pattern only.\n")
        # Even without the library we can explain the approach.
        for tool in mcp_tools:
            print(f"  Would register: {tool.name}")
            for p in tool.parameters:
                req = "required" if p.required else f"default={p.default}"
                print(f"    {p.name} ({p.type}) -- {p.description} [{req}]")
        print()
        return

    registry = ToolRegistry()

    # For each MCP tool we create a thin wrapper that calls the remote
    # server over the MCP protocol.  The wrapper is registered in the
    # ToolRegistry so it can be used alongside local tools.
    for tool in mcp_tools:
        # Capture `tool` in the closure via a default argument.
        def _make_wrapper(t: McpTool) -> callable:
            def wrapper(**kwargs: object) -> dict:
                # In production this would call the MCP server:
                #
                #   with McpClient(server_url) as client:
                #       return client.call_tool_sync(t.name, kwargs)
                #
                # For this example we return a placeholder.
                return {
                    "tool": t.name,
                    "args": kwargs,
                    "note": "placeholder -- connect a real MCP server",
                }

            # Give the wrapper the tool's name and docstring so the
            # registry can use them for schema generation.
            wrapper.__name__ = t.name
            wrapper.__doc__ = t.description
            wrapper.__annotations__ = {
                p.name: str
                if p.type == "string"
                else int
                if p.type == "integer"
                else float
                if p.type == "number"
                else object
                for p in t.parameters
            }
            wrapper.__annotations__["return"] = dict
            return wrapper

        fn = _make_wrapper(tool)
        registry.tool()(fn)

    # Show what we registered
    print("Registered MCP tools in ToolRegistry:")
    for t in registry.list_tools():
        print(f"  {t.name}: {t.short_description}")
        for p in t.parameters:
            req = "required" if p.required else f"default={p.default}"
            print(f"    {p.name} ({p.type}) -- {p.description} [{req}]")
    print()

    # The JSON Schema export is what an LLM sees during function calling.
    print("JSON Schema export (for LLM function calling):")
    for schema in registry.to_json_schema():
        props = schema.get("parameters", {}).get("properties", {})
        print(f"  {schema['name']}: {len(props)} params")
    print()


# ── Part 3: A Quartermaster graph that would use MCP tools ─────────────────
# In a real deployment the MCP tool wrappers from Part 2 would be
# passed to an Agent or Instruction node via the ``tools=`` parameter.
# Below we build the graph structure to show how it all fits together.


def demo_graph_with_mcp_tools() -> None:
    """Build a graph that incorporates MCP-provided tools."""
    print("=" * 60)
    print("PART 3 -- Graph using MCP tools")
    print("=" * 60)
    print()

    try:
        import quartermaster_sdk as qm
    except ImportError:
        print("quartermaster-sdk is not installed; skipping graph demo.")
        return

    # Imagine the MCP server exposes "weather_lookup" and "web_search".
    # After wrapping them (Part 2) they live in a ToolRegistry.
    # An Agent node references them by name via ``tools=[...]``.
    graph = (
        qm.Graph("MCP Research Agent")
        .user("What would you like to research?")
        .agent(
            "Researcher",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            system_instruction=(
                "You are a research assistant with access to web search "
                "and weather lookup tools. Use them to answer the user's "
                "question thoroughly."
            ),
            # These tool names match what we registered in Part 2.
            # At runtime the engine resolves them from the ToolRegistry.
            tools=["web_search", "weather_lookup"],
            max_iterations=5,
        )
        .instruction(
            "Summariser",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            system_instruction=(
                "Summarise the researcher's findings into a clear, "
                "concise answer for the user."
            ),
        )
    )

    # Print the graph structure — we call .build() here just to inspect the
    # validated spec; ``qm.run(graph, ...)`` also accepts the builder
    # directly (auto-builds internally).
    built = graph.build()
    print("Graph structure:")
    for node in built.nodes:
        print(f"  [{node.node_type.value:15s}] {node.name}")
    print()
    print(f"Edges: {len(built.edges)}")
    for edge in built.edges:
        src = next((n.name for n in built.nodes if n.id == edge.source), "?")
        tgt = next((n.name for n in built.nodes if n.id == edge.target), "?")
        label = f" ({edge.label})" if edge.label else ""
        print(f"  {src} -> {tgt}{label}")
    print()

    # To actually run this graph:
    #
    #   import quartermaster_sdk as qm
    #   qm.run(graph, "What is the weather in Tokyo?")
    #
    # The runner wires the ToolRegistry into the FlowRunner so the
    # Agent node can call tools during its reasoning loop.
    print(
        "To execute, set an API key and call:\n"
        "  qm.run(graph, 'What is the weather in Tokyo?')\n"
    )


# ── Part 4: Sync API one-liner ─────────────────────────────────────────────
# For scripts that do not use asyncio the sync wrappers are convenient.


def demo_sync_api() -> None:
    """Demonstrate the synchronous (non-async) MCP client API."""
    print("=" * 60)
    print("PART 4 -- Synchronous API")
    print("=" * 60)
    print()

    print("The sync API mirrors the async API with a _sync suffix:")
    print()
    print("    with McpClient('http://localhost:8000/mcp') as client:")
    print("        info  = client.server_info_sync()")
    print("        tools = client.list_tools_sync()")
    print("        result = client.call_tool_sync('weather', {'city': 'Berlin'})")
    print()
    print("Sync methods run the async code on a thread when called from")
    print("inside an existing event loop, or via asyncio.run() otherwise.")
    print()

    # Quick demonstration -- this will fail gracefully if no server runs.
    try:
        with McpClient(MCP_SERVER_URL, timeout=3.0, max_retries=1) as client:
            tools = client.list_tools_sync()
            print(f"Sync discovery found {len(tools)} tools.")
    except McpConnectionError:
        print(f"(No server at {MCP_SERVER_URL} -- sync demo skipped.)")
    except McpTimeoutError:
        print(f"(Server at {MCP_SERVER_URL} timed out -- sync demo skipped.)")
    print()


# ── Main ────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Part 1: connect to a live MCP server (graceful if unavailable)
    discovered_tools = await demo_mcp_client()

    # Part 2: bridge MCP tools into the Quartermaster tool system
    mcp_to_quartermaster_tools(discovered_tools)

    # Part 3: build a graph that uses those tools
    demo_graph_with_mcp_tools()

    # Part 4: show the sync API
    demo_sync_api()

    # Wrap-up
    print("=" * 60)
    print("Done.  Key takeaways:")
    print("  - McpClient discovers tools from any MCP-compatible server")
    print("  - McpTool metadata maps cleanly to ToolRegistry wrappers")
    print("  - Agent nodes reference MCP tools by name via tools=[...]")
    print("  - Both async and sync APIs are available")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
