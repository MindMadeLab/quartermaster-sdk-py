"""Example 24 -- Expose Quartermaster tools as an MCP server.

Pairs with example 19 (MCP client).  Take the ``@tool()``-decorated
callables you already wrote for your agents (see examples 16, 20, 21),
plug them into a ``FastMCP`` server with a few lines, and any
MCP-compatible client -- Claude Desktop, Cursor, Continue, this SDK's
own ``McpClient`` (example 19) -- can call them.

The bridge is small on purpose: the Quartermaster ``ToolRegistry``
already exports MCP-shaped descriptors via ``to_mcp_tools()``.  Each
``FunctionTool`` is also a Python callable (the ``@tool()`` decorator
preserves the original function), so we just iterate the registry and
register each tool with FastMCP.

Install:
    pip install mcp                         # official Python MCP SDK
    # or: pip install quartermaster-sdk[mcp] for both client + server stack

Run:
    uv run examples/24_mcp_server.py        # listens on stdio for an MCP client

Hook it up to Claude Desktop by adding to your ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "quartermaster-demo": {
          "command": "uv",
          "args": ["run", "examples/24_mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import sys

from quartermaster_tools import ToolRegistry

# ---------------------------------------------------------------------------
# 1. Define tools the normal Quartermaster way
# ---------------------------------------------------------------------------
#
# Identical to how you'd register them for an in-process agent (see
# example 16) -- there is no MCP-specific decorator.

registry = ToolRegistry()


@registry.tool()
def add(a: int, b: int) -> dict:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return {"result": a + b}


@registry.tool()
def multiply(a: int, b: int) -> dict:
    """Multiply two integers.

    Args:
        a: First factor.
        b: Second factor.
    """
    return {"result": a * b}


@registry.tool()
def lookup_capital(country: str) -> dict:
    """Look up the capital city of a country.

    Args:
        country: The country name to look up.
    """
    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "japan": "Tokyo",
        "slovenia": "Ljubljana",
    }
    key = country.lower().strip()
    if key in capitals:
        return {"country": country, "capital": capitals[key]}
    return {"error": f"Capital not found for '{country}'"}


# ---------------------------------------------------------------------------
# 2. Wrap them as an MCP server
# ---------------------------------------------------------------------------
#
# FastMCP is the official high-level wrapper from the Python MCP SDK
# (``pip install mcp``).  It accepts plain Python callables and infers
# the JSON Schema from type hints + docstrings -- which is exactly the
# metadata our ``@tool()`` decorator already preserves on the
# ``FunctionTool`` callable, so no extra plumbing is needed.

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "mcp is not installed. Install the official Python MCP SDK:\n"
        "    pip install mcp\n"
        "    # or: pip install quartermaster-sdk[mcp]\n",
        file=sys.stderr,
    )
    sys.exit(1)


mcp = FastMCP("quartermaster-tools-demo")

# ``to_mcp_tools()`` returns the canonical MCP-shaped descriptors
# (name / description / inputSchema).  ``registry.get(name)`` returns
# the underlying ``FunctionTool``, which is itself callable -- so we
# can hand it straight to ``mcp.tool()`` and FastMCP will wire it up
# using the description we pass.
for spec in registry.to_mcp_tools():
    fn = registry.get(spec["name"])
    mcp.tool(
        name=spec["name"],
        description=spec.get("description", ""),
    )(fn)


# ---------------------------------------------------------------------------
# 3. Serve over stdio
# ---------------------------------------------------------------------------
#
# stdio is the default transport MCP clients (Claude Desktop, Cursor,
# Continue, the SDK's own ``McpClient``) speak when launching a server
# as a subprocess.  Switch to ``transport="streamable-http"`` if you
# want to host the server behind a URL instead.

if __name__ == "__main__":
    mcp.run(transport="stdio")
