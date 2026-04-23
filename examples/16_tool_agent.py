"""Example 16 -- Agent with tools.

Demonstrates the full agentic loop: an Agent node receives the user's
request, picks one or more registered ``@tool()`` callables, the SDK
actually executes them, feeds the results back into the model, and
the model produces a final text answer.

This replaces the pre-v0.1 "format the prompt with a TOOL:/ARGS:/REASONING:
template and ask the model to roleplay tool calling" pattern. With
``.agent(tools=[...])`` the SDK does the dispatch for you and the
caller can introspect every tool invocation via ``result.trace.tool_calls``.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/16_tool_agent.py
"""

from __future__ import annotations

import json

import quartermaster_sdk as qm
from quartermaster_tools import ToolRegistry

# ---------------------------------------------------------------------------
# 1. Define custom tools with @registry.tool()
# ---------------------------------------------------------------------------
#
# v0.3.1: register tools on a ToolRegistry the agent loop can dispatch
# from.  The registry is passed through ``qm.run(..., tool_registry=)``
# and the engine looks tool names up by string.

registry = ToolRegistry()


@registry.tool()
def calculate(expression: str) -> dict:
    """Evaluate a math expression safely.

    Args:
        expression: A mathematical expression to evaluate (e.g. '2 + 2 * 3').
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Expression contains disallowed characters"}
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


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
        "brazil": "Brasilia",
        "australia": "Canberra",
        "canada": "Ottawa",
        "italy": "Rome",
    }
    key = country.lower().strip()
    if key in capitals:
        return {"country": country, "capital": capitals[key]}
    return {"error": f"Capital not found for '{country}'"}


@registry.tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city (mock data).

    Args:
        city: The city name to check weather for.
        units: Temperature units -- celsius or fahrenheit.
    """
    # Mock weather data
    forecasts = {
        "paris": {"temperature": 18, "condition": "partly cloudy"},
        "berlin": {"temperature": 14, "condition": "overcast"},
        "tokyo": {"temperature": 24, "condition": "sunny"},
        "ljubljana": {"temperature": 16, "condition": "clear"},
    }
    key = city.lower().strip()
    data = forecasts.get(key, {"temperature": 20, "condition": "unknown"})
    return {"city": city, "units": units, **data}


# ---------------------------------------------------------------------------
# 2. Inspect registered tools (for orientation -- not required for execution)
# ---------------------------------------------------------------------------

print("Registered tools:")
for t in registry.list_tools():
    print(f"  {t.name}: {t.short_description}")
    for p in t.parameters:
        req = "required" if p.required else f"optional, default={p.default}"
        print(f"    - {p.name} ({p.type}): {p.description} [{req}]")

# Direct call -- the @tool() decorator preserves callable behaviour, so
# you can still invoke them like normal Python functions in tests.
print(
    f"\nDirect call -- calculate('(3 + 4) * 2'): {calculate(expression='(3 + 4) * 2')}"
)
print(
    f"Direct call -- lookup_capital('Slovenia'): {lookup_capital(country='Slovenia')}"
)

# ---------------------------------------------------------------------------
# 3. Export JSON schemas for downstream MCP servers / multi-provider plumbing
# ---------------------------------------------------------------------------
#
# Useful for shipping the same tool set to a remote MCP server (see
# example 24) or hand-rolling a provider that doesn't speak the
# Quartermaster registry directly.

schemas = registry.to_json_schema()
print("\nJSON schemas for LLM:")
for s in schemas:
    print(f"  {s['name']}: {json.dumps(s, indent=2)[:120]}...")

# Also available in provider-specific formats
anthropic_tools = registry.to_anthropic_tools()
print(f"\nAnthropic format: {len(anthropic_tools)} tools exported")

# ---------------------------------------------------------------------------
# 4. Build a graph that uses .agent(tools=[...]) -- the SDK runs the loop
# ---------------------------------------------------------------------------
#
# v0.3.1: use .agent(tools=[name, name, ...]) so the SDK actually
# executes the tools in a reasoning loop (was .instruction() with a
# prompt-formatted "TOOL: ... / ARGS: ... / REASONING: ..." template
# that the model only roleplayed).  The agent node:
#   - receives the user's request
#   - asks the LLM which tool(s) to call
#   - dispatches them via the tool_registry passed to qm.run
#   - feeds tool results back into the next iteration
#   - returns the final natural-language answer

agent = (
    qm.Graph("Tool Agent")
    .user("Ask anything that needs a tool")
    .agent(
        "assistant",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "You are a helpful assistant.  When you need to look up a "
            "country's capital, call lookup_capital.  For weather use "
            "get_weather.  For math use calculate.  Combine the results "
            "into one concise answer."
        ),
        tools=["calculate", "lookup_capital", "get_weather"],
        max_iterations=5,
        capture_as="response",
    )
)

# ---------------------------------------------------------------------------
# 5. Run with qm.run(...) so we can pass the tool_registry and inspect
#    result.trace.tool_calls afterwards
# ---------------------------------------------------------------------------

result = qm.run(
    agent,
    user_input="What is the capital of Japan and what's the weather there?",
    tool_registry=registry,
)

print("\n" + "=" * 60)
print("Final answer:")
print("=" * 60)
print(result.text)

print("\n" + "=" * 60)
print(f"Tool calls executed during the run ({len(result.trace.tool_calls)}):")
print("=" * 60)
for call in result.trace.tool_calls:
    args = call.get("arguments", {})
    out = call.get("result")
    if out is not None and len(str(out)) > 80:
        out = str(out)[:77] + "..."
    print(f"  {call['tool']}({args}) -> {out}")
