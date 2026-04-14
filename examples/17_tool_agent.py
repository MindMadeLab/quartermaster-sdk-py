"""Example 17 -- Agent with tools.

Demonstrates creating custom tools with @tool(), registering them,
exporting JSON schemas, and showing how they integrate with agent graphs.

The FlowRunner doesn't natively wire tool calling at runtime (that
requires the nodes package's InstructionProgram node), so this example
shows the pattern: define tools, register them, generate schemas, and
pass those schemas to an instruction node so the LLM can reason about
available capabilities.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/17_tool_agent.py
"""

from __future__ import annotations

import json

from quartermaster_graph import Graph
from quartermaster_tools import ToolRegistry
from quartermaster_engine import run_graph

# ---------------------------------------------------------------------------
# 1. Define custom tools with @tool()
# ---------------------------------------------------------------------------

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
# 2. Inspect registered tools
# ---------------------------------------------------------------------------

print("Registered tools:")
for t in registry.list_tools():
    print(f"  {t.name}: {t.short_description}")
    for p in t.parameters:
        req = "required" if p.required else f"optional, default={p.default}"
        print(f"    - {p.name} ({p.type}): {p.description} [{req}]")

# Call a tool directly (the decorator preserves callable behavior)
result = calculate(expression="(3 + 4) * 2")
print(f"\nDirect call -- calculate('(3 + 4) * 2'): {result}")

result = lookup_capital(country="Slovenia")
print(f"Direct call -- lookup_capital('Slovenia'): {result}")

# ---------------------------------------------------------------------------
# 3. Export JSON schemas for LLM function calling
# ---------------------------------------------------------------------------

schemas = registry.to_json_schema()
print("\nJSON schemas for LLM:")
for s in schemas:
    print(f"  {s['name']}: {json.dumps(s, indent=2)[:120]}...")

# Also available in provider-specific formats
anthropic_tools = registry.to_anthropic_tools()
print(f"\nAnthropic format: {len(anthropic_tools)} tools exported")

# ---------------------------------------------------------------------------
# 4. Build a graph that passes tool schemas to the LLM
# ---------------------------------------------------------------------------

tool_descriptions = json.dumps(schemas, indent=2)

agent = (
    Graph("Tool Agent")
    .start()
    .user("Ask something that needs a tool")
    .instruction(
        "Reason about tools",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "You are an assistant with access to the following tools:\n\n"
            f"{tool_descriptions}\n\n"
            "When the user asks a question that could be answered by one of "
            "these tools, explain which tool you would call and with what "
            "arguments. Format your response as:\n"
            "TOOL: <tool_name>\n"
            "ARGS: <json arguments>\n"
            "REASONING: <why this tool is appropriate>\n\n"
            "If no tool is needed, just answer directly."
        ),
    )
    .instruction(
        "Summarise",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "You received a tool-reasoning response from a previous step. "
            "Summarise the tool that would be called and why, or provide "
            "the direct answer. Be concise -- one or two sentences."
        ),
    )
    .end()
)

# ---------------------------------------------------------------------------
# 5. Run the graph
# ---------------------------------------------------------------------------

run_graph(agent, user_input="What is the capital of Japan and what's the weather there?")
