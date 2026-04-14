"""Tool creation with the @tool() decorator.

Demonstrates the FastMCP-style decorator pattern for creating tools
from plain functions. The decorator extracts metadata from type hints
and Google-style docstrings automatically.

This example does NOT execute a graph -- it shows standalone tool
registration and schema export. Tools are used within graphs via
``ToolRegistry`` when wiring tool-enabled agents.

Usage:
    uv run examples/06_tool_decorator.py
"""

from __future__ import annotations

from quartermaster_tools import ToolRegistry

# NOTE: Tools are used within graphs via ToolRegistry. This example
# demonstrates standalone registration and schema export only.

registry = ToolRegistry()


@registry.tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    # In production this would call a weather API
    return {"city": city, "temperature": 22, "condition": "sunny", "units": units}


@registry.tool()
def search_database(query: str, limit: int = 10) -> dict:
    """Search the knowledge database.

    Args:
        query: Search query string.
        limit: Maximum results to return.
    """
    return {"results": [f"Result for: {query}"], "count": 1}


@registry.tool()
def send_notification(recipient: str, message: str, channel: str = "email") -> dict:
    """Send a notification to a user.

    Args:
        recipient: Who to notify.
        message: Notification content.
        channel: Delivery channel (email, slack, sms).
    """
    return {"sent": True, "channel": channel, "recipient": recipient}


# Inspect registered tools
print("Registered tools:")
for t in registry.list_tools():
    print(f"  {t.name}: {t.short_description}")
    for p in t.parameters:
        req = "required" if p.required else f"optional, default={p.default}"
        print(f"    - {p.name} ({p.type}): {p.description} [{req}]")

# Use a tool directly (the decorator preserves callable behavior)
result = get_weather(city="Amsterdam", units="celsius")
print(f"\nWeather result: {result}")

# Export as JSON Schema for LLM function calling
print("\nJSON Schema export:")
for schema in registry.to_json_schema():
    print(f"  {schema['name']}: {len(schema['parameters'].get('properties', {}))} params")
