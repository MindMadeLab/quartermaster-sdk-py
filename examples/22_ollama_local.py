"""Example 22 -- Local Gemma 4 with Ollama: vision, tool calling, streaming.

Runs entirely on your machine — no cloud API keys needed. Demonstrates
Google's Gemma 4 (26B MoE) via Ollama with vision analysis, tool-aware
agents, and multi-step pipelines — all with streaming output.

Prerequisites:
    ollama serve
    ollama pull gemma4:26b    # 17GB, MoE, 256K context, vision + tools

Usage:
    uv run examples/22_ollama_local.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_tools import ToolRegistry, tool
from quartermaster_engine import run_graph


MODEL = "gemma4:26b"
PROVIDER = "ollama"


# ============================================================================
# Local tools for Gemma to reason about
# ============================================================================

registry = ToolRegistry()


@registry.tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    weather = {
        "Ljubljana": {"temp": 22, "condition": "sunny", "humidity": 45},
        "Tokyo": {"temp": 28, "condition": "humid", "humidity": 80},
        "New York": {"temp": 18, "condition": "cloudy", "humidity": 60},
    }
    data = weather.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})
    if units == "fahrenheit":
        data["temp"] = round(data["temp"] * 9 / 5 + 32)
    return {"city": city, **data, "units": units}


@registry.tool()
def search_knowledge(query: str) -> dict:
    """Search the local knowledge base.

    Args:
        query: Search query string.
    """
    kb = [
        {"title": "Quartermaster", "fact": "AI agent orchestration framework by MindMade"},
        {"title": "Gemma 4", "fact": "Google's open model with vision and tool calling"},
        {"title": "Slovenia", "fact": "Country in Central Europe, capital Ljubljana"},
    ]
    results = [r for r in kb if query.lower() in r["title"].lower() or query.lower() in r["fact"].lower()]
    return {"query": query, "results": results}


# Build tool descriptions for the system prompt
tool_list = "\n".join(f"- {s['name']}: {s.get('description', '')}" for s in registry.to_json_schema())


# ============================================================================
# Demo 1: Vision — describe a scene
# ============================================================================

print("=" * 60)
print(f"  Gemma 4 Local Demo ({MODEL} via Ollama)")
print("=" * 60)

print("\n--- Demo 1: Vision Analysis ---\n")

vision_agent = (
    Graph("Gemma Vision")
    .start()
    .user("Describe the scene")
    .vision(
        "Analyze scene",
        model=MODEL, provider=PROVIDER,
        system_instruction=(
            "You are a visual analyst. Describe what you see: "
            "subjects, colors, composition, mood. 3-4 vivid sentences."
        ),
    )
    .end()
)

run_graph(
    vision_agent,
    user_input=(
        "A cozy wooden cabin in the Swiss Alps at sunset. Snow-capped peaks "
        "reflect golden and purple light. Warm amber glow from the cabin windows. "
        "Pink and orange clouds layered across a deep blue sky."
    ),

)


# ============================================================================
# Demo 2: Tool-aware agent
# ============================================================================

print("\n--- Demo 2: Tool Calling ---\n")

tool_agent = (
    Graph("Gemma Tools")
    .start()
    .user("Ask anything")
    .instruction(
        "Think and call tools",
        model=MODEL, provider=PROVIDER,
        system_instruction=(
            "You are a helpful assistant with tools. When you need data, "
            "describe which tool you would call and why.\n\n"
            f"Available tools:\n{tool_list}\n\n"
            "Reason step-by-step about which tools to use, then answer."
        ),
    )
    .end()
)

run_graph(
    tool_agent,
    user_input="What's the weather in Ljubljana and what do you know about Slovenia?",

)


# ============================================================================
# Demo 3: Multi-step research pipeline
# ============================================================================

print("\n--- Demo 3: Multi-step Pipeline ---\n")

pipeline = (
    Graph("Gemma Pipeline")
    .start()
    .user("Topic")
    .instruction(
        "Research",
        model=MODEL, provider=PROVIDER,
        system_instruction="Write 3 key facts about the given topic. Be concise and specific.",
    )
    .instruction(
        "Summarize",
        model=MODEL, provider=PROVIDER,
        system_instruction=(
            "Take the research above and write a single compelling paragraph "
            "for a non-expert. Under 100 words."
        ),
    )
    .end()
)

run_graph(
    pipeline,
    user_input="The history of artificial intelligence",

)


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("  All demos completed locally — zero cloud API calls!")
print(f"  Model: {MODEL}")
print("  Features: vision, tool awareness, streaming, multi-step")
print("=" * 60)
