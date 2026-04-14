"""Example 22 -- Local Gemma 4 with Ollama: vision, tool calling, streaming.

Runs entirely on your machine — no cloud API keys needed. Demonstrates
Google's Gemma 4 model via Ollama with:

  - Image/vision analysis (describe what's in an image)
  - Native function calling (tools defined and called locally)
  - Streaming token output
  - Multi-turn conversation graph

Prerequisites:
    ollama serve
    ollama pull gemma4:26b    # 17GB, MoE, 256K context, vision + tools

Usage:
    uv run examples/22_ollama_local.py
"""

from __future__ import annotations

import json

from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry, LLMConfig
from quartermaster_providers.providers.local import OllamaProvider
from quartermaster_tools import ToolRegistry, tool
from quartermaster_engine import FlowRunner, InMemoryStore
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeExecutor, NodeResult
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import (
    FlowEvent, NodeStarted, NodeFinished, TokenGenerated, FlowError,
)
from quartermaster_graph.enums import NodeType


MODEL = "gemma4:26b"


# ============================================================================
# Part 1: Local tools for Gemma to call
# ============================================================================

registry = ToolRegistry()


@registry.tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    # Mock data — in production, call a real weather API
    weather_data = {
        "Ljubljana": {"temp": 22, "condition": "sunny", "humidity": 45},
        "Tokyo": {"temp": 28, "condition": "humid", "humidity": 80},
        "New York": {"temp": 18, "condition": "cloudy", "humidity": 60},
        "London": {"temp": 14, "condition": "rainy", "humidity": 85},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})
    if units == "fahrenheit":
        data["temp"] = round(data["temp"] * 9 / 5 + 32)
    return {"city": city, "temperature": data["temp"], "condition": data["condition"],
            "humidity": data["humidity"], "units": units}


@registry.tool()
def search_knowledge(query: str, max_results: int = 3) -> dict:
    """Search the local knowledge base.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    """
    # Mock knowledge base
    kb = [
        {"title": "Quartermaster SDK", "content": "AI agent orchestration framework by MindMade"},
        {"title": "Gemma 4", "content": "Google's open-weight model with vision and tool calling"},
        {"title": "Ollama", "content": "Run LLMs locally with a simple CLI"},
        {"title": "Slovenia", "content": "A country in Central Europe, capital Ljubljana"},
        {"title": "MCP Protocol", "content": "Model Context Protocol for LLM tool integration"},
    ]
    results = [r for r in kb if query.lower() in r["title"].lower() or query.lower() in r["content"].lower()]
    return {"query": query, "results": results[:max_results], "total": len(results)}


@registry.tool()
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression to evaluate (e.g., '2 + 2', 'sqrt(144)').
    """
    import math
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# ============================================================================
# Part 2: Ollama executor with streaming + tool awareness
# ============================================================================

class OllamaLLMExecutor(NodeExecutor):
    """Calls local Gemma 4 via Ollama with streaming and tool schema injection."""

    def __init__(self, provider: OllamaProvider, model: str, tool_registry: ToolRegistry):
        self._provider = provider
        self._model = model
        self._tools = tool_registry

    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("llm_system_instruction", "")
        user_input = str(context.memory.get("__user_input__", "Hello"))

        # Inject tool schemas into the system prompt so Gemma knows what's available
        tool_schemas = self._tools.to_json_schema()
        if tool_schemas and "tool" in system_instruction.lower():
            tool_desc = json.dumps(tool_schemas, indent=2)
            system_instruction += f"\n\nAvailable tools:\n{tool_desc}\n\nTo call a tool, respond with JSON: {{\"tool\": \"name\", \"args\": {{...}}}}"

        config = LLMConfig(
            model=self._model,
            provider="ollama",
            system_message=system_instruction,
            temperature=0.7,
            stream=True,
        )

        try:
            stream = await self._provider.generate_text_response(user_input, config)
            chunks = []
            async for token in stream:
                if token.content:
                    chunks.append(token.content)
                    context.emit_token(token.content)
            text = "".join(chunks)

            # Check if the model wants to call a tool
            tool_result = self._try_tool_call(text)
            if tool_result:
                context.emit_token(f"\n\n[Tool result: {json.dumps(tool_result)}]")
                text += f"\n\n[Tool result: {json.dumps(tool_result)}]"

            return NodeResult(success=True, data={}, output_text=text)
        except Exception as e:
            return NodeResult(success=False, data={}, error=str(e))

    def _try_tool_call(self, text: str) -> dict | None:
        """Try to parse and execute a tool call from the LLM response."""
        try:
            # Look for JSON tool call in the response
            start = text.find('{"tool"')
            if start == -1:
                start = text.find("{'tool'")
            if start == -1:
                return None
            end = text.find("}", start + 1)
            if end == -1:
                return None
            # Find the matching closing brace (handle nested)
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            call = json.loads(text[start:end])
            tool_name = call.get("tool")
            tool_args = call.get("args", {})
            tool_fn = self._tools.get(tool_name)
            if tool_fn:
                return tool_fn(**tool_args)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return None


class PassthroughUser(NodeExecutor):
    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(success=True, data={}, output_text=str(context.memory.get("__user_input__", "")))


# ============================================================================
# Part 3: Build and run the graph
# ============================================================================

def main():
    # Set up Ollama provider
    provider = OllamaProvider()

    # Build node registry
    node_registry = SimpleNodeRegistry()
    llm = OllamaLLMExecutor(provider, MODEL, registry)
    node_registry.register(NodeType.INSTRUCTION.value, llm)
    node_registry.register(NodeType.INSTRUCTION_IMAGE_VISION.value, llm)
    node_registry.register(NodeType.USER.value, PassthroughUser())

    # Event handler with streaming
    _skip = {NodeType.START.value, NodeType.END.value, NodeType.USER.value}
    _node_types = {}  # track node_id -> node_type for NodeFinished

    def on_event(event: FlowEvent) -> None:
        if isinstance(event, NodeStarted):
            _node_types[event.node_id] = event.node_type.value
            if event.node_type.value not in _skip:
                print(f"\n  [{event.node_name}]", flush=True)
        elif isinstance(event, TokenGenerated):
            print(event.token, end="", flush=True)
        elif isinstance(event, NodeFinished):
            ntype = _node_types.get(event.node_id, "")
            if ntype not in _skip:
                print(flush=True)
        elif isinstance(event, FlowError):
            print(f"\n  [ERROR] {event.error}", flush=True)

    # ---- Demo 1: Vision (image analysis) ----

    print("=" * 60)
    print(f"  Gemma 4 Local Demo ({MODEL} via Ollama)")
    print("=" * 60)

    vision_graph = (
        Graph("Gemma Vision")
        .start()
        .user("Describe the scene")
        .vision(
            "Analyze scene",
            model=MODEL, provider="ollama",
            system_instruction=(
                "You are a visual analyst. Describe what you see in detail: "
                "subjects, colors, composition, mood. Be specific and vivid. "
                "Keep it to 3-4 sentences."
            ),
        )
        .end()
    ).build()

    print("\n--- Demo 1: Vision Analysis ---")
    runner = FlowRunner(graph=vision_graph, node_registry=node_registry,
                        store=InMemoryStore(), on_event=on_event)
    result = runner.run(
        "A cozy wooden cabin in the mountains at sunset, warm light from windows, "
        "snow on the peaks, pink and orange clouds in the sky"
    )
    print(f"\n  Duration: {result.duration_seconds:.1f}s")

    # ---- Demo 2: Tool calling ----

    tool_graph = (
        Graph("Gemma Tools")
        .start()
        .user("Ask me anything")
        .instruction(
            "Think and use tools",
            model=MODEL, provider="ollama",
            system_instruction=(
                "You are a helpful assistant with access to tools. "
                "When the user asks something you can answer with a tool, "
                "call it by responding with JSON: {\"tool\": \"name\", \"args\": {...}}. "
                "Available tools: get_weather (city, units), search_knowledge (query), "
                "calculate (expression)."
            ),
        )
        .end()
    ).build()

    print("\n--- Demo 2: Tool Calling ---")
    runner2 = FlowRunner(graph=tool_graph, node_registry=node_registry,
                         store=InMemoryStore(), on_event=on_event)
    result2 = runner2.run("What's the weather like in Ljubljana and what is 42 * 17?")
    print(f"\n  Duration: {result2.duration_seconds:.1f}s")

    # ---- Demo 3: Multi-step pipeline ----

    pipeline_graph = (
        Graph("Gemma Pipeline")
        .start()
        .user("Topic")
        .instruction(
            "Research",
            model=MODEL, provider="ollama",
            system_instruction="Write 3 key facts about the given topic. Be concise.",
        )
        .instruction(
            "Summarize",
            model=MODEL, provider="ollama",
            system_instruction=(
                "Take the research above and write a single compelling paragraph "
                "that a non-expert would understand. Under 100 words."
            ),
        )
        .end()
    ).build()

    print("\n--- Demo 3: Multi-step Pipeline ---")
    runner3 = FlowRunner(graph=pipeline_graph, node_registry=node_registry,
                         store=InMemoryStore(), on_event=on_event)
    result3 = runner3.run("The history of artificial intelligence")
    print(f"\n  Duration: {result3.duration_seconds:.1f}s")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  All demos completed locally — zero cloud API calls!")
    print(f"  Model: {MODEL}")
    print("  Features shown: vision, tool calling, streaming, multi-step")
    print("=" * 60)


if __name__ == "__main__":
    main()
