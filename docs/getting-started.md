# Getting Started

This guide walks you through installing Quartermaster, building your first agent graph, and running it.

## Prerequisites

- **Python 3.11+** (3.12 or 3.13 recommended)
- **pip** or any PEP 517-compatible installer (uv, poetry, pdm)
- An API key for at least one LLM provider (OpenAI, Anthropic, Google, Groq, or xAI)

Optional:
- **Docker** -- Required only if you use `quartermaster-code-runner` for sandboxed code execution

## Installation

### Individual Packages

Each package can be installed independently. Install only what you need:

```bash
# LLM provider abstraction (OpenAI, Anthropic, Google, Groq, xAI)
pip install quartermaster-providers

# Tool framework (standalone, no LLM dependency)
pip install quartermaster-tools

# Graph schema and builder
pip install quartermaster-graph

# Full execution engine (pulls in quartermaster-graph, quartermaster-nodes, quartermaster-tools)
pip install quartermaster-engine
```

### Standalone Packages

These two packages have no dependency on the core framework:

```bash
# MCP protocol client (SSE + Streamable HTTP)
pip install quartermaster-mcp-client

# Docker-based sandboxed code execution
pip install quartermaster-code-runner
```

### From Source

```bash
git clone https://github.com/MindMadeLab/quartermaster-sdk-py.git
cd quartermaster-sdk-py

# Install a specific package in development mode
pip install -e ./quartermaster-graph
pip install -e ./quartermaster-providers
pip install -e ./quartermaster-engine
```

## Your First Agent in 5 Minutes

### Step 1: Set Up a Provider

Configure your LLM provider with an API key:

```python
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.providers.openai import OpenAIProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key="sk-...")
```

The registry supports automatic model-to-provider inference. When the engine encounters `model="gpt-4o"`, it resolves to the `openai` provider automatically.

### Step 2: Build a Graph

Use the `Graph` fluent API to define your agent's workflow:

```python
from quartermaster_graph import Graph

graph = (
    Graph("My First Agent")
    .start()
    .user("What would you like to know?")
    .instruction("Analyze input", model="gpt-4o", temperature=0.7)
    .end()
)
```

This creates a minimal graph: Start -> User (collect input) -> Instruction (LLM call) -> End.

### Step 3: Run the Graph

```python
from quartermaster_engine import FlowRunner
from quartermaster_engine.nodes import SimpleNodeRegistry

node_registry = SimpleNodeRegistry()
runner = FlowRunner(graph=graph, node_registry=node_registry)

result = runner.run("Explain quantum computing in one sentence.")
print(result.final_output)
print(f"Success: {result.success}")
print(f"Duration: {result.duration_seconds:.2f}s")
```

### Step 4: Add Branching

Make it more interesting with a decision node:

```python
graph = (
    Graph("Sentiment Analyzer")
    .start()
    .user("Enter text to analyze")
    .instruction(
        "Classify sentiment",
        model="gpt-4o",
        system_instruction="Classify the user's message as Positive or Negative. Reply with one word only.",
    )
    .decision("Sentiment?", options=["Positive", "Negative"])
    .on("Positive")
        .instruction("Positive reply", system_instruction="Write an enthusiastic response.")
        .end()
    .on("Negative")
        .instruction("Negative reply", system_instruction="Write an empathetic response.")
        .end()
    # Decision picks ONE branch -- no merge needed. Branches converge on the next node.
    .end()
)
```

### Step 5: Stream Events

For real-time feedback from the SDK, use the filtered stream iterators
(v0.3.0):

```python
import quartermaster_sdk as qm

# Typewriter effect -- just the model tokens.
for token in qm.run.stream(graph, "I just got promoted!").tokens():
    print(token, end="", flush=True)

# Or listen to tool calls / progress / custom events instead:
for call in qm.run.stream(graph, "Research topic X").tool_calls():
    print(f"[TOOL] {call.tool}({call.args})")

for prog in qm.run.stream(graph, "Crunch the dataset").progress():
    print(f"[{prog.percent:.0%}] {prog.message}")
```

Streams are single-pass -- pick one consumer per stream. See
[engine.md](engine.md) for the low-level `FlowEvent` API that the SDK
chunk filters are built on top of.

### Step 5b: OpenTelemetry (optional)

For production observability, install the telemetry extra and flip it
on with a single call:

```bash
pip install 'quartermaster-sdk[telemetry]'
```

```python
from quartermaster_sdk import telemetry

telemetry.instrument()    # every subsequent run emits OTEL GenAI spans
```

Point your exporter at Jaeger, Tempo, Honeycomb, Logfire, Phoenix, or
any OTLP collector. Spans follow the OpenTelemetry GenAI semantic
conventions (`gen_ai.system`, `gen_ai.operation.name`,
`gen_ai.tool.name`, `gen_ai.usage.input_tokens`, ...).

## Project Structure

A typical project using Quartermaster looks like this:

```
my-agent/
    agents/
        analyzer.py      # Graph definitions (Graph)
        summarizer.py
    config.py            # Provider registry setup
    main.py              # FlowRunner execution
    requirements.txt     # quartermaster-engine (+ quartermaster-providers, quartermaster-graph, etc.)
```

## Environment Variables

Store API keys in environment variables rather than in code:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
import os
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.providers.anthropic import AnthropicProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key=os.environ["OPENAI_API_KEY"])
registry.register("anthropic", AnthropicProvider, api_key=os.environ["ANTHROPIC_API_KEY"])
```

### Step 6: Create Custom Tools

Use the `@tool` decorator to define tools from plain functions:

```python
from quartermaster_tools import tool

@tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    return {"city": city, "temperature": 22, "units": units}

# That's it -- the decorator defines the tool. No extra registration needed.
result = get_weather(city="Amsterdam")  # Call directly
schema = get_weather.to_json_schema()   # Export for LLM function calling
```

## Next Steps

- [Graph Building](graph-building.md) -- Complete Graph builder API, node types, edge types, and patterns
- [Providers](providers.md) -- Configure multiple LLM providers, streaming, cost estimation
- [Tools](tools.md) -- Create custom tools with the `@tool` decorator and register them for agent use
- [Engine](engine.md) -- Dispatchers, memory stores, error handling, streaming
- [Architecture](architecture.md) -- Full system design and data flow
- [Security](security.md) -- Security best practices for production deployments
