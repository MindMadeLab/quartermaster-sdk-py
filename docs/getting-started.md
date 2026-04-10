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
git clone https://github.com/MindMade/quartermaster-opensource.git
cd quartermaster-opensource

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

Use the `GraphBuilder` fluent API to define your agent's workflow:

```python
from quartermaster_graph import GraphBuilder

graph = (
    GraphBuilder("My First Agent")
    .start()
    .instruction("Analyze input", model="gpt-4o", temperature=0.7)
    .end()
    .build()
)
```

This creates a minimal graph: Start -> Instruction (LLM call) -> End.

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
    GraphBuilder("Sentiment Analyzer")
    .start()
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
    .build()
)
```

### Step 5: Stream Events

For real-time feedback, use the async streaming API:

```python
import asyncio

async def main():
    runner = FlowRunner(graph=graph, node_registry=node_registry)

    async for event in runner.run_async("I just got promoted!"):
        match event:
            case NodeStarted(node_name=name):
                print(f"[started] {name}")
            case TokenGenerated(token=tok):
                print(tok, end="", flush=True)
            case NodeFinished(result=res):
                print(f"\n[finished] {res[:50]}...")
            case FlowFinished(final_output=out):
                print(f"\n\nFinal: {out}")

asyncio.run(main())
```

## Project Structure

A typical project using Quartermaster looks like this:

```
my-agent/
    agents/
        analyzer.py      # Graph definitions (GraphBuilder)
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

## Next Steps

- [Graph Building](graph-building.md) -- Complete GraphBuilder API, all 50+ node types, edge types, and templates
- [Providers](providers.md) -- Configure multiple LLM providers, streaming, cost estimation
- [Tools](tools.md) -- Create custom tools and register them for agent use
- [Engine](engine.md) -- Dispatchers, memory stores, error handling, streaming
- [Architecture](architecture.md) -- Full system design and data flow
- [Security](security.md) -- Security best practices for production deployments
