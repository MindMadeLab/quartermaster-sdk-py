# Quartermaster

[![CI](https://github.com/MindMade/quartermaster/actions/workflows/ci.yml/badge.svg)](https://github.com/MindMade/quartermaster/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Modular AI agent orchestration framework. Build agent workflows as directed graphs, wire them with a fluent Python API, and run them with any LLM provider.

Built by [MindMade](https://mindmade.si) in Slovenia.

## Install

```bash
# Everything (recommended)
pip install quartermaster-sdk

# With a specific LLM provider
pip install quartermaster-sdk[openai]
pip install quartermaster-sdk[anthropic]

# With uv
uv pip install quartermaster-sdk
```

## Quick Start

```python
from quartermaster_sdk import Graph

agent = (
    Graph("My Agent")
    .start()
    .user("What can I help you with?")
    .instruction("Respond", model="gpt-4o", system_instruction="You are a helpful assistant.")
    .end()
)
```

### Decision Routing

The LLM classifies input and picks ONE branch. No merge needed.

```python
agent = (
    Graph("Router")
    .start()
    .user("Describe your issue")
    .instruction("Classify", system_instruction="Classify as: Technical or General.")
    .decision("Category?", options=["Technical", "General"])
    .on("Technical")
        .instruction("Tech response", system_instruction="Give a technical answer.")
    .end()
    .on("General")
        .instruction("General response", system_instruction="Give a general answer.")
    .end()
    .end()
)
```

### Parallel Execution

All branches run concurrently, then merge.

```python
agent = (
    Graph("Code Review")
    .start()
    .user("Paste your code")
    .parallel()
    .branch()
        .instruction("Security audit", system_instruction="Check for vulnerabilities.")
    .end()
    .branch()
        .instruction("Performance check", system_instruction="Check for performance issues.")
    .end()
    .static_merge("Collect results")
    .instruction("Final report", system_instruction="Combine all findings.")
    .end()
)
```

### User Forms and Templates

```python
agent = (
    Graph("Registration")
    .start()
    .user("Welcome!")
    .user_form("Details", parameters=[
        {"name": "full_name", "type": "text", "label": "Name", "required": "true"},
        {"name": "email",     "type": "email", "label": "Email", "required": "true"},
    ])
    .var("Capture name", variable="name", expression="full_name")
    .text("Confirm", template="Thanks {{full_name}}, we'll email {{email}} with details.")
    .end()
)
```

See [`examples/`](./examples/) for 16 runnable examples covering every pattern.

## Packages

| Package | Description |
|---------|-------------|
| [`quartermaster-sdk`](./quartermaster-sdk/) | Meta-package -- installs all core packages |
| [`quartermaster-graph`](./quartermaster-graph/) | Graph schema, fluent builder API, validation |
| [`quartermaster-providers`](./quartermaster-providers/) | LLM provider abstraction (OpenAI, Anthropic, Google, Groq, Ollama, vLLM) |
| [`quartermaster-tools`](./quartermaster-tools/) | Tool definition, registry, 50+ built-in tools |
| [`quartermaster-nodes`](./quartermaster-nodes/) | Node execution protocols and 40+ node implementations |
| [`quartermaster-engine`](./quartermaster-engine/) | Flow execution, traversal, memory, streaming |
| [`quartermaster-mcp-client`](./quartermaster-mcp-client/) | MCP protocol client -- standalone, no framework dependency |
| [`quartermaster-code-runner`](./quartermaster-code-runner/) | Docker sandboxed code execution -- standalone FastAPI service |

## Architecture

```
Your Application
       |
       v
quartermaster-engine        Flow execution, traversal, streaming
  |         |         |
  v         v         v
graph     nodes     tools   Schema/builder, node executors, tool registry
            |
            v
         providers          OpenAI, Anthropic, Google, Groq, Ollama, vLLM, ...

quartermaster-mcp-client    Standalone MCP protocol client
quartermaster-code-runner   Standalone Docker code execution
```

## Key Concepts

- **Graph** -- A directed acyclic graph of nodes and edges. Built with the fluent `Graph("name").start()...end()` API.
- **Nodes** -- Units of work: LLM calls, decisions, user input, memory, tools, templates.
- **Edges** -- Directed connections between nodes. Decision/IF/Switch edges carry labels.
- **Thoughts** -- Runtime containers that carry text and variables (metadata) between nodes.
- **Memory** -- Flow-scoped persistent storage accessible from any node via `write_memory`/`read_memory`.
- **Providers** -- Pluggable LLM backends. Model name auto-resolves to the right provider.

### Branching Rules

| Node Type | Behavior | Merge Needed? |
|-----------|----------|---------------|
| `decision()` | LLM picks ONE branch | No |
| `if_node()` | Boolean expression picks ONE branch | No |
| `switch()` | Expression picks ONE branch | No |
| `parallel()` | ALL branches run concurrently | Yes -- use `static_merge()` |

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./docs/getting-started.md) | Installation and first agent |
| [Graph Building](./docs/graph-building.md) | Builder API, node types, patterns |
| [Architecture](./docs/architecture.md) | System overview and data flow |
| [Providers](./docs/providers.md) | LLM providers including local (Ollama, vLLM) |
| [Tools Catalog](./docs/tools-catalog.md) | All 50+ built-in tools with parameters |
| [Engine](./docs/engine.md) | Execution engine internals |
| [Security](./docs/security.md) | Safe eval, sandboxing, API key management |
| [Node Reference](./docs/nodes/README.md) | Detailed node documentation by category |

## Development

```bash
# Clone
git clone https://github.com/MindMade/quartermaster.git
cd quartermaster

# Install all packages (with uv)
uv pip install -e quartermaster-providers -e quartermaster-graph \
  -e quartermaster-tools -e quartermaster-nodes -e quartermaster-engine \
  -e quartermaster-sdk

# Run tests for a single package
cd quartermaster-graph && pytest

# Run all examples
for f in examples/*.py; do python "$f"; done
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development guide.

## License

[Apache 2.0](./LICENSE) -- Built by [MindMade](https://mindmade.si) in Slovenia.
