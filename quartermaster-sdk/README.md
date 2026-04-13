# Quartermaster SDK

**Modular AI agent orchestration framework by [MindMade](https://mindmade.io).**

Quartermaster lets you build AI agent workflows as directed graphs — define nodes (LLM calls, decisions, user input, tools), connect them with edges, and execute them with a pluggable engine.

## Quick Install

```bash
# Core framework (graph + providers + tools + nodes + engine)
pip install quartermaster-sdk

# With OpenAI
pip install quartermaster-sdk[openai]

# With everything (all providers, all tools, MCP client, code runner)
pip install quartermaster-sdk[all]
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

## Packages

| Package | Description |
|---------|-------------|
| `quartermaster-graph` | Graph schema, builder API, validation |
| `quartermaster-providers` | LLM provider abstraction (OpenAI, Anthropic, Google, Groq, local) |
| `quartermaster-tools` | Tool definition, registry, 50+ built-in tools |
| `quartermaster-nodes` | Node execution protocols and implementations |
| `quartermaster-engine` | Flow execution, traversal, memory, streaming |
| `quartermaster-mcp-client` | MCP protocol client (standalone) |
| `quartermaster-code-runner` | Docker sandboxed code execution (standalone) |

## Documentation

See the [docs/](https://github.com/MindMade/quartermaster/tree/master/docs) directory:

- [Getting Started](docs/getting-started.md)
- [Graph Building](docs/graph-building.md)
- [Architecture](docs/architecture.md)
- [Tools Catalog](docs/tools-catalog.md)
- [Providers](docs/providers.md)
- [Security](docs/security.md)

## License

Apache 2.0
