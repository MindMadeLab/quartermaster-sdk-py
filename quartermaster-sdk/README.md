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

## Quick Start (local Ollama, zero config)

```bash
ollama pull gemma4:26b      # or any model you've pulled
```

```python
from quartermaster_sdk import Graph, FlowRunner, register_local

provider_registry = register_local(
    "ollama",
    base_url="http://localhost:11434",   # or set $OLLAMA_HOST
    default_model="gemma4:26b",
)

graph = Graph("chat").start().user().agent().end().build()
runner = FlowRunner(graph=graph, provider_registry=provider_registry)
result = runner.run("Pozdravljen, koliko je ura?")
print(result.final_output)
```

## Quick Start (cloud provider)

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

## Sync chat shim (no graph needed)

For one-shot LLM calls from sync code (Celery workers, Django views, CLI scripts) —
no `asgiref.async_to_sync` wrapper required:

```python
from quartermaster_providers.providers.local import OllamaProvider

provider = OllamaProvider(default_model="gemma4:26b")
result = provider.chat(
    messages=[{"role": "user", "content": "Pozdravljen!"}],
    max_output_tokens=128,
    thinking_level="off",
)
print(result.content)        # promoted from `reasoning` if `content` is empty
print(result.usage)          # {prompt_tokens, completion_tokens, total_tokens}
```

## Packages

| Package | Description |
|---------|-------------|
| `quartermaster-graph` | Graph schema, builder API, validation |
| `quartermaster-providers` | LLM provider abstraction (OpenAI, Anthropic, Google, Groq, local) |
| `quartermaster-tools` | Tool definition, registry, built-in tools |
| `quartermaster-nodes` | Node execution protocols and implementations |
| `quartermaster-engine` | Flow execution, traversal, memory, streaming |
| `quartermaster-mcp-client` | MCP protocol client (standalone) |
| `quartermaster-code-runner` | Docker sandboxed code execution (standalone) |

## Documentation

See the [docs/](https://github.com/MindMadeLab/quartermaster-sdk-py/tree/master/docs) directory:

- [Getting Started](../docs/getting-started.md)
- [Graph Building](../docs/graph-building.md)
- [Architecture](../docs/architecture.md)
- [Tools Catalog](../docs/tools-catalog.md)
- [Providers](../docs/providers.md)
- [Security](../docs/security.md)

## License

Apache 2.0
