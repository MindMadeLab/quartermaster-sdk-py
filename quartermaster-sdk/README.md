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
import quartermaster_sdk as qm

qm.configure(
    provider="ollama",
    base_url="http://localhost:11434",   # or set $OLLAMA_HOST
    default_model="gemma4:26b",
)

# Graph() auto-creates Start; .end() / .build() are both optional when running via qm.run().
result = qm.run(qm.Graph("chat").user().agent(), "Pozdravljen, koliko je ura?")
print(result.text)
```

## Single-shot helpers (no graph visible)

```python
# prompt → str
reply = qm.instruction(system="Respond in Slovenian.", user="Pozdravljen!")

# prompt → Pydantic model (typed JSON extraction)
from pydantic import BaseModel

class Classification(BaseModel):
    category: str
    priority: str

data = qm.instruction_form(Classification, system="Classify.", user=email_body)
```

## Reading specific node outputs with `capture_as=`

```python
graph = (
    qm.Graph("enrich")
    .agent("Research", tools=[...], capture_as="notes")
    .instruction_form(CustomerData, system="Extract.", capture_as="data")
)
result = qm.run(graph, "VT-Treyd Slovenija")
result["notes"].output_text    # agent's free-text research
result["data"].output_text     # extracted JSON
```

## Streaming

```python
for chunk in qm.run.stream(qm.Graph("chat").user().agent(), "Tell me a story"):
    if chunk.type == "token":
        print(chunk.content, end="", flush=True)
    elif chunk.type == "done":
        final = chunk.result    # qm.Result
```

## Quick Start (cloud provider)

```python
agent = (
    qm.Graph("My Agent")
    .user("What can I help you with?")
    .instruction("Respond", model="gpt-4o", system_instruction="You are a helpful assistant.")
)
result = qm.run(agent, "How does photosynthesis work?")
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
