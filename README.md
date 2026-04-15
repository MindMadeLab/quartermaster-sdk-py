# Quartermaster

[![CI](https://github.com/MindMadeLab/quartermaster-sdk-py/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/MindMadeLab/quartermaster-sdk-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/quartermaster-sdk.svg)](https://pypi.org/project/quartermaster-sdk/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Modular AI agent orchestration framework. Build agent workflows as directed graphs, wire them with a fluent Python API, and run them with any LLM provider.

Built by [MindMade](https://mindmade.io) in Slovenia.

## Install

```bash
# Everything (recommended)
pip install quartermaster-sdk

# With a specific LLM provider
pip install quartermaster-sdk[openai]
pip install quartermaster-sdk[anthropic]

# From source (for development or running examples)
git clone https://github.com/MindMadeLab/quartermaster-sdk-py.git
cd quartermaster-sdk-py
uv sync
```

## Quick Start

The simplest possible graph — running against a local Ollama in four lines
(no `.start()`, no `.end()`, no `.build()`, no `FlowRunner` import):

```python
import quartermaster_sdk as qm

qm.configure(provider="ollama", base_url="http://localhost:11434", default_model="gemma4:26b")

result = qm.run(qm.Graph("chat").user().agent(), "Pozdravljen, koliko je ura?")
print(result.text)
```

`qm.run()` accepts the builder directly and finalises it internally —
`.build()` is only needed when you want the validated `GraphSpec` for
serialisation or inspection. For single-shot calls skip the graph entirely:

```python
reply = qm.instruction(system="Respond in Slovenian.", user="Pozdravljen!")
# reply is a str.
```

For typed JSON extraction:

```python
from pydantic import BaseModel

class Classification(BaseModel):
    category: str
    priority: str

data = qm.instruction_form(Classification, system="Classify.", user=email_body)
# data is a Classification instance.
```

For richer flows you keep the explicit per-node configuration:

```python
agent = (
    qm.Graph("My Agent")
    .user("What can I help you with?")
    .instruction("Respond", model="gpt-4o", system_instruction="You are a helpful assistant.")
)
result = qm.run(agent, "...")
```

### Reading specific node outputs with `capture_as=`

Attach a name to any node and read its output from `result.captures`:

```python
graph = (
    qm.Graph("enrich")
    .agent("Research", tools=[...], capture_as="notes")
    .instruction_form(CustomerData, system="Extract.", capture_as="data")
)
result = qm.run(graph, "VT-Treyd Slovenija")
result["notes"].output_text    # agent's free-text research
result["data"].output_text     # form-parsed JSON
```

### Sync `OllamaProvider.chat()` for non-graph callers

For email classification, OCR pipelines, Celery workers, Django views — anything
where you'd rather call an LLM than build a graph — use the synchronous native
shim. Talks Ollama's `/api/chat` directly via `httpx`, no `async_to_sync` wrapper:

```python
from quartermaster_providers.providers.local import OllamaProvider

provider = OllamaProvider(default_model="gemma4:26b")  # honours $OLLAMA_HOST
result = provider.chat(
    messages=[
        {"role": "system", "content": "Respond in Slovenian. Keep it short."},
        {"role": "user", "content": "Pozdravljen, koliko je ura?"},
    ],
    max_output_tokens=128,        # honoured — capped at Ollama's `num_predict`
    thinking_level="off",         # off / low / medium / high
)
print(result.content)             # promoted from `reasoning` if `content` is empty
print(result.tool_calls)          # list[ToolCall]
print(result.usage)               # {prompt_tokens, completion_tokens, total_tokens}
```

`ServiceUnavailableError` raises on connection failures (instead of silently
returning an empty result), and `ProviderError` on HTTP errors with status code
attached.

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

### Custom Tools with @tool()

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

# Call it directly
result = get_weather(city="Amsterdam")

# Export JSON Schema for LLM function calling
schema = get_weather.info().to_input_schema()

# Or register in a ToolRegistry and export all at once
from quartermaster_tools import ToolRegistry

registry = ToolRegistry()
registry.register(get_weather)
schemas = registry.to_json_schema()
```

See [`examples/`](./examples/) for runnable examples covering every pattern.

## Running Your Graph

```python
from quartermaster_engine import run_graph

# Run — each node uses the provider/model it declares
run_graph(agent, user_input="What is quantum computing?")

# Interactive mode — pauses at User nodes and prompts stdin
run_graph(agent)  # no user_input = interactive
```

Nodes declare their own provider and model:
```python
.instruction("Respond", model="claude-haiku-4-5-20251001", provider="anthropic", ...)
.instruction("Fast reply", model="llama-3.3-70b-versatile", provider="groq", ...)
.instruction("Local", model="gemma4:26b", provider="ollama", ...)
```

Set up your API keys in a `.env` file at the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
```

Output streams token-by-token in real time. Use `show_output=False` on nodes
to hide internal steps (variables, conditions) from the output.

## Packages

| Package | Description |
|---------|-------------|
| [`quartermaster-sdk`](./quartermaster-sdk/) | Meta-package -- installs all core packages |
| [`quartermaster-graph`](./quartermaster-graph/) | Graph schema, fluent builder API, validation |
| [`quartermaster-providers`](./quartermaster-providers/) | LLM provider abstraction (OpenAI, Anthropic, Google, Groq, Ollama, vLLM) |
| [`quartermaster-tools`](./quartermaster-tools/) | Tool definition, registry, built-in tools |
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

- **Graph** -- A directed graph (supports cycles via `connect()` for loops) of nodes and edges. Built with the fluent `Graph("name").start().user("Input")...end()` API.
- **GraphSpec** -- The serializable graph model (`GraphSpec` in quartermaster-graph). `qm.run(graph, ...)` finalises the builder for you; explicit `Graph.build()` only matters when you want the validated spec to serialise / inspect. `AgentGraph` remains as a deprecated backward-compat alias.
- **User Node** -- Every graph starts with `.user()` after `.start()` to collect user input.
- **Nodes** -- Units of work: LLM calls, decisions, user input, memory, tools, templates.
- **Edges** -- Directed connections between nodes. Decision/IF/Switch edges carry labels.
- **Thoughts** -- Runtime containers that carry text and variables (metadata) between nodes.
- **Memory** -- Flow-scoped persistent storage accessible from any node via `write_memory`/`read_memory`.
- **Providers** -- Pluggable LLM backends. Model name auto-resolves to the right provider.
- **Tools** -- `@tool()` decorator for custom tools, built-in tools, JSON Schema export via `tool.info().to_input_schema()`.
- **Loops** -- `connect("Continue", "Start")` creates back-edges for iterative flows.
- **Streaming** -- Token-by-token output from LLM nodes in real time.
- **Multi-provider** -- Different LLM providers for different nodes in the same graph.

### Branching Rules

| Node Type | Behavior | Merge Needed? |
|-----------|----------|---------------|
| `decision()` | LLM picks ONE branch | No |
| `if_node()` | Boolean expression picks ONE branch | No |
| `switch()` | Expression picks ONE branch | No |
| `parallel()` | ALL branches run concurrently | Yes -- use `static_merge()` |
| `connect()` | Manual edge by node name | Creates loops/cycles |

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](./docs/getting-started.md) | Installation and first agent |
| [Graph Building](./docs/graph-building.md) | Builder API, node types, patterns |
| [Architecture](./docs/architecture.md) | System overview and data flow |
| [Providers](./docs/providers.md) | LLM providers including local (Ollama, vLLM) |
| [Tools Catalog](./docs/tools-catalog.md) | All built-in tools with parameters |
| [Engine](./docs/engine.md) | Execution engine internals |
| [Security](./docs/security.md) | Safe eval, sandboxing, API key management |
| [Node Reference](./docs/nodes/README.md) | Detailed node documentation by category |

## Development

```bash
# Clone
git clone https://github.com/MindMadeLab/quartermaster-sdk-py.git
cd quartermaster-sdk-py

# Install everything (uv workspace -- one command)
uv sync

# Run an example
uv run examples/01_hello_agent.py

# Run tests for a single package
uv run pytest quartermaster-graph/tests/

# Run all tests
uv run pytest quartermaster-graph/tests/ quartermaster-tools/tests/ quartermaster-engine/tests/
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development guide.

## License

[Apache 2.0](./LICENSE) -- Built by [MindMade](https://mindmade.io) in Slovenia.
