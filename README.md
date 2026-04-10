# Quartermaster

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Open-source AI agent orchestration framework. Build, compose, and run multi-step AI agent workflows with any LLM provider.

Built by [MindMade](https://mindmade.si) in Slovenia.

## Packages

| Package | Description | Status |
|---|---|---|
| [`quartermaster-mcp-client`](./quartermaster-mcp-client/) | MCP Protocol client (SSE + Streamable HTTP) | **Ready to publish** |
| [`quartermaster-providers`](./quartermaster-providers/) | Multi-LLM provider abstraction (OpenAI, Anthropic, Google, Groq, xAI) | **Ready to publish** |
| [`quartermaster-code-runner`](./quartermaster-code-runner/) | Docker-based sandboxed code execution (Python, Node, Go, Rust, Deno, Bun) | Alpha |
| [`quartermaster-tools`](./quartermaster-tools/) | Tool definition, registry, and execution framework | Alpha |
| [`quartermaster-nodes`](./quartermaster-nodes/) | 38+ composable node types (LLM, control flow, memory, user interaction) | Alpha |
| [`quartermaster-graph`](./quartermaster-graph/) | Agent graph schema and builder (DAG-based flow definitions) | Alpha |
| [`quartermaster-engine`](./quartermaster-engine/) | Flow execution engine (traversal, branching, memory, streaming) | Alpha |

## Architecture

```
                    +-----------------------------------------+
                    |          Your Application               |
                    +-------------------+---------------------+
                                        |
                    +-------------------v---------------------+
                    |              quartermaster-engine                  |
                    |   Flow execution, traversal, streaming  |
                    +----+----------+---------------+---------+
                         |          |               |
            +------------v--+  +---v--------+  +---v-----------+
            |   quartermaster-graph    |  |  quartermaster-nodes  |  |   quartermaster-tools    |
            |   Schema &    |  |  38+ node  |  |   Registry &  |
            |   builder     |  |  types     |  |   execution   |
            +---------------+  +-----+------+  +-------+-------+
                                     |                 |
                    +----------------v-----------------v------+
                    |             quartermaster-providers                |
                    |   OpenAI, Anthropic, Google, Groq, xAI  |
                    +-----------------------------------------+

            +------------------+    +------------------+
            |  quartermaster-mcp-client   |    |  quartermaster-code-runner  |
            |  (Standalone)    |    |  (Standalone)     |
            +------------------+    +------------------+
```

## Installation

Each package can be installed independently:

```bash
# Standalone packages (no framework dependency)
pip install quartermaster-mcp-client
pip install quartermaster-code-runner

# LLM provider abstraction
pip install quartermaster-providers

# Tool framework
pip install quartermaster-tools

# Node types
pip install quartermaster-nodes

# Graph schema and builder
pip install quartermaster-graph

# Full execution engine
pip install quartermaster-engine
```

## Quick Start

Build and run a simple agent graph:

```python
from quartermaster_graph import GraphBuilder
from quartermaster_engine import FlowRunner
from quartermaster_engine.nodes import SimpleNodeRegistry

# Build a simple agent graph
graph = (
    GraphBuilder("My Agent")
    .start()
    .instruction("Analyze this text", model="gpt-4o")
    .end()
    .build()
)

# Set up the node registry (registers node executors by type)
node_registry = SimpleNodeRegistry()

# Run the graph
runner = FlowRunner(graph=graph, node_registry=node_registry)
result = runner.run("Hello world")
print(result.final_output)
```

The `ProviderRegistry` from `quartermaster-providers` manages LLM provider connections:

```python
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.providers import OpenAIProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key="sk-...")

# Auto-infer provider from model name
provider = registry.get_for_model("gpt-4o")
```

For more examples, see the [`examples/`](./examples/) directory.

## Package Documentation

Each package has its own README with detailed documentation:

- [quartermaster-mcp-client](./quartermaster-mcp-client/README.md) -- MCP protocol client
- [quartermaster-code-runner](./quartermaster-code-runner/README.md) -- Docker-based code execution
- [quartermaster-providers](./quartermaster-providers/README.md) -- Multi-LLM provider abstraction
- [quartermaster-tools](./quartermaster-tools/README.md) -- Tool definition and registry
- [quartermaster-nodes](./quartermaster-nodes/README.md) -- Composable node types
- [quartermaster-graph](./quartermaster-graph/README.md) -- Agent graph schema and builder
- [quartermaster-engine](./quartermaster-engine/README.md) -- Flow execution engine

## License

[Apache 2.0](./LICENSE)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, code style, and PR process.

---

Built by [MindMade](https://mindmade.si) in Slovenia.
