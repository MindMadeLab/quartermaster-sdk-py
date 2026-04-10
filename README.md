# Quartermaster

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Open-source AI agent orchestration framework. Build, compose, and run multi-step AI agent workflows with any LLM provider.

Built by [MindMade](https://mindmade.si) in Slovenia.

## Packages

| Package | Description | Status |
|---|---|---|
| [`qm-mcp-client`](./qm-mcp-client/) | MCP Protocol client (SSE + Streamable HTTP) | **Ready to publish** |
| [`qm-providers`](./qm-providers/) | Multi-LLM provider abstraction (OpenAI, Anthropic, Google, Groq, xAI) | **Ready to publish** |
| [`qm-code-runner`](./qm-code-runner/) | Docker-based sandboxed code execution (Python, Node, Go, Rust, Deno, Bun) | Alpha |
| [`qm-tools`](./qm-tools/) | Tool definition, registry, and execution framework | Alpha |
| [`qm-nodes`](./qm-nodes/) | 38+ composable node types (LLM, control flow, memory, user interaction) | Alpha |
| [`qm-graph`](./qm-graph/) | Agent graph schema and builder (DAG-based flow definitions) | Alpha |
| [`qm-engine`](./qm-engine/) | Flow execution engine (traversal, branching, memory, streaming) | Alpha |

## Architecture

```
                    +-----------------------------------------+
                    |          Your Application               |
                    +-------------------+---------------------+
                                        |
                    +-------------------v---------------------+
                    |              qm-engine                  |
                    |   Flow execution, traversal, streaming  |
                    +----+----------+---------------+---------+
                         |          |               |
            +------------v--+  +---v--------+  +---v-----------+
            |   qm-graph    |  |  qm-nodes  |  |   qm-tools    |
            |   Schema &    |  |  38+ node  |  |   Registry &  |
            |   builder     |  |  types     |  |   execution   |
            +---------------+  +-----+------+  +-------+-------+
                                     |                 |
                    +----------------v-----------------v------+
                    |             qm-providers                |
                    |   OpenAI, Anthropic, Google, Groq, xAI  |
                    +-----------------------------------------+

            +------------------+    +------------------+
            |  qm-mcp-client   |    |  qm-code-runner  |
            |  (Standalone)    |    |  (Standalone)     |
            +------------------+    +------------------+
```

## Installation

Each package can be installed independently:

```bash
# Standalone packages (no framework dependency)
pip install qm-mcp-client
pip install qm-code-runner

# LLM provider abstraction
pip install qm-providers

# Tool framework
pip install qm-tools

# Node types
pip install qm-nodes

# Graph schema and builder
pip install qm-graph

# Full execution engine
pip install qm-engine
```

## Quick Start

Build and run a simple agent graph:

```python
from qm_graph import GraphBuilder
from qm_engine import FlowRunner
from qm_engine.nodes import SimpleNodeRegistry

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

The `ProviderRegistry` from `qm-providers` manages LLM provider connections:

```python
from qm_providers import ProviderRegistry
from qm_providers.providers import OpenAIProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key="sk-...")

# Auto-infer provider from model name
provider = registry.get_for_model("gpt-4o")
```

For more examples, see the [`examples/`](./examples/) directory.

## Package Documentation

Each package has its own README with detailed documentation:

- [qm-mcp-client](./qm-mcp-client/README.md) -- MCP protocol client
- [qm-code-runner](./qm-code-runner/README.md) -- Docker-based code execution
- [qm-providers](./qm-providers/README.md) -- Multi-LLM provider abstraction
- [qm-tools](./qm-tools/README.md) -- Tool definition and registry
- [qm-nodes](./qm-nodes/README.md) -- Composable node types
- [qm-graph](./qm-graph/README.md) -- Agent graph schema and builder
- [qm-engine](./qm-engine/README.md) -- Flow execution engine

## License

[Apache 2.0](./LICENSE)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, code style, and PR process.

---

Built by [MindMade](https://mindmade.si) in Slovenia.
