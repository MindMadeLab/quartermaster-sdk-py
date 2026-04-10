# Quartermaster Examples

Runnable example scripts demonstrating the core APIs of the Quartermaster
monorepo.  Each script is self-contained and can be executed directly.

## Prerequisites

Install the packages in development mode from the repository root:

```bash
pip install -e quartermaster-graph
pip install -e quartermaster-providers
pip install -e quartermaster-tools
pip install -e quartermaster-tools[web]    # for WebRequestTool (needs httpx)
pip install -e quartermaster-engine
pip install -e quartermaster-mcp-client    # for the MCP example (needs httpx)
```

## Examples

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_simple_graph.py` | Build and inspect a Start -> Instruction -> End graph using `GraphBuilder` |
| 02 | `02_decision_flow.py` | Decision and If-node branching with labelled branches |
| 03 | `03_tool_usage.py` | Define a custom `AbstractTool`, register it in `ToolRegistry`, export JSON Schema |
| 04 | `04_builtin_tools.py` | Use the built-in `ReadFileTool`, `WriteFileTool`, and `WebRequestTool` |
| 05 | `05_provider_setup.py` | Configure `LLMConfig`, use `ProviderRegistry`, infer providers from model names |
| 06 | `06_graph_templates.py` | Use `Templates` factory methods (chatbot, decision tree, RAG, tool agent, etc.) |
| 07 | `07_yaml_roundtrip.py` | Serialize a graph to YAML, reimport it, and verify round-trip fidelity |
| 08 | `08_mcp_client.py` | Connect to an MCP server, list tools, call a tool (sync and async) |
| 09 | `09_full_pipeline.py` | End-to-end: build graph + register node executors + run `FlowRunner` |
| 10 | `10_custom_node.py` | Create a custom node type with a custom `NodeExecutor` |

## Running

```bash
# Run any example directly
python examples/01_simple_graph.py

# Examples that need external services will print a helpful message
# if the service is unavailable (e.g., 08_mcp_client.py)
```

## Notes

- Examples 01-07 run fully offline with no API keys required.
- Example 08 requires a running MCP server (prints a message if unavailable).
- Examples 09 and 10 use stub executors so they run without real LLM providers.
- All examples handle missing dependencies with `try/except ImportError`.
