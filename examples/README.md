# Quartermaster Examples

Progressive examples demonstrating the Quartermaster AI agent framework,
from the simplest agent to a full enterprise-grade multi-department assistant.

## Prerequisites

Install the packages in development mode from the repository root:

```bash
pip install -e quartermaster-graph
pip install -e quartermaster-tools
```

## Examples

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_hello_agent.py` | Simplest possible agent: user input, LLM response, done |
| 02 | `02_decision_flow.py` | Decision node with two branches that merge back together |
| 03 | `03_multi_decision.py` | Multiple decisions in sequence: categorization, urgency check, feedback |
| 04 | `04_sub_graphs.py` | Composable sub-graphs inlined with `.use()` |
| 05 | `05_memory_flow.py` | Customer service scenario with var, text, write/read/update memory |
| 06 | `06_tool_decorator.py` | `@tool()` decorator for creating tools from plain functions |
| 07 | `07_switch_router.py` | Multi-way switch (5 branches) for language routing |
| 08 | `08_reasoning.py` | Reasoning and summarization node types for analytical workflows |
| 09 | `09_parallel_agents.py` | Parallel agent sessions with `SessionManager` |
| 10 | `10_enterprise_agent.py` | Full enterprise agent: department sub-graphs, decisions, IF gates, parallel checks, memory, notifications |
| 11 | `11_nested_control_flow.py` | Whiteboard pattern: parallel fan-out with IF decisions nested inside branches |
| 12 | `12_full_showcase.py` | Kitchen-sink: AI Research Assistant using every major pattern in one graph |

## Running

```bash
# Run any example directly
python examples/01_hello_agent.py

# Examples 01-08 build graphs offline -- no API keys needed
# Example 09 runs parallel sessions with simulated work
# Examples 10-12 compose sub-graphs into complex agent pipelines
```

## Patterns demonstrated

- **Fluent builder**: All graphs use the chainable `Graph("name").start()...end()` API
- **No `.build()` needed**: Access `.nodes` and `.edges` directly on the builder
- **Branching**: `decision()` for multi-way, `if_node()` for boolean, both with `.on(label)`
- **Merging**: `.merge()` collects all branch endpoints back into the main flow
- **Parallel fan-out**: `.parallel()` + `.branch()` for concurrent paths, with nested control flow inside branches
- **Sub-graphs**: `.use(sub_graph)` inlines a sub-graph (accepts `GraphBuilder` or `AgentVersion`)
- **Memory**: `.var()`, `.read_memory()`, `.write_memory()`, `.update_memory()` for state management
- **Text templates**: `.text()` with `{{variable}}` interpolation
- **Reasoning & summarization**: `.reasoning()` and `.summarize()` for analytical pipelines
- **Notifications & logging**: `.notification()`, `.log()`, `.webhook()` for observability
- **Tools**: `@registry.tool()` decorator with automatic type/docstring extraction
- **Parallelism**: `SessionManager` for concurrent agent sessions with thread-based execution
