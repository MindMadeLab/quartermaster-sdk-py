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
| 05 | `05_memory_flow.py` | Variable storage, write-memory, and read-memory nodes |
| 06 | `06_tool_decorator.py` | `@tool()` decorator for creating tools from plain functions |
| 07 | `07_switch_router.py` | Multi-way switch (5 branches) for language routing |
| 08 | `08_reasoning.py` | Reasoning and summarization node types for analytical workflows |
| 09 | `09_parallel_agents.py` | Parallel agent sessions with `SessionManager` |
| 10 | `10_enterprise_agent.py` | Full showcase: sub-graphs, decisions, if-nodes, memory, logging |

## Running

```bash
# Run any example directly
python examples/01_hello_agent.py

# Examples 01-08 build graphs offline -- no API keys needed
# Example 09 runs parallel sessions with simulated work
# Example 10 composes sub-graphs into a complex enterprise agent
```

## Patterns demonstrated

- **Fluent builder**: All graphs use the chainable `Graph("name").start()...end()` API
- **Branching**: `decision()` for multi-way, `if_node()` for boolean, both with `.on(label)`
- **Merging**: `.merge()` collects all branch endpoints back into the main flow
- **Sub-graphs**: `.use(sub_graph)` inlines a built sub-graph, stripping its START/END
- **Memory**: `VAR`, `READ_MEMORY`, `WRITE_MEMORY` node types for state management
- **Tools**: `@registry.tool()` decorator with automatic type/docstring extraction
- **Parallelism**: `SessionManager` for concurrent agent sessions with thread-based execution
