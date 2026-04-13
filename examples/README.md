# Quartermaster Examples

Progressive examples demonstrating the Quartermaster AI agent framework,
from the simplest agent to a full enterprise-grade multi-department assistant.

## Prerequisites

```bash
# With uv (recommended)
uv pip install -e quartermaster-graph -e quartermaster-tools \
  -e quartermaster-providers -e quartermaster-nodes -e quartermaster-engine

# Or install the SDK
pip install -e quartermaster-sdk
```

## Examples

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_hello_agent.py` | Simplest possible agent: user input, LLM response, done |
| 02 | `02_decision_flow.py` | Decision node routing to different branches |
| 03 | `03_multi_decision.py` | Multiple decisions in sequence: categorization, urgency, feedback |
| 04 | `04_sub_graphs.py` | Composable sub-graphs inlined with `.use()` |
| 05 | `05_memory_flow.py` | Customer service with var, text templates, write/read/update memory |
| 06 | `06_tool_decorator.py` | `@tool()` decorator for creating tools from plain functions |
| 07 | `07_switch_router.py` | Multi-way switch (5 branches) for language routing |
| 08 | `08_reasoning.py` | Reasoning and summarization nodes for analytical workflows |
| 09 | `09_parallel_agents.py` | Parallel agent sessions with `SessionManager` |
| 10 | `10_enterprise_agent.py` | Enterprise agent: department sub-graphs, decisions, IF gates, parallel, memory |
| 11 | `11_nested_control_flow.py` | Parallel fan-out with IF decisions nested inside branches |
| 12 | `12_full_showcase.py` | Kitchen-sink: AI Research Assistant using every major pattern |
| 13 | `13_orchestrator.py` | Agent orchestrator with spawn/collect session tools |
| 14 | `14_local_providers.py` | Local LLM providers: Ollama, vLLM, mixed cloud+local |
| 15 | `15_user_forms.py` | User forms, variables, IF conditions, and Jinja2 templates |
| 16 | `16_courtroom_debate.py` | Multi-round courtroom drama: parallel prep, loop-back debate, judge verdict |
| -- | `run_interactive.py` | Real LLM demo with Anthropic/OpenAI auto-detection and decision routing |

## Running

```bash
# Run any example (no API keys needed -- these build graphs offline)
python examples/01_hello_agent.py

# Or with uv
uv run examples/01_hello_agent.py
```

All examples build and validate graphs without calling any LLM API.
They print graph structure, node counts, and edge lists to verify correctness.

## Patterns Demonstrated

- **Fluent builder**: `Graph("name").start().user("Input")...end()` -- chainable API, no `.build()` needed
- **User input**: Every graph starts with `.user()` after `.start()` to collect input
- **Decision routing**: `decision()` picks ONE branch via LLM -- no merge needed
- **If/else**: `if_node()` with safe AST-evaluated boolean expressions -- no merge needed
- **Switch**: `switch()` for multi-way expression branching -- no merge needed
- **Parallel fan-out**: `parallel()` + `branch()` for concurrent paths, joined with `static_merge()`
- **Sub-graphs**: `.use(sub_graph)` inlines a reusable sub-graph
- **Memory**: `var()`, `write_memory()`, `read_memory()`, `update_memory()` for state
- **Templates**: `text()` with `{{variable}}` Jinja2 interpolation
- **User forms**: `user_form()` with typed parameters for structured data collection
- **Reasoning**: `reasoning()` and `summarize()` for analytical pipelines
- **Tools**: `@tool()` and `@registry.tool()` decorator with automatic type and docstring extraction
- **Local LLMs**: `register_local("ollama")` for self-hosted inference
