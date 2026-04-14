# Quartermaster Examples

Progressive examples demonstrating the Quartermaster AI agent framework,
from the simplest agent to a full enterprise-grade multi-department assistant.

## Setup

```bash
# Clone the repo
git clone git@github.com:MindMadeLab/quartermaster-sdk-py.git
cd quartermaster-sdk-py

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install all packages
uv sync

# Run any example
uv run examples/01_hello_agent.py
```

All examples (except `run_interactive.py`) build and validate graphs offline
-- no API keys or LLM calls needed.

### Running the interactive demo

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Run
uv run examples/run_interactive.py
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

## Patterns Demonstrated

- **Fluent builder**: `Graph("name").start().user("Input")...end()` -- chainable API
- **User input**: `.user()` pauses flow and waits for human input
- **Decision routing**: `decision()` picks ONE branch via LLM
- **If/else**: `if_node()` with safe AST-evaluated boolean expressions
- **Switch**: `switch()` for multi-way expression branching
- **Parallel fan-out**: `parallel()` + `branch()` for concurrent paths, joined with `static_merge()`
- **Sub-graphs**: `.use(sub_graph)` inlines a reusable sub-graph
- **Memory**: `var()`, `write_memory()`, `read_memory()`, `update_memory()` for state
- **Templates**: `text()` with `{{variable}}` Jinja2 interpolation
- **User forms**: `user_form()` with typed parameters for structured data collection
- **Reasoning**: `reasoning()` and `summarize()` for analytical pipelines
- **Tools**: `@tool()` and `@registry.tool()` decorator with automatic type and docstring extraction
- **Local LLMs**: `register_local("ollama")` for self-hosted inference
