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

### API Keys

Create a `.env` file at the repo root (see `.env.example`):
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Or export them directly: `export ANTHROPIC_API_KEY=...`

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
| 08 | `08_iterative_refinement.py` | Writer-critic loop: 3 rounds of draft, review, and improve |
| 09 | `09_parallel_agents.py` | Parallel agent sessions with `SessionManager` |
| 10 | `10_enterprise_agent.py` | Enterprise agent: department sub-graphs, decisions, IF gates, parallel, memory |
| 11 | `11_nested_control_flow.py` | Parallel fan-out with IF decisions nested inside branches |
| 12 | `12_full_showcase.py` | Kitchen-sink: AI Research Assistant using every major pattern |
| 13 | `13_orchestrator.py` | Agent orchestrator with spawn/collect session tools |
| 14 | `14_local_providers.py` | Local LLM providers: Ollama, vLLM, mixed cloud+local |
| 15 | `15_user_forms.py` | User forms, variables, IF conditions, and Jinja2 templates |
| 16 | `16_courtroom_debate.py` | Multi-round courtroom drama: loop-back debate across 3 providers, judge verdict |
| 17 | `17_tool_agent.py` | Custom tools with `@tool()`, JSON schema export, tool-aware LLM agent |
| 18 | `18_streaming_events.py` | FlowRunner direct usage with event streaming (no API keys needed) |
| 19 | `19_data_pipeline.py` | Multi-stage data pipeline with parallel analysis and mixed providers |
| 20 | `20_compliance_guard.py` | PII detection/redaction, EU AI Act risk classification, audit logging |
| 21 | `21_mcp_client.py` | MCP protocol client: discover tools, bridge to Quartermaster graphs |
| 22 | `22_ollama_local.py` | Local Ollama with Gemma model, streaming, no cloud API keys |
| 23 | `23_vision.py` | Image vision analysis with multimodal LLM pipeline |
| -- | `run_interactive.py` | Interactive stdin conversation loop with Ctrl+C exit |

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
- **Summarize**: `summarize()` for condensing conversation history
- **Tools**: `@tool()` and `@registry.tool()` decorator with automatic type and docstring extraction
- **Local LLMs**: `register_local("ollama")` for self-hosted inference
- **Loop-back edges**: `connect()` for iterative/cyclic flows (examples 08, 16)
- **Output control**: `show_output=False` hides internal nodes from display
- **Streaming**: Token-by-token real-time output from all LLM nodes
- **Interactive mode**: `run_graph()` without `user_input` prompts stdin at User nodes
- **Multi-provider**: Different LLMs for different nodes (Anthropic + OpenAI + Groq + xAI)
- **`run_graph()`**: One-line execution with auto-detected provider and streaming
