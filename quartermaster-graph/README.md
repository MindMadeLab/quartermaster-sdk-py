# quartermaster-graph

Framework-agnostic graph schema for defining AI agent workflows as directed graphs.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-graph)](https://pypi.org/project/quartermaster-graph/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **Pydantic-based models**: `Agent`, `GraphSpec`, `GraphNode`, `GraphEdge` with full validation
- **GraphBuilder** fluent API for programmatic graph construction (`Graph` convenience alias)
- **40+ node types** covering LLM, control flow, data, user interaction, memory, and utility
- **YAML/JSON serialization** with round-trip fidelity
- **Graph validation**: start/end nodes, cycle detection, orphan detection, edge label checks
- **Graph traversal**: topological sort, path finding, successor/predecessor queries
- **Pre-built templates**: simple chat, decision tree, multi-agent supervisor, and more
- **Typed metadata schemas** for each node type

### New in v0.5.0

- **`program_runner(program=<callable>)`** -- pass a `@tool()`-decorated function
  directly instead of its name string; the graph builder auto-registers it.
  Parity with `.agent(tools=[...])`.

## Installation

```bash
pip install quartermaster-graph
```

**Dependencies**: `pydantic>=2.0`, `pyyaml>=6.0`

## Quick Start

### Build a Graph with the Fluent API

```python
from quartermaster_graph import Graph

graph = (
    Graph("Customer Support Agent")
    .user("How can I help you?")
    .instruction("Classify intent", model="gpt-4o", system_instruction="Classify the user's intent.")
    .decision("Route by intent", options=["billing", "technical", "general"])
    .on("billing").instruction("Handle billing", model="gpt-4o").end()
    .on("technical").instruction("Handle technical", model="gpt-4o").end()
    .on("general").instruction("Handle general", model="gpt-4o").end()
    .end()
)
```

`Graph` is a convenience alias for `GraphBuilder`. Both `.build()` and `.to_graph()` return a `GraphSpec`:

```python
from quartermaster_graph import GraphBuilder

builder = GraphBuilder("My Agent")
builder.instruction("Process", model="gpt-4o").end()

spec = builder.build()       # returns GraphSpec
spec = builder.to_graph()    # same thing
```

### Load a Graph from YAML

```yaml
# agent.yaml
agent_id: "550e8400-e29b-41d4-a716-446655440000"
start_node_id: "00000000-0000-0000-0000-000000000001"
nodes:
  - id: "00000000-0000-0000-0000-000000000001"
    type: "Start1"
    name: "Start"
  - id: "00000000-0000-0000-0000-000000000002"
    type: "User1"
    name: "Input"
  - id: "00000000-0000-0000-0000-000000000003"
    type: "Instruction1"
    name: "Process"
    metadata:
      llm_system_instruction: "Analyze user input"
      llm_model: "gpt-4o"
  - id: "00000000-0000-0000-0000-000000000004"
    type: "End1"
    name: "End"
edges:
  - source_id: "00000000-0000-0000-0000-000000000001"
    target_id: "00000000-0000-0000-0000-000000000002"
  - source_id: "00000000-0000-0000-0000-000000000002"
    target_id: "00000000-0000-0000-0000-000000000003"
  - source_id: "00000000-0000-0000-0000-000000000003"
    target_id: "00000000-0000-0000-0000-000000000004"
```

```python
from quartermaster_graph import from_yaml

with open("agent.yaml") as f:
    spec = from_yaml(f.read())
```

## API Reference

### Core Models

| Model | Description |
|-------|-------------|
| `Agent` | Top-level agent definition (name, description, tags) |
| `GraphSpec` | Graph definition: nodes, edges, start node, features |
| `GraphNode` | A node with type, metadata, traversal config, error handling |
| `GraphEdge` | Directed edge with optional label and routing points |
| `NodePosition` | Visual position for editor rendering |

> `AgentGraph` and `AgentVersion` both exist as deprecated backward-compatibility aliases for `GraphSpec`.

### GraphSpec Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_node(node_id)` | `GraphNode \| None` | Find a node by ID |
| `get_start_node()` | `GraphNode \| None` | Get the start node |
| `get_successors(node_id)` | `list[GraphNode]` | All successor nodes |
| `get_predecessors(node_id)` | `list[GraphNode]` | All predecessor nodes |
| `get_edges_from(node_id)` | `list[GraphEdge]` | All edges from a node |
| `get_edges_to(node_id)` | `list[GraphEdge]` | All edges to a node |

### GraphBuilder Methods

**Start / End**

| Method | Returns | Description |
|--------|---------|-------------|
| `.start()` | `GraphBuilder` | Add a Start node (auto-inserted since v0.2.0 — rarely needed explicitly) |
| `.end()` | `GraphBuilder` | Add an End node (or close a branch) |

**LLM Nodes**

| Method | Returns | Description |
|--------|---------|-------------|
| `.instruction(name, model, provider, temperature, system_instruction, **kwargs)` | `GraphBuilder` | LLM text generation, no tools, streams response |
| `.decision(name, model, provider, temperature, prefix_message, suffix_message, options, **kwargs)` | `GraphBuilder` | LLM picks one path via `pick_path` tool (non-streaming) |

| `.summarize(name, model, provider, temperature, system_instruction, **kwargs)` | `GraphBuilder` | LLM condenses conversation history |
| `.agent(name, model, provider, system_instruction, tools, max_iterations, **kwargs)` | `GraphBuilder` | Agentic loop with tools, up to `max_iterations` |
| `.vision(name, model, provider, system_instruction, **kwargs)` | `GraphBuilder` | Image vision/analysis node |
| `.merge(name, model, provider, temperature, system_instruction, prefix_message, suffix_message, **kwargs)` | `GraphBuilder` | LLM merges parallel branch outputs |

**Control Flow**

| Method | Returns | Description |
|--------|---------|-------------|
| `.on(label)` | `BranchBuilder` | Start a named branch for a decision/if/switch |
| `.if_node(name, expression)` | `GraphBuilder` | Conditional branch via expression, no LLM |
| `.static_decision(name, expression)` | `GraphBuilder` | Expression-based branching, no LLM |
| `.user_decision(name)` | `GraphBuilder` | User picks which path to follow |
| `.switch(name, cases, default_edge_id)` | `GraphBuilder` | Multi-way switch, first matching case wins |
| `.parallel(name)` | `GraphBuilder` | Start a parallel fan-out |
| `.branch()` | `BranchBuilder` | Start a parallel branch |
| `.static_merge(name, text)` | `GraphBuilder` | Merge parallel branches without LLM |
| `.break_node(name, targets)` | `GraphBuilder` | Stop backward message collection |

**User Interaction**

| Method | Returns | Description |
|--------|---------|-------------|
| `.user(name, prompts)` | `GraphBuilder` | Pause flow for user input |
| `.user_form(name, parameters)` | `GraphBuilder` | Show a structured form to the user |

**Data Nodes**

| Method | Returns | Description |
|--------|---------|-------------|
| `.static(name, text)` | `GraphBuilder` | Fixed text content, no LLM |
| `.code(name, code, filename)` | `GraphBuilder` | Code execution node |
| `.text(name, template)` | `GraphBuilder` | Jinja2 template rendering |
| `.var(name, variable, expression)` | `GraphBuilder` | Evaluate expression, store as variable |
| `.text_to_variable(name, variable, source)` | `GraphBuilder` | Convert text output to a variable |
| `.program_runner(name, program, **kwargs)` | `GraphBuilder` | Run a program/tool inline |

**Memory**

| Method | Returns | Description |
|--------|---------|-------------|
| `.read_memory(name, memory_name, memory_type, variable_names)` | `GraphBuilder` | Read from persistent memory |
| `.write_memory(name, memory_name, memory_type, variables)` | `GraphBuilder` | Write to persistent memory |
| `.update_memory(name, memory_name, memory_type, variables)` | `GraphBuilder` | Update existing memory variables |
| `.flow_memory(name, memory_name, initial_data)` | `GraphBuilder` | Define flow-scoped memory |
| `.user_memory(name, memory_name, initial_data)` | `GraphBuilder` | Define user-scoped persistent memory |

**Composition / Utility**

| Method | Returns | Description |
|--------|---------|-------------|
| `.sub_agent(name, graph_id)` | `GraphBuilder` | Call another agent graph synchronously |
| `.use(sub_graph)` | `GraphBuilder` | Inline a sub-graph (accepts `GraphSpec` or `GraphBuilder`) |
| `.comment(name, text)` | `GraphBuilder` | Documentation-only node, no runtime logic |
| `.allowed_agents(*agent_ids)` | `GraphBuilder` | Restrict which sub-agents can be spawned |
| `.node(node_type, name, metadata, **kwargs)` | `GraphBuilder` | Add any node type generically |
| `.edge(source_id, target_id, label, is_main)` | `GraphBuilder` | Manually add an edge |
| `.connect(from_name, to_name, label)` | `GraphBuilder` | Create an edge between two nodes by name |

**Build / Export**

| Method | Returns | Description |
|--------|---------|-------------|
| `.build(validate=True)` | `GraphSpec` | Build the graph, optionally validate |
| `.to_graph(validate=True, agent_id=None)` | `GraphSpec` | Build with optional explicit agent ID |
| `.to_agent(validate=True)` | `Agent` | Export as a full `Agent` model |

### Node Configuration Kwargs

All builder methods accept optional keyword arguments:

| Kwarg | Type | Description |
|-------|------|-------------|
| `traverse_in` | `TraverseIn` | When to execute: `AWAIT_FIRST` or `AWAIT_ALL` |
| `traverse_out` | `TraverseOut` | Which successors to trigger |
| `thought_type` | `ThoughtType` | How to build conversation context |
| `message_type` | `MessageType` | What message role to use |
| `show_output` | `bool` | Whether to display this node's output (default `True`) |
| `error_handling` | `ErrorStrategy` | What to do on failure |

### Loops and Cycles

Use `connect()` to create back-edges for iterative flows:

```python
agent = (
    Graph("Refiner")
    .user("Input")
    .var("Init", variable="round", expression="1")
    .text("Header", template="Round {{round}}", traverse_in=TraverseIn.AWAIT_FIRST)
    .instruction("Process", system_instruction="Improve the text")
    .var("Increment", variable="round", expression="round + 1")
    .if_node("Done?", expression="round > 3")
    .on("true").text("Done", template="Complete").end()
    .on("false").text("Continue", template="", show_output=False).end()
    .end()
)
agent.connect("Continue", "Header", label="loop")
graph = agent.build(validate=False)  # skip validation for intentional cycles
```

### Validation

```python
from quartermaster_graph import validate_graph

errors = validate_graph(agent_graph)
for err in errors:
    print(f"[{err.severity}] {err.code}: {err.message}")
```

Validation checks:
- Exactly one Start node, at least one End node
- All edge source/target IDs reference existing nodes
- Orphan detection (unreachable from Start)
- Cycle detection (DAG property)
- Decision/If/Switch nodes have proper edge labels

### Serialization

```python
from quartermaster_graph import to_json, from_json, to_yaml, from_yaml, json_schema

# JSON round-trip
data = to_json(agent_graph)
restored = from_json(data)

# YAML round-trip
yaml_str = to_yaml(agent_graph)
restored = from_yaml(yaml_str)

# JSON Schema for cross-language validation
schema = json_schema()
```

### Traversal Utilities

```python
from quartermaster_graph import (
    get_start_node, get_successors, get_predecessors,
    get_path, topological_sort, find_merge_points, find_decision_points,
)

start = get_start_node(agent_graph)
ordered = topological_sort(agent_graph)
path = get_path(agent_graph, start_id, end_id)
```

### Enums

| Enum | Values |
|------|--------|
| `NodeType` | 40+ types: `START`, `END`, `INSTRUCTION`, `DECISION`, `IF`, `SWITCH`, `AGENT`, `USER`, `USER_FORM`, `USER_DECISION`, `CODE`, `MERGE`, `STATIC`, `STATIC_MERGE`, `STATIC_DECISION`, `VAR`, `TEXT`, `SUMMARIZE`, `SUB_ASSISTANT`, `BREAK`, `COMMENT`, etc. |
| `TraverseIn` | `AWAIT_ALL`, `AWAIT_FIRST` |
| `TraverseOut` | `SPAWN_ALL`, `SPAWN_NONE`, `SPAWN_START`, `SPAWN_PICKED` |
| `ThoughtType` | `SKIP`, `NEW`, `NEW_HIDDEN`, `NEW_COLLAPSED`, `INHERIT`, `CONTINUE`, `EDIT_OR_NEW`, `EDIT_SAME`, `APPEND`, `USE_PREVIOUS`, etc. |
| `MessageType` | `AUTOMATIC`, `USER`, `ASSISTANT`, `SYSTEM`, `TOOL`, `VARIABLE` |
| `ErrorStrategy` | `STOP`, `RETRY`, `SKIP`, `CONTINUE`, `CUSTOM` |
| `ExceptionResolution` | `RETRY`, `BREAK`, `CONTINUE` |

## Integration with Sibling Packages

```python
# Build a graph (quartermaster-graph)
from quartermaster_graph import Graph

graph = Graph("Agent").user("Input").instruction("Process").end()

# Execute it (quartermaster-engine)
from quartermaster_engine import FlowRunner

runner = FlowRunner(graph=graph.build(), node_registry=registry)
result = runner.run("Hello!")
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
