# quartermaster-graph

Framework-agnostic graph schema for defining AI agent workflows as directed acyclic graphs.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-graph)](https://pypi.org/project/quartermaster-graph/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **Pydantic-based models**: Agent, AgentVersion, GraphNode, GraphEdge with full validation
- **GraphBuilder** fluent API for programmatic graph construction (`Graph` alias)
- **40 node types** covering LLM, control flow, data, user interaction, memory, and utility
- **YAML/JSON serialization** with round-trip fidelity
- **Graph validation**: start/end nodes, cycle detection, orphan detection, edge label checks
- **Semantic versioning**: create, bump, fork, and diff agent versions
- **Graph traversal**: topological sort, path finding, successor/predecessor queries
- **Pre-built templates**: simple chat, decision tree, multi-agent supervisor, and more
- **Typed metadata schemas** for each node type

## Installation

```bash
pip install quartermaster-graph
```

**Dependencies**: `pydantic>=2.0`, `pyyaml>=6.0`

## Quick Start

### Build a Graph with the Fluent API

```python
from quartermaster_graph import Graph

agent = (
    Graph("Customer Support Agent")
    .start()
    .user("How can I help you?")
    .instruction("Classify intent", model="gpt-4o", system_instruction="Classify the user's intent.")
    .decision("Route by intent", options=["billing", "technical", "general"])
    .on("billing").instruction("Handle billing", model="gpt-4o").end()
    .on("technical").instruction("Handle technical", model="gpt-4o").end()
    .on("general").instruction("Handle general", model="gpt-4o").end()
    .end()
)
```

### Load a Graph from YAML

```yaml
# agent.yaml
version: "1.0.0"
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
      system_instruction: "Analyze user input"
      model: "gpt-4o"
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
    agent_version = from_yaml(f.read())
```

## API Reference

### Core Models

| Model | Description |
|-------|-------------|
| `Agent` | Top-level agent definition (name, description, tags) |
| `AgentVersion` | Versioned snapshot: nodes, edges, start node, features |
| `GraphNode` | A node with type, metadata, traversal config, error handling |
| `GraphEdge` | Directed edge with optional label and routing points |
| `NodePosition` | Visual position for editor rendering |
| `GraphDiff` | Difference between two graph versions |

### GraphBuilder Methods

| Method | Description |
|--------|-------------|
| `.start()` | Add a Start node |
| `.end()` | Add an End node (or close a branch) |
| `.user(name)` | Add a user input node |
| `.instruction(name, model, provider, temperature, system_instruction)` | Add an LLM instruction node |
| `.decision(name, options)` | Add a decision node |
| `.on(label) -> BranchBuilder` | Start building a named branch |
| `.if_node(name, expression)` | Add a conditional branch |
| `.switch(name, expression, cases)` | Add a multi-way switch |
| `.static(name, content)` | Add a static content node |
| `.code(name, code, language)` | Add a code execution node |
| `.text(name, template)` | Add a text template node |
| `.var(name, variable, expression)` | Add a variable node |
| `.user_form(name, parameters)` | Add a user form node |
| `.parallel(name)` | Add a parallel fork |
| `.branch()` | Start a parallel branch |
| `.static_merge(name)` | Merge parallel branches |
| `.merge(name)` | Add a merge node |
| `.reasoning(name, ...)` | Add a reasoning node |
| `.summarize(name, ...)` | Add a summarize node |
| `.write_memory(name, ...)` | Add a write memory node |
| `.read_memory(name, ...)` | Add a read memory node |
| `.use(sub_graph)` | Inline a sub-graph |
| `.node(node_type, name, metadata)` | Add any node type |
| `.edge(source_id, target_id, label)` | Manually add an edge |
| `.build(validate) -> AgentVersion` | Build and optionally validate |
| `.to_version(validate, version) -> AgentVersion` | Build with explicit version |

### Validation

```python
from quartermaster_graph import validate_graph

errors = validate_graph(agent_version)
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
data = to_json(agent_version)
restored = from_json(data)

# YAML round-trip
yaml_str = to_yaml(agent_version)
restored = from_yaml(yaml_str)

# JSON Schema for cross-language validation
schema = json_schema()
```

### Versioning

```python
from quartermaster_graph import create_version, fork, bump_major, bump_minor, bump_patch

assert bump_patch("1.2.3") == "1.2.4"
assert bump_minor("1.2.3") == "1.3.0"
assert bump_major("1.2.3") == "2.0.0"
```

### Traversal Utilities

```python
from quartermaster_graph import (
    get_start_node, get_successors, get_predecessors,
    get_path, topological_sort, find_merge_points, find_decision_points,
)

start = get_start_node(agent_version)
ordered = topological_sort(agent_version)
```

### Enums

| Enum | Values |
|------|--------|
| `NodeType` | 40 types: `START`, `END`, `INSTRUCTION`, `DECISION`, `IF`, `SWITCH`, `AGENT`, `USER`, `USER_FORM`, `CODE`, `MERGE`, `STATIC`, `STATIC_MERGE`, `VAR`, `TEXT`, etc. |
| `TraverseIn` | `AWAIT_ALL`, `AWAIT_FIRST` |
| `TraverseOut` | `SPAWN_ALL`, `SPAWN_NONE`, `SPAWN_START`, `SPAWN_PICKED` |
| `ThoughtType` | `SKIP`, `NEW`, `NEW_HIDDEN`, `NEW_COLLAPSED`, `EDIT_OR_NEW`, `USE_PREVIOUS`, etc. |
| `MessageType` | `AUTOMATIC`, `USER`, `ASSISTANT`, `SYSTEM`, `TOOL`, `VARIABLE` |
| `ErrorStrategy` | `STOP`, `RETRY`, `SKIP`, `CONTINUE`, `CUSTOM` |

## Integration with Sibling Packages

```python
# Build a graph (quartermaster-graph)
from quartermaster_graph import Graph

agent = Graph("Agent").start().user("Input").instruction("Process").end()

# Execute it (quartermaster-engine)
from quartermaster_engine import FlowRunner

runner = FlowRunner(graph=agent.build(), node_registry=registry)
result = runner.run("Hello!")
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
