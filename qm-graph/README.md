# qm-graph

Framework-agnostic graph schema for defining AI agent workflows as directed acyclic graphs.

[![PyPI version](https://img.shields.io/pypi/v/qm-graph)](https://pypi.org/project/qm-graph/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **Pydantic-based models**: Agent, AgentVersion, GraphNode, GraphEdge with full validation
- **GraphBuilder** fluent API for programmatic graph construction
- **39 node types** covering LLM, control flow, data, user interaction, memory, and utility
- **YAML/JSON serialization** with round-trip fidelity
- **Graph validation**: start/end nodes, cycle detection, orphan detection, edge label checks
- **Semantic versioning**: create, bump, fork, and diff agent versions
- **Graph traversal**: topological sort, path finding, successor/predecessor queries
- **Pre-built templates**: simple chat, decision tree, RAG pipeline, multi-agent supervisor, and more
- **Typed metadata schemas** for each node type

## Installation

```bash
pip install qm-graph
```

**Dependencies**: `pydantic>=2.0`, `pyyaml>=6.0`

## Quick Start

### Build a Graph with the Fluent API

```python
from qm_graph import GraphBuilder

agent_version = (
    GraphBuilder("Customer Support Agent")
    .start()
    .instruction("Classify intent", model="gpt-4o", system_instruction="Classify the user's intent.")
    .decision("Route by intent", options=["billing", "technical", "general"])
    .on("billing").instruction("Handle billing", model="gpt-4o").end()
    .on("technical").instruction("Handle technical", model="gpt-4o").end()
    .on("general").instruction("Handle general", model="gpt-4o").end()
    .build()
)
```

### Use a Pre-built Template

```python
from qm_graph import Templates

# Simple chat loop: Start -> Instruction -> User -> End
chat = Templates.simple_chat(
    model="gpt-4o",
    system_instruction="You are a helpful assistant.",
)

# RAG pipeline: Start -> Tool(retrieve) -> Instruction(generate) -> End
rag = Templates.rag_pipeline(retrieval_tool="vector_search")

# Multi-agent supervisor with specialized workers
supervisor = Templates.multi_agent_supervisor(
    worker_names=["Researcher", "Writer", "Coder"],
    model="gpt-4o",
)

# Advanced RAG with query rewriting and reranking
advanced_rag = Templates.advanced_rag(
    retrieval_tool="vector_search",
    rerank_tool="reranker",
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
    type: "Instruction1"
    name: "Process"
    metadata:
      system_instruction: "Analyze user input"
      model: "gpt-4o"
      temperature: 0.7
  - id: "00000000-0000-0000-0000-000000000003"
    type: "End1"
    name: "End"
edges:
  - source_id: "00000000-0000-0000-0000-000000000001"
    target_id: "00000000-0000-0000-0000-000000000002"
  - source_id: "00000000-0000-0000-0000-000000000002"
    target_id: "00000000-0000-0000-0000-000000000003"
```

```python
from qm_graph import from_yaml

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

### GraphNode Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `NodeType` | required | Node type enum |
| `name` | `str` | `""` | Display name |
| `traverse_in` | `TraverseIn` | `AWAIT_ALL` | How to handle multiple inputs |
| `traverse_out` | `TraverseOut` | `SPAWN_ALL` | How to dispatch outputs |
| `thought_type` | `ThoughtType` | `NEW` | How execution is displayed |
| `message_type` | `MessageType` | `AUTOMATIC` | Role of produced message |
| `error_handling` | `ErrorStrategy` | `STOP` | Error strategy |
| `metadata` | `dict` | `{}` | Node-specific configuration |
| `max_retries` | `int` | `3` | Retry count for error handling |
| `timeout` | `float \| None` | `None` | Execution timeout in seconds |

### AgentVersion Methods

| Method | Description |
|--------|-------------|
| `get_node(node_id) -> GraphNode \| None` | Find a node by ID |
| `get_start_node() -> GraphNode \| None` | Get the start node |
| `get_successors(node_id) -> list[GraphNode]` | Successor nodes |
| `get_predecessors(node_id) -> list[GraphNode]` | Predecessor nodes |
| `get_edges_from(node_id) -> list[GraphEdge]` | Outgoing edges |
| `get_edges_to(node_id) -> list[GraphEdge]` | Incoming edges |

### GraphBuilder Methods

| Method | Description |
|--------|-------------|
| `.start()` | Add a Start node |
| `.end()` | Add an End node |
| `.instruction(name, model, provider, temperature, system_instruction)` | Add an LLM instruction node |
| `.decision(name, options)` | Add a decision node |
| `.on(label) -> BranchBuilder` | Start building a named branch |
| `.if_node(name, expression)` | Add a conditional branch |
| `.static(name, content)` | Add a static content node |
| `.code(name, code, language)` | Add a code execution node |
| `.user(name)` | Add a user input node |
| `.tool(name, tool_name)` | Add a tool invocation node |
| `.sub_agent(name, agent_id)` | Add a sub-agent node |
| `.parallel(name)` | Add a parallel fork |
| `.loop(name, max_iterations, break_condition)` | Add a loop node |
| `.merge(name) -> GraphNode` | Add a merge node (returns node for `merge_to()`) |
| `.node(node_type, name, metadata)` | Add any node type |
| `.edge(source_id, target_id, label, is_main)` | Manually add an edge |
| `.build(validate, version) -> AgentVersion` | Build and optionally validate |

### Validation

```python
from qm_graph import validate_graph

errors = validate_graph(agent_version)
for err in errors:
    print(f"[{err.severity}] {err.code}: {err.message}")
```

Validation checks:
- Exactly one Start node, at least one End node
- All edge source/target IDs reference existing nodes
- Orphan detection (unreachable from Start)
- Cycle detection (DAG property, with Loop node awareness)
- Decision/If/Switch nodes have proper edge labels

### Serialization

```python
from qm_graph import to_json, from_json, to_yaml, from_yaml, json_schema

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
from qm_graph import create_version, fork, bump_major, bump_minor, bump_patch
from qm_graph.versioning import diff

# Bump versions
assert bump_patch("1.2.3") == "1.2.4"
assert bump_minor("1.2.3") == "1.3.0"
assert bump_major("1.2.3") == "2.0.0"

# Create a new version of an agent
v2 = create_version(agent, version="1.1.0", nodes=nodes, edges=edges, start_node_id=start_id)

# Fork a version to a new agent (deep copy with fresh IDs)
forked = fork(v1, new_agent)
print(forked.forked_from)  # Original version ID

# Compare two versions
changes = diff(v1, v2)
print(f"Changed: {len(changes.node_diffs)} nodes, {len(changes.edge_diffs)} edges")
```

### Traversal Utilities

```python
from qm_graph import (
    get_start_node,
    get_successors,
    get_predecessors,
    get_path,
    topological_sort,
    find_merge_points,
    find_decision_points,
)

start = get_start_node(agent_version)
successors = get_successors(agent_version, node.id)
predecessors = get_predecessors(agent_version, node.id)
path = get_path(agent_version, start_id, end_id)
ordered = topological_sort(agent_version)
merges = find_merge_points(agent_version)
decisions = find_decision_points(agent_version)
```

### Typed Metadata Schemas

```python
from qm_graph import get_metadata_class, InstructionMetadata, NodeType

# Get the metadata class for a node type
cls = get_metadata_class(NodeType.INSTRUCTION)  # Returns InstructionMetadata
meta = cls(model="gpt-4o", temperature=0.5, system_instruction="Be helpful")
```

Available metadata classes: `InstructionMetadata`, `DecisionMetadata`, `IfMetadata`, `SwitchMetadata`, `StaticMetadata`, `CodeMetadata`, `VarMetadata`, `UserFormMetadata`.

### Enums

| Enum | Values |
|------|--------|
| `NodeType` | 39 types: `START`, `END`, `INSTRUCTION`, `DECISION`, `IF`, `AGENT`, `USER`, `TOOL`, `CODE`, `MERGE`, `LOOP`, `PARALLEL`, etc. |
| `TraverseIn` | `AWAIT_ALL`, `AWAIT_FIRST` |
| `TraverseOut` | `SPAWN_ALL`, `SPAWN_NONE`, `SPAWN_START`, `SPAWN_PICKED` |
| `ThoughtType` | `SKIP`, `NEW`, `NEW_HIDDEN`, `NEW_COLLAPSED`, `EDIT_OR_NEW`, `USE_PREVIOUS`, etc. |
| `MessageType` | `AUTOMATIC`, `USER`, `ASSISTANT`, `SYSTEM`, `TOOL`, `VARIABLE` |
| `ErrorStrategy` | `STOP`, `RETRY`, `SKIP`, `CONTINUE`, `CUSTOM` |

## Integration with Sibling Packages

### With qm-nodes (node implementations)

qm-graph defines the schema; qm-nodes implements behavior:

```python
# qm-graph defines WHAT the graph looks like
from qm_graph import GraphBuilder, NodeType

version = GraphBuilder("My Agent").start().instruction("Process").end().build()

# qm-nodes defines WHAT each node does
from qm_nodes.nodes import InstructionNodeV1
InstructionNodeV1.think(ctx)  # Executes the instruction node logic
```

### With qm-engine (execution runtime)

qm-engine consumes graph definitions and orchestrates execution:

```python
from qm_graph import GraphBuilder
from qm_engine import FlowRunner

graph = GraphBuilder("Agent").start().instruction("Greet").end().build()
runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("Hello!")
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
