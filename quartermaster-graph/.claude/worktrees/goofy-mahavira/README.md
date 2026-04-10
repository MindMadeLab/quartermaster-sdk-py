# quartermaster-graph

Framework-agnostic agent graph schema for defining AI agent workflows as directed acyclic graphs (DAGs).

`quartermaster-graph` provides the **blueprint format** for AI agent flows — how they're defined, validated, versioned, and stored. Any visual editor can output this format, and any execution engine can consume it.

## Installation

```bash
pip install quartermaster-graph
```

**Requirements:** Python 3.10+ | Pydantic 2.x | PyYAML 6.x

## Quick Start

### Define a graph with the fluent builder

```python
from quartermaster_graph import GraphBuilder

agent = (
    GraphBuilder("My Agent")
    .start()
    .instruction("Analyze the input", model="gpt-4o")
    .decision("Is it positive?", options=["Yes", "No"])
    .on("Yes").instruction("Generate positive response").end()
    .on("No").instruction("Generate negative response").end()
    .build()
)
```

### Use a pre-built template

```python
from quartermaster_graph import Templates

# Simple chat loop
chat = Templates.simple_chat(model="claude-3", system_instruction="You are helpful.")

# RAG pipeline
rag = Templates.rag_pipeline(retrieval_tool="vector_search")

# Decision tree
tree = Templates.decision_tree(question="Route?", options=["Sales", "Support", "Billing"])
```

### Define a graph as YAML

```yaml
version: "1.0.0"
agent_id: "550e8400-e29b-41d4-a716-446655440000"
start_node_id: "node-start"
nodes:
  - id: "node-start"
    type: "Start1"
    name: "Start"
  - id: "node-process"
    type: "Instruction1"
    name: "Process Input"
    metadata:
      system_instruction: "Analyze user input"
      model: "gpt-4o"
      temperature: 0.7
  - id: "node-end"
    type: "End1"
    name: "End"
edges:
  - source_id: "node-start"
    target_id: "node-process"
  - source_id: "node-process"
    target_id: "node-end"
```

Load it:

```python
from quartermaster_graph import from_yaml

with open("agent.yaml") as f:
    version = from_yaml(f.read())
```

## Core Concepts

### Models

| Model | Description |
|---|---|
| `Agent` | Top-level agent definition (name, tags, version) |
| `AgentVersion` | Versioned snapshot of the graph (nodes + edges) |
| `GraphNode` | A node in the DAG (type, metadata, traversal config) |
| `GraphEdge` | A directed edge between two nodes |
| `NodePosition` | Visual position for editor rendering |
| `GraphDiff` | Difference between two versions |

### Node Types (39 types)

Instruction, Decision, Reasoning, Agent, Start, End, Merge, If, Switch, Break, User, UserDecision, UserForm, Static, Var, Text, Code, ProgramRunner, FlowMemory, ReadMemory, WriteMemory, Tool, ApiCall, Webhook, Timer, Loop, Parallel, SubAgent, Template, Validator, Transformer, Filter, Aggregator, Router, ErrorHandler, Log, Notification, Custom, Comment

### Traversal Strategies

- **TraverseIn**: `AwaitAll` (wait for all inputs) | `AwaitFirst` (proceed on first input)
- **TraverseOut**: `SpawnAll` (activate all outputs) | `SpawnNone` | `SpawnStart` | `SpawnPickedNode`

## API Reference

### Validation

```python
from quartermaster_graph import validate_graph

errors = validate_graph(version)
for err in errors:
    print(f"[{err.severity}] {err.code}: {err.message}")
```

Checks: start/end nodes, edge references, orphan nodes, cycle detection, edge labels for decision/if/switch nodes.

### Versioning

```python
from quartermaster_graph import create_version, fork, bump_minor
from quartermaster_graph.versioning import diff

# Create a new version
v2 = create_version(agent, version="1.1.0", nodes=nodes, edges=edges, start_node_id=start.id)

# Fork to a new agent
forked = fork(v1, new_agent)

# Compare versions
changes = diff(v1, v2)
print(f"Changed: {len(changes.node_diffs)} nodes, {len(changes.edge_diffs)} edges")

# Semver helpers
assert bump_minor("1.2.3") == "1.3.0"
```

### Serialization

```python
from quartermaster_graph import to_json, from_json, to_yaml, from_yaml, json_schema

# JSON
data = to_json(version)
restored = from_json(data)

# YAML
yaml_str = to_yaml(version)
restored = from_yaml(yaml_str)

# JSON Schema (for cross-language validation)
schema = json_schema()
```

### Traversal

```python
from quartermaster_graph import (
    get_start_node, get_successors, get_predecessors,
    get_path, topological_sort, find_merge_points, find_decision_points,
)

start = get_start_node(version)
succs = get_successors(version, node.id)
preds = get_predecessors(version, node.id)
path = get_path(version, start_id, end_id)
ordered = topological_sort(version)
merges = find_merge_points(version)
decisions = find_decision_points(version)
```

### Builder

```python
from quartermaster_graph import GraphBuilder

version = (
    GraphBuilder("RAG Agent", description="Retrieval-augmented generation")
    .start()
    .tool("Retrieve", tool_name="vector_search")
    .instruction("Generate", model="gpt-4o", system_instruction="Answer from context")
    .user("Feedback")
    .end()
    .build()
)
```

Available builder methods: `.start()`, `.end()`, `.instruction()`, `.decision()`, `.on()`, `.if_node()`, `.static()`, `.code()`, `.user()`, `.tool()`, `.sub_agent()`, `.parallel()`, `.loop()`, `.merge()`, `.node()`, `.edge()`

### Metadata Schemas

Each node type has an optional typed metadata schema:

```python
from quartermaster_graph import get_metadata_class, InstructionMetadata

cls = get_metadata_class(NodeType.INSTRUCTION)  # InstructionMetadata
meta = cls(model="gpt-4o", temperature=0.5, system_instruction="Be helpful")
```

## Architecture

```
quartermaster-graph (this package)    — WHAT an agent flow looks like (the blueprint)
quartermaster-engine                  — HOW it executes (the runtime)
quartermaster-nodes                   — WHAT each node does (the behavior)
quartermaster-tools                   — Tools referenced in node metadata
quartermaster-providers               — LLM providers referenced in node metadata
```

## Development

```bash
git clone https://github.com/MindMade/quartermaster-graph.git
cd quartermaster-graph
python -m venv .venv && source .venv/bin/activate
pip install pydantic pyyaml pytest pytest-cov mypy ruff
PYTHONPATH=src pytest
```

## License

MIT
