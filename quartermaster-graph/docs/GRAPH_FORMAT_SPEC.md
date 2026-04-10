# quartermaster-graph Format Specification v0.1

This document specifies the JSON interchange format for agent graphs. Any frontend editor can produce this format, and any execution engine can consume it.

## Overview

An agent graph is a **directed acyclic graph (DAG)** where:
- **Nodes** represent operations (LLM calls, decisions, code execution, user interactions)
- **Edges** represent the flow of execution between nodes
- **Versions** snapshot the graph at a point in time

## Top-Level Structure: AgentVersion

```json
{
  "id": "uuid",
  "agent_id": "uuid",
  "version": "X.Y.Z",
  "start_node_id": "uuid",
  "nodes": [...],
  "edges": [...],
  "features": "string",
  "is_published": false,
  "forked_from": "uuid | null",
  "created_at": "ISO 8601 datetime"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | UUID | auto | Unique identifier for this version |
| `agent_id` | UUID | yes | The agent this version belongs to |
| `version` | string | yes | Semver version string (e.g., "1.0.0") |
| `start_node_id` | UUID | yes | ID of the Start node |
| `nodes` | array | yes | List of GraphNode objects |
| `edges` | array | yes | List of GraphEdge objects |
| `features` | string | no | Release notes / changelog |
| `is_published` | bool | no | Whether this version is live |
| `forked_from` | UUID | no | If forked, the source version ID |
| `created_at` | datetime | auto | When this version was created |

## GraphNode

```json
{
  "id": "uuid",
  "type": "NodeType enum value",
  "name": "string",
  "traverse_in": "AwaitAll | AwaitFirst",
  "traverse_out": "SpawnAll | SpawnNone | SpawnStart | SpawnPickedNode",
  "thought_type": "ThoughtType enum value",
  "message_type": "Automatic | User | Variable | Assistant | System",
  "error_handling": "Stop | Retry | Skip | Custom",
  "metadata": { ... },
  "position": { "x": 0, "y": 0, "icon": "string | null" }
}
```

### Node Types

| Value | Description |
|---|---|
| `Start1` | Entry point of the graph |
| `End1` | Exit point of the graph |
| `Instruction1` | LLM instruction / prompt execution |
| `Decision1` | LLM-driven branching decision |
| `Reasoning1` | Chain-of-thought reasoning step |
| `Agent1` | Delegate to another agent |
| `Merge1` | Merge multiple branches |
| `If1` | Conditional branch (expression-based) |
| `Switch1` | Multi-way conditional |
| `Break1` | Break out of a loop |
| `User1` | Wait for user input |
| `UserDecision1` | User picks a branch |
| `UserForm1` | User fills a form |
| `Static1` | Static content output |
| `Var1` | Variable assignment |
| `Text1` | Text template |
| `Code1` | Code execution |
| `ProgramRunner1` | External program |
| `FlowMemory1` | Flow-scoped memory |
| `ReadMemory1` | Read from memory |
| `WriteMemory1` | Write to memory |
| `Tool1` | Tool invocation |
| `ApiCall1` | HTTP API call |
| `Webhook1` | Webhook trigger/receiver |
| `Timer1` | Delay / timer |
| `Loop1` | Loop construct |
| `Parallel1` | Parallel execution fork |
| `SubAgent1` | Invoke sub-agent |
| `Template1` | Template expansion |
| `Validator1` | Data validation |
| `Transformer1` | Data transformation |
| `Filter1` | Data filtering |
| `Aggregator1` | Data aggregation |
| `Router1` | Dynamic routing |
| `ErrorHandler1` | Error handling |
| `Log1` | Logging |
| `Notification1` | Send notification |
| `Custom1` | Custom node type |
| `Comment1` | Visual comment (no execution) |

### Traverse Strategies

**TraverseIn** — how a node handles multiple incoming edges:
- `AwaitAll`: Wait until all incoming branches have completed
- `AwaitFirst`: Proceed as soon as the first incoming branch completes

**TraverseOut** — how a node dispatches to outgoing edges:
- `SpawnAll`: Activate all outgoing edges (parallel fan-out)
- `SpawnNone`: Terminal node, no outgoing activation
- `SpawnStart`: Return to the start node
- `SpawnPickedNode`: Activate only the edge selected by the node's logic (for decisions)

### Metadata

The `metadata` field is a JSON object whose schema depends on the node type. Common schemas:

**InstructionMetadata** (for `Instruction1`, `Reasoning1`):
```json
{
  "system_instruction": "string",
  "model": "gpt-4o",
  "provider": "openai",
  "temperature": 0.7,
  "max_tokens": null,
  "tools": [],
  "response_format": null
}
```

**DecisionMetadata** (for `Decision1`):
```json
{
  "system_instruction": "string",
  "model": "gpt-4o",
  "provider": "openai",
  "temperature": 0.7,
  "decision_prompt": "string"
}
```

**IfMetadata** (for `If1`):
```json
{
  "expression": "x > 0",
  "variable": "x"
}
```

**CodeMetadata** (for `Code1`, `ProgramRunner1`):
```json
{
  "language": "python",
  "code": "string",
  "timeout_seconds": 30
}
```

**ToolMetadata** (for `Tool1`):
```json
{
  "tool_name": "string",
  "tool_args": { "key": "value" }
}
```

## GraphEdge

```json
{
  "id": "uuid",
  "source_id": "uuid",
  "target_id": "uuid",
  "label": "string",
  "is_main": true,
  "points": [[x, y], ...]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | UUID | auto | Unique identifier |
| `source_id` | UUID | yes | Tail node ID |
| `target_id` | UUID | yes | Head node ID |
| `label` | string | no | Edge label (e.g., "Yes", "No" for decisions) |
| `is_main` | bool | no | Whether this is the primary flow direction |
| `points` | array | no | Bezier control points for visual rendering |

## Validation Rules

A well-formed graph must satisfy:

1. **Exactly one Start node** (`type == "Start1"`)
2. **At least one End node** (`type == "End1"`)
3. **`start_node_id` references a Start node**
4. **All edge `source_id` and `target_id` reference existing nodes**
5. **No orphan nodes** (all nodes reachable from Start, except Comments)
6. **No cycles** (DAG property; Loop nodes get a warning instead of error)
7. **Decision/Switch edges are labeled** when there are multiple outgoing edges
8. **If node edges** should use true/false labels

## Versioning

Versions follow semantic versioning (semver):
- **Major**: Breaking graph structure changes
- **Minor**: New nodes/features added
- **Patch**: Metadata or configuration changes

Forking creates a deep copy of a version under a new agent, with all IDs regenerated.
