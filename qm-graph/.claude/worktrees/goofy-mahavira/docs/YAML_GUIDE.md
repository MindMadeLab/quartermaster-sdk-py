# YAML Agent Definition Guide

Define AI agent workflows as YAML files for version control, review, and portability.

## Minimal Example

```yaml
agent_id: "550e8400-e29b-41d4-a716-446655440000"
version: "1.0.0"
start_node_id: &start "a1b2c3d4-0000-0000-0000-000000000001"

nodes:
  - id: *start
    type: Start1
    name: Start

  - id: "a1b2c3d4-0000-0000-0000-000000000002"
    type: Instruction1
    name: Greet User
    metadata:
      system_instruction: "Greet the user warmly"
      model: gpt-4o

  - id: "a1b2c3d4-0000-0000-0000-000000000003"
    type: End1
    name: End

edges:
  - source_id: *start
    target_id: "a1b2c3d4-0000-0000-0000-000000000002"
  - source_id: "a1b2c3d4-0000-0000-0000-000000000002"
    target_id: "a1b2c3d4-0000-0000-0000-000000000003"
```

## Loading in Python

```python
from qm_graph import from_yaml, validate_graph

with open("agent.yaml") as f:
    version = from_yaml(f.read())

errors = validate_graph(version)
if errors:
    for e in errors:
        print(f"  [{e.severity}] {e.code}: {e.message}")
else:
    print("Graph is valid!")
```

## Decision Tree Example

```yaml
agent_id: "550e8400-e29b-41d4-a716-446655440000"
version: "1.0.0"
start_node_id: &start "00000000-0000-0000-0000-000000000001"

nodes:
  - id: *start
    type: Start1
    name: Start

  - id: &decide "00000000-0000-0000-0000-000000000002"
    type: Decision1
    name: Classify Intent
    traverse_out: SpawnPickedNode
    metadata:
      system_instruction: "Classify the user's intent"
      model: gpt-4o
      decision_prompt: "Is this a question, complaint, or feedback?"

  - id: &question "00000000-0000-0000-0000-000000000003"
    type: Instruction1
    name: Answer Question
    metadata:
      system_instruction: "Answer the user's question helpfully"
      model: gpt-4o

  - id: &complaint "00000000-0000-0000-0000-000000000004"
    type: Instruction1
    name: Handle Complaint
    metadata:
      system_instruction: "Acknowledge and address the complaint empathetically"
      model: gpt-4o

  - id: &feedback "00000000-0000-0000-0000-000000000005"
    type: Instruction1
    name: Process Feedback
    metadata:
      system_instruction: "Thank for feedback and summarize key points"
      model: gpt-4o

  - id: &end1 "00000000-0000-0000-0000-000000000006"
    type: End1
    name: End

edges:
  - source_id: *start
    target_id: *decide

  - source_id: *decide
    target_id: *question
    label: Question

  - source_id: *decide
    target_id: *complaint
    label: Complaint

  - source_id: *decide
    target_id: *feedback
    label: Feedback

  - source_id: *question
    target_id: *end1
  - source_id: *complaint
    target_id: *end1
  - source_id: *feedback
    target_id: *end1
```

## RAG Pipeline Example

```yaml
agent_id: "660e8400-e29b-41d4-a716-446655440000"
version: "1.0.0"
start_node_id: &start "10000000-0000-0000-0000-000000000001"

nodes:
  - id: *start
    type: Start1
    name: Start

  - id: &retrieve "10000000-0000-0000-0000-000000000002"
    type: Tool1
    name: Retrieve Context
    metadata:
      tool_name: vector_search
      tool_args:
        collection: knowledge_base
        top_k: "5"

  - id: &generate "10000000-0000-0000-0000-000000000003"
    type: Instruction1
    name: Generate Answer
    metadata:
      system_instruction: |
        Answer the user's question using ONLY the retrieved context.
        If the context doesn't contain the answer, say so.
      model: gpt-4o
      temperature: 0.3

  - id: &end "10000000-0000-0000-0000-000000000004"
    type: End1
    name: End

edges:
  - source_id: *start
    target_id: *retrieve
  - source_id: *retrieve
    target_id: *generate
  - source_id: *generate
    target_id: *end
```

## Tips

- Use YAML anchors (`&name` / `*name`) to reference node IDs cleanly
- Use `|` for multi-line strings in metadata (like system instructions)
- Keep one agent per YAML file for clean git diffs
- Use `version` field to track changes with semver
- Run `validate_graph()` after loading to catch structural issues
