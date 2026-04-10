# quartermaster-engine

Execution engine for AI agent graphs. Takes a graph definition, resolves node implementations, and orchestrates the execution: traversal, branching, merging, memory, message passing, and error handling.

## Installation

```bash
pip install quartermaster-engine
```

### Optional dependencies

```bash
pip install quartermaster-engine[sqlite]   # SQLite-backed persistent store
pip install quartermaster-engine[redis]    # Redis-backed high-performance store
pip install quartermaster-engine[all]      # All optional dependencies
```

## Quick Start

```python
from uuid import uuid4
from quartermaster_engine import (
    FlowRunner, AgentVersion, GraphNode, GraphEdge,
    NodeType, TraverseIn, TraverseOut,
)
from quartermaster_engine.nodes import SimpleNodeRegistry

# 1. Define a graph
start_id, instruction_id, end_id = uuid4(), uuid4(), uuid4()

graph = AgentVersion(
    id=uuid4(),
    agent_id=uuid4(),
    version="1.0.0",
    start_node_id=start_id,
    nodes=[
        GraphNode(id=start_id, type=NodeType.START, name="Start"),
        GraphNode(
            id=instruction_id,
            type=NodeType.INSTRUCTION,
            name="Greet",
            metadata={"system_instruction": "You are a friendly assistant."},
        ),
        GraphNode(
            id=end_id,
            type=NodeType.END, name="End",
            traverse_out=TraverseOut.SPAWN_NONE,
        ),
    ],
    edges=[
        GraphEdge(id=uuid4(), source_id=start_id, target_id=instruction_id),
        GraphEdge(id=uuid4(), source_id=instruction_id, target_id=end_id),
    ],
)

# 2. Register node executors
registry = SimpleNodeRegistry()
# registry.register("Instruction1", MyInstructionExecutor())

# 3. Run the flow
runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("Hello, world!")
print(result.final_output)
```

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Graph   │───▶│  FlowRunner  │───▶│   Results    │
│(AgentVer)│    │              │    │ (FlowResult) │
└──────────┘    │  ┌────────┐  │    └──────────────┘
                │  │Traverse│  │
                │  │ In/Out │  │    ┌──────────────┐
                │  └────────┘  │───▶│   Events     │
                │  ┌────────┐  │    │(FlowEvent[]) │
                │  │Message │  │    └──────────────┘
                │  │Router  │  │
                │  └────────┘  │
                │  ┌────────┐  │
                │  │Memory  │  │
                │  │System  │  │
                │  └────────┘  │
                └──────────────┘
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
        ┌─────────┐┌────────┐┌──────┐
        │InMemory ││SQLite  ││Redis │
        │ Store   ││ Store  ││Store │
        └─────────┘└────────┘└──────┘
```

## Key Concepts

### Execution Flow
1. **Start** at the graph's start node
2. **Traverse In** — synchronization gate (AwaitAll / AwaitFirst)
3. **Execute** — resolve node implementation, build context, run
4. **Traverse Out** — branching gate (SpawnAll / SpawnNone / SpawnPicked / SpawnStart)
5. **Dispatch** — trigger successor nodes via pluggable dispatcher
6. **Repeat** until all branches reach End nodes

### Pluggable Components
- **ExecutionStore** — where execution state lives (in-memory, SQLite, Redis, PostgreSQL)
- **TaskDispatcher** — how parallel branches execute (sync, threads, asyncio, Celery)
- **NodeRegistry** — maps node types to executable implementations
- **ContextManager** — manages LLM context window truncation

### Error Handling
Per-node error strategies:
- **Stop** — halt entire flow on error
- **Retry** — retry with configurable max retries and backoff
- **Skip** — skip failed node, continue to successors
- **Custom** — invoke error handling sub-flow

### Memory System
- **FlowMemory** — scoped to a single flow execution (key-value store)
- **PersistentMemory** — cross-flow memory that survives between executions

### Event Streaming
Real-time events for UI integration:
- `NodeStarted`, `TokenGenerated`, `NodeFinished`
- `FlowFinished`, `UserInputRequired`, `FlowError`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## License

Apache-2.0
