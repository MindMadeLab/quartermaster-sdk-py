# quartermaster-engine

Execution engine for AI agent graphs with pluggable storage, dispatching, and memory.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-engine)](https://pypi.org/project/quartermaster-engine/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **FlowRunner** orchestrates graph execution: traversal, branching, merging, and error handling
- **Pluggable dispatchers**: SyncDispatcher, ThreadDispatcher, AsyncDispatcher
- **Pluggable storage**: InMemoryStore, SQLiteStore, or implement your own ExecutionStore
- **Two memory layers**: FlowMemory (per-execution) and PersistentMemory (cross-execution)
- **Real-time event streaming**: NodeStarted, TokenGenerated, NodeFinished, FlowError, UserInputRequired
- **Per-node error strategies**: Stop, Retry (with configurable max retries), Skip
- **Flow pause/resume** for user interaction nodes
- **Sync and async execution** modes with `run()` and `run_async()`

## Installation

```bash
pip install quartermaster-engine

# With SQLite persistent store
pip install quartermaster-engine[sqlite]
```

## Quick Start

### Run a Simple Graph

```python
from uuid import uuid4
from quartermaster_engine import FlowRunner, InMemoryStore
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeResult
from quartermaster_engine.types import AgentVersion, GraphNode, GraphEdge, NodeType, TraverseOut

# 1. Define the graph
start_id, process_id, end_id = uuid4(), uuid4(), uuid4()

graph = AgentVersion(
    id=uuid4(),
    agent_id=uuid4(),
    version="1.0.0",
    start_node_id=start_id,
    nodes=[
        GraphNode(id=start_id, type=NodeType.START, name="Start"),
        GraphNode(
            id=process_id,
            type=NodeType.INSTRUCTION,
            name="Process",
            metadata={"system_instruction": "Summarize the input."},
        ),
        GraphNode(id=end_id, type=NodeType.END, name="End"),
    ],
    edges=[
        GraphEdge(id=uuid4(), source_id=start_id, target_id=process_id),
        GraphEdge(id=uuid4(), source_id=process_id, target_id=end_id),
    ],
)

# 2. Register node executors
registry = SimpleNodeRegistry()
# registry.register("Instruction1", my_instruction_executor)

# 3. Run the flow
runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("Please summarize this article about AI safety.")

print(result.success)       # True/False
print(result.final_output)  # The final text output
print(result.duration_seconds)
```

### Stream Events in Real Time

```python
from quartermaster_engine import FlowRunner, FlowEvent, NodeStarted, NodeFinished, TokenGenerated, FlowError

def handle_event(event: FlowEvent):
    if isinstance(event, NodeStarted):
        print(f"[START] {event.node_name} ({event.node_type})")
    elif isinstance(event, TokenGenerated):
        print(event.token, end="", flush=True)
    elif isinstance(event, NodeFinished):
        print(f"\n[DONE] Node finished: {event.result[:50]}...")
    elif isinstance(event, FlowError):
        print(f"[ERROR] {event.error} (recoverable={event.recoverable})")

runner = FlowRunner(graph=graph, node_registry=registry, on_event=handle_event)
result = runner.run("Hello!")
```

### Async Execution

```python
import asyncio
from quartermaster_engine import FlowRunner, NodeStarted, TokenGenerated

async def run_flow():
    runner = FlowRunner(graph=graph, node_registry=registry)
    async for event in runner.run_async("Hello!"):
        if isinstance(event, TokenGenerated):
            print(event.token, end="", flush=True)

asyncio.run(run_flow())
```

## API Reference

### FlowRunner

The core orchestration class.

```python
runner = FlowRunner(
    graph=agent_version,         # AgentVersion from quartermaster-graph
    node_registry=registry,      # Maps node types to executors
    store=InMemoryStore(),       # Execution state storage
    dispatcher=SyncDispatcher(), # How branches are dispatched
    context_manager=ContextManager(),  # LLM context window management
    on_event=handle_event,       # Real-time event callback
)
```

| Method | Description |
|--------|-------------|
| `run(input_message, flow_id=None) -> FlowResult` | Execute synchronously |
| `run_async(input_message, flow_id=None) -> AsyncIterator[FlowEvent]` | Execute asynchronously, yielding events |
| `resume(flow_id, user_input) -> FlowResult` | Resume a paused flow with user input |
| `stop(flow_id)` | Stop a running flow |

### FlowResult

| Field | Type | Description |
|-------|------|-------------|
| `flow_id` | `UUID` | Unique execution identifier |
| `success` | `bool` | Whether all nodes completed successfully |
| `final_output` | `str` | Text output from the End node |
| `output_data` | `dict` | Structured output data |
| `error` | `str \| None` | Error message if failed |
| `node_results` | `dict[UUID, NodeResult]` | Per-node results |
| `duration_seconds` | `float` | Total execution time |

### Dispatchers

Control how successor nodes are executed.

| Dispatcher | Description |
|------------|-------------|
| `SyncDispatcher` | Execute nodes sequentially in the calling thread. Simple and predictable. |
| `ThreadDispatcher(max_workers=4)` | Execute branches in parallel using a thread pool. Good for I/O-bound nodes. |
| `AsyncDispatcher` | Execute branches concurrently using asyncio tasks. For async web applications. |

All dispatchers implement the `TaskDispatcher` protocol:

```python
class TaskDispatcher(Protocol):
    def dispatch(self, flow_id, node_id, execute_fn) -> None: ...
    def wait_all(self) -> None: ...
    def shutdown(self) -> None: ...
```

### Execution Stores

Pluggable storage for flow state, memory, and messages.

| Store | Description |
|-------|-------------|
| `InMemoryStore` | Dict-backed, no persistence. Great for testing. |
| `SQLiteStore(db_path)` | SQLite-backed with WAL mode. For local development. |

Implement `ExecutionStore` for custom backends (Redis, PostgreSQL, etc.):

```python
from quartermaster_engine import ExecutionStore

class RedisStore:
    def save_node_execution(self, flow_id, node_id, execution) -> None: ...
    def get_node_execution(self, flow_id, node_id) -> NodeExecution | None: ...
    def get_all_node_executions(self, flow_id) -> dict[UUID, NodeExecution]: ...
    def save_memory(self, flow_id, key, value) -> None: ...
    def get_memory(self, flow_id, key) -> Any: ...
    def get_all_memory(self, flow_id) -> dict[str, Any]: ...
    def delete_memory(self, flow_id, key) -> None: ...
    def save_messages(self, flow_id, node_id, messages) -> None: ...
    def get_messages(self, flow_id, node_id) -> list[Message]: ...
    def append_message(self, flow_id, node_id, message) -> None: ...
    def clear_flow(self, flow_id) -> None: ...
```

### Memory System

**FlowMemory** -- scoped to a single flow execution:

```python
from quartermaster_engine import FlowMemory, InMemoryStore

store = InMemoryStore()
memory = FlowMemory(flow_id=my_flow_id, store=store)

memory.set("user_name", "Alice")
memory.set("preferences", {"language": "en"})

name = memory.get("user_name")            # "Alice"
all_data = memory.get_all()                # {"user_name": "Alice", ...}
keys = memory.list_keys()                  # ["user_name", "preferences"]
memory.delete("preferences")
memory.clear()
```

**PersistentMemory** -- cross-flow memory that survives between executions:

```python
from quartermaster_engine import PersistentMemory, InMemoryPersistence

persistence = InMemoryPersistence()

persistence.write(agent_id, "user_pref", "dark_mode")
value = persistence.read(agent_id, "user_pref")     # "dark_mode"
persistence.update(agent_id, "user_pref", "light_mode")

results = persistence.search(agent_id, "pref")      # Substring search
keys = persistence.list_keys(agent_id)               # ["user_pref"]
persistence.delete(agent_id, "user_pref")
```

### Error Handling

Per-node error strategies configured via `GraphNode.error_handling`:

| Strategy | Behavior |
|----------|----------|
| `ErrorStrategy.STOP` | Halt entire flow on error (default) |
| `ErrorStrategy.RETRY` | Retry up to `max_retries` times with `retry_delay` backoff |
| `ErrorStrategy.SKIP` | Skip failed node, continue to successors |

```python
from quartermaster_engine.types import GraphNode, NodeType, ErrorStrategy

node = GraphNode(
    type=NodeType.INSTRUCTION,
    name="Unreliable API",
    error_handling=ErrorStrategy.RETRY,
    max_retries=3,
    retry_delay=2.0,
    timeout=30.0,
)
```

### Events

Real-time events emitted during flow execution:

| Event | Fields | Description |
|-------|--------|-------------|
| `NodeStarted` | `node_id`, `node_type`, `node_name` | Node begins execution |
| `TokenGenerated` | `node_id`, `token` | Streaming token from LLM |
| `NodeFinished` | `node_id`, `result`, `output_data` | Node completed |
| `FlowFinished` | `final_output`, `output_data` | Entire flow completed |
| `UserInputRequired` | `node_id`, `prompt`, `options` | Flow paused for user input |
| `FlowError` | `node_id`, `error`, `recoverable` | Node failed |

### ExecutionContext

The runtime context passed to each node executor:

| Field | Type | Description |
|-------|------|-------------|
| `flow_id` | `UUID` | Flow execution identifier |
| `node_id` | `UUID` | Current node identifier |
| `graph` | `AgentVersion` | Full graph definition |
| `current_node` | `GraphNode` | Current node definition |
| `messages` | `list[Message]` | Conversation history |
| `memory` | `dict[str, Any]` | Flow-scoped memory snapshot |
| `metadata` | `dict[str, Any]` | Node metadata |
| `on_token` | `Callable` | Callback for streaming tokens |

### Node Execution Protocol

Implement `NodeExecutor` to add custom node types:

```python
from quartermaster_engine.nodes import NodeExecutor, NodeResult
from quartermaster_engine.context.execution_context import ExecutionContext

class MyInstructionExecutor:
    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("system_instruction", "")
        # Call your LLM here...
        response_text = "Generated response"

        # Stream tokens in real time
        for token in response_text.split():
            context.emit_token(token + " ")

        return NodeResult(
            success=True,
            data={"model": "gpt-4o"},
            output_text=response_text,
        )
```

## Integration with Sibling Packages

### With quartermaster-graph (graph definition)

Use the GraphBuilder to create graphs, then execute them with FlowRunner:

```python
from quartermaster_graph import GraphBuilder
from quartermaster_engine import FlowRunner

graph = (
    GraphBuilder("Support Agent")
    .start()
    .instruction("Classify", model="gpt-4o")
    .decision("Route", options=["billing", "technical"])
    .on("billing").instruction("Handle billing").end()
    .on("technical").instruction("Handle technical").end()
    .build()
)

runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("I need help with my invoice")
```

### With quartermaster-nodes (node implementations)

Bridge quartermaster-nodes node classes to the engine's NodeExecutor protocol:

```python
from quartermaster_nodes import NodeRegistry as QMNodeRegistry
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeExecutor, NodeResult

# Discover all quartermaster-nodes implementations
node_registry = QMNodeRegistry()
node_registry.discover("quartermaster_nodes.nodes")

# Register them with the engine
engine_registry = SimpleNodeRegistry()
# Adapt each quartermaster-nodes class to the NodeExecutor protocol
```

## Configuration

### SQLiteStore

```python
from quartermaster_engine.stores.sqlite_store import SQLiteStore

store = SQLiteStore(db_path="my_agent.db")
runner = FlowRunner(graph=graph, node_registry=registry, store=store)
```

Tables are created automatically on first use. Uses WAL mode for concurrent read access.

### ThreadDispatcher

```python
from quartermaster_engine.dispatchers.thread_dispatcher import ThreadDispatcher

dispatcher = ThreadDispatcher(max_workers=8)
runner = FlowRunner(graph=graph, node_registry=registry, dispatcher=dispatcher)
result = runner.run("Process this in parallel")
dispatcher.shutdown()  # Clean up thread pool
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
