# Execution Engine

The `qm-engine` package is the runtime that executes agent graphs. It handles traversal, branching, merging, memory management, message routing, error handling, and streaming. The central class is `FlowRunner`.

## FlowRunner

### Construction

```python
from qm_engine import FlowRunner
from qm_engine.nodes import SimpleNodeRegistry
from qm_engine.stores.memory_store import InMemoryStore
from qm_engine.dispatchers.sync_dispatcher import SyncDispatcher

runner = FlowRunner(
    graph=graph,                          # AgentVersion from GraphBuilder
    node_registry=SimpleNodeRegistry(),   # Maps node types to executors
    store=InMemoryStore(),                # Execution state storage (optional)
    dispatcher=SyncDispatcher(),          # Execution strategy (optional)
    on_event=lambda e: print(e),          # Event callback (optional)
)
```

### Synchronous Execution

```python
result = runner.run("Hello, analyze this text.")

print(result.final_output)       # The text output from the End node
print(result.success)            # True if all nodes succeeded
print(result.duration_seconds)   # Wall clock time
print(result.error)              # Error message if failed, else None
print(result.node_results)       # Dict[UUID, NodeResult] for each node
```

### Async Streaming Execution

```python
import asyncio
from qm_engine.events import (
    NodeStarted, NodeFinished, TokenGenerated,
    FlowFinished, FlowError, UserInputRequired,
)

async def main():
    async for event in runner.run_async("Tell me about Python"):
        if isinstance(event, NodeStarted):
            print(f"Node started: {event.node_name}")
        elif isinstance(event, TokenGenerated):
            print(event.token, end="", flush=True)
        elif isinstance(event, NodeFinished):
            print(f"\nNode finished: {event.result[:80]}")
        elif isinstance(event, FlowError):
            print(f"Error at node {event.node_id}: {event.error}")
        elif isinstance(event, UserInputRequired):
            print(f"Waiting for input: {event.prompt}")
        elif isinstance(event, FlowFinished):
            print(f"Done: {event.final_output[:80]}")

asyncio.run(main())
```

### Resuming After User Input

When a `User` node is encountered, execution pauses and emits a `UserInputRequired` event. Resume with:

```python
result = runner.run("Start the conversation")
# result indicates waiting for user input

result = runner.resume(flow_id=result.flow_id, user_input="Yes, proceed")
```

### Stopping a Flow

```python
runner.stop(flow_id=result.flow_id)
```

This marks all active nodes as failed and prevents further dispatching.

## FlowResult

The return type of `runner.run()`:

| Field | Type | Description |
|-------|------|-------------|
| `flow_id` | `UUID` | Unique identifier for this execution |
| `success` | `bool` | `True` if all nodes completed without errors |
| `final_output` | `str` | Text output from the End node |
| `output_data` | `dict` | Structured data from the End node |
| `error` | `str\|None` | Semicolon-joined error messages, or `None` |
| `node_results` | `dict[UUID, NodeResult]` | Per-node execution results |
| `duration_seconds` | `float` | Total execution wall time |

## Flow Events

Events are emitted during execution via the `on_event` callback or the async iterator:

| Event | Fields | When |
|-------|--------|------|
| `NodeStarted` | `flow_id`, `node_id`, `node_type`, `node_name` | A node begins execution |
| `TokenGenerated` | `flow_id`, `node_id`, `token` | A streaming token arrives from an LLM |
| `NodeFinished` | `flow_id`, `node_id`, `result`, `output_data` | A node completes |
| `UserInputRequired` | `flow_id`, `node_id`, `prompt`, `options` | Execution pauses for user input |
| `FlowError` | `flow_id`, `node_id`, `error`, `recoverable` | A node fails |
| `FlowFinished` | `flow_id`, `final_output`, `output_data` | All branches complete |

## Dispatchers

Dispatchers control how successor nodes are executed. All dispatchers implement the same interface:

```python
class Dispatcher:
    def dispatch(self, flow_id, node_id, execute_fn): ...
    def wait_all(self): ...
    def shutdown(self): ...
```

### SyncDispatcher

Executes nodes immediately in the calling thread. No parallelism. Simple and predictable.

```python
from qm_engine.dispatchers.sync_dispatcher import SyncDispatcher

runner = FlowRunner(graph=graph, node_registry=registry, dispatcher=SyncDispatcher())
```

Best for: testing, debugging, simple sequential flows.

### ThreadDispatcher

Executes nodes in parallel using a `ThreadPoolExecutor`. Ideal for I/O-bound workloads (LLM API calls).

```python
from qm_engine.dispatchers.thread_dispatcher import ThreadDispatcher

runner = FlowRunner(
    graph=graph,
    node_registry=registry,
    dispatcher=ThreadDispatcher(max_workers=4),
)
```

`wait_all()` blocks until all parallel branches complete. Exceptions from any branch are collected and raised as an `ExceptionGroup`.

Best for: production deployments with parallel branches, I/O-bound LLM calls.

### AsyncDispatcher

Executes nodes as `asyncio.Task` instances. For use in async web applications (FastAPI, aiohttp).

```python
from qm_engine.dispatchers.async_dispatcher import AsyncDispatcher

runner = FlowRunner(
    graph=graph,
    node_registry=registry,
    dispatcher=AsyncDispatcher(),
)
```

The synchronous `execute_fn` is wrapped in `loop.run_in_executor` so it does not block the event loop.

Best for: async web applications, high-concurrency scenarios.

## Execution Stores

The `ExecutionStore` protocol defines how flow state is persisted. The engine is storage-agnostic.

### ExecutionStore Protocol

```python
class ExecutionStore(Protocol):
    def save_node_execution(self, flow_id, node_id, execution): ...
    def get_node_execution(self, flow_id, node_id) -> NodeExecution | None: ...
    def get_all_node_executions(self, flow_id) -> dict[UUID, NodeExecution]: ...
    def save_memory(self, flow_id, key, value): ...
    def get_memory(self, flow_id, key) -> Any: ...
    def get_all_memory(self, flow_id) -> dict[str, Any]: ...
    def delete_memory(self, flow_id, key): ...
    def save_messages(self, flow_id, node_id, messages): ...
    def get_messages(self, flow_id, node_id) -> list[Message]: ...
    def append_message(self, flow_id, node_id, message): ...
    def clear_flow(self, flow_id): ...
```

### InMemoryStore

Default store. All state is held in Python dictionaries. Fast, but lost when the process exits.

```python
from qm_engine.stores.memory_store import InMemoryStore

store = InMemoryStore()
runner = FlowRunner(graph=graph, node_registry=registry, store=store)
```

Best for: testing, short-lived scripts, development.

### SQLiteStore

Persists state to a SQLite database file. Survives process restarts.

```python
from qm_engine.stores.sqlite_store import SQLiteStore

store = SQLiteStore(db_path="./flow_state.db")
runner = FlowRunner(graph=graph, node_registry=registry, store=store)
```

Best for: local development, single-process deployments, CLI tools.

### Custom Stores

Implement the `ExecutionStore` protocol for any backend:

```python
class RedisStore:
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)

    def save_node_execution(self, flow_id, node_id, execution):
        key = f"flow:{flow_id}:node:{node_id}"
        self.client.set(key, execution.model_dump_json())

    # ... implement remaining methods ...
```

## Error Handling

Each node has an `error_handling` strategy and related configuration:

| Strategy | Behavior |
|----------|----------|
| `ErrorStrategy.STOP` | Halt the flow immediately. The node is marked as failed. Default. |
| `ErrorStrategy.RETRY` | Retry execution up to `max_retries` times with `retry_delay` between attempts. |
| `ErrorStrategy.SKIP` | Mark the node as skipped and continue to successors. Emits a recoverable `FlowError`. |
| `ErrorStrategy.CONTINUE` | Continue execution despite the error. |
| `ErrorStrategy.CUSTOM` | Halt with a custom error (same as STOP currently). |

### Retry Configuration

```python
from qm_graph.models import GraphNode
from qm_graph.enums import NodeType, ErrorStrategy

node = GraphNode(
    type=NodeType.INSTRUCTION,
    name="Unreliable API call",
    error_handling=ErrorStrategy.RETRY,
    max_retries=3,       # Try up to 3 additional times
    retry_delay=1.0,     # Wait 1 second between retries
    timeout=30.0,        # Abort if node takes > 30 seconds
)
```

### Error Event Handling

```python
def on_event(event):
    if isinstance(event, FlowError):
        if event.recoverable:
            print(f"Warning: {event.error} (continuing)")
        else:
            print(f"Fatal: {event.error}")

runner = FlowRunner(graph=graph, node_registry=registry, on_event=on_event)
```

## Node Timeout Enforcement

When a `GraphNode` has `timeout` set (in seconds), the engine wraps the executor call in `asyncio.wait_for`:

```python
node = GraphNode(
    type=NodeType.INSTRUCTION,
    name="Fast response",
    timeout=10.0,  # Must complete within 10 seconds
)
```

If the timeout expires, a `TimeoutError` is raised and handled according to the node's `error_handling` strategy.

## Traversal Gates

### TraverseInGate

Controls when a node is ready to execute:

- **AwaitAll** (default) -- Waits for all predecessor nodes to complete before executing.
- **AwaitFirst** -- Executes as soon as any one predecessor completes.

### TraverseOutGate

Controls which successor nodes are triggered after a node finishes:

- **SpawnAll** (default) -- Triggers all successor nodes (parallel fork).
- **SpawnPicked** -- Triggers only the successor whose edge label matches the node's output (decision routing).
- **SpawnNone** -- Does not trigger any successors (terminal).
- **SpawnStart** -- Loops back to the start node (used by loop nodes).

## Memory Management

Flow memory is a key-value store scoped to each execution:

```python
# The engine stores the initial user input automatically
store.save_memory(flow_id, "__user_input__", "user's message")

# Nodes can read and write memory via their execution context
memory = store.get_all_memory(flow_id)
store.save_memory(flow_id, "extracted_name", "Alice")
value = store.get_memory(flow_id, "extracted_name")

# Nodes can produce memory updates in their result
result = NodeResult(
    success=True,
    data={"memory_updates": {"key1": "value1", "key2": "value2"}},
    output_text="Done",
)
```

## Message Routing

The `MessageRouter` and `ContextManager` handle conversation history assembly:

1. For each node, the router collects output messages from all predecessor nodes.
2. The `ContextManager` applies truncation (max messages, token limits) to keep the context window manageable.
3. An input message is built based on the node's `MessageType` setting.
4. The complete message list is passed to the node executor.

## See Also

- [Architecture](architecture.md) -- System overview and data flow
- [Graph Building](graph-building.md) -- Building the graphs that the engine executes
- [Providers](providers.md) -- LLM providers called by instruction nodes
- [Security](security.md) -- Security considerations for execution
