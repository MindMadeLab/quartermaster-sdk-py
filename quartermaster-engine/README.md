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

### New in v0.4.0

- **Cooperative cancellation** -- `ctx.cancelled` flag + `Cancelled` exception for clean stream teardown.
- **Per-node tool scoping** -- `agent(tools=[...])` strictly enforced at the engine level; `tool_scope="permissive"` escape.
- **Inline `@tool` callables** -- `agent(tools=[my_func])` accepts bare callables alongside registry names.

## Installation

```bash
pip install quartermaster-engine

# With SQLite persistent store
pip install quartermaster-engine[sqlite]
```

## Quick Start

### High-level path — `provider_registry`

The simplest way to run a graph: hand `FlowRunner` a provider registry and let
it build the default node registry (covering every node type the bundled DSL
emits — including `AgentExecutor` for `agent()` nodes with the canonical
Quartermaster tool loop):

```python
from quartermaster_engine import FlowRunner
from quartermaster_graph import Graph
from quartermaster_providers import register_local

provider_registry = register_local(
    "ollama",
    base_url="http://localhost:11434",   # or set $OLLAMA_HOST
    default_model="gemma4:26b",
)

graph = Graph("chat").user().agent().end().build()
runner = FlowRunner(graph=graph, provider_registry=provider_registry)
result = runner.run("Pozdravljen!")
print(result.success, result.final_output)
```

If you have a `quartermaster_tools.ToolRegistry`, pass it as
`tool_registry=` so `AgentExecutor` can actually execute the tools your
graph's `.agent(tools=[...])` nodes request.

### Low-level path — bring your own `node_registry`

For full control over which executor handles each node type (custom
executors, alternative storage, etc.) hand `FlowRunner` a
`SimpleNodeRegistry` directly. Use the helper
`build_default_registry(provider_registry)` if you only want to swap one
executor; pass the result back into `FlowRunner(node_registry=...)`.

```python
from uuid import uuid4
from quartermaster_engine import FlowRunner, InMemoryStore
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeResult
from quartermaster_graph import GraphSpec, GraphNode, GraphEdge, NodeType

# 1. Define the graph
start_id, process_id, end_id = uuid4(), uuid4(), uuid4()

graph = GraphSpec(
    id=uuid4(),
    agent_id=uuid4(),
    start_node_id=start_id,
    nodes=[
        GraphNode(id=start_id, type=NodeType.START, name="Start"),
        GraphNode(
            id=process_id,
            type=NodeType.INSTRUCTION,
            name="Process",
            metadata={"llm_system_instruction": "Summarize the input.", "llm_model": "gpt-4o"},
        ),
        GraphNode(id=end_id, type=NodeType.END, name="End"),
    ],
    edges=[
        GraphEdge(source_id=start_id, target_id=process_id),
        GraphEdge(source_id=process_id, target_id=end_id),
    ],
)

# 2. Register node executors
registry = SimpleNodeRegistry()
# registry.register("Instruction1", my_instruction_executor)

# 3. Run the flow
runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("Please summarize this article about AI safety.")

print(result.success)          # True/False
print(result.final_output)     # The final text output
print(result.duration_seconds)
```

### Using GraphBuilder from quartermaster-graph

```python
from quartermaster_graph import GraphBuilder
from quartermaster_engine import FlowRunner
from quartermaster_engine.nodes import SimpleNodeRegistry

graph = (
    GraphBuilder("Support Agent")
    .instruction("Classify", model="gpt-4o")
    .decision("Route", options=["billing", "technical"])
    .on("billing").instruction("Handle billing").end()
    .on("technical").instruction("Handle technical").end()
    .build()
)

registry = SimpleNodeRegistry()
runner = FlowRunner(graph=graph, node_registry=registry)
result = runner.run("I need help with my invoice")
```

### Stream Events in Real Time (low-level)

The engine's native streaming surface is a single `on_event` callback
that fires for every `FlowEvent`. This is the primitive layer; most
application code uses the SDK's chunk filters layered on top (see
below).

```python
from quartermaster_engine import (
    FlowRunner, FlowEvent,
    NodeStarted, NodeFinished, TokenGenerated,
    ToolCallStarted, ToolCallFinished,
    ProgressEvent, CustomEvent,
    FlowError,
)

def handle_event(event: FlowEvent):
    if isinstance(event, NodeStarted):
        print(f"[START] {event.node_name} ({event.node_type})")
    elif isinstance(event, TokenGenerated):
        print(event.token, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n[TOOL ->] {event.tool}({event.arguments})")
    elif isinstance(event, ToolCallFinished):
        print(f"[TOOL <-] {event.tool} = {event.result!r}")
    elif isinstance(event, ProgressEvent):        # new in v0.3.0
        print(f"[PROGRESS] {event.message} ({event.percent})")
    elif isinstance(event, CustomEvent):          # new in v0.3.0
        print(f"[{event.name}] {event.payload}")
    elif isinstance(event, NodeFinished):
        print(f"\n[DONE] Node finished: {event.result[:50]}...")
    elif isinstance(event, FlowError):
        print(f"[ERROR] {event.error} (recoverable={event.recoverable})")

runner = FlowRunner(graph=graph, node_registry=registry, on_event=handle_event)
result = runner.run("Hello!")
```

### SDK chunk filters (high-level)

The `quartermaster_sdk` package translates each `FlowEvent` into a
typed `Chunk` via `_event_to_chunk` and wraps the iterator so consumers
can filter by chunk family instead of writing `isinstance` ladders:

```python
import quartermaster_sdk as qm

for token in qm.run.stream(graph, "Hello!").tokens():            # TokenGenerated
    print(token, end="")

for call in qm.run.stream(graph, "Research x").tool_calls():    # ToolCallStarted
    ui.tool_card(call.tool, call.args)

for prog in qm.run.stream(graph, "Crunch").progress():          # ProgressEvent
    ui.status(prog.message, prog.percent)

for evt in qm.run.stream(graph, "Research").custom(name="src"): # CustomEvent by name
    ui.add(evt.payload)
```

### Post-mortem trace

`qm.run(...)` also attaches a structured `Trace` to the returned
`Result`, built from the same `FlowEvent` stream after the run
finishes:

```python
result = qm.run(graph, "Hello!")
result.trace.text                        # concatenated TokenGenerated.token
result.trace.tool_calls                  # list[dict] from every ToolCallFinished
result.trace.progress                    # list[ProgressEvent]
result.trace.by_node["Researcher"].text  # per-node slice
print(result.trace.as_jsonl())
```

### Async Execution

```python
import asyncio
from quartermaster_engine import FlowRunner, TokenGenerated

async def run_flow():
    runner = FlowRunner(graph=graph, node_registry=registry)
    async for event in runner.run_async("Hello!"):
        if isinstance(event, TokenGenerated):
            print(event.token, end="", flush=True)

asyncio.run(run_flow())
```

### Quick Execution with `run_graph()`

For rapid prototyping, use the convenience function:

```python
from quartermaster_engine import run_graph

# Non-interactive (provide input)
run_graph(agent, user_input="Explain quantum computing")

# Interactive (prompts stdin at User nodes)
run_graph(agent)

# Force provider
run_graph(agent, user_input="Hello", provider="openai")
```

`run_graph()` handles provider detection, node registration, streaming output,
and the pause/resume cycle for interactive User nodes.

## API Reference

### FlowRunner

The core orchestration class. Accepts a `GraphSpec` from `quartermaster-graph` (`AgentGraph` still works as a deprecated alias).

```python
from quartermaster_engine import FlowRunner
from quartermaster_engine.dispatchers.sync_dispatcher import SyncDispatcher
from quartermaster_engine.messaging.context_manager import ContextManager

runner = FlowRunner(
    graph=spec,                      # GraphSpec from quartermaster-graph
    node_registry=registry,          # Maps node types to executors
    store=InMemoryStore(),           # Execution state storage
    dispatcher=SyncDispatcher(),     # How branches are dispatched
    context_manager=ContextManager(),  # LLM context window management
    on_event=handle_event,           # Real-time event callback
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
| `NodeStarted` | `flow_id`, `node_id`, `node_type`, `node_name` | Node begins execution |
| `TokenGenerated` | `flow_id`, `node_id`, `token` | Streaming token from LLM |
| `ToolCallStarted` | `flow_id`, `node_id`, `tool`, `arguments`, `iteration` | Agent dispatched a tool call |
| `ToolCallFinished` | `flow_id`, `node_id`, `tool`, `arguments`, `result`, `raw`, `error`, `iteration` | Tool call completed (or errored) |
| `ProgressEvent` (v0.3.0) | `flow_id`, `node_id`, `message`, `percent`, `data` | `ctx.emit_progress(...)` from app code |
| `CustomEvent` (v0.3.0) | `flow_id`, `node_id`, `name`, `payload` | `ctx.emit_custom(...)` from app code |
| `NodeFinished` | `flow_id`, `node_id`, `result`, `output_data` | Node completed |
| `FlowFinished` | `flow_id`, `final_output`, `output_data` | Entire flow completed |
| `UserInputRequired` | `flow_id`, `node_id`, `prompt`, `options` | Flow paused for user input |
| `FlowError` | `flow_id`, `node_id`, `error`, `recoverable` | Node failed |

`run_graph()` uses streaming by default -- `TokenGenerated` events are printed
as they arrive, giving real-time output from LLM nodes without extra setup.
`ProgressEvent` and `CustomEvent` fire whenever application code calls
`ExecutionContext.emit_progress(...)` / `emit_custom(...)`, reachable from
inside tools via `quartermaster_engine.context.current_context()`.

### ExecutionContext

The runtime context passed to each node executor:

| Field | Type | Description |
|-------|------|-------------|
| `flow_id` | `UUID` | Flow execution identifier |
| `node_id` | `UUID` | Current node identifier |
| `graph` | `GraphSpec` | Full graph definition |
| `current_node` | `GraphNode` | Current node definition |
| `messages` | `list[Message]` | Conversation history |
| `memory` | `dict[str, Any]` | Flow-scoped memory snapshot |
| `metadata` | `dict[str, Any]` | Node metadata |
| `on_token` | `Callable[[str], None] \| None` | Callback for streaming tokens |

Helper methods and properties:

| Method / Property | Description |
|--------|-------------|
| `get_meta(key, default=None)` | Get value from node metadata, falling back to context metadata |
| `set_meta(key, value)` | Set a metadata value on this context |
| `emit_token(token)` | Emit a streaming token via callback |
| `emit_message(content)` | Emit a complete message via callback |
| `cancelled` (property, v0.4.0) | `True` when the flow has been asked to stop (cooperative cancellation) |

### Node Execution Protocol

Implement `NodeExecutor` to add custom node types:

```python
from quartermaster_engine.nodes import NodeExecutor, NodeResult
from quartermaster_engine.context.execution_context import ExecutionContext

class MyInstructionExecutor:
    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("llm_system_instruction", "")
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

Register executors with `SimpleNodeRegistry`:

```python
from quartermaster_engine.nodes import SimpleNodeRegistry

registry = SimpleNodeRegistry()
registry.register("Instruction1", MyInstructionExecutor())
registry.register("Decision1", MyDecisionExecutor())

# List registered types
registry.list_types()  # ["Instruction1", "Decision1"]
```

### NodeResult

Returned by node executors after execution:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether execution succeeded |
| `data` | `dict[str, Any]` | Structured output data |
| `error` | `str \| None` | Error message if failed |
| `picked_node` | `str \| None` | For decision nodes: which successor to trigger |
| `output_text` | `str \| None` | Main text output |
| `wait_for_user` | `bool` | If True, flow pauses for user input |
| `user_prompt` | `str \| None` | Prompt to show the user |
| `user_options` | `list[str] \| None` | Options for user selection |

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
