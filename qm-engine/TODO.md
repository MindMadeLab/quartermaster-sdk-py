# qm-engine — Extraction TODO

The runtime execution engine for AI agent graphs. Takes a graph definition (from `qm-graph`), resolves node implementations (from `qm-nodes`), and orchestrates the execution: traversal, branching, merging, memory, message passing, and error handling. This is the brain that makes agent flows run.

## Source Files

Extract from `quartermaster/be/flows/`:

| Source File | Purpose |
|---|---|
| `models.py` | Flow, FlowNode, FlowEdge, FlowMemory Django models |
| `services/flow_context/base.py` | `BaseFlowContext` — execution context dataclass |
| `services/flow_context/protocol.py` | Protocol definitions / interfaces |
| `services/flow_context/flow_node_lifecycle.py` | Node execution lifecycle (before/after hooks) |
| `services/flow_context/thought_factory.py` | Thought creation and management |
| `services/v1/traverse_in.py` | `TraverseIn` — synchronization gate (wait for predecessors) |
| `services/v1/traverse_out.py` | `TraverseOut` — branching gate (spawn successors) |
| `services/v2/traverse_out.py` | `TraverseOutV2` — single-task runner variant |
| `services/v2/__init__.py` | `SingleTaskRunner(TraverseIn, TraverseOutV2)` |
| `services/graph/flow_edge.py` | Edge management utilities |
| `services/graph/inline_nodes.py` | Inline node handling |
| `services/memory/flow_memory.py` | `FlowMemoryService` — flow-scoped variable storage |
| `services/memory/user_memory.py` | `UserAgentMemoryService` — persistent user memory |
| `services/messaging/messaging_service.py` | Internal message routing between nodes |
| `services/messaging/message_helpers.py` | Message formatting utilities |
| `tasks.py` | Celery tasks for async node execution |

Also references:
| Source File | Purpose |
|---|---|
| `quartermaster/be/thoughts/models.py` | `ThoughtMemory` — execution state tracking |
| `quartermaster/be/flows/services/agent_control.py` | Stop/pause running agents |
| `quartermaster/be/flows/services/hot_agents.py` | Real-time agent management |

## Extractability: 6/10

The hardest extraction. The flow engine is deeply coupled to Django ORM (FlowNode, FlowEdge stored in DB), Celery (async task dispatching for parallel branches), and the Thought system (execution state tracking). Requires significant refactoring to make framework-agnostic.

## Phase 1: Execution Context ✅

### 1.1 Replace FlowContext
- [x] Extract `BaseFlowContext` concept — `ExecutionContext` dataclass with flow_id, node_id, graph, messages, memory, metadata, callbacks

### 1.2 Node Status Tracking
- [x] `NodeStatus` enum: PENDING, RUNNING, WAITING_USER, WAITING_TOOL, FINISHED, FAILED, SKIPPED
- [x] `NodeExecution` dataclass with lifecycle methods (start, finish, fail, skip, wait_for_user, wait_for_tool)

### 1.3 Execution State Store
- [x] `ExecutionStore` protocol (pluggable storage)
- [x] `InMemoryStore` — default implementation (dict-based, with deepcopy isolation)
- [x] `SQLiteStore` — persistent implementation for local development
- [ ] `RedisStore` — high-performance implementation (optional extra)
- [ ] Platform can plug in Django ORM store (proprietary)

## Phase 2: Traversal Engine ✅

### 2.1 Traverse In (Synchronization Gate)
- [x] `TraverseInGate` with `AwaitAll` and `AwaitFirst` strategies
- [x] Handles: no predecessors, single predecessor, multiple predecessors, dead branches

### 2.2 Traverse Out (Branching Gate)
- [x] `TraverseOutGate` with `SpawnAll`, `SpawnNone`, `SpawnPickedNode`, `SpawnStart`
- [x] Matching by node name, UUID string, and edge label

### 2.3 Task Dispatcher
- [x] `TaskDispatcher` protocol
- [x] `SyncDispatcher` — execute immediately (single-threaded)
- [x] `ThreadDispatcher` — use threading for parallel branches
- [x] `AsyncDispatcher` — use asyncio for parallel branches
- [ ] Platform can plug in `CeleryDispatcher` (proprietary)

## Phase 3: Flow Runner ✅

### 3.1 Core Runner
- [x] `FlowRunner` with `run()`, `run_async()`, `resume()`, `stop()` methods
- [x] Pluggable store, dispatcher, context_manager, and node_registry

### 3.2 Execution Loop
- [x] Full execution loop: start → traverse_in → execute → traverse_out → dispatch → repeat

### 3.3 Flow Events (Streaming)
- [x] `NodeStarted`, `TokenGenerated`, `NodeFinished`, `FlowFinished`, `UserInputRequired`, `FlowError`

### 3.4 Error Handling
- [x] `Stop` — halt entire flow on error
- [x] `Retry` — retry node execution (configurable max retries)
- [x] `Skip` — skip this node, continue to successors
- [ ] `Custom` — invoke error handling sub-flow
- [x] Timeout enforcement per node

## Phase 4: Memory System ✅

### 4.1 Flow Memory (Scoped Variables)
- [x] `FlowMemory` with set/get/delete/list_keys/get_all/clear
- [x] Backed by ExecutionStore (pluggable persistence)

### 4.2 Persistent Memory (Cross-Flow)
- [x] `PersistentMemory` protocol
- [x] `InMemoryPersistence` — simple dict (for testing)
- [ ] `SQLitePersistence` — file-based persistence
- [ ] Platform provides: Django ORM + pgvector semantic search (proprietary)

## Phase 5: Message System ✅

### 5.1 Message Routing
- [x] `MessageRouter` with thought types: Skip, New, NewHidden, Inherit, Continue
- [x] Input message building based on MessageType: Automatic, User, Variable, Assistant

### 5.2 Context Window Management
- [x] `ContextManager` with truncation by count and tokens
- [x] System message preservation, most-recent-first strategy
- [x] Pluggable token counter

## Phase 6: Testing ✅

### 6.1 Unit Tests
- [x] Traverse In: test AwaitAll with 2 predecessors (1 done, both done)
- [x] Traverse In: test AwaitFirst with 2 predecessors (1 done = go)
- [x] Traverse Out: test SpawnAll with 3 successors
- [x] Traverse Out: test SpawnPickedNode with decision result
- [x] Traverse Out: test SpawnStart (loop)
- [x] Traverse Out: test SpawnNone (stop)
- [x] Memory: test set/get/delete/list
- [x] Context: test message truncation
- [x] Stores: InMemoryStore and SQLiteStore (shared test mixin)
- [x] MessageRouter: thought types and input message building

### 6.2 Integration Tests (Full Flow Execution)
- [x] Simple: Start → Instruction → End (mock LLM)
- [x] Decision: Start → Decision → [A, B] → End
- [x] Parallel: Start → Fork → [A, B, C] → Merge → End
- [x] Memory: Start → WriteMemory → ReadMemory → End
- [x] Error: Start → Instruction (fails) → error handling (Stop, Skip, Retry)
- [x] User input: Start → Instruction → User (waits) → resume → End
- [x] Event streaming verification
- [x] Flow stop
- [x] Loop: Start → Instruction → If (condition) → loop back or End
- [x] Sub-agent: Start → Agent (calls sub-flow) → End

### 6.3 Benchmark Tests
- [x] 10/50/100-node linear chain execution time
- [x] 5/10/20-branch parallel fan-out
- [x] Memory throughput (1000 ops)
- [x] Large message history (500 messages)

## Phase 7: Documentation ✅

### 7.1 README
- [x] Quick start: define a graph, run it, get output
- [x] FlowRunner API reference
- [x] Pluggable stores and dispatchers
- [x] Architecture diagram
- [x] Error handling guide

### 7.2 Architecture Guide
- [ ] Execution flow diagram (traverse_in → think → traverse_out)
- [ ] How parallel branches work
- [ ] How memory scoping works
- [ ] How to plug in custom stores (Redis, PostgreSQL, etc.)

### 7.3 Integration Guide
- [ ] Using with qm-graph (define the flow)
- [ ] Using with qm-nodes (implement the behavior)
- [ ] Using with qm-providers (call LLMs)
- [ ] Using with qm-tools (execute tools)
- [ ] Full example: build and run a RAG agent from scratch

## Phase 8: CI/CD & PyPI ✅

- [x] GitHub Actions: lint, typecheck, unit + integration tests (Python 3.11/3.12/3.13)
- [x] PyPI package config: `qm-engine` with hatchling
- [x] Optional extras: `qm-engine[sqlite]`, `qm-engine[redis]`, `qm-engine[all]`
- [ ] PyPI publishing workflow
- [ ] Dependencies: `qm-graph`, `qm-nodes`, `qm-providers`, `qm-tools` (when published)

## Architecture Notes

### Three Runner Modes (from QM source)
QM has three runner versions internally:
1. **V1 (Per-Node Celery)** — each node is a separate Celery task. Good for distributed execution but heavy overhead.
2. **V2 (Single-Task)** — entire flow runs in one task. Faster but no true parallelism.
3. **V3 (Hybrid)** — single-task with Celery for parallel branches.

For open-source, we implemented:
- `SyncDispatcher` (single-threaded, V2 equivalent) — simplest, great for testing and simple agents ✅
- `ThreadDispatcher` (V1 equivalent) — true parallel branches via threads ✅
- `AsyncDispatcher` (asyncio, V2 equivalent) — non-blocking, good for web apps (TODO)

### What Stays Proprietary
- Django ORM ExecutionStore (platform uses PostgreSQL)
- Celery TaskDispatcher (platform uses distributed workers)
- Real-time WebSocket streaming (Django Channels)
- Agent monitoring and observability dashboard
- Hot agent management (live pause/resume/inspect)
- Multi-tenant execution isolation

## Current Stats

- **132 tests**, all passing
- **90% code coverage**
- **Zero lint errors** (ruff)
- **Zero type errors** (mypy strict)
- **Python 3.11+** compatible
