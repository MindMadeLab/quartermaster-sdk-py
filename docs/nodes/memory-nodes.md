# Memory Nodes

Memory nodes provide persistent storage that survives beyond a single node
execution. They come in two scopes and three operations.

> See also: [Data Nodes](data-nodes.md) for variable computation and text
> rendering within a single thought.

---

## Memory scopes

### Flow memory

Flow memory is scoped to a **single flow execution**. Every node in the same
flow run can read from and write to it, but the data is discarded once the flow
completes. Use flow memory for intermediate results that only matter during the
current conversation turn.

### User memory

User memory is scoped to a **specific user** and persists **across flow
executions**. It survives between separate conversations, making it suitable for
preferences, accumulated history, or any state that should follow the user over
time.

---

## FlowMemoryNode

| | |
|---|---|
| **Type enum** | `FlowMemory` |
| **Class** | `FlowMemoryNode` |
| **Version** | 1.0 |

Declares a flow-scoped memory store. This node acts as a definition -- it tells
the engine to initialize a named memory space with optional seed data when the
flow starts. It does not execute any logic during `think()`.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `memory_name` | `str` | `"default"` | Unique name identifying this memory store |
| `initial_data` | `list` | `[]` | Seed data loaded when the flow starts |

### How it works

1. The node is registered in the graph with `accepts_incoming_edges=False` and
   `accepts_outgoing_edges=False` -- it is a standalone declaration.
2. The engine (via StartNode) reads `memory_name` and `initial_data` and
   provisions the memory store before any other node runs.
3. `think()` is a no-op.

### Common use cases

- Initialize a shopping cart, counter, or accumulator at flow start.
- Provide default configuration values for the flow.
- Declare a named memory that [WriteMemoryNode](#writememorynode) and
  [ReadMemoryNode](#readmemorynode) reference later.

---

## UserMemoryNode

| | |
|---|---|
| **Type enum** | `UserMemory1` |
| **Class** | `UserMemoryNode` |
| **Version** | 1.0 |

Declares a user-scoped memory store. Like FlowMemoryNode, this is a
declaration node that the engine provisions at startup. The key difference is
that data persists across flow executions for the same user.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `memory_name` | `str` | `"default"` | Unique name identifying this user memory store |
| `initial_data` | `list` | `[]` | Seed data loaded on first access (if memory does not yet exist) |

### How it works

1. Standalone declaration node (no incoming/outgoing edges).
2. The engine provisions user-scoped storage keyed by `memory_name` and user
   identity.
3. `initial_data` is applied only when the memory is created for the first time
   for that user; subsequent flow runs find the existing data intact.
4. `think()` is a no-op.

### Common use cases

- Store user preferences (language, timezone, display name).
- Maintain a running history of interactions.
- Track long-lived state such as loyalty points or onboarding progress.

---

## ReadMemoryNode

| | |
|---|---|
| **Type enum** | `ReadMemory1` |
| **Class** | `ReadMemoryNode` |
| **Version** | 1.0 |

Reads one or more variables from a persistent memory store and merges them into
the current thought metadata, making them available to downstream nodes.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `memory_name` | `str` | `"default"` | Name of the memory store to read from |
| `memory_type` | `str` | `"flow"` | Scope: `"flow"` or `"user"` |
| `variable_names` | `list` | `[]` | List of variable names to load; empty = load all |

### How it works

1. Retrieves the `_memory_reader` callback from the node context.
2. Calls `memory_reader(memory_name, memory_type, variable_names, ctx)`.
3. Merges the returned dict into thought metadata via
   `ctx.handle.update_metadata(variables)`.

### Read pattern example

```
ReadMemoryNode
  memory_name: "cart"
  memory_type: "flow"
  variable_names: ["items", "total"]
    -> TextNode (text: "Your cart has {{ items|length }} items, total: ${{ total }}")
```

### Common use cases

- Load previously stored results before an LLM summarization step.
- Retrieve user preferences at the start of a flow.
- Access flow-level counters or accumulators.

---

## WriteMemoryNode

| | |
|---|---|
| **Type enum** | `WriteMemory1` |
| **Class** | `WriteMemoryNode` |
| **Version** | 1.0 |

Evaluates expressions against thought metadata and writes the results to a
persistent memory store. Each variable is defined as a name/expression pair.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `memory_name` | `str` | `"default"` | Name of the memory store to write to |
| `memory_type` | `str` | `"flow"` | Scope: `"flow"` or `"user"` |
| `variables` | `list` | `[]` | List of `{"name": "...", "expression": "..."}` objects |

### How it works

1. Iterates over the `variables` list.
2. For each entry, evaluates `expression` against thought metadata (using the
   `_expression_evaluator` or restricted `eval()`).
3. Collects results into a dict keyed by `name`.
4. Calls `memory_writer(memory_name, memory_type, data, ctx)` to persist.

### Write pattern example

```
VarNode (name="order_total", expression="price * qty")
  -> WriteMemoryNode
       memory_name: "session"
       memory_type: "flow"
       variables:
         - { name: "total", expression: "order_total" }
         - { name: "timestamp", expression: "'2024-01-15'" }
```

### Common use cases

- Persist computed values so later nodes (or future flows) can read them.
- Save LLM-generated results to user memory for long-term recall.
- Store intermediate calculations in flow memory for a downstream merge.

---

## UpdateMemoryNode

| | |
|---|---|
| **Type enum** | `UpdateMemory1` |
| **Class** | `UpdateMemoryNode` |
| **Version** | 1.0 |

Modifies existing variables in a persistent memory store. Structurally
identical to WriteMemoryNode but calls `_memory_updater` instead of
`_memory_writer`, signaling to the engine that these are partial updates rather
than full overwrites.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `memory_name` | `str` | `"default"` | Name of the memory store to update |
| `memory_type` | `str` | `"flow"` | Scope: `"flow"` or `"user"` |
| `variables` | `list` | `[]` | List of `{"name": "...", "expression": "..."}` objects |

### How it works

1. Same expression evaluation loop as WriteMemoryNode.
2. Calls `memory_updater(memory_name, memory_type, data, ctx)` -- the engine
   merges the new values into the existing memory rather than replacing it.

### Update pattern example

```
ReadMemoryNode (memory_name="stats", variable_names=["visit_count"])
  -> VarNode (name="new_count", expression="visit_count + 1")
  -> UpdateMemoryNode
       memory_name: "stats"
       memory_type: "user"
       variables:
         - { name: "visit_count", expression: "new_count" }
```

### Common use cases

- Increment counters or scores.
- Append items to an existing list.
- Patch specific fields without affecting the rest of the memory store.

---

## End-to-end example: Initialize, write, read

The following graph shows the full memory lifecycle within a single flow.

```
FlowMemoryNode (memory_name="order", initial_data=[])
   (standalone -- no edges)

StartNode
  -> VarNode (name="item", expression="'Widget'")
  -> VarNode (name="price", expression="9.99")
  -> WriteMemoryNode
       memory_name: "order"
       memory_type: "flow"
       variables:
         - { name: "item",  expression: "item" }
         - { name: "price", expression: "price" }
  -> ... (other processing) ...
  -> ReadMemoryNode
       memory_name: "order"
       memory_type: "flow"
       variable_names: ["item", "price"]
  -> TextNode (text: "You ordered {{ item }} for ${{ price }}.")
```

**What happens:**

1. `FlowMemoryNode` declares an `"order"` memory store.
2. Two `VarNode` instances create `item` and `price` in thought metadata.
3. `WriteMemoryNode` persists those values to the `"order"` store.
4. Later, `ReadMemoryNode` loads them back into a fresh thought's metadata.
5. `TextNode` renders the final user-facing message.

For user-scoped persistence, replace `FlowMemoryNode` with `UserMemoryNode`
and set `memory_type: "user"` on the read/write/update nodes.
