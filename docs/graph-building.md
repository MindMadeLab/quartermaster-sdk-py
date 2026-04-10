# Graph Building

Agent workflows in Quartermaster are represented as directed graphs. The `quartermaster-graph` package provides a fluent builder API, Pydantic-based models, and a validation system for constructing and verifying these graphs.

## Core Concepts

A graph consists of:
- **Nodes** (`GraphNode`) -- Units of work (LLM calls, decisions, user input, etc.)
- **Edges** (`GraphEdge`) -- Directed connections between nodes, optionally labeled
- **AgentVersion** -- A versioned snapshot containing all nodes and edges, with a designated start node

## GraphBuilder API

The `GraphBuilder` class provides a fluent (chainable) interface for constructing graphs programmatically.

### Creating a Builder

```python
from quartermaster_graph import GraphBuilder

builder = GraphBuilder("My Agent", description="Processes user queries")
```

### Linear Chains

The simplest pattern is a linear sequence of nodes:

```python
graph = (
    GraphBuilder("Simple Chain")
    .start()
    .instruction("Step 1", model="gpt-4o", system_instruction="Summarize the input.")
    .instruction("Step 2", model="gpt-4o", system_instruction="Translate to French.")
    .end()
    .build()
)
```

### Decision Branching

Use `.decision()` followed by `.on(label)` to create conditional branches:

```python
graph = (
    GraphBuilder("Router")
    .start()
    .instruction(
        "Classify",
        model="gpt-4o",
        system_instruction="Classify as: Technical, Creative, or General.",
    )
    .decision("Category?", options=["Technical", "Creative", "General"])
    .on("Technical")
        .instruction("Tech response", system_instruction="Give a technical answer.")
        .end()
    .on("Creative")
        .instruction("Creative response", system_instruction="Give a creative answer.")
        .end()
    .on("General")
        .instruction("General response", system_instruction="Give a general answer.")
        .end()
    .build()
)
```

### If/Else Branching

For boolean conditions, use `.if_node()`:

```python
graph = (
    GraphBuilder("Conditional")
    .start()
    .if_node("Check length", expression="len(input) > 100", variable="input")
    .on("true")
        .instruction("Summarize", system_instruction="The input is long. Summarize it.")
        .end()
    .on("false")
        .instruction("Expand", system_instruction="The input is short. Elaborate on it.")
        .end()
    .build()
)
```

### Merging Branches

Use `.merge()` and `.merge_to()` to rejoin branches:

```python
builder = GraphBuilder("Merge Example")
builder.start()
builder.decision("Route?", options=["A", "B"])

merge_node = builder.merge("Combine results")

(builder
    .on("A")
        .instruction("Path A")
        .merge_to(merge_node.id))

(builder
    .on("B")
        .instruction("Path B")
        .merge_to(merge_node.id))

# Continue after merge
builder.instruction("Final step")
builder.end()
graph = builder.build()
```

### Parallel Execution

Fork execution into parallel branches:

```python
graph = (
    GraphBuilder("Parallel")
    .start()
    .parallel("Fork")
    # Parallel branches are connected via manual edges or the on() pattern
    .build(validate=False)  # Parallel graphs may need manual edge wiring
)
```

### Other Node Types

```python
# Static content (no LLM call)
builder.static("Welcome", content="Hello! How can I help you today?")

# User input (pauses execution, waits for user response)
builder.user("Ask for clarification")

# Code execution
builder.code("Calculate", code="result = 2 + 2", language="python")

# Tool invocation
builder.tool("Search", tool_name="web_search", query="latest news")

# Sub-agent delegation
builder.sub_agent("Delegate", agent_id="agent-uuid-here")

# Loop
builder.loop("Retry loop", max_iterations=5, break_condition="success == True")

# Generic node (any NodeType)
from quartermaster_graph.enums import NodeType
builder.node(NodeType.SUMMARIZE, name="Summarize conversation")
```

### Manual Edge Wiring

For complex topologies, add edges explicitly:

```python
builder.edge(source_id=node_a.id, target_id=node_b.id, label="custom label")
```

### Build and Validate

```python
# Build with validation (default)
graph = builder.build()

# Build with a specific version tag
graph = builder.build(version="1.2.0")

# Skip validation (useful during development)
graph = builder.build(validate=False)
```

## Node Types Reference

The `NodeType` enum defines all available node types. Here is the complete list organized by category:

### Core Flow Control

| Type | Enum Value | Description |
|------|-----------|-------------|
| `START` | `Start1` | Entry point of the graph. Exactly one per graph. |
| `END` | `End1` | Terminal node. At least one per graph. |
| `DECISION` | `Decision1` | LLM-powered branching based on classification. |
| `STATIC_DECISION` | `StaticDecision1` | Rule-based branching (no LLM). |
| `USER_DECISION` | `UserDecision1` | User selects the branch to follow. |
| `IF` | `If1` | Boolean condition evaluation. |
| `SWITCH` | `Switch1` | Multi-way branching on a variable value. |
| `MERGE` | `Merge1` | Waits for all incoming branches, then continues. |
| `STATIC_MERGE` | `StaticMerge1` | Merge with static content injection. |
| `BREAK` | `Break1` | Exit a loop early. |
| `LOOP` | `Loop1` | Repeat a subgraph up to N iterations. |
| `PARALLEL` | `Parallel1` | Fork execution into concurrent branches. |

### LLM Instruction

| Type | Enum Value | Description |
|------|-----------|-------------|
| `INSTRUCTION` | `Instruction1` | Standard LLM call with system instruction. |
| `INSTRUCTION_IMAGE_VISION` | `InstructionImageVision1` | LLM call with image/vision input. |
| `INSTRUCTION_PARAMETERS` | `InstructionParameters1` | LLM call with structured parameters. |
| `INSTRUCTION_PROGRAM` | `InstructionProgram1` | LLM call that produces tool/program calls. |
| `INSTRUCTION_PROGRAM_PARAMETERS` | `InstructionProgramParameters1` | Combined instruction + tool parameters. |
| `REASONING` | `Reasoning1` | Extended thinking / chain-of-thought mode. |
| `SUMMARIZE` | `Summarize1` | Summarize conversation or content. |

### User Interaction

| Type | Enum Value | Description |
|------|-----------|-------------|
| `USER` | `User1` | Pause and wait for user text input. |
| `USER_FORM` | `UserForm1` | Present a structured form to the user. |
| `USER_PROGRAM_FORM` | `UserProgramForm1` | User form with program/tool parameters. |

### Memory and Variables

| Type | Enum Value | Description |
|------|-----------|-------------|
| `FLOW_MEMORY` | `FlowMemory1` | Access flow-scoped memory. |
| `READ_MEMORY` | `ReadMemory1` | Read a value from memory. |
| `WRITE_MEMORY` | `WriteMemory1` | Write a value to memory. |
| `UPDATE_MEMORY` | `UpdateMemory1` | Update an existing memory value. |
| `USER_MEMORY` | `UserMemory1` | Access user-scoped persistent memory. |
| `VAR` | `Var1` | Set or read a variable. |
| `TEXT_TO_VARIABLE` | `TextToVariable1` | Extract text into a named variable. |

### Content and Code

| Type | Enum Value | Description |
|------|-----------|-------------|
| `STATIC` | `Static1` | Inject static text content. |
| `TEXT` | `Text1` | Text processing node. |
| `CODE` | `Code1` | Execute code (Python, JS, etc.). |
| `PROGRAM_RUNNER` | `ProgramRunner1` | Run an external program/tool. |
| `TEMPLATE` | `Template1` | Render a template with variables. |

### Integration and Agents

| Type | Enum Value | Description |
|------|-----------|-------------|
| `SUB_ASSISTANT` | `SubAssistant1` | Delegate to a sub-assistant. |
| `SUB_AGENT` | `SubAgent1` | Delegate to a sub-agent graph. |
| `TOOL` | `Tool1` | Invoke a registered tool. |
| `API_CALL` | `ApiCall1` | Make an external API call. |
| `WEBHOOK` | `Webhook1` | Trigger or receive a webhook. |

### Utility

| Type | Enum Value | Description |
|------|-----------|-------------|
| `AGENT` | `Agent1` | Agent identity node. |
| `BLANK` | `Blank1` | No-op placeholder. |
| `COMMENT` | `Comment1` | Documentation node (not executed). |
| `TIMER` | `Timer1` | Delay execution for a duration. |
| `VALIDATOR` | `Validator1` | Validate data against rules. |
| `TRANSFORMER` | `Transformer1` | Transform data between formats. |
| `FILTER` | `Filter1` | Filter data based on criteria. |
| `AGGREGATOR` | `Aggregator1` | Aggregate data from multiple sources. |
| `ROUTER` | `Router1` | Route messages to different handlers. |
| `ERROR_HANDLER` | `ErrorHandler1` | Custom error handling logic. |
| `LOG` | `Log1` | Log data for debugging. |
| `NOTIFICATION` | `Notification1` | Send a notification. |
| `CUSTOM` | `Custom1` | User-defined custom node type. |
| `VIEW_METADATA` | `ViewMetadata1` | Inspect node metadata. |
| `USE_ENVIRONMENT` | `UseEnvironment1` | Load environment configuration. |
| `UNSELECT_ENVIRONMENT` | `UnselectEnvironment1` | Unload environment configuration. |
| `USE_FILE` | `UseFile1` | Load a file into the context. |

## Edge Types and Traversal

### TraverseIn -- How Nodes Handle Incoming Edges

| Strategy | Enum Value | Behavior |
|----------|-----------|----------|
| `AWAIT_ALL` | `AwaitAll` | Wait for all predecessors to complete before executing. Default. |
| `AWAIT_FIRST` | `AwaitFirst` | Execute as soon as any one predecessor completes. |

### TraverseOut -- How Nodes Dispatch to Successors

| Strategy | Enum Value | Behavior |
|----------|-----------|----------|
| `SPAWN_ALL` | `SpawnAll` | Trigger all successor nodes. Default. |
| `SPAWN_PICKED` | `SpawnPickedNode` | Trigger only the successor matching the result (decisions). |
| `SPAWN_NONE` | `SpawnNone` | Do not trigger any successors (terminal). |
| `SPAWN_START` | `SpawnStart` | Loop back to the start node. |

### Edge Labels

Edges from Decision, If, and Switch nodes carry labels that determine which branch to follow. The engine matches the node's output against edge labels to route execution.

## Validation Rules

The `validate_graph()` function checks these rules before execution:

1. **Exactly one Start node** -- The graph must have a single `NodeType.START` node.
2. **At least one End node** -- The graph must terminate somewhere.
3. **Valid start_node_id** -- Must reference an existing Start-type node.
4. **No orphan nodes** -- All nodes must be reachable from the Start node (except Comment nodes).
5. **No cycles** -- The graph must be a DAG. Cycles involving Loop nodes produce a warning instead of an error.
6. **Valid edge references** -- All edge source and target IDs must reference existing nodes.
7. **Decision/If/Switch edge labels** -- Branching nodes with multiple outgoing edges must have labeled edges.

## GraphNode Configuration

Each `GraphNode` carries configuration beyond its type:

```python
from quartermaster_graph.models import GraphNode
from quartermaster_graph.enums import NodeType, ErrorStrategy, ThoughtType, TraverseIn, TraverseOut

node = GraphNode(
    type=NodeType.INSTRUCTION,
    name="My Node",
    traverse_in=TraverseIn.AWAIT_ALL,       # Wait for all predecessors
    traverse_out=TraverseOut.SPAWN_ALL,      # Trigger all successors
    thought_type=ThoughtType.NEW,            # Create a new thought
    error_handling=ErrorStrategy.RETRY,      # Retry on failure
    max_retries=3,                           # Up to 3 retries
    retry_delay=1.0,                         # 1 second between retries
    timeout=30.0,                            # 30 second timeout
    metadata={                               # Node-specific configuration
        "model": "gpt-4o",
        "temperature": 0.7,
        "system_instruction": "You are a helpful assistant.",
    },
)
```

## See Also

- [Architecture](architecture.md) -- System overview and data flow
- [Engine](engine.md) -- How the engine executes graphs
- [Providers](providers.md) -- LLM provider configuration for instruction nodes
- [Tools](tools.md) -- Tool definitions for tool nodes
