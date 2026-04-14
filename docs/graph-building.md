# Graph Building

Agent workflows in Quartermaster are represented as directed graphs. The `quartermaster-graph` package provides a fluent builder API, Pydantic-based models, and a validation system for constructing and verifying these graphs.

## Core Concepts

A graph consists of:
- **Nodes** (`GraphNode`) -- Units of work (LLM calls, decisions, user input, etc.)
- **Edges** (`GraphEdge`) -- Directed connections between nodes, optionally labeled
- **GraphSpec** -- The full graph definition: nodes, edges, and the start node. (`AgentGraph` and `AgentVersion` remain as deprecated aliases.)

## Graph Builder API

The `Graph` class (alias for `GraphBuilder`) provides a fluent (chainable) interface for constructing graphs programmatically. The builder itself IS the graph -- you can access `.nodes` and `.edges` directly without calling `.build()`.

### Creating a Graph

```python
from quartermaster_graph import Graph

graph = Graph("My Agent", description="Processes user queries")
```

### Linear Chains

The simplest pattern is a linear sequence of nodes. Every graph should have a `.user()` node after `.start()` to collect input:

```python
graph = (
    Graph("Simple Chain")
    .start()
    .user("Enter text to process")
    .instruction("Step 1", model="gpt-4o", system_instruction="Summarize the input.")
    .instruction("Step 2", model="gpt-4o", system_instruction="Translate to French.")
    .end()
)
```

### Decision Branching

Use `.decision()` followed by `.on(label)` to create conditional branches.
Decision uses an LLM to pick ONE path — only one branch executes.
No merge is needed after a decision; branches converge directly on the
next node.

```python
graph = (
    Graph("Router")
    .start()
    .user("What would you like help with?")
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
    # No .merge() -- only one branch fires, they converge on .end()
    .end()
)
```

### If/Else Branching

For boolean conditions, use `.if_node()`:

```python
graph = (
    Graph("Conditional")
    .start()
    .user("Enter your text")
    .if_node("Check length", expression="len(input) > 100")
    .on("true")
        .instruction("Summarize", system_instruction="The input is long. Summarize it.")
        .end()
    .on("false")
        .instruction("Expand", system_instruction="The input is short. Elaborate on it.")
        .end()
    .end()
)
```

### Merging Parallel Branches

Merge nodes rejoin **parallel** branches (where ALL branches run
concurrently). Do NOT use merge after decision or if nodes — those pick
only one branch, so there is nothing to merge.

Two flavours:

* **`.static_merge()`** — joins branch outputs into one context without
  calling an LLM. Use this when the **next** node is an LLM instruction
  that will work with all the outputs. This is the most common case.
* **`.merge()`** — calls an LLM to *compress* multiple branch outputs
  into a single coherent message. Use only when you need the merge itself
  to synthesize/summarize before continuing.

```python
graph = (
    Graph("Parallel with Merge")
    .start()
    .user("Enter your request")
    .instruction("Prepare")
    .parallel()
    .branch()
        .instruction("Path A")
    .end()
    .branch()
        .instruction("Path B")
    .end()
    .static_merge("Combine results")  # joins parallel outputs into one context
    .instruction("Final step")        # this LLM sees both outputs
    .end()
)
```

### Parallel Execution

Fork execution into concurrent branches with `.parallel()`, define each
branch with `.branch()`, and rejoin with `.static_merge()`:

```python
graph = (
    Graph("Parallel Research")
    .start()
    .user("Paste your code for review")
    .instruction("Prepare", system_instruction="Prepare the research task")
    .parallel()
    .branch()
        .instruction("Security audit", system_instruction="Check for security issues")
    .end()
    .branch()
        .instruction("Performance audit", system_instruction="Check for performance issues")
    .end()
    .branch()
        .instruction("Code quality", system_instruction="Review code quality")
    .end()
    .static_merge("Collect all audits")
    .instruction("Final report", system_instruction="Combine all audit results into a report")
    .end()
)
```

All three branches run concurrently. `.static_merge()` waits for all to
complete and joins their outputs. The next instruction node sees all
three outputs in its context.

### Other Node Types

```python
# User input (pauses execution, waits for user response)
graph.user("Ask for clarification")

# Static content (no LLM call)
graph.static("Welcome", text="Hello! How can I help you today?")

# Text template with Jinja2 interpolation
graph.text("Greeting", template="Hello {{user_name}}, welcome!")

# Variable capture and expression evaluation
graph.var("Capture name", variable="user_name", expression="input.strip()")

# Code execution
graph.code("Calculate", code="result = 2 + 2", filename="calc.py")

# Sub-agent delegation
graph.sub_agent("Delegate", graph_id="agent-uuid-here")

# Agent node (autonomous tool-use loop)
graph.agent("Researcher", model="gpt-4o", tools=["web-search-v1"], max_iterations=10)

# Summarize conversation
graph.summarize("Summary", system_instruction="Summarize the conversation.")

# Generic node (any NodeType)
from quartermaster_graph.enums import NodeType
graph.node(NodeType.SUMMARIZE, name="Summarize conversation")
```

## Node Types Reference

The `NodeType` enum defines all available node types. Here is the complete list organized by category:

### Core Flow Control

| Type | Enum Value | Description |
|------|-----------|-------------|
| `START` | `Start1` | Entry point of the graph. Exactly one per graph. |
| `END` | `End1` | Terminal node. At least one per graph. |
| `DECISION` | `Decision1` | LLM picks ONE branch via tool call (SpawnPicked). No merge needed after. |
| `STATIC_DECISION` | `StaticDecision1` | Expression-based branching, no LLM (SpawnPicked). |
| `USER_DECISION` | `UserDecision1` | User selects which branch to follow (SpawnPicked). |
| `IF` | `If1` | Boolean expression -> true/false branch (SpawnPicked). No merge needed. |
| `SWITCH` | `Switch1` | Multi-way expression branching, first match wins (SpawnPicked). |
| `MERGE` | `Merge1` | LLM combines parallel branch outputs into one message. Use after parallel(). |
| `STATIC_MERGE` | `StaticMerge1` | Joins parallel branch outputs without LLM. Use after parallel(). |
| `BREAK` | `Break1` | Message collection boundary -- stops backward context traversal. |

### LLM Nodes

| Type | Enum Value | Description |
|------|-----------|-------------|
| `INSTRUCTION` | `Instruction1` | Standard LLM call with system instruction (streaming). |
| `AGENT` | `Agent1` | Autonomous agentic loop WITH tools, iterates up to max_iterations. |
| `INSTRUCTION_IMAGE_VISION` | `InstructionImageVision1` | LLM call with image/vision input. |
| `INSTRUCTION_PARAMETERS` | `InstructionParameters1` | LLM call with structured parameter extraction. |
| `INSTRUCTION_PROGRAM` | `InstructionProgram1` | LLM call with tool execution. |
| `INSTRUCTION_PROGRAM_PARAMETERS` | `InstructionProgramParameters1` | Combined instruction + tool + structured parameters. |

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
| `FLOW_MEMORY` | `FlowMemory1` | Define flow-scoped memory (standalone declaration). |
| `USER_MEMORY` | `UserMemory1` | Define user-scoped persistent memory (standalone declaration). |
| `READ_MEMORY` | `ReadMemory1` | Read variables from a memory store into thought metadata. |
| `WRITE_MEMORY` | `WriteMemory1` | Write variables to a memory store. |
| `UPDATE_MEMORY` | `UpdateMemory1` | Update existing variables in a memory store. |
| `VAR` | `Var1` | Evaluate expression and store result as a named variable. |
| `TEXT_TO_VARIABLE` | `TextToVariable1` | Store thought text as a named variable. |

### Content and Code

| Type | Enum Value | Description |
|------|-----------|-------------|
| `STATIC` | `Static1` | Output fixed text content (no LLM). |
| `TEXT` | `Text1` | Render Jinja2 template using thought metadata. |
| `CODE` | `Code1` | Execute code (Python, JS, etc.). |
| `PROGRAM_RUNNER` | `ProgramRunner1` | Run a registered tool/program inline. |
| `STATIC_PROGRAM_PARAMETERS` | `StaticProgramParameters1` | Inject static tool parameters into metadata. |

### Integration

| Type | Enum Value | Description |
|------|-----------|-------------|
| `SUB_ASSISTANT` | `SubAssistant1` | Invoke another agent graph synchronously (blocks until complete). |

### Utility

| Type | Enum Value | Description |
|------|-----------|-------------|
| `BLANK` | `Blank1` | No-op placeholder. |
| `COMMENT` | `Comment1` | Documentation node (not executed, no edges). |
| `VIEW_METADATA` | `ViewMetadata1` | Debug node -- inspect thought metadata. |
| `USE_ENVIRONMENT` | `UseEnvironment1` | Activate a runtime environment. |
| `UNSELECT_ENVIRONMENT` | `UnselectEnvironment1` | Deactivate a runtime environment. |
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
        "llm_model": "gpt-4o",
        "llm_provider": "openai",
        "llm_temperature": 0.7,
        "llm_system_instruction": "You are a helpful assistant.",
    },
)
```

## See Also

- [Architecture](architecture.md) -- System overview and data flow
- [Engine](engine.md) -- How the engine executes graphs
- [Providers](providers.md) -- LLM provider configuration for instruction nodes
- [Tools](tools.md) -- Tool definitions and the `@tool` decorator
