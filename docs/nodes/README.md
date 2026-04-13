# Node Reference

Nodes are the building blocks of Quartermaster flows. Each node encapsulates a
discrete unit of work -- calling an LLM, waiting for user input, branching on a
condition, manipulating data, or managing memory. Nodes are wired together with
directed edges to form an **agent graph** that the engine traverses at runtime.

## How nodes work

Every node class inherits from `AbstractAssistantNode` (or the LLM-specific
`AbstractLLMAssistantNode`) and exposes four class methods:

| Method | Purpose |
|---|---|
| `info()` | Returns an `AssistantInfo` with description, version, and default metadata. |
| `flow_config()` | Returns a `FlowNodeConf` defining traversal, thought type, and message type. |
| `think(ctx)` | Core logic -- receives a `NodeContext` and produces output through a `ThoughtHandle`. |
| `name()` / `version()` | Identity used by the node registry. |

Nodes are **stateless**. All mutable state lives in the `NodeContext`:

* `ctx.thought` / `ctx.thought_id` -- the current thought in the conversation.
* `ctx.node_metadata` -- configuration dict (model name, temperature, etc.).
* `ctx.handle` -- a `ThoughtHandle` for writing results back.
* `ctx.flow_node_id` -- unique ID of this node instance in the graph.
* `ctx.assistant_node` -- the graph-level node object (edges, position, etc.).

### Traversal model

Two settings on `FlowNodeConf` control how the engine schedules a node:

| Setting | Options | Meaning |
|---|---|---|
| `traverse_in` | `AwaitFirst` | Fire as soon as **any** incoming edge arrives. |
| | `AwaitAll` | Wait until **all** incoming edges have arrived. |
| `traverse_out` | `SpawnAll` | Activate every outgoing edge after completion. |
| | `SpawnPickedNode` | Activate only the edge chosen at runtime (branching). |
| | `SpawnStart` | Spawn the Start node of connected agents (End node). |
| | `SpawnNone` | Do not activate any outgoing edge. |

### Chain handlers (LLM nodes)

LLM nodes delegate work to a `Chain` of handlers executed in sequence:

1. **ValidateMemoryID** -- ensures a valid thought exists.
2. **PrepareMessages** -- collects conversation history from the thought tree.
3. **ContextManager** -- trims messages to fit token/message limits.
4. **TransformToProvider** -- converts messages to provider-specific format.
5. **GenerateStreamResponse** / **GenerateNativeResponse** / **GenerateToolCall** -- calls the LLM.
6. **ProcessStreamResponse** -- writes the result back to the thought.

## Node catalogue (39 nodes)

### Control Flow (6)

| Node name | Class | Description | Docs |
|---|---|---|---|
| StartNode | `StartNodeV1` | Flow entry point; initializes memory | [control-flow-nodes.md](control-flow-nodes.md) |
| EndNode | `EndNodeV1` | Flow termination; spawns connected agents | [control-flow-nodes.md](control-flow-nodes.md) |
| IfNode1 | `IfNode` | Binary conditional (true/false) | [control-flow-nodes.md](control-flow-nodes.md) |
| Switch1 | `SwitchNode1` | Multi-way branching with default | [control-flow-nodes.md](control-flow-nodes.md) |
| Break1 | `BreakNode1` | Message collection boundary | [control-flow-nodes.md](control-flow-nodes.md) |
| SubAssistant1 | `SubAssistant1` | Invoke a nested sub-flow | [control-flow-nodes.md](control-flow-nodes.md) |

### LLM (10)

| Node name | Class | Description | Docs |
|---|---|---|---|
| InstructionNode | `InstructionNodeV1` | Core LLM instruction node | [llm-nodes.md](llm-nodes.md) |
| AgentNode | `AgentNodeV1` | Autonomous agent with tool loop | [llm-nodes.md](llm-nodes.md) |
| Decision1 | `Decision1` | LLM picks which edge to follow | [llm-nodes.md](llm-nodes.md) |
| Merge1 | `Merge1` | Synthesize parallel (SpawnAll) branch outputs via LLM | [llm-nodes.md](llm-nodes.md) |
| ReasoningNode | `ReasoningV1` | Chain-of-thought reasoning | [llm-nodes.md](llm-nodes.md) |
| Summarize1 | `Summarize1` | LLM summarization | [llm-nodes.md](llm-nodes.md) |
| InstructionParameters1 | `InstructionParameters1` | Structured output via tool calling | [llm-nodes.md](llm-nodes.md) |
| InstructionProgram1 | `InstructionProgram1` | LLM with tool execution | [llm-nodes.md](llm-nodes.md) |
| InstructionProgramParameters1 | `InstructionProgramParameters1` | Tools + structured output | [llm-nodes.md](llm-nodes.md) |
| InstructionImageVision1 | `InstructionImageVision1` | Vision-enabled LLM instruction | [llm-nodes.md](llm-nodes.md) |

### Data (9)

| Node name | Class | Description |
|---|---|---|
| StaticAssistant | `StaticNode1` | Output static text content |
| StaticDecision1 | `StaticDecision1` | Rule-based path selection |
| StaticMerge1 | `StaticMerge1` | Join parallel (SpawnAll) branch outputs without LLM |
| StaticProgramParameters1 | `StaticProgramParameters1` | Static structured parameters |
| TextNode1 | `TextNode` | Template-rendered text |
| TextToVariableNode1 | `TextToVariableNode` | Store text into a variable |
| VarNode1 | `VarNode` | Read/write thought variables |
| Code1 | `CodeNode` | Execute code snippets |
| ProgramRunner1 | `ProgramRunner1` | Run external programs/tools |

### Memory (5)

| Node name | Class | Description |
|---|---|---|
| FlowMemory | `FlowMemoryNode` | Flow-scoped key-value store |
| UserMemory1 | `UserMemoryNode` | User-scoped persistent memory |
| ReadMemory1 | `ReadMemoryNode` | Read from memory store |
| WriteMemory1 | `WriteMemoryNode` | Write to memory store |
| UpdateMemory1 | `UpdateMemoryNode` | Update existing memory entry |

### User Interaction (4)

| Node name | Class | Description |
|---|---|---|
| UserAssistant1 | `UserNode1` | Wait for user text input |
| UserDecision1 | `UserDecisionV1` | User picks a path |
| UserForm1 | `UserFormV1` | Structured form input |
| UserProgramForm1 | `UserProgramFormV1` | Form with tool execution |

### Utility (6)

| Node name | Class | Description |
|---|---|---|
| Blank1 | `BlankNode` | No-op placeholder |
| Comment1 | `CommentNode` | Developer comment (no runtime effect) |
| UseEnvironment1 | `UseEnvironmentNode` | Activate an environment |
| UnselectEnvironment1 | `UnselectEnvironmentNode` | Deactivate an environment |
| UseFile1 | `UseFileNode` | Attach a file to context |
| ViewMetadata1 | `ViewMetadataNode` | Inspect thought metadata |

## Creating custom nodes

Extend `AbstractAssistantNode` for non-LLM nodes or `AbstractLLMAssistantNode`
for nodes that call a language model.

```python
from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)

class HttpRequestNode(AbstractAssistantNode):
    @classmethod
    def name(cls) -> str:
        return "HttpRequest1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Make an HTTP request"
        info.metadata = {"url": "", "method": "GET"}
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )

    @classmethod
    def think(cls, ctx) -> None:
        url = cls.get_metadata_key_value(ctx, "url", "")
        # Perform the request and write result to ctx.handle
```

Register the node so the engine can discover it:

```python
from quartermaster_nodes.registry import registry
registry.register(HttpRequestNode)
```

## Thought types

The `thought_type` on a node's `FlowNodeConf` controls how results are stored:

| Thought type | Behavior |
|---|---|
| `SkipThought1` | Node produces no thought (pass-through). |
| `NewThought1` | Creates a new thought in the conversation. |
| `NewHiddenThought1` | Creates a thought hidden from the user. |
| `NewCollapsedThought1` | Creates a thought shown collapsed by default. |
| `EditSameOrAddNew1` | Edits the current thought or creates a new one. |
| `UsePreviousThought1` | Reuses the incoming thought without creating a new one. |

## Graph builder quick start

The `Graph` builder from `quartermaster-graph` provides a fluent API for
assembling graphs in code. Every graph should start with `.user()` after
`.start()` to collect input:

```python
from quartermaster_graph import Graph

agent = (
    Graph("Support Bot")
    .start()
    .user("What can I help you with?")
    .instruction("Answer", model="gpt-4o", system_instruction="You are a helpful assistant.")
    .end()
)
```

See [graph-building.md](../graph-building.md) for the full builder reference.
