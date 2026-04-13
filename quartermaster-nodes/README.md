# quartermaster-nodes

Composable node types for building AI agent graphs.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-nodes)](https://pypi.org/project/quartermaster-nodes/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **40 built-in node types** across 6 categories: LLM, control flow, data, user interaction, memory, and utility
- **Framework-agnostic** via the `NodeContext` protocol -- integrates with any runtime
- **Stateless class-based design** -- all nodes use `@classmethod` with no instance state
- **Chain-of-Responsibility pattern** for composable LLM processing pipelines
- **NodeRegistry** with auto-discovery, version tracking, and catalog generation
- **Configurable flow behavior** per node: traversal strategy, thought type, message type, error handling

## Installation

```bash
pip install quartermaster-nodes
```

**Dependencies**: `jinja2>=3.1`, `quartermaster-graph` (enums), `quartermaster-providers` (LLM config)

## Quick Start

### Register and Discover Nodes

```python
from quartermaster_nodes import NodeRegistry
from quartermaster_nodes.nodes import InstructionNodeV1, StartNodeV1, EndNodeV1, Decision1

# Manual registration
registry = NodeRegistry()
registry.register(StartNodeV1)
registry.register(InstructionNodeV1)
registry.register(Decision1)
registry.register(EndNodeV1)

# Or auto-discover all 40 built-in nodes
registry = NodeRegistry()
count = registry.discover("quartermaster_nodes.nodes")
print(f"Discovered {count} nodes")

# Look up a node by name
node_cls = registry.get("InstructionNode")
print(node_cls.info().description)

# Generate a JSON catalog of all nodes
catalog = registry.catalog_json()
```

### Creating a Custom Node

```python
from quartermaster_nodes import AbstractAssistantNode, AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableTraversingIn,
    AvailableTraversingOut,
    AvailableThoughtTypes,
    AvailableMessageTypes,
)

class HttpRequestNode(AbstractAssistantNode):
    """Fetch data from an HTTP endpoint."""

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Make an HTTP request and return the response"
        info.metadata = {"url": "", "method": "GET"}
        return info

    @classmethod
    def name(cls) -> str:
        return "HttpRequest"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Tool,
        )

    @classmethod
    def think(cls, ctx) -> None:
        import urllib.request
        url = cls.get_metadata_key_value(ctx, "url", "")
        method = cls.get_metadata_key_value(ctx, "method", "GET")

        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req) as response:
            body = response.read().decode()

        ctx.handle.append_text(body)
```

### Implementing the NodeContext Protocol

Nodes are framework-agnostic. Implement the `NodeContext` protocol to integrate with your runtime:

```python
from quartermaster_nodes.protocols import NodeContext

class MyRuntimeContext:
    """Adapts your runtime to the NodeContext protocol."""

    def __init__(self, node_config, thought):
        self._config = node_config
        self._thought = thought

    @property
    def node_metadata(self) -> dict:
        return self._config.metadata

    @property
    def flow_node_id(self):
        return self._config.id

    @property
    def thought_id(self):
        return self._thought.id if self._thought else None

    @property
    def thought(self):
        return self._thought

    @property
    def handle(self):
        return self._thought  # Must implement ThoughtHandle protocol

    @property
    def assistant_node(self):
        return self._config

    @property
    def chat_id(self):
        return None

# Execute a node
ctx = MyRuntimeContext(node_config, thought)
InstructionNodeV1.think(ctx)
```

## Node Categories

### LLM Nodes (10)

| Node | Description |
|------|-------------|
| `InstructionNodeV1` | Generate response from system instructions and conversation history |
| `InstructionImageVision1` | LLM with image/vision input support |
| `InstructionParameters1` | LLM with structured parameter extraction |
| `InstructionProgram1` | LLM with tool/function execution |
| `InstructionProgramParameters1` | LLM with tools and structured output |
| `Decision1` | LLM picks a branch path |
| `ReasoningV1` | Extended thinking / chain-of-thought reasoning |
| `AgentNodeV1` | Autonomous agent with agentic loop and tool orchestration |
| `Summarize1` | LLM-powered summarization |
| `Merge1` | Combine parallel branches via LLM |

### Control Flow Nodes (6)

| Node | Description |
|------|-------------|
| `StartNodeV1` | Flow entry point, initializes memory |
| `EndNodeV1` | Flow termination |
| `BreakNode1` | Message collection boundary |
| `IfNode` | Binary conditional branching via expression evaluation |
| `SwitchNode1` | Multi-way branching |
| `SubAssistant1` | Invoke a sub-flow |

### Data Nodes (9)

| Node | Description |
|------|-------------|
| `StaticNode1` | Output static text content |
| `StaticMerge1` | Merge with static content |
| `StaticDecision1` | Rule-based decision (no LLM) |
| `StaticProgramParameters1` | Static tool parameters |
| `VarNode` | Variable assignment via expression |
| `TextNode` | Jinja2 template rendering |
| `TextToVariableNode` | Extract text into a variable |
| `CodeNode` | Custom code definition |
| `ProgramRunner1` | Execute a registered tool |

### User Interaction Nodes (4)

| Node | Description |
|------|-------------|
| `UserNode1` | Wait for user message input |
| `UserDecisionV1` | Present choices to the user |
| `UserFormV1` | Collect structured input via form |
| `UserProgramFormV1` | User selects and configures a tool |

### Memory Nodes (5)

| Node | Description |
|------|-------------|
| `FlowMemoryNode` | Flow-scoped persistent memory |
| `ReadMemoryNode` | Read from persistent memory |
| `WriteMemoryNode` | Write to persistent memory |
| `UpdateMemoryNode` | Update existing memory entries |
| `UserMemoryNode` | User-scoped persistent memory |

### Utility Nodes (6)

| Node | Description |
|------|-------------|
| `BlankNode` | No-op placeholder |
| `CommentNode` | Documentation / annotation node |
| `ViewMetadataNode` | Debug: inspect thought metadata |
| `UseEnvironmentNode` | Activate an execution environment |
| `UnselectEnvironmentNode` | Deactivate current environment |
| `UseFileNode` | Attach a file to the context |

## API Reference

### AbstractAssistantNode

Base class for all nodes. All methods are `@classmethod`.

| Method | Description |
|--------|-------------|
| `info() -> AssistantInfo` | Node metadata (description, default config) |
| `name() -> str` | Display name |
| `version() -> str` | Version string |
| `flow_config() -> FlowNodeConf` | Traversal, thought type, message type, error handling |
| `think(ctx: NodeContext) -> None` | Core execution logic |
| `get_metadata_key_value(ctx, key, default)` | Read from node metadata |
| `store_metadata_key_value(ctx, key, value)` | Write to node metadata |
| `deprecated() -> bool` | Whether this node is deprecated (default: False) |

### AbstractLLMAssistantNode

Extends AbstractAssistantNode with LLM configuration.

| Method | Description |
|--------|-------------|
| `llm_config(ctx) -> LLMConfig` | Build LLM config from node metadata |
| `context_manager_config(ctx, llm_config) -> ContextManagerConfig` | Build context management config |

Configurable metadata keys: `llm_model`, `llm_provider`, `llm_temperature`, `llm_max_input_tokens`, `llm_max_output_tokens`, `llm_max_messages`, `llm_stream`, `llm_vision`, `llm_system_instruction`, `llm_thinking_level`.

### FlowNodeConf

Defines a node's flow behavior constraints.

| Field | Type | Description |
|-------|------|-------------|
| `traverse_in` | `AvailableTraversingIn` | AwaitFirst or AwaitAll |
| `traverse_out` | `AvailableTraversingOut` | SpawnAll, SpawnNone, SpawnStart, SpawnPickedNode |
| `thought_type` | `AvailableThoughtTypes` | How thoughts are created/displayed |
| `message_type` | `AvailableMessageTypes` | Message role (Automatic, User, Assistant, etc.) |
| `error_handling_strategy` | `AvailableErrorHandlingStrategies` | Stop, Continue, or Retry |

### NodeRegistry

| Method | Description |
|--------|-------------|
| `register(node_class)` | Register a node class |
| `get(name, version=None)` | Look up by name and optional version |
| `has(name, version=None) -> bool` | Check if a node is registered |
| `list_nodes() -> list[dict]` | List all nodes with metadata |
| `catalog_json() -> list[dict]` | JSON-serializable catalog |
| `discover(package) -> int` | Auto-discover nodes from a package |
| `count -> int` | Number of registered nodes |

### Chain-of-Responsibility

LLM nodes use a composable handler chain for processing:

```python
from quartermaster_nodes.chain import Chain
from quartermaster_nodes.chain.handlers import (
    ValidateMemoryID,
    PrepareMessages,
    ContextManager,
    TransformToProvider,
    GenerateStreamResponse,
    ProcessStreamResponse,
)

chain = (
    Chain()
    .add_handler(ValidateMemoryID())
    .add_handler(PrepareMessages(client, config))
    .add_handler(ContextManager(client, config, ctx_config))
    .add_handler(TransformToProvider(transformer))
    .add_handler(GenerateStreamResponse(client, config))
    .add_handler(ProcessStreamResponse())
)

result = chain.run({"memory_id": thought_id, "flow_node_id": node_id, "ctx": ctx})
```

## Integration with Sibling Packages

### With quartermaster-graph (graph schema)

Enums are re-exported from quartermaster-graph for consistency:

```python
from quartermaster_nodes.enums import AvailableNodeTypes  # Alias for quartermaster_graph.enums.NodeType
from quartermaster_graph.enums import NodeType             # Canonical source
```

### With quartermaster-engine (execution runtime)

The engine resolves node types from the registry and calls `think()` through an adapter:

```python
from quartermaster_nodes import NodeRegistry
from quartermaster_nodes.nodes import InstructionNodeV1

registry = NodeRegistry()
registry.discover("quartermaster_nodes.nodes")

# The engine looks up nodes by name and invokes think()
node_cls = registry.get("InstructionNode", "1.0")
node_cls.think(execution_context)
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
