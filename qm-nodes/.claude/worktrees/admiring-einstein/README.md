# qm-nodes

Library of 38 composable node types for building AI agent graphs. Each node is a self-contained unit of execution: LLM calls, control flow (branching, merging, loops), user interaction, memory operations, code execution, and more.

## Installation

```bash
pip install qm-nodes
```

Or install from source:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from qm_nodes import NodeRegistry
from qm_nodes.nodes import (
    StartNodeV1,
    InstructionNodeV1,
    EndNodeV1,
    StaticNode1,
)

# Register nodes
registry = NodeRegistry()
registry.register(StartNodeV1)
registry.register(InstructionNodeV1)
registry.register(EndNodeV1)
registry.register(StaticNode1)

# Or auto-discover all nodes
registry = NodeRegistry()
registry.discover("qm_nodes.nodes")

# Get node catalog as JSON
catalog = registry.catalog_json()
```

## Architecture

### Node Protocol

All nodes implement the same interface — they receive a `NodeContext`, do work, and return results through the context's handle:

```python
from qm_nodes import AbstractAssistantNode, AssistantInfo, FlowNodeConf

class MyNode(AbstractAssistantNode):
    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "My custom node"
        info.metadata = {"my_config_key": "default_value"}
        return info

    @classmethod
    def name(cls) -> str:
        return "MyCustomNode"

    @classmethod
    def version(cls) -> str:
        return "1.0"

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
        value = cls.get_metadata_key_value(ctx, "my_config_key", "default")
        ctx.handle.append_text(f"Processed: {value}")
```

### Chain Handler Pattern

LLM nodes use a composable handler chain for processing:

```python
from qm_nodes.chain import Chain
from qm_nodes.chain.handlers import (
    ValidateMemoryID,
    PrepareMessages,
    ContextManager,
    TransformToProvider,
    GenerateStreamResponse,
    ProcessStreamResponse,
)

chain = Chain() \
    .add_handler(ValidateMemoryID()) \
    .add_handler(PrepareMessages(client, config)) \
    .add_handler(ContextManager(client, config, ctx_config)) \
    .add_handler(TransformToProvider(transformer)) \
    .add_handler(GenerateStreamResponse(client, config)) \
    .add_handler(ProcessStreamResponse())

result = chain.run(initial_data)
```

### Framework-Agnostic Integration

Nodes are decoupled from any specific framework via the `NodeContext` protocol. Implement it to integrate with your runtime:

```python
from qm_nodes.protocols import NodeContext

class DjangoFlowContext:
    """Adapts Django models to the NodeContext protocol."""

    def __init__(self, flow_node, thought):
        self._flow_node = flow_node
        self._thought = thought

    @property
    def node_metadata(self) -> dict:
        return self._flow_node.metadata

    @property
    def flow_node_id(self):
        return self._flow_node.pk

    @property
    def thought_id(self):
        return self._thought.pk if self._thought else None

    # ... implement remaining protocol methods
```

## Node Categories

### LLM Nodes (10)

| Node | Description |
|------|-------------|
| `InstructionNodeV1` | Send prompt to LLM, get text response |
| `InstructionImageVision1` | LLM with image/vision input |
| `InstructionParameters1` | LLM with structured parameter output |
| `InstructionProgram1` | LLM + tool execution |
| `InstructionProgramParameters1` | LLM + tools + structured output |
| `Decision1` | LLM picks a path (branch selection) |
| `ReasoningV1` | Extended thinking / chain-of-thought |
| `AgentNodeV1` | Autonomous agent with tool orchestration |
| `Summarize1` | LLM summarization |
| `Merge1` | Combine parallel branches via LLM |

### Control Flow Nodes (6)

| Node | Description |
|------|-------------|
| `StartNodeV1` | Flow entry point |
| `EndNodeV1` | Flow termination |
| `BreakNode1` | Message collection boundary |
| `IfNode` | Binary conditional branching |
| `SwitchNode1` | Multi-way branching |
| `SubAssistant1` | Invoke sub-flow |

### Data Nodes (9)

| Node | Description |
|------|-------------|
| `StaticNode1` | Static text output |
| `StaticMerge1` | Merge with static content |
| `StaticDecision1` | Rule-based decision (no LLM) |
| `StaticProgramParameters1` | Static tool parameters |
| `VarNode` | Variable assignment via expression |
| `TextNode` | Jinja2 template rendering |
| `TextToVariableNode` | Extract text into variable |
| `CodeNode` | Custom code definition |
| `ProgramRunner1` | Execute registered tool |

### User Interaction Nodes (4)

| Node | Description |
|------|-------------|
| `UserNode1` | Wait for user message |
| `UserDecisionV1` | Present choices to user |
| `UserFormV1` | Collect structured input |
| `UserProgramFormV1` | User selects + configures tool |

### Memory Nodes (5)

| Node | Description |
|------|-------------|
| `FlowMemoryNode` | Flow-scoped persistent memory |
| `ReadMemoryNode` | Read from persistent memory |
| `WriteMemoryNode` | Write to persistent memory |
| `UpdateMemoryNode` | Update existing memory |
| `UserMemoryNode` | User-scoped persistent memory |

### Utility Nodes (6)

| Node | Description |
|------|-------------|
| `BlankNode` | No-op placeholder |
| `CommentNode` | Documentation node |
| `ViewMetadataNode` | Debug: inspect metadata |
| `UseEnvironmentNode` | Activate execution environment |
| `UnselectEnvironmentNode` | Deactivate environment |
| `UseFileNode` | Attach file to context |

## Creating Custom Nodes

### Basic Node

```python
from qm_nodes import AbstractAssistantNode, AssistantInfo, FlowNodeConf
from qm_nodes.enums import *

class HttpRequestNode(AbstractAssistantNode):
    metadata_url_key = "url"
    metadata_method_key = "method"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Make an HTTP request"
        info.metadata = {"url": "", "method": "GET"}
        return info

    @classmethod
    def name(cls) -> str:
        return "HttpRequest1"

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

### Custom Chain Handler

```python
from qm_nodes.chain import Handler

class LoggingHandler(Handler):
    def handle(self, data):
        print(f"Processing: {list(data.keys())}")
        return data

class CachingHandler(Handler):
    def __init__(self, cache):
        self.cache = cache

    def handle(self, data):
        key = str(data.get("memory_id"))
        if key in self.cache:
            data["response"] = self.cache[key]
        return data
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qm_nodes --cov-report=html

# Run specific test file
pytest tests/test_nodes/test_control_flow.py -v
```

## Dependencies

- **Required**: `jinja2>=3.1` (for TextNode template rendering)
- **Dev**: `pytest`, `pytest-cov`, `mypy`, `ruff`
- **Optional**: `qm-providers` (for LLM calls), `qm-tools` (for tool execution), `qm-code-runner` (for code execution)

## License

MIT
