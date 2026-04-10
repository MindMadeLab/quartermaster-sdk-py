# qm-nodes — Extraction TODO

Library of 38 composable node types for building AI agent graphs. Each node is a self-contained unit of execution: LLM calls, control flow (branching, merging, loops), user interaction, memory operations, code execution, and more.

## Source Files

Extract from `quartermaster/be/assistants/nodes/`:

### LLM Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `instruction.py` | `InstructionNodeV1` | Send prompt to LLM, get text response |
| `instruction_image_vision.py` | `InstructionImageVision1` | LLM with image/vision input |
| `instruction_parameters.py` | `InstructionParameters1` | LLM with structured parameter output |
| `instruction_program.py` | `InstructionProgram1` | LLM + tool execution |
| `instruction_program_parameters.py` | `InstructionProgramParameters1` | LLM + tools + structured output |
| `decision.py` | `Decision1` | LLM picks a path (branch selection) |
| `reasoning.py` | `Reasoning1` | Extended thinking / chain-of-thought |
| `agent.py` | `Agent1` | Sub-agent invocation (recursive composition) |
| `summarize.py` | `Summarize1` | LLM summarization node |

### Control Flow Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `start.py` | `StartNodeV1` | Flow entry point |
| `end.py` | `End1` | Flow termination |
| `break_node.py` | `Break1` | Loop break |
| `merge.py` | `Merge1` | Synchronize parallel branches |
| `if_node.py` | `If1` | Conditional branching (expression-based) |
| `switch.py` | `Switch1` | Multi-way branching |
| `sub_assistant.py` | `SubAssistant1` | Invoke sub-flow |

### User Interaction Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `user.py` | `User1` | Wait for user message |
| `user_decision.py` | `UserDecision1` | Present choices to user |
| `user_form.py` | `UserForm1` | Collect structured user input |
| `user_program_form.py` | `UserProgramForm1` | User selects + configures tool |

### Memory Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `flow_memory.py` | `FlowMemory1` | Read/write flow-scoped variables |
| `read_memory.py` | `ReadMemory1` | Read from persistent memory |
| `write_memory.py` | `WriteMemory1` | Write to persistent memory |
| `update_memory.py` | `UpdateMemory1` | Update existing memory |
| `user_memory.py` | `UserMemory1` | User-scoped persistent memory |

### Data Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `static.py` | `Static1` | Static content (no LLM) |
| `static_merge.py` | `StaticMerge1` | Merge with static content |
| `static_decision.py` | `StaticDecision1` | Rule-based decision (no LLM) |
| `static_program_parameters.py` | `StaticProgramParameters1` | Static tool parameters |
| `var.py` | `Var1` | Variable assignment |
| `text.py` | `Text1` | Text template rendering |
| `text_to_variable.py` | `TextToVariable1` | Extract text into variable |
| `code.py` | `Code1` | Execute code |
| `program_runner.py` | `ProgramRunner1` | Execute registered tool/program |

### Utility Nodes
| Source File | Node Class | Purpose |
|---|---|---|
| `blank.py` | `Blank1` | No-op / placeholder |
| `comment.py` | `Comment1` | Documentation node |
| `view_metadata.py` | `ViewMetadata1` | Debug: inspect node metadata |
| `use_environment.py` | `UseEnvironment1` | Activate execution environment |
| `unselect_environment.py` | `UnselectEnvironment1` | Deactivate environment |
| `use_file.py` | `UseFile1` | Attach file to context |

### Base Classes
| Source File | Class | Purpose |
|---|---|---|
| `quartermaster/be/assistants/abstract.py` | `AbstractAssistantNode` | Base for all nodes |
| `quartermaster/be/assistants/abstract.py` | `AbstractLLMAssistantNode` | Base for LLM-calling nodes |
| `quartermaster/be/assistants/catalog_models.py` | Enums | Node types, traverse strategies, thought types |
| `quartermaster/be/assistants/config.py` | `FlowNodeConf`, `AssistantInfo` | Configuration models |

### Chain Handlers (used by LLM nodes)
| Source File | Purpose |
|---|---|
| `quartermaster/be/chain/utils.py` | `Chain` class — handler pipeline |
| `quartermaster/be/chain/handlers/base.py` | `Handler` ABC |
| `quartermaster/be/chain/handlers/llm_handlers.py` | PrepareMessages, ContextManager, TransformToProvider, GenerateStreamResponse, ProcessStreamResponse |
| `quartermaster/be/chain/handlers/credit_handlers.py` | Billing handlers (EXCLUDE from open-source) |

## Extractability: 8/10

Nodes are mostly stateless — they receive context, do work, return results. Main coupling is to `AbstractAssistantNode` base class which depends on FlowContext and ThoughtMemory. The chain handler pattern is completely portable.

## Phase 1: Extract Chain Handler Pattern

### 1.1 Core Chain
- [ ] Extract `Chain` class (10 lines, zero dependencies)
- [ ] Extract `Handler` ABC (5 lines, zero dependencies)
- [ ] These are the foundation — everything else builds on them

### 1.2 LLM Chain Handlers
- [ ] Extract `ValidateMemoryID` — check thought exists
- [ ] Extract `PrepareMessages` — load message history from context
- [ ] Extract `ContextManager` — truncate messages to fit token window
- [ ] Extract `TransformToProvider` — convert messages to provider-specific format
- [ ] Extract `GenerateStreamResponse` — call LLM via provider abstraction
- [ ] Extract `ProcessStreamResponse` — parse and store streaming results
- [ ] Extract `GenerateToolCall` — structured tool calling
- [ ] **EXCLUDE** `credit_handlers.py` — billing is proprietary
- [ ] Replace `FlowContext` dependency with a protocol/interface:
  ```python
  class NodeContext(Protocol):
      config: LLMConfig
      messages: list[Message]
      tools: list[ToolDefinition]
      metadata: dict
  ```

## Phase 2: Extract Node Base Classes

### 2.1 AbstractAssistantNode
- [ ] Extract from `be/assistants/abstract.py`
- [ ] Define interface:
  - `info() -> NodeInfo` (metadata about the node)
  - `name() -> str`
  - `version() -> str`
  - `flow_config() -> FlowNodeConf` (traverse_in, traverse_out, thought_type, message_type)
  - `think(context) -> NodeResult` (main execution method)
- [ ] Replace Django model dependencies with dataclasses
- [ ] `FlowNodeConf` becomes standalone config:
  ```python
  @dataclass
  class FlowNodeConf:
      traverse_in: TraverseIn  # AwaitAll | AwaitFirst
      traverse_out: TraverseOut  # SpawnAll | SpawnNone | SpawnPickedNode | SpawnStart
      thought_type: ThoughtType
      message_type: MessageType
      error_handling: ErrorStrategy
  ```

### 2.2 AbstractLLMAssistantNode
- [ ] Extends `AbstractAssistantNode`
- [ ] Adds: model selection, provider selection, temperature, system message from metadata
- [ ] Builds and runs `Chain` of handlers
- [ ] Replace `select_text_generation_service_provider()` with provider injection

### 2.3 Catalog Enums
- [ ] Extract all enums from `catalog_models.py`:
  - `AvailableNodeTypes` (38 types)
  - `AvailableTraversingIn` (AwaitFirst, AwaitAll)
  - `AvailableTraversingOut` (SpawnAll, SpawnNone, SpawnStart, SpawnPickedNode)
  - `AvailableThoughtTypes` (SkipThought1, NewThought1, etc.)
  - `AvailableMessageTypes` (Automatic, User, Variable, Assistant)
  - `AvailableErrorHandlingStrategies` (Stop, Retry, Skip, Custom)

## Phase 3: Extract Individual Nodes

### 3.1 Priority 1 — Core LLM Nodes
- [ ] `InstructionNodeV1` — the most important node. Send prompt, get response.
- [ ] `Decision1` — LLM picks which edge to follow
- [ ] `Reasoning1` — Extended thinking with chain-of-thought
- [ ] `Agent1` — Sub-agent invocation

### 3.2 Priority 2 — Control Flow
- [ ] `StartNodeV1` — Entry point (minimal logic)
- [ ] `End1` — Termination
- [ ] `Merge1` — Synchronize parallel branches
- [ ] `If1` — Expression-based conditional
- [ ] `Switch1` — Multi-way branch
- [ ] `Break1` — Loop exit

### 3.3 Priority 3 — Data & Memory
- [ ] `Static1` — Pass-through static content
- [ ] `Var1` — Variable assignment
- [ ] `Text1` — Template rendering
- [ ] `FlowMemory1` — Read/write flow variables
- [ ] `ReadMemory1` / `WriteMemory1` — Persistent memory
- [ ] `Code1` — Code execution (depends on qm-code-runner)

### 3.4 Priority 4 — User Interaction
- [ ] `User1` — Wait for user input
- [ ] `UserDecision1` — Present choices
- [ ] `UserForm1` — Structured input collection
- [ ] `ProgramRunner1` — Execute tool

### 3.5 For Each Node
- [ ] Remove Django ORM dependencies
- [ ] Replace metadata dict keys with typed config
- [ ] Ensure node is testable in isolation (no DB, no Celery)
- [ ] Add type hints
- [ ] Add docstring explaining what the node does

## Phase 4: Node Registry

### 4.1 Registration
- [ ] `NodeRegistry` — register nodes by type name + version
- [ ] Auto-discovery via decorators: `@register_node("Instruction1")`
- [ ] Version-aware lookup: `registry.get("Instruction", version=1)`
- [ ] Allow third-party node registration (extensible)

### 4.2 Node Catalog
- [ ] Generate catalog from registered nodes
- [ ] Each entry: name, version, description, config schema, input/output types
- [ ] Exportable as JSON (for frontend editors to consume)

## Phase 5: Testing

### 5.1 Per-Node Tests
- [ ] Each node gets a test file: `test_instruction.py`, `test_decision.py`, etc.
- [ ] Test with mock `NodeContext` (no real LLM calls)
- [ ] Test config parsing from metadata dict
- [ ] Test error handling per node

### 5.2 Chain Handler Tests
- [ ] Test each handler in isolation
- [ ] Test full chain pipeline with mock provider
- [ ] Test chain composition (add/remove handlers)

### 5.3 Integration Tests
- [ ] Build a simple 3-node flow: Start → Instruction → End
- [ ] Build a decision flow: Start → Decision → [Path A, Path B] → End
- [ ] Build a loop: Start → Instruction → If → (loop back or End)

## Phase 6: Documentation

### 6.1 Node Reference
- [ ] One page per node type: description, config options, inputs, outputs, example usage
- [ ] Visual diagram showing node icon + connections
- [ ] "When to use" guide for each node

### 6.2 Custom Node Guide
- [ ] How to create your own node type
- [ ] Extend `AbstractAssistantNode`
- [ ] Register with `NodeRegistry`
- [ ] Example: custom "HTTP Request" node

### 6.3 Chain Handler Guide
- [ ] How the handler pipeline works
- [ ] How to add custom handlers (logging, caching, rate limiting)
- [ ] Example: add a caching handler before LLM call

## Architecture Notes

### Dependencies on Other QM Packages
- `qm-providers` — LLM nodes use provider abstraction for model calls
- `qm-tools` — ProgramRunner uses tool abstraction
- `qm-code-runner` — Code1 node uses code execution (optional)
- `qm-graph` — Nodes need graph schema for traverse config

### Why This Is Valuable
- Composable building blocks for AI agent graphs
- 38 battle-tested node types covering every common pattern
- Chain handler pattern is flexible and extensible
- Nodes are framework-agnostic once extracted (usable outside QM)

## Timeline Estimate

- Phase 1 (Chain): 2 days
- Phase 2 (Base classes): 3 days
- Phase 3 (Nodes): 5-7 days
- Phase 4 (Registry): 1 day
- Phase 5 (Testing): 3-5 days
- Phase 6 (Docs): 2-3 days

**Total: 3-4 weeks**
