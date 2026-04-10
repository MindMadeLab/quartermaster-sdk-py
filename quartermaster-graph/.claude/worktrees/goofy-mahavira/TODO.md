# quartermaster-graph — Extraction TODO

Framework-agnostic agent graph schema. Defines the structure for building AI agent workflows as directed acyclic graphs (DAGs): agents, versions, nodes, edges, metadata, and traversal strategies. This is the "blueprint" format — how agent flows are defined and stored.

## Source Files

Extract from `quartermaster/be/assistants/`:

| Source File | Purpose |
|---|---|
| `models.py` | Django models: Assistant, AssistantVersion, AssistantNode, AssistantEdge, NodeMemory, AssistantNodeFE, AssistantInfo |
| `catalog_models.py` | Enums: AvailableNodeTypes (38), TraversingIn/Out, ThoughtTypes, MessageTypes, ErrorHandling |
| `config.py` | `FlowNodeConf`, `AssistantInfo` configuration models |
| `abstract.py` | `AbstractAssistantNode` base class (interface for node implementations) |
| `serializers.py` | DRF serializers (reference for JSON schema) |
| `services/version_service.py` | Versioning logic (create version, fork, diff) |

## Extractability: 7/10

The graph schema is conceptually clean — it's a standard DAG with typed nodes and edges. But it's deeply embedded in Django ORM. Extraction means converting Django models to Pydantic models while preserving all the relationships and validation logic.

## Phase 1: Core Schema (Pydantic Models)

### 1.1 Graph Definition
- [ ] `Agent` — top-level agent definition
  ```python
  class Agent(BaseModel):
      id: UUID
      name: str
      description: str
      tags: list[str] = []
      created_at: datetime
      updated_at: datetime
      current_version: Optional[str] = None  # semver
  ```

- [ ] `AgentVersion` — versioned snapshot of agent graph
  ```python
  class AgentVersion(BaseModel):
      id: UUID
      agent_id: UUID
      version: str  # semver "X.Y.Z"
      start_node_id: UUID
      nodes: list[GraphNode]
      edges: list[GraphEdge]
      features: str = ""
      is_published: bool = False
      forked_from: Optional[UUID] = None
      created_at: datetime
  ```

- [ ] `GraphNode` — node in the agent graph
  ```python
  class GraphNode(BaseModel):
      id: UUID
      type: NodeType  # enum of 38+ types
      name: str = ""
      traverse_in: TraverseIn  # AwaitAll | AwaitFirst
      traverse_out: TraverseOut  # SpawnAll | SpawnNone | SpawnPickedNode | SpawnStart
      thought_type: ThoughtType
      message_type: MessageType
      error_handling: ErrorStrategy  # Stop | Retry | Skip | Custom
      metadata: dict = {}  # node-specific configuration
      position: Optional[NodePosition] = None  # x, y for visual editor
  ```

- [ ] `GraphEdge` — connection between nodes
  ```python
  class GraphEdge(BaseModel):
      id: UUID
      source_id: UUID  # tail node
      target_id: UUID  # head node
      label: str = ""  # edge label (e.g., "Yes", "No" for decision)
      is_main: bool = True  # primary flow direction
      points: list[tuple[float, float]] = []  # bezier points for visual
  ```

- [ ] `NodePosition` — visual position
  ```python
  class NodePosition(BaseModel):
      x: int
      y: int
      icon: Optional[str] = None
  ```

### 1.2 Enums
- [ ] Extract all enums from `catalog_models.py`:
  ```python
  class NodeType(str, Enum):
      INSTRUCTION = "Instruction1"
      DECISION = "Decision1"
      REASONING = "Reasoning1"
      AGENT = "Agent1"
      START = "Start1"
      END = "End1"
      MERGE = "Merge1"
      IF = "If1"
      SWITCH = "Switch1"
      BREAK = "Break1"
      USER = "User1"
      USER_DECISION = "UserDecision1"
      USER_FORM = "UserForm1"
      STATIC = "Static1"
      VAR = "Var1"
      TEXT = "Text1"
      CODE = "Code1"
      PROGRAM_RUNNER = "ProgramRunner1"
      FLOW_MEMORY = "FlowMemory1"
      READ_MEMORY = "ReadMemory1"
      WRITE_MEMORY = "WriteMemory1"
      # ... all 38 types
  
  class TraverseIn(str, Enum):
      AWAIT_ALL = "AwaitAll"
      AWAIT_FIRST = "AwaitFirst"
  
  class TraverseOut(str, Enum):
      SPAWN_ALL = "SpawnAll"
      SPAWN_NONE = "SpawnNone"
      SPAWN_START = "SpawnStart"
      SPAWN_PICKED = "SpawnPickedNode"
  
  class ThoughtType(str, Enum):
      SKIP = "SkipThought1"
      NEW = "NewThought1"
      NEW_HIDDEN = "NewHiddenThought1"
      NEW_COLLAPSED = "NewCollapsedThought1"
      EDIT_OR_NEW = "EditSameOrAddNew1"
      # ... etc
  
  class MessageType(str, Enum):
      AUTOMATIC = "Automatic"
      USER = "User"
      VARIABLE = "Variable"
      ASSISTANT = "Assistant"
  
  class ErrorStrategy(str, Enum):
      STOP = "Stop"
      RETRY = "Retry"
      SKIP = "Skip"
      CUSTOM = "Custom"
  ```

### 1.3 Node Metadata Schemas
- [ ] Define typed metadata schemas per node type (instead of raw dict):
  ```python
  class InstructionMetadata(BaseModel):
      system_instruction: str = ""
      model: str = "gpt-4o"
      provider: str = "openai"
      temperature: float = 0.7
      max_tokens: Optional[int] = None
      tools: list[str] = []  # tool names from registry
  
  class DecisionMetadata(InstructionMetadata):
      decision_prompt: str = ""
      # edges define the options
  
  class IfMetadata(BaseModel):
      expression: str  # Python expression to evaluate
      variable: str  # variable name to check
  
  class StaticMetadata(BaseModel):
      content: str = ""
  
  class CodeMetadata(BaseModel):
      language: str = "python"
      code: str = ""
  ```
- [ ] Map `NodeType` → metadata schema class
- [ ] Validate metadata on node creation

## Phase 2: Graph Operations

### 2.1 Validation
- [ ] `validate_graph(version: AgentVersion) -> list[ValidationError]`
  - Every graph must have exactly one Start node
  - Every graph must have at least one End node
  - No orphan nodes (every node reachable from Start)
  - No cycles (DAG validation) — unless explicitly allowed for loops
  - Decision/If nodes must have matching edge labels
  - All edge source/target IDs reference existing nodes

### 2.2 Versioning
- [ ] `create_version(agent: Agent, version: str) -> AgentVersion` — snapshot current graph
- [ ] `fork(version: AgentVersion, new_agent: Agent) -> AgentVersion` — deep copy to new agent
- [ ] `diff(v1: AgentVersion, v2: AgentVersion) -> GraphDiff` — what changed between versions
- [ ] Semver helpers: `bump_major()`, `bump_minor()`, `bump_patch()`

### 2.3 Serialization
- [ ] `to_json(version: AgentVersion) -> dict` — full JSON representation
- [ ] `from_json(data: dict) -> AgentVersion` — parse from JSON
- [ ] `to_yaml(version: AgentVersion) -> str` — YAML format (human-readable)
- [ ] `from_yaml(yaml_str: str) -> AgentVersion` — parse from YAML
- [ ] JSON Schema for the graph format (enables validation in any language)

### 2.4 Graph Traversal Utilities
- [ ] `get_start_node(version) -> GraphNode`
- [ ] `get_successors(node_id) -> list[GraphNode]`
- [ ] `get_predecessors(node_id) -> list[GraphNode]`
- [ ] `get_path(start, end) -> list[GraphNode]` — shortest path
- [ ] `topological_sort(version) -> list[GraphNode]`
- [ ] `find_merge_points(version) -> list[GraphNode]` — nodes where branches converge
- [ ] `find_decision_points(version) -> list[GraphNode]` — nodes where branches diverge

## Phase 3: Graph Builder (Fluent API)

### 3.1 Programmatic Graph Construction
- [ ] Fluent builder for creating graphs in code (alternative to visual editor):
  ```python
  graph = (
      GraphBuilder("My Agent")
      .start()
      .instruction("Analyze the input", model="gpt-4o")
      .decision("Is it positive?", options=["Yes", "No"])
      .on("Yes").instruction("Generate positive response").end()
      .on("No").instruction("Generate negative response").end()
      .build()
  )
  ```
- [ ] Support for parallel branches (fork/merge)
- [ ] Support for loops (with break conditions)
- [ ] Support for sub-agents (composition)
- [ ] Validate graph on `.build()`

### 3.2 Graph Templates
- [ ] Pre-built templates for common patterns:
  - Simple chat agent (Start → Instruction → User → loop)
  - Decision tree (Start → Decision → branches → End)
  - RAG pipeline (Start → Retrieve → Instruction → End)
  - Multi-step agent (Start → Instruction → ProgramRunner → Instruction → End)
  - Parallel processing (Start → Fork → [A, B, C] → Merge → End)

## Phase 4: Testing

### 4.1 Schema Tests
- [ ] Test all Pydantic models serialize/deserialize correctly
- [ ] Test enum values match QM originals
- [ ] Test metadata validation per node type
- [ ] Test JSON/YAML round-trip (serialize → deserialize → compare)

### 4.2 Validation Tests
- [ ] Valid graph passes validation
- [ ] Missing start node → error
- [ ] Orphan node → error
- [ ] Cycle detection → error (or warning for loops)
- [ ] Invalid edge references → error

### 4.3 Builder Tests
- [ ] Build simple graph → validate → serialize
- [ ] Build complex graph with branches → validate
- [ ] Build from template → verify structure

### 4.4 Traversal Tests
- [ ] Topological sort on various graph shapes
- [ ] Successor/predecessor lookups
- [ ] Path finding

## Phase 5: Documentation

### 5.1 README
- [ ] Quick start: define a graph in Python or YAML
- [ ] Schema reference (all models, all enums)
- [ ] Builder API reference
- [ ] Template gallery
- [ ] Visual examples (Mermaid diagrams of sample graphs)

### 5.2 Graph Format Spec
- [ ] Document the JSON format as a specification
- [ ] This becomes the interchange format between frontend editors and backend engines
- [ ] Any frontend can build graphs in this format
- [ ] Any engine can execute graphs in this format

### 5.3 YAML Format Guide
- [ ] Show how to define agents as YAML files (great for version control)
- [ ] Example: full agent definition in a single YAML file

## Phase 6: CI/CD & PyPI

- [ ] GitHub Actions: lint, typecheck, tests
- [ ] PyPI package: `quartermaster-graph` or `quartermaster-graph`
- [ ] Zero external dependencies except Pydantic

## Architecture Notes

### This Is The Schema Layer, Not The Engine
- `quartermaster-graph` defines WHAT an agent flow looks like (the blueprint)
- `quartermaster-engine` defines HOW it executes (the runtime)
- `quartermaster-nodes` defines WHAT each node does (the behavior)
- This separation means you can use quartermaster-graph with a completely different engine

### Why This Is Valuable
- Standardized graph format for AI agent workflows
- Any visual editor can output this format
- Any execution engine can consume this format
- YAML support means agents can be version-controlled in git
- Builder API means no visual editor needed for simple agents
- Graph validation catches errors before execution

### Relationship to Other Packages
- `quartermaster-nodes` — implements the behavior for each `NodeType`
- `quartermaster-engine` — executes graphs defined by `quartermaster-graph`
- `quartermaster-tools` — tools referenced in node metadata
- `quartermaster-providers` — LLM providers referenced in node metadata

## Timeline Estimate

- Phase 1 (Schema): 3-4 days
- Phase 2 (Operations): 2-3 days
- Phase 3 (Builder): 2-3 days
- Phase 4 (Testing): 2-3 days
- Phase 5 (Docs): 2 days
- Phase 6 (CI/CD): 1 day

**Total: 2-3 weeks**
