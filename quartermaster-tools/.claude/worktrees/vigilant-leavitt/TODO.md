# quartermaster-tools — Extraction TODO

Framework for defining, registering, and executing tools (programs) that AI agents can invoke. Supports multiple tool types: local execution, API calls, MCP servers, and custom implementations. Includes a registry pattern with version-aware lookup.

## Source Files

Extract from `quartermaster/be/programs/`:

| Source File | Purpose |
|---|---|
| `internal_programs/abstract.py` | `AbstractProgram`, `AbstractLocalProgram`, `ProgramDescriber` |
| `internal_programs/registry.py` | `InternalProgramRegistry` — lazy init, version-aware lookup |
| `internal_programs/utils.py` | Constants: `THOUGHT_ID`, `USER_ID`, `LAST_THOUGHT_FLOW_NODE_ID` |
| `services/__init__.py` | `ParameterContainer`, `ParameterOptionContainer` — tool parameter models |
| `services/terminal.py` | `TerminalCommandExecutor` — local command execution |
| `models.py` | Django models for Program, ProgramVersion (reference only) |

### Internal Program Examples (for reference, not all extracted)
| Directory | Programs | Notes |
|---|---|---|
| `internal_programs/crawler/` | 15+ web crawling tools | Could be extracted as separate package |
| `internal_programs/environment/` | 15+ file/folder operations | Useful for file-based agents |
| `internal_programs/excel/` | 9 Excel tools | Spreadsheet manipulation |
| `internal_programs/pdf/` | 15+ PDF tools | PDF manipulation |
| `internal_programs/code_runner/` | 4 code execution tools | Bridges to quartermaster-code-runner |
| `internal_programs/web/` | 5 web tools | HTTP requests, Google search |
| `internal_programs/local/` | 5 local tools | Git, npm, shell, text processing |
| `internal_programs/agent/` | 30+ agent manipulation tools | QM-specific, keep proprietary |
| `internal_programs/voice/` | 2 voice tools | TTS, STT |

## Extractability: 8/10

The abstraction layer (`AbstractProgram`, `ProgramDescriber`, registry) is clean and portable. Individual tool implementations vary — some are standalone (web, file ops), others are deeply coupled to QM (agent manipulation, hub discovery).

## Phase 1: Core Abstraction

### 1.1 Extract Tool Base
- [ ] Extract `AbstractProgram` ABC:
  ```python
  class AbstractTool(ABC):
      @classmethod
      @abstractmethod
      def run(cls, context: ToolContext, **kwargs) -> ToolResult:
          ...
      
      @classmethod
      @abstractmethod
      def name(cls) -> str: ...
      
      @classmethod
      @abstractmethod
      def version(cls) -> str: ...
      
      @classmethod
      @abstractmethod
      def parameters(cls) -> list[ToolParameter]: ...
      
      @classmethod
      @abstractmethod
      def info(cls) -> ToolInfo: ...
  ```
- [ ] Rename `AbstractProgram` → `AbstractTool` (industry standard naming)
- [ ] Rename `ProgramDescriber` → `ToolInfo`
- [ ] Replace `Tuple[str, List[UUID]]` return type with proper `ToolResult` dataclass:
  ```python
  @dataclass
  class ToolResult:
      output: str
      files: list[str] = field(default_factory=list)  # file paths instead of UUIDs
      metadata: dict = field(default_factory=dict)
      success: bool = True
      error: Optional[str] = None
  ```

### 1.2 Extract Local Tool Base
- [ ] Extract `AbstractLocalProgram` → `AbstractLocalTool`
- [ ] Methods: `prepare_command()`, `result()`
- [ ] Replace `ProcessLocalTerminalStopException` with cleaner error handling
- [ ] Replace `TerminalCommandExecutor` with subprocess wrapper

### 1.3 Tool Parameter Model
- [ ] Extract `ParameterContainer` → `ToolParameter`:
  ```python
  @dataclass
  class ToolParameter:
      name: str
      type: str  # string, integer, float, boolean, array, object
      description: str
      required: bool = False
      default: Any = None
      options: list[ToolParameterOption] = field(default_factory=list)
      enum: list[str] = field(default_factory=list)
  ```
- [ ] Extract `ParameterOptionContainer` → `ToolParameterOption`
- [ ] Support JSON Schema generation from `ToolParameter` list (for LLM tool calling)

### 1.4 Remove Dependencies
- [ ] Remove Django model references (`UUID` from Django files system)
- [ ] Remove `exceptions.exceptions` imports — define own exceptions
- [ ] Remove `programs.services.terminal` — replace with own subprocess wrapper
- [ ] Remove `THOUGHT_ID`, `USER_ID` constants — these are QM-specific context

## Phase 2: Tool Registry

### 2.1 Registry Pattern
- [ ] Extract and enhance `InternalProgramRegistry` → `ToolRegistry`:
  ```python
  class ToolRegistry:
      def register(self, tool_cls: type[AbstractTool]) -> None: ...
      def get(self, name: str, version: Optional[str] = None) -> type[AbstractTool]: ...
      def list_tools(self) -> list[ToolInfo]: ...
      def to_json_schema(self) -> list[dict]: ...  # For LLM function calling
  ```
- [ ] Lazy initialization (tools loaded on first access)
- [ ] Version-aware lookup: `registry.get("web_search", version="1.0")`
- [ ] Decorator registration: `@register_tool`
- [ ] Plugin discovery: auto-find tools in installed packages via entry points

### 2.2 Tool Discovery
- [ ] Scan Python packages for `quartermaster_tools` entry point
- [ ] Support explicit registration and auto-discovery
- [ ] Namespace support: `registry.get("web.search")` vs `registry.get("file.read")`

## Phase 3: Built-in Tool Packs (Optional Extras)

### 3.1 Web Tools Pack (`quartermaster-tools[web]`)
- [ ] `WebRequest` — HTTP GET/POST/PUT/DELETE with response parsing
- [ ] `WebSearch` — Google search (via SerpAPI or similar)
- [ ] `WebScreenshot` — Take screenshot of URL (via Playwright)
- [ ] `WebCrawl` — Crawl and extract content from URL

### 3.2 File Tools Pack (`quartermaster-tools[files]`)
- [ ] `ReadFile` — Read file content
- [ ] `WriteFile` — Write content to file
- [ ] `ListDirectory` — List files in directory
- [ ] `MoveFile` / `CopyFile` — File operations
- [ ] `DeleteFile` — Delete file

### 3.3 Code Tools Pack (`quartermaster-tools[code]`)
- [ ] `RunPython` — Execute Python code (via quartermaster-code-runner)
- [ ] `RunNode` — Execute Node.js code
- [ ] `RunShell` — Execute shell command

### 3.4 Data Tools Pack (`quartermaster-tools[data]`)
- [ ] `ReadExcel` / `WriteExcel` — Spreadsheet operations
- [ ] `ReadPDF` / `WritePDF` — PDF operations
- [ ] `ReadCSV` / `WriteCSV` — CSV operations

### Note
- Not all QM internal programs need extraction
- Agent manipulation tools (30+) stay proprietary — they're QM platform-specific
- Hub/discovery tools stay proprietary
- Voice tools could be a separate optional pack

## Phase 4: JSON Schema Bridge

### 4.1 LLM Integration
- [ ] `ToolParameter.to_json_schema() -> dict` — convert to JSON Schema for tool calling
- [ ] `ToolRegistry.to_openai_tools() -> list[dict]` — OpenAI function calling format
- [ ] `ToolRegistry.to_anthropic_tools() -> list[dict]` — Anthropic tool format
- [ ] `ToolRegistry.to_mcp_tools() -> list[dict]` — MCP tool format
- [ ] This bridges `quartermaster-tools` with `quartermaster-providers`

## Phase 5: Testing

### 5.1 Core Tests
- [ ] Test `AbstractTool` interface contract
- [ ] Test `ToolResult` serialization
- [ ] Test `ToolParameter` to JSON Schema conversion
- [ ] Test `ToolRegistry` register/get/list
- [ ] Test version-aware lookup
- [ ] Test decorator registration

### 5.2 Built-in Tool Tests
- [ ] Test each built-in tool with mock filesystem / mock HTTP
- [ ] Test error handling (file not found, network error, timeout)
- [ ] Test parameter validation

## Phase 6: Documentation

### 6.1 README
- [ ] Quick start: define a tool, register it, execute it
- [ ] Built-in tools reference
- [ ] Custom tool creation guide
- [ ] Integration with quartermaster-providers (tool calling)
- [ ] Integration with quartermaster-mcp-client (MCP tools)

### 6.2 Custom Tool Tutorial
- [ ] Step-by-step: create "WeatherTool" that calls a weather API
- [ ] Register it, use it with LLM tool calling
- [ ] Package it as installable plugin

## Architecture Notes

### Relationship to Other Packages
- `quartermaster-providers` — Tools are formatted via `prepare_tool()` for each LLM provider
- `quartermaster-mcp-client` — MCP tools are a type of tool discovered at runtime
- `quartermaster-nodes` — ProgramRunner node executes tools from the registry
- `quartermaster-code-runner` — Code execution tools delegate to code-runner

### Why This Is Valuable
- Standardized tool interface across any framework
- Registry pattern with versioning is production-grade
- JSON Schema bridge makes tools work with any LLM's function calling
- Plugin system via entry points means community can contribute tools

## Timeline Estimate

- Phase 1 (Core): 2-3 days
- Phase 2 (Registry): 1-2 days
- Phase 3 (Built-in packs): 3-5 days
- Phase 4 (JSON Schema bridge): 1-2 days
- Phase 5 (Testing): 2-3 days
- Phase 6 (Docs): 1-2 days

**Total: 2-3 weeks**
