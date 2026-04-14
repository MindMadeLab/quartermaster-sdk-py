# quartermaster-tools

Lightweight tool abstraction framework for AI agent orchestration.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-tools)](https://pypi.org/project/quartermaster-tools/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **`@tool()` decorator** -- the primary way to create tools from plain functions
- **FunctionTool** instances with metadata introspection (`info()`, `parameters()`)
- **Built-in tools**: file I/O, web requests, data parsing, math evaluation, code execution, and more
- **ToolRegistry** with version-aware lookup, plugin discovery via entry points, and decorator registration
- **Chain-of-Responsibility** pattern for composable data processing pipelines
- **LLM bridge methods**: `to_openai_tools()`, `to_anthropic_tools()`, `to_mcp_tools()` on both ToolDescriptor and ToolRegistry
- **AbstractTool** base class available for advanced use cases
- **Zero required dependencies** (httpx optional for WebRequestTool)

## Installation

```bash
pip install quartermaster-tools

# With optional HTTP support for WebRequestTool
pip install quartermaster-tools[web]

# With quartermaster-providers bridge for to_tool_definition()
pip install quartermaster-tools[llm]
```

## Quick Start

### Creating a Tool with @tool()

The `@tool()` decorator is the primary way to define tools. It extracts parameter metadata from the function signature, type hints, and Google-style docstring automatically.

```python
from quartermaster_tools import tool

@tool()
def analyze_sentiment(text: str, language: str = "en") -> dict:
    """Analyze text sentiment.

    Returns positive, negative, or neutral sentiment with a confidence score.

    Args:
        text: Text to analyze for sentiment.
        language: ISO language code for the input text.
    """
    # Your sentiment logic here
    return {"sentiment": "positive", "score": 0.92}

# Call it like a normal function
result = analyze_sentiment(text="I love this product!")

# Or execute via the tool interface (returns ToolResult)
tool_result = analyze_sentiment.run(text="I love this product!")
print(tool_result.data)  # {"sentiment": "positive", "score": 0.92}

# Introspect metadata
print(analyze_sentiment.name())        # "analyze_sentiment"
print(analyze_sentiment.parameters())  # [ToolParameter(name="text", ...), ToolParameter(name="language", ...)]

# Export JSON Schema for LLM function calling
schema = analyze_sentiment.info().to_input_schema()
```

You can override the tool name and description:

```python
@tool(name="sentiment_v2", description="Advanced sentiment analysis")
def analyze(text: str) -> dict:
    ...
```

### Using the Tool Registry

```python
from quartermaster_tools import ToolRegistry, tool

registry = ToolRegistry()

# Register existing FunctionTool instances (built-in tools are already FunctionTool instances)
from quartermaster_tools import ReadFileTool, WriteFileTool, WebRequestTool

registry.register(ReadFileTool)
registry.register(WriteFileTool)
registry.register(WebRequestTool)

# Or use the registry's own @tool decorator to create and register in one step
@registry.tool()
def summarize(text: str, max_length: int = 100) -> dict:
    """Summarize text to a given length.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary.
    """
    return {"summary": text[:max_length]}

# Lookup by name (returns latest version)
reader = registry.get("read_file")
result = reader.run(path="/tmp/data.txt")

# List all registered tools
for desc in registry.list_tools():
    print(f"{desc.name} v{desc.version}: {desc.short_description}")

# Export for LLM function calling
openai_tools = registry.to_openai_tools()
anthropic_tools = registry.to_anthropic_tools()
mcp_tools = registry.to_mcp_tools()
json_schemas = registry.to_json_schema()
```

### Building a Handler Chain

```python
from quartermaster_tools import Chain, Handler

class ValidateInput(Handler):
    def handle(self, data: dict) -> dict:
        if "query" not in data:
            raise ValueError("Missing required field: query")
        return data

class NormalizeText(Handler):
    def handle(self, data: dict) -> dict:
        data["query"] = data["query"].strip().lower()
        return data

class AddTimestamp(Handler):
    def handle(self, data: dict) -> dict:
        from datetime import datetime
        data["timestamp"] = datetime.now().isoformat()
        return data

chain = (
    Chain()
    .add_handler(ValidateInput())
    .add_handler(NormalizeText())
    .add_handler(AddTimestamp())
)

result = chain.run({"query": "  Hello World  "})
# {"query": "hello world", "timestamp": "2025-01-15T10:30:00"}
```

### Advanced: Creating a Tool with AbstractTool

For cases where you need full control over tool construction (e.g., custom validation logic, stateful tools), subclass `AbstractTool` directly:

```python
from quartermaster_tools import AbstractTool, ToolDescriptor, ToolParameter, ToolResult

class DatabaseQueryTool(AbstractTool):
    def __init__(self, connection_string: str):
        self._conn = connection_string

    def name(self) -> str:
        return "db_query"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="sql", description="SQL query to execute", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute a database query",
            long_description="Runs a SQL query against the configured database.",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs) -> ToolResult:
        sql = kwargs.get("sql", "")
        # Execute query...
        return ToolResult(success=True, data={"rows": []})

tool = DatabaseQueryTool(connection_string="sqlite:///mydb.db")
result = tool.safe_run(sql="SELECT * FROM users LIMIT 10")
```

## API Reference

### @tool() Decorator

Converts a plain function into a `FunctionTool` instance. Parameters are extracted from the function signature and type hints. Descriptions come from the Google-style docstring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Override tool name (defaults to function `__name__`) |
| `description` | `str \| None` | `None` | Override short description (defaults to docstring first line) |

### FunctionTool

A tool created by the `@tool()` decorator. Subclass of `AbstractTool`.

| Method | Description |
|--------|-------------|
| `name() -> str` | Tool name |
| `version() -> str` | Version string (always `"1.0.0"`) |
| `parameters() -> list[ToolParameter]` | Parameter definitions extracted from the function |
| `info() -> ToolDescriptor` | Full tool metadata |
| `run(**kwargs) -> ToolResult` | Execute the tool, returning a ToolResult |
| `__call__(*args, **kwargs)` | Call the underlying function directly |

### AbstractTool

The base class for advanced tool implementations.

| Method | Description |
|--------|-------------|
| `name() -> str` | Unique tool identifier |
| `version() -> str` | Semantic version string (default: `"1.0.0"`) |
| `parameters() -> list[ToolParameter]` | Parameter definitions |
| `info() -> ToolDescriptor` | Tool metadata |
| `run(**kwargs) -> ToolResult` | Execute the tool |
| `validate_params(**kwargs) -> list[str]` | Validate parameters, returns error list |
| `safe_run(**kwargs) -> ToolResult` | Validate then run; returns error result on validation failure |

### AbstractLocalTool

Extends AbstractTool for subprocess-based tools.

| Method | Description |
|--------|-------------|
| `prepare_command(**kwargs) -> list[str]` | Build the command-line arguments |
| `parse_output(stdout, stderr, returncode) -> ToolResult` | Convert subprocess output to result |
| `timeout() -> int` | Max execution time in seconds (default: 30) |
| `working_directory() -> str \| None` | Working directory for the subprocess |

### ToolDescriptor

Metadata describing a tool, with LLM bridge methods.

| Method | Description |
|--------|-------------|
| `to_input_schema() -> dict` | JSON Schema for the tool's parameters |
| `to_openai_tools() -> dict` | OpenAI function-calling format |
| `to_anthropic_tools() -> dict` | Anthropic tool-use format |
| `to_tool_definition() -> ToolDefinition` | quartermaster-providers ToolDefinition (requires `quartermaster-tools[llm]`) |

### ToolRegistry

Central registry with version-aware lookup and plugin discovery.

| Method | Description |
|--------|-------------|
| `register(tool)` | Register a tool instance |
| `get(name, version=None)` | Look up by name (latest if no version) |
| `list_tools() -> list[ToolDescriptor]` | All registered tool descriptors |
| `list_names() -> list[str]` | All registered tool names |
| `unregister(name, version=None)` | Remove a tool |
| `tool(name=None, description=None)` | Decorator to create and register a tool in one step |
| `load_plugins()` | Discover tools from `quartermaster_tools` entry points |
| `to_json_schema()` | Export all tools as JSON Schema |
| `to_openai_tools()` | Export in OpenAI function-calling format |
| `to_anthropic_tools()` | Export in Anthropic tool-use format |
| `to_mcp_tools()` | Export in MCP (Model Context Protocol) format |

### Built-in Tools

All built-in tools are `FunctionTool` instances created with `@tool()`. The `*Tool` names (e.g., `ReadFileTool`) are backward-compatible aliases pointing to the same `FunctionTool` instance.

79 tools across 15 categories:

| Category | Tool | Description |
|----------|------|-------------|
| **File I/O** | `read_file` | Read content from a file with path validation and size limits |
| **File I/O** | `write_file` | Write or append content to a file with blocked-path and size enforcement |
| **Filesystem** | `list_directory` | List directory entries with type, size, and modification time |
| **Filesystem** | `find_files` | Find files using glob patterns and optional regex name filtering |
| **Filesystem** | `grep` | Search file contents for a regex pattern with context lines |
| **Filesystem** | `copy_file` | Copy a file or directory |
| **Filesystem** | `move_file` | Move or rename a file or directory |
| **Filesystem** | `delete_file` | Delete a file or directory (requires confirmation) |
| **Filesystem** | `create_directory` | Create a directory with optional parent creation |
| **Filesystem** | `file_info` | Get file metadata (size, type, permissions, MIME type) |
| **Code** | `python_executor` | Execute Python code in a subprocess with timeout enforcement |
| **Code** | `javascript_executor` | Execute JavaScript code via Node.js subprocess |
| **Code** | `shell_executor` | Execute shell commands with blocked-command safety list |
| **Code** | `eval_math` | Safely evaluate math expressions via AST parsing (no eval/exec) |
| **Data** | `parse_json` | Parse JSON from a file or string with optional JMESPath queries |
| **Data** | `parse_csv` | Parse CSV from a file or string with custom delimiters and headers |
| **Data** | `parse_yaml` | Parse YAML from a file or string (requires pyyaml) |
| **Data** | `parse_xml` | Parse XML from a file or string with optional XPath queries |
| **Data** | `convert_format` | Convert data between CSV, JSON, and YAML formats |
| **Data** | `data_filter` | Filter, sort, and limit structured data with safe expressions |
| **Database** | `sqlite_query` | Execute read-only SQL queries on a SQLite database |
| **Database** | `sqlite_write` | Execute write SQL statements on SQLite (requires confirm=True) |
| **Database** | `sqlite_schema` | Introspect SQLite database schema (tables and columns) |
| **Web** | `web_request` | HTTP requests (GET/POST/PUT/DELETE/PATCH) with SSRF protection |
| **Web Search** | `duckduckgo_search` | Search the web via DuckDuckGo HTML (no API key needed) |
| **Web Search** | `brave_search` | Search the web via Brave Search API (requires BRAVE_API_KEY) |
| **Web Search** | `google_search` | Search the web via Google Custom Search API |
| **Web Search** | `web_scraper` | Fetch a web page and return content as text, markdown, or HTML |
| **Web Search** | `json_api` | Call a JSON API with optional JMESPath response filtering |
| **Vector/RAG** | `embed_text` | Generate text embeddings (built-in hash-based or sentence-transformers) |
| **Vector/RAG** | `vector_store` | Store text with vector embeddings in memory or JSON files |
| **Vector/RAG** | `vector_search` | Cosine-similarity search over a vector collection |
| **Vector/RAG** | `hybrid_search` | Combined semantic + keyword search with configurable weighting |
| **Vector/RAG** | `document_index` | Chunk and index a document for vector search |
| **Browser** | `browser_navigate` | Navigate browser to a URL and wait for page load |
| **Browser** | `browser_wait` | Wait for an element to reach a desired state (visible/hidden/attached) |
| **Browser** | `browser_click` | Click an element on the page by CSS selector |
| **Browser** | `browser_type` | Type text into an input field by CSS selector |
| **Browser** | `browser_eval` | Execute JavaScript in the browser page context |
| **Browser** | `browser_extract` | Extract text or HTML content from the page or an element |
| **Browser** | `browser_screenshot` | Capture a PNG screenshot of the page or an element |
| **Email** | `send_email` | Send email via SMTP with TLS (rate limited to 10/min) |
| **Email** | `read_email` | Read emails from an IMAP mailbox with unread filtering |
| **Email** | `search_email` | Search emails via IMAP by text, date range, and sender |
| **Messaging** | `slack_message` | Send a message to a Slack channel via Web API |
| **Messaging** | `slack_read` | Read recent messages from a Slack channel |
| **Messaging** | `webhook_notify` | Send a JSON POST to any webhook URL |
| **Messaging** | `discord_message` | Send a message to Discord via webhook |
| **A2A** | `a2a_discover` | Discover a remote A2A agent's capabilities via Agent Card |
| **A2A** | `a2a_send_task` | Send a task to a remote A2A agent via JSON-RPC 2.0 |
| **A2A** | `a2a_check_status` | Check the status of a remote A2A task |
| **A2A** | `a2a_collect_result` | Collect results from a completed A2A task |
| **A2A** | `a2a_register` | Generate an A2A Agent Card for local agent registration |
| **Agents** | `spawn_agent` | Create and start a parallel agent session in one step |
| **Agents** | `create_agent_session` | Create a new parallel agent session for later start |
| **Agents** | `start_agent_session` | Start a previously created agent session with a task |
| **Agents** | `inject_agent_message` | Inject a message into a running agent session |
| **Agents** | `get_agent_session_status` | Get detailed status of an agent session |
| **Agents** | `list_agent_sessions` | List all agent sessions with optional status filtering |
| **Agents** | `wait_agent_session` | Block until an agent session completes or times out |
| **Agents** | `collect_agent_results` | Collect results from multiple agent sessions |
| **Agents** | `cancel_agent_session` | Cancel a running agent session |
| **Agents** | `add_agent_finish_hook` | Register a log or notify callback for session completion |
| **Agents** | `notify_parent` | Send a status update from a sub-agent to its parent |
| **Memory** | `set_variable` | Store a key-value pair in shared in-memory storage |
| **Memory** | `get_variable` | Retrieve a stored variable from in-memory storage |
| **Memory** | `list_variables` | List stored variable names with optional prefix filter |
| **Privacy** | `detect_pii` | Detect PII entities in text (email, phone, SSN, credit card, etc.) |
| **Privacy** | `redact_pii` | Redact PII from text using redact, mask, or hash strategies |
| **Privacy** | `scan_file_pii` | Scan a file for PII entities with line-number reporting |
| **Compliance** | `risk_classifier` | Classify AI system risk level per EU AI Act Annex III |
| **Compliance** | `audit_log` | Append to a tamper-evident (SHA-256 chained) audit trail |
| **Compliance** | `read_audit_log` | Query and verify audit trail integrity |
| **Compliance** | `compliance_checklist` | Generate EU AI Act compliance checklists by risk level |
| **Observability** | `trace` | Create a trace span for distributed observability |
| **Observability** | `performance_profile` | Record tool execution timing with summary statistics |
| **Observability** | `log` | Write structured JSON log entries (in-memory + optional file) |
| **Observability** | `metric` | Record custom metrics (counter, gauge, or histogram) |
| **Observability** | `cost_tracker` | Track LLM API call costs with built-in pricing tables |

### Decorator Registration

```python
from quartermaster_tools import register_tool, get_default_registry, AbstractTool

@register_tool
class MyTool(AbstractTool):
    ...

# The tool is automatically registered in the default registry
registry = get_default_registry()
tool = registry.get("my_tool")
```

### Plugin Discovery

Register tools as entry points in your `pyproject.toml`:

```toml
[project.entry-points.quartermaster_tools]
my_tool = "my_package.tools:MyTool"
```

The registry discovers and loads these automatically on first access.

## Integration with Sibling Packages

### With quartermaster-nodes (tool execution in agent graphs)

```python
from quartermaster_tools import ToolRegistry, ReadFileTool

registry = ToolRegistry()
registry.register(ReadFileTool)

# Convert tool descriptors to node-compatible format
for desc in registry.list_tools():
    openai_format = desc.to_openai_tools()
    # Pass to agent nodes for LLM function calling
```

### With quartermaster-providers (LLM provider integration)

```python
from quartermaster_tools import ToolRegistry, ReadFileTool, WriteFileTool

registry = ToolRegistry()
registry.register(ReadFileTool)
registry.register(WriteFileTool)

# Export tools in the format your LLM provider expects
anthropic_tools = registry.to_anthropic_tools()
openai_tools = registry.to_openai_tools()
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
