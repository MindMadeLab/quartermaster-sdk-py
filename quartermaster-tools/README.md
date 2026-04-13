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

| Alias | Function | Tool Name | Description |
|-------|----------|-----------|-------------|
| `ReadFileTool` | `read_file` | `read_file` | Read file content with path validation and size limits |
| `WriteFileTool` | `write_file` | `write_file` | Write/append to files with blocked paths and size enforcement |
| `WebRequestTool` | `web_request` | `web_request` | HTTP GET/POST with SSRF protection |
| `EvalMathTool` | `eval_math` | `eval_math` | Safe mathematical expression evaluation via AST parsing |
| `PythonExecutorTool` | `python_executor` | `python_executor` | Execute Python code |
| `JavaScriptExecutorTool` | `javascript_executor` | `javascript_executor` | Execute JavaScript code |
| `ShellExecutorTool` | `shell_executor` | `shell_executor` | Execute shell commands |
| `ParseJSONTool` | `parse_json` | `parse_json` | Parse JSON strings |
| `ParseXMLTool` | `parse_xml` | `parse_xml` | Parse XML strings |
| `ParseCSVTool` | `parse_csv` | `parse_csv` | Parse CSV strings |
| `ParseYAMLTool` | `parse_yaml` | `parse_yaml` | Parse YAML strings |
| `ConvertFormatTool` | `convert_format` | `convert_format` | Convert between data formats |
| `DataFilterTool` | `data_filter` | `data_filter` | Filter data with expressions |
| `GrepTool` | `grep` | `grep` | Search file contents |
| `FindFilesTool` | `find_files` | `find_files` | Find files by pattern |
| `ListDirectoryTool` | `list_directory` | `list_directory` | List directory contents |
| `CopyFileTool` | `copy_file` | `copy_file` | Copy files |
| `MoveFileTool` | `move_file` | `move_file` | Move/rename files |
| `DeleteFileTool` | `delete_file` | `delete_file` | Delete files |
| `CreateDirectoryTool` | `create_directory` | `create_directory` | Create directories |
| `FileInfoTool` | `file_info` | `file_info` | Get file metadata |
| `DuckDuckGoSearchTool` | `duckduckgo_search` | `duckduckgo_search` | Web search via DuckDuckGo |
| `WebScraperTool` | `web_scraper` | `web_scraper` | Scrape web pages |
| `JsonApiTool` | `json_api` | `json_api` | Call JSON APIs |

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
