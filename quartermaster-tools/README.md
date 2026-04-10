# quartermaster-tools

Lightweight tool abstraction framework for AI agent orchestration.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-tools)](https://pypi.org/project/quartermaster-tools/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)

## Features

- **AbstractTool** base class with parameter validation, safe execution, and JSON Schema export
- **AbstractLocalTool** for subprocess-based tools with timeout and output parsing
- **Built-in tools**: ReadFileTool, WriteFileTool, WebRequestTool with security hardening
- **ToolRegistry** with version-aware lookup, plugin discovery via entry points, and decorator registration
- **Chain-of-Responsibility** pattern for composable data processing pipelines
- **LLM bridge methods**: `to_openai_tools()`, `to_anthropic_tools()`, `to_mcp_tools()` on both ToolDescriptor and ToolRegistry
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

### Creating a Custom Tool

```python
from quartermaster_tools import AbstractTool, ToolDescriptor, ToolParameter, ToolResult

class SentimentTool(AbstractTool):
    def name(self) -> str:
        return "analyze_sentiment"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                description="Text to analyze for sentiment",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Analyze text sentiment",
            long_description="Returns positive, negative, or neutral sentiment.",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        # Your sentiment logic here
        return ToolResult(success=True, data={"sentiment": "positive", "score": 0.92})

# Execute the tool
tool = SentimentTool()
result = tool.safe_run(text="I love this product!")
print(result.data)  # {"sentiment": "positive", "score": 0.92}
```

### Using the Tool Registry

```python
from quartermaster_tools import ToolRegistry, ReadFileTool, WriteFileTool, WebRequestTool

registry = ToolRegistry()
registry.register(ReadFileTool(allowed_base_dir="/tmp/sandbox"))
registry.register(WriteFileTool(allowed_base_dir="/tmp/sandbox", create_dirs=True))
registry.register(WebRequestTool(timeout=15))

# Lookup by name (returns latest version)
reader = registry.get("read_file")
result = reader.run(path="/tmp/sandbox/data.txt")

# List all registered tools
for desc in registry.list_tools():
    print(f"{desc.name} v{desc.version}: {desc.short_description}")

# Export for LLM function calling
openai_tools = registry.to_openai_tools()
anthropic_tools = registry.to_anthropic_tools()
mcp_tools = registry.to_mcp_tools()
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

## API Reference

### AbstractTool

The base class all tools must implement.

| Method | Description |
|--------|-------------|
| `name() -> str` | Unique tool identifier |
| `version() -> str` | Semantic version string |
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
| `load_plugins()` | Discover tools from `quartermaster_tools` entry points |
| `to_json_schema()` | Export all tools as JSON Schema |
| `to_openai_tools()` | Export in OpenAI function-calling format |
| `to_anthropic_tools()` | Export in Anthropic tool-use format |
| `to_mcp_tools()` | Export in MCP (Model Context Protocol) format |

### Built-in Tools

| Tool | Name | Description |
|------|------|-------------|
| `ReadFileTool` | `read_file` | Read file content with path validation, size limits, and SSRF protection |
| `WriteFileTool` | `write_file` | Write/append to files with blocked paths and size enforcement |
| `WebRequestTool` | `web_request` | HTTP GET/POST with SSRF protection and streaming response limits |

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
from quartermaster_nodes.protocols import ProgramContainer, ParameterContainer

registry = ToolRegistry()
registry.register(ReadFileTool())

# Convert tool descriptors to node-compatible format
for desc in registry.list_tools():
    openai_format = desc.to_openai_tools()
    # Pass to agent nodes for LLM function calling
```

### With quartermaster-providers (LLM provider integration)

```python
from quartermaster_tools import ToolRegistry, ReadFileTool, WriteFileTool

registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WriteFileTool())

# Export tools in the format your LLM provider expects
anthropic_tools = registry.to_anthropic_tools()
openai_tools = registry.to_openai_tools()
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 -- see [LICENSE](../LICENSE) for details.
