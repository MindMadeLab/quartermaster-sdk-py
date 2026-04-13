# Tool System

The `quartermaster-tools` package provides a framework for defining, registering, and executing tools that AI agents can invoke. Tools are units of functionality -- file operations, web requests, calculations, or any custom logic -- that extend what an agent can do beyond LLM text generation.

## Quick Start: The @tool Decorator

The `@tool` decorator turns a plain Python function into a tool. No separate registration step needed:

```python
from quartermaster_tools import tool

@tool()
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to look up.
        units: Temperature units (celsius or fahrenheit).
    """
    return {"city": city, "temperature": 22, "units": units}

# That's it. The decorator defines the tool completely.
result = get_weather(city="Amsterdam")          # Call directly
schema = get_weather.to_json_schema()           # Export for LLM function calling
print(get_weather.name())                       # "get_weather"
print(get_weather.parameters())                 # [ToolParameter(name="city", ...), ...]
```

To manage multiple tools together, use a `ToolRegistry`:

```python
from quartermaster_tools import ToolRegistry

registry = ToolRegistry()

@registry.tool()
def search_database(query: str, limit: int = 10) -> dict:
    """Search the knowledge database.

    Args:
        query: Search query string.
        limit: Maximum results to return.
    """
    return {"results": [f"Result for: {query}"], "count": 1}

@registry.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {"sent": True, "to": to}

# All tools are registered by the decorator -- no add_tool() calls needed
schemas = registry.to_json_schema()  # Export all tools for LLM function calling
```

The decorator extracts metadata from type hints and Google-style docstrings. It supports sync and async functions, all standard Python types (`str` -> `string`, `int` -> `integer`, `float` -> `number`, `bool` -> `boolean`, `list` -> `array`, `dict` -> `object`), and default values for optional parameters.

## Core Types

### AbstractTool

For more control, inherit from `AbstractTool` and implement five methods:

```python
from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

class MyTool(AbstractTool):
    def name(self) -> str:
        return "my_tool"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                description="The input text to process",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="max_length",
                description="Maximum output length",
                type="integer",
                required=False,
                default=100,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Processes input text",
            long_description="A detailed description of what this tool does.",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs) -> ToolResult:
        text = kwargs["input"]
        max_len = kwargs.get("max_length", 100)
        result = text[:max_len].upper()
        return ToolResult(success=True, data={"output": result})
```

### ToolParameter

Defines a single input parameter for a tool:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Parameter identifier |
| `description` | `str` | Human-readable description for the LLM |
| `type` | `str` | JSON Schema type: `string`, `integer`, `number`, `boolean`, `array`, `object` |
| `required` | `bool` | Whether the parameter is mandatory (default: `False`) |
| `default` | `Any` | Default value when not provided |
| `options` | `list[ToolParameterOption]` | Enumerated choices (rendered as JSON Schema `enum`) |
| `validation` | `Any` | Optional callable for custom validation |

### ToolParameterOption

Defines a selectable option for a parameter:

```python
from quartermaster_tools.types import ToolParameter, ToolParameterOption

param = ToolParameter(
    name="format",
    description="Output format",
    type="string",
    options=[
        ToolParameterOption(label="JSON", value="json"),
        ToolParameterOption(label="CSV", value="csv"),
        ToolParameterOption(label="Plain Text", value="text"),
    ],
)
```

### ToolDescriptor

Metadata describing a tool. Used for registration, discovery, and schema export:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool identifier |
| `short_description` | `str` | Brief description (shown to LLMs) |
| `long_description` | `str` | Detailed documentation |
| `version` | `str` | Semantic version string |
| `parameters` | `list[ToolParameter]` | Input parameters |
| `is_local` | `bool` | Whether the tool runs locally |

### ToolResult

Returned by `tool.run()`:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether execution succeeded |
| `data` | `dict` | Output data (arbitrary key-value) |
| `error` | `str` | Error message if failed |
| `metadata` | `dict` | Additional metadata |

`ToolResult` is truthy when `success=True`, so you can use it in boolean contexts: `if result: ...`

## ToolRegistry

The `ToolRegistry` manages tool instances with version-aware lookup, decorator registration, and plugin discovery.

### Basic Usage

```python
from quartermaster_tools.registry import ToolRegistry

registry = ToolRegistry()

# Register a tool instance
tool = MyTool()
registry.register(tool)

# Look up by name (returns latest version)
tool = registry.get("my_tool")

# Look up by name and version
tool = registry.get("my_tool", version="1.0.0")

# List all registered tools
descriptors = registry.list_tools()  # returns list[ToolDescriptor]
names = registry.list_names()        # returns list[str]

# Check if registered
if "my_tool" in registry:
    ...

# Remove
registry.unregister("my_tool")
registry.unregister("my_tool", version="1.0.0")  # specific version only
registry.clear()  # remove all
```

### Decorator Registration

Register tools using the `@registry.tool()` decorator for functions:

```python
registry = ToolRegistry()

@registry.tool()
def fetch_weather(city: str) -> dict:
    """Fetch current weather for a city."""
    return {"city": city, "temp": 72}

tool = registry.get("fetch_weather")
```

Or use the `@register_tool` class decorator to auto-register with the module-level default registry:

```python
from quartermaster_tools.registry import register_tool, get_default_registry

@register_tool
class WebSearchTool(AbstractTool):
    def name(self) -> str:
        return "web_search"
    # ... implement remaining methods ...

# Retrieve from the default registry
registry = get_default_registry()
tool = registry.get("web_search")
```

### Plugin Discovery

The registry can auto-discover tools from installed packages using Python entry points:

```toml
# In your package's pyproject.toml
[project.entry-points.quartermaster_tools]
my_tool = "my_package.tools:MyTool"
```

```python
registry = ToolRegistry()
registry.load_plugins()  # discovers and registers all quartermaster_tools entry points

# Or let it happen lazily on first access
tool = registry.get("my_tool")  # triggers load_plugins() if not yet called
```

### Version-Aware Lookup

Multiple versions of the same tool can coexist:

```python
registry.register(MyToolV1())  # name="my_tool", version="1.0.0"
registry.register(MyToolV2())  # name="my_tool", version="2.0.0"

latest = registry.get("my_tool")                # returns v2.0.0
specific = registry.get("my_tool", version="1.0.0")  # returns v1.0.0
```

## Parameter Validation

`AbstractTool` provides built-in parameter validation:

```python
tool = MyTool()

# Manual validation
errors = tool.validate_params(input="hello")  # [] (valid)
errors = tool.validate_params()               # ["Missing required parameter: input"]

# Safe execution (validates first, returns error result on failure)
result = tool.safe_run(input="hello")  # runs normally
result = tool.safe_run()               # ToolResult(success=False, error="Missing required parameter: input")
```

## AbstractLocalTool

For tools that execute local subprocess commands, extend `AbstractLocalTool`:

```python
from quartermaster_tools.base import AbstractLocalTool

class GitStatusTool(AbstractLocalTool):
    def name(self) -> str:
        return "git_status"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="path", description="Repository path", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Get git status for a repository",
            long_description="Runs git status in the specified directory.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def prepare_command(self, **kwargs) -> list[str]:
        return ["git", "status", "--short"]

    def working_directory(self) -> str | None:
        return kwargs.get("path")

    def timeout(self) -> int:
        return 10  # 10 second timeout
```

`AbstractLocalTool` handles subprocess execution, timeout enforcement, and output parsing automatically. Override `parse_output()` for custom result formatting.

## JSON Schema Export

The registry exports tool definitions in formats required by different LLM providers:

### Generic JSON Schema

```python
schemas = registry.to_json_schema()
# [{"name": "my_tool", "description": "...", "parameters": {"type": "object", ...}}]
```

### OpenAI Function Calling Format

```python
tools = registry.to_openai_tools()
# [{"type": "function", "function": {"name": "my_tool", "description": "...", "parameters": {...}}}]
```

### Anthropic Tool Use Format

```python
tools = registry.to_anthropic_tools()
# [{"name": "my_tool", "description": "...", "input_schema": {"type": "object", ...}}]
```

### MCP (Model Context Protocol) Format

```python
tools = registry.to_mcp_tools()
# [{"name": "my_tool", "description": "...", "inputSchema": {"type": "object", ...}}]
```

### Per-Tool Export

Individual `ToolDescriptor` instances can also export their schema:

```python
descriptor = tool.info()

# OpenAI format
openai_tool = descriptor.to_openai_tools()

# Anthropic format
anthropic_tool = descriptor.to_anthropic_tools()

# quartermaster-providers ToolDefinition (requires quartermaster-providers installed)
tool_def = descriptor.to_tool_definition()
```

## Bridging to quartermaster-providers

The `ToolDescriptor.to_tool_definition()` method converts a tool descriptor to a `quartermaster_providers.types.ToolDefinition`, which can be passed directly to any `AbstractLLMProvider.generate_tool_parameters()` call:

```python
from quartermaster_providers.types import ToolDefinition

# Convert tool descriptors for use with providers
tool_defs = [tool.info().to_tool_definition() for tool in my_tools]

# Pass to provider
response = await provider.generate_tool_parameters(
    prompt="Find information about Python",
    tools=tool_defs,
    config=config,
)

# Execute the tool calls
for call in response.tool_calls:
    tool = registry.get(call.tool_name)
    result = tool.safe_run(**call.parameters)
```

This bridge requires the optional `quartermaster-providers` dependency. Install with `pip install quartermaster-tools[llm]`.

## See Also

- [Tools Catalog](tools-catalog.md) -- Complete reference for all built-in tools with parameters, examples, and categories
- [Architecture](architecture.md) -- How tools fit into the system
- [Providers](providers.md) -- LLM providers that consume tool definitions
- [Graph Building](graph-building.md) -- Tool nodes in agent graphs
- [Security](security.md) -- Security considerations for tool execution
