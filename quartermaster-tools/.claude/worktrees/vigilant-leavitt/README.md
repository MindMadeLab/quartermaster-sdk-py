# quartermaster-tools

Tool abstraction framework and chain-of-responsibility handler pattern for AI agent orchestration.

`quartermaster-tools` provides a lightweight, dependency-free foundation for building composable AI agent tools and processing pipelines. Define tool capabilities, chain handlers together, and build sophisticated agent workflows.

## Features

- **AbstractTool**: Base class for defining AI agent capabilities with parameters and results
- **Chain-of-Responsibility Pattern**: Composable handler pipelines for data processing
- **Parameter System**: Type-safe tool parameter definitions with validation support
- **Tool Registry**: Lazy-loaded, version-aware tool management
- **Zero Dependencies**: Pure Python, no external dependencies required

## Installation

```bash
pip install quartermaster-tools
```

## Quick Start

### Creating a Tool

```python
from quartermaster_tools import AbstractTool, ToolDescriptor, ToolParameter, ToolResult

class CalculatorTool(AbstractTool):
    """A simple calculator tool."""

    def name(self) -> str:
        return "calculator"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="operation",
                description="Operation to perform: add, subtract, multiply, divide",
                type="string",
                required=True,
                options=[
                    ToolParameterOption(label="Add", value="add"),
                    ToolParameterOption(label="Subtract", value="subtract"),
                    ToolParameterOption(label="Multiply", value="multiply"),
                    ToolParameterOption(label="Divide", value="divide"),
                ],
            ),
            ToolParameter(
                name="a",
                description="First operand",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="b",
                description="Second operand",
                type="number",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Perform basic arithmetic operations",
            long_description="A tool that can add, subtract, multiply, or divide two numbers",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs) -> ToolResult:
        operation = kwargs.get("operation")
        a = kwargs.get("a")
        b = kwargs.get("b")

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return ToolResult(
                    success=False,
                    error="Division by zero",
                )
            result = a / b
        else:
            return ToolResult(
                success=False,
                error=f"Unknown operation: {operation}",
            )

        return ToolResult(
            success=True,
            data={"result": result},
        )
```

### Building a Handler Chain

```python
from quartermaster_tools import Chain, Handler

class LoggingHandler(Handler):
    """A handler that logs data."""

    def handle(self, data: dict) -> dict:
        print(f"Processing: {data}")
        return data

class ValidationHandler(Handler):
    """A handler that validates required fields."""

    def handle(self, data: dict) -> dict:
        if "input" not in data:
            raise ValueError("Missing required field: input")
        return data

class TransformHandler(Handler):
    """A handler that transforms data."""

    def handle(self, data: dict) -> dict:
        data["output"] = data["input"].upper()
        return data

# Build the chain
chain = (
    Chain()
    .add_handler(LoggingHandler())
    .add_handler(ValidationHandler())
    .add_handler(TransformHandler())
)

# Execute
result = chain.run({"input": "hello"})
print(result)  # {'input': 'hello', 'output': 'HELLO'}
```

## Architecture

### Tool Abstraction

Tools are the primary abstraction for AI agent capabilities. Each tool:

1. Declares its **name** and **version**
2. Exposes **parameters** that define inputs (with types, descriptions, validation options)
3. Implements **run()** to execute the tool logic
4. Returns a **ToolResult** with success/error status and data

Tools are composable — agents can reference tools internally, creating tool hierarchies.

### Chain-of-Responsibility Pattern

Handlers form a processing pipeline where each handler:

1. Receives input data
2. Processes or modifies it
3. Passes it to the next handler
4. Or halts the chain (by raising an exception)

Chains are reusable, composable, and testable in isolation.

### Parameter System

Tool parameters are declarative:

- **name**: Parameter identifier
- **type**: Type constraint (string, number, boolean, array, object)
- **description**: Human-readable description
- **required**: Is this parameter mandatory?
- **options**: Pre-defined choices (for UI dropdowns, etc.)
- **validation**: Custom validation rules (optional, extensible)

## Advanced Usage

### Custom Parameter Validation

```python
from quartermaster_tools import ToolParameter

def validate_positive(value):
    if value <= 0:
        raise ValueError("Value must be positive")
    return value

param = ToolParameter(
    name="count",
    description="Number of items",
    type="number",
    required=True,
    validation=validate_positive,
)
```

### Error Handling in Handlers

```python
class SafeHandler(Handler):
    """A handler with error recovery."""

    def handle(self, data: dict) -> dict:
        try:
            # Risky operation
            return self._process(data)
        except Exception as e:
            print(f"Handler error: {e}")
            # Either recover or re-raise
            data["error"] = str(e)
            return data

    def _process(self, data: dict) -> dict:
        raise NotImplementedError()
```

### Tool Registry

```python
from quartermaster_tools import ToolRegistry

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(StringTool())

# Lookup by name and version
tool = registry.get("calculator", "1.0.0")

# List all tools
all_tools = registry.list_tools()
```

## Testing Tools

```python
import pytest
from quartermaster_tools import ToolResult

def test_calculator_add():
    tool = CalculatorTool()
    result = tool.run(operation="add", a=2, b=3)

    assert result.success
    assert result.data["result"] == 5

def test_calculator_divide_by_zero():
    tool = CalculatorTool()
    result = tool.run(operation="divide", a=5, b=0)

    assert not result.success
    assert "Division by zero" in result.error
```

## Best Practices

1. **Single Responsibility**: Each tool should do one thing well
2. **Clear Parameters**: Document what each parameter does and what values it accepts
3. **Graceful Errors**: Return ToolResult with error message instead of raising exceptions (unless fatal)
4. **Versioning**: Follow semantic versioning for tool versions
5. **Composability**: Design tools to work together; avoid tight coupling
6. **Testing**: Test tools in isolation before integrating into chains
7. **Immutability**: Avoid side effects in handler chains; return new data instead of mutating input

## Contributing

Contributions welcome! Please ensure:

- All tests pass: `pytest`
- Code is formatted: `ruff format`
- Types check: `mypy`
- No lint errors: `ruff check`

## License

Apache License 2.0 — see LICENSE file for details.

## Related Projects

- **quartermaster-graph** — Agent graph definition and serialization
- **quartermaster-engine** — DAG-based flow execution engine
- **quartermaster-providers** — LLM provider abstractions
- **Quartermaster** — Full AI agent ecosystem platform
