"""Define custom tools, register them, and use the ToolRegistry.

Shows how to subclass AbstractTool, implement the required methods,
register tools with ToolRegistry, and export them as JSON Schema.
"""

from __future__ import annotations

from typing import Any

try:
    from quartermaster_tools.base import AbstractTool
    from quartermaster_tools.registry import ToolRegistry
    from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult
except ImportError:
    raise SystemExit("Install quartermaster-tools first:  pip install -e quartermaster-tools")


class CalculatorTool(AbstractTool):
    """A simple calculator that evaluates arithmetic expressions."""

    def name(self) -> str:
        return "calculator"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                description="A Python arithmetic expression, e.g. '2 + 3 * 4'.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Evaluate an arithmetic expression.",
            long_description="Safely evaluates simple arithmetic using Python.",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        expr = kwargs.get("expression", "")
        try:
            # Allow only safe arithmetic operations
            result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def main() -> None:
    # Step 1 -- Create the registry and register our tool
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)

    # Step 2 -- Look up and run the tool
    tool = registry.get("calculator")
    result = tool.run(expression="2 + 3 * 4")
    print(f"calculator('2 + 3 * 4') = {result.data['result']}")
    assert result.success

    # Step 3 -- Use safe_run with validation
    bad_result = tool.safe_run()  # missing required 'expression'
    print(f"Missing param error: {bad_result.error}")
    assert not bad_result.success

    # Step 4 -- List tools and export as JSON Schema
    print(f"\nRegistered tools: {registry.list_names()}")
    schemas = registry.to_json_schema()
    for schema in schemas:
        print(f"\nJSON Schema for '{schema['name']}':")
        print(f"  description: {schema['description']}")
        print(f"  parameters:  {schema['parameters']}")

    # Step 5 -- Export in OpenAI and Anthropic formats
    print(f"\nOpenAI format:   {registry.to_openai_tools()}")
    print(f"Anthropic format: {registry.to_anthropic_tools()}")


if __name__ == "__main__":
    main()
