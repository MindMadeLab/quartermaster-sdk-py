"""Tests for the @tool() decorator and FunctionTool."""

from __future__ import annotations

import asyncio

import pytest

from quartermaster_tools import FunctionTool, ToolRegistry, tool
from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


# ---------------------------------------------------------------------------
# 1. Basic function -> tool conversion
# ---------------------------------------------------------------------------


class TestBasicConversion:
    def test_basic_function_to_tool(self):
        @tool()
        def greet(name: str) -> dict:
            """Say hello."""
            return {"greeting": f"Hello, {name}!"}

        assert isinstance(greet, FunctionTool)
        assert isinstance(greet, AbstractTool)

    def test_extracts_name_from_function_name(self):
        @tool()
        def my_awesome_tool(x: str) -> dict:
            """Does stuff."""
            return {}

        assert greet_tool_name(my_awesome_tool) == "my_awesome_tool"

    def test_extracts_description_from_docstring_first_line(self):
        @tool()
        def summarize(text: str) -> dict:
            """Summarize a piece of text.

            This tool takes text and produces a summary.
            """
            return {}

        assert summarize.info().short_description == "Summarize a piece of text."

    def test_extracts_long_description_from_full_docstring(self):
        @tool()
        def summarize(text: str) -> dict:
            """Summarize a piece of text.

            This tool takes text and produces a summary.
            """
            return {}

        info = summarize.info()
        assert "This tool takes text and produces a summary." in info.long_description


# ---------------------------------------------------------------------------
# 2. Type annotation parsing
# ---------------------------------------------------------------------------


class TestTypeAnnotations:
    def test_str_maps_to_string(self):
        @tool()
        def f(x: str) -> dict:
            return {}

        assert f.parameters()[0].type == "string"

    def test_int_maps_to_integer(self):
        @tool()
        def f(x: int) -> dict:
            return {}

        assert f.parameters()[0].type == "integer"

    def test_float_maps_to_number(self):
        @tool()
        def f(x: float) -> dict:
            return {}

        assert f.parameters()[0].type == "number"

    def test_bool_maps_to_boolean(self):
        @tool()
        def f(x: bool) -> dict:
            return {}

        assert f.parameters()[0].type == "boolean"

    def test_list_maps_to_array(self):
        @tool()
        def f(x: list) -> dict:
            return {}

        assert f.parameters()[0].type == "array"

    def test_dict_maps_to_object(self):
        @tool()
        def f(x: dict) -> dict:
            return {}

        assert f.parameters()[0].type == "object"


# ---------------------------------------------------------------------------
# 3. Required vs optional parameters
# ---------------------------------------------------------------------------


class TestRequiredOptional:
    def test_required_params_no_default(self):
        @tool()
        def f(x: str) -> dict:
            return {}

        param = f.parameters()[0]
        assert param.required is True
        assert param.default is None

    def test_optional_params_with_default(self):
        @tool()
        def f(x: str = "hello") -> dict:
            return {}

        param = f.parameters()[0]
        assert param.required is False
        assert param.default == "hello"

    def test_only_required_params(self):
        @tool()
        def f(a: str, b: int) -> dict:
            return {}

        assert all(p.required for p in f.parameters())

    def test_only_optional_params(self):
        @tool()
        def f(a: str = "x", b: int = 5) -> dict:
            return {}

        assert all(not p.required for p in f.parameters())


# ---------------------------------------------------------------------------
# 4. Google-style docstring Args parsing
# ---------------------------------------------------------------------------


class TestDocstringArgsParsing:
    def test_parses_google_style_args(self):
        @tool()
        def get_weather(city: str, units: str = "celsius") -> dict:
            """Get current weather for a city.

            Args:
                city: The city name to look up weather for.
                units: Temperature units (celsius or fahrenheit).
            """
            return {}

        params = {p.name: p for p in get_weather.parameters()}
        assert params["city"].description == "The city name to look up weather for."
        assert params["units"].description == "Temperature units (celsius or fahrenheit)."

    def test_multiline_param_description(self):
        @tool()
        def f(query: str) -> dict:
            """Search for things.

            Args:
                query: The search query string.
                    Can be multiple lines.
            """
            return {}

        params = {p.name: p for p in f.parameters()}
        assert "Can be multiple lines" in params["query"].description


# ---------------------------------------------------------------------------
# 5. Custom name/description overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_custom_name_override(self):
        @tool(name="custom_tool_name")
        def my_func(x: str) -> dict:
            """Original description."""
            return {}

        assert my_func.name() == "custom_tool_name"

    def test_custom_description_override(self):
        @tool(description="Overridden description")
        def my_func(x: str) -> dict:
            """Original description."""
            return {}

        assert my_func.info().short_description == "Overridden description"


# ---------------------------------------------------------------------------
# 6. Return value handling
# ---------------------------------------------------------------------------


class TestReturnValueHandling:
    def test_returns_tool_result_when_function_returns_dict(self):
        @tool()
        def f(x: str) -> dict:
            return {"key": "value"}

        result = f.run(x="test")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == {"key": "value"}

    def test_returns_tool_result_when_function_returns_non_dict(self):
        @tool()
        def f(x: int) -> int:
            return 42

        result = f.run(x=1)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == {"result": 42}

    def test_returns_tool_result_when_function_returns_tool_result(self):
        @tool()
        def f() -> ToolResult:
            return ToolResult(success=True, data={"custom": "data"})

        result = f.run()
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == {"custom": "data"}

    def test_exception_returns_error_result(self):
        @tool()
        def f() -> dict:
            raise ValueError("something went wrong")

        result = f.run()
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "something went wrong" in result.error


# ---------------------------------------------------------------------------
# 7. FunctionTool is callable
# ---------------------------------------------------------------------------


class TestCallable:
    def test_function_tool_is_callable(self):
        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        # Calling the FunctionTool directly invokes the underlying function
        assert add(a=3, b=4) == 7

    def test_callable_with_positional_args(self):
        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# 8. Tool info / parameters / name
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_info_returns_tool_descriptor(self):
        @tool()
        def my_tool(x: str) -> dict:
            """Short desc.

            Long description here.
            """
            return {}

        info = my_tool.info()
        assert isinstance(info, ToolDescriptor)
        assert info.name == "my_tool"
        assert info.short_description == "Short desc."
        assert info.version == "1.0.0"

    def test_parameters_returns_list(self):
        @tool()
        def f(a: str, b: int = 5) -> dict:
            return {}

        params = f.parameters()
        assert isinstance(params, list)
        assert len(params) == 2
        assert all(isinstance(p, ToolParameter) for p in params)

    def test_name_returns_string(self):
        @tool()
        def hello(x: str) -> dict:
            return {}

        assert hello.name() == "hello"


# ---------------------------------------------------------------------------
# 9. ToolRegistry.tool() decorator
# ---------------------------------------------------------------------------


class TestRegistryTool:
    def test_registry_tool_registers(self):
        registry = ToolRegistry()

        @registry.tool()
        def my_tool(x: str) -> dict:
            """A tool."""
            return {}

        assert "my_tool" in registry
        assert registry.get("my_tool") is my_tool

    def test_registry_tool_with_custom_name(self):
        registry = ToolRegistry()

        @registry.tool(name="renamed")
        def my_tool(x: str) -> dict:
            """A tool."""
            return {}

        assert "renamed" in registry
        assert my_tool.name() == "renamed"

    def test_multiple_tools_on_same_registry(self):
        registry = ToolRegistry()

        @registry.tool()
        def tool_a(x: str) -> dict:
            return {}

        @registry.tool()
        def tool_b(y: int) -> dict:
            return {}

        assert len(registry) == 2
        assert "tool_a" in registry
        assert "tool_b" in registry


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_function_with_no_type_hints(self):
        @tool()
        def f(x, y):
            """No hints."""
            return {"x": x, "y": y}

        # All params default to "string"
        for p in f.parameters():
            assert p.type == "string"

    def test_function_with_no_docstring(self):
        @tool()
        def f(x: str) -> dict:
            return {}

        assert f.info().short_description == ""
        assert f.info().long_description == ""

    def test_skips_ctx_parameter(self):
        @tool()
        def f(ctx, x: str) -> dict:
            return {}

        param_names = [p.name for p in f.parameters()]
        assert "ctx" not in param_names
        assert "x" in param_names

    def test_skips_context_parameter(self):
        @tool()
        def f(context, x: str) -> dict:
            return {}

        param_names = [p.name for p in f.parameters()]
        assert "context" not in param_names
        assert "x" in param_names

    def test_skips_self_parameter(self):
        @tool()
        def f(self, x: str) -> dict:
            return {}

        param_names = [p.name for p in f.parameters()]
        assert "self" not in param_names

    def test_validate_params_works_on_function_tool(self):
        @tool()
        def f(x: str, y: int = 5) -> dict:
            return {}

        # Missing required param
        errors = f.validate_params()
        assert len(errors) == 1
        assert "x" in errors[0]

        # All required params present
        errors = f.validate_params(x="hello")
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# 11. Async function support
# ---------------------------------------------------------------------------


class TestAsyncSupport:
    def test_async_function_tool(self):
        @tool()
        async def async_fetch(url: str) -> dict:
            """Fetch a URL."""
            return {"url": url, "status": 200}

        result = async_fetch.run(url="https://example.com")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 12. repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        @tool()
        def my_tool(x: str) -> dict:
            return {}

        assert repr(my_tool) == "FunctionTool('my_tool')"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def greet_tool_name(t: FunctionTool) -> str:
    return t.name()
