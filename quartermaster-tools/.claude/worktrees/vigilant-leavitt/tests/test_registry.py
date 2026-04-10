"""Tests for quartermaster_tools.registry — ToolRegistry, register_tool, JSON Schema export."""

from typing import Any

import pytest

from quartermaster_tools import (
    AbstractTool,
    ToolDescriptor,
    ToolParameter,
    ToolParameterOption,
    ToolRegistry,
    ToolResult,
    register_tool,
    get_default_registry,
)
from quartermaster_tools.registry import _tool_to_json_schema, _param_to_json_schema


# --- Test tools ---


class SearchTool(AbstractTool):
    def name(self) -> str:
        return "search"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="query", description="Search query", type="string", required=True),
            ToolParameter(name="limit", description="Max results", type="integer", default=10),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Search the web",
            long_description="Full text web search",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"results": []})


class SearchToolV2(AbstractTool):
    def name(self) -> str:
        return "search"

    def version(self) -> str:
        return "2.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="query", description="Search query", type="string", required=True),
            ToolParameter(
                name="engine",
                description="Search engine",
                type="string",
                options=[
                    ToolParameterOption(label="Google", value="google"),
                    ToolParameterOption(label="Bing", value="bing"),
                ],
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Search the web v2",
            long_description="Enhanced web search",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"results": [], "engine": kwargs.get("engine")})


class GreetTool(AbstractTool):
    def name(self) -> str:
        return "greet"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="name", description="Name", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Greet someone",
            long_description="Greet",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"greeting": f"Hello {kwargs['name']}"})


# --- Registry tests ---


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True  # Skip plugin loading for tests
        reg.register(SearchTool())
        tool = reg.get("search")
        assert tool.name() == "search"
        assert tool.version() == "1.0.0"

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(SearchTool())

    def test_get_nonexistent_raises(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        with pytest.raises(KeyError, match="No tool registered"):
            reg.get("nonexistent")

    def test_version_aware_lookup(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(SearchToolV2())

        v1 = reg.get("search", "1.0.0")
        assert v1.version() == "1.0.0"

        v2 = reg.get("search", "2.0.0")
        assert v2.version() == "2.0.0"

    def test_get_latest_version(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(SearchToolV2())
        latest = reg.get("search")
        assert latest.version() == "2.0.0"

    def test_get_specific_version_not_found(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        with pytest.raises(KeyError, match="version '9.9.9' not found"):
            reg.get("search", "9.9.9")

    def test_list_tools(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(GreetTool())
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"search", "greet"}

    def test_list_names(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(GreetTool())
        assert set(reg.list_names()) == {"search", "greet"}

    def test_unregister_all_versions(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(SearchToolV2())
        reg.unregister("search")
        assert "search" not in reg

    def test_unregister_specific_version(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(SearchToolV2())
        reg.unregister("search", "1.0.0")
        tool = reg.get("search")
        assert tool.version() == "2.0.0"
        with pytest.raises(KeyError):
            reg.get("search", "1.0.0")

    def test_unregister_last_version_removes_name(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.unregister("search", "1.0.0")
        assert "search" not in reg

    def test_unregister_nonexistent_raises(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_unregister_nonexistent_version_raises(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        with pytest.raises(KeyError):
            reg.unregister("search", "9.9.9")

    def test_clear(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(GreetTool())
        reg.clear()
        assert len(reg) == 0

    def test_len(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        assert len(reg) == 0
        reg.register(SearchTool())
        assert len(reg) == 1
        reg.register(SearchToolV2())
        assert len(reg) == 2
        reg.register(GreetTool())
        assert len(reg) == 3

    def test_contains(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        assert "search" in reg
        assert "nonexistent" not in reg

    def test_multiple_tools_multiple_versions(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(SearchToolV2())
        reg.register(GreetTool())
        assert len(reg) == 3
        tools = reg.list_tools()
        assert len(tools) == 3


# --- Decorator tests ---


class TestRegisterToolDecorator:
    def test_register_tool_decorator(self):
        # Use a fresh default registry
        import quartermaster_tools.registry as reg_module

        old = reg_module._default_registry
        reg_module._default_registry = ToolRegistry()
        reg_module._default_registry._plugins_loaded = True

        try:

            @register_tool
            class MyTool(AbstractTool):
                def name(self):
                    return "my_tool"

                def version(self):
                    return "1.0.0"

                def parameters(self):
                    return []

                def info(self):
                    return ToolDescriptor(
                        name="my_tool",
                        short_description="My",
                        long_description="My",
                        version="1.0.0",
                    )

                def run(self, **kwargs):
                    return ToolResult(success=True)

            registry = get_default_registry()
            assert "my_tool" in registry
            tool = registry.get("my_tool")
            assert tool.name() == "my_tool"
        finally:
            reg_module._default_registry = old


# --- JSON Schema tests ---


class TestJsonSchema:
    def test_param_to_json_schema_string(self):
        param = ToolParameter(name="q", description="Query", type="string", required=True)
        schema = _param_to_json_schema(param)
        assert schema["type"] == "string"
        assert schema["description"] == "Query"

    def test_param_to_json_schema_integer(self):
        param = ToolParameter(name="n", description="Count", type="integer", default=5)
        schema = _param_to_json_schema(param)
        assert schema["type"] == "integer"
        assert schema["default"] == 5

    def test_param_to_json_schema_with_options(self):
        param = ToolParameter(
            name="engine",
            description="Engine",
            type="string",
            options=[
                ToolParameterOption(label="Google", value="google"),
                ToolParameterOption(label="Bing", value="bing"),
            ],
        )
        schema = _param_to_json_schema(param)
        assert schema["enum"] == ["google", "bing"]

    def test_param_type_mapping(self):
        mappings = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        for input_type, expected in mappings.items():
            param = ToolParameter(name="x", description="X", type=input_type)
            schema = _param_to_json_schema(param)
            assert schema["type"] == expected, f"{input_type} should map to {expected}"

    def test_tool_to_json_schema(self):
        tool = SearchTool()
        schema = _tool_to_json_schema(tool)
        assert schema["name"] == "search"
        assert schema["description"] == "Search the web"
        assert "properties" in schema["parameters"]
        assert "query" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["query"]

    def test_tool_to_json_schema_no_required(self):
        class OptionalTool(AbstractTool):
            def name(self):
                return "opt"

            def version(self):
                return "1.0.0"

            def parameters(self):
                return [
                    ToolParameter(name="x", description="X", type="string"),
                ]

            def info(self):
                return ToolDescriptor(
                    name="opt",
                    short_description="Optional params",
                    long_description="Opt",
                    version="1.0.0",
                )

            def run(self, **kwargs):
                return ToolResult(success=True)

        schema = _tool_to_json_schema(OptionalTool())
        assert "required" not in schema["parameters"]

    def test_registry_to_json_schema(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        reg.register(GreetTool())
        schemas = reg.to_json_schema()
        assert len(schemas) == 2

    def test_registry_to_openai_tools(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        tools = reg.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"

    def test_registry_to_anthropic_tools(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        tools = reg.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert "input_schema" in tools[0]
        assert tools[0]["input_schema"]["type"] == "object"

    def test_registry_to_mcp_tools(self):
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchTool())
        tools = reg.to_mcp_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert "inputSchema" in tools[0]

    def test_all_formats_consistent_structure(self):
        """All export formats should produce the same underlying schema."""
        reg = ToolRegistry()
        reg._plugins_loaded = True
        reg.register(SearchToolV2())

        openai = reg.to_openai_tools()[0]["function"]
        anthropic = reg.to_anthropic_tools()[0]
        mcp = reg.to_mcp_tools()[0]
        json_schema = reg.to_json_schema()[0]

        # All should have same name
        assert openai["name"] == anthropic["name"] == mcp["name"] == json_schema["name"]

        # All should have same properties
        openai_props = openai["parameters"]["properties"]
        anthropic_props = anthropic["input_schema"]["properties"]
        mcp_props = mcp["inputSchema"]["properties"]
        json_props = json_schema["parameters"]["properties"]

        assert set(openai_props.keys()) == set(anthropic_props.keys())
        assert set(mcp_props.keys()) == set(json_props.keys())
        assert set(openai_props.keys()) == set(json_props.keys())
