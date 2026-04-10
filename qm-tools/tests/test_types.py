"""Tests for qm_tools.types — ToolParameter, ToolParameterOption, ToolDescriptor, ToolResult."""

from qm_tools import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult


class TestToolParameterOption:
    def test_creation(self):
        opt = ToolParameterOption(label="Yes", value="yes")
        assert opt.label == "Yes"
        assert opt.value == "yes"

    def test_equality(self):
        a = ToolParameterOption(label="X", value="x")
        b = ToolParameterOption(label="X", value="x")
        assert a == b


class TestToolParameter:
    def test_required_param(self):
        p = ToolParameter(name="query", description="Search query", type="string", required=True)
        assert p.name == "query"
        assert p.type == "string"
        assert p.required is True
        assert p.default is None
        assert p.options == []
        assert p.validation is None

    def test_optional_param_with_default(self):
        p = ToolParameter(name="limit", description="Max results", type="integer", default=10)
        assert p.required is False
        assert p.default == 10

    def test_param_with_options(self):
        opts = [
            ToolParameterOption(label="Add", value="add"),
            ToolParameterOption(label="Sub", value="sub"),
        ]
        p = ToolParameter(name="op", description="Operation", type="string", options=opts)
        assert len(p.options) == 2
        assert p.options[0].value == "add"

    def test_param_with_validation(self):
        def positive(v):
            if v <= 0:
                raise ValueError("Must be positive")
            return v

        p = ToolParameter(name="count", description="Count", type="integer", validation=positive)
        assert p.validation is not None
        assert p.validation(5) == 5


class TestToolDescriptor:
    def test_creation(self):
        d = ToolDescriptor(
            name="search",
            short_description="Search the web",
            long_description="Full web search tool",
            version="1.0.0",
        )
        assert d.name == "search"
        assert d.version == "1.0.0"
        assert d.parameters == []
        assert d.is_local is False

    def test_with_parameters(self):
        params = [ToolParameter(name="q", description="Query", type="string", required=True)]
        d = ToolDescriptor(
            name="search",
            short_description="Search",
            long_description="Search tool",
            version="1.0.0",
            parameters=params,
        )
        assert len(d.parameters) == 1


class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, data={"answer": 42})
        assert r.success is True
        assert r.data == {"answer": 42}
        assert r.error == ""
        assert bool(r) is True

    def test_error(self):
        r = ToolResult(success=False, error="Something went wrong")
        assert r.success is False
        assert r.error == "Something went wrong"
        assert bool(r) is False

    def test_with_metadata(self):
        r = ToolResult(success=True, metadata={"latency_ms": 150})
        assert r.metadata["latency_ms"] == 150

    def test_default_values(self):
        r = ToolResult(success=True)
        assert r.data == {}
        assert r.error == ""
        assert r.metadata == {}


class TestToolParameterJsonSchema:
    def test_basic_schema(self):
        p = ToolParameter(name="query", description="Search query", type="string")
        schema = p.to_json_schema()
        assert schema == {"type": "string", "description": "Search query"}

    def test_schema_with_default(self):
        p = ToolParameter(name="limit", description="Max results", type="integer", default=10)
        schema = p.to_json_schema()
        assert schema["default"] == 10

    def test_schema_with_options(self):
        opts = [
            ToolParameterOption(label="Add", value="add"),
            ToolParameterOption(label="Sub", value="sub"),
        ]
        p = ToolParameter(name="op", description="Operation", type="string", options=opts)
        schema = p.to_json_schema()
        assert schema["enum"] == ["add", "sub"]


class TestToolDescriptorBridge:
    def _make_descriptor(self):
        params = [
            ToolParameter(name="query", description="Search query", type="string", required=True),
            ToolParameter(name="limit", description="Max results", type="integer", default=10),
        ]
        return ToolDescriptor(
            name="search",
            short_description="Search the web",
            long_description="Full web search tool",
            version="1.0.0",
            parameters=params,
        )

    def test_to_input_schema(self):
        d = self._make_descriptor()
        schema = d.to_input_schema()
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["default"] == 10

    def test_to_input_schema_no_required(self):
        d = ToolDescriptor(
            name="noop",
            short_description="No-op",
            long_description="Does nothing",
            version="0.1.0",
        )
        schema = d.to_input_schema()
        assert "required" not in schema

    def test_to_openai_tools(self):
        d = self._make_descriptor()
        result = d.to_openai_tools()
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search the web"
        assert result["function"]["parameters"]["type"] == "object"
        assert "query" in result["function"]["parameters"]["properties"]

    def test_to_anthropic_tools(self):
        d = self._make_descriptor()
        result = d.to_anthropic_tools()
        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert result["input_schema"]["type"] == "object"
        assert "query" in result["input_schema"]["properties"]

    def test_to_tool_definition_import_error(self):
        """to_tool_definition raises ImportError when qm-providers is not installed."""
        import sys
        import unittest.mock

        d = self._make_descriptor()
        # Temporarily block qm_providers import
        with unittest.mock.patch.dict(sys.modules, {"qm_providers": None, "qm_providers.types": None}):
            import pytest

            with pytest.raises(ImportError, match="qm-providers is required"):
                d.to_tool_definition()
