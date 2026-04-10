"""Tests for quartermaster_tools.types — ToolParameter, ToolParameterOption, ToolDescriptor, ToolResult."""

from quartermaster_tools import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult


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
