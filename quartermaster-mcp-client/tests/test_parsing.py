"""Unit tests for parsing functions: SSE, JSON Schema types, tool parameters."""

from __future__ import annotations

import pytest

from quartermaster_mcp_client.client import (
    parse_json_schema_type,
    parse_sse_response,
    parse_tool_parameters,
)
from quartermaster_mcp_client.errors import McpProtocolError


# === parse_sse_response ===


class TestParseSSEResponse:
    def test_plain_json(self) -> None:
        result = parse_sse_response('{"jsonrpc": "2.0", "id": 1, "result": {}}')
        assert result == {"jsonrpc": "2.0", "id": 1, "result": {}}

    def test_data_prefix(self) -> None:
        result = parse_sse_response('data: {"key": "value"}\n\n')
        assert result == {"key": "value"}

    def test_single_sse_event(self) -> None:
        """parse_sse_response handles a single SSE event (not multi-event)."""
        text = 'data: {"result": "ok"}\n\n'
        result = parse_sse_response(text)
        assert result == {"result": "ok"}

    def test_empty_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="Empty SSE response"):
            parse_sse_response("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="Empty SSE response"):
            parse_sse_response("   \n  \n  ")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="Invalid JSON"):
            parse_sse_response("data: {not json}")

    def test_data_prefix_with_extra_spaces(self) -> None:
        result = parse_sse_response('data:   {"key": "value"}  \n\n')
        assert result == {"key": "value"}


# === parse_json_schema_type ===


class TestParseJsonSchemaType:
    def test_simple_string(self) -> None:
        assert parse_json_schema_type({"type": "string"}) == "string"

    def test_simple_number(self) -> None:
        assert parse_json_schema_type({"type": "number"}) == "number"

    def test_simple_integer(self) -> None:
        assert parse_json_schema_type({"type": "integer"}) == "integer"

    def test_simple_boolean(self) -> None:
        assert parse_json_schema_type({"type": "boolean"}) == "boolean"

    def test_simple_array(self) -> None:
        assert parse_json_schema_type({"type": "array"}) == "array"

    def test_simple_object(self) -> None:
        assert parse_json_schema_type({"type": "object"}) == "object"

    def test_type_array_nullable_string(self) -> None:
        assert parse_json_schema_type({"type": ["string", "null"]}) == "string"

    def test_type_array_nullable_number(self) -> None:
        assert parse_json_schema_type({"type": ["null", "number"]}) == "number"

    def test_type_array_all_null(self) -> None:
        assert parse_json_schema_type({"type": ["null"]}) == "null"

    def test_no_type_field(self) -> None:
        assert parse_json_schema_type({"description": "no type"}) == "object"

    def test_empty_schema(self) -> None:
        assert parse_json_schema_type({}) == "object"

    def test_non_dict(self) -> None:
        assert parse_json_schema_type("string") == "object"  # type: ignore[arg-type]

    def test_any_of(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert parse_json_schema_type(schema) == "string"

    def test_one_of(self) -> None:
        schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
        assert parse_json_schema_type(schema) == "integer"

    def test_all_of(self) -> None:
        schema = {"allOf": [{"type": "object", "properties": {}}]}
        assert parse_json_schema_type(schema) == "object"

    def test_any_of_all_null(self) -> None:
        schema = {"anyOf": [{"type": "null"}]}
        assert parse_json_schema_type(schema) == "object"


# === parse_tool_parameters ===


class TestParseToolParameters:
    def test_basic_parameters(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        assert len(params) == 6

        names = {p.name for p in params}
        assert names == {"query", "limit", "format", "verbose", "tags", "options"}

    def test_required_field(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        query = next(p for p in params if p.name == "query")
        limit = next(p for p in params if p.name == "limit")

        assert query.required is True
        assert limit.required is False

    def test_types(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        param_map = {p.name: p for p in params}

        assert param_map["query"].type == "string"
        assert param_map["limit"].type == "integer"
        assert param_map["format"].type == "string"
        assert param_map["verbose"].type == "boolean"
        assert param_map["tags"].type == "array"
        assert param_map["options"].type == "object"

    def test_enum_values(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        fmt = next(p for p in params if p.name == "format")

        assert fmt.enum == ["json", "csv", "xml"]
        assert len(fmt.options) == 3
        assert fmt.options[0].label == "json"
        assert fmt.options[0].value == "json"

    def test_default_values(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        param_map = {p.name: p for p in params}

        assert param_map["limit"].default == 10
        assert param_map["verbose"].default is False
        assert param_map["query"].default is None

    def test_constraints(self, sample_input_schema: dict) -> None:
        params = parse_tool_parameters(sample_input_schema)
        param_map = {p.name: p for p in params}

        assert param_map["query"].min_length == 1
        assert param_map["query"].max_length == 500
        assert param_map["limit"].min_value == 1
        assert param_map["limit"].max_value == 100

    def test_empty_schema(self) -> None:
        params = parse_tool_parameters({})
        assert params == []

    def test_no_properties(self) -> None:
        params = parse_tool_parameters({"type": "object"})
        assert params == []

    def test_non_dict_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="input_schema must be a dict"):
            parse_tool_parameters("not a dict")  # type: ignore[arg-type]

    def test_skips_non_dict_property(self) -> None:
        schema = {
            "properties": {
                "good": {"type": "string", "description": "ok"},
                "bad": "not a dict",
            }
        }
        params = parse_tool_parameters(schema)
        assert len(params) == 1
        assert params[0].name == "good"

    def test_description_defaults_empty(self) -> None:
        schema = {
            "properties": {
                "x": {"type": "integer"},
            }
        }
        params = parse_tool_parameters(schema)
        assert params[0].description == ""

    def test_pattern_constraint(self) -> None:
        schema = {
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Email address",
                    "pattern": r"^[\w.]+@[\w.]+$",
                }
            }
        }
        params = parse_tool_parameters(schema)
        assert params[0].pattern == r"^[\w.]+@[\w.]+$"
