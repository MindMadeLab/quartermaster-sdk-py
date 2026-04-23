"""Tests for the data serialization tools."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from quartermaster_tools.builtin.data import (
    convert_format,
    data_filter,
    parse_csv,
    parse_json,
    parse_xml,
    parse_yaml,
)
from quartermaster_tools.builtin.data.convert_format import _convert as convert_format_data
from quartermaster_tools.builtin.data.data_filter import _filter_data as filter_data
from quartermaster_tools.builtin.data.parse_csv import _parse_csv as parse_csv_data
from quartermaster_tools.builtin.data.parse_json import _parse_json as parse_json_data
from quartermaster_tools.builtin.data.parse_xml import _parse_xml as parse_xml_data
from quartermaster_tools.builtin.data.parse_yaml import _parse_yaml as parse_yaml_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_temp(content: str, suffix: str = ".txt") -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# parse_csv
# ---------------------------------------------------------------------------


class TestParseCSVTool:
    """Tests for parse_csv function tool."""

    def test_parse_string_with_headers(self) -> None:
        """Parse CSV string with headers returns list of dicts."""
        csv_text = "name,age\nAlice,30\nBob,25"
        result = parse_csv_data(csv_text)
        assert result == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]

    def test_parse_string_no_headers(self) -> None:
        """Parse CSV string without headers returns list of lists."""
        csv_text = "Alice,30\nBob,25"
        result = parse_csv_data(csv_text, has_headers=False)
        assert result == [["Alice", "30"], ["Bob", "25"]]

    def test_parse_custom_delimiter(self) -> None:
        """Parse CSV with tab delimiter."""
        csv_text = "name\tage\nAlice\t30\nBob\t25"
        result = parse_csv_data(csv_text, delimiter="\t")
        assert result == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]

    def test_parse_semicolon_delimiter(self) -> None:
        """Parse CSV with semicolon delimiter."""
        csv_text = "name;age\nAlice;30"
        result = parse_csv_data(csv_text, delimiter=";")
        assert result == [{"name": "Alice", "age": "30"}]

    def test_parse_from_file(self) -> None:
        """Parse CSV from a file path."""
        csv_text = "x,y\n1,2\n3,4"
        path = _write_temp(csv_text, suffix=".csv")
        try:
            result = parse_csv_data(path)
            assert result == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]
        finally:
            os.unlink(path)

    def test_parse_empty_raises(self) -> None:
        """Empty CSV source raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_csv_data("")

    def test_run_returns_tool_result(self) -> None:
        """The run() method returns a ToolResult with rows."""
        csv_text = "a,b\n1,2"
        result = parse_csv.run(source=csv_text)
        assert result.success
        assert result.data["count"] == 1

    def test_run_missing_source(self) -> None:
        """run() without source returns error."""
        result = parse_csv.run()
        assert not result.success

    def test_name_and_version(self) -> None:
        """Tool reports correct name and version."""
        assert parse_csv.name() == "parse_csv"
        assert parse_csv.version() == "1.0.0"

    def test_info_descriptor(self) -> None:
        """info() returns a valid ToolDescriptor."""
        info = parse_csv.info()
        assert info.name == "parse_csv"
        assert len(info.parameters) == 3


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------


class TestParseJSONTool:
    """Tests for parse_json function tool."""

    def test_parse_string_object(self) -> None:
        """Parse JSON object string."""
        result = parse_json_data('{"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}

    def test_parse_string_array(self) -> None:
        """Parse JSON array string."""
        result = parse_json_data("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_from_file(self) -> None:
        """Parse JSON from a file path."""
        data = {"items": [1, 2, 3]}
        path = _write_temp(json.dumps(data), suffix=".json")
        try:
            result = parse_json_data(path)
            assert result == data
        finally:
            os.unlink(path)

    def test_parse_with_jmespath_query(self) -> None:
        """Parse JSON with JMESPath query if available."""
        try:
            import jmespath  # noqa: F401
        except ImportError:
            pytest.skip("jmespath not installed")
        source = '{"people": [{"name": "Alice"}, {"name": "Bob"}]}'
        result = parse_json_data(source, query="people[*].name")
        assert result == ["Alice", "Bob"]

    def test_parse_invalid_json(self) -> None:
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_json_data("{not valid json}")

    def test_run_returns_tool_result(self) -> None:
        """run() returns ToolResult with parsed data."""
        result = parse_json.run(source='{"a": 1}')
        assert result.success
        assert result.data["result"] == {"a": 1}

    def test_run_missing_source(self) -> None:
        """run() without source returns error."""
        result = parse_json.run()
        assert not result.success

    def test_name_and_version(self) -> None:
        """Tool reports correct name and version."""
        assert parse_json.name() == "parse_json"
        assert parse_json.version() == "1.0.0"


# ---------------------------------------------------------------------------
# parse_yaml
# ---------------------------------------------------------------------------


class TestParseYAMLTool:
    """Tests for parse_yaml function tool."""

    def test_parse_string(self) -> None:
        """Parse YAML string."""
        yaml_text = "name: Alice\nage: 30\n"
        result = parse_yaml_data(yaml_text)
        assert result == {"name": "Alice", "age": 30}

    def test_parse_list(self) -> None:
        """Parse YAML list."""
        yaml_text = "- one\n- two\n- three\n"
        result = parse_yaml_data(yaml_text)
        assert result == ["one", "two", "three"]

    def test_parse_from_file(self) -> None:
        """Parse YAML from a file path."""
        yaml_text = "key: value\nnested:\n  a: 1\n"
        path = _write_temp(yaml_text, suffix=".yaml")
        try:
            result = parse_yaml_data(path)
            assert result == {"key": "value", "nested": {"a": 1}}
        finally:
            os.unlink(path)

    def test_parse_nested_structures(self) -> None:
        """Parse complex nested YAML."""
        yaml_text = "servers:\n  - host: a.com\n    port: 80\n  - host: b.com\n    port: 443\n"
        result = parse_yaml_data(yaml_text)
        assert len(result["servers"]) == 2
        assert result["servers"][0]["host"] == "a.com"

    def test_run_returns_tool_result(self) -> None:
        """run() returns ToolResult."""
        result = parse_yaml.run(source="x: 1")
        assert result.success
        assert result.data["result"] == {"x": 1}

    def test_run_missing_source(self) -> None:
        """run() without source returns error."""
        result = parse_yaml.run()
        assert not result.success

    def test_name_and_version(self) -> None:
        """Tool reports correct name and version."""
        assert parse_yaml.name() == "parse_yaml"


# ---------------------------------------------------------------------------
# parse_xml
# ---------------------------------------------------------------------------


class TestParseXMLTool:
    """Tests for parse_xml function tool."""

    def test_parse_simple_xml(self) -> None:
        """Parse simple XML string."""
        xml_text = "<root><name>Alice</name><age>30</age></root>"
        result = parse_xml_data(xml_text)
        assert result["name"] == {"#text": "Alice"}
        assert result["age"] == {"#text": "30"}

    def test_parse_with_attributes(self) -> None:
        """Parse XML with attributes."""
        xml_text = '<item id="1" type="book"><title>Test</title></item>'
        result = parse_xml_data(xml_text)
        assert result["@id"] == "1"
        assert result["@type"] == "book"
        assert result["title"] == {"#text": "Test"}

    def test_parse_with_xpath(self) -> None:
        """Parse XML with XPath query."""
        xml_text = "<root><item>A</item><item>B</item><item>C</item></root>"
        result = parse_xml_data(xml_text, xpath=".//item")
        assert len(result) == 3
        assert result[0] == {"#text": "A"}

    def test_parse_xpath_single_match(self) -> None:
        """XPath with single match returns dict, not list."""
        xml_text = "<root><name>Alice</name><age>30</age></root>"
        result = parse_xml_data(xml_text, xpath=".//name")
        assert result == {"#text": "Alice"}

    def test_parse_from_file(self) -> None:
        """Parse XML from a file."""
        xml_text = "<root><key>value</key></root>"
        path = _write_temp(xml_text, suffix=".xml")
        try:
            result = parse_xml_data(path)
            assert result["key"] == {"#text": "value"}
        finally:
            os.unlink(path)

    def test_parse_invalid_xml(self) -> None:
        """Invalid XML raises ValueError."""
        with pytest.raises(ValueError, match="Invalid XML"):
            parse_xml_data("<unclosed>")

    def test_parse_repeated_children(self) -> None:
        """Repeated child tags produce a list."""
        xml_text = "<root><tag>1</tag><tag>2</tag></root>"
        result = parse_xml_data(xml_text)
        assert isinstance(result["tag"], list)
        assert len(result["tag"]) == 2

    def test_run_returns_tool_result(self) -> None:
        """run() returns ToolResult."""
        result = parse_xml.run(source="<r><a>1</a></r>")
        assert result.success

    def test_run_missing_source(self) -> None:
        """run() without source returns error."""
        result = parse_xml.run()
        assert not result.success


# ---------------------------------------------------------------------------
# convert_format
# ---------------------------------------------------------------------------


class TestConvertFormatTool:
    """Tests for convert_format function tool."""

    def test_json_to_csv(self) -> None:
        """Convert JSON array of objects to CSV."""
        source = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        output = convert_format_data(source, "json", "csv")
        assert "name,age" in output
        assert "Alice,30" in output

    def test_csv_to_json(self) -> None:
        """Convert CSV to JSON."""
        source = "name,age\nAlice,30\nBob,25"
        output = convert_format_data(source, "csv", "json")
        data = json.loads(output)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"

    def test_json_to_yaml(self) -> None:
        """Convert JSON to YAML."""
        source = '{"name": "Alice", "age": 30}'
        output = convert_format_data(source, "json", "yaml")
        assert "name: Alice" in output
        assert "age: 30" in output

    def test_yaml_to_json(self) -> None:
        """Convert YAML to JSON."""
        source = "name: Alice\nage: 30\n"
        output = convert_format_data(source, "yaml", "json")
        data = json.loads(output)
        assert data["name"] == "Alice"

    def test_csv_to_yaml(self) -> None:
        """Convert CSV to YAML."""
        source = "x,y\n1,2"
        output = convert_format_data(source, "csv", "yaml")
        assert "x:" in output

    def test_unsupported_format_raises(self) -> None:
        """Unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            convert_format_data("{}", "xml", "json")

    def test_run_returns_tool_result(self) -> None:
        """run() returns ToolResult with output."""
        result = convert_format.run(source='[{"a":1}]', from_format="json", to_format="csv")
        assert result.success
        assert "output" in result.data

    def test_run_missing_params(self) -> None:
        """run() with missing params returns error."""
        assert not convert_format.run(source="x").success
        assert not convert_format.run(source="x", from_format="json").success


# ---------------------------------------------------------------------------
# data_filter
# ---------------------------------------------------------------------------


class TestDataFilterTool:
    """Tests for data_filter function tool."""

    def setup_method(self) -> None:
        self.sample_data: list[dict[str, Any]] = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "NYC"},
            {"name": "Diana", "age": 28, "city": "Chicago"},
        ]

    def test_filter_expression(self) -> None:
        """Filter rows with expression."""
        result = filter_data(self.sample_data, filter_expression="row['age'] > 28")
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"Alice", "Charlie"}

    def test_filter_string_expression(self) -> None:
        """Filter with string comparison."""
        result = filter_data(self.sample_data, filter_expression="row['city'] == 'NYC'")
        assert len(result) == 2

    def test_sort_by(self) -> None:
        """Sort by a key."""
        result = filter_data(self.sample_data, sort_by="age")
        assert result[0]["name"] == "Bob"
        assert result[-1]["name"] == "Charlie"

    def test_limit(self) -> None:
        """Limit number of results."""
        result = filter_data(self.sample_data, limit=2)
        assert len(result) == 2

    def test_filter_sort_limit_combined(self) -> None:
        """Combine filter, sort, and limit."""
        result = filter_data(
            self.sample_data,
            filter_expression="row['age'] >= 28",
            sort_by="age",
            limit=2,
        )
        assert len(result) == 2
        assert result[0]["name"] == "Diana"
        assert result[1]["name"] == "Alice"

    def test_blocked_expression_rejected(self) -> None:
        """Expressions with blocked names are rejected."""
        with pytest.raises(ValueError, match="blocked"):
            filter_data(self.sample_data, filter_expression="__import__('os')")

    def test_dunder_rejected(self) -> None:
        """Dunder attributes are rejected."""
        with pytest.raises(ValueError, match="Dunder"):
            filter_data(self.sample_data, filter_expression="row.__class__")

    def test_empty_data(self) -> None:
        """Filter on empty list returns empty list."""
        result = filter_data([], filter_expression="row['x'] > 1")
        assert result == []

    def test_no_filters_returns_copy(self) -> None:
        """No filter/sort/limit returns a copy of data."""
        result = filter_data(self.sample_data)
        assert result == self.sample_data
        assert result is not self.sample_data

    def test_run_returns_tool_result(self) -> None:
        """run() returns ToolResult."""
        result = data_filter.run(data=self.sample_data, limit=1)
        assert result.success
        assert result.data["count"] == 1

    def test_run_missing_data(self) -> None:
        """run() without data returns error."""
        result = data_filter.run()
        assert not result.success

    def test_invalid_data_type(self) -> None:
        """Non-list data raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            filter_data("not a list")  # type: ignore[arg-type]

    def test_name_and_info(self) -> None:
        """Tool reports correct metadata."""
        assert data_filter.name() == "data_filter"
        info = data_filter.info()
        assert info.name == "data_filter"
        assert len(info.parameters) == 4
