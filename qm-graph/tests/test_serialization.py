"""Tests for JSON, YAML, and JSON Schema serialization."""

import json

import yaml

from qm_graph.serialization import from_json, from_yaml, json_schema, to_json, to_yaml


class TestJsonSerialization:
    def test_roundtrip(self, simple_graph):
        data = to_json(simple_graph)
        restored = from_json(data)
        assert restored.version == simple_graph.version
        assert len(restored.nodes) == len(simple_graph.nodes)
        assert len(restored.edges) == len(simple_graph.edges)

    def test_json_is_dict(self, simple_graph):
        data = to_json(simple_graph)
        assert isinstance(data, dict)
        assert "nodes" in data
        assert "edges" in data

    def test_json_serializable(self, simple_graph):
        data = to_json(simple_graph)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        restored = from_json(parsed)
        assert restored.version == simple_graph.version

    def test_decision_graph_roundtrip(self, decision_graph):
        data = to_json(decision_graph)
        restored = from_json(data)
        assert len(restored.nodes) == 6
        assert len(restored.edges) == 5


class TestYamlSerialization:
    def test_roundtrip(self, simple_graph):
        yaml_str = to_yaml(simple_graph)
        restored = from_yaml(yaml_str)
        assert restored.version == simple_graph.version
        assert len(restored.nodes) == len(simple_graph.nodes)

    def test_yaml_is_string(self, simple_graph):
        yaml_str = to_yaml(simple_graph)
        assert isinstance(yaml_str, str)
        assert "version:" in yaml_str

    def test_yaml_parseable(self, simple_graph):
        yaml_str = to_yaml(simple_graph)
        data = yaml.safe_load(yaml_str)
        assert isinstance(data, dict)
        assert data["version"] == "0.1.0"

    def test_decision_graph_yaml_roundtrip(self, decision_graph):
        yaml_str = to_yaml(decision_graph)
        restored = from_yaml(yaml_str)
        assert len(restored.nodes) == 6

    def test_json_yaml_equivalence(self, simple_graph):
        """JSON and YAML should produce equivalent results."""
        json_data = to_json(simple_graph)
        yaml_str = to_yaml(simple_graph)
        yaml_data = yaml.safe_load(yaml_str)
        assert json_data == yaml_data


class TestJsonSchema:
    def test_generates_schema(self):
        schema = json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_schema_has_required_fields(self):
        schema = json_schema()
        props = schema.get("properties", {})
        assert "nodes" in props
        assert "edges" in props
        assert "version" in props
        assert "start_node_id" in props
