"""Serialization — JSON, YAML, and JSON Schema generation."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import TypeAdapter

from quartermaster_graph.models import GraphSpec


def to_json(version: GraphSpec) -> dict[str, Any]:
    """Serialize an GraphSpec to a JSON-compatible dict."""
    return version.model_dump(mode="json")


def from_json(data: dict[str, Any]) -> GraphSpec:
    """Deserialize an GraphSpec from a JSON dict."""
    return GraphSpec.model_validate(data)


def to_yaml(version: GraphSpec) -> str:
    """Serialize an GraphSpec to a YAML string."""
    data = version.model_dump(mode="json")
    result: str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return result


def from_yaml(yaml_str: str) -> GraphSpec:
    """Deserialize an GraphSpec from a YAML string."""
    data = yaml.safe_load(yaml_str)
    return GraphSpec.model_validate(data)


def json_schema() -> dict[str, Any]:
    """Generate a JSON Schema for the GraphSpec model."""
    adapter = TypeAdapter(GraphSpec)
    return adapter.json_schema()
