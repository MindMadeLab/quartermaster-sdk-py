"""Serialization — JSON, YAML, and JSON Schema generation."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import TypeAdapter

from quartermaster_graph.models import AgentGraph


def to_json(version: AgentGraph) -> dict[str, Any]:
    """Serialize an AgentGraph to a JSON-compatible dict."""
    return version.model_dump(mode="json")


def from_json(data: dict[str, Any]) -> AgentGraph:
    """Deserialize an AgentGraph from a JSON dict."""
    return AgentGraph.model_validate(data)


def to_yaml(version: AgentGraph) -> str:
    """Serialize an AgentGraph to a YAML string."""
    data = version.model_dump(mode="json")
    result: str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return result


def from_yaml(yaml_str: str) -> AgentGraph:
    """Deserialize an AgentGraph from a YAML string."""
    data = yaml.safe_load(yaml_str)
    return AgentGraph.model_validate(data)


def json_schema() -> dict[str, Any]:
    """Generate a JSON Schema for the AgentGraph model."""
    adapter = TypeAdapter(AgentGraph)
    return adapter.json_schema()
