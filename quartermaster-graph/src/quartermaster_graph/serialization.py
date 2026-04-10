"""Serialization — JSON, YAML, and JSON Schema generation."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import TypeAdapter

from quartermaster_graph.models import AgentVersion


def to_json(version: AgentVersion) -> dict[str, Any]:
    """Serialize an AgentVersion to a JSON-compatible dict."""
    return version.model_dump(mode="json")


def from_json(data: dict[str, Any]) -> AgentVersion:
    """Deserialize an AgentVersion from a JSON dict."""
    return AgentVersion.model_validate(data)


def to_yaml(version: AgentVersion) -> str:
    """Serialize an AgentVersion to a YAML string."""
    data = version.model_dump(mode="json")
    result: str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return result


def from_yaml(yaml_str: str) -> AgentVersion:
    """Deserialize an AgentVersion from a YAML string."""
    data = yaml.safe_load(yaml_str)
    return AgentVersion.model_validate(data)


def json_schema() -> dict[str, Any]:
    """Generate a JSON Schema for the AgentVersion model."""
    adapter = TypeAdapter(AgentVersion)
    return adapter.json_schema()
