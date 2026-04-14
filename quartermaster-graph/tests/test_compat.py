"""Backward-compatibility aliases for the GraphSpec rename.

``AgentGraph`` was the class name before the 0.2.0 rename; ``AgentVersion``
was the name before that. Both resolve to ``GraphSpec`` so downstream code
that still imports the old name keeps working.
"""

from quartermaster_graph import AgentGraph, AgentVersion, GraphSpec
from quartermaster_graph.models import (
    AgentGraph as AgentGraphFromModels,
    AgentVersion as AgentVersionFromModels,
    GraphSpec as GraphSpecFromModels,
)


def test_agentgraph_is_alias_for_graphspec() -> None:
    """Old code that does ``from quartermaster_graph import AgentGraph`` must keep working."""
    assert AgentGraph is GraphSpec
    assert AgentGraphFromModels is GraphSpecFromModels


def test_agentversion_is_alias_for_graphspec() -> None:
    """Even-older code using ``AgentVersion`` must keep working."""
    assert AgentVersion is GraphSpec
    assert AgentVersionFromModels is GraphSpecFromModels


def test_sdk_reexport_matches() -> None:
    """quartermaster-sdk re-exports the same objects."""
    from quartermaster_sdk import AgentGraph as SdkAgentGraph
    from quartermaster_sdk import GraphSpec as SdkGraphSpec

    assert SdkAgentGraph is GraphSpec
    assert SdkGraphSpec is GraphSpec


def test_instance_check_works_with_both_names() -> None:
    """isinstance checks should return True for both the alias and the canonical name."""
    from uuid import uuid4

    spec = GraphSpec(agent_id=uuid4(), start_node_id=uuid4())
    assert isinstance(spec, GraphSpec)
    assert isinstance(spec, AgentGraph)     # alias
    assert isinstance(spec, AgentVersion)   # alias


def test_roundtrip_serialization_preserved() -> None:
    """JSON round-trip still works identically — no schema break."""
    from uuid import uuid4

    from quartermaster_graph import from_json, to_json

    spec = GraphSpec(agent_id=uuid4(), start_node_id=uuid4())
    restored = from_json(to_json(spec))

    assert isinstance(restored, GraphSpec)
    assert restored.agent_id == spec.agent_id
    assert restored.start_node_id == spec.start_node_id
