"""Negative regression tests — v0.6.0 removed the ``AgentGraph`` /
``AgentVersion`` aliases for ``GraphSpec``.

These tests guard against accidental reintroduction of the aliases and
document the v0.5 → v0.6 migration: use ``GraphSpec`` directly.
"""

from __future__ import annotations

import pytest

import quartermaster_graph
import quartermaster_graph.models as _models
from quartermaster_graph import GraphSpec


def test_graph_spec_still_importable() -> None:
    """The canonical name must keep working — we did not rename GraphSpec."""
    assert GraphSpec is _models.GraphSpec
    assert GraphSpec.__name__ == "GraphSpec"


def test_agentgraph_alias_removed_from_top_level() -> None:
    with pytest.raises(ImportError):
        from quartermaster_graph import AgentGraph  # noqa: F401


def test_agentversion_alias_removed_from_top_level() -> None:
    with pytest.raises(ImportError):
        from quartermaster_graph import AgentVersion  # noqa: F401


def test_agentgraph_alias_removed_from_models() -> None:
    with pytest.raises(ImportError):
        from quartermaster_graph.models import AgentGraph  # noqa: F401


def test_agentversion_alias_removed_from_models() -> None:
    with pytest.raises(ImportError):
        from quartermaster_graph.models import AgentVersion  # noqa: F401


def test_aliases_not_in_dunder_all() -> None:
    assert "AgentGraph" not in quartermaster_graph.__all__
    assert "AgentVersion" not in quartermaster_graph.__all__


def test_sdk_does_not_re_export_agentgraph() -> None:
    """The SDK layer used to forward AgentGraph; it no longer should."""
    import quartermaster_sdk

    assert "AgentGraph" not in quartermaster_sdk.__all__
    with pytest.raises(ImportError):
        from quartermaster_sdk import AgentGraph  # noqa: F401


def test_roundtrip_serialization_still_works_via_graph_spec() -> None:
    """Sanity: GraphSpec itself still round-trips through JSON — we only
    removed the alias, never the class."""
    from uuid import uuid4

    from quartermaster_graph import from_json, to_json

    spec = GraphSpec(agent_id=uuid4(), start_node_id=uuid4())
    restored = from_json(to_json(spec))

    assert isinstance(restored, GraphSpec)
    assert restored.agent_id == spec.agent_id
    assert restored.start_node_id == spec.start_node_id
