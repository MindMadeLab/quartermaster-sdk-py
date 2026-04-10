"""Graph versioning — create, fork, diff, and bump versions."""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from uuid import UUID, uuid4

from quartermaster_graph.models import (
    Agent,
    AgentVersion,
    EdgeDiff,
    GraphDiff,
    GraphEdge,
    GraphNode,
    NodeDiff,
)

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _parse_semver(v: str) -> tuple[int, int, int]:
    m = _SEMVER.match(v)
    if not m:
        raise ValueError(f"Invalid semver string: {v!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump_major(version: str) -> str:
    """Bump major version: 1.2.3 -> 2.0.0."""
    major, _, _ = _parse_semver(version)
    return f"{major + 1}.0.0"


def bump_minor(version: str) -> str:
    """Bump minor version: 1.2.3 -> 1.3.0."""
    major, minor, _ = _parse_semver(version)
    return f"{major}.{minor + 1}.0"


def bump_patch(version: str) -> str:
    """Bump patch version: 1.2.3 -> 1.2.4."""
    major, minor, patch = _parse_semver(version)
    return f"{major}.{minor}.{patch + 1}"


def create_version(
    agent: Agent,
    version: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    start_node_id: UUID,
    features: str = "",
) -> AgentVersion:
    """Create a new version snapshot of an agent graph."""
    _parse_semver(version)
    return AgentVersion(
        agent_id=agent.id,
        version=version,
        start_node_id=start_node_id,
        nodes=copy.deepcopy(nodes),
        edges=copy.deepcopy(edges),
        features=features,
        created_at=datetime.now(timezone.utc),
    )


def fork(version: AgentVersion, new_agent: Agent) -> AgentVersion:
    """Deep-copy a version to a new agent, assigning fresh IDs."""
    old_to_new: dict[UUID, UUID] = {}
    new_nodes: list[GraphNode] = []
    for node in version.nodes:
        new_id = uuid4()
        old_to_new[node.id] = new_id
        new_node = node.model_copy(deep=True)
        new_node.id = new_id
        new_nodes.append(new_node)

    new_edges: list[GraphEdge] = []
    for edge in version.edges:
        new_edge = edge.model_copy(deep=True)
        new_edge.id = uuid4()
        new_edge.source_id = old_to_new.get(edge.source_id, edge.source_id)
        new_edge.target_id = old_to_new.get(edge.target_id, edge.target_id)
        new_edges.append(new_edge)

    new_start_id = old_to_new.get(version.start_node_id, version.start_node_id)

    return AgentVersion(
        agent_id=new_agent.id,
        version="0.1.0",
        start_node_id=new_start_id,
        nodes=new_nodes,
        edges=new_edges,
        features=version.features,
        forked_from=version.id,
        created_at=datetime.now(timezone.utc),
    )


def diff(v1: AgentVersion, v2: AgentVersion) -> GraphDiff:
    """Compute the difference between two graph versions."""
    v1_nodes = {n.id: n for n in v1.nodes}
    v2_nodes = {n.id: n for n in v2.nodes}
    v1_edges = {e.id: e for e in v1.edges}
    v2_edges = {e.id: e for e in v2.edges}

    node_diffs: list[NodeDiff] = []
    for nid, node in v1_nodes.items():
        if nid not in v2_nodes:
            node_diffs.append(NodeDiff(node_id=nid, change="removed", old=node))
        elif node != v2_nodes[nid]:
            node_diffs.append(
                NodeDiff(node_id=nid, change="modified", old=node, new=v2_nodes[nid])
            )
    for nid, node in v2_nodes.items():
        if nid not in v1_nodes:
            node_diffs.append(NodeDiff(node_id=nid, change="added", new=node))

    edge_diffs: list[EdgeDiff] = []
    for eid, edge in v1_edges.items():
        if eid not in v2_edges:
            edge_diffs.append(EdgeDiff(edge_id=eid, change="removed", old=edge))
        elif edge != v2_edges[eid]:
            edge_diffs.append(
                EdgeDiff(edge_id=eid, change="modified", old=edge, new=v2_edges[eid])
            )
    for eid, edge in v2_edges.items():
        if eid not in v1_edges:
            edge_diffs.append(EdgeDiff(edge_id=eid, change="added", new=edge))

    return GraphDiff(
        version_from=v1.version,
        version_to=v2.version,
        node_diffs=node_diffs,
        edge_diffs=edge_diffs,
    )
