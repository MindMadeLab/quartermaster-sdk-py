"""Graph traversal utilities."""

from __future__ import annotations

from collections import deque
from uuid import UUID

from quartermaster_graph.enums import NodeType
from quartermaster_graph.models import GraphSpec, GraphNode


def _build_adj(version: GraphSpec) -> tuple[dict[UUID, list[UUID]], dict[UUID, list[UUID]]]:
    """Build forward and reverse adjacency lists."""
    forward: dict[UUID, list[UUID]] = {n.id: [] for n in version.nodes}
    reverse: dict[UUID, list[UUID]] = {n.id: [] for n in version.nodes}
    for edge in version.edges:
        if edge.source_id in forward:
            forward[edge.source_id].append(edge.target_id)
        if edge.target_id in reverse:
            reverse[edge.target_id].append(edge.source_id)
    return forward, reverse


def _node_map(version: GraphSpec) -> dict[UUID, GraphNode]:
    return {n.id: n for n in version.nodes}


def get_start_node(version: GraphSpec) -> GraphNode:
    """Get the start node of the graph. Raises ValueError if not found."""
    nmap = _node_map(version)
    if version.start_node_id in nmap:
        return nmap[version.start_node_id]
    for node in version.nodes:
        if node.type == NodeType.START:
            return node
    raise ValueError("No start node found in graph")


def get_successors(version: GraphSpec, node_id: UUID) -> list[GraphNode]:
    """Get all direct successor nodes of a given node."""
    forward, _ = _build_adj(version)
    nmap = _node_map(version)
    return [nmap[nid] for nid in forward.get(node_id, []) if nid in nmap]


def get_predecessors(version: GraphSpec, node_id: UUID) -> list[GraphNode]:
    """Get all direct predecessor nodes of a given node."""
    _, reverse = _build_adj(version)
    nmap = _node_map(version)
    return [nmap[nid] for nid in reverse.get(node_id, []) if nid in nmap]


def get_path(version: GraphSpec, start_id: UUID, end_id: UUID) -> list[GraphNode]:
    """Find the shortest path between two nodes using BFS.

    Returns a list of nodes from start to end (inclusive), or empty list if no path.
    """
    forward, _ = _build_adj(version)
    nmap = _node_map(version)

    if start_id not in nmap or end_id not in nmap:
        return []

    visited: set[UUID] = set()
    parent: dict[UUID, UUID | None] = {start_id: None}
    queue: deque[UUID] = deque([start_id])

    while queue:
        current = queue.popleft()
        if current == end_id:
            path: list[GraphNode] = []
            nid: UUID | None = end_id
            while nid is not None:
                path.append(nmap[nid])
                nid = parent.get(nid)
            path.reverse()
            return path
        if current in visited:
            continue
        visited.add(current)
        for succ in forward.get(current, []):
            if succ not in visited and succ not in parent:
                parent[succ] = current
                queue.append(succ)

    return []


def topological_sort(version: GraphSpec) -> list[GraphNode]:
    """Return nodes in topological order (Kahn's algorithm).

    Raises ValueError if the graph has a cycle.
    """
    nmap = _node_map(version)
    in_degree: dict[UUID, int] = {n.id: 0 for n in version.nodes}
    forward: dict[UUID, list[UUID]] = {n.id: [] for n in version.nodes}

    for edge in version.edges:
        if edge.source_id in forward and edge.target_id in in_degree:
            forward[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

    queue: deque[UUID] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result: list[GraphNode] = []

    while queue:
        nid = queue.popleft()
        result.append(nmap[nid])
        for succ in forward[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(result) < len(version.nodes):
        raise ValueError("Graph has a cycle — cannot topologically sort")

    return result


def find_merge_points(version: GraphSpec) -> list[GraphNode]:
    """Find nodes where branches converge (>1 incoming edge)."""
    _, reverse = _build_adj(version)
    nmap = _node_map(version)
    return [nmap[nid] for nid, preds in reverse.items() if len(preds) > 1]


def find_decision_points(version: GraphSpec) -> list[GraphNode]:
    """Find nodes where branches diverge (>1 outgoing edge)."""
    forward, _ = _build_adj(version)
    nmap = _node_map(version)
    return [nmap[nid] for nid, succs in forward.items() if len(succs) > 1]
