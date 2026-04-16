"""Graph validation — ensures agent graphs are well-formed DAGs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from uuid import UUID

from quartermaster_graph.enums import NodeType


@dataclass
class ValidationError:
    """A single validation issue found in a graph."""

    code: str
    message: str
    node_id: UUID | None = None
    edge_id: UUID | None = None
    severity: str = "error"  # "error" or "warning"


def validate_graph(version: GraphSpec) -> list[ValidationError]:  # type: ignore[name-defined]  # noqa: F821
    """Validate an agent graph version, returning all issues found.

    Checks:
    - Exactly one Start node
    - At least one End node
    - No orphan nodes (all reachable from start)
    - No cycles (DAG property) — loops with Loop nodes get a warning
    - Decision/If/Switch nodes have matching edge labels
    - All edge source/target IDs reference existing nodes
    - Start node ID is valid
    """
    from quartermaster_graph.models import GraphSpec

    assert isinstance(version, GraphSpec)
    errors: list[ValidationError] = []
    node_map = {n.id: n for n in version.nodes}

    # --- Start node ---
    start_nodes = [n for n in version.nodes if n.type == NodeType.START]
    if len(start_nodes) == 0:
        errors.append(
            ValidationError(
                code="no_start",
                message="Graph must have exactly one Start node",
            )
        )
    elif len(start_nodes) > 1:
        for sn in start_nodes[1:]:
            errors.append(
                ValidationError(
                    code="multiple_starts",
                    message="Graph has multiple Start nodes",
                    node_id=sn.id,
                )
            )

    # --- End node (optional since v0.2.0) ---
    # Pre-0.2.0 a graph without an explicit End node was rejected here.
    # The runner has always been fine with "no End" — it falls back to the
    # last finished node's output in ``FlowResult``.  Making ``.end()``
    # optional means single-node flows (``Graph("x").instruction(...).build()``)
    # don't need a trailing ``.end()`` boilerplate line.

    # --- Start node ID valid ---
    if version.start_node_id not in node_map:
        errors.append(
            ValidationError(
                code="invalid_start_id",
                message=(
                    f"start_node_id {version.start_node_id} does not reference a node in the graph"
                ),
            )
        )
    elif node_map[version.start_node_id].type != NodeType.START:
        errors.append(
            ValidationError(
                code="start_id_not_start_type",
                message="start_node_id does not point to a Start-type node",
                node_id=version.start_node_id,
            )
        )

    # --- Edge references ---
    for edge in version.edges:
        if edge.source_id not in node_map:
            errors.append(
                ValidationError(
                    code="invalid_edge_source",
                    message=f"Edge source {edge.source_id} not found in nodes",
                    edge_id=edge.id,
                )
            )
        if edge.target_id not in node_map:
            errors.append(
                ValidationError(
                    code="invalid_edge_target",
                    message=f"Edge target {edge.target_id} not found in nodes",
                    edge_id=edge.id,
                )
            )

    # --- Reachability (orphan detection) ---
    if start_nodes and version.start_node_id in node_map:
        adj: dict[UUID, list[UUID]] = {n.id: [] for n in version.nodes}
        for edge in version.edges:
            if edge.source_id in adj:
                adj[edge.source_id].append(edge.target_id)

        reachable: set[UUID] = set()
        queue: deque[UUID] = deque([version.start_node_id])
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for succ in adj.get(nid, []):
                if succ not in reachable:
                    queue.append(succ)

        for node in version.nodes:
            if node.id not in reachable and node.type != NodeType.COMMENT:
                errors.append(
                    ValidationError(
                        code="orphan_node",
                        message=(
                            f"Node '{node.name or node.type.value}' is not reachable from Start"
                        ),
                        node_id=node.id,
                    )
                )

    # --- Cycle detection (Kahn's algorithm) ---
    if version.nodes:
        in_degree: dict[UUID, int] = {n.id: 0 for n in version.nodes}
        adj_cycle: dict[UUID, list[UUID]] = {n.id: [] for n in version.nodes}
        for edge in version.edges:
            if edge.source_id in adj_cycle and edge.target_id in in_degree:
                adj_cycle[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        queue_cycle: deque[UUID] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited_count = 0
        while queue_cycle:
            nid = queue_cycle.popleft()
            visited_count += 1
            for succ in adj_cycle[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue_cycle.append(succ)

        if visited_count < len(version.nodes):
            # v0.3.0: End → Start back-edges are implicit (they live in
            # the runner's dispatch logic, not in ``version.edges``),
            # so a graph that uses the new loop semantics passes
            # without effort.  User-written cycles via explicit edges
            # are still flagged, but only as a WARNING because the
            # runner has a ``_is_loop_target`` / ``max_loop_iterations``
            # guard that makes intentional back-edges safe at run time.
            errors.append(
                ValidationError(
                    code="cycle_detected",
                    message=(
                        "Graph contains a cycle via explicit edges. "
                        "Intentional loops are supported at run time "
                        "(``FlowRunner.max_loop_iterations`` caps the "
                        "dispatches), and the default v0.3.0 End-node "
                        "semantics loop back to Start implicitly — so "
                        "you rarely need an explicit back-edge.  "
                        "Consider replacing the cycle with .end()."
                    ),
                    severity="warning",
                )
            )

    # --- Decision/If/Switch edge label checks ---
    outgoing: dict[UUID, list[GraphEdge]] = {}  # type: ignore[name-defined]  # noqa: F821
    for edge in version.edges:
        outgoing.setdefault(edge.source_id, []).append(edge)

    for node in version.nodes:
        out_edges = outgoing.get(node.id, [])
        if node.type in (NodeType.DECISION, NodeType.USER_DECISION):
            if len(out_edges) > 1:
                unlabeled = [e for e in out_edges if not e.label]
                if unlabeled:
                    errors.append(
                        ValidationError(
                            code="decision_unlabeled_edges",
                            message=(
                                f"Decision node '{node.name}' has "
                                f"{len(unlabeled)} unlabeled outgoing edges"
                            ),
                            node_id=node.id,
                        )
                    )
        elif node.type == NodeType.IF:
            labels = {e.label for e in out_edges}
            bool_labels = {
                "true",
                "false",
                "True",
                "False",
                "yes",
                "no",
                "Yes",
                "No",
            }
            if out_edges and not labels.intersection(bool_labels):
                errors.append(
                    ValidationError(
                        code="if_missing_labels",
                        message=f"If node '{node.name}' edges should have true/false labels",
                        node_id=node.id,
                        severity="warning",
                    )
                )
        elif node.type == NodeType.SWITCH and len(out_edges) > 1:
            unlabeled = [e for e in out_edges if not e.label]
            if unlabeled:
                errors.append(
                    ValidationError(
                        code="switch_unlabeled_edges",
                        message=(f"Switch node '{node.name}' has unlabeled outgoing edges"),
                        node_id=node.id,
                    )
                )

    # --- capture_as collisions with reserved node metadata keys ---
    # v0.2.0 lets callers name a node's output via ``capture_as="x"``.
    # A YAML-loaded graph could set ``capture_as: llm_model`` — that's
    # not a correctness bug (captures are a separate namespace from
    # node metadata), but the resulting ``result.captures["llm_model"]``
    # is confusing to debug because it shadows a well-known concept.
    # Warn so the operator sees it in logs.
    _reserved_capture_names = frozenset(
        {
            "llm_model",
            "llm_provider",
            "llm_temperature",
            "llm_system_instruction",
            "llm_max_output_tokens",
            "llm_max_input_tokens",
            "llm_stream",
            "llm_vision",
            "llm_thinking_level",
            "show_output",
            "capture_as",
        }
    )
    for node in version.nodes:
        capture_name = node.metadata.get("capture_as")
        if isinstance(capture_name, str) and capture_name in _reserved_capture_names:
            errors.append(
                ValidationError(
                    code="capture_as_shadows_reserved_key",
                    message=(
                        f"Node '{node.name}' has capture_as={capture_name!r} which "
                        f"collides with a reserved node-metadata key. Captures "
                        f"live in a separate namespace so this still works, but "
                        f"rename to something unambiguous for readability."
                    ),
                    node_id=node.id,
                    severity="warning",
                )
            )

    return errors
