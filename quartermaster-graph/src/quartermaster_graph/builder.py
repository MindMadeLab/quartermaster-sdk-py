"""Fluent graph builder API for programmatic graph construction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from quartermaster_graph.enums import (
    NodeType,
    TraverseIn,
    TraverseOut,
)
from quartermaster_graph.models import Agent, AgentVersion, GraphEdge, GraphNode, NodePosition
from quartermaster_graph.validation import validate_graph


def _inline_subgraph(
    sub_graph: AgentVersion,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    connect_from_id: UUID | None,
    advance_position: Callable[[], NodePosition],
) -> UUID | None:
    """Inline a sub-graph's nodes/edges, skipping START and END nodes.

    Returns the last non-END node ID of the inlined sub-graph (the new
    "current node" for further chaining), or the original *connect_from_id*
    if the sub-graph was effectively empty.
    """
    # Build old-id -> new-id mapping; skip START and END nodes
    id_map: dict[UUID, UUID] = {}
    kept_nodes: list[GraphNode] = []
    for n in sub_graph.nodes:
        if n.type in (NodeType.START, NodeType.END):
            continue
        new_id = uuid4()
        id_map[n.id] = new_id
        copied = n.model_copy(update={"id": new_id, "position": advance_position()})
        kept_nodes.append(copied)

    if not kept_nodes:
        return connect_from_id

    nodes.extend(kept_nodes)

    # Determine the first node after START
    start_id = sub_graph.start_node_id
    first_after_start: UUID | None = None
    for e in sub_graph.edges:
        if e.source_id == start_id and e.target_id in id_map:
            first_after_start = id_map[e.target_id]
            break

    # If no edge from START was found, fall back to the first kept node
    if first_after_start is None and kept_nodes:
        first_after_start = kept_nodes[0].id

    # Connect caller's current node to first inlined node
    if connect_from_id is not None and first_after_start is not None:
        edges.append(GraphEdge(source_id=connect_from_id, target_id=first_after_start))

    # Copy edges, remapping IDs; skip edges involving START/END
    end_ids = {n.id for n in sub_graph.nodes if n.type == NodeType.END}
    for e in sub_graph.edges:
        src = id_map.get(e.source_id)
        tgt = id_map.get(e.target_id)
        if src is not None and tgt is not None:
            edges.append(
                GraphEdge(source_id=src, target_id=tgt, label=e.label, is_main=e.is_main)
            )

    # Find last node(s) -- those that had edges going to END, or if none, the last kept node
    nodes_to_end: list[UUID] = []
    for e in sub_graph.edges:
        if e.target_id in end_ids and e.source_id in id_map:
            nodes_to_end.append(id_map[e.source_id])
    if not nodes_to_end:
        nodes_to_end = [kept_nodes[-1].id]

    return nodes_to_end[-1]


class _BranchBuilder:
    """Builder for a specific branch (e.g., after a decision)."""

    def __init__(self, parent: GraphBuilder, label: str) -> None:
        self._parent = parent
        self._label = label
        self._last_node_id: UUID | None = None

    def _add_node(self, node: GraphNode) -> _BranchBuilder:
        self._parent._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._parent._edges.append(edge)
        self._last_node_id = node.id
        return self

    def instruction(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.7,
        system_instruction: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an instruction node to this branch."""
        meta = {
            "system_instruction": system_instruction,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            **kwargs,
        }
        node = GraphNode(type=NodeType.INSTRUCTION, name=name, metadata=meta)
        return self._add_node(node)

    def user(self, name: str = "User Input") -> _BranchBuilder:
        """Add a user input node to this branch."""
        node = GraphNode(type=NodeType.USER, name=name)
        return self._add_node(node)

    def static(self, name: str, content: str = "") -> _BranchBuilder:
        """Add a static content node."""
        node = GraphNode(type=NodeType.STATIC, name=name, metadata={"content": content})
        return self._add_node(node)

    def code(self, name: str, code: str = "", language: str = "python") -> _BranchBuilder:
        """Add a code execution node."""
        node = GraphNode(
            type=NodeType.CODE,
            name=name,
            metadata={"language": language, "code": code},
        )
        return self._add_node(node)

    def node(
        self,
        node_type: NodeType,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a generic node."""
        node = GraphNode(type=node_type, name=name, metadata=metadata or {}, **kwargs)
        return self._add_node(node)

    def use(self, sub_graph: AgentVersion) -> _BranchBuilder:
        """Inline a sub-graph into this branch.

        Copies all nodes except START/END from the sub-graph, remaps their
        IDs, and connects them into the current branch chain.
        """
        new_last = _inline_subgraph(
            sub_graph,
            self._parent._nodes,
            self._parent._edges,
            self._last_node_id,
            self._parent._advance_position,
        )
        if new_last is not None:
            self._last_node_id = new_last
        return self

    def end(self) -> GraphBuilder:
        """End this branch and return to the parent builder.

        Records the branch endpoint so that a subsequent call on the parent
        (e.g. ``.merge()``, ``.instruction()``, etc.) can auto-merge the
        branches.
        """
        if self._last_node_id is not None:
            self._parent._branch_endpoints.append(self._last_node_id)
        return self._parent

    def merge_to(self, merge_node_id: UUID) -> GraphBuilder:
        """Connect this branch to an existing merge node, then return to parent."""
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=merge_node_id)
            self._parent._edges.append(edge)
        return self._parent


class GraphBuilder:
    """Fluent builder for creating agent graphs programmatically.

    Example::

        graph = (
            GraphBuilder("My Agent")
            .start()
            .instruction("Analyze input", model="gpt-4o")
            .decision("Is it positive?", options=["Yes", "No"])
            .on("Yes").instruction("Positive response").end()
            .on("No").instruction("Negative response").end()
            .build()
        )
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._start_node_id: UUID | None = None
        self._last_node_id: UUID | None = None
        self._decision_node_id: UUID | None = None
        self._pending_branches: dict[str, UUID] = {}
        self._branch_endpoints: list[UUID] = []
        self._position_x = 0
        self._position_y = 0

    def _advance_position(self) -> NodePosition:
        pos = NodePosition(x=self._position_x, y=self._position_y)
        self._position_y += 120
        return pos

    def _auto_merge_if_needed(self) -> None:
        """If there are pending branch endpoints, auto-insert a merge node
        and connect all endpoints to it, making it the current node."""
        if not self._branch_endpoints:
            return
        merge_node = GraphNode(
            type=NodeType.MERGE,
            name="Merge",
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._advance_position(),
        )
        self._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        self._last_node_id = merge_node.id

    def _add_node(self, node: GraphNode) -> GraphBuilder:
        self._auto_merge_if_needed()
        if node.position is None:
            node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._last_node_id = node.id
        return self

    def start(self) -> GraphBuilder:
        """Add a Start node."""
        node = GraphNode(type=NodeType.START, name="Start")
        self._start_node_id = node.id
        return self._add_node(node)

    def instruction(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.7,
        system_instruction: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an instruction (LLM call) node."""
        meta = {
            "system_instruction": system_instruction,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            **kwargs,
        }
        node = GraphNode(type=NodeType.INSTRUCTION, name=name, metadata=meta)
        return self._add_node(node)

    def decision(self, name: str, options: list[str] | None = None) -> GraphBuilder:
        """Add a decision node. Use ``.on(label)`` to build each branch."""
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.DECISION,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"decision_prompt": name},
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None  # branches take over
        if options:
            for opt in options:
                self._pending_branches[opt] = node.id
        return self

    def on(self, label: str) -> _BranchBuilder:
        """Start building a branch for a decision option."""
        if self._decision_node_id is None:
            raise ValueError("on() must be called after decision()")
        branch = _BranchBuilder(self, label)
        # The first node added to this branch will be connected from the decision node
        branch._last_node_id = self._decision_node_id
        # We need to override _add_node to set the edge label for the first edge
        original_add = branch._add_node

        def labeled_add(node: GraphNode) -> _BranchBuilder:
            if node.position is None:
                node.position = self._advance_position()
            self._nodes.append(node)
            assert self._decision_node_id is not None
            if branch._last_node_id == self._decision_node_id:
                edge = GraphEdge(
                    source_id=self._decision_node_id,
                    target_id=node.id,
                    label=label,
                )
                self._edges.append(edge)
            elif branch._last_node_id is not None:
                edge = GraphEdge(source_id=branch._last_node_id, target_id=node.id)
                self._edges.append(edge)
            branch._last_node_id = node.id
            branch._add_node = original_add  # type: ignore[method-assign]
            return branch

        branch._add_node = labeled_add  # type: ignore[method-assign]
        return branch

    def if_node(self, name: str, expression: str = "", variable: str = "") -> GraphBuilder:
        """Add an If conditional node. Use ``.on("true")`` / ``.on("false")`` for branches."""
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.IF,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"expression": expression, "variable": variable},
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def static(self, name: str, content: str = "") -> GraphBuilder:
        """Add a static content node."""
        node = GraphNode(type=NodeType.STATIC, name=name, metadata={"content": content})
        return self._add_node(node)

    def code(self, name: str, code: str = "", language: str = "python") -> GraphBuilder:
        """Add a code execution node."""
        node = GraphNode(
            type=NodeType.CODE,
            name=name,
            metadata={"language": language, "code": code},
        )
        return self._add_node(node)

    def user(self, name: str = "User Input") -> GraphBuilder:
        """Add a user input node."""
        node = GraphNode(type=NodeType.USER, name=name)
        return self._add_node(node)

    def merge(self, name: str = "Merge") -> GraphBuilder:
        """Add a merge node that collects all pending branch endpoints.

        All branch endpoints are connected to this merge node, which becomes
        the new current node for further chaining. Returns ``self`` for fluent
        chaining (unlike the legacy signature that returned the ``GraphNode``).
        """
        merge_node = GraphNode(
            type=NodeType.MERGE,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._advance_position(),
        )
        self._nodes.append(merge_node)
        # Connect all pending branch endpoints
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        # Also connect from current node if set (backward compat with merge_to pattern)
        if self._last_node_id is not None:
            self._edges.append(
                GraphEdge(source_id=self._last_node_id, target_id=merge_node.id)
            )
        self._last_node_id = merge_node.id
        return self

    def use(self, sub_graph: AgentVersion) -> GraphBuilder:
        """Inline a sub-graph into the current graph.

        Copies all nodes except START/END from the sub-graph, remaps their
        IDs, and connects them into the main chain.
        """
        self._auto_merge_if_needed()
        new_last = _inline_subgraph(
            sub_graph,
            self._nodes,
            self._edges,
            self._last_node_id,
            self._advance_position,
        )
        if new_last is not None:
            self._last_node_id = new_last
        return self

    def tool(self, name: str, tool_name: str = "", **tool_args: str) -> GraphBuilder:
        """Add a tool invocation node."""
        node = GraphNode(
            type=NodeType.TOOL,
            name=name,
            metadata={"tool_name": tool_name, "tool_args": tool_args},
        )
        return self._add_node(node)

    def sub_agent(self, name: str, agent_id: str = "") -> GraphBuilder:
        """Add a sub-agent node."""
        node = GraphNode(
            type=NodeType.SUB_AGENT,
            name=name,
            metadata={"agent_id": agent_id},
        )
        return self._add_node(node)

    def parallel(self, name: str = "Parallel") -> GraphBuilder:
        """Add a parallel fork node."""
        node = GraphNode(
            type=NodeType.PARALLEL,
            name=name,
            traverse_out=TraverseOut.SPAWN_ALL,
        )
        return self._add_node(node)

    def loop(
        self, name: str, max_iterations: int = 10, break_condition: str = ""
    ) -> GraphBuilder:
        """Add a loop node."""
        node = GraphNode(
            type=NodeType.LOOP,
            name=name,
            metadata={"max_iterations": max_iterations, "break_condition": break_condition},
        )
        return self._add_node(node)

    def end(self) -> GraphBuilder:
        """Add an End node.

        If there are pending branch endpoints, auto-merges them first so
        the End node is reachable from all branches.
        """
        node = GraphNode(type=NodeType.END, name="End")
        return self._add_node(node)

    def node(
        self,
        node_type: NodeType,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a generic node of any type."""
        node = GraphNode(type=node_type, name=name, metadata=metadata or {}, **kwargs)
        return self._add_node(node)

    def edge(
        self,
        source_id: UUID,
        target_id: UUID,
        label: str = "",
        is_main: bool = True,
    ) -> GraphBuilder:
        """Manually add an edge between two nodes."""
        e = GraphEdge(source_id=source_id, target_id=target_id, label=label, is_main=is_main)
        self._edges.append(e)
        return self

    def build(self, validate: bool = True, version: str = "1.0.0") -> AgentVersion:
        """Build the AgentVersion, optionally validating the graph.

        Raises ValueError if validation fails and validate=True.
        """
        if self._start_node_id is None:
            raise ValueError("Graph must have a Start node -- call .start() first")

        # If branches are still pending without a merge/end, add END nodes for each
        if self._branch_endpoints:
            for ep in self._branch_endpoints:
                end_node = GraphNode(
                    type=NodeType.END,
                    name="End",
                    position=self._advance_position(),
                )
                self._nodes.append(end_node)
                self._edges.append(GraphEdge(source_id=ep, target_id=end_node.id))
            self._branch_endpoints.clear()

        agent = Agent(name=self._name, description=self._description)
        agent_version = AgentVersion(
            agent_id=agent.id,
            version=version,
            start_node_id=self._start_node_id,
            nodes=self._nodes,
            edges=self._edges,
            created_at=datetime.now(timezone.utc),
        )

        if validate:
            errors = validate_graph(agent_version)
            real_errors = [e for e in errors if e.severity == "error"]
            if real_errors:
                msg = "; ".join(e.message for e in real_errors)
                raise ValueError(f"Graph validation failed: {msg}")

        return agent_version
