"""Fluent graph builder API for programmatic graph construction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Union
from uuid import UUID, uuid4

from quartermaster_graph.enums import (
    NodeType,
    TraverseIn,
    TraverseOut,
)
from quartermaster_graph.models import Agent, AgentVersion, GraphEdge, GraphNode, NodePosition
from quartermaster_graph.validation import validate_graph


# ── Helpers ──────────────────────────────────────────────────────────

def _llm_meta(
    model: str = "gpt-4o",
    provider: str = "openai",
    temperature: float = 0.5,
    system_instruction: str = "",
    stream: bool = True,
    max_output_tokens: int = 2048,
    max_input_tokens: int = 16385,
    vision: bool = False,
    thinking_level: str = "off",
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata dict using the ``llm_*`` key names that match
    the actual quartermaster-nodes ``AbstractLLMAssistantNode``."""
    meta: dict[str, Any] = {
        "llm_model": model,
        "llm_provider": provider,
        "llm_temperature": temperature,
        "llm_system_instruction": system_instruction,
        "llm_stream": stream,
        "llm_max_output_tokens": max_output_tokens,
        "llm_max_input_tokens": max_input_tokens,
        "llm_vision": vision,
        "llm_thinking_level": thinking_level,
    }
    meta.update(extra)
    return meta


def _inline_subgraph(
    sub_graph: AgentVersion | GraphBuilder,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    connect_from_id: UUID | None,
    advance_position: Callable[[], NodePosition],
) -> UUID | None:
    """Inline a sub-graph's nodes/edges, skipping START and END nodes.

    Returns the last non-END node ID of the inlined sub-graph (the new
    "current node" for further chaining), or the original *connect_from_id*
    if the sub-graph was effectively empty.

    Accepts either an ``AgentVersion`` or a ``GraphBuilder`` instance.
    """
    if isinstance(sub_graph, GraphBuilder):
        sub_graph._finalize()
        src_nodes = sub_graph._nodes
        src_edges = sub_graph._edges
        start_id = sub_graph._start_node_id
    else:
        src_nodes = sub_graph.nodes
        src_edges = sub_graph.edges
        start_id = sub_graph.start_node_id

    # Build old-id -> new-id mapping; skip START and END nodes
    id_map: dict[UUID, UUID] = {}
    kept_nodes: list[GraphNode] = []
    for n in src_nodes:
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
    first_after_start: UUID | None = None
    for e in src_edges:
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
    end_ids = {n.id for n in src_nodes if n.type == NodeType.END}
    for e in src_edges:
        src = id_map.get(e.source_id)
        tgt = id_map.get(e.target_id)
        if src is not None and tgt is not None:
            edges.append(
                GraphEdge(source_id=src, target_id=tgt, label=e.label, is_main=e.is_main)
            )

    # Find last node(s) -- those that had edges going to END, or if none, the last kept node
    nodes_to_end: list[UUID] = []
    for e in src_edges:
        if e.target_id in end_ids and e.source_id in id_map:
            nodes_to_end.append(id_map[e.source_id])
    if not nodes_to_end:
        nodes_to_end = [kept_nodes[-1].id]

    return nodes_to_end[-1]


# Type alias for the return type of end() — either a GraphBuilder or another
# _BranchBuilder when branches are nested.
_Parent = Union["GraphBuilder", "_BranchBuilder"]


class _BranchBuilder:
    """Builder for a branch (decision option, parallel path, or nested control flow).

    Supports full nesting: you can use ``if_node()``, ``decision()``,
    ``on()``, ``parallel()``, ``branch()``, and ``merge()`` inside a
    branch to create arbitrarily deep control flow.
    """

    def __init__(
        self,
        graph: GraphBuilder,
        parent: _Parent,
        label: str,
    ) -> None:
        self._graph = graph          # root GraphBuilder — owns all nodes/edges
        self._parent = parent        # immediate parent (GraphBuilder or _BranchBuilder)
        self._label = label
        self._last_node_id: UUID | None = None
        self._decision_node_id: UUID | None = None
        self._branch_endpoints: list[UUID] = []
        # Set by the creating on() method so use() can detect the first call
        self._origin_decision_id: UUID | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_node(self, node: GraphNode) -> _BranchBuilder:
        self._wire_pending_endpoints(node)
        if node.position is None:
            node.position = self._graph._advance_position()
        self._graph._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._graph._edges.append(edge)
        self._last_node_id = node.id
        return self

    def _wire_pending_endpoints(self, target_node: GraphNode) -> None:
        """Wire pending branch endpoints directly to *target_node*.

        After a decision/if/switch only ONE branch fires, so there is
        nothing to merge — the branches simply converge on the next node.
        For parallel fan-out use ``.merge()`` or ``.static_merge()``
        explicitly.
        """
        if not self._branch_endpoints:
            return
        for ep in self._branch_endpoints:
            self._graph._edges.append(
                GraphEdge(source_id=ep, target_id=target_node.id)
            )
        self._branch_endpoints.clear()
        # Clear _last_node_id so _add_node doesn't add a duplicate edge
        self._last_node_id = None

    def _auto_merge_if_needed(self) -> None:
        """No-op — kept for call-site compatibility.

        Branch endpoints are now wired directly to the next node in
        ``_add_node`` / ``_wire_pending_endpoints``.  Explicit
        ``.merge()`` or ``.static_merge()`` should be used after
        ``parallel()`` when all branches run concurrently.
        """

    # ------------------------------------------------------------------
    # LLM nodes
    # ------------------------------------------------------------------

    def instruction(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an Instruction node — pure LLM text generation, NO tools.

        Streams the LLM response. Use ``agent()`` instead if you need tool use.
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        node = GraphNode(type=NodeType.INSTRUCTION, name=name, metadata=meta)
        return self._add_node(node)

    def reasoning(
        self,
        name: str,
        model: str = "o1-mini",
        provider: str = "openai",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Reasoning node — specialized for o-series reasoning models.

        Does NOT use system instructions (reasoning models handle internally).
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=0.0,
            system_instruction="", **kwargs,
        )
        node = GraphNode(type=NodeType.REASONING, name=name, metadata=meta)
        return self._add_node(node)

    def summarize(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "Summarize the given conversation concisely.",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Summarize node — LLM condenses conversation history."""
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        node = GraphNode(type=NodeType.SUMMARIZE, name=name, metadata=meta)
        return self._add_node(node)

    def agent(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        system_instruction: str = "",
        tools: list[str] | None = None,
        max_iterations: int = 25,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an Agent node — autonomous agentic loop WITH tools.

        Iterates up to *max_iterations* times. Each iteration the LLM
        may call tools; the loop continues until no tool calls are returned.

        Args:
            tools: List of tool/program-version IDs the agent may call.
            max_iterations: Maximum loop iterations before forced stop.
        """
        meta = _llm_meta(
            model=model, provider=provider,
            system_instruction=system_instruction, **kwargs,
        )
        meta["program_version_ids"] = tools or []
        meta["max_iterations"] = max_iterations
        node = GraphNode(type=NodeType.AGENT, name=name, metadata=meta)
        return self._add_node(node)

    def vision(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        system_instruction: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an image vision/analysis node."""
        meta = _llm_meta(
            model=model, provider=provider,
            system_instruction=system_instruction, vision=True, **kwargs,
        )
        node = GraphNode(type=NodeType.INSTRUCTION_IMAGE_VISION, name=name, metadata=meta)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # User interaction nodes
    # ------------------------------------------------------------------

    def user(self, name: str = "User Input", prompts: list[str] | None = None) -> _BranchBuilder:
        """Add a User input node — pauses flow and awaits user response.

        Args:
            prompts: Optional text snippets to show the user.
        """
        meta: dict[str, Any] = {}
        if prompts:
            meta["text_snippets"] = prompts
        node = GraphNode(type=NodeType.USER, name=name, metadata=meta)
        return self._add_node(node)

    def user_form(
        self, name: str, parameters: list[dict] | None = None,
    ) -> _BranchBuilder:
        """Show a structured form to the user — pauses flow until submitted.

        Args:
            parameters: List of form field definitions.
        """
        node = GraphNode(
            type=NodeType.USER_FORM, name=name,
            metadata={"parameters": parameters or []},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Data nodes
    # ------------------------------------------------------------------

    def static(self, name: str, text: str = "") -> _BranchBuilder:
        """Add a Static node — outputs fixed text content, NO LLM."""
        node = GraphNode(
            type=NodeType.STATIC, name=name,
            metadata={"static_text": text},
        )
        return self._add_node(node)

    def text(self, name: str, template: str = "") -> _BranchBuilder:
        """Add a Text node — renders Jinja2 template using thought metadata."""
        node = GraphNode(
            type=NodeType.TEXT, name=name,
            metadata={"text": template},
        )
        return self._add_node(node)

    def var(self, name: str, variable: str = "", expression: str = "") -> _BranchBuilder:
        """Add a Var node — evaluates Python expression, stores result in metadata.

        Args:
            variable: Name of the variable to create.
            expression: Python expression to evaluate.
        """
        node = GraphNode(
            type=NodeType.VAR, name=name,
            metadata={"name": variable, "expression": expression},
        )
        return self._add_node(node)

    def code(self, name: str, code: str = "", filename: str = "") -> _BranchBuilder:
        """Add a Code node — code execution (handled by runtime environment)."""
        node = GraphNode(
            type=NodeType.CODE, name=name,
            metadata={"code": code, "filename": filename},
        )
        return self._add_node(node)

    def text_to_variable(self, name: str, variable: str = "", source: str = "") -> _BranchBuilder:
        """Convert text output to a variable."""
        node = GraphNode(
            type=NodeType.TEXT_TO_VARIABLE, name=name,
            metadata={"variable": variable, "source": source},
        )
        return self._add_node(node)

    def program_runner(self, name: str, program: str = "", **kwargs: Any) -> _BranchBuilder:
        """Run a program/tool inline."""
        node = GraphNode(
            type=NodeType.PROGRAM_RUNNER, name=name,
            metadata={"program": program, **kwargs},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Control flow (nested)
    # ------------------------------------------------------------------

    def if_node(self, name: str, expression: str = "") -> _BranchBuilder:
        """Add an IF node — evaluates Python expression, picks true/false branch.

        NO LLM call. Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.IF, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"if_expression": expression},
        )
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def decision(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        prefix_message: str = "",
        suffix_message: str = "",
        options: list[str] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Decision node — LLM picks ONE path via ``pick_path`` tool.

        The LLM sees the available edge labels and calls an internal tool
        to select which branch to follow. Does NOT stream.

        Use ``.on(label)`` for each option.
        """
        self._auto_merge_if_needed()
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            stream=False, **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        node = GraphNode(
            type=NodeType.DECISION, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata=meta,
        )
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def static_decision(self, name: str, expression: str = "") -> _BranchBuilder:
        """Add a StaticDecision node — expression-based branching, NO LLM.

        Like ``if_node()`` but uses ``StaticDecision1`` node type.
        Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.STATIC_DECISION, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"expression": expression},
        )
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def user_decision(self, name: str) -> _BranchBuilder:
        """Add a UserDecision node — presents choices to user, user picks path.

        Waits for ALL incoming branches (traverse_in=AwaitAll), then
        pauses flow until the user selects a path.

        Use ``.on(label)`` for each option.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.USER_DECISION, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def on(self, label: str) -> _BranchBuilder:
        """Start a sub-branch for a decision/if option inside this branch."""
        if self._decision_node_id is None:
            raise ValueError("on() must be called after if_node(), decision(), static_decision(), or user_decision()")
        decision_id = self._decision_node_id
        sub = _BranchBuilder(self._graph, self, label)
        sub._last_node_id = decision_id
        sub._origin_decision_id = decision_id
        original_add = sub._add_node

        def labeled_add(node: GraphNode) -> _BranchBuilder:
            if node.position is None:
                node.position = self._graph._advance_position()
            self._graph._nodes.append(node)
            if sub._last_node_id == decision_id:
                edge = GraphEdge(
                    source_id=decision_id,
                    target_id=node.id,
                    label=label,
                )
                self._graph._edges.append(edge)
            elif sub._last_node_id is not None:
                self._graph._edges.append(
                    GraphEdge(source_id=sub._last_node_id, target_id=node.id)
                )
            sub._last_node_id = node.id
            sub._add_node = original_add  # type: ignore[method-assign]
            return sub

        sub._add_node = labeled_add  # type: ignore[method-assign]
        return sub

    def switch(
        self, name: str,
        cases: list[dict[str, str]] | None = None,
        default_edge_id: str = "",
    ) -> _BranchBuilder:
        """Add a Switch node — evaluates multiple cases, first match wins. NO LLM.

        Args:
            cases: List of ``{"expression": "...", "edge_id": "..."}`` dicts.
            default_edge_id: Fallback edge ID if no case matches.
        """
        meta: dict[str, Any] = {
            "cases": cases or [],
            "default_edge_id": default_edge_id,
        }
        node = GraphNode(
            type=NodeType.SWITCH, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata=meta,
        )
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def break_node(self, name: str = "Break", targets: list[str] | None = None) -> _BranchBuilder:
        """Add a Break node — stops backward message collection.

        Args:
            targets: What to clear: ``[]`` (full break), ``['tools']``, ``['thinking']``.
        """
        node = GraphNode(
            type=NodeType.BREAK, name=name,
            metadata={"break_targets": targets or []},
        )
        return self._add_node(node)

    def parallel(self, name: str = "Parallel") -> _BranchBuilder:
        """Start a parallel fan-out inside this branch.

        Use ``.branch()`` to define each parallel path and ``.merge()``
        to rejoin them.
        """
        self._auto_merge_if_needed()
        if self._last_node_id is not None:
            for n in self._graph._nodes:
                if n.id == self._last_node_id:
                    n.traverse_out = TraverseOut.SPAWN_ALL
                    break
        self._decision_node_id = self._last_node_id
        self._last_node_id = None
        return self

    def branch(self) -> _BranchBuilder:
        """Start a parallel branch inside this branch."""
        if self._decision_node_id is None:
            raise ValueError("branch() must be called after parallel()")
        sub = _BranchBuilder(self._graph, self, "")
        sub._last_node_id = self._decision_node_id
        return sub

    def merge(
        self, name: str = "Merge",
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "",
        prefix_message: str = "Compress following conversations into one",
        suffix_message: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Merge node — LLM combines all parallel branch outputs.

        Waits for ALL branches (traverse_in=AwaitAll), then sends the
        combined content to the LLM for compression into one message.

        Use ``static_merge()`` if you don't need LLM processing.
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        merge_node = GraphNode(
            type=NodeType.MERGE, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._graph._advance_position(),
            metadata=meta,
        )
        self._graph._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._graph._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._graph._edges.append(
                GraphEdge(source_id=self._last_node_id, target_id=merge_node.id)
            )
        self._last_node_id = merge_node.id
        return self

    def static_merge(self, name: str = "Merge", text: str = "") -> _BranchBuilder:
        """Add a StaticMerge node — combines branches WITHOUT LLM.

        Waits for ALL branches (traverse_in=AwaitAll), appends static text,
        then continues. No LLM call involved.
        """
        merge_node = GraphNode(
            type=NodeType.STATIC_MERGE, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._graph._advance_position(),
            metadata={"static_text": text},
        )
        self._graph._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._graph._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._graph._edges.append(
                GraphEdge(source_id=self._last_node_id, target_id=merge_node.id)
            )
        self._last_node_id = merge_node.id
        return self

    # ------------------------------------------------------------------
    # Memory nodes
    # ------------------------------------------------------------------

    def read_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variable_names: list[str] | None = None,
    ) -> _BranchBuilder:
        """Read variables from persistent memory into thought metadata.

        Args:
            memory_name: Name of the memory store.
            memory_type: ``"flow"`` (scoped to this flow) or ``"user"`` (persists across flows).
            variable_names: Which variables to load.
        """
        node = GraphNode(
            type=NodeType.READ_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variable_names": variable_names or [],
            },
        )
        return self._add_node(node)

    def write_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
    ) -> _BranchBuilder:
        """Write variables from thought metadata to persistent memory.

        Args:
            memory_name: Name of the memory store.
            memory_type: ``"flow"`` or ``"user"``.
            variables: List of ``{"name": "...", "expression": "..."}`` dicts.
        """
        node = GraphNode(
            type=NodeType.WRITE_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
        )
        return self._add_node(node)

    def update_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
    ) -> _BranchBuilder:
        """Update existing persistent memory variables."""
        node = GraphNode(
            type=NodeType.UPDATE_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
        )
        return self._add_node(node)

    def flow_memory(
        self, name: str = "Flow Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
    ) -> _BranchBuilder:
        """Define flow-scoped persistent memory (not connected to flow edges).

        This is a *definition* node — the Start node initialises it at
        runtime via ``_memory_initializer``.
        """
        node = GraphNode(
            type=NodeType.FLOW_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
        )
        return self._add_node(node)

    def user_memory(
        self, name: str = "User Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
    ) -> _BranchBuilder:
        """Define user-scoped persistent memory (survives across flow executions)."""
        node = GraphNode(
            type=NodeType.USER_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Utility nodes
    # ------------------------------------------------------------------

    def comment(self, name: str, text: str = "") -> _BranchBuilder:
        """Add a Comment node — documentation only, no runtime logic."""
        node = GraphNode(
            type=NodeType.COMMENT, name=name,
            metadata={"comment": text},
        )
        return self._add_node(node)

    def sub_agent(self, name: str, graph_id: str = "") -> _BranchBuilder:
        """Call another agent graph synchronously (blocks until sub-graph completes).

        This is different from ``spawn_agent`` (session tool) which runs agents
        in background sessions. Sub-agent nodes execute inline and return
        their result to the current flow.

        Args:
            graph_id: ID of the agent graph to execute as a sub-flow.
        """
        node = GraphNode(
            type=NodeType.SUB_ASSISTANT, name=name,
            metadata={"sub_assistant_id": graph_id},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------

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

    def use(self, sub_graph: AgentVersion | GraphBuilder) -> _BranchBuilder:
        """Inline a sub-graph into this branch.

        Copies all nodes except START/END from the sub-graph, remaps their
        IDs, and connects them into the current branch chain.

        Accepts either an ``AgentVersion`` or a ``GraphBuilder`` instance.
        """
        # Detect whether this is the first call after on() — need to label the edge
        is_first = (
            self._origin_decision_id is not None
            and self._last_node_id == self._origin_decision_id
        )
        new_last = _inline_subgraph(
            sub_graph,
            self._graph._nodes,
            self._graph._edges,
            self._last_node_id,
            self._graph._advance_position,
        )
        # Patch the label on the connecting edge if this was the first call
        if is_first and self._graph._edges:
            for edge in reversed(self._graph._edges):
                if edge.source_id == self._origin_decision_id:
                    edge.label = self._label
                    break
        if new_last is not None:
            self._last_node_id = new_last
        return self

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def end(self) -> Any:
        """End this branch and return to the parent.

        Records the branch endpoint so that a subsequent ``.merge()``
        or auto-merge can connect all branches.

        Returns the parent — either a ``GraphBuilder`` or another
        ``_BranchBuilder`` for nested control flow.
        """
        if self._last_node_id is not None:
            self._parent._branch_endpoints.append(self._last_node_id)
        return self._parent

    def merge_to(self, merge_node_id: UUID) -> Any:
        """Connect this branch to an existing merge node, then return to parent."""
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=merge_node_id)
            self._graph._edges.append(edge)
        return self._parent


class GraphBuilder:
    """Fluent builder for creating agent graphs programmatically.

    The ``GraphBuilder`` itself IS the graph -- you can access ``.nodes``
    and ``.edges`` directly without calling ``.build()``.  The ``.build()``
    method is retained for backward compatibility and returns an
    ``AgentVersion``.

    Node semantics match the actual quartermaster-nodes implementations:

    * **Instruction** — pure LLM text generation, NO tools, streams response.
    * **Decision** — LLM picks ONE path via internal ``pick_path`` tool (non-streaming).
    * **StaticDecision** — evaluates Python expression, picks true/false. NO LLM.
    * **UserDecision** — user picks which path to follow.
    * **Merge** — LLM combines parallel branch outputs into one message.
    * **StaticMerge** — combines branches WITHOUT LLM (just joins thoughts).
    * **Agent** — autonomous agentic loop WITH tools, iterates up to max_iterations.
    * **SubAssistant** — calls another graph synchronously, blocks until complete.
    * **If** / **Switch** — expression evaluation, NO LLM call.

    Example::

        graph = (
            GraphBuilder("My Agent")
            .start()
            .instruction("Analyze input", model="gpt-4o")
            .decision("Route?")
            .on("Yes").instruction("Positive response").end()
            .on("No").instruction("Negative response").end()
            .merge("Combine")
            .end()
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
        self._allowed_agents: list[str] = []

    # ------------------------------------------------------------------
    # Agent control
    # ------------------------------------------------------------------

    def allowed_agents(self, *agent_ids: str) -> GraphBuilder:
        """Declare which sub-agents this graph is allowed to spawn.

        Call this on the graph to restrict which agent IDs can be used
        with ``.sub_agent()`` nodes and the ``spawn_agent`` tool at
        runtime.  An empty list means "allow all" (default).

        Example::

            graph = (
                Graph("Coordinator")
                .allowed_agents("researcher", "writer", "reviewer")
                .start()
                .user("Task")
                .sub_agent("Do research", graph_id="researcher")
                .end()
            )
        """
        self._allowed_agents = list(agent_ids)
        return self

    # ------------------------------------------------------------------
    # Graph-like properties -- auto-finalize on access
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[GraphNode]:
        """Return all nodes. Auto-finalizes pending branches."""
        self._finalize()
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        """Return all edges. Auto-finalizes pending branches."""
        self._finalize()
        return self._edges

    @property
    def start_node_id(self) -> UUID:
        if self._start_node_id is None:
            raise ValueError("No start node")
        return self._start_node_id

    def _finalize(self) -> None:
        """Auto-merge any pending branches and ensure graph is valid."""
        if self._branch_endpoints:
            for ep in self._branch_endpoints:
                end_node = GraphNode(
                    type=NodeType.END, name="End", position=self._advance_position()
                )
                self._nodes.append(end_node)
                self._edges.append(GraphEdge(source_id=ep, target_id=end_node.id))
            self._branch_endpoints.clear()

    def get_node(self, node_id: UUID) -> GraphNode | None:
        """Find a node by its ID."""
        return next((n for n in self._nodes if n.id == node_id), None)

    def get_successors(self, node_id: UUID) -> list[GraphNode]:
        """Return all successor nodes of the given node."""
        target_ids = [e.target_id for e in self._edges if e.source_id == node_id]
        return [n for n in self._nodes if n.id in target_ids]

    def get_edges_from(self, node_id: UUID) -> list[GraphEdge]:
        """Return all edges originating from the given node."""
        return [e for e in self._edges if e.source_id == node_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_position(self) -> NodePosition:
        pos = NodePosition(x=self._position_x, y=self._position_y)
        self._position_y += 120
        return pos

    def _auto_merge_if_needed(self) -> None:
        """No-op — kept for call-site compatibility.

        Branch endpoints are now wired directly to the next node in
        ``_add_node`` / ``_wire_pending_endpoints``.  Explicit
        ``.merge()`` or ``.static_merge()`` should be used after
        ``parallel()`` when all branches run concurrently.
        """

    def _wire_pending_endpoints(self, target_node: GraphNode) -> None:
        """Wire pending branch endpoints directly to *target_node*.

        After a decision/if/switch only ONE branch fires, so there is
        nothing to merge — the branches simply converge on the next node.
        """
        if not self._branch_endpoints:
            return
        for ep in self._branch_endpoints:
            self._edges.append(
                GraphEdge(source_id=ep, target_id=target_node.id)
            )
        self._branch_endpoints.clear()
        self._last_node_id = None

    def _add_node(self, node: GraphNode) -> GraphBuilder:
        self._wire_pending_endpoints(node)
        if node.position is None:
            node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._last_node_id = node.id
        return self

    # ------------------------------------------------------------------
    # Start / End
    # ------------------------------------------------------------------

    def start(self) -> GraphBuilder:
        """Add a Start node."""
        node = GraphNode(type=NodeType.START, name="Start")
        self._start_node_id = node.id
        return self._add_node(node)

    def end(self) -> GraphBuilder:
        """Add an End node.

        If there are pending branch endpoints, auto-merges them first so
        the End node is reachable from all branches.
        """
        node = GraphNode(type=NodeType.END, name="End")
        return self._add_node(node)

    # ------------------------------------------------------------------
    # LLM nodes
    # ------------------------------------------------------------------

    def instruction(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an Instruction node — pure LLM text generation, NO tools.

        Streams the LLM response. Use ``agent()`` instead if you need tool use.
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        node = GraphNode(type=NodeType.INSTRUCTION, name=name, metadata=meta)
        return self._add_node(node)

    def reasoning(
        self,
        name: str,
        model: str = "o1-mini",
        provider: str = "openai",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Reasoning node — specialized for o-series reasoning models.

        Does NOT use system instructions (reasoning models handle internally).
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=0.0,
            system_instruction="", **kwargs,
        )
        node = GraphNode(type=NodeType.REASONING, name=name, metadata=meta)
        return self._add_node(node)

    def summarize(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "Summarize the given conversation concisely.",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Summarize node — LLM condenses conversation history."""
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        node = GraphNode(type=NodeType.SUMMARIZE, name=name, metadata=meta)
        return self._add_node(node)

    def agent(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        system_instruction: str = "",
        tools: list[str] | None = None,
        max_iterations: int = 25,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an Agent node — autonomous agentic loop WITH tools.

        Iterates up to *max_iterations* times. Each iteration the LLM
        may call tools; the loop continues until no tool calls are returned.

        Args:
            tools: List of tool/program-version IDs the agent may call.
            max_iterations: Maximum loop iterations before forced stop.
        """
        meta = _llm_meta(
            model=model, provider=provider,
            system_instruction=system_instruction, **kwargs,
        )
        meta["program_version_ids"] = tools or []
        meta["max_iterations"] = max_iterations
        node = GraphNode(type=NodeType.AGENT, name=name, metadata=meta)
        return self._add_node(node)

    def vision(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        system_instruction: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an image vision/analysis node (INSTRUCTION_IMAGE_VISION)."""
        meta = _llm_meta(
            model=model, provider=provider,
            system_instruction=system_instruction, vision=True, **kwargs,
        )
        node = GraphNode(type=NodeType.INSTRUCTION_IMAGE_VISION, name=name, metadata=meta)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def decision(
        self,
        name: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        prefix_message: str = "",
        suffix_message: str = "",
        options: list[str] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Decision node — LLM picks ONE path via ``pick_path`` tool.

        The LLM sees the available edge labels and calls an internal tool
        to select which branch to follow. Does NOT stream.

        Use ``.on(label)`` to define each branch.
        """
        self._auto_merge_if_needed()
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            stream=False, **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        node = GraphNode(
            type=NodeType.DECISION, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata=meta,
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

    def static_decision(self, name: str, expression: str = "") -> GraphBuilder:
        """Add a StaticDecision node — expression-based branching, NO LLM.

        Evaluates a Python expression and picks the true/false branch.
        Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.STATIC_DECISION, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"expression": expression},
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def switch(
        self, name: str,
        cases: list[dict[str, str]] | None = None,
        default_edge_id: str = "",
    ) -> GraphBuilder:
        """Add a Switch node — evaluates multiple cases, first match wins. NO LLM.

        Args:
            cases: List of ``{"expression": "...", "edge_id": "..."}`` dicts.
            default_edge_id: Fallback edge ID if no case matches.
        """
        self._auto_merge_if_needed()
        meta: dict[str, Any] = {
            "cases": cases or [],
            "default_edge_id": default_edge_id,
        }
        node = GraphNode(
            type=NodeType.SWITCH, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata=meta,
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        if cases:
            for c in cases:
                label = c.get("expression", "")
                self._pending_branches[label] = node.id
        return self

    def on(self, label: str) -> _BranchBuilder:
        """Start building a branch for a decision/if/switch option."""
        if self._decision_node_id is None:
            raise ValueError("on() must be called after decision(), if_node(), static_decision(), or user_decision()")
        decision_id = self._decision_node_id
        branch = _BranchBuilder(self, self, label)
        # The first node added to this branch will be connected from the decision node
        branch._last_node_id = decision_id
        branch._origin_decision_id = decision_id
        # We need to override _add_node to set the edge label for the first edge
        original_add = branch._add_node

        def labeled_add(node: GraphNode) -> _BranchBuilder:
            if node.position is None:
                node.position = self._advance_position()
            self._nodes.append(node)
            if branch._last_node_id == decision_id:
                edge = GraphEdge(
                    source_id=decision_id,
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

    def if_node(self, name: str, expression: str = "") -> GraphBuilder:
        """Add an If node — evaluates Python expression, picks true/false branch.

        NO LLM call. Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.IF, name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"if_expression": expression},
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def break_node(self, name: str = "Break", targets: list[str] | None = None) -> GraphBuilder:
        """Add a Break node — stops backward message collection.

        Args:
            targets: What to clear: ``[]`` (full break), ``['tools']``, ``['thinking']``.
        """
        node = GraphNode(
            type=NodeType.BREAK, name=name,
            metadata={"break_targets": targets or []},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Data nodes
    # ------------------------------------------------------------------

    def static(self, name: str, text: str = "") -> GraphBuilder:
        """Add a Static node — outputs fixed text content, NO LLM."""
        node = GraphNode(
            type=NodeType.STATIC, name=name,
            metadata={"static_text": text},
        )
        return self._add_node(node)

    def code(self, name: str, code: str = "", filename: str = "") -> GraphBuilder:
        """Add a Code node — code execution (handled by runtime environment)."""
        node = GraphNode(
            type=NodeType.CODE, name=name,
            metadata={"code": code, "filename": filename},
        )
        return self._add_node(node)

    def var(self, name: str, variable: str = "", expression: str = "") -> GraphBuilder:
        """Add a Var node — evaluates Python expression, stores result in metadata.

        Args:
            variable: Name of the variable to create.
            expression: Python expression to evaluate.
        """
        node = GraphNode(
            type=NodeType.VAR, name=name,
            metadata={"name": variable, "expression": expression},
        )
        return self._add_node(node)

    def text(self, name: str, template: str = "") -> GraphBuilder:
        """Add a Text node — renders Jinja2 template using thought metadata."""
        node = GraphNode(
            type=NodeType.TEXT, name=name,
            metadata={"text": template},
        )
        return self._add_node(node)

    def text_to_variable(self, name: str, variable: str = "", source: str = "") -> GraphBuilder:
        """Convert text output to a variable."""
        node = GraphNode(
            type=NodeType.TEXT_TO_VARIABLE, name=name,
            metadata={"variable": variable, "source": source},
        )
        return self._add_node(node)

    def program_runner(self, name: str, program: str = "", **kwargs: Any) -> GraphBuilder:
        """Run a program/tool inline."""
        node = GraphNode(
            type=NodeType.PROGRAM_RUNNER, name=name,
            metadata={"program": program, **kwargs},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Memory nodes
    # ------------------------------------------------------------------

    def read_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variable_names: list[str] | None = None,
    ) -> GraphBuilder:
        """Read variables from persistent memory into thought metadata."""
        node = GraphNode(
            type=NodeType.READ_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variable_names": variable_names or [],
            },
        )
        return self._add_node(node)

    def write_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
    ) -> GraphBuilder:
        """Write variables from thought metadata to persistent memory."""
        node = GraphNode(
            type=NodeType.WRITE_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
        )
        return self._add_node(node)

    def update_memory(
        self, name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
    ) -> GraphBuilder:
        """Update existing persistent memory variables."""
        node = GraphNode(
            type=NodeType.UPDATE_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
        )
        return self._add_node(node)

    def flow_memory(
        self, name: str = "Flow Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
    ) -> GraphBuilder:
        """Define flow-scoped persistent memory."""
        node = GraphNode(
            type=NodeType.FLOW_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
        )
        return self._add_node(node)

    def user_memory(
        self, name: str = "User Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
    ) -> GraphBuilder:
        """Define user-scoped persistent memory (survives across flow executions)."""
        node = GraphNode(
            type=NodeType.USER_MEMORY, name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # User interaction nodes
    # ------------------------------------------------------------------

    def user(self, name: str = "User Input", prompts: list[str] | None = None) -> GraphBuilder:
        """Add a User input node — pauses flow and awaits user response.

        Args:
            prompts: Optional text snippets to show the user.
        """
        meta: dict[str, Any] = {}
        if prompts:
            meta["text_snippets"] = prompts
        node = GraphNode(type=NodeType.USER, name=name, metadata=meta)
        return self._add_node(node)

    def user_decision(self, name: str) -> GraphBuilder:
        """Add a UserDecision node — presents choices to user, user picks path.

        Waits for ALL incoming branches (traverse_in=AwaitAll), then
        pauses flow until the user selects a path.

        Use ``.on(label)`` for each option.
        """
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.USER_DECISION, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            traverse_out=TraverseOut.SPAWN_PICKED,
        )
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def user_form(
        self, name: str, parameters: list[dict] | None = None,
    ) -> GraphBuilder:
        """Show a structured form to the user — pauses flow until submitted.

        Args:
            parameters: List of form field definitions.
        """
        node = GraphNode(
            type=NodeType.USER_FORM, name=name,
            metadata={"parameters": parameters or []},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Utility nodes
    # ------------------------------------------------------------------

    def comment(self, name: str, text: str = "") -> GraphBuilder:
        """Add a Comment node — documentation only, no runtime logic."""
        node = GraphNode(
            type=NodeType.COMMENT, name=name,
            metadata={"comment": text},
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Composition / sub-graphs
    # ------------------------------------------------------------------

    def sub_agent(self, name: str, graph_id: str = "") -> GraphBuilder:
        """Call another agent graph synchronously (blocks until sub-graph completes).

        This is different from ``spawn_agent`` (session tool) which runs agents
        in background sessions. Sub-agent nodes execute inline and return
        their result to the current flow.

        Args:
            graph_id: ID of the agent graph to execute as a sub-flow.
        """
        node = GraphNode(
            type=NodeType.SUB_ASSISTANT, name=name,
            metadata={"sub_assistant_id": graph_id},
        )
        return self._add_node(node)

    def parallel(self, name: str = "Parallel") -> GraphBuilder:
        """Start a parallel fan-out from the current node.

        Unlike ``.decision()`` (which picks ONE branch via LLM),
        ``.parallel()`` executes ALL branches concurrently.  Use
        ``.branch()`` to define each parallel path and ``.merge()``
        (LLM-based) or ``.static_merge()`` (no LLM) to rejoin.
        """
        self._auto_merge_if_needed()
        if self._last_node_id is not None:
            for n in self._nodes:
                if n.id == self._last_node_id:
                    n.traverse_out = TraverseOut.SPAWN_ALL
                    break
        self._decision_node_id = self._last_node_id
        self._last_node_id = None
        return self

    def branch(self) -> _BranchBuilder:
        """Start a parallel branch from the current fan-out point."""
        if self._decision_node_id is None:
            raise ValueError("branch() must be called after parallel()")
        b = _BranchBuilder(self, self, "")
        b._last_node_id = self._decision_node_id
        return b

    def merge(
        self, name: str = "Merge",
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.5,
        system_instruction: str = "",
        prefix_message: str = "Compress following conversations into one",
        suffix_message: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Merge node — LLM combines all parallel branch outputs.

        Waits for ALL branches (traverse_in=AwaitAll), then sends the
        combined content to the LLM for compression into one message.

        Use ``static_merge()`` instead if you don't need LLM processing.
        """
        meta = _llm_meta(
            model=model, provider=provider, temperature=temperature,
            system_instruction=system_instruction, **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        merge_node = GraphNode(
            type=NodeType.MERGE, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._advance_position(),
            metadata=meta,
        )
        self._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._edges.append(
                GraphEdge(source_id=self._last_node_id, target_id=merge_node.id)
            )
        self._last_node_id = merge_node.id
        return self

    def static_merge(self, name: str = "Merge", text: str = "") -> GraphBuilder:
        """Add a StaticMerge node — combines branches WITHOUT LLM.

        Waits for ALL branches (traverse_in=AwaitAll), appends static text,
        then continues. No LLM call involved.
        """
        merge_node = GraphNode(
            type=NodeType.STATIC_MERGE, name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._advance_position(),
            metadata={"static_text": text},
        )
        self._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._edges.append(
                GraphEdge(source_id=self._last_node_id, target_id=merge_node.id)
            )
        self._last_node_id = merge_node.id
        return self

    def use(self, sub_graph: AgentVersion | GraphBuilder) -> GraphBuilder:
        """Inline a sub-graph into the current graph."""
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

    # ------------------------------------------------------------------
    # Generic node
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Build / export
    # ------------------------------------------------------------------

    def to_version(
        self,
        validate: bool = True,
        version: str = "0.1.0",
        agent_id: UUID | None = None,
    ) -> AgentVersion:
        """Export the graph as an ``AgentVersion``."""
        if self._start_node_id is None:
            raise ValueError("Graph has no start node — call .start() first")
        self._finalize()
        ver = AgentVersion(
            id=uuid4(),
            agent_id=agent_id or uuid4(),
            version=version,
            nodes=list(self._nodes),
            edges=list(self._edges),
            start_node_id=self._start_node_id,
            created_at=datetime.now(timezone.utc),
        )
        if validate:
            validate_graph(ver)
        return ver

    def build(self, validate: bool = True) -> AgentVersion:
        """Build and return the ``AgentVersion``.  Alias for ``to_version()``."""
        return self.to_version(validate=validate)

    def to_agent(self, validate: bool = True, version: str = "0.1.0") -> Agent:
        """Export as a full ``Agent`` (with a single version)."""
        ver = self.to_version(validate=validate, version=version)
        return Agent(
            id=uuid4(),
            name=self._name,
            description=self._description,
            versions=[ver],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
