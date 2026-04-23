"""Fluent graph builder API for programmatic graph construction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Union
from uuid import UUID, uuid4

from quartermaster_graph.enums import (
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)
from quartermaster_graph.models import Agent, GraphSpec, GraphEdge, GraphNode, NodePosition
from quartermaster_graph.validation import validate_graph


# ── Flow config keys that can be passed as kwargs to any node method ──

_FLOW_CONFIG_KEYS = {
    "traverse_in",
    "traverse_out",
    "thought_type",
    "message_type",
    "error_handling",
}

# Metadata-level config keys (stored in node.metadata, not as node attributes).
# ``capture_as`` (v0.2.0) lets callers name a node's output so
# ``Result.captures["name"]`` can retrieve it post-run without fishing through
# UUID-keyed ``node_results``.
_META_CONFIG_KEYS = {"show_output", "capture_as"}

# Node-metadata key under which ``capture_as`` is stored.  Kept as a named
# constant so the engine side can import it rather than hard-coding the
# string (see ``quartermaster_engine.runner.flow_runner``).
CAPTURE_AS_METADATA_KEY = "capture_as"

# v0.7.0: node-metadata key under which the ``retry_max_attempts`` integer
# (normalised budget for node-level retries) is stored.  Kept as a named
# constant so the engine side can import it rather than hard-coding.
RETRY_MAX_ATTEMPTS_METADATA_KEY = "retry_max_attempts"

# Combined set for extraction from kwargs
_ALL_CONFIG_KEYS = _FLOW_CONFIG_KEYS | _META_CONFIG_KEYS


def _apply_flow_config(node: GraphNode, kwargs: dict[str, Any]) -> None:
    """Apply optional flow config kwargs (traverse_in, traverse_out, etc.) to a node."""
    for key in _FLOW_CONFIG_KEYS:
        if key in kwargs:
            setattr(node, key, kwargs.pop(key))
    for key in _META_CONFIG_KEYS:
        if key in kwargs:
            value = kwargs.pop(key)
            if value is not None:
                node.metadata[key] = value


# ── Helpers ──────────────────────────────────────────────────────────


def _llm_meta(
    model: str = "",
    provider: str = "",
    temperature: float = 0.5,
    system_instruction: str = "",
    stream: bool = True,
    max_output_tokens: int = 2048,
    max_input_tokens: int = 16385,
    vision: bool = False,
    thinking_level: str = "off",
    extra_body: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a metadata dict using the ``llm_*`` key names that match
    the actual quartermaster-nodes ``AbstractLLMAssistantNode``.

    The ``extra_body`` arg is v0.6.0 — a provider-specific OpenAI-compat
    escape hatch (spliced into the outgoing ``chat.completions.create(...,
    extra_body=...)`` payload). Typical use: toggle Gemma-4 thinking via
    ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
    """
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
    if extra_body:
        # Stash on the node metadata. The engine's executor layer reads
        # ``llm_extra_body`` and forwards it into ``LLMConfig.extra_body``.
        meta["llm_extra_body"] = dict(extra_body)
    meta.update({k: v for k, v in extra.items() if k not in _ALL_CONFIG_KEYS})
    return meta


def _apply_retry_spec(
    node: GraphNode,
    retry: dict[str, Any] | None,
    predicate_registry: dict[str, Any],
) -> None:
    """Apply a v0.7.0 retry spec to *node* metadata and *predicate_registry*.

    The retry spec is a dict with shape::

        {"max_attempts": int, "on": Callable[[NodeResult], bool] | None}

    Only the integer budget lands on ``node.metadata[RETRY_MAX_ATTEMPTS_METADATA_KEY]``
    because node metadata must survive JSON / YAML round-trips.  The ``on``
    predicate (if any) is stashed in *predicate_registry* keyed by node name
    so the engine can look it up at run time via the same inline-tools-style
    side-channel — it does NOT survive serialisation.

    ``max_attempts`` values ``<= 0`` are normalised to ``1`` (no retries,
    just the initial attempt).  The spec is a no-op when ``retry`` is
    ``None`` or an empty dict — metadata stays clean so round-trips don't
    acquire stale keys.
    """
    if not retry:
        return
    max_attempts = retry.get("max_attempts", 1)
    try:
        max_attempts_int = int(max_attempts)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"retry=: 'max_attempts' must be int-coercible, got {max_attempts!r}"
        ) from exc
    if max_attempts_int <= 0:
        max_attempts_int = 1
    node.metadata[RETRY_MAX_ATTEMPTS_METADATA_KEY] = max_attempts_int

    predicate = retry.get("on")
    if predicate is not None:
        if not callable(predicate):
            raise TypeError(
                f"retry={{'on': ...}}: expected a callable predicate, got "
                f"{type(predicate).__name__}"
            )
        # Keyed by node NAME — the engine resolves predicates by looking
        # the node's name up in ``FlowRunner.retry_predicates`` at run time.
        # Callable objects can't survive JSON/YAML so they live only in
        # this in-memory registry (mirrors the inline-tools design).
        predicate_registry[node.name] = predicate


def _normalise_agent_tools(
    tools: list[Any] | None,
    inline_registry: dict[str, Any],
) -> list[str]:
    """Normalise an ``agent(tools=...)`` argument to a list of tool NAMES.

    The v0.4.0 surface accepts a mix of:

    * ``str`` — a tool name the run-time registry already knows about
      (classic v0.3.x behaviour, unchanged).
    * A ``FunctionTool`` / :class:`AbstractTool` instance — typically
      what ``@tool()`` produces. Stashed in *inline_registry* by name
      so the SDK runner can register it lazily at run time.
    * A plain callable (undecorated ``def``) — auto-decorated on the
      fly via :func:`quartermaster_tools.auto_decorate`, then stashed
      the same way.
    * A lambda / unintrospectable callable — raises ``ValueError`` with
      an actionable message telling the caller to decorate explicitly.

    Returns the ordered list of NAMES to store on the node metadata.
    Mutates *inline_registry* in place with ``{name: tool_instance}``
    entries for every callable encountered.
    """
    if not tools:
        return []
    try:
        from quartermaster_tools import (
            auto_decorate,
            is_quartermaster_tool,
        )
    except ImportError as exc:  # pragma: no cover — quartermaster-tools is a hard dep
        raise ImportError(
            "quartermaster_tools is required to pass callables to "
            "agent(tools=[...]). Install quartermaster-tools or pass tool "
            "name strings instead."
        ) from exc

    result: list[str] = []
    for item in tools:
        if isinstance(item, str):
            result.append(item)
            continue
        if is_quartermaster_tool(item):
            tool_name = item.name()
            inline_registry[tool_name] = item
            result.append(tool_name)
            continue
        if callable(item):
            try:
                wrapped = auto_decorate(item)
            except ValueError as exc:
                raise ValueError(
                    f"tools=[{item!r}] — function is not @tool()-decorated "
                    f"and cannot be auto-decorated ({exc}). Either decorate "
                    "it with @tool() or register it explicitly via "
                    "get_default_registry().register(...) and pass the "
                    "name string."
                ) from exc
            tool_name = wrapped.name()
            inline_registry[tool_name] = wrapped
            result.append(tool_name)
            continue
        raise TypeError(
            f"tools=[{item!r}] — unsupported item type {type(item).__name__}. "
            "Expected a tool name string, a @tool()-decorated function, "
            "or an AbstractTool instance."
        )
    return result


def _normalise_program(
    program: Any,
    inline_registry: dict[str, Any],
) -> str:
    """Normalise a ``program_runner(program=...)`` argument to a tool name.

    Accepts the same shapes as :func:`_normalise_agent_tools`:

    * ``str`` — a tool name the run-time registry already knows about
      (unchanged behaviour).
    * A ``FunctionTool`` / :class:`AbstractTool` instance — typically
      what ``@tool()`` produces. Stashed in *inline_registry* by name so
      the runner can register it lazily.
    * A plain callable (undecorated ``def``) — auto-decorated via
      :func:`quartermaster_tools.auto_decorate`, then stashed the same way.

    Returns the tool's name string — which is what the engine's
    ProgramRunnerExecutor looks up at runtime. Mutates *inline_registry*
    in place when a callable was supplied.
    """
    if program is None or program == "":
        return ""
    if isinstance(program, str):
        return program
    try:
        from quartermaster_tools import (
            auto_decorate,
            is_quartermaster_tool,
        )
    except ImportError as exc:  # pragma: no cover — quartermaster-tools is a hard dep
        raise ImportError(
            "quartermaster_tools is required to pass a callable to "
            "program_runner(program=...). Install quartermaster-tools or "
            "pass the tool's name string instead."
        ) from exc

    if is_quartermaster_tool(program):
        tool_name = program.name()
        inline_registry[tool_name] = program
        return tool_name
    if callable(program):
        try:
            wrapped = auto_decorate(program)
        except ValueError as exc:
            raise ValueError(
                f"program_runner(program={program!r}) — function is not "
                f"@tool()-decorated and cannot be auto-decorated ({exc}). "
                "Either decorate it with @tool() or pass the tool's name "
                "string instead."
            ) from exc
        tool_name = wrapped.name()
        inline_registry[tool_name] = wrapped
        return tool_name
    raise TypeError(
        f"program_runner(program={program!r}) — unsupported type "
        f"{type(program).__name__}. Expected a tool name string, a "
        "@tool()-decorated function, or an AbstractTool instance."
    )


def _inline_subgraph(
    sub_graph: GraphSpec | GraphBuilder,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    connect_from_id: UUID | None,
    advance_position: Callable[[], NodePosition],
) -> UUID | None:
    """Inline a sub-graph's nodes/edges, skipping START and END nodes.

    Returns the last non-END node ID of the inlined sub-graph (the new
    "current node" for further chaining), or the original *connect_from_id*
    if the sub-graph was effectively empty.

    Accepts either an ``GraphSpec`` or a ``GraphBuilder`` instance.
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
            edges.append(GraphEdge(source_id=src, target_id=tgt, label=e.label, is_main=e.is_main))

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
        self._graph = graph  # root GraphBuilder — owns all nodes/edges
        self._parent = parent  # immediate parent (GraphBuilder or _BranchBuilder)
        self._label = label
        self._last_node_id: UUID | None = None
        self._decision_node_id: UUID | None = None
        self._branch_endpoints: list[UUID] = []
        # Set by the creating on() method so use() can detect the first call
        self._origin_decision_id: UUID | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_node(self, node: GraphNode, **flow_config: Any) -> _BranchBuilder:
        _apply_flow_config(node, flow_config)
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
            self._graph._edges.append(GraphEdge(source_id=ep, target_id=target_node.id))
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
        model: str = "",
        provider: str = "",
        temperature: float = 0.5,
        system_instruction: str = "",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an Instruction node — pure LLM text generation, NO tools.

        Streams the LLM response. Use ``agent()`` instead if you need tool use.

        Args:
            retry: Optional v0.7.0 node-level retry spec. A dict with
                ``max_attempts`` (int) and optional ``on`` (predicate
                ``(NodeResult) -> bool``). Only the int budget survives
                ``to_json``/``from_json`` round-trips — the predicate is
                held in memory on the builder and re-attached to the
                engine by the SDK runner.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.INSTRUCTION, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._graph._retry_predicates)
        return self._add_node(node)

    def instruction_form(
        self,
        name: str,
        schema: type | dict | None = None,
        model: str = "",
        provider: str = "",
        temperature: float = 0.1,
        system_instruction: str = "",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an InstructionForm node — LLM returns typed JSON.

        The schema is injected into the system prompt so the LLM knows
        the target shape. The executor validates the response and stores
        the parsed dict in ``NodeResult.data["parsed"]``.

        Args:
            schema: Pydantic BaseModel subclass or dict JSON Schema.
            retry: Optional v0.7.0 node-level retry spec — see
                :meth:`instruction` for the exact shape.
        """
        import json as _json

        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}

        schema_json = ""
        if schema is not None:
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema_json = _json.dumps(schema.model_json_schema(), separators=(",", ":"))
            elif isinstance(schema, dict):
                schema_json = _json.dumps(schema, separators=(",", ":"))

        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["schema_json"] = schema_json
        meta["schema_class"] = (
            f"{schema.__module__}.{schema.__qualname__}" if isinstance(schema, type) else ""
        )

        node = GraphNode(
            type=NodeType.INSTRUCTION_FORM,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._graph._retry_predicates)
        return self._add_node(node)

    def summarize(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        temperature: float = 0.5,
        system_instruction: str = "Summarize the given conversation concisely.",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Summarize node — LLM condenses conversation history."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.SUMMARIZE, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def agent(
        self,
        name: str = "Agent",
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        tools: list[Any] | None = None,
        max_iterations: int = 25,
        tool_scope: str = "strict",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an Agent node — autonomous agentic loop with optional tools.

        With no tools the node degenerates to a plain text completion against
        the configured provider, so ``.agent()`` works bare on any branch
        with no positional args.

        Args:
            tools: Mix of tool name strings, ``@tool()``-decorated
                callables, or :class:`AbstractTool` instances. Callables
                are auto-registered into a per-run tool registry by the
                SDK runner. Empty / unset → single-shot text generation,
                no loop.
            max_iterations: Maximum loop iterations before forced stop
                (only relevant when *tools* is non-empty).
            tool_scope: ``"strict"`` (default, v0.4.0 behaviour) — the
                model can ONLY call tools listed in *tools*; a
                hallucinated out-of-list name returns a structured error
                to the model so it can correct itself.
                ``"permissive"`` — the legacy pre-v0.4.0 behaviour where
                every tool registered on the shared registry is reachable
                from this node; intended as a migration escape hatch.
            retry: Optional v0.7.0 node-level retry spec — see
                :meth:`instruction` for the exact shape.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["program_version_ids"] = _normalise_agent_tools(tools, self._graph._inline_tools)
        meta["max_iterations"] = max_iterations
        meta["tool_scope"] = tool_scope
        node = GraphNode(
            type=NodeType.AGENT, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._graph._retry_predicates)
        return self._add_node(node)

    def instruction_program(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an InstructionProgram node — LLM emits tool calls in one round.

        Unlike ``agent()`` which loops (call tools → feed results → call
        again), this node makes ONE LLM call that produces tool-call
        parameters. The tools execute, and the combined result is the
        node's output. Use this when the LLM should decide WHICH tool
        to call and with what arguments, but you don't need a multi-turn
        reasoning loop.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            **kwargs,
        )
        if tools:
            meta["program_version_ids"] = _normalise_agent_tools(tools, self._graph._inline_tools)
        node = GraphNode(
            type=NodeType.INSTRUCTION_PROGRAM,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_program_form(
        self,
        name: str,
        parameters: list[dict] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a UserProgramForm node — form that collects tool parameters from the user.

        Combines a user form (structured input fields) with tool
        invocation: the user fills in the parameters, the tools execute
        with those values.

        Args:
            parameters: Form field definitions (same as ``user_form``).
            tools: Tools to invoke with the collected parameters.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta: dict[str, Any] = {"parameters": parameters or []}
        if tools:
            meta["program_version_ids"] = _normalise_agent_tools(tools, self._graph._inline_tools)
        meta.update(kwargs)
        node = GraphNode(
            type=NodeType.USER_PROGRAM_FORM,
            name=name,
            metadata=meta,
            message_type=MessageType.USER,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def view_metadata(self, name: str = "View Metadata", **kwargs: Any) -> _BranchBuilder:
        """Add a ViewMetadata node — debug/inspection node that exposes flow state.

        Outputs the current flow memory, node metadata, and conversation
        history as text. Useful for debugging graph execution.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.VIEW_METADATA,
            name=name,
            metadata=dict(kwargs),
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def vision(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add an image vision/analysis node."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            vision=True,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.INSTRUCTION_IMAGE_VISION,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # User interaction nodes
    # ------------------------------------------------------------------

    def user(
        self, name: str = "User Input", prompts: list[str] | None = None, **kwargs: Any
    ) -> _BranchBuilder:
        """Add a User input node — pauses flow and awaits user response.

        Args:
            prompts: Optional text snippets to show the user.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta: dict[str, Any] = {}
        if prompts:
            meta["text_snippets"] = prompts
        node = GraphNode(
            type=NodeType.USER, name=name, metadata=meta, message_type=MessageType.USER
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_form(
        self,
        name: str,
        parameters: list[dict] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Show a structured form to the user — pauses flow until submitted.

        Args:
            parameters: List of form field definitions.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.USER_FORM,
            name=name,
            metadata={"parameters": parameters or []},
            message_type=MessageType.USER,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Data nodes
    # ------------------------------------------------------------------

    def static(self, name: str, text: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a Static node — outputs fixed text content, NO LLM."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.STATIC,
            name=name,
            metadata={"static_text": text},
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def text(self, name: str, template: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a Text node — renders Jinja2 template using thought metadata."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.TEXT,
            name=name,
            metadata={"text": template},
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def var(
        self, name: str, variable: str = "", expression: str = "", **kwargs: Any
    ) -> _BranchBuilder:
        """Add a Var node — evaluates Python expression, stores result in metadata.

        Args:
            variable: Name of the variable to create.
            expression: Python expression to evaluate.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.VAR,
            name=name,
            metadata={"name": variable, "expression": expression},
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def code(self, name: str, code: str = "", filename: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a Code node — code execution (handled by runtime environment)."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.CODE,
            name=name,
            metadata={"code": code, "filename": filename},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def text_to_variable(
        self, name: str, variable: str = "", source: str = "", **kwargs: Any
    ) -> _BranchBuilder:
        """Convert text output to a variable."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.TEXT_TO_VARIABLE,
            name=name,
            metadata={"variable": variable, "source": source},
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def program_runner(self, name: str, program: Any = "", **kwargs: Any) -> _BranchBuilder:
        """Run a program/tool inline.

        *program* accepts either:

        * a tool-name string ("web_scraper") — classic behaviour, resolved
          against the run-time tool registry, or
        * a ``@tool()``-decorated function — auto-registered into the
          builder's inline-tools so the SDK runner can dispatch it without
          a manual ``get_default_registry().register(...)`` call.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        program_name = _normalise_program(program, self._graph._inline_tools)
        node = GraphNode(
            type=NodeType.PROGRAM_RUNNER,
            name=name,
            metadata={"program": program_name, **kwargs},
            message_type=MessageType.TOOL,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Control flow (nested)
    # ------------------------------------------------------------------

    def if_node(self, name: str, expression: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add an IF node — evaluates Python expression, picks true/false branch.

        NO LLM call. Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.IF,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata={"if_expression": expression},
        )
        _apply_flow_config(node, flow_cfg)
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def decision(
        self,
        name: str,
        model: str = "",
        provider: str = "",
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        node = GraphNode(
            type=NodeType.DECISION,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata=meta,
        )
        _apply_flow_config(node, flow_cfg)
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def static_decision(self, name: str, expression: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a StaticDecision node — expression-based branching, NO LLM.

        Like ``if_node()`` but uses ``StaticDecision1`` node type.
        Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.STATIC_DECISION,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata={"expression": expression},
        )
        _apply_flow_config(node, flow_cfg)
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def user_decision(self, name: str, **kwargs: Any) -> _BranchBuilder:
        """Add a UserDecision node — presents choices to user, user picks path.

        Waits for ALL incoming branches (traverse_in=AwaitAll), then
        pauses flow until the user selects a path.

        Use ``.on(label)`` for each option.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.USER_DECISION,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def on(self, label: str) -> _BranchBuilder:
        """Start a sub-branch for a decision/if option inside this branch."""
        if self._decision_node_id is None:
            raise ValueError(
                "on() must be called after if_node(), decision(), static_decision(), or user_decision()"
            )
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
                self._graph._edges.append(GraphEdge(source_id=sub._last_node_id, target_id=node.id))
            sub._last_node_id = node.id
            sub._add_node = original_add  # type: ignore[method-assign]
            return sub

        sub._add_node = labeled_add  # type: ignore[method-assign]
        return sub

    def switch(
        self,
        name: str,
        cases: list[dict[str, str]] | None = None,
        default_edge_id: str = "",
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Add a Switch node — evaluates multiple cases, first match wins. NO LLM.

        Args:
            cases: List of ``{"expression": "...", "edge_id": "..."}`` dicts.
            default_edge_id: Fallback edge ID if no case matches.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta: dict[str, Any] = {
            "cases": cases or [],
            "default_edge_id": default_edge_id,
        }
        node = GraphNode(
            type=NodeType.SWITCH,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata=meta,
        )
        _apply_flow_config(node, flow_cfg)
        self._add_node(node)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def break_node(
        self, name: str = "Break", targets: list[str] | None = None, **kwargs: Any
    ) -> _BranchBuilder:
        """Add a Break node — stops backward message collection.

        Args:
            targets: What to clear: ``[]`` (full break), ``['tools']``, ``['thinking']``.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.BREAK,
            name=name,
            metadata={"break_targets": targets or []},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.SYSTEM,
        )
        _apply_flow_config(node, flow_cfg)
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
        self,
        name: str = "Merge",
        model: str = "",
        provider: str = "",
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        merge_node = GraphNode(
            type=NodeType.MERGE,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            message_type=MessageType.ASSISTANT,
            position=self._graph._advance_position(),
            metadata=meta,
        )
        _apply_flow_config(merge_node, flow_cfg)
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

    def static_merge(self, name: str = "Merge", text: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a StaticMerge node — combines branches WITHOUT LLM.

        Waits for ALL branches (traverse_in=AwaitAll), appends static text,
        then continues. No LLM call involved.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        merge_node = GraphNode(
            type=NodeType.STATIC_MERGE,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._graph._advance_position(),
            metadata={"static_text": text},
        )
        _apply_flow_config(merge_node, flow_cfg)
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
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variable_names: list[str] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Read variables from persistent memory into thought metadata.

        Args:
            memory_name: Name of the memory store.
            memory_type: ``"flow"`` (scoped to this flow) or ``"user"`` (persists across flows).
            variable_names: Which variables to load.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.READ_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variable_names": variable_names or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def write_memory(
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Write variables from thought metadata to persistent memory.

        Args:
            memory_name: Name of the memory store.
            memory_type: ``"flow"`` or ``"user"``.
            variables: List of ``{"name": "...", "expression": "..."}`` dicts.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.WRITE_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def update_memory(
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Update existing persistent memory variables."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.UPDATE_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def flow_memory(
        self,
        name: str = "Flow Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Define flow-scoped persistent memory (not connected to flow edges).

        This is a *definition* node — the Start node initialises it at
        runtime via ``_memory_initializer``.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.FLOW_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_memory(
        self,
        name: str = "User Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> _BranchBuilder:
        """Define user-scoped persistent memory (survives across flow executions)."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.USER_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Utility nodes
    # ------------------------------------------------------------------

    def comment(self, name: str, text: str = "", **kwargs: Any) -> _BranchBuilder:
        """Add a Comment node — documentation only, no runtime logic."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.COMMENT,
            name=name,
            metadata={"comment": text},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def sub_agent(self, name: str, graph_id: str = "", **kwargs: Any) -> _BranchBuilder:
        """Call another agent graph synchronously (blocks until sub-graph completes).

        This is different from ``spawn_agent`` (session tool) which runs agents
        in background sessions. Sub-agent nodes execute inline and return
        their result to the current flow.

        Args:
            graph_id: ID of the agent graph to execute as a sub-flow.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.SUB_ASSISTANT,
            name=name,
            metadata={"sub_assistant_id": graph_id},
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(type=node_type, name=name, metadata=metadata or {}, **kwargs)
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def use(self, sub_graph: GraphSpec | GraphBuilder) -> _BranchBuilder:
        """Inline a sub-graph into this branch.

        Copies all nodes except START/END from the sub-graph, remaps their
        IDs, and connects them into the current branch chain.

        Accepts either an ``GraphSpec`` or a ``GraphBuilder`` instance.
        """
        # Detect whether this is the first call after on() — need to label the edge
        is_first = (
            self._origin_decision_id is not None and self._last_node_id == self._origin_decision_id
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
    # Explicit wiring
    # ------------------------------------------------------------------

    def connect(self, from_name: str, to_name: str, label: str = "") -> _BranchBuilder:
        """Create an edge between two nodes by name."""
        node_map = {n.name: n for n in self._graph._nodes}
        source = node_map.get(from_name)
        target = node_map.get(to_name)
        if not source:
            raise ValueError(f"Node '{from_name}' not found")
        if not target:
            raise ValueError(f"Node '{to_name}' not found")
        self._graph._edges.append(GraphEdge(source_id=source.id, target_id=target.id, label=label))
        return self

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def end(self) -> Any:
        """End this branch.

        v0.3.1 restores the pre-v0.3.0 behaviour: ``.end()`` on a
        branch REGISTERS the branch's last node as a pending endpoint
        so the next outer node (typically the outer ``.end()`` or a
        ``.merge()``) automatically wires every branch to it.  No End
        node is appended inside the branch — the outer End handles
        termination for all branches uniformly.

        For an explicit branch-terminal that stops the whole flow
        immediately (instead of merging), add a terminal node yourself
        (e.g. a ``.text(...)``-like terminal, or fall through to the
        outer ``.end()``).  For loops, use :meth:`back` instead.

        Returns the parent — either a ``GraphBuilder`` or another
        ``_BranchBuilder`` for nested control flow.
        """
        if self._last_node_id is not None:
            self._parent._branch_endpoints.append(self._last_node_id)
        return self._parent

    def back(self) -> Any:
        """Emit a Back node on this branch — "loop / return to parent".

        Appends a ``NodeType.BACK`` node to this branch and does NOT
        register a pending endpoint for auto-merge.  The runner will
        dispatch Start (main graph) or hand control back to the parent
        flow (sub-graph) when execution reaches this node.

        Use this on the "keep looping" arm of a decision/IF, and
        ``.end()`` on the "we're done" arm.
        """
        back_node = GraphNode(
            type=NodeType.BACK,
            name="Back",
            traverse_out=TraverseOut.SPAWN_NONE,
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        # Route through ``_add_node`` so .on(label)'s labelling hook
        # applies — ``.on("false").back()`` as the FIRST call on a
        # branch must label the decision → Back edge with "false".
        self._add_node(back_node)
        # Don't register as a branch endpoint — Back is terminal for this branch.
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
    ``GraphSpec``.

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

    def __init__(self, name: str, description: str = "", *, auto_start: bool = True) -> None:
        """Build a new graph.

        Args:
            name: Human-readable graph name.
            description: Optional description stored on the spec.
            auto_start: When ``True`` (the default, v0.2.0+) a ``Start``
                node is added automatically so chains can open with
                ``Graph("x").user()...`` instead of ``.start().user()...``.
                Set to ``False`` only if you want to construct the start
                yourself (rare — legacy compatibility path).
        """
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
        # v0.4.0: side-channel dict of {tool_name: AbstractTool-like instance}
        # populated by ``.agent(tools=[...])`` when callables are passed
        # inline. The SDK runner reads this at run time and merges the
        # entries into the run-scoped tool registry — callables never land
        # in the serialisable GraphSpec (they can't survive JSON/YAML
        # round-trips), only the resolved tool NAMES do.
        self._inline_tools: dict[str, Any] = {}
        # v0.7.0: side-channel dict of {node_name: retry-predicate callable}
        # populated by ``.instruction(retry=...)`` / ``.agent(retry=...)`` /
        # ``.instruction_form(retry=...)``.  The engine resolves predicates
        # by name at run time.  Callable objects can't survive JSON/YAML
        # so they live only in this in-memory registry (mirrors
        # ``_inline_tools``).
        self._retry_predicates: dict[str, Any] = {}
        if auto_start:
            self._create_start_node()

    def _create_start_node(self) -> None:
        """Append the graph's ``Start`` node and mark it as the entry point.

        Internal helper used by ``__init__`` (via ``auto_start=True``) and
        by the now-idempotent :meth:`start` method so explicit callers
        don't accidentally create two Start nodes.
        """
        node = GraphNode(
            type=NodeType.START,
            name="Start",
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        self._start_node_id = node.id
        self._add_node(node)

    # ------------------------------------------------------------------
    # Agent control
    # ------------------------------------------------------------------

    def allowed_agents(self, *agent_ids: str, **kwargs: Any) -> GraphBuilder:
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
                    type=NodeType.END,
                    name="End",
                    traverse_out=TraverseOut.SPAWN_NONE,
                    thought_type=ThoughtType.SKIP,
                    message_type=MessageType.VARIABLE,
                    position=self._advance_position(),
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
            self._edges.append(GraphEdge(source_id=ep, target_id=target_node.id))
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
        """Add a Start node (idempotent since v0.2.0).

        ``Graph(name)`` auto-creates the Start node, so chained calls to
        ``.start()`` are no-ops — the method is kept only for readability
        of legacy snippets.  Explicitly creating a second Start via
        ``auto_start=False`` + a manual ``.start()`` call is still
        supported.
        """
        if self._start_node_id is not None:
            # Auto-start already happened (or the caller previously
            # called .start()) — don't create a duplicate.
            return self
        self._create_start_node()
        return self

    def end(self) -> GraphBuilder:
        """Add an End node — the graph terminates here.

        v0.3.1 semantics (reverted from v0.3.0 Proposal A):

        * In the **main graph**, reaching an End node **stops the
          flow**.  Sets ``traverse_out=SPAWN_NONE`` — the runner does
          not dispatch anything after this node.  (This restores the
          v0.2.x behaviour after the short-lived v0.3.0 "loop to
          Start" default was found to silently break every existing
          trailing-``.end()`` graph.)
        * In a **sub-graph** (spawned via a ``SUB_ASSISTANT`` node) an
          End node still signals "return to parent" — the runner
          inspects ``ExecutionContext.parent_context`` at run time.
        * For explicit "loop back to Start / return to parent" use
          :meth:`back` instead.

        If there are pending branch endpoints, auto-merges them first
        so the End node is reachable from all branches.
        """
        node = GraphNode(
            type=NodeType.END,
            name="End",
            traverse_out=TraverseOut.SPAWN_NONE,
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        return self._add_node(node)

    def back(self) -> GraphBuilder:
        """Add a Back node — explicit "loop to Start / return to parent".

        * In the **main graph**, reaching a Back node dispatches the
          graph's Start node again, producing an explicit loop.  The
          runner's ``max_loop_iterations`` safety cap (default 100)
          still applies so a broken loop fails fast instead of
          hanging.
        * In a **sub-graph** a Back node returns control to the parent
          flow, same as the sub-graph End-node behaviour.

        Use ``.back()`` on the "keep looping" arm of a decision / IF and
        ``.end()`` on the "we're done" arm.  No LLM call is made when
        a Back node executes — it's pure control flow.
        """
        node = GraphNode(
            type=NodeType.BACK,
            name="Back",
            traverse_out=TraverseOut.SPAWN_NONE,
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        return self._add_node(node)

    # ------------------------------------------------------------------
    # LLM nodes
    # ------------------------------------------------------------------

    def instruction(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        temperature: float = 0.5,
        system_instruction: str = "",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an Instruction node — pure LLM text generation, NO tools.

        Streams the LLM response. Use ``agent()`` instead if you need tool use.

        Args:
            retry: Optional v0.7.0 node-level retry spec. Shape:
                ``{"max_attempts": int, "on": (NodeResult) -> bool}``.
                Only the integer budget survives ``to_json``/``from_json``
                round-trips — the predicate is in-memory only (builder
                side-channel, re-attached by the SDK runner).
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.INSTRUCTION, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._retry_predicates)
        return self._add_node(node)

    def instruction_form(
        self,
        name: str,
        schema: type | dict | None = None,
        model: str = "",
        provider: str = "",
        temperature: float = 0.1,
        system_instruction: str = "",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an InstructionForm node — LLM returns typed JSON.

        See :meth:`_BranchBuilder.instruction_form`. Accepts the v0.7.0
        ``retry=`` spec; see :meth:`instruction` for the exact shape.
        """
        import json as _json

        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}

        schema_json = ""
        if schema is not None:
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema_json = _json.dumps(schema.model_json_schema(), separators=(",", ":"))
            elif isinstance(schema, dict):
                schema_json = _json.dumps(schema, separators=(",", ":"))

        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["schema_json"] = schema_json
        meta["schema_class"] = (
            f"{schema.__module__}.{schema.__qualname__}" if isinstance(schema, type) else ""
        )

        node = GraphNode(
            type=NodeType.INSTRUCTION_FORM,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._retry_predicates)
        return self._add_node(node)

    def summarize(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        temperature: float = 0.5,
        system_instruction: str = "Summarize the given conversation concisely.",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Summarize node — LLM condenses conversation history."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.SUMMARIZE, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def agent(
        self,
        name: str = "Agent",
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        tools: list[Any] | None = None,
        max_iterations: int = 25,
        tool_scope: str = "strict",
        retry: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an Agent node — autonomous agentic loop with optional tools.

        With no tools the node degenerates to a plain text completion against
        the configured provider, which is exactly what the simplest
        ``start → user → agent → end`` graph wants.

        Args:
            name: Display name (defaults to "Agent" so ``.agent()`` works
                bare, with no positional args).
            tools: Mix of tool name strings, ``@tool()``-decorated
                callables, or :class:`AbstractTool` instances. Callables
                are auto-registered into a per-run tool registry by the
                SDK runner; strings are resolved against the registry
                the integrator passes to ``qm.run(tool_registry=...)``
                (or the module-level default). Empty / unset → single-
                shot text generation, no loop.
            max_iterations: Maximum loop iterations before forced stop
                (only relevant when *tools* is non-empty).
            tool_scope: ``"strict"`` (default, v0.4.0 behaviour) — the
                model can ONLY call tools listed in *tools*; a
                hallucinated out-of-list name returns a structured error
                to the model so it can correct itself.
                ``"permissive"`` — the legacy pre-v0.4.0 behaviour where
                every tool registered on the shared registry is reachable
                from this node; intended as a migration escape hatch.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["program_version_ids"] = _normalise_agent_tools(tools, self._inline_tools)
        meta["max_iterations"] = max_iterations
        meta["tool_scope"] = tool_scope
        node = GraphNode(
            type=NodeType.AGENT, name=name, metadata=meta, message_type=MessageType.ASSISTANT
        )
        _apply_flow_config(node, flow_cfg)
        _apply_retry_spec(node, retry, self._retry_predicates)
        return self._add_node(node)

    def instruction_program(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an InstructionProgram node — single-round LLM tool dispatch.

        See :meth:`_BranchBuilder.instruction_program`.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            **kwargs,
        )
        if tools:
            meta["program_version_ids"] = _normalise_agent_tools(tools, self._inline_tools)
        node = GraphNode(
            type=NodeType.INSTRUCTION_PROGRAM,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_program_form(
        self,
        name: str,
        parameters: list[dict] | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a UserProgramForm node — form + tool execution.

        See :meth:`_BranchBuilder.user_program_form`.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta: dict[str, Any] = {"parameters": parameters or []}
        if tools:
            meta["program_version_ids"] = _normalise_agent_tools(tools, self._inline_tools)
        meta.update(kwargs)
        node = GraphNode(
            type=NodeType.USER_PROGRAM_FORM,
            name=name,
            metadata=meta,
            message_type=MessageType.USER,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def view_metadata(self, name: str = "View Metadata", **kwargs: Any) -> GraphBuilder:
        """Add a ViewMetadata debug node. See :meth:`_BranchBuilder.view_metadata`."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.VIEW_METADATA,
            name=name,
            metadata=dict(kwargs),
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def vision(
        self,
        name: str,
        model: str = "",
        provider: str = "",
        system_instruction: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add an image vision/analysis node (INSTRUCTION_IMAGE_VISION)."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            system_instruction=system_instruction,
            vision=True,
            **kwargs,
        )
        node = GraphNode(
            type=NodeType.INSTRUCTION_IMAGE_VISION,
            name=name,
            metadata=meta,
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def decision(
        self,
        name: str,
        model: str = "",
        provider: str = "",
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        node = GraphNode(
            type=NodeType.DECISION,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata=meta,
        )
        _apply_flow_config(node, flow_cfg)
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

    def static_decision(self, name: str, expression: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a StaticDecision node — expression-based branching, NO LLM.

        Evaluates a Python expression and picks the true/false branch.
        Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.STATIC_DECISION,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata={"expression": expression},
        )
        _apply_flow_config(node, flow_cfg)
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def switch(
        self,
        name: str,
        cases: list[dict[str, str]] | None = None,
        default_edge_id: str = "",
        **kwargs: Any,
    ) -> GraphBuilder:
        """Add a Switch node — evaluates multiple cases, first match wins. NO LLM.

        Args:
            cases: List of ``{"expression": "...", "edge_id": "..."}`` dicts.
            default_edge_id: Fallback edge ID if no case matches.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        meta: dict[str, Any] = {
            "cases": cases or [],
            "default_edge_id": default_edge_id,
        }
        node = GraphNode(
            type=NodeType.SWITCH,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata=meta,
        )
        _apply_flow_config(node, flow_cfg)
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
            raise ValueError(
                "on() must be called after decision(), if_node(), static_decision(), or user_decision()"
            )
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

    def if_node(self, name: str, expression: str = "", **kwargs: Any) -> GraphBuilder:
        """Add an If node — evaluates Python expression, picks true/false branch.

        NO LLM call. Use ``.on("true")`` / ``.on("false")`` for branches.

        Args:
            expression: Python expression that evaluates to truthy/falsy.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.IF,
            name=name,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
            metadata={"if_expression": expression},
        )
        _apply_flow_config(node, flow_cfg)
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def break_node(
        self, name: str = "Break", targets: list[str] | None = None, **kwargs: Any
    ) -> GraphBuilder:
        """Add a Break node — stops backward message collection.

        Args:
            targets: What to clear: ``[]`` (full break), ``['tools']``, ``['thinking']``.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.BREAK,
            name=name,
            metadata={"break_targets": targets or []},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.SYSTEM,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Data nodes
    # ------------------------------------------------------------------

    def static(self, name: str, text: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a Static node — outputs fixed text content, NO LLM."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.STATIC,
            name=name,
            metadata={"static_text": text},
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def code(self, name: str, code: str = "", filename: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a Code node — code execution (handled by runtime environment)."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.CODE,
            name=name,
            metadata={"code": code, "filename": filename},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def var(
        self, name: str, variable: str = "", expression: str = "", **kwargs: Any
    ) -> GraphBuilder:
        """Add a Var node — evaluates Python expression, stores result in metadata.

        Args:
            variable: Name of the variable to create.
            expression: Python expression to evaluate.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.VAR,
            name=name,
            metadata={"name": variable, "expression": expression},
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def text(self, name: str, template: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a Text node — renders Jinja2 template using thought metadata."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.TEXT,
            name=name,
            metadata={"text": template},
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def text_to_variable(
        self, name: str, variable: str = "", source: str = "", **kwargs: Any
    ) -> GraphBuilder:
        """Convert text output to a variable."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.TEXT_TO_VARIABLE,
            name=name,
            metadata={"variable": variable, "source": source},
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def program_runner(self, name: str, program: Any = "", **kwargs: Any) -> GraphBuilder:
        """Run a program/tool inline.

        *program* accepts either a tool-name string or a ``@tool()``-
        decorated function — callables are auto-registered into the
        builder's inline-tools so the SDK runner can dispatch them
        without a manual ``get_default_registry().register(...)`` call.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        program_name = _normalise_program(program, self._inline_tools)
        node = GraphNode(
            type=NodeType.PROGRAM_RUNNER,
            name=name,
            metadata={"program": program_name, **kwargs},
            message_type=MessageType.TOOL,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Memory nodes
    # ------------------------------------------------------------------

    def read_memory(
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variable_names: list[str] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Read variables from persistent memory into thought metadata."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.READ_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variable_names": variable_names or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def write_memory(
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Write variables from thought metadata to persistent memory."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.WRITE_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def update_memory(
        self,
        name: str,
        memory_name: str = "",
        memory_type: str = "flow",
        variables: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Update existing persistent memory variables."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.UPDATE_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "memory_type": memory_type,
                "variables": variables or [],
            },
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def flow_memory(
        self,
        name: str = "Flow Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Define flow-scoped persistent memory."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.FLOW_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_memory(
        self,
        name: str = "User Memory",
        memory_name: str = "",
        initial_data: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Define user-scoped persistent memory (survives across flow executions)."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.USER_MEMORY,
            name=name,
            metadata={
                "memory_name": memory_name,
                "initial_data": initial_data or [],
            },
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # User interaction nodes
    # ------------------------------------------------------------------

    def user(
        self, name: str = "User Input", prompts: list[str] | None = None, **kwargs: Any
    ) -> GraphBuilder:
        """Add a User input node — pauses flow and awaits user response.

        Args:
            prompts: Optional text snippets to show the user.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta: dict[str, Any] = {}
        if prompts:
            meta["text_snippets"] = prompts
        node = GraphNode(
            type=NodeType.USER, name=name, metadata=meta, message_type=MessageType.USER
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def user_decision(self, name: str, **kwargs: Any) -> GraphBuilder:
        """Add a UserDecision node — presents choices to user, user picks path.

        Waits for ALL incoming branches (traverse_in=AwaitAll), then
        pauses flow until the user selects a path.

        Use ``.on(label)`` for each option.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        self._auto_merge_if_needed()
        node = GraphNode(
            type=NodeType.USER_DECISION,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            traverse_out=TraverseOut.SPAWN_PICKED,
            thought_type=ThoughtType.USE_PREVIOUS,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        node.position = self._advance_position()
        self._nodes.append(node)
        if self._last_node_id is not None:
            edge = GraphEdge(source_id=self._last_node_id, target_id=node.id)
            self._edges.append(edge)
        self._decision_node_id = node.id
        self._last_node_id = None
        return self

    def user_form(
        self,
        name: str,
        parameters: list[dict] | None = None,
        **kwargs: Any,
    ) -> GraphBuilder:
        """Show a structured form to the user — pauses flow until submitted.

        Args:
            parameters: List of form field definitions.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.USER_FORM,
            name=name,
            metadata={"parameters": parameters or []},
            message_type=MessageType.USER,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Utility nodes
    # ------------------------------------------------------------------

    def comment(self, name: str, text: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a Comment node — documentation only, no runtime logic."""
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.COMMENT,
            name=name,
            metadata={"comment": text},
            thought_type=ThoughtType.SKIP,
            message_type=MessageType.VARIABLE,
        )
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    # ------------------------------------------------------------------
    # Composition / sub-graphs
    # ------------------------------------------------------------------

    def sub_agent(self, name: str, graph_id: str = "", **kwargs: Any) -> GraphBuilder:
        """Call another agent graph synchronously (blocks until sub-graph completes).

        This is different from ``spawn_agent`` (session tool) which runs agents
        in background sessions. Sub-agent nodes execute inline and return
        their result to the current flow.

        Args:
            graph_id: ID of the agent graph to execute as a sub-flow.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(
            type=NodeType.SUB_ASSISTANT,
            name=name,
            metadata={"sub_assistant_id": graph_id},
            message_type=MessageType.ASSISTANT,
        )
        _apply_flow_config(node, flow_cfg)
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
        self,
        name: str = "Merge",
        model: str = "",
        provider: str = "",
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        meta = _llm_meta(
            model=model,
            provider=provider,
            temperature=temperature,
            system_instruction=system_instruction,
            **kwargs,
        )
        meta["prefix_message"] = prefix_message
        meta["suffix_message"] = suffix_message
        merge_node = GraphNode(
            type=NodeType.MERGE,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            message_type=MessageType.ASSISTANT,
            position=self._advance_position(),
            metadata=meta,
        )
        _apply_flow_config(merge_node, flow_cfg)
        self._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._edges.append(GraphEdge(source_id=self._last_node_id, target_id=merge_node.id))
        self._last_node_id = merge_node.id
        return self

    def static_merge(self, name: str = "Merge", text: str = "", **kwargs: Any) -> GraphBuilder:
        """Add a StaticMerge node — combines branches WITHOUT LLM.

        Waits for ALL branches (traverse_in=AwaitAll), appends static text,
        then continues. No LLM call involved.
        """
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        merge_node = GraphNode(
            type=NodeType.STATIC_MERGE,
            name=name,
            traverse_in=TraverseIn.AWAIT_ALL,
            position=self._advance_position(),
            metadata={"static_text": text},
        )
        _apply_flow_config(merge_node, flow_cfg)
        self._nodes.append(merge_node)
        for ep in self._branch_endpoints:
            self._edges.append(GraphEdge(source_id=ep, target_id=merge_node.id))
        self._branch_endpoints.clear()
        if self._last_node_id is not None:
            self._edges.append(GraphEdge(source_id=self._last_node_id, target_id=merge_node.id))
        self._last_node_id = merge_node.id
        return self

    def use(self, sub_graph: GraphSpec | GraphBuilder) -> GraphBuilder:
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
        flow_cfg = {k: kwargs.pop(k) for k in list(kwargs) if k in _ALL_CONFIG_KEYS}
        node = GraphNode(type=node_type, name=name, metadata=metadata or {}, **kwargs)
        _apply_flow_config(node, flow_cfg)
        return self._add_node(node)

    def connect(self, from_name: str, to_name: str, label: str = "") -> GraphBuilder:
        """Create an edge between two nodes by name."""
        node_map = {n.name: n for n in self._nodes}
        source = node_map.get(from_name)
        target = node_map.get(to_name)
        if not source:
            raise ValueError(f"Node '{from_name}' not found")
        if not target:
            raise ValueError(f"Node '{to_name}' not found")
        self._edges.append(GraphEdge(source_id=source.id, target_id=target.id, label=label))
        return self

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

    def to_graph(
        self,
        validate: bool = True,
        agent_id: UUID | None = None,
    ) -> GraphSpec:
        """Export the graph as an ``GraphSpec``."""
        if self._start_node_id is None:
            raise ValueError("Graph has no start node — call .start() first")
        self._finalize()
        ver = GraphSpec(
            id=uuid4(),
            agent_id=agent_id or uuid4(),
            nodes=list(self._nodes),
            edges=list(self._edges),
            start_node_id=self._start_node_id,
            created_at=datetime.now(timezone.utc),
        )
        if validate:
            validate_graph(ver)
        # v0.4.0: transfer the builder's inline-tools side-channel onto
        # the built spec via a private attribute.  ``GraphSpec`` is a
        # Pydantic model so we can't add this as a declared field (it
        # would serialise to JSON, which callables can't survive); using
        # ``object.__setattr__`` keeps the attribute out of ``model_dump``
        # yet accessible to the SDK runner via ``getattr(spec, "_inline_tools")``.
        if self._inline_tools:
            object.__setattr__(ver, "_inline_tools", dict(self._inline_tools))
        # v0.7.0: retry predicates follow the same side-channel pattern as
        # inline tools — callables can't survive JSON/YAML, so they live as
        # a non-declared attribute on the spec for the SDK runner to read.
        if self._retry_predicates:
            object.__setattr__(ver, "_retry_predicates", dict(self._retry_predicates))
        return ver

    def build(self, validate: bool = True) -> GraphSpec:
        """Build and return the ``GraphSpec``.  Alias for ``to_graph()``."""
        return self.to_graph(validate=validate)

    def to_agent(self, validate: bool = True) -> Agent:
        """Export as a full ``Agent``."""
        self.to_graph(validate=validate)
        return Agent(
            id=uuid4(),
            name=self._name,
            description=self._description,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
