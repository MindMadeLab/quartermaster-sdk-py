"""v0.7.0 — graph-level per-node retry primitive (engine side).

The builder puts ``retry_max_attempts`` on node metadata and (optionally)
stashes an ``on=`` predicate in a side-channel dict that the SDK runner
forwards to :class:`FlowRunner` as ``retry_predicates={name: callable}``.
The engine wraps the executor invocation in a loop:

* Up to ``max_attempts`` total attempts per node execution.
* Predicate ``(NodeResult) -> bool`` gates the retry. When absent the
  default "retry on failure" kicks in (``NodeResult.success is False``).
* Exceptions count as a retry trigger regardless of predicate.
* Emits ``CustomEvent(name="node.retried")`` for each re-attempt.
* Counter is per-node-execution — not shared across sub-graph invocations.
* Non-retryable node types (e.g. ``START``) ignore the metadata.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from quartermaster_engine import FlowRunner
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import CustomEvent, FlowEvent
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.types import (
    GraphSpec,
    GraphEdge,
    GraphNode,
    NodeType,
    ThoughtType,
    MessageType,
    TraverseOut,
)


# ── Test doubles ────────────────────────────────────────────────────


class _ScriptedExecutor:
    """Executor whose ``execute()`` returns / raises from a scripted list.

    Each call pops the next element:

    * A :class:`NodeResult` is returned as-is.
    * An :class:`Exception` instance is raised.
    * A bare string is converted to a successful ``NodeResult(output_text=...)``.

    Tracks ``call_count`` for assertions.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.call_count = 0

    async def execute(self, context: ExecutionContext) -> NodeResult:
        self.call_count += 1
        if not self._script:
            # Shouldn't happen in tests — surface loudly rather than looping.
            raise AssertionError("scripted executor ran out of responses")
        next_item = self._script.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        if isinstance(next_item, NodeResult):
            return next_item
        return NodeResult(success=True, data={}, output_text=str(next_item))


def _build_graph(
    node_type: NodeType = NodeType.INSTRUCTION,
    node_name: str = "Target",
    retry_max_attempts: int | None = None,
) -> tuple[GraphSpec, GraphNode]:
    """Tiny graph: Start → Target → End.  Only the ``Target`` node is
    interesting; the Start / End nodes pass through unchanged.

    Returns ``(spec, target_node)`` so tests can reference the node's
    UUID when scripting registries / asserting captures.
    """
    start = GraphNode(
        id=uuid4(),
        type=NodeType.START,
        name="Start",
        thought_type=ThoughtType.SKIP,
        message_type=MessageType.VARIABLE,
    )
    metadata: dict[str, Any] = {}
    if retry_max_attempts is not None:
        metadata["retry_max_attempts"] = retry_max_attempts
    target = GraphNode(
        id=uuid4(),
        type=node_type,
        name=node_name,
        metadata=metadata,
        message_type=MessageType.ASSISTANT,
    )
    end = GraphNode(
        id=uuid4(),
        type=NodeType.END,
        name="End",
        traverse_out=TraverseOut.SPAWN_NONE,
        thought_type=ThoughtType.SKIP,
        message_type=MessageType.VARIABLE,
    )
    edges = [
        GraphEdge(source_id=start.id, target_id=target.id),
        GraphEdge(source_id=target.id, target_id=end.id),
    ]
    spec = GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=start.id,
        nodes=[start, target, end],
        edges=edges,
    )
    return spec, target


def _make_runner(
    spec: GraphSpec,
    executor: _ScriptedExecutor,
    target_type: NodeType,
    events: list[FlowEvent] | None = None,
    retry_predicates: dict[str, Any] | None = None,
) -> FlowRunner:
    """Build a runner with just the executor we care about + passthroughs
    for the non-interesting node types."""
    from quartermaster_engine.example_runner import PassthroughExecutor

    reg = SimpleNodeRegistry()
    reg.register(target_type.value, executor)
    # Cover the other node types the test graph uses so the runner
    # doesn't bail on "no executor registered" before reaching Target.
    reg.register(NodeType.START.value, PassthroughExecutor())
    reg.register(NodeType.END.value, PassthroughExecutor())

    on_event = None
    if events is not None:

        def _capture(ev: FlowEvent) -> None:
            events.append(ev)

        on_event = _capture

    return FlowRunner(
        graph=spec,
        node_registry=reg,
        on_event=on_event,
        retry_predicates=retry_predicates,
    )


# ── Default predicate: retry on success=False ────────────────────────


def test_failing_then_succeeding_produces_success_on_second_attempt() -> None:
    """``retry_max_attempts=3`` + fail, then succeed → success, 2 calls."""
    spec, target = _build_graph(retry_max_attempts=3)
    executor = _ScriptedExecutor(
        [
            NodeResult(success=False, data={}, error="transient"),
            NodeResult(success=True, data={}, output_text="recovered"),
        ]
    )
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION)
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 2
    # The successful attempt's output flows to the End node.
    assert "recovered" in fr.final_output


def test_always_failing_exhausts_budget_exactly_max_attempts_times() -> None:
    """Always-failing executor with budget=3 → 3 calls, final NodeResult.success=False."""
    spec, target = _build_graph(retry_max_attempts=3)
    executor = _ScriptedExecutor(
        [
            NodeResult(success=False, data={}, error="boom-1"),
            NodeResult(success=False, data={}, error="boom-2"),
            NodeResult(success=False, data={}, error="boom-3"),
        ]
    )
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION)
    fr = runner.run("go")
    assert executor.call_count == 3
    assert fr.success is False
    # The final error surfaces via the flow-level error string.
    assert fr.error is not None
    assert "boom-3" in fr.error


def test_no_retry_spec_runs_node_exactly_once() -> None:
    """No ``retry_max_attempts`` metadata → legacy single-call behaviour."""
    spec, target = _build_graph(retry_max_attempts=None)
    executor = _ScriptedExecutor([NodeResult(success=True, data={}, output_text="once")])
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION)
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 1


def test_retry_emits_node_retried_custom_event_with_attempt_and_reason() -> None:
    """Each retry fires a CustomEvent(name='node.retried') with ``attempt`` and ``reason``."""
    spec, target = _build_graph(retry_max_attempts=3)
    executor = _ScriptedExecutor(
        [
            NodeResult(success=False, data={}, error="first"),
            NodeResult(success=False, data={}, error="second"),
            NodeResult(success=True, data={}, output_text="ok"),
        ]
    )
    events: list[FlowEvent] = []
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION, events=events)
    fr = runner.run("go")
    assert fr.success is True, fr.error

    retries = [ev for ev in events if isinstance(ev, CustomEvent) and ev.name == "node.retried"]
    # 3 attempts → 2 retries.
    assert len(retries) == 2
    # Attempt counter starts at 1 for the first retry, 2 for the second.
    assert retries[0].payload["attempt"] == 1
    assert retries[1].payload["attempt"] == 2
    # All retries reference the target node by name.
    for ev in retries:
        assert ev.payload["node"] == "Target"
        assert ev.payload["reason"] == "predicate"
        assert ev.node_id == target.id


# ── Custom predicate ─────────────────────────────────────────────────


def test_predicate_based_retry_runs_once_then_stops() -> None:
    """Predicate ``on=lambda r: "retry me" in r.output_text`` re-runs when
    the first result has that text and stops when the second doesn't."""
    spec, target = _build_graph(retry_max_attempts=5)
    executor = _ScriptedExecutor(
        [
            NodeResult(success=True, data={}, output_text="please retry me"),
            NodeResult(success=True, data={}, output_text="final answer"),
        ]
    )

    def _predicate(capture: NodeResult) -> bool:
        return "retry me" in (capture.output_text or "")

    runner = _make_runner(
        spec,
        executor,
        NodeType.INSTRUCTION,
        retry_predicates={"Target": _predicate},
    )
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 2
    assert "final answer" in fr.final_output


def test_predicate_never_triggers_runs_once() -> None:
    """Predicate that never returns True → single call even with budget=5."""
    spec, target = _build_graph(retry_max_attempts=5)
    executor = _ScriptedExecutor([NodeResult(success=True, data={}, output_text="done")])
    runner = _make_runner(
        spec,
        executor,
        NodeType.INSTRUCTION,
        retry_predicates={"Target": lambda r: False},
    )
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 1


def test_exception_path_retries_with_reason_exception() -> None:
    """An executor exception counts as a retry trigger; reason='exception'."""
    spec, target = _build_graph(retry_max_attempts=3)
    executor = _ScriptedExecutor(
        [
            RuntimeError("network glitch"),
            NodeResult(success=True, data={}, output_text="recovered"),
        ]
    )
    events: list[FlowEvent] = []
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION, events=events)
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 2

    retries = [ev for ev in events if isinstance(ev, CustomEvent) and ev.name == "node.retried"]
    assert len(retries) == 1
    assert retries[0].payload["reason"] == "exception"


# ── Agent node ───────────────────────────────────────────────────────


def test_agent_node_retries_on_predicate_match() -> None:
    """Same contract applies to ``NodeType.AGENT``."""
    spec, target = _build_graph(node_type=NodeType.AGENT, retry_max_attempts=2)
    executor = _ScriptedExecutor(
        [
            NodeResult(success=True, data={}, output_text="bad <|tool_call>"),
            NodeResult(success=True, data={}, output_text="clean"),
        ]
    )

    def _predicate(r: NodeResult) -> bool:
        return "<|tool_call>" in (r.output_text or "")

    runner = _make_runner(
        spec,
        executor,
        NodeType.AGENT,
        retry_predicates={"Target": _predicate},
    )
    fr = runner.run("go")
    assert fr.success is True, fr.error
    assert executor.call_count == 2


# ── Non-retryable node types ─────────────────────────────────────────


def test_retry_metadata_ignored_on_non_retryable_types() -> None:
    """A node type outside {INSTRUCTION, INSTRUCTION_FORM, AGENT} must
    ignore ``retry_max_attempts``; the loop is intentionally limited to
    the builder's ``retry=`` surface."""
    from quartermaster_engine.types import NodeType as NT

    # DECISION is logic but not part of the retry surface.
    spec, target = _build_graph(node_type=NT.STATIC, retry_max_attempts=5)
    executor = _ScriptedExecutor([NodeResult(success=False, data={}, error="one-shot")])
    runner = _make_runner(spec, executor, NT.STATIC)
    runner.run("go")
    # One call only — metadata was present but the type wasn't eligible.
    assert executor.call_count == 1


# ── Zero / negative budget is normalised ─────────────────────────────


def test_zero_max_attempts_runs_exactly_once() -> None:
    spec, target = _build_graph(retry_max_attempts=0)
    executor = _ScriptedExecutor([NodeResult(success=False, data={}, error="boom")])
    runner = _make_runner(spec, executor, NodeType.INSTRUCTION)
    runner.run("go")
    assert executor.call_count == 1
