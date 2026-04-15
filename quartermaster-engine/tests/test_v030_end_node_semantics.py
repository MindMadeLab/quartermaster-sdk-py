"""Regression tests for v0.3.1 reverted End-node semantics.

v0.3.0 briefly tried "End loops back to Start by default" (Proposal A)
— that silently broke every trailing-``.end()`` graph written against
v0.2.x, so v0.3.1 reverts: End again means **stop here**.  The
explicit "loop back / return to parent" behaviour has moved to the
new :meth:`GraphBuilder.back` builder method backed by
``NodeType.BACK`` — see ``test_v031_back_node.py`` for that
behaviour.

This file covers the reverted End semantics:

* End in the main graph stops (no implicit loop).
* End inside a sub-graph still returns control to the parent flow
  via the ``__return_to_parent__`` sentinel.
* A graph with no End auto-stops at its last node (unchanged).
* The validator no longer errors on explicit cycles — it warns — so
  Back nodes can form legitimate cycles.
"""

from __future__ import annotations

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import (
    NodeType,
    TraverseOut,
)

from tests.conftest import make_edge, make_graph, make_node


# ── helpers ──────────────────────────────────────────────────────────


class _CountingExecutor:
    """Increments a module-local counter on every ``execute`` call."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, context: ExecutionContext) -> NodeResult:
        self.call_count += 1
        return NodeResult(
            success=True,
            data={"call_count": self.call_count},
            output_text=f"n={self.call_count}",
        )


# ── 1. End stops in main graph ─────────────────────────────────────


def test_end_stops_at_end_node():
    """Reaching an End node in the main graph stops the flow —
    ``traverse_out=SPAWN_NONE`` is the v0.3.1 default produced by
    :meth:`GraphBuilder.end`, so the executor runs exactly once.
    """
    start = make_node(NodeType.START, name="Start")
    inst = make_node(NodeType.INSTRUCTION, name="Once")
    end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

    graph = make_graph(
        [start, inst, end],
        [make_edge(start, inst), make_edge(inst, end)],
        start,
    )

    counter_exec = _CountingExecutor()
    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, counter_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    runner.max_loop_iterations = 3
    result = runner.run("once")
    assert result.success, result.error
    assert counter_exec.call_count == 1, (
        f"Instruction should run exactly once with a trailing End "
        f"(v0.3.1 revert), got {counter_exec.call_count}"
    )


# ── 2. graph without End auto-stops at last node ───────────────────


def test_no_end_auto_stops_at_last_node():
    """A graph with no End node reaches its last node and stops — the
    pre-0.3.0 "implicit auto-stop" behaviour is unchanged."""
    start = make_node(NodeType.START, name="Start")
    inst = make_node(NodeType.INSTRUCTION, name="OneShot")

    graph = make_graph(
        [start, inst],
        [make_edge(start, inst)],
        start,
    )

    counter_exec = _CountingExecutor()
    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, counter_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    runner.max_loop_iterations = 3
    result = runner.run("once")
    assert result.success, result.error
    # The last node's output becomes final_output.
    assert "n=1" in result.final_output
    assert counter_exec.call_count == 1


# ── 3. sub-graph End returns to parent ─────────────────────────────


def test_sub_graph_end_returns_to_parent():
    """A SUB_ASSISTANT node spawns a child FlowRunner with
    ``parent_context=<caller ExecutionContext>``; the child's End node
    detects the parent_context and stamps ``__return_to_parent__`` on
    its NodeResult so ``_dispatch_successors`` does nothing further in
    the child.  The parent's SUB_ASSISTANT node then dispatches its
    own successors as normal.
    """
    from quartermaster_engine.example_runner import SubAssistantExecutor

    # ── Sub-graph: Start → ChildNode → End ──
    sub_start = make_node(NodeType.START, name="SubStart")
    child = make_node(NodeType.INSTRUCTION, name="ChildNode")
    sub_end = make_node(NodeType.END, name="SubEnd", traverse_out=TraverseOut.SPAWN_NONE)
    sub_graph = make_graph(
        [sub_start, child, sub_end],
        [make_edge(sub_start, child), make_edge(child, sub_end)],
        sub_start,
    )

    child_exec = _CountingExecutor()
    child_registry = SimpleNodeRegistry()
    child_registry.register(NodeType.INSTRUCTION.value, child_exec)

    # ── Main graph: Start → SubAssistant → SecondNode → End ──
    main_start = make_node(NodeType.START, name="MainStart")
    sub = make_node(
        NodeType.SUB_ASSISTANT,
        name="InvokeChild",
        metadata={"sub_assistant_id": "child"},
    )
    second = make_node(NodeType.INSTRUCTION, name="SecondNode")
    main_end = make_node(NodeType.END, name="MainEnd", traverse_out=TraverseOut.SPAWN_NONE)

    main_graph = make_graph(
        [main_start, sub, second, main_end],
        [
            make_edge(main_start, sub),
            make_edge(sub, second),
            make_edge(second, main_end),
        ],
        main_start,
    )

    second_exec = _CountingExecutor()
    resolver = lambda sid: sub_graph if sid == "child" else None
    main_registry = SimpleNodeRegistry()
    main_registry.register(NodeType.INSTRUCTION.value, second_exec)
    main_registry.register(
        NodeType.SUB_ASSISTANT.value,
        SubAssistantExecutor(resolver=resolver, node_registry=child_registry),
    )

    runner = FlowRunner(graph=main_graph, node_registry=main_registry)
    runner.max_loop_iterations = 3
    result = runner.run("hello")
    assert result.success, result.error

    assert child_exec.call_count == 1, (
        f"Child node should run exactly once in the sub-graph, got {child_exec.call_count}"
    )
    assert second_exec.call_count == 1, (
        f"Parent's SecondNode should run exactly once after the "
        f"sub-graph returns, got {second_exec.call_count}"
    )


# ── 4. parent_context is propagated into the sub-graph ──────────────


def test_parent_context_propagated_into_subgraph():
    """The sub-graph's ExecutionContext must carry a non-None
    ``parent_context`` pointing at the parent's live ExecutionContext —
    without that wiring a sub-graph's End/Back wouldn't know to return
    control to its caller.
    """
    from quartermaster_engine.example_runner import SubAssistantExecutor

    captured: dict = {}

    class _ParentCapturingExecutor:
        async def execute(self, context: ExecutionContext) -> NodeResult:
            return NodeResult(success=True, data={}, output_text="parent-ran")

    class _ChildCapturingExecutor:
        async def execute(self, context: ExecutionContext) -> NodeResult:
            captured["parent_context"] = context.parent_context
            return NodeResult(success=True, data={}, output_text="child-ran")

    sub_start = make_node(NodeType.START, name="SubStart")
    child = make_node(NodeType.INSTRUCTION, name="ChildPeek")
    sub_end = make_node(NodeType.END, name="SubEnd", traverse_out=TraverseOut.SPAWN_NONE)
    sub_graph = make_graph(
        [sub_start, child, sub_end],
        [make_edge(sub_start, child), make_edge(child, sub_end)],
        sub_start,
    )

    child_registry = SimpleNodeRegistry()
    child_registry.register(NodeType.INSTRUCTION.value, _ChildCapturingExecutor())

    main_start = make_node(NodeType.START, name="MainStart")
    sub = make_node(
        NodeType.SUB_ASSISTANT,
        name="InvokeChild",
        metadata={"sub_assistant_id": "child"},
    )
    parent_peek = make_node(NodeType.INSTRUCTION, name="ParentPeek")
    main_end = make_node(NodeType.END, name="MainEnd", traverse_out=TraverseOut.SPAWN_NONE)
    main_graph = make_graph(
        [main_start, parent_peek, sub, main_end],
        [
            make_edge(main_start, parent_peek),
            make_edge(parent_peek, sub),
            make_edge(sub, main_end),
        ],
        main_start,
    )

    resolver = lambda sid: sub_graph if sid == "child" else None
    main_registry = SimpleNodeRegistry()
    main_registry.register(NodeType.INSTRUCTION.value, _ParentCapturingExecutor())
    main_registry.register(
        NodeType.SUB_ASSISTANT.value,
        SubAssistantExecutor(resolver=resolver, node_registry=child_registry),
    )

    runner = FlowRunner(graph=main_graph, node_registry=main_registry)
    runner.max_loop_iterations = 3
    result = runner.run("hi")
    assert result.success, result.error

    assert captured.get("parent_context") is not None, (
        "child's current_node.parent_context must not be None; it should "
        "point at the parent's live ExecutionContext"
    )
    assert isinstance(captured["parent_context"], ExecutionContext)


# ── 5. validator tolerates Back-driven cycles (warning only) ───────


def test_validator_allows_back_cycles_as_warning():
    """Building a graph that loops via a Back node must NOT be
    rejected by the validator — Back nodes dispatch Start implicitly
    through the runner, but if a user also wires an explicit cycle
    the validator emits a WARNING (not an ERROR) so intentional loops
    don't break the build.
    """
    import quartermaster_sdk as qm

    # Minimal "loop via Back" shape — the If node's two arms
    # guarantee a hard stop eventually.  We only care that the graph
    # builds cleanly under the default validator (``validate=True``).
    graph = (
        qm.Graph("loopy")
        .if_node("keep?", expression="False")
        .on("true")
        .text("Done")
        .end()
        .on("false")
        .back()
    ).build()

    # Sanity: the graph has exactly one Back node — the ``.on("false")``
    # branch emits it and the runner routes the dispatch back to Start.
    back_nodes = [n for n in graph.nodes if n.type == NodeType.BACK]
    assert len(back_nodes) == 1, "expected exactly one Back node"
    assert back_nodes[0].traverse_out == TraverseOut.SPAWN_NONE
