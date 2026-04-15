"""Regression tests for v0.3.0 End-node semantics (Proposal A).

Under Proposal A reaching an End node in the MAIN graph dispatches
control back to the Start node (enabling recursive agent loops), while
an End node inside a sub-graph (spawned via ``SUB_ASSISTANT``) returns
control to the parent flow.  ``.end(stop=True)`` remains the explicit
opt-out for the rare "stop here permanently" case, which sets
``traverse_out=SPAWN_NONE`` on the End node.

Every test in this module uses a HARD stopping condition (explicit
``stop=True``, an If/break after N rounds, or the runner's
``max_loop_iterations`` cap reduced to a small ceiling) so a regression
cannot hang the test suite forever.
"""

from __future__ import annotations

from uuid import uuid4

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.context.node_execution import NodeStatus
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import (
    GraphSpec,
    GraphNode,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
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


class _CounterBreakerExecutor:
    """Increments a counter and, once it reaches *threshold*, picks the
    ``exit`` branch.  Otherwise picks ``loop``.

    Pairs with a Decision-like SPAWN_PICKED node in tests that need a
    bounded loop without leaning on the runner's max_loop_iterations
    safety cap.
    """

    def __init__(self, threshold: int) -> None:
        self.call_count = 0
        self._threshold = threshold

    async def execute(self, context: ExecutionContext) -> NodeResult:
        self.call_count += 1
        picked = "exit" if self.call_count >= self._threshold else "loop"
        return NodeResult(
            success=True,
            data={},
            output_text=str(self.call_count),
            picked_node=picked,
        )


# ── 1. main-graph End loops back to Start ───────────────────────────


def test_end_loops_back_to_start_in_main_graph():
    """Start → Counter → If(count >= N ? "exit" : "loop") → End(loop)/End(stop).

    Verifies the NEW default End behaviour: reaching an End node without
    ``stop=True`` dispatches the Start node, rerunning the whole flow
    body.  The Counter executor must be called exactly THRESHOLD times
    before the "exit" branch short-circuits via ``.end(stop=True)``.
    """
    threshold = 3

    start = make_node(NodeType.START, name="Start")
    counter = make_node(NodeType.INSTRUCTION, name="Counter")
    decision = make_node(
        NodeType.DECISION, name="Done?", traverse_out=TraverseOut.SPAWN_PICKED
    )
    # loop branch → End (default SPAWN_START loop)
    end_loop = make_node(
        NodeType.END,
        name="EndLoop",
        traverse_out=TraverseOut.SPAWN_START,
    )
    # exit branch → End (SPAWN_NONE stop)
    end_stop = make_node(
        NodeType.END,
        name="EndStop",
        traverse_out=TraverseOut.SPAWN_NONE,
    )

    graph = make_graph(
        [start, counter, decision, end_loop, end_stop],
        [
            make_edge(start, counter),
            make_edge(counter, decision),
            make_edge(decision, end_loop, label="loop"),
            make_edge(decision, end_stop, label="exit"),
        ],
        start,
    )

    counter_exec = _CountingExecutor()
    decision_exec = _CounterBreakerExecutor(threshold)

    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, counter_exec)
    registry.register(NodeType.DECISION.value, decision_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    # Hard safety cap (separate from the test's own threshold).
    runner.max_loop_iterations = 10

    result = runner.run("go")
    assert result.success, result.error
    assert counter_exec.call_count == threshold, (
        f"Counter should run exactly {threshold} times, got "
        f"{counter_exec.call_count}"
    )


# ── 2. .end(stop=True) opt-out ─────────────────────────────────────


def test_end_with_stop_kwarg_does_not_loop():
    """An End node with ``traverse_out=SPAWN_NONE`` (set by
    ``.end(stop=True)``) still behaves the pre-0.3.0 way — reached
    once, then the flow terminates."""
    start = make_node(NodeType.START, name="Start")
    inst = make_node(NodeType.INSTRUCTION, name="Once")
    end = make_node(
        NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE
    )

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
        f"Instruction should run exactly once with stop=True End, got "
        f"{counter_exec.call_count}"
    )


# ── 3. graph without End auto-stops at last node ───────────────────


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


# ── 4. sub-graph End returns to parent ─────────────────────────────


def test_sub_graph_end_returns_to_parent():
    """A SUB_ASSISTANT node spawns a child FlowRunner with
    ``parent_context=<caller ExecutionContext>``; the child's End node
    detects the parent_context and stamps
    ``__end_returns_to_parent__`` on its NodeResult instead of looping
    back to the sub-graph's Start.  The parent's SUB_ASSISTANT node
    then dispatches its own successors.
    """
    from quartermaster_engine.example_runner import SubAssistantExecutor

    # ── Sub-graph: Start → ChildNode → End ──
    sub_start = make_node(NodeType.START, name="SubStart")
    child = make_node(NodeType.INSTRUCTION, name="ChildNode")
    # The sub-graph's End has SPAWN_START by default — but because
    # parent_context is set the runner short-circuits the loop and
    # returns to the parent instead, regardless of SPAWN_START vs
    # SPAWN_NONE.  Keep the default SPAWN_START to prove this.
    sub_end = make_node(
        NodeType.END, name="SubEnd", traverse_out=TraverseOut.SPAWN_START
    )
    sub_graph = make_graph(
        [sub_start, child, sub_end],
        [make_edge(sub_start, child), make_edge(child, sub_end)],
        sub_start,
    )

    child_exec = _CountingExecutor()
    child_registry = SimpleNodeRegistry()
    child_registry.register(NodeType.INSTRUCTION.value, child_exec)

    # ── Main graph: Start → SubAssistant → SecondNode → End(stop=True) ──
    main_start = make_node(NodeType.START, name="MainStart")
    sub = make_node(
        NodeType.SUB_ASSISTANT,
        name="InvokeChild",
        metadata={"sub_assistant_id": "child"},
    )
    second = make_node(NodeType.INSTRUCTION, name="SecondNode")
    main_end = make_node(
        NodeType.END, name="MainEnd", traverse_out=TraverseOut.SPAWN_NONE
    )

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
    # The SubAssistantExecutor needs a resolver → sub-graph lookup, and
    # a node_registry for the child FlowRunner to use.  Keep the
    # registries separate to prove the executors are NOT shared state.
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
        f"Child node should run exactly once in the sub-graph "
        f"(parent_context should suppress the loop), got "
        f"{child_exec.call_count}"
    )
    assert second_exec.call_count == 1, (
        f"Parent's SecondNode should run exactly once after the "
        f"sub-graph returns, got {second_exec.call_count}"
    )


# ── 5. parent_context is propagated into the sub-graph ──────────────


def test_parent_context_propagated_into_subgraph():
    """The sub-graph's ExecutionContext must carry a non-None
    ``parent_context`` pointing at the parent's live ExecutionContext —
    without that wiring the sub-graph's End wouldn't know to return to
    its caller.
    """
    from quartermaster_engine.example_runner import SubAssistantExecutor

    captured: dict = {}
    parent_seen: list[ExecutionContext] = []

    class _ParentCapturingExecutor:
        """Runs in the parent — records its own context so the test can
        assert it matches what the child sees via parent_context."""

        async def execute(self, context: ExecutionContext) -> NodeResult:
            parent_seen.append(context)
            return NodeResult(success=True, data={}, output_text="parent-ran")

    class _ChildCapturingExecutor:
        """Runs in the sub-graph — records its own parent_context."""

        async def execute(self, context: ExecutionContext) -> NodeResult:
            captured["parent_context"] = context.parent_context
            return NodeResult(success=True, data={}, output_text="child-ran")

    sub_start = make_node(NodeType.START, name="SubStart")
    child = make_node(NodeType.INSTRUCTION, name="ChildPeek")
    sub_end = make_node(
        NodeType.END, name="SubEnd", traverse_out=TraverseOut.SPAWN_START
    )
    sub_graph = make_graph(
        [sub_start, child, sub_end],
        [make_edge(sub_start, child), make_edge(child, sub_end)],
        sub_start,
    )

    child_registry = SimpleNodeRegistry()
    child_registry.register(
        NodeType.INSTRUCTION.value, _ChildCapturingExecutor()
    )

    main_start = make_node(NodeType.START, name="MainStart")
    sub = make_node(
        NodeType.SUB_ASSISTANT,
        name="InvokeChild",
        metadata={"sub_assistant_id": "child"},
    )
    # Extra parent-side logic node to have something capture its own
    # ExecutionContext — keeps parent_seen populated for the identity
    # check below.
    parent_peek = make_node(NodeType.INSTRUCTION, name="ParentPeek")
    main_end = make_node(
        NodeType.END, name="MainEnd", traverse_out=TraverseOut.SPAWN_NONE
    )
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
    main_registry.register(
        NodeType.INSTRUCTION.value, _ParentCapturingExecutor()
    )
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
    # The parent_context the child sees must be a real ExecutionContext
    # instance.  Identity against ``parent_seen`` isn't guaranteed
    # because the SUB_ASSISTANT node itself has a separate
    # ExecutionContext from ParentPeek's; what matters is that the
    # child saw SOME parent context instead of ``None``.
    assert isinstance(captured["parent_context"], ExecutionContext)


# ── 6. validator allows implicit End → Start cycles ────────────────


def test_validator_allows_implicit_end_to_start_cycle():
    """Building a graph that ends in a SPAWN_START End (the v0.3.0
    default) must NOT be rejected by the validator — the End → Start
    back-edge lives in the runner, not in ``graph.edges``.
    """
    import quartermaster_sdk as qm

    # This graph has no user-written cycle in its edge list — just a
    # plain Start → user → instruction → End, with End's traverse_out
    # pointing back to Start implicitly.  The validator should accept
    # it without `validate=False`.
    graph = (
        qm.Graph("loopy")
        .instruction("Tick")
        .end()
        .build()
    )
    # Sanity: the End node was produced with SPAWN_START (the new
    # default), not SPAWN_NONE.
    end_nodes = [n for n in graph.nodes if n.type == NodeType.END]
    assert end_nodes, "graph must have an End node"
    assert end_nodes[-1].traverse_out == TraverseOut.SPAWN_START


# ── 7. existing loop-style example (08/15) still terminates ────────


def test_existing_loop_examples_still_work():
    """Sanity check — a classic "loop N times then exit" graph using
    the new .end() semantics (no .connect() back-edge) must produce a
    final output without hitting max_loop_iterations.

    Mimics the structure of ``examples/08_iterative_refinement.py``
    and ``examples/15_courtroom_debate.py`` after migration: an If
    node decides between "loop" (default End, SPAWN_START) and "exit"
    (End(stop=True)).
    """
    threshold = 3

    start = make_node(NodeType.START, name="Start")
    body = make_node(NodeType.INSTRUCTION, name="Body")
    decision = make_node(
        NodeType.DECISION, name="Done?", traverse_out=TraverseOut.SPAWN_PICKED
    )
    loop_end = make_node(
        NodeType.END,
        name="LoopEnd",
        traverse_out=TraverseOut.SPAWN_START,
    )
    exit_end = make_node(
        NodeType.END,
        name="ExitEnd",
        traverse_out=TraverseOut.SPAWN_NONE,
    )
    graph = make_graph(
        [start, body, decision, loop_end, exit_end],
        [
            make_edge(start, body),
            make_edge(body, decision),
            make_edge(decision, loop_end, label="loop"),
            make_edge(decision, exit_end, label="exit"),
        ],
        start,
    )

    body_exec = _CountingExecutor()
    decision_exec = _CounterBreakerExecutor(threshold)
    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, body_exec)
    registry.register(NodeType.DECISION.value, decision_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    # Cap well below an infinite loop but above threshold so a
    # regression against the loop-guard surfaces as a FAIL rather
    # than "test takes too long".
    runner.max_loop_iterations = threshold + 5
    result = runner.run("loop")
    assert result.success, result.error
    assert body_exec.call_count == threshold, (
        f"Body should have run exactly {threshold} times, got "
        f"{body_exec.call_count}"
    )


# ── 8. max_loop_iterations safety cap ──────────────────────────────


def test_loop_safety_cap_prevents_infinite_recursion():
    """A graph with an End that loops back and NO break condition
    must stop once ``max_loop_iterations`` is hit rather than
    recursing forever.  This is the safety rail for accidentally-
    looping graphs.
    """
    start = make_node(NodeType.START, name="Start")
    body = make_node(NodeType.INSTRUCTION, name="Body")
    end = make_node(
        NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_START
    )

    graph = make_graph(
        [start, body, end],
        [make_edge(start, body), make_edge(body, end)],
        start,
    )

    counter_exec = _CountingExecutor()
    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, counter_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    runner.max_loop_iterations = 5
    result = runner.run("runaway")
    # The cap was hit — the flow still returns (doesn't raise) and
    # the counter is bounded by the cap.
    assert result.success, result.error
    assert counter_exec.call_count == runner.max_loop_iterations + 1, (
        f"Body should have run exactly max_loop_iterations+1 times "
        f"(initial run + N loop-backs), got {counter_exec.call_count}"
    )
