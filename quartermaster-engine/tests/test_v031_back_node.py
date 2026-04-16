"""Regression tests for v0.3.1 ``NodeType.BACK`` / ``.back()`` semantics.

``Back`` is the explicit "loop back to Start / return to parent"
marker introduced in v0.3.1.  Reaching a Back node in the main graph
dispatches the graph's Start node again (subject to
``FlowRunner.max_loop_iterations``); reaching one inside a sub-graph
returns control to the parent flow via the shared
``__return_to_parent__`` sentinel.

Every test uses a HARD stopping condition (explicit ``.end()``, an
If / Decision break after N rounds, or the runner's
``max_loop_iterations`` cap reduced to a small ceiling) so a regression
cannot hang the test suite forever.
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
    """Increments a local counter on every ``execute`` call."""

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
    ``exit`` branch.  Otherwise picks ``loop``.  Pairs with a
    SPAWN_PICKED decision-shaped node to build a bounded loop that
    uses a Back node on the loop arm.
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


# ── 1. Back loops to Start in the main graph ───────────────────────


def test_back_loops_to_start_in_main_graph():
    """Graph shape::

        Start → Counter → Decision ─[exit]→ End
                              │
                              └─[loop]→ Back   (→ Start)

    The Counter executor increments each iteration; after THRESHOLD
    runs the decision picks ``exit`` and hits End, terminating the
    flow.  Asserts Counter was called exactly THRESHOLD times.
    """
    threshold = 3

    start = make_node(NodeType.START, name="Start")
    counter = make_node(NodeType.INSTRUCTION, name="Counter")
    decision = make_node(NodeType.DECISION, name="Done?", traverse_out=TraverseOut.SPAWN_PICKED)
    back = make_node(NodeType.BACK, name="Back", traverse_out=TraverseOut.SPAWN_NONE)
    end_stop = make_node(NodeType.END, name="EndStop", traverse_out=TraverseOut.SPAWN_NONE)

    graph = make_graph(
        [start, counter, decision, back, end_stop],
        [
            make_edge(start, counter),
            make_edge(counter, decision),
            make_edge(decision, back, label="loop"),
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
    # Hard safety cap above the expected threshold so regressions surface
    # as "wrong count" rather than "hung forever".
    runner.max_loop_iterations = 10

    result = runner.run("go")
    assert result.success, result.error
    assert counter_exec.call_count == threshold, (
        f"Counter should run exactly {threshold} times, got {counter_exec.call_count}"
    )


# ── 2. Back returns to parent in a sub-graph ──────────────────────


def test_back_returns_to_parent_in_sub_graph():
    """Mirror of ``test_sub_graph_end_returns_to_parent`` but uses a
    Back node instead of End inside the sub-graph::

        Main: Start → SubAssistant → Second → End
        Sub:  Start → Child → Back

    Expected: Child runs once, then control returns to the parent;
    Second runs once; End stops.  Specifically, the Back node in the
    sub-graph MUST NOT dispatch the sub-graph's Start again — it
    signals return-to-parent via the ``__return_to_parent__``
    sentinel.
    """
    from quartermaster_engine.example_runner import SubAssistantExecutor

    # Sub-graph: Start → Child → Back
    sub_start = make_node(NodeType.START, name="SubStart")
    child = make_node(NodeType.INSTRUCTION, name="Child")
    sub_back = make_node(NodeType.BACK, name="SubBack", traverse_out=TraverseOut.SPAWN_NONE)
    sub_graph = make_graph(
        [sub_start, child, sub_back],
        [make_edge(sub_start, child), make_edge(child, sub_back)],
        sub_start,
    )

    child_exec = _CountingExecutor()
    child_registry = SimpleNodeRegistry()
    child_registry.register(NodeType.INSTRUCTION.value, child_exec)

    # Main graph: Start → SubAssistant → Second → End
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
        f"Child node should run exactly once inside the sub-graph "
        f"(Back should return to parent, not loop), got "
        f"{child_exec.call_count}"
    )
    assert second_exec.call_count == 1, (
        f"Parent's SecondNode should run exactly once after the "
        f"sub-graph's Back returns, got {second_exec.call_count}"
    )


# ── 3. Back can live inside an on(...) branch ──────────────────────


def test_back_can_be_inside_branch():
    """Fluent builder integration test::

        Graph("loopy")
          .instruction("Tick")
          .if_node("done?", expression="...")
          .on("true").text("Done").end()
          .on("false").back()

    Counts how many times "Tick" runs.  The ``_PickingIfExecutor``
    picks "false" for the first N iterations (which triggers the
    ``.back()`` arm — loop to Start), then "true" on iteration N+1
    (which flows into the "Done" text node and auto-merges into the
    graph-level End).  Asserts the loop executed exactly N times.
    """
    import quartermaster_sdk as qm

    # Shared counter across evaluations via a closure.
    state = {"n": 0}
    threshold = 3

    class _BranchCounterExecutor:
        async def execute(self, context: ExecutionContext) -> NodeResult:
            state["n"] += 1
            return NodeResult(success=True, data={}, output_text=f"n={state['n']}")

    class _PickingIfExecutor:
        """Mimics an If node: pick 'true' once threshold hit, else 'false'."""

        async def execute(self, context: ExecutionContext) -> NodeResult:
            picked = "true" if state["n"] >= threshold else "false"
            return NodeResult(success=True, data={}, output_text=picked, picked_node=picked)

    # Build via the fluent builder so we also exercise .back() wiring.
    # The "Done" text node on the true arm gives the labelled-edge
    # hook something to attach to (.on("true")) so SPAWN_PICKED can
    # actually resolve the pick to the right successor.
    builder = (
        qm.Graph("loopy")
        .instruction("Tick")
        .if_node("done?", expression="False")
        .on("true")
        .text("Done", template="done")
        .end()
        .on("false")
        .back()
    )
    graph = builder.build()

    # Registry: swap the real LLM / If executors for deterministic ones.
    # A stub Text executor keeps the "Done" node on the true arm happy.
    class _NoopTextExecutor:
        async def execute(self, context: ExecutionContext) -> NodeResult:
            return NodeResult(success=True, data={}, output_text="done")

    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, _BranchCounterExecutor())
    registry.register(NodeType.IF.value, _PickingIfExecutor())
    registry.register(NodeType.TEXT.value, _NoopTextExecutor())

    runner = FlowRunner(graph=graph, node_registry=registry)
    runner.max_loop_iterations = 10
    result = runner.run("go")
    assert result.success, result.error

    assert state["n"] == threshold, (
        f"Tick should run exactly {threshold} times; the ``.back()`` "
        f"arm loops until the ``true`` arm short-circuits, got {state['n']}"
    )

    # Sanity: the builder emitted exactly one Back node (from the
    # ``.on("false")`` branch).
    back_nodes = [n for n in graph.nodes if n.type == NodeType.BACK]
    assert len(back_nodes) == 1, "expected one Back node from the .on('false') branch"


# ── 4. Back respects max_loop_iterations ──────────────────────────


def test_back_respects_max_loop_iterations():
    """A graph that unconditionally loops via Back must stop when
    ``max_loop_iterations`` is hit, not recurse forever.  This is the
    safety rail for graphs whose loop-break condition is broken.
    """
    start = make_node(NodeType.START, name="Start")
    body = make_node(NodeType.INSTRUCTION, name="Body")
    back = make_node(NodeType.BACK, name="Back", traverse_out=TraverseOut.SPAWN_NONE)

    graph = make_graph(
        [start, body, back],
        [make_edge(start, body), make_edge(body, back)],
        start,
    )

    counter_exec = _CountingExecutor()
    registry = SimpleNodeRegistry()
    registry.register(NodeType.INSTRUCTION.value, counter_exec)

    runner = FlowRunner(graph=graph, node_registry=registry)
    runner.max_loop_iterations = 5
    result = runner.run("runaway")
    # The cap was hit — the flow still returns (doesn't raise) and the
    # counter is bounded by the cap.
    assert result.success, result.error
    assert counter_exec.call_count == runner.max_loop_iterations + 1, (
        f"Body should have run exactly max_loop_iterations+1 times "
        f"(initial run + N Back-driven loops), got "
        f"{counter_exec.call_count}"
    )
