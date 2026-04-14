"""Integration tests for event streaming and token generation during flow execution."""

from __future__ import annotations

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import (
    FlowEvent,
    FlowFinished,
    NodeFinished,
    NodeStarted,
    TokenGenerated,
)
from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import (
    NodeType,
    TraverseIn,
    TraverseOut,
)
from tests.conftest import (
    EchoExecutor,
    SlowExecutor,
    make_edge,
    make_graph,
    make_node,
)


class _StreamingExecutor:
    """Emits tokens via the context callback during execution."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or ["Hello", " ", "world", "!"]

    async def execute(self, context: ExecutionContext) -> NodeResult:
        for token in self._tokens:
            context.emit_token(token)
        full_text = "".join(self._tokens)
        return NodeResult(success=True, data={}, output_text=full_text)


class _PartialStreamExecutor:
    """Emits some tokens then returns a result with remaining text."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        context.emit_token("partial-")
        context.emit_token("stream")
        return NodeResult(success=True, data={}, output_text="partial-stream-complete")


class TestTokenStreaming:
    """Token-level streaming via on_token callback."""

    def test_tokens_emitted_during_execution(self):
        start = make_node(NodeType.START, name="Start")
        streaming = make_node(NodeType.INSTRUCTION, name="Streamer")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, streaming, end],
            [make_edge(start, streaming), make_edge(streaming, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _StreamingExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("stream test")

        assert result.success
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        assert len(token_events) == 4
        tokens = [e.token for e in token_events]
        assert tokens == ["Hello", " ", "world", "!"]

    def test_token_events_contain_correct_node_id(self):
        start = make_node(NodeType.START, name="Start")
        streaming = make_node(NodeType.INSTRUCTION, name="Streamer")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, streaming, end],
            [make_edge(start, streaming), make_edge(streaming, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _StreamingExecutor(["a", "b"]))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("node id test")

        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        for te in token_events:
            assert te.node_id == streaming.id

    def test_custom_tokens(self):
        start = make_node(NodeType.START, name="Start")
        streaming = make_node(NodeType.INSTRUCTION, name="Streamer")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, streaming, end],
            [make_edge(start, streaming), make_edge(streaming, end)],
            start,
        )

        custom_tokens = ["The", " answer", " is", " 42"]
        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _StreamingExecutor(custom_tokens))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("custom tokens")

        assert result.success
        tokens = [e.token for e in events if isinstance(e, TokenGenerated)]
        assert tokens == custom_tokens
        assert result.final_output == "The answer is 42"


class TestPartialStreaming:
    """Partial streaming: some tokens emitted, then final result."""

    def test_partial_stream_then_complete(self):
        start = make_node(NodeType.START, name="Start")
        node = make_node(NodeType.INSTRUCTION, name="Partial")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, node, end],
            [make_edge(start, node), make_edge(node, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _PartialStreamExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("partial test")

        assert result.success
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        assert len(token_events) == 2
        tokens = [e.token for e in token_events]
        assert tokens == ["partial-", "stream"]
        assert "partial-stream-complete" in result.final_output


class TestEventOrdering:
    """Events are emitted in the correct order."""

    def test_start_before_finish_before_flow_finished(self):
        start = make_node(NodeType.START, name="Start")
        inst = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, inst, end],
            [make_edge(start, inst), make_edge(inst, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("ordering test")

        event_types = [type(e).__name__ for e in events]
        # NodeStarted should come before NodeFinished for the same node
        start_idx = event_types.index("NodeStarted")
        finish_idx = event_types.index("NodeFinished")
        flow_finish_idx = event_types.index("FlowFinished")
        assert start_idx < finish_idx < flow_finish_idx

    def test_tokens_between_start_and_finish(self):
        """Token events should appear between NodeStarted and NodeFinished."""
        start = make_node(NodeType.START, name="Start")
        streaming = make_node(NodeType.INSTRUCTION, name="Streamer")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, streaming, end],
            [make_edge(start, streaming), make_edge(streaming, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _StreamingExecutor(["a"]))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("token ordering")

        # Find events for the Streamer node specifically
        streamer_events = [
            e
            for e in events
            if isinstance(e, (NodeStarted, NodeFinished, TokenGenerated))
            and getattr(e, "node_id", None) == streaming.id
        ]
        types = [type(e).__name__ for e in streamer_events]
        assert types == ["NodeStarted", "TokenGenerated", "NodeFinished"]


class TestMultiNodeStreaming:
    """Streaming across multiple nodes in a flow."""

    def test_two_streaming_nodes_sequential(self):
        """Two streaming nodes in sequence both emit tokens."""
        start = make_node(NodeType.START, name="Start")
        s1 = make_node(NodeType.INSTRUCTION, name="S1")
        s2 = make_node(NodeType.STATIC, name="S2")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, s1, s2, end],
            [
                make_edge(start, s1),
                make_edge(s1, s2),
                make_edge(s2, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, _StreamingExecutor(["A", "B"]))
        registry.register(NodeType.STATIC.value, _StreamingExecutor(["C", "D"]))

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        result = runner.run("multi stream")

        assert result.success
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        assert len(token_events) == 4
        # S1 tokens come first, then S2 tokens (sequential)
        s1_tokens = [e.token for e in token_events if e.node_id == s1.id]
        s2_tokens = [e.token for e in token_events if e.node_id == s2.id]
        assert s1_tokens == ["A", "B"]
        assert s2_tokens == ["C", "D"]

    def test_no_tokens_from_non_streaming_node(self):
        """Nodes that don't emit tokens should produce no TokenGenerated events."""
        start = make_node(NodeType.START, name="Start")
        inst = make_node(NodeType.INSTRUCTION, name="Work")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, inst, end],
            [make_edge(start, inst), make_edge(inst, end)],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(NodeType.INSTRUCTION.value, EchoExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=registry, on_event=events.append)
        runner.run("no tokens")

        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        assert len(token_events) == 0
