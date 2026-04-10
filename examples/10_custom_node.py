"""Create a custom node type and use it in a graph.

Shows how to use NodeType.CUSTOM with the GraphBuilder, and how to
register a custom NodeExecutor for it in the engine's SimpleNodeRegistry.
"""

from __future__ import annotations

from typing import Any

try:
    from quartermaster_graph.builder import GraphBuilder
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

try:
    from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
    from quartermaster_engine.context.execution_context import ExecutionContext
    from quartermaster_engine.runner.flow_runner import FlowRunner
    from quartermaster_engine.events import NodeStarted, NodeFinished
except ImportError:
    raise SystemExit("Install quartermaster-engine first:  pip install -e quartermaster-engine")


class SentimentAnalyzer:
    """Custom node executor that performs keyword-based sentiment analysis."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        # Get user input from memory (set by FlowRunner) or from messages
        text = str(context.memory.get("__user_input__", ""))
        if context.messages:
            text += " " + " ".join(m.content for m in context.messages if m.content)
        text = text.lower()
        positive = sum(1 for w in ["good", "great", "happy", "love", "excellent"] if w in text)
        negative = sum(1 for w in ["bad", "terrible", "sad", "hate", "awful"] if w in text)

        if positive > negative:
            sentiment = "positive"
        elif negative > positive:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return NodeResult(
            success=True,
            data={"sentiment": sentiment, "positive": positive, "negative": negative},
            output_text=f"Sentiment: {sentiment} (positive={positive}, negative={negative})",
        )


def main() -> None:
    # Step 1 -- Build a graph with a custom node
    graph = (
        GraphBuilder("Custom Node Demo")
        .start()
        .node(
            NodeType.CUSTOM,
            name="Sentiment Check",
            metadata={"analyzer": "keyword", "version": "1.0"},
        )
        .end()
        .build(version="1.0.0")
    )

    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    for node in graph.nodes:
        print(f"  {node.type.value:15s}  {node.name!r}  metadata={dict(node.metadata)}")

    # Step 2 -- Register the custom executor
    registry = SimpleNodeRegistry()
    registry.register(NodeType.CUSTOM.value, SentimentAnalyzer())
    print(f"\nRegistered types: {registry.list_types()}")

    # Step 3 -- Run with a positive message
    def on_event(event: Any) -> None:
        if isinstance(event, NodeFinished) and event.result:
            print(f"  Node result: {event.result}")

    runner = FlowRunner(graph=graph, node_registry=registry, on_event=on_event)
    print("\nRunning with positive input...")
    result = runner.run("I love this great product, it's excellent!")
    print(f"Final: {result.final_output}")

    # Step 4 -- Run with a negative message
    runner2 = FlowRunner(graph=graph, node_registry=registry, on_event=on_event)
    print("\nRunning with negative input...")
    result2 = runner2.run("This is terrible and bad, I hate it.")
    print(f"Final: {result2.final_output}")


if __name__ == "__main__":
    main()
