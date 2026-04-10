"""End-to-end pipeline: build graph + configure providers + run engine.

Ties together quartermaster-graph, quartermaster-providers, and quartermaster-engine to show the full
Quartermaster execution flow.  Uses in-memory stores and a stub node
registry so the example runs without external API keys.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

try:
    from quartermaster_graph.builder import GraphBuilder
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

try:
    from quartermaster_engine.runner.flow_runner import FlowRunner, FlowResult
    from quartermaster_engine.nodes import NodeResult, SimpleNodeRegistry
    from quartermaster_engine.context.execution_context import ExecutionContext
    from quartermaster_engine.events import FlowEvent, NodeStarted, NodeFinished
except ImportError:
    raise SystemExit("Install quartermaster-engine first:  pip install -e quartermaster-engine")


# -- Stub node executor for demonstration ------------------------------------

class EchoExecutor:
    """A simple executor that echoes input back, prefixed with the node name."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        user_input = str(context.memory.get("__user_input__", ""))
        output = f"[{context.current_node.name}] processed: {user_input}"
        return NodeResult(success=True, data={}, output_text=output)


def main() -> None:
    # Step 1 -- Build the graph
    graph = (
        GraphBuilder("Full Pipeline Demo")
        .start()
        .instruction("Analyze", system_instruction="Analyze the input.")
        .instruction("Summarize", system_instruction="Summarize the analysis.")
        .end()
        .build(version="1.0.0")
    )
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Step 2 -- Set up the node registry with our stub executor
    registry = SimpleNodeRegistry()
    echo = EchoExecutor()
    registry.register(NodeType.INSTRUCTION.value, echo)
    print(f"Registered executors: {registry.list_types()}")

    # Step 3 -- Wire up an event callback
    events: list[FlowEvent] = []

    def on_event(event: FlowEvent) -> None:
        events.append(event)
        if isinstance(event, NodeStarted):
            print(f"  >> Started: {event.node_name}")
        elif isinstance(event, NodeFinished):
            print(f"  << Finished: {event.result[:60] if event.result else ''}")

    # Step 4 -- Create and run the FlowRunner
    runner = FlowRunner(graph=graph, node_registry=registry, on_event=on_event)
    print("\nRunning flow...")
    result: FlowResult = runner.run("Hello, Quartermaster!")

    # Step 5 -- Inspect the result
    print(f"\nFlow ID       : {result.flow_id}")
    print(f"Success       : {result.success}")
    print(f"Final output  : {result.final_output}")
    print(f"Duration      : {result.duration_seconds:.3f}s")
    print(f"Events fired  : {len(events)}")
    print(f"Node results  : {len(result.node_results)}")

    if result.error:
        print(f"Error         : {result.error}")


if __name__ == "__main__":
    main()
