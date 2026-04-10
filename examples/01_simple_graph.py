"""Build and inspect a simple Start -> Instruction -> End graph.

Demonstrates the core GraphBuilder API: creating nodes, chaining them,
and building a validated AgentVersion object.
"""

from __future__ import annotations

try:
    from quartermaster_graph.builder import GraphBuilder
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")


def main() -> None:
    # Step 1 -- Build a minimal three-node graph
    graph = (
        GraphBuilder("Hello World Agent", description="A minimal example agent")
        .start()
        .instruction(
            "Greet the user",
            model="gpt-4o",
            provider="openai",
            temperature=0.5,
            system_instruction="You are a friendly greeter.",
        )
        .end()
        .build(version="1.0.0")
    )

    # Step 2 -- Inspect the result
    print(f"Agent version : {graph.version}")
    print(f"Agent ID      : {graph.agent_id}")
    print(f"Start node    : {graph.start_node_id}")
    print(f"Total nodes   : {len(graph.nodes)}")
    print(f"Total edges   : {len(graph.edges)}")

    # Step 3 -- Walk the nodes
    print("\nNodes:")
    for node in graph.nodes:
        print(f"  [{node.type.value:15s}] {node.name!r}  id={node.id}")

    # Step 4 -- Walk the edges
    print("\nEdges:")
    for edge in graph.edges:
        print(f"  {edge.source_id} -> {edge.target_id}")

    # Step 5 -- Verify the graph structure
    start = graph.get_start_node()
    assert start is not None and start.type == NodeType.START
    successors = graph.get_successors(start.id)
    assert len(successors) == 1 and successors[0].type == NodeType.INSTRUCTION
    print("\nGraph structure verified successfully.")


if __name__ == "__main__":
    main()
