"""Build a graph with If/Decision branching.

Demonstrates decision nodes that route execution to different branches
based on labelled options, and the if_node for boolean branching.
"""

from __future__ import annotations

try:
    from qm_graph.builder import GraphBuilder
    from qm_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install qm-graph first:  pip install -e qm-graph")


def main() -> None:
    # -- Decision node example ------------------------------------------------
    # Start -> Decision("Sentiment?") -> Yes branch / No branch -> End
    graph = (
        GraphBuilder("Sentiment Router")
        .start()
        .instruction("Analyze sentiment", system_instruction="Classify the input as positive or negative.")
        .decision("Is it positive?", options=["Yes", "No"])
        .on("Yes").instruction("Positive path", system_instruction="Respond enthusiastically.").end()
        .on("No").instruction("Negative path", system_instruction="Respond empathetically.").end()
        .build(version="0.1.0")
    )

    print("=== Decision Graph ===")
    print(f"Nodes: {len(graph.nodes)}   Edges: {len(graph.edges)}")
    for node in graph.nodes:
        print(f"  {node.type.value:15s}  {node.name}")
    for edge in graph.edges:
        label = f" [{edge.label}]" if edge.label else ""
        print(f"  Edge: {edge.source_id} -> {edge.target_id}{label}")

    # -- If-node example ------------------------------------------------------
    # Start -> If("has_api_key") -> true/false branches -> End
    if_graph = (
        GraphBuilder("API Key Check")
        .start()
        .if_node("Check key", expression="has_api_key == true", variable="has_api_key")
        .on("true").instruction("Call API", system_instruction="Make the API call.").end()
        .on("false").static("No key", content="Please set your API key.").end()
        .build(version="0.1.0")
    )

    print("\n=== If-Node Graph ===")
    print(f"Nodes: {len(if_graph.nodes)}   Edges: {len(if_graph.edges)}")
    for node in if_graph.nodes:
        print(f"  {node.type.value:15s}  {node.name}")


if __name__ == "__main__":
    main()
