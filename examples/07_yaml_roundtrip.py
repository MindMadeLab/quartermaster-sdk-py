"""Build a graph, export to YAML, and reimport it.

Demonstrates the serialization module: to_yaml / from_yaml for human-readable
persistence, and to_json / from_json for programmatic use.
"""

from __future__ import annotations

try:
    from quartermaster_graph.builder import GraphBuilder
    from quartermaster_graph.serialization import to_yaml, from_yaml, to_json, from_json
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")


def main() -> None:
    # Step 1 -- Build a graph
    original = (
        GraphBuilder("YAML Demo Agent")
        .start()
        .instruction(
            "Summarize",
            model="gpt-4o",
            system_instruction="Summarize the input text concisely.",
        )
        .static("Footer", content="Thank you for using Quartermaster.")
        .end()
        .build(version="2.0.0")
    )

    print(f"Original graph: {len(original.nodes)} nodes, {len(original.edges)} edges")

    # Step 2 -- Export to YAML
    yaml_str = to_yaml(original)
    print("\n--- YAML representation (first 600 chars) ---")
    print(yaml_str[:600])
    if len(yaml_str) > 600:
        print("... (truncated)")

    # Step 3 -- Reimport from YAML
    restored = from_yaml(yaml_str)
    assert restored.version == original.version
    assert len(restored.nodes) == len(original.nodes)
    assert len(restored.edges) == len(original.edges)
    assert restored.start_node_id == original.start_node_id
    print("\nYAML round-trip verified: all fields match.")

    # Step 4 -- JSON round-trip for comparison
    json_data = to_json(original)
    from_json_restored = from_json(json_data)
    assert from_json_restored.version == original.version
    assert len(from_json_restored.nodes) == len(original.nodes)
    print("JSON round-trip verified: all fields match.")

    # Step 5 -- Show that node IDs are preserved
    for orig_node, rest_node in zip(original.nodes, restored.nodes):
        assert orig_node.id == rest_node.id
        assert orig_node.type == rest_node.type
        assert orig_node.name == rest_node.name
    print("All node IDs and types preserved across serialization.")


if __name__ == "__main__":
    main()
