"""Variable management and memory nodes.

Demonstrates storing values in variables, writing them to long-term
memory, and reading them back in later nodes. Uses the generic
.node() method for memory/variable node types.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Memory Agent")
    .start()
    .user("What's your name?")
    .node(NodeType.VAR, "Store name", metadata={"variable": "user_name"})
    .instruction("Greet user", system_instruction="Greet {{user_name}} warmly")
    .node(NodeType.WRITE_MEMORY, "Remember user", metadata={"key": "user_name"})
    .user("Ask me something")
    .node(NodeType.READ_MEMORY, "Recall user", metadata={"key": "user_name"})
    .instruction("Personalized response", system_instruction="Respond using the user's name")
    .end()
    .build(version="1.0.0")
)

print(f"Memory Agent: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nGraph structure:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
    if node.metadata:
        for k, v in node.metadata.items():
            print(f"      {k}: {v}")
