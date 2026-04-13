"""The simplest possible agent: user asks, LLM responds.

Demonstrates the minimal Graph API -- three nodes chained together
with the fluent builder.
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# The simplest agent: ask user -> LLM responds
agent = (
    Graph("Hello Agent")
    .start()
    .user("Ask me anything")
    .instruction("Respond", model="gpt-4o", system_instruction="You are a helpful assistant.")
    .end()
)

print(f"Nodes: {len(agent.nodes)}")
print(f"Edges: {len(agent.edges)}")

# Walk the graph
print("\nGraph structure:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
for edge in agent.edges:
    print(f"  Edge: {edge.source_id} -> {edge.target_id}")
