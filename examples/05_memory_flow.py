"""Variable management and memory nodes.

Demonstrates storing values in variables, writing them to long-term
memory, and reading them back in later nodes using dedicated fluent
methods.
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Memory Agent")
    .start()
    .user("What's your name?")
    .var("Store name", variable="user_name")
    .instruction("Greet user", system_instruction="Greet {{user_name}} warmly")
    .write_memory("Remember user", key="user_name")
    .user("Ask me something")
    .read_memory("Recall user", key="user_name")
    .instruction("Personalized response", system_instruction="Respond using the user's name")
    .end()
)

print(f"Memory Agent: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nGraph structure:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
    if node.metadata:
        for k, v in node.metadata.items():
            print(f"      {k}: {v}")
