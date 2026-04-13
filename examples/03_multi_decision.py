"""Multiple decision/if nodes in sequence.

Demonstrates chaining several branching nodes one after another:
a decision for categorization, an if-node for urgency, and a
decision for user feedback -- all in a single graph.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Multi-Stage Classifier")
    .start()
    .user("Describe your issue")

    # First decision: categorize the request
    .decision("Category?", options=["technical", "billing", "general"])
    .on("technical")
        .instruction("Tech triage", system_instruction="Assess technical severity")
    .end()
    .on("billing")
        .instruction("Billing check", system_instruction="Look up account status")
    .end()
    .on("general")
        .instruction("General info", system_instruction="Provide general help")
    .end()
    .merge("After categorization")

    # Second decision: urgency check (boolean if-node)
    .if_node("Is urgent?", expression="severity == 'high'")
    .on("true")
        .instruction("Escalate", system_instruction="Create urgent ticket")
        .node(NodeType.NOTIFICATION, "Alert team", metadata={"channel": "slack", "message": "Urgent issue!"})
    .end()
    .on("false")
        .instruction("Standard response", system_instruction="Provide standard help")
    .end()
    .merge("After urgency check")

    # Third decision: user feedback (manual choice)
    .decision("Was this helpful?", options=["yes", "no"])
    .on("yes")
        .instruction("Thank user", system_instruction="Thank them and close")
    .end()
    .on("no")
        .instruction("Escalate to human", system_instruction="Transfer to human agent")
    .end()
    .end()
    .build(version="1.0.0")
)

print(f"Built multi-decision agent with {len(agent.nodes)} nodes and {len(agent.edges)} edges")
print("\nNode types used:")
node_types: dict[str, int] = {}
for n in agent.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
for name, count in sorted(node_types.items()):
    print(f"  {name}: {count}")
