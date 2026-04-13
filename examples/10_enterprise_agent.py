"""Complex enterprise agent combining sub-graphs, decisions, if-nodes, memory, and logging.

The big showcase -- demonstrates composing multiple sub-graphs, chaining
several branching nodes, using memory and notification node types, and
building a production-style multi-department support agent.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# --- Sub-graphs for different departments -----------------------------------

hr_flow = (
    Graph("HR Handler")
    .start()
    .instruction("HR analysis", system_instruction="Analyze HR-related query")
    .if_node("Needs approval?", expression="requires_manager_approval")
    .on("true")
        .node(NodeType.NOTIFICATION, "Alert manager", metadata={"channel": "email"})
        .user("Awaiting manager response")
    .end()
    .on("false")
        .instruction("Direct response", system_instruction="Provide HR information")
    .end()
    .end()
    .build(version="1.0.0")
)

it_flow = (
    Graph("IT Handler")
    .start()
    .instruction("IT diagnosis", system_instruction="Diagnose the IT issue")
    .instruction("Suggest fix", system_instruction="Provide troubleshooting steps")
    .end()
    .build(version="1.0.0")
)

# --- Main enterprise agent --------------------------------------------------

agent = (
    Graph("Enterprise Assistant")
    .start()
    .user("How can I help you today?")
    .instruction("Classify request", system_instruction="Classify: hr/it/finance/general")

    .decision("Department?", options=["hr", "it", "finance", "general"])
    .on("hr").use(hr_flow).end()
    .on("it").use(it_flow).end()
    .on("finance")
        .instruction("Finance query", system_instruction="Handle finance questions")
        .node(NodeType.WRITE_MEMORY, "Log query", metadata={"key": "last_finance_query"})
    .end()
    .on("general")
        .instruction("General help", system_instruction="Provide general assistance")
    .end()
    .merge("Collect department response")

    # Quality check
    .instruction("Quality check", system_instruction="Review the response for quality")
    .if_node("Quality OK?", expression="quality_score > 0.8")
    .on("true")
        .instruction("Deliver", system_instruction="Format and deliver the final response")
    .end()
    .on("false")
        .instruction("Improve", system_instruction="Improve the response quality")
        .instruction("Re-deliver", system_instruction="Format and deliver improved response")
    .end()
    .merge("Final output")

    .node(NodeType.LOG, "Audit", metadata={"message": "Request completed", "level": "info"})
    .end()
    .build(version="1.0.0")
)

# --- Print graph stats -------------------------------------------------------

print("Enterprise Agent")
print(f"  Nodes: {len(agent.nodes)}")
print(f"  Edges: {len(agent.edges)}")

node_types: dict[str, int] = {}
for n in agent.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
print(f"  Node types: {dict(sorted(node_types.items()))}")

print("\nFull node list:")
for node in agent.nodes:
    meta_summary = ""
    if "system_instruction" in node.metadata:
        meta_summary = f' -- "{node.metadata["system_instruction"][:50]}"'
    print(f"  [{node.type.value:15s}] {node.name}{meta_summary}")
