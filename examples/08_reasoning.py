"""Reasoning and summarization nodes.

Demonstrates using REASONING and SUMMARIZE node types for deep
analytical workflows. These node types trigger chain-of-thought
prompting and condensation steps respectively.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Research Analyst")
    .start()
    .user("What topic should I research?")
    .node(
        NodeType.REASONING,
        "Deep analysis",
        metadata={"model": "gpt-4o", "system_instruction": "Think step by step about this topic"},
    )
    .instruction("Gather evidence", system_instruction="List key facts and sources")
    .node(
        NodeType.REASONING,
        "Evaluate evidence",
        metadata={"system_instruction": "Critically evaluate the evidence"},
    )
    .node(
        NodeType.SUMMARIZE,
        "Executive summary",
        metadata={"system_instruction": "Create a concise executive summary"},
    )
    .end()
    .build(version="1.0.0")
)

print(f"Research Analyst: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nPipeline:")
for i, node in enumerate(agent.nodes):
    arrow = "  ->" if i > 0 else "   "
    print(f"{arrow} [{node.type.value:15s}] {node.name}")
