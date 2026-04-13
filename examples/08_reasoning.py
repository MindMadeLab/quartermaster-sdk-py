"""Reasoning and summarization nodes.

Demonstrates using reasoning and summarize fluent methods for deep
analytical workflows. These node types trigger chain-of-thought
prompting and condensation steps respectively.
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Research Analyst")
    .start()
    .user("What topic should I research?")
    .reasoning("Deep analysis", model="gpt-4o", system_instruction="Think step by step about this topic")
    .instruction("Gather evidence", system_instruction="List key facts and sources")
    .reasoning("Evaluate evidence", system_instruction="Critically evaluate the evidence")
    .summarize("Executive summary", system_instruction="Create a concise executive summary")
    .end()
)

print(f"Research Analyst: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nPipeline:")
for i, node in enumerate(agent.nodes):
    arrow = "  ->" if i > 0 else "   "
    print(f"{arrow} [{node.type.value:15s}] {node.name}")
