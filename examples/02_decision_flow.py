"""Decision flow with two branches that merge.

Demonstrates decision nodes that route to different branches based on
labelled options, then merge back into a single flow.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Sentiment Analyzer")
    .start()
    .user("Enter text to analyze")
    .instruction("Analyze sentiment", system_instruction="Classify as positive or negative")
    .decision("Sentiment?", options=["positive", "negative"])
    .on("positive")
        .instruction("Positive response", system_instruction="Generate an enthusiastic response")
    .end()
    .on("negative")
        .instruction("Negative response", system_instruction="Generate an empathetic response")
    .end()
    .merge("Combine results")
    .instruction("Final summary", system_instruction="Summarize the analysis")
    .end()
    .build(version="1.0.0")
)

print(f"Nodes: {len(agent.nodes)}   Edges: {len(agent.edges)}")
print("\nGraph structure:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
for edge in agent.edges:
    label = f" [{edge.label}]" if edge.label else ""
    print(f"  Edge: {edge.source_id} -> {edge.target_id}{label}")
