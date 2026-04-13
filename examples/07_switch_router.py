"""Switch node with multiple branches.

Demonstrates a multi-way switch (more than two branches) for routing
based on detected values. Uses a decision node with many options to
model a switch/case pattern.
"""

from __future__ import annotations

try:
    from quartermaster_graph import GraphBuilder as Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# Multi-language support agent with switch-style routing
agent = (
    Graph("Multi-Language Agent")
    .start()
    .user("Enter your message")
    .instruction("Detect language", system_instruction="Detect the language. Output: en/es/fr/de/other")
    .decision("Language?", options=["en", "es", "fr", "de", "other"])
    .on("en").instruction("English handler", system_instruction="Respond in English").end()
    .on("es").instruction("Spanish handler", system_instruction="Responde en espanol").end()
    .on("fr").instruction("French handler", system_instruction="Repondez en francais").end()
    .on("de").instruction("German handler", system_instruction="Antworten Sie auf Deutsch").end()
    .on("other").instruction("Fallback", system_instruction="Respond in English, note language").end()
    .merge("Collect response")
    .end()
    .build(version="1.0.0")
)

print(f"Language agent: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nBranch structure:")
for edge in agent.edges:
    label = f" [{edge.label}]" if edge.label else ""
    print(f"  {edge.source_id} -> {edge.target_id}{label}")
