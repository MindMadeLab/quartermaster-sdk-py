"""Composable sub-graphs with .use().

Demonstrates building reusable sub-graphs and inlining them into a
main agent graph via the .use() method. Sub-graph START/END nodes are
stripped and the internal nodes are wired into the parent chain.
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# --- Reusable sub-graphs ---------------------------------------------------

research_flow = (
    Graph("Research")
    .start()
    .instruction("Search web", system_instruction="Find relevant information")
    .instruction("Summarize findings", system_instruction="Create a concise summary")
    .end()
)

code_review_flow = (
    Graph("Code Review")
    .start()
    .instruction("Analyze code", system_instruction="Review code for bugs and style")
    .instruction("Generate suggestions", system_instruction="Suggest improvements")
    .end()
)

# --- Compose into a main agent ---------------------------------------------

agent = (
    Graph("Dev Assistant")
    .start()
    .user("What do you need help with?")
    .decision("Task type?", options=["research", "code_review", "chat"])
    .on("research").use(research_flow).end()
    .on("code_review").use(code_review_flow).end()
    .on("chat")
        .instruction("Chat", system_instruction="Have a helpful conversation")
    .end()
    # No merge — decision picks ONE branch.
    .instruction("Deliver answer", system_instruction="Present the final result clearly")
    .end()
)

print(f"Dev Assistant: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print("\nAll nodes:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
