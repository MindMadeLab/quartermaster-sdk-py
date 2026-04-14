"""Multiple decision/if nodes in sequence.

Demonstrates chaining several branching nodes one after another:
a decision for categorization, an if-node for urgency, and a
decision for user feedback -- all in a single graph. Executed with
a real LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/03_multi_decision.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_engine import run_graph

agent = (
    Graph("Multi-Stage Classifier")
    .start()
    .user("Describe your issue")

    # First decision: categorize the request
    .decision("Category?", options=["technical", "billing", "general"])
    .on("technical")
        .instruction("Tech triage", model="claude-haiku-4-5-20251001", system_instruction="Assess technical severity")
    .end()
    .on("billing")
        .instruction("Billing check", model="claude-haiku-4-5-20251001", system_instruction="Look up account status")
    .end()
    .on("general")
        .instruction("General info", model="claude-haiku-4-5-20251001", system_instruction="Provide general help")
    .end()
    # No merge -- decision picks ONE branch, they converge on the next node.

    # Second decision: urgency check (boolean if-node)
    .if_node("Is urgent?", expression="severity == 'high'")
    .on("true")
        .instruction("Escalate", model="claude-haiku-4-5-20251001", system_instruction="Create urgent ticket")
        .static("Alert team", text="Urgent issue!")
    .end()
    .on("false")
        .instruction("Standard response", model="claude-haiku-4-5-20251001", system_instruction="Provide standard help")
    .end()
    # No merge -- IF picks ONE branch.

    # Third decision: user feedback (manual choice)
    .decision("Was this helpful?", options=["yes", "no"])
    .on("yes")
        .instruction("Thank user", model="claude-haiku-4-5-20251001", system_instruction="Thank them and close")
    .end()
    .on("no")
        .instruction("Escalate to human", model="claude-haiku-4-5-20251001", system_instruction="Transfer to human agent")
    .end()
    .end()
)

run_graph(agent, user_input="My server keeps crashing with out-of-memory errors every few hours")
