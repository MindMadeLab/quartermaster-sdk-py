"""Nested control flow inside parallel branches -- the whiteboard pattern.

Demonstrates the most advanced graph pattern: parallel fan-out where
individual branches contain their own IF decisions and nested logic,
then everything merges back together.

This is the "whiteboard" pattern -- the kind of graph you'd sketch on
a whiteboard when designing a complex agent pipeline.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/11_nested_control_flow.py
"""

from __future__ import annotations

import quartermaster_sdk as qm

agent = (
    qm.Graph("Whiteboard Agent")
    .user("Describe your task")
    .instruction(
        "Analyse input",
        model="claude-haiku-4-5-20251001",
        system_instruction="Break the task into components for parallel processing",
    )
    # --- Parallel fan-out: 3 branches with nested control flow ----------------
    .parallel("Fan out")
    # Branch A: deep analysis pipeline
    .branch()
    .text("Prepare context", template="Task context: {{user_input}}")
    .instruction(
        "Deep analysis",
        model="claude-haiku-4-5-20251001",
        system_instruction="Perform thorough analysis of the task",
    )
    .end()
    # Branch B: independent lightweight check (pass-through)
    .branch()
    .instruction(
        "Quick check",
        model="claude-haiku-4-5-20251001",
        system_instruction="Perform a fast independent assessment",
    )
    .end()
    # Branch C: conditional quality gate (IF picks one path — no merge needed)
    .branch()
    .if_node("Confidence high?", expression="confidence > 0.7")
    .on("true")
    .text("Accept result", template="Result meets confidence threshold -- approved")
    .end()
    .on("false")
    .text("Flag for review", template="Low confidence -- manual review recommended")
    .end()
    # IF branches converge on this static node, which becomes the branch endpoint
    .static("Quality gate result", text="Quality gate complete")
    .end()
    .static_merge("Combine all branches")
    # --- Final synthesis -------------------------------------------------------
    .summarize(
        "Final synthesis",
        model="claude-haiku-4-5-20251001",
        system_instruction="Combine all branch results into a coherent response",
    )
)

# Execute with a real LLM
qm.run_graph(
    agent,
    user_input="Design a Python microservice that handles user authentication with JWT tokens and rate limiting",
)
