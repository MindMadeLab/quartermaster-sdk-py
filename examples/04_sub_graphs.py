"""Composable sub-graphs with .use().

Demonstrates building reusable sub-graphs and inlining them into a
main agent graph via the .use() method. Sub-graph START/END nodes are
stripped and the internal nodes are wired into the parent chain.
Executed with a real LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/04_sub_graphs.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from _runner import run_graph

# --- Reusable sub-graphs ---------------------------------------------------

research_flow = (
    Graph("Research")
    .start()
    .instruction("Search web", model="claude-sonnet-4-20250514", system_instruction="Find relevant information")
    .instruction("Summarize findings", model="claude-sonnet-4-20250514", system_instruction="Create a concise summary")
    .end()
)

code_review_flow = (
    Graph("Code Review")
    .start()
    .instruction("Analyze code", model="claude-sonnet-4-20250514", system_instruction="Review code for bugs and style")
    .instruction("Generate suggestions", model="claude-sonnet-4-20250514", system_instruction="Suggest improvements")
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
        .instruction("Chat", model="claude-sonnet-4-20250514", system_instruction="Have a helpful conversation")
    .end()
    # No merge -- decision picks ONE branch.
    .instruction("Deliver answer", model="claude-sonnet-4-20250514", system_instruction="Present the final result clearly")
    .end()
)

run_graph(agent, user_input="Research the latest trends in WebAssembly for server-side applications")
