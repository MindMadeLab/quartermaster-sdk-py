"""Reasoning and summarization nodes.

Demonstrates using reasoning and summarize fluent methods for deep
analytical workflows. These node types trigger chain-of-thought
prompting and condensation steps respectively. Executed with a real
LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/08_reasoning.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from _runner import run_graph

agent = (
    Graph("Research Analyst")
    .start()
    .user("What topic should I research?")
    .reasoning("Deep analysis", model="claude-sonnet-4-20250514")
    .instruction("Gather evidence", model="claude-sonnet-4-20250514", system_instruction="List key facts and sources")
    .reasoning("Evaluate evidence", model="claude-sonnet-4-20250514")
    .summarize("Executive summary", model="claude-sonnet-4-20250514", system_instruction="Create a concise executive summary")
    .end()
)

run_graph(agent, user_input="The impact of large language models on software engineering productivity")
