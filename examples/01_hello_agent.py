"""The simplest possible agent: user asks, LLM responds.

Demonstrates the minimal Graph API -- three nodes chained together
with the fluent builder, then executed with a real LLM.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/01_hello_agent.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_engine import run_graph

# The simplest agent: ask user -> LLM responds
agent = (
    Graph("Hello Agent")
    .start()
    .user("Ask me anything")
    .instruction("Respond", model="claude-haiku-4-5-20251001", system_instruction="You are a helpful assistant. Be concise.")
    .end()
)

# Execute with a real LLM
run_graph(agent, user_input="What is the capital of Slovenia?")
