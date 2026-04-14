"""Decision flow with two branches that merge.

Demonstrates decision nodes that route to different branches based on
labelled options, then merge back into a single flow. Executed with a
real LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/02_decision_flow.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_engine import run_graph

agent = (
    Graph("Sentiment Analyzer")
    .start()
    .user("Enter text to analyze")
    .instruction("Analyze sentiment", model="claude-haiku-4-5-20251001", system_instruction="Classify as positive or negative")
    .decision("Sentiment?", options=["positive", "negative"])
    .on("positive")
        .instruction("Positive response", model="claude-haiku-4-5-20251001", system_instruction="Generate an enthusiastic response")
    .end()
    .on("negative")
        .instruction("Negative response", model="claude-haiku-4-5-20251001", system_instruction="Generate an empathetic response")
    .end()
    # No merge needed -- decision picks ONE branch, so branches converge
    # directly on the next node.
    .instruction("Final summary", model="claude-haiku-4-5-20251001", system_instruction="Summarize the analysis")
    .end()
)

run_graph(agent, user_input="I absolutely love this product, it changed my life!")
