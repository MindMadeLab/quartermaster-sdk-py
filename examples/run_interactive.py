#!/usr/bin/env python3
"""Interactive agent demo -- runs a real conversation with an LLM provider.

The graph pauses at User nodes and prompts you for input via stdin.
Your message is then classified and routed through either a quick
analysis or deep research pipeline.

Usage:
    uv run examples/run_interactive.py
    uv run examples/run_interactive.py --provider anthropic
    uv run examples/run_interactive.py --provider openai
"""

from __future__ import annotations

import argparse

from quartermaster_graph import Graph
from quartermaster_engine import run_graph


def build_assistant_graph(model: str) -> Graph:
    """Build an interactive research assistant graph."""

    quick_analysis = (
        Graph("Quick Analysis")
        .start()
        .instruction(
            "Analyse", model=model,
            system_instruction="Provide a clear, concise analysis. Structure with key points.",
        )
        .end()
    )

    deep_research = (
        Graph("Deep Research")
        .start()
        .instruction("Break down", model=model, system_instruction="Break the question into 2-3 research angles.")
        .parallel()
        .branch()
            .instruction("Factual research", model=model, system_instruction="Research factual and technical aspects.")
        .end()
        .branch()
            .instruction("Perspective analysis", model=model, system_instruction="Analyse different perspectives.")
        .end()
        .static_merge("Combine research")
        .instruction(
            "Synthesize", model=model,
            system_instruction="Synthesize research into a comprehensive response with facts and perspectives.",
        )
        .end()
    )

    return (
        Graph("Interactive Assistant")
        .start()
        .user("What would you like to explore?")
        .instruction(
            "Classify", model=model,
            system_instruction="Classify as 'quick' or 'deep'. Respond with exactly one word.",
        )
        .decision("Route", options=["quick", "deep"])
        .on("quick").use(quick_analysis).end()
        .on("deep").use(deep_research).end()
        .end()
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive Quartermaster agent demo")
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "xai"])
    args = parser.parse_args()

    agent = build_assistant_graph("claude-haiku-4-5-20251001")

    # interactive=True: User nodes prompt stdin instead of using a hardcoded input
    run_graph(agent, provider=args.provider)


if __name__ == "__main__":
    main()
