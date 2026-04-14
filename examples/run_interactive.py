#!/usr/bin/env python3
"""Interactive agent demo -- runs a real conversation with an LLM provider.

Builds a graph with decision routing, parallel research branches, and
memory, then executes it with a real LLM via the _runner helper.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/run_interactive.py

    # Force a specific provider
    uv run examples/run_interactive.py --provider anthropic
    uv run examples/run_interactive.py --provider openai
"""

from __future__ import annotations

import argparse

from quartermaster_graph import Graph
from quartermaster_engine import run_graph


def build_assistant_graph(model: str) -> Graph:
    """Build an interactive research assistant graph."""

    # Sub-graph: quick analysis pipeline
    quick_analysis = (
        Graph("Quick Analysis")
        .start()
        .instruction(
            "Analyse",
            model=model,
            system_instruction=(
                "Provide a clear, concise analysis of the user's question. "
                "Structure your response with key points."
            ),
        )
        .end()
    )

    # Sub-graph: deep research pipeline with parallel branches
    deep_research = (
        Graph("Deep Research")
        .start()
        .instruction(
            "Break down",
            model=model,
            system_instruction="Break the question into 2-3 research angles.",
        )
        .parallel()
        .branch()
            .instruction(
                "Factual research",
                model=model,
                system_instruction="Research the factual and technical aspects of this topic.",
            )
        .end()
        .branch()
            .instruction(
                "Perspective analysis",
                model=model,
                system_instruction="Analyse different perspectives and opinions on this topic.",
            )
        .end()
        .static_merge("Combine research")
        .instruction(
            "Synthesize",
            model=model,
            system_instruction=(
                "Synthesize the research from both angles into a comprehensive, "
                "well-structured response. Include key facts and multiple perspectives."
            ),
        )
        .end()
    )

    # Main graph
    agent = (
        Graph("Interactive Assistant")
        .start()
        .user("What would you like to explore?")
        .var("Capture question", variable="user_question")

        # Classify the request
        .instruction(
            "Classify complexity",
            model=model,
            system_instruction=(
                "Classify this request as either 'quick' (simple factual question, "
                "greeting, or brief task) or 'deep' (complex topic needing research "
                "from multiple angles). Respond with exactly one word: quick or deep."
            ),
        )

        # Route based on complexity
        .decision("Complexity?", options=["quick", "deep"])

        .on("quick")
            .use(quick_analysis)
        .end()

        .on("deep")
            .use(deep_research)
        .end()

        # Decision picks ONE branch -- no merge needed.

        .write_memory("Save exchange", memory_name="last_topic")
        .end()
    )

    return agent


def main():
    parser = argparse.ArgumentParser(description="Interactive Quartermaster agent demo")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"],
        help="Force a specific provider (default: auto-detect from API keys)",
    )
    args = parser.parse_args()

    # Build the graph with a placeholder model; _runner will override
    # with the detected provider's default model anyway.
    model = "claude-haiku-4-5-20251001"
    agent = build_assistant_graph(model)

    run_graph(
        agent,
        user_input="What are the trade-offs between microservices and monolithic architectures for a startup?",
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
