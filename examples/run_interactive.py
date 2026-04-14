#!/usr/bin/env python3
"""Interactive agent demo -- continuous conversation with an LLM.

The graph pauses at User nodes and prompts for your input via stdin.
After the LLM responds, it loops back for your next question.
Press Ctrl+C to exit.

Usage:
    uv run examples/run_interactive.py
    uv run examples/run_interactive.py --provider openai
"""

from __future__ import annotations

import argparse

from quartermaster_graph import Graph
from quartermaster_engine import run_graph


def main():
    parser = argparse.ArgumentParser(description="Interactive Quartermaster agent")
    parser.add_argument("--provider", choices=["anthropic", "openai", "groq", "xai"])
    args = parser.parse_args()

    agent = (
        Graph("Interactive Assistant")
        .start()
        .user("You")
        .instruction(
            "Respond",
            system_instruction=(
                "You are a helpful assistant. Be concise and clear. "
                "Respond in the same language the user writes in."
            ),
        )
        .end()
    )

    print("Interactive Assistant (Ctrl+C to exit)\n")

    while True:
        try:
            result = run_graph(agent, provider=args.provider)
            if result is None:
                break
        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
