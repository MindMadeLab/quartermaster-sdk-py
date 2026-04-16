#!/usr/bin/env python3
"""Interactive agent demo -- continuous conversation with an LLM.

The graph pauses at User nodes and prompts for your input via stdin.
After the LLM responds, ``.back()`` loops control back to Start so the
User node prompts again — no Python ``while True`` needed.  The loop
lives inside the graph itself.  Press Ctrl+C to exit.

Usage:
    uv run examples/run_interactive.py
    uv run examples/run_interactive.py --model claude-haiku-4-5-20251001 --provider anthropic
    uv run examples/run_interactive.py --model gemma4:26b --provider ollama
"""

from __future__ import annotations

import argparse

import quartermaster_sdk as qm


def main():
    parser = argparse.ArgumentParser(description="Interactive Quartermaster agent")
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001", help="Model to use"
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="Provider (anthropic, openai, groq, xai, ollama)",
    )
    args = parser.parse_args()

    # The graph loops natively via .back():
    #   Start → User (stdin) → Instruction (LLM) → Back → Start → User → ...
    # No external while-loop required.
    agent = (
        qm.Graph("Interactive Assistant")
        .user("You")
        .instruction(
            "Respond",
            model=args.model,
            provider=args.provider,
            system_instruction=(
                "You are a helpful assistant. Be concise and clear. "
                "Respond in the same language the user writes in."
            ),
        )
        .back()  # loop back to Start → User prompts again
    )

    print(f"Interactive Assistant ({args.model} via {args.provider})")
    print("Ctrl+C to exit\n")

    try:
        qm.run_graph(agent)
    except (KeyboardInterrupt, EOFError):
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
