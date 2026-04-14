"""Example 23 -- Image vision analysis.

Demonstrates the vision node for analyzing images with multimodal LLMs.
The graph takes an image URL, analyzes it with a vision-capable model,
then generates a structured description.

Uses Claude's vision capabilities to process images alongside text prompts.

Usage:
    uv run examples/23_vision.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_engine import run_graph


# -- Build vision pipeline --------------------------------------------------

agent = (
    Graph("Image Analyzer")
    .start()
    .user("Provide an image URL or description")

    # Vision node — analyzes the image
    .vision(
        "Analyze image",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "You are an expert image analyst. The user will describe or provide "
            "an image. Analyze it thoroughly:\n"
            "1. Main subject and composition\n"
            "2. Colors, lighting, mood\n"
            "3. Notable details and context\n"
            "4. Technical quality assessment\n\n"
            "Be specific and observational."
        ),
    )

    # Generate structured output from the analysis
    .instruction(
        "Structure findings",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "Take the image analysis and create a structured report:\n\n"
            "## Image Analysis Report\n"
            "### Subject\n"
            "### Composition & Style\n"
            "### Technical Details\n"
            "### Suggested Use Cases\n\n"
            "Be concise but thorough."
        ),
    )
    .end()
)

# -- Run --------------------------------------------------------------------

print("Image Vision Analysis Pipeline")
print("Note: Vision node analyzes images using multimodal LLMs.")
print("Without an actual image, the model analyzes based on the text description.")
print()

run_graph(
    agent,
    user_input=(
        "A dramatic sunset over the Swiss Alps, with snow-capped peaks "
        "reflecting golden and purple light. A small wooden cabin sits "
        "in the foreground with warm light glowing from its windows. "
        "The sky has layers of orange, pink, and deep blue clouds."
    ),
)
