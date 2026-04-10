"""Decision agent example -- an agent with branching logic.

This demonstrates how to build an agent that makes decisions and
follows different paths based on the outcome. The graph uses a
decision node with labeled branches.

Prerequisites:
    pip install qm-graph qm-engine
"""

from qm_graph import GraphBuilder
from qm_engine import FlowRunner
from qm_engine.nodes import SimpleNodeRegistry

# ------------------------------------------------------------------
# Step 1: Build a graph with decision branching
# ------------------------------------------------------------------
# After a decision node, use .on("label") to define what happens
# for each branch. Each branch can have its own chain of nodes
# and must end with .end() to terminate or .merge_to() to rejoin.

graph = (
    GraphBuilder("Sentiment Router", description="Routes input by sentiment")
    .start()

    # First, classify the sentiment of the input
    .instruction(
        "Classify sentiment",
        model="gpt-4o",
        provider="openai",
        system_instruction=(
            "Classify the sentiment of the user's message as exactly one of: "
            "Positive, Negative, or Neutral. Reply with only the label."
        ),
    )

    # Decision node: the LLM's output determines which branch to follow
    .decision("Route by sentiment", options=["Positive", "Negative", "Neutral"])

    # Branch: Positive sentiment
    .on("Positive")
        .instruction(
            "Positive response",
            model="gpt-4o",
            provider="openai",
            system_instruction="The user expressed something positive. Respond enthusiastically and encourage them.",
        )
        .end()

    # Branch: Negative sentiment
    .on("Negative")
        .instruction(
            "Negative response",
            model="gpt-4o",
            provider="openai",
            system_instruction="The user expressed something negative. Respond with empathy and offer constructive suggestions.",
        )
        .end()

    # Branch: Neutral sentiment
    .on("Neutral")
        .instruction(
            "Neutral response",
            model="gpt-4o",
            provider="openai",
            system_instruction="The user's message is neutral. Provide a balanced, informative response.",
        )
        .end()

    .build()
)

# ------------------------------------------------------------------
# Step 2: Set up the node registry and run
# ------------------------------------------------------------------

node_registry = SimpleNodeRegistry()
runner = FlowRunner(graph=graph, node_registry=node_registry)

# ------------------------------------------------------------------
# Step 3: Run with different inputs
# ------------------------------------------------------------------

inputs = [
    "I just got promoted at work!",
    "My project deadline was missed and the client is upset.",
    "The weather forecast says it will be 20 degrees tomorrow.",
]

for user_input in inputs:
    print(f"\n{'=' * 60}")
    print(f"Input: {user_input}")
    print(f"{'=' * 60}")

    result = runner.run(user_input)

    print(f"Success: {result.success}")
    print(f"Output: {result.final_output}")

    if result.error:
        print(f"Error: {result.error}")
