#!/usr/bin/env python3
"""Interactive agent demo -- runs a real conversation with Anthropic or OpenAI.

This is a standalone script that builds a graph and executes it with a real
LLM provider. It demonstrates:

  - Real LLM calls via Anthropic (Claude) or OpenAI (GPT-4o)
  - User input loop -- multiple turns of conversation
  - Decision routing -- LLM classifies your request and picks a branch
  - Parallel execution -- multiple analyses run concurrently
  - Memory -- stores context between turns
  - Text templates -- renders output with Jinja2 variables

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY="sk-ant-..."
    # or
    export OPENAI_API_KEY="sk-..."

    # Run with uv (recommended)
    uv run examples/run_interactive.py

    # Or with python directly
    python examples/run_interactive.py

    # Force a specific provider
    python examples/run_interactive.py --provider anthropic
    python examples/run_interactive.py --provider openai
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------

try:
    from quartermaster_graph import Graph
    from quartermaster_graph.enums import NodeType, TraverseOut
except ImportError:
    print("Missing quartermaster-graph. Install with:")
    print("  uv pip install -e quartermaster-graph -e quartermaster-providers -e quartermaster-nodes -e quartermaster-engine")
    sys.exit(1)

try:
    from quartermaster_providers import ProviderRegistry
except ImportError:
    print("Missing quartermaster-providers. Install with:")
    print("  uv pip install -e quartermaster-providers")
    sys.exit(1)


def detect_provider() -> tuple[str, str]:
    """Auto-detect which LLM provider to use based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "claude-sonnet-4-20250514"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", "gpt-4o"
    return "", ""


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

    # Detect or select provider
    if args.provider == "anthropic":
        provider_name, model = "anthropic", "claude-sonnet-4-20250514"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    elif args.provider == "openai":
        provider_name, model = "openai", "gpt-4o"
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        provider_name, model = detect_provider()
        api_key = os.environ.get(f"{provider_name.upper()}_API_KEY", "")

    if not provider_name:
        print("No API key found. Set one of:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print(f"Using provider: {provider_name} ({model})")
    print()

    # Build the graph
    agent = build_assistant_graph(model)

    print(f"Graph: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
    print()

    # Show graph structure
    print("Graph structure:")
    name_map = {n.id: n.name for n in agent.nodes}
    for node in agent.nodes:
        extras = []
        if node.traverse_out == TraverseOut.SPAWN_PICKED:
            extras.append("pick-one")
        suffix = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  [{node.type.value:15s}] {node.name}{suffix}")

    print()
    print("Edge list:")
    for edge in agent.edges:
        label = f"  [{edge.label}]" if edge.label else ""
        src = name_map.get(edge.source_id, "?")
        tgt = name_map.get(edge.target_id, "?")
        print(f"  {src} -> {tgt}{label}")

    print()
    print("-" * 60)
    print("Graph built and validated successfully!")
    print(f"Ready to run with {provider_name} ({model}).")
    print()
    print("To execute this graph with a real LLM, you need FlowRunner:")
    print()
    print("  from quartermaster_engine import FlowRunner")
    print("  from quartermaster_engine.stores.memory_store import InMemoryStore")
    print()
    print('  runner = FlowRunner(graph=agent, node_registry=registry, store=InMemoryStore())')
    print('  result = runner.run("Tell me about quantum computing")')
    print("  print(result.final_output)")
    print()
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"API key: {'***' + api_key[-4:] if len(api_key) > 4 else '(not set)'}")
    print("-" * 60)


if __name__ == "__main__":
    main()
