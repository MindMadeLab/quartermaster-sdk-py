"""Example 13 -- Parallel orchestrator with session-based sub-agents.

Demonstrates the pattern where an Agent node (the orchestrator) spawns
multiple sub-agents in parallel via ``spawn_agent``, waits for all of
them with ``collect_agent_results``, and synthesises a final answer.

Key concepts
~~~~~~~~~~~~
* ``allowed_agents()`` -- restricts which agent IDs the orchestrator can spawn.
* ``spawn_agent`` tool -- creates & starts a session in one call (runs in a
  background thread).  The LLM can call it multiple times in a single
  turn to launch agents in parallel.
* ``collect_agent_results`` tool -- blocks until every listed session
  completes (or times out), then returns combined results.
* ``notify_parent`` tool -- sub-agents can push status updates back to
  the orchestrator's session via a webhook-style notification.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/13_orchestrator.py
"""

from quartermaster_graph import Graph
from quartermaster_engine import run_graph

# -- Build the graph ----------------------------------------------------------
graph = (
    Graph("ParallelOrchestrator")
    .allowed_agents("researcher", "writer", "reviewer")
    .start()
    # 1. Collect the user's request
    .user("Describe the project you'd like researched, written, and reviewed.")
    # 2. Orchestrator -- an Agent node with session-management tools
    .agent(
        "Orchestrator",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "You are a project manager.  Given the user's request:\n"
            "1. Spawn three agents IN PARALLEL using spawn_agent:\n"
            "   - researcher  -- deep-dive research on the topic\n"
            "   - writer      -- draft the document\n"
            "   - reviewer    -- prepare a review checklist\n"
            "2. Call collect_agent_results with all three session IDs.\n"
            "3. Return a combined status summary so the next node can "
            "synthesise the final output."
        ),
        tools=[
            "spawn_agent",
            "collect_agent_results",
            "get_agent_session_status",
            "cancel_agent_session",
        ],
        max_iterations=15,
    )
    # 3. Final summariser -- pure Instruction (no tools), just LLM text
    .instruction(
        "Summariser",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "Using the orchestrator's collected results, produce a polished "
            "final document that integrates the research, the draft, and the "
            "reviewer's feedback."
        ),
    )
    .end()
)

# Execute with a real LLM
run_graph(graph, user_input="Write a technical blog post about WebAssembly's role in server-side computing")
