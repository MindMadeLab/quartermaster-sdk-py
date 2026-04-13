"""Example 13 — Parallel orchestrator with session-based sub-agents.

Demonstrates the pattern where an Agent node (the orchestrator) spawns
multiple sub-agents in parallel via ``spawn_agent``, waits for all of
them with ``collect_agent_results``, and synthesises a final answer.

Key concepts
~~~~~~~~~~~~
* ``allowed_agents()`` — restricts which agent IDs the orchestrator can spawn.
* ``spawn_agent`` tool — creates & starts a session in one call (runs in a
  background thread).  The LLM can call it multiple times in a single
  turn to launch agents in parallel.
* ``collect_agent_results`` tool — blocks until every listed session
  completes (or times out), then returns combined results.
* ``notify_parent`` tool — sub-agents can push status updates back to
  the orchestrator's session via a webhook-style notification.

Graph topology
~~~~~~~~~~~~~~
::

    START
      │
      ▼
    [User] "Describe the project"
      │
      ▼
    [Orchestrator]  ← Agent node with 4 session tools
      │   ├─ spawn_agent("researcher", ...)
      │   ├─ spawn_agent("writer", ...)
      │   └─ spawn_agent("reviewer", ...)
      │
      │  (orchestrator calls collect_agent_results)
      │
      ▼
    [Summariser]  ← Instruction node (no tools) renders final answer
      │
      ▼
    END
"""

from quartermaster_graph import Graph, NodeType

# ── Build the graph ──────────────────────────────────────────────────
graph = (
    Graph("ParallelOrchestrator")
    .allowed_agents("researcher", "writer", "reviewer")
    .start()
    # 1. Collect the user's request
    .user("Describe the project you'd like researched, written, and reviewed.")
    # 2. Orchestrator — an Agent node with session-management tools
    .agent(
        "Orchestrator",
        model="gpt-4o",
        system_instruction=(
            "You are a project manager.  Given the user's request:\n"
            "1. Spawn three agents IN PARALLEL using spawn_agent:\n"
            "   • researcher  — deep-dive research on the topic\n"
            "   • writer      — draft the document\n"
            "   • reviewer    — prepare a review checklist\n"
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
    # 3. Final summariser — pure Instruction (no tools), just LLM text
    .instruction(
        "Summariser",
        model="gpt-4o",
        system_instruction=(
            "Using the orchestrator's collected results, produce a polished "
            "final document that integrates the research, the draft, and the "
            "reviewer's feedback."
        ),
    )
    .end()
)


# ── Inspect the graph ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Graph : {graph._name}")
    print(f"Nodes : {len(graph.nodes)}")
    print(f"Edges : {len(graph.edges)}")
    print(f"Allowed agents: {graph._allowed_agents}")
    print()

    for node in graph.nodes:
        label = f"[{node.type.value}] {node.name}"
        if node.type == NodeType.AGENT:
            tools = node.metadata.get("program_version_ids", [])
            iters = node.metadata.get("max_iterations", "?")
            print(f"  {label}  tools={tools}  max_iter={iters}")
        else:
            print(f"  {label}")

    print()
    for edge in graph.edges:
        src = next((n.name for n in graph.nodes if n.id == edge.source_id), "?")
        tgt = next((n.name for n in graph.nodes if n.id == edge.target_id), "?")
        lbl = f" [{edge.label}]" if edge.label else ""
        print(f"  {src} → {tgt}{lbl}")


# ── What happens at runtime (conceptual) ────────────────────────────
#
# The engine gives the Orchestrator node these tool instances:
#
#   SpawnAgentTool(manager=session_mgr, allowed_agents=["researcher", "writer", "reviewer"])
#   CollectResultsTool(manager=session_mgr)
#   GetSessionStatusTool(manager=session_mgr)
#   CancelSessionTool(manager=session_mgr)
#
# The Orchestrator's agentic loop (up to 15 iterations) lets the LLM
# decide how many agents to spawn and when to collect.  A single LLM
# turn can include multiple tool_calls, so all three spawn_agent calls
# happen in parallel:
#
#   Turn 1  →  tool_calls: [
#       spawn_agent(agent_id="researcher", task="Research ..."),
#       spawn_agent(agent_id="writer",     task="Draft ..."),
#       spawn_agent(agent_id="reviewer",   task="Prepare review ..."),
#   ]
#   Turn 2  →  tool_calls: [
#       collect_agent_results(session_ids="sid-1,sid-2,sid-3", timeout=120)
#   ]
#   Turn 3  →  final text summary (loop exits)
#
# Each spawned session runs its own Agent graph in a background thread.
# The SessionManager tracks them all and wait_all() blocks until done.
