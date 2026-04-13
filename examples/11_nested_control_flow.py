"""Nested control flow inside parallel branches -- the whiteboard pattern.

Demonstrates the most advanced graph pattern: parallel fan-out where
individual branches contain their own IF decisions and nested logic,
then everything merges back together.

This is the "whiteboard" pattern -- the kind of graph you'd sketch on
a whiteboard when designing a complex agent pipeline.

Architecture::

    START
      |
    User("Describe your task")
      |
    Instruction("Analyse input")
      |
    PARALLEL --------+--------------------------+------------------+
      |              |                          |                  |
    [Branch A]     [Branch B]                [Branch C]           |
    Text(context)  Instruction(independent)  IF(confidence>0.7)   |
    Instruction      |                        |          |        |
    (deep analysis)  |                      true       false      |
      |              |                        |          |        |
      |              |                      Text       Text       |
      |              |                     (accept)  (needs work) |
      |              |                        |          |        |
      |              |                        +----+-----+        |
      |              |                             |              |
      |              |                       IF-merge             |
      +--------------+-----------------------------+--------------+
                              |
                           MERGE-1
                              |
                     Summarize("Combine all")
                              |
                            END

Key patterns shown:
  - parallel() with 3+ branches of varying complexity
  - IF node nested inside a parallel branch
  - Empty/pass-through branch (Branch B goes straight through)
  - merge() collecting all branch endpoints
  - Summarize node for final combination
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
    from quartermaster_graph.enums import TraverseOut
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Whiteboard Agent")
    .start()
    .user("Describe your task")
    .instruction("Analyse input", system_instruction="Break the task into components for parallel processing")

    # --- Parallel fan-out: 3 branches with nested control flow ----------------
    .parallel("Fan out")

    # Branch A: deep analysis pipeline
    .branch()
        .text("Prepare context", template="Task context: {{user_input}}")
        .instruction("Deep analysis", system_instruction="Perform thorough analysis of the task")
    .end()

    # Branch B: independent lightweight check (pass-through)
    .branch()
        .instruction("Quick check", system_instruction="Perform a fast independent assessment")
    .end()

    # Branch C: conditional quality gate (IF picks one path — no merge needed)
    .branch()
        .if_node("Confidence high?", expression="confidence > 0.7")
        .on("true")
            .text("Accept result", template="Result meets confidence threshold -- approved")
        .end()
        .on("false")
            .text("Flag for review", template="Low confidence -- manual review recommended")
        .end()
        # IF branches converge on this static node, which becomes the branch endpoint
        .static("Quality gate result", text="Quality gate complete")
    .end()

    .merge("Combine all branches")

    # --- Final synthesis -------------------------------------------------------
    .summarize("Final synthesis", system_instruction="Combine all branch results into a coherent response")
    .end()
)

# ---------------------------------------------------------------------------
# Print graph structure
# ---------------------------------------------------------------------------

print(f"Whiteboard Agent: {len(agent.nodes)} nodes, {len(agent.edges)} edges")

print("\nNode list:")
outgoing: dict[str, int] = {}
for e in agent.edges:
    outgoing[str(e.source_id)] = outgoing.get(str(e.source_id), 0) + 1
for node in agent.nodes:
    count = outgoing.get(str(node.id), 0)
    extras = []
    if count > 1 and node.traverse_out == TraverseOut.SPAWN_PICKED:
        extras.append("pick-one")
    elif count > 1:
        extras.append(f"fan-out x{count}")
    suffix = f"  [{', '.join(extras)}]" if extras else ""
    print(f"  [{node.type.value:15s}] {node.name}{suffix}")

print("\nEdge list:")
name_map = {n.id: n.name for n in agent.nodes}
for edge in agent.edges:
    label = f"  [{edge.label}]" if edge.label else ""
    print(f"  {name_map[edge.source_id]} -> {name_map[edge.target_id]}{label}")
