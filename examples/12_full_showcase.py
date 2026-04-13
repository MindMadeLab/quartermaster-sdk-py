"""Full showcase -- AI Research Assistant using every major pattern.

A "kitchen sink" example that demonstrates every major graph-builder
pattern in one realistic workflow: an AI Research Assistant that takes
a user's question, researches it from multiple angles, reviews quality,
synthesises findings, and delivers a polished report.

Patterns demonstrated
---------------------
  - User input node
  - Decision routing (research strategy)
  - Sub-graphs (web research, academic research)
  - Parallel fan-out with nested IF
  - Memory read/write/update
  - Text templating with {{variables}}
  - Var capture
  - Reasoning and summarisation nodes
  - Logging and notifications
  - Multiple merge points

Architecture::

    START
      |
    User("What do you want to research?")
      |
    VAR(topic) -> WRITE_MEMORY(topic)
      |
    Instruction("Classify research type")
      |
    DECISION(strategy) ---+--- academic ---+--- general ---+--- technical ---+
      |                   |                |               |                 |
      |              [Academic sub]   [Web sub]       [Technical]            |
      |                   |                |               |                 |
      +-------------------+----------------+---------------+-----------------+
                                    |
                                 MERGE-1
                                    |
                            READ_MEMORY(topic)
                                    |
                           PARALLEL(review) ----+-------------------+
                              |                 |                   |
                         [Fact-check]    [Bias check]        [Completeness]
                         IF(verified?)          |            IF(gaps?)
                          T / F                 |             T / F
                              |                 |               |
                              +--------+--------+-------+-------+
                                       |
                                    MERGE-2
                                       |
                               Reasoning("Synthesise")
                                       |
                              Summarize("Executive summary")
                                       |
                         UPDATE_MEMORY(research_status)
                                       |
                    Notification("Report ready") -> Log("done")
                                       |
                                      END
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
    from quartermaster_graph.enums import TraverseOut
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# ---------------------------------------------------------------------------
# Sub-graph: Web research pipeline
# ---------------------------------------------------------------------------

web_research = (
    Graph("Web Research")
    .start()
    .instruction("Web search", system_instruction="Search the web for recent information on the topic")
    .instruction("Extract key facts", system_instruction="Extract and list the key facts from search results")
    .instruction("Assess source quality", system_instruction="Rate the reliability of each source (1-5)")
    .end()
)

# ---------------------------------------------------------------------------
# Sub-graph: Academic research pipeline
# ---------------------------------------------------------------------------

academic_research = (
    Graph("Academic Research")
    .start()
    .instruction("Search papers", system_instruction="Search academic databases for peer-reviewed papers")
    .instruction("Summarise papers", system_instruction="Create structured summaries of the top papers")
    .instruction("Identify consensus", system_instruction="Identify areas of scientific consensus and debate")
    .end()
)

# ---------------------------------------------------------------------------
# Main graph: AI Research Assistant
# ---------------------------------------------------------------------------

agent = (
    Graph("AI Research Assistant")
    .start()

    # --- Input and topic capture ----------------------------------------------
    .user("What do you want to research?")
    .var("Capture topic", variable="research_topic")
    .write_memory("Store topic", memory_name="research_topic")
    .text("Acknowledge", template="Researching: {{research_topic}}")

    # --- Strategy selection ----------------------------------------------------
    .instruction("Classify research", system_instruction="Classify the research type: academic, general, or technical")

    .decision("Research strategy?", options=["academic", "general", "technical"])

    .on("academic")
        .use(academic_research)
    .end()

    .on("general")
        .use(web_research)
    .end()

    .on("technical")
        .instruction("Technical deep-dive", system_instruction="Perform in-depth technical analysis with code examples")
        .instruction("Benchmark review", system_instruction="Review benchmarks and performance comparisons")
    .end()

    # No merge — decision picks ONE research strategy.

    # --- Quality review: parallel checks with nested IF -----------------------
    .read_memory("Recall topic", memory_name="research_topic")

    .parallel("Quality review")

    # Branch 1: Fact-checking with pass/fail gate
    .branch()
        .instruction("Fact-check", system_instruction="Verify all factual claims against sources")
        .if_node("Facts verified?", expression="verification_score > 0.9")
        .on("true")
            .text("Facts OK", template="All facts verified successfully")
        .end()
        .on("false")
            .instruction("Fix errors", system_instruction="Correct any unverified or inaccurate claims")
        .end()
        # IF branches converge on a result node
        .static("Fact-check done", text="Fact-check complete")
    .end()

    # Branch 2: Bias assessment (no conditional, straight-through)
    .branch()
        .instruction("Bias assessment", system_instruction="Check for confirmation bias, source bias, and framing issues")
    .end()

    # Branch 3: Completeness check with gap detection
    .branch()
        .if_node("Coverage gaps?", expression="has_coverage_gaps")
        .on("true")
            .instruction("Fill gaps", system_instruction="Research and fill identified coverage gaps")
        .end()
        .on("false")
            .text("Coverage complete", template="Research covers all key aspects of {{research_topic}}")
        .end()
        # IF branches converge on a result node
        .static("Coverage done", text="Coverage check complete")
    .end()

    .merge("Quality review complete")

    # --- Synthesis and delivery -----------------------------------------------
    .reasoning("Synthesise findings")
    .summarize("Executive summary", system_instruction="Create a concise executive summary with key takeaways")

    # --- Audit trail ----------------------------------------------------------
    .update_memory("Update status", memory_name="research_status")
    .notification("Report ready", channel="email", message="Research report on {{research_topic}} is ready for review")
    .log("Completed", message="Research pipeline completed for: {{research_topic}}", level="info")
    .end()
)

# ---------------------------------------------------------------------------
# Print comprehensive graph stats
# ---------------------------------------------------------------------------

print("=" * 60)
print("AI Research Assistant -- Full Showcase")
print("=" * 60)
print(f"\n  Total nodes: {len(agent.nodes)}")
print(f"  Total edges: {len(agent.edges)}")

# Node type breakdown
node_types: dict[str, int] = {}
for n in agent.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
print(f"\n  Node type breakdown:")
for t, count in sorted(node_types.items()):
    print(f"    {t:20s} x{count}")

# Branching nodes (nodes with multiple outgoing edges)
outgoing: dict[str, int] = {}
for e in agent.edges:
    outgoing[str(e.source_id)] = outgoing.get(str(e.source_id), 0) + 1
print(f"\n  Branching nodes:")
for n in agent.nodes:
    count = outgoing.get(str(n.id), 0)
    if count > 1:
        kind = "PICK-ONE" if n.traverse_out == TraverseOut.SPAWN_PICKED else "FAN-OUT"
        print(f"    {kind:8s}  {n.name} ({count} outgoing edges)")

# Memory operations
print(f"\n  Memory operations:")
for n in agent.nodes:
    ntype = n.type.value.upper()
    if "MEMORY" in ntype or ntype == "VAR":
        key = n.metadata.get("memory_name", n.metadata.get("variable", ""))
        print(f"    {n.type.value:15s}  {n.name:30s}  key={key}")

# Full edge list
print(f"\n  Edge list ({len(agent.edges)} edges):")
name_map = {n.id: n.name for n in agent.nodes}
for edge in agent.edges:
    label = f"  [{edge.label}]" if edge.label else ""
    print(f"    {name_map[edge.source_id]} -> {name_map[edge.target_id]}{label}")
