"""Courtroom debate -- two AI agents argue a case with a loop-back for multiple rounds.

Two attorneys (prosecution and defense) prepare arguments in parallel,
then enter a debate loop where each round consists of prosecution
argument, defense rebuttal, and a judge evaluation. The judge decides
after each round whether to continue or deliver a final verdict.

This example demonstrates:
  - Parallel sub-graphs (both sides prepare simultaneously)
  - Loop node with max_iterations for multi-round debate
  - IF node inside loop to check whether to continue
  - Manual edge wiring for the loop back-edge
  - Memory (accumulates arguments across rounds)
  - Decision routing (judge picks guilty/not_guilty/mistrial)
  - Text templates for dramatic courtroom narration
  - Reasoning node for judicial deliberation

Architecture::

    START -> User -> VAR -> WRITE_MEMORY -> Text("Court in session")
      |
    PARALLEL(preparation)
      |── Prosecution: "Build case"
      |── Defense: "Build defense"
      |
    STATIC_MERGE -> VAR(round=1)
      |
    LOOP(max=3) <─────────────────────────────────────────────+
      |                                                       |
    Text("Round {{round_number}}")                            |
    Instruction("Prosecution argues")                         |
    Instruction("Defense rebuts")                             |
    Instruction("Judge evaluates round")                      |
    UPDATE_MEMORY(transcript) -> VAR(round+1)                 |
      |                                                       |
    IF(round > 3)                                             |
      |── true:  Text("Enough") ─→ verdict                   |
      |── false: Text("Continue") ────────────────────────────+
      |
    Reasoning("Deliberate") -> Instruction("Final verdict")
      |
    DECISION(verdict)
      |── guilty:     Text(GUILTY)
      |── not_guilty: Text(NOT GUILTY)
      |── mistrial:   Text(MISTRIAL)
      |
    WRITE_MEMORY(verdict) -> Text("Adjourned") -> END
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
    from quartermaster_graph.enums import NodeType, TraverseOut
    from quartermaster_graph.models import GraphEdge
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")


# ---------------------------------------------------------------------------
# Sub-graph: Prosecution preparation
# ---------------------------------------------------------------------------

prosecution_prep = (
    Graph("Prosecution Prep")
    .start()
    .instruction(
        "Build prosecution case",
        system_instruction=(
            "You are the lead prosecutor. Prepare your legal strategy:\n"
            "1. Key evidence (access logs, code similarity, timeline)\n"
            "2. Witnesses to call\n"
            "3. Legal precedents supporting the prosecution\n"
            "4. Defense weaknesses to exploit"
        ),
    )
    .end()
)

# ---------------------------------------------------------------------------
# Sub-graph: Defense preparation
# ---------------------------------------------------------------------------

defense_prep = (
    Graph("Defense Prep")
    .start()
    .instruction(
        "Build defense case",
        system_instruction=(
            "You are the lead defense attorney. Prepare your strategy:\n"
            "1. Evidence supporting the defendant's innocence\n"
            "2. Holes in the prosecution's narrative\n"
            "3. Legal precedents favoring the defense\n"
            "4. Arguments for why the non-compete is unenforceable"
        ),
    )
    .end()
)

# ---------------------------------------------------------------------------
# Main courtroom graph -- built with chained fluent API, then manually
# wire the loop back-edge at the end.
# ---------------------------------------------------------------------------

trial = (
    Graph("The People v. AI Engineer")
    .start()

    # --- Case setup ----------------------------------------------------------
    .user("Describe the case to be tried")
    .var("Capture case", variable="case_description")
    .write_memory("File case", memory_name="case_file")
    .text("Court opens", template=(
        "=== SUPERIOR COURT - DEPARTMENT 7 ===\n"
        "Case: The People v. AI Engineer\n"
        "Charge: Theft of trade secrets, breach of contract\n"
        "\n"
        "{{case_description}}\n"
        "\n"
        "BAILIFF: All rise. Court is now in session."
    ))

    # --- Parallel: both sides prepare simultaneously -------------------------
    .parallel("Case preparation")
    .branch()
        .use(prosecution_prep)
    .end()
    .branch()
        .use(defense_prep)
    .end()
    .static_merge("Both sides ready")

    # --- Initialize round counter --------------------------------------------
    .var("Init round", variable="round_number", expression="1")

    # --- LOOP: multi-round debate (up to 3 rounds) --------------------------
    .instruction("Debate loop", system_instruction="Manage the debate loop (up to 3 rounds)")

    .text("Round header", template=(
        "\n--- ROUND {{round_number}} ---\n"
        "JUDGE: Prosecution, you may proceed."
    ))

    .instruction(
        "Prosecution argues",
        system_instruction=(
            "You are the prosecutor. This is round {{round_number}} of the trial.\n"
            "Round 1: Opening statement and initial evidence.\n"
            "Round 2: Present your strongest evidence and witness testimony.\n"
            "Round 3: Closing argument -- make it powerful and memorable.\n"
            "Adapt your strategy to the round. 2-3 paragraphs."
        ),
    )

    .instruction(
        "Defense rebuts",
        system_instruction=(
            "You are the defense attorney. This is round {{round_number}}.\n"
            "You just heard the prosecution. Counter their argument directly:\n"
            "Round 1: Challenge their narrative, present your opening.\n"
            "Round 2: Cross-examine evidence, expose weaknesses.\n"
            "Round 3: Closing argument -- fight for acquittal.\n"
            "Be sharp, find contradictions. 2-3 paragraphs."
        ),
    )

    .instruction(
        "Judge evaluates round",
        system_instruction=(
            "You are the presiding judge. You just heard round {{round_number}}.\n"
            "Assess briefly:\n"
            "1. Which side was more compelling this round?\n"
            "2. What key points were made?\n"
            "3. Are there unresolved questions?\n"
            "End with: 'Sufficient evidence: yes' or 'Sufficient evidence: no'"
        ),
    )

    .update_memory("Log round", memory_name="trial_transcript")
    .var("Next round", variable="round_number", expression="round_number + 1")

    # --- Check: continue or break out? ---------------------------------------
    .if_node("Enough evidence?", expression="round_number > 3")

    .on("true")
        .text("Debate complete", template=(
            "\nJUDGE: The court has heard sufficient argument across all rounds.\n"
            "We will now proceed to deliberation."
        ))
    .end()

    .on("false")
        .text("Continue debate", template=(
            "\nJUDGE: The court requires further argument.\n"
            "We will proceed to round {{round_number}}."
        ))
    .end()
    # IF picks one branch -- no merge needed.
    # The "true" branch continues linearly to the verdict.
    # The "false" branch loops back (wired below).

    # --- Final Verdict -------------------------------------------------------
    .text("Deliberation header", template="\n=== JUDICIAL DELIBERATION ===\n")

    .reasoning("Judge deliberates on all rounds")

    .instruction(
        "Final verdict",
        system_instruction=(
            "You are the presiding judge. You have heard the full trial across "
            "multiple rounds. Review the complete transcript.\n\n"
            "Deliver your verdict:\n"
            "1. Summarize the strongest points from each side\n"
            "2. Identify which evidence was most compelling\n"
            "3. Apply the legal standard for theft of trade secrets\n"
            "4. Consider whether the non-compete is enforceable\n\n"
            "End with exactly one word on its own line: guilty, not_guilty, or mistrial."
        ),
    )

    .decision("Verdict?", options=["guilty", "not_guilty", "mistrial"])

    .on("guilty")
        .text("Guilty verdict", template=(
            "\n=== VERDICT: GUILTY ===\n"
            "JUDGE: The defendant is found GUILTY of theft of trade secrets.\n"
            "Sentencing will be scheduled.\n"
            "BAILIFF: Order in the court!"
        ))
    .end()

    .on("not_guilty")
        .text("Not guilty verdict", template=(
            "\n=== VERDICT: NOT GUILTY ===\n"
            "JUDGE: The defendant is found NOT GUILTY.\n"
            "The charges are dismissed. The defendant is free to go.\n"
            "DEFENSE: Justice has been served."
        ))
    .end()

    .on("mistrial")
        .text("Mistrial declared", template=(
            "\n=== MISTRIAL DECLARED ===\n"
            "JUDGE: The court declares a MISTRIAL.\n"
            "The case may be retried."
        ))
    .end()

    # Decision picks ONE verdict -- no merge needed.
    .write_memory("Record verdict", memory_name="verdict")
    .text("Court adjourned", template=(
        "\n--- Court is adjourned ---\n"
        "Case concluded after multiple rounds of argument."
    ))
    .end()
)

# ---------------------------------------------------------------------------
# Wire the loop back-edge: "Continue debate" -> "Debate loop"
# ---------------------------------------------------------------------------
# Find the nodes by name so we can create the back-edge
node_by_name = {n.name: n for n in trial.nodes}
continue_node = node_by_name["Continue debate"]
loop_node = node_by_name["Debate loop"]

trial._edges.append(
    GraphEdge(source_id=continue_node.id, target_id=loop_node.id, label="next_round")
)

# Build the final AgentVersion (skip validation -- intentional cycle)
agent = trial.build(validate=False)


# ---------------------------------------------------------------------------
# Print graph structure
# ---------------------------------------------------------------------------

print("=" * 60)
print("THE PEOPLE v. AI ENGINEER")
print("A Multi-Round Courtroom Drama with Loop-Back")
print("=" * 60)
print(f"\n  Nodes: {len(agent.nodes)}")
print(f"  Edges: {len(agent.edges)}")

# Count by type
node_types: dict[str, int] = {}
for n in agent.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
print(f"\n  Node types:")
for t, count in sorted(node_types.items()):
    print(f"    {t:20s} x{count}")

# Show the flow
print(f"\n  Edge list ({len(agent.edges)} edges):")
name_map = {n.id: n.name for n in agent.nodes}
for edge in agent.edges:
    label = f"  [{edge.label}]" if edge.label else ""
    src = name_map.get(edge.source_id, "?")
    tgt = name_map.get(edge.target_id, "?")
    is_loop = "  <-- LOOP BACK" if "Debate loop" in tgt and "Init" not in src else ""
    print(f"    {src} -> {tgt}{label}{is_loop}")

# Debate structure
print(f"\n  Debate loop (up to 3 rounds):")
print(f"    Each round:")
print(f"      1. Prosecution argues (adapts strategy per round)")
print(f"      2. Defense rebuts (counters prosecution directly)")
print(f"      3. Judge evaluates (decides if more rounds needed)")
print(f"      4. IF round > 3: break -> final verdict")
print(f"      5. IF round <= 3: loop back to Debate loop")

# Roles
print(f"\n  Agents in the courtroom:")
for n in agent.nodes:
    name_lower = n.name.lower()
    if "judge" in name_lower or "deliberat" in name_lower or "verdict" in name_lower:
        role = "JUDGE"
    elif "defense" in name_lower:
        role = "DEFENSE"
    elif "prosecution" in name_lower:
        role = "PROSECUTION"
    else:
        continue
    if n.type.value in ("Instruction1", "Reasoning1"):
        print(f"    [{role:11s}] {n.name}")
