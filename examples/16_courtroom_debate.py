"""Courtroom debate -- two AI agents argue a case before a judge.

Two attorneys (prosecution and defense) prepare arguments in parallel,
then take turns presenting their case across three rounds. A judge
agent evaluates both sides and delivers a verdict.

This example demonstrates:
  - Parallel sub-graphs (both sides prepare simultaneously)
  - Sequential multi-round debate (argument -> counter -> rebuttal)
  - Memory (case file shared across all nodes)
  - Decision routing (judge picks guilty/not_guilty/mistrial)
  - Text templates for dramatic courtroom narration
  - Reasoning node for judicial deliberation

Architecture::

    START
      |
    User("Describe the case")
      |
    VAR(case) -> WRITE_MEMORY(case_file)
      |
    Text("Court is now in session...")
      |
    PARALLEL ─────────────────────────────────+
      |                                       |
    [Prosecution]                         [Defense]
    Instruction("Build case")            Instruction("Build defense")
      |                                       |
    +─────────────────────────────────────────+
      |
    STATIC_MERGE("Both sides ready")
      |
    ── Round 1 ──
    Instruction("Prosecution opening statement")
    Instruction("Defense opening statement")
      |
    ── Round 2 ──
    Instruction("Prosecution presents evidence")
    Instruction("Defense cross-examines")
      |
    ── Round 3 ──
    Instruction("Prosecution closing argument")
    Instruction("Defense closing argument")
      |
    ── Verdict ──
    Reasoning("Judge deliberates")
    DECISION(verdict) ── guilty / not_guilty / mistrial
      |           |              |
    Text       Text           Text
    (guilty)   (acquitted)    (mistrial)
      |           |              |
    WRITE_MEMORY(verdict)
      |
    Text("Court is adjourned")
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
# The case
# ---------------------------------------------------------------------------

CASE_CONTEXT = (
    "The defendant, a senior AI engineer at a major tech company, is accused "
    "of secretly training a personal AI model using proprietary company data "
    "and computational resources, then launching a competing startup. The "
    "prosecution alleges theft of trade secrets and breach of contract. The "
    "defense argues the engineer used only publicly available data and open-"
    "source tools, and that the non-compete clause is unenforceable."
)

# ---------------------------------------------------------------------------
# Sub-graph: Prosecution preparation
# ---------------------------------------------------------------------------

prosecution_prep = (
    Graph("Prosecution Prep")
    .start()
    .instruction(
        "Build prosecution case",
        system_instruction=(
            "You are the lead prosecutor. Based on the case file, prepare your "
            "legal strategy. Identify:\n"
            "1. Key evidence points (access logs, code similarity, timeline)\n"
            "2. Witnesses to call\n"
            "3. Legal precedents that support the prosecution\n"
            "4. Weaknesses in the defense you plan to exploit\n"
            "Be aggressive but factual."
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
            "You are the lead defense attorney. Based on the case file, prepare "
            "your defense strategy. Identify:\n"
            "1. Evidence that supports the defendant's innocence\n"
            "2. Holes in the prosecution's narrative\n"
            "3. Legal precedents favoring the defense\n"
            "4. Arguments for why the non-compete is unenforceable\n"
            "Be persuasive and find every angle to protect your client."
        ),
    )
    .end()
)

# ---------------------------------------------------------------------------
# Main courtroom graph
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
        "\n"
        "Case: The People v. AI Engineer\n"
        "Charge: Theft of trade secrets, breach of contract\n"
        "\n"
        "{{case_description}}\n"
        "\n"
        "BAILIFF: All rise. Court is now in session.\n"
        "JUDGE: Be seated. Both counsels, are you ready to proceed?"
    ))

    # --- Parallel preparation: both sides prepare simultaneously -------------
    .parallel("Case preparation")

    .branch()
        .use(prosecution_prep)
    .end()

    .branch()
        .use(defense_prep)
    .end()

    .static_merge("Both sides ready")

    # --- Round 1: Opening statements -----------------------------------------
    .text("Round 1 header", template="\n--- ROUND 1: OPENING STATEMENTS ---\n")

    .instruction(
        "Prosecution opening",
        system_instruction=(
            "You are the prosecutor. Deliver your opening statement to the jury. "
            "Paint a clear picture of the crime: the defendant betrayed their "
            "employer's trust, stole proprietary data, and used company resources "
            "to build a competing product. Be dramatic but credible. Address the "
            "jury directly. Keep it to 2-3 paragraphs."
        ),
    )

    .instruction(
        "Defense opening",
        system_instruction=(
            "You are the defense attorney. Deliver your opening statement. The "
            "prosecution just made their case -- now counter it. Argue that your "
            "client is an innovator, not a thief. The data was public, the tools "
            "were open-source, and the non-compete is a relic of corporate "
            "overreach. Make the jury sympathize with the underdog. 2-3 paragraphs."
        ),
    )

    # --- Round 2: Evidence and cross-examination ------------------------------
    .text("Round 2 header", template="\n--- ROUND 2: EVIDENCE & CROSS-EXAMINATION ---\n")

    .instruction(
        "Prosecution evidence",
        system_instruction=(
            "You are the prosecutor. Present your key evidence:\n"
            "- Server access logs showing late-night data transfers\n"
            "- Code similarity analysis between company IP and the startup\n"
            "- Testimony from a colleague who saw the defendant copying files\n"
            "- The signed employment contract with non-compete clause\n"
            "Walk through each piece methodically. This is your strongest moment."
        ),
    )

    .instruction(
        "Defense cross-examination",
        system_instruction=(
            "You are the defense attorney. Cross-examine and dismantle the "
            "prosecution's evidence:\n"
            "- The access logs show normal work patterns, not theft\n"
            "- Code similarity is expected when solving the same problem\n"
            "- The colleague has a grudge (was passed over for promotion)\n"
            "- The non-compete clause is overly broad and likely unenforceable "
            "under state law\n"
            "Be sharp. Find contradictions. Plant reasonable doubt."
        ),
    )

    # --- Round 3: Closing arguments ------------------------------------------
    .text("Round 3 header", template="\n--- ROUND 3: CLOSING ARGUMENTS ---\n")

    .instruction(
        "Prosecution closing",
        system_instruction=(
            "You are the prosecutor. Deliver a powerful closing argument. "
            "Summarize the evidence, remind the jury of the defendant's actions, "
            "and make the moral case: if we allow employees to steal and compete, "
            "no company will invest in innovation. Ask for a guilty verdict. "
            "Make it memorable. 2-3 paragraphs."
        ),
    )

    .instruction(
        "Defense closing",
        system_instruction=(
            "You are the defense attorney. This is your final chance. Deliver "
            "a passionate closing. Remind the jury that the prosecution has not "
            "proven their case beyond reasonable doubt. Your client built "
            "something new with public knowledge. Non-competes stifle "
            "innovation and hurt workers. Ask for acquittal. Make the jury "
            "feel they're standing up for the little guy. 2-3 paragraphs."
        ),
    )

    # --- Verdict: Judge deliberates -------------------------------------------
    .text("Deliberation header", template="\n--- JUDICIAL DELIBERATION ---\n")

    .reasoning("Judge deliberates")

    .instruction(
        "Judge's analysis",
        system_instruction=(
            "You are the presiding judge. You have heard both sides across three "
            "rounds. Now deliver your analysis:\n"
            "1. Summarize the strongest points from each side\n"
            "2. Identify which evidence was most compelling\n"
            "3. Note where each side was weakest\n"
            "4. Apply the legal standard: was theft of trade secrets proven?\n"
            "5. Render your verdict: guilty, not_guilty, or mistrial\n"
            "\n"
            "End your analysis with exactly one word on its own line: "
            "guilty, not_guilty, or mistrial."
        ),
    )

    .decision("Verdict?", options=["guilty", "not_guilty", "mistrial"])

    .on("guilty")
        .text("Guilty verdict", template=(
            "\n=== VERDICT: GUILTY ===\n"
            "\n"
            "JUDGE: The defendant is found GUILTY of theft of trade secrets.\n"
            "Sentencing will be scheduled for a later date.\n"
            "The defendant is remanded into custody.\n"
            "\n"
            "BAILIFF: Order in the court!"
        ))
    .end()

    .on("not_guilty")
        .text("Not guilty verdict", template=(
            "\n=== VERDICT: NOT GUILTY ===\n"
            "\n"
            "JUDGE: The defendant is found NOT GUILTY.\n"
            "The charges are dismissed. The defendant is free to go.\n"
            "\n"
            "DEFENSE ATTORNEY: Justice has been served.\n"
            "PROSECUTOR: The People are disappointed but respect the court's decision."
        ))
    .end()

    .on("mistrial")
        .text("Mistrial declared", template=(
            "\n=== MISTRIAL DECLARED ===\n"
            "\n"
            "JUDGE: Due to insufficient evidence and procedural concerns,\n"
            "the court declares a MISTRIAL. The case may be retried.\n"
            "\n"
            "Both counsels are dismissed."
        ))
    .end()

    # Decision picks ONE verdict -- no merge needed.

    .write_memory("Record verdict", memory_name="verdict")
    .text("Court adjourned", template=(
        "\n--- Court is adjourned ---\n"
        "Case file archived. Thank you, counselors."
    ))
    .end()
)


# ---------------------------------------------------------------------------
# Print graph structure
# ---------------------------------------------------------------------------

print("=" * 60)
print("THE PEOPLE v. AI ENGINEER")
print("A Courtroom Drama in Three Rounds")
print("=" * 60)
print(f"\n  Nodes: {len(trial.nodes)}")
print(f"  Edges: {len(trial.edges)}")

# Count by type
node_types: dict[str, int] = {}
for n in trial.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
print(f"\n  Node types:")
for t, count in sorted(node_types.items()):
    print(f"    {t:20s} x{count}")

# Show the dramatic flow
print(f"\n  Flow:")
name_map = {n.id: n.name for n in trial.nodes}
for edge in trial.edges:
    label = f"  [{edge.label}]" if edge.label else ""
    src = name_map.get(edge.source_id, "?")
    tgt = name_map.get(edge.target_id, "?")
    print(f"    {src} -> {tgt}{label}")

# Key nodes
print(f"\n  Debate rounds:")
instructions = [n for n in trial.nodes if n.type.value == "Instruction1"]
for n in instructions:
    name_lower = n.name.lower()
    if "judge" in name_lower:
        role = "JUDGE"
    elif "defense" in name_lower:
        role = "DEFENSE"
    elif "prosecution" in name_lower or "prosecutor" in name_lower:
        role = "PROSECUTION"
    else:
        role = ""
    if role:
        print(f"    [{role:11s}] {n.name}")
