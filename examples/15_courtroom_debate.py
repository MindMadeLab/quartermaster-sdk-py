"""Courtroom debate -- multi-round prosecution vs defense with loop-back.

Demonstrates loops, IF branching, memory, show_output, connect(), and
mixed providers in a courtroom trial with 5 rounds of escalating argument.

Each round covers different ground:
  Round 1: Opening statements
  Round 2: Evidence presentation and witness testimony
  Round 3: Cross-examination and challenges
  Round 4: Rebuttal and counter-evidence
  Round 5: Closing arguments

Providers:
  - Prosecution: xAI Grok (aggressive style)
  - Defense: OpenAI GPT-4o (methodical style)
  - Judge: Anthropic Claude (balanced deliberation)

Usage:
    uv run examples/16_courtroom_debate.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_graph.enums import TraverseIn
from quartermaster_engine import run_graph


# -- Provider config --------------------------------------------------------

PROSECUTOR = dict(model="grok-3-mini-fast", provider="xai")
DEFENSE    = dict(model="gpt-4o", provider="openai")
JUDGE      = dict(model="claude-haiku-4-5-20251001", provider="anthropic")

MAX_ROUNDS = 5


# -- Main trial graph -------------------------------------------------------

trial = (
    Graph("The People v. Dr. Sarah Chen")
    .start()
    .user("Describe the case")
    .var("Capture case", variable="case_description", show_output=False)
    .write_memory("File case", memory_name="case_file", show_output=False)

    .text("Court opens", template=(
        "═══════════════════════════════════════════\n"
        "  SUPERIOR COURT — DEPARTMENT 7\n"
        "  TechCorp Inc. v. Dr. Sarah Chen\n"
        "  Charges: Trade secret theft, NDA breach\n"
        "═══════════════════════════════════════════\n\n"
        "BAILIFF: All rise. Court is now in session.\n"
        f"JUDGE: We will hear {MAX_ROUNDS} rounds of argument.\n"
    ))

    # -- Initialize loop counter --
    .var("Init round", variable="round_number", expression="1", show_output=False)

    # -- Loop target --
    .text("Round announce", template=(
        "\n──── Round {{round_number}} of 5 ────"
    ), traverse_in=TraverseIn.AWAIT_FIRST)

    # -- Round context injected as a text node so LLMs see it --
    .text("Round context", template=(
        "COURT CLERK: This is round {{round_number}} of 5.\n"
        "{% if round_number == 1 %}"
        "Phase: OPENING STATEMENTS. Present your theory of the case."
        "{% elif round_number == 2 %}"
        "Phase: EVIDENCE. Present forensic evidence, call expert witnesses, cite specific data."
        "{% elif round_number == 3 %}"
        "Phase: CROSS-EXAMINATION. Challenge the opposing side's evidence and witnesses directly."
        "{% elif round_number == 4 %}"
        "Phase: REBUTTAL. Address weaknesses in your case, introduce new counter-evidence."
        "{% else %}"
        "Phase: CLOSING ARGUMENTS. This is your final chance. Make it count."
        "{% endif %}"
    ), show_output=False)

    # -- Prosecution --
    .instruction(
        "Prosecution argues", **PROSECUTOR,
        system_instruction=(
            "You are the lead prosecutor in TechCorp v. Dr. Sarah Chen.\n"
            "The court clerk has announced the current phase. Follow it STRICTLY:\n\n"
            "OPENING STATEMENTS: Lay out your theory — charges, motive, what you'll prove.\n"
            "EVIDENCE: Present specific forensic findings — access logs showing bulk downloads "
            "48 hours before resignation, expert testimony on code architecture matching, "
            "the 78% similarity breakdown showing proprietary vs open-source components.\n"
            "CROSS-EXAMINATION: Attack the defense's claims directly — name specific flaws "
            "in their open-source argument, challenge their expert's credentials.\n"
            "REBUTTAL: Address your case's weaknesses head-on, then present the timeline "
            "evidence and internal communications that corroborate premeditation.\n"
            "CLOSING: Powerful emotional and legal summary. Connect all evidence. Ask for guilty.\n\n"
            "IMPORTANT: Do NOT repeat arguments from previous rounds. Build on them.\n"
            "2-3 paragraphs. Address the judge as 'Your Honor'."
        ),
    )

    # -- Defense --
    .instruction(
        "Defense rebuts", **DEFENSE,
        system_instruction=(
            "You are the defense attorney for Dr. Sarah Chen.\n"
            "The court clerk has announced the current phase. Follow it STRICTLY:\n\n"
            "OPENING STATEMENTS: Present your theory — Dr. Chen is innocent, the non-compete "
            "is unenforceable, her work is original.\n"
            "EVIDENCE: Present your counter-evidence — independent code audit showing open-source "
            "origins, Dr. Chen's published research predating TechCorp work, industry expert "
            "testimony on common AI patterns.\n"
            "CROSS-EXAMINATION: Directly challenge prosecution's forensic expert — question "
            "methodology, show the 78% figure includes standard library code, present "
            "alternative similarity analysis.\n"
            "REBUTTAL: Address prosecution's strongest points, present character witnesses, "
            "show Dr. Chen's ethical track record.\n"
            "CLOSING: Passionate plea — invoke reasonable doubt, the right to work, "
            "the danger of criminalizing common knowledge.\n\n"
            "IMPORTANT: Do NOT repeat arguments from previous rounds. Build on them.\n"
            "Directly reference and counter what the prosecution just said.\n"
            "2-3 paragraphs. Address the judge as 'Your Honor'."
        ),
    )

    # -- Increment and check --
    .var("Increment", variable="round_number", expression="round_number + 1", show_output=False)
    .if_node("More rounds?", expression=f"round_number > {MAX_ROUNDS}", show_output=False)

    # TRUE: verdict
    .on("true")
        .text("Debate over", template=(
            "\n═══════════════════════════════════════════\n"
            "  ALL ARGUMENTS HEARD — DELIVERING VERDICT\n"
            "═══════════════════════════════════════════"
        ))
        .instruction(
            "Judge delivers verdict", **JUDGE,
            system_instruction=(
                "You are the presiding judge delivering the FINAL VERDICT.\n"
                "You heard 5 rounds: openings, evidence, cross-examination, rebuttal, closings.\n\n"
                "Structure your verdict:\n"
                "1. FINDINGS OF FACT — what was established\n"
                "2. PROSECUTION'S CASE — strongest points and weaknesses\n"
                "3. DEFENSE'S CASE — strongest points and weaknesses\n"
                "4. EVIDENCE EVALUATION — the 78% code similarity, access logs, expert testimony\n"
                "5. NON-COMPETE RULING — enforceable or not, with legal reasoning\n"
                "6. VERDICT — GUILTY, NOT GUILTY, or MISTRIAL with clear rationale\n\n"
                "Be thorough and judicial. This is a formal ruling."
            ),
        )
        .text("Adjournment", template="\n--- Court is adjourned ---")
    .end()

    # FALSE: loop back
    .on("false")
        .text("Next round", template="", show_output=False)
    .end()

    .end()
)

# -- Wire the loop back-edge -----------------------------------------------

trial.connect("Next round", "Round announce", label="next_round")

# -- Build and run ----------------------------------------------------------

agent = trial.build(validate=False)

print(f"Graph: {len(agent.nodes)} nodes, {len(agent.edges)} edges")
print()

run_graph(
    agent,
    user_input=(
        "Dr. Sarah Chen, a senior AI engineer, left TechCorp after 5 years "
        "to co-found NeuralStart. TechCorp alleges she copied proprietary "
        "training data and model architectures, violating her NDA and "
        "2-year non-compete. Forensic analysis shows 78% code similarity. "
        "The defense argues the non-compete is overly broad and the code "
        "comes from open-source frameworks."
    ),
)
