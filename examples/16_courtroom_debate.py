"""Courtroom debate -- prosecution vs defense with loop-back for multiple rounds.

Demonstrates a LOOP in the graph: prosecution argues, defense rebuts, a round
counter increments, and an IF node checks whether to loop back or break out
to the verdict. The back-edge is wired manually after the graph is built.

  Opening   -> Prosecution argues -> Defense rebuts -> Increment round
            -> IF round > 5: break to verdict
            -> ELSE: loop back to "Prosecution argues"

Providers:
  - Prosecution: xAI Grok
  - Defense: OpenAI GPT-4o
  - Judge: Anthropic Claude

Usage:
    # Set API keys in .env at repo root
    uv run examples/16_courtroom_debate.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_graph.enums import TraverseIn
from quartermaster_engine import run_graph


PROSECUTION_MODEL = "grok-3-mini-fast"
PROSECUTION_PROVIDER = "xai"

DEFENSE_MODEL = "gpt-4o"
DEFENSE_PROVIDER = "openai"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROVIDER = "anthropic"

MAX_ROUNDS = 5

# =====================================================================
# Build the graph — the IF node's "true" branch leads to the verdict,
# while the "false" branch is a dead-end that we wire to the loop target.
# =====================================================================

trial = (
    Graph("The People v. AI Engineer")
    .start()

    # --- Case filing ---
    .user("Describe the case")
    .var("Capture case", variable="case_description")
    .write_memory("File case", memory_name="case_file")

    .text("Court opens", template=(
        "=== SUPERIOR COURT - DEPARTMENT 7 ===\n"
        "Case: TechCorp Inc. v. Dr. Sarah Chen\n"
        "Charges: Theft of trade secrets, breach of non-compete\n\n"
        "{{case_description}}\n\n"
        "BAILIFF: All rise. Court is now in session.\n"
        f"JUDGE: We will hear up to {MAX_ROUNDS} rounds of argument.\n"
    ))

    # --- Initialize round counter ---
    .var("Init round", variable="round_number", expression="1")

    # =====================================================================
    # DEBATE LOOP — Round header is the loop target
    # =====================================================================

    .text("Round header", template="\n━━━ ROUND {{round_number}} ━━━\n", traverse_in=TraverseIn.AWAIT_FIRST)

    .instruction(
        "Prosecution argues",
        model=PROSECUTION_MODEL, provider=PROSECUTION_PROVIDER,
        system_instruction=(
            "You are the lead prosecutor in a trade secrets case. "
            "Adapt your argument to the current round:\n"
            "  Round 1: Opening statement — lay out charges, theory, key evidence.\n"
            "  Round 2-3: Present evidence — forensic analysis, access logs, expert testimony.\n"
            "  Round 4: Rebut the defense's challenges, reinforce your strongest points.\n"
            "  Round 5: Closing argument — powerful summary, ask for guilty verdict.\n"
            "Be forceful. 2-3 paragraphs. Address the judge."
        ),
    )

    .instruction(
        "Defense rebuts",
        model=DEFENSE_MODEL, provider=DEFENSE_PROVIDER,
        system_instruction=(
            "You are the lead defense attorney. You just heard the prosecution. "
            "Counter their argument directly based on the round:\n"
            "  Round 1: Opening — challenge narrative, present your theory of innocence.\n"
            "  Round 2-3: Cross-examine evidence — challenge methodology, expose weaknesses.\n"
            "  Round 4: Present counter-evidence, show prosecution's case has holes.\n"
            "  Round 5: Closing — passionate plea for acquittal, invoke reasonable doubt.\n"
            "Be sharp, find contradictions. 2-3 paragraphs. Address the judge."
        ),
    )

    # --- Increment round and check loop condition ---
    .var("Next round", variable="round_number", expression="round_number + 1")

    .if_node("More rounds?", expression=f"round_number > {MAX_ROUNDS}")

    # TRUE branch: debate is over → proceed to verdict
    .on("true")
        .text("Debate complete", template=(
            f"\n━━━ DEBATE CONCLUDED AFTER {MAX_ROUNDS} ROUNDS ━━━\n"
            "JUDGE: The court has heard sufficient argument. We proceed to deliberation.\n"
        ))
        .instruction(
            "Judge deliberates",
            model=JUDGE_MODEL, provider=JUDGE_PROVIDER,
            system_instruction=(
                "You are the presiding judge. You have heard multiple rounds of argument "
                "in this trade secrets case. Deliver your FINAL VERDICT.\n\n"
                "Include:\n"
                "1. Summary of prosecution's strongest points\n"
                "2. Summary of defense's strongest points\n"
                "3. Analysis of key evidence (code similarity, access logs, timeline)\n"
                "4. Whether trade secret misappropriation was proven\n"
                "5. Whether the non-compete is enforceable\n"
                "6. Your ruling with clear reasoning\n\n"
                "End with exactly one word on its own line: guilty, not_guilty, or mistrial."
            ),
        )
        .decision("Verdict?", options=["guilty", "not_guilty", "mistrial"])
        .on("guilty")
            .text("Guilty verdict", template=(
                "\n══════════════════════════════════\n"
                "       VERDICT: G U I L T Y\n"
                "══════════════════════════════════\n"
                "JUDGE: The defendant is found GUILTY of theft of trade secrets.\n"
                "Sentencing hearing scheduled for next month.\n"
                "BAILIFF: Order in the court!"
            ))
        .end()
        .on("not_guilty")
            .text("Not guilty verdict", template=(
                "\n══════════════════════════════════\n"
                "    VERDICT: N O T  G U I L T Y\n"
                "══════════════════════════════════\n"
                "JUDGE: The defendant is found NOT GUILTY. Charges dismissed.\n"
                "Dr. Chen, you are free to go."
            ))
        .end()
        .on("mistrial")
            .text("Mistrial declared", template=(
                "\n══════════════════════════════════\n"
                "    VERDICT: M I S T R I A L\n"
                "══════════════════════════════════\n"
                "JUDGE: The court declares a MISTRIAL.\n"
                "The case may be retried."
            ))
        .end()
        .text("Court adjourned", template="\n--- Court is adjourned ---")
    .end()

    # FALSE branch: loop back (dead-end text — we wire the back-edge manually)
    .on("false")
        .text("Continue debate", template="\nJUDGE: The court will hear further argument.")
    .end()

    .end()
)

# ---------------------------------------------------------------------------
# Wire the loop back-edge: "Continue debate" -> "Round header"
# ---------------------------------------------------------------------------
trial.connect("Continue debate", "Round header", label="next_round")

# Build (skip validation — intentional cycle)
agent = trial.build(validate=False)

print(f"Graph: {len(agent.nodes)} nodes, {len(agent.edges)} edges (includes back-edge for loop)")
print()

run_graph(
    agent,
    user_input=(
        "A senior AI engineer, Dr. Sarah Chen, left TechCorp after 5 years to "
        "co-found NeuralStart, an AI startup. TechCorp alleges she copied "
        "proprietary training datasets, custom model architectures, and internal "
        "benchmark results before leaving, violating her NDA and 2-year non-compete. "
        "Forensic analysis shows 78% code similarity between TechCorp's and "
        "NeuralStart's codebases. The defense argues the non-compete is overly "
        "broad, the similar code comes from common open-source frameworks, and "
        "Dr. Chen's new work is based on publicly available research papers."
    ),
)
