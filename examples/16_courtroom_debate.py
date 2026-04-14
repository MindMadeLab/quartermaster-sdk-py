"""Courtroom debate -- multi-round prosecution vs defense with loop-back.

Demonstrates sub-graphs, loops, IF branching, memory, and mixed providers
in a courtroom trial simulation. The debate round is a reusable sub-graph
that gets looped 5 times via a back-edge.

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


# -- Sub-graph: one round of debate ----------------------------------------

debate_round = (
    Graph("Debate Round")
    .start()
    .instruction(
        "Prosecution argues", **PROSECUTOR,
        system_instruction=(
            "You are the lead prosecutor. Adapt to the round:\n"
            "  Early rounds: present evidence, call witnesses, build your case.\n"
            "  Later rounds: rebut defense, reinforce strongest points.\n"
            "  Final round: powerful closing argument.\n"
            "Be forceful. 2-3 paragraphs. Address the judge."
        ),
    )
    .instruction(
        "Defense rebuts", **DEFENSE,
        system_instruction=(
            "You are the defense attorney. Counter the prosecution directly:\n"
            "  Early rounds: challenge evidence, present your theory.\n"
            "  Later rounds: cross-examine, expose contradictions.\n"
            "  Final round: passionate plea for acquittal.\n"
            "Be sharp. 2-3 paragraphs. Address the judge."
        ),
    )
    .end()
)


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

    # -- Loop target: round header --
    .text("Round announce", template=(
        "\n──── Round {{round_number}} of 5 ────"
    ), traverse_in=TraverseIn.AWAIT_FIRST)

    # -- Inline the debate sub-graph --
    .use(debate_round)

    # -- Increment and check --
    .var("Increment", variable="round_number", expression="round_number + 1", show_output=False)
    .if_node("More rounds?", expression=f"round_number > {MAX_ROUNDS}", show_output=False)

    # TRUE: done debating -> verdict
    .on("true")
        .text("Debate over", template=(
            "\n═══════════════════════════════════════════\n"
            "  CLOSING DEBATE — PROCEEDING TO VERDICT\n"
            "═══════════════════════════════════════════"
        ))
        .instruction(
            "Judge delivers verdict", **JUDGE,
            system_instruction=(
                "You are the presiding judge. After hearing all rounds:\n"
                "1. Summarize each side's strongest points\n"
                "2. Evaluate the key evidence\n"
                "3. Rule on the non-compete enforceability\n"
                "4. Deliver your verdict with reasoning\n\n"
                "End with: VERDICT: GUILTY, VERDICT: NOT GUILTY, or VERDICT: MISTRIAL"
            ),
        )
        .text("Adjournment", template="\n--- Court is adjourned ---")
    .end()

    # FALSE: loop back
    .on("false")
        .text("Next round", template="JUDGE: We continue.", show_output=False)
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
