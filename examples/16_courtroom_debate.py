"""Courtroom debate -- prosecution vs defense across three rounds, judged by AI.

Three rounds of structured legal argument between two AI attorneys, each
using a different LLM provider. A judge (third provider) evaluates each
round and delivers the final verdict.

  Round 1: Opening statements (prosecution presents, defense responds)
  Round 2: Evidence and cross-examination (strongest arguments from both sides)
  Round 3: Closing arguments (final appeals to the court)
  Verdict: Judge deliberates and delivers judgment

Providers:
  - Prosecution: xAI Grok (aggressive, confrontational style)
  - Defense: OpenAI GPT-4o (methodical, precedent-based style)
  - Judge: Anthropic Claude (balanced, analytical deliberation)

Usage:
    export ANTHROPIC_API_KEY="..."
    export OPENAI_API_KEY="..."
    export XAI_API_KEY="..."
    uv run examples/16_courtroom_debate.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from _runner import run_graph


PROSECUTION_MODEL = "grok-3-mini-fast"
PROSECUTION_PROVIDER = "xai"

DEFENSE_MODEL = "gpt-4o"
DEFENSE_PROVIDER = "openai"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROVIDER = "anthropic"


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
        "JUDGE: We will hear three rounds of argument.\n"
    ))

    # =====================================================================
    # ROUND 1 — Opening Statements
    # =====================================================================

    .text("Round 1 header", template="\n━━━ ROUND 1: OPENING STATEMENTS ━━━\n")

    .instruction(
        "Prosecution opening",
        model=PROSECUTION_MODEL, provider=PROSECUTION_PROVIDER,
        system_instruction=(
            "You are the lead prosecutor in a trade secrets theft case. "
            "This is your OPENING STATEMENT to the court.\n\n"
            "Present:\n"
            "- The charges against Dr. Sarah Chen\n"
            "- Your theory of the case: she copied proprietary data before leaving\n"
            "- Key evidence you will present (access logs, code similarity, timeline)\n"
            "- Why the non-compete agreement is valid and was breached\n\n"
            "Be forceful and persuasive. 3-4 paragraphs. Address the judge directly."
        ),
    )

    .instruction(
        "Defense opening",
        model=DEFENSE_MODEL, provider=DEFENSE_PROVIDER,
        system_instruction=(
            "You are the lead defense attorney. You just heard the prosecution's "
            "opening statement. Now deliver YOUR opening statement.\n\n"
            "Present:\n"
            "- Why your client Dr. Chen is innocent\n"
            "- Challenge the prosecution's narrative point by point\n"
            "- Your theory: she only used publicly available techniques\n"
            "- Why the non-compete is overly broad and unenforceable under state law\n"
            "- Legal precedents supporting your position\n\n"
            "Be confident and methodical. 3-4 paragraphs. Address the judge directly."
        ),
    )

    .instruction(
        "Judge evaluates Round 1",
        model=JUDGE_MODEL, provider=JUDGE_PROVIDER,
        system_instruction=(
            "You are the presiding judge. You just heard opening statements from "
            "both prosecution and defense.\n\n"
            "Provide a brief evaluation (2 paragraphs):\n"
            "1. Note the key claims from each side\n"
            "2. Identify what evidence will be critical to resolve\n"
            "3. Any procedural notes\n\n"
            "End with: 'The court will now hear evidence. Prosecution, call your first witness.'"
        ),
    )

    .update_memory("Log Round 1", memory_name="trial_transcript")

    # =====================================================================
    # ROUND 2 — Evidence and Cross-Examination
    # =====================================================================

    .text("Round 2 header", template="\n━━━ ROUND 2: EVIDENCE & CROSS-EXAMINATION ━━━\n")

    .instruction(
        "Prosecution presents evidence",
        model=PROSECUTION_MODEL, provider=PROSECUTION_PROVIDER,
        system_instruction=(
            "You are the prosecutor. Present your STRONGEST EVIDENCE:\n\n"
            "1. Forensic evidence: server access logs showing bulk downloads\n"
            "2. Expert testimony: code similarity analysis (78%% structural match)\n"
            "3. Timeline: downloads happened 48 hours before resignation\n"
            "4. Witness: IT security officer testifying about data exfiltration alerts\n\n"
            "Present this as direct examination of your expert witness. Be specific "
            "with technical details. 3-4 paragraphs."
        ),
    )

    .instruction(
        "Defense cross-examines",
        model=DEFENSE_MODEL, provider=DEFENSE_PROVIDER,
        system_instruction=(
            "You are the defense attorney conducting CROSS-EXAMINATION of the "
            "prosecution's expert witness.\n\n"
            "Attack their evidence:\n"
            "1. Challenge the code similarity analysis methodology\n"
            "2. Show that 'bulk downloads' were routine backups she always did\n"
            "3. Present counter-evidence: the startup's code is based on open-source\n"
            "4. Question the expert's qualifications and potential bias\n"
            "5. Introduce your own expert's findings contradicting the prosecution\n\n"
            "Be sharp and surgical. Find contradictions. 3-4 paragraphs."
        ),
    )

    .instruction(
        "Prosecution redirect",
        model=PROSECUTION_MODEL, provider=PROSECUTION_PROVIDER,
        system_instruction=(
            "You are the prosecutor. The defense just challenged your evidence. "
            "Conduct a REDIRECT examination to rehabilitate your case:\n\n"
            "1. Address the defense's challenges to your code similarity analysis\n"
            "2. Present additional corroborating evidence\n"
            "3. Highlight inconsistencies in the defense's counter-narrative\n"
            "4. Reinforce the timeline of suspicious activity\n\n"
            "Be direct and focused. 2-3 paragraphs."
        ),
    )

    .instruction(
        "Judge evaluates Round 2",
        model=JUDGE_MODEL, provider=JUDGE_PROVIDER,
        system_instruction=(
            "You are the presiding judge evaluating Round 2 evidence.\n\n"
            "Assess (2 paragraphs):\n"
            "1. Which evidence was most compelling?\n"
            "2. Were the defense's challenges to the forensic evidence effective?\n"
            "3. What remains unresolved?\n\n"
            "End with: 'We will now hear closing arguments.'"
        ),
    )

    .update_memory("Log Round 2", memory_name="trial_transcript")

    # =====================================================================
    # ROUND 3 — Closing Arguments
    # =====================================================================

    .text("Round 3 header", template="\n━━━ ROUND 3: CLOSING ARGUMENTS ━━━\n")

    .instruction(
        "Prosecution closing",
        model=PROSECUTION_MODEL, provider=PROSECUTION_PROVIDER,
        system_instruction=(
            "You are the prosecutor delivering your CLOSING ARGUMENT.\n\n"
            "This is your last chance to convince the judge. Be powerful:\n"
            "1. Summarize the evidence that proves guilt beyond reasonable doubt\n"
            "2. Connect the dots: motive, opportunity, and means\n"
            "3. Address and dismiss the defense's arguments\n"
            "4. Appeal to the importance of protecting intellectual property\n"
            "5. Ask the judge for a guilty verdict\n\n"
            "Make it memorable and compelling. 3-4 paragraphs."
        ),
    )

    .instruction(
        "Defense closing",
        model=DEFENSE_MODEL, provider=DEFENSE_PROVIDER,
        system_instruction=(
            "You are the defense attorney delivering your CLOSING ARGUMENT.\n\n"
            "Fight for acquittal with everything you have:\n"
            "1. Reasonable doubt: the prosecution hasn't proven their case\n"
            "2. The code similarity can be explained by common patterns\n"
            "3. The non-compete is unconscionable and unenforceable\n"
            "4. Your client's new work is original and based on public research\n"
            "5. The real motive: TechCorp is trying to suppress competition\n\n"
            "End with a passionate plea for justice. 3-4 paragraphs."
        ),
    )

    # =====================================================================
    # VERDICT — Judge deliberates and decides
    # =====================================================================

    .text("Deliberation header", template=(
        "\n━━━ JUDICIAL DELIBERATION ━━━\n"
        "BAILIFF: The court will now deliberate. Please remain seated.\n"
    ))

    .instruction("Judge deliberates", model=JUDGE_MODEL, provider=JUDGE_PROVIDER, system_instruction="You are the judge. Deliberate carefully on all evidence and arguments presented. Weigh the prosecution's and defense's cases methodically before reaching your verdict.")

    .instruction(
        "Final verdict",
        model=JUDGE_MODEL, provider=JUDGE_PROVIDER,
        system_instruction=(
            "You are the presiding judge delivering the FINAL VERDICT after hearing "
            "three rounds of argument in this trade secrets case.\n\n"
            "Your verdict must include:\n"
            "1. Summary of the prosecution's strongest points\n"
            "2. Summary of the defense's strongest points\n"
            "3. Your analysis of the key evidence (code similarity, access logs, timeline)\n"
            "4. Legal standard: was trade secret misappropriation proven?\n"
            "5. Assessment of the non-compete clause enforceability\n"
            "6. Your ruling with clear reasoning\n\n"
            "Deliver the verdict formally. End with exactly one word on its own line: "
            "guilty, not_guilty, or mistrial."
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
            "Dr. Chen, you are free to go.\n"
            "DEFENSE: Justice has been served."
        ))
    .end()

    .on("mistrial")
        .text("Mistrial declared", template=(
            "\n══════════════════════════════════\n"
            "    VERDICT: M I S T R I A L\n"
            "══════════════════════════════════\n"
            "JUDGE: The court declares a MISTRIAL due to insufficient evidence.\n"
            "The case may be retried."
        ))
    .end()

    .text("Court adjourned", template=(
        "\n--- Court is adjourned ---\n"
        "Three rounds of argument heard. Verdict delivered."
    ))
    .end()
)

agent = trial.build()

run_graph(
    agent,
    user_input=(
        "A senior AI engineer, Dr. Sarah Chen, left TechCorp after 5 years to "
        "co-found NeuralStart, an AI startup. TechCorp alleges she copied "
        "proprietary training datasets, custom model architectures, and internal "
        "benchmark results before leaving, violating her NDA and 2-year non-compete. "
        "Forensic analysis shows 78%% code similarity between TechCorp's and "
        "NeuralStart's codebases. The defense argues the non-compete is overly "
        "broad, the similar code comes from common open-source frameworks, and "
        "Dr. Chen's new work is based on publicly available research papers."
    ),
)
