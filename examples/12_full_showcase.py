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
  - Summarisation nodes
  - Logging and notifications
  - Multiple merge points

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/12_full_showcase.py
"""

from __future__ import annotations

import quartermaster_sdk as qm

# ---------------------------------------------------------------------------
# Sub-graph: Web research pipeline
# ---------------------------------------------------------------------------

web_research = (
    qm.Graph("Web Research")
    .instruction(
        "Web search",
        model="claude-haiku-4-5-20251001",
        system_instruction="Search the web for recent information on the topic",
    )
    .instruction(
        "Extract key facts",
        model="claude-haiku-4-5-20251001",
        system_instruction="Extract and list the key facts from search results",
    )
    .instruction(
        "Assess source quality",
        model="claude-haiku-4-5-20251001",
        system_instruction="Rate the reliability of each source (1-5)",
    )
)

# ---------------------------------------------------------------------------
# Sub-graph: Academic research pipeline
# ---------------------------------------------------------------------------

academic_research = (
    qm.Graph("Academic Research")
    .instruction(
        "Search papers",
        model="claude-haiku-4-5-20251001",
        system_instruction="Search academic databases for peer-reviewed papers",
    )
    .instruction(
        "Summarise papers",
        model="claude-haiku-4-5-20251001",
        system_instruction="Create structured summaries of the top papers",
    )
    .instruction(
        "Identify consensus",
        model="claude-haiku-4-5-20251001",
        system_instruction="Identify areas of scientific consensus and debate",
    )
)

# ---------------------------------------------------------------------------
# Main graph: AI Research Assistant
# ---------------------------------------------------------------------------

agent = (
    qm.Graph("AI Research Assistant")
    # --- Input and topic capture ----------------------------------------------
    .user("What do you want to research?")
    .var("Capture topic", variable="research_topic")
    .write_memory("Store topic", memory_name="research_topic")
    .text("Acknowledge", template="Researching: {{research_topic}}")
    # --- Strategy selection ----------------------------------------------------
    .instruction(
        "Classify research",
        model="claude-haiku-4-5-20251001",
        system_instruction="Classify the research type: academic, general, or technical",
    )
    .decision("Research strategy?", options=["academic", "general", "technical"])
    .on("academic")
    .use(academic_research)
    .end()
    .on("general")
    .use(web_research)
    .end()
    .on("technical")
    .instruction(
        "Technical deep-dive",
        model="claude-haiku-4-5-20251001",
        system_instruction="Perform in-depth technical analysis with code examples",
    )
    .instruction(
        "Benchmark review",
        model="claude-haiku-4-5-20251001",
        system_instruction="Review benchmarks and performance comparisons",
    )
    .end()
    # No merge — decision picks ONE research strategy.
    # --- Quality review: parallel checks with nested IF -----------------------
    .read_memory("Recall topic", memory_name="research_topic")
    .parallel("Quality review")
    # Branch 1: Fact-checking with pass/fail gate
    .branch()
    .instruction(
        "Fact-check",
        model="claude-haiku-4-5-20251001",
        system_instruction="Verify all factual claims against sources",
    )
    .if_node("Facts verified?", expression="verification_score > 0.9")
    .on("true")
    .text("Facts OK", template="All facts verified successfully")
    .end()
    .on("false")
    .instruction(
        "Fix errors",
        model="claude-haiku-4-5-20251001",
        system_instruction="Correct any unverified or inaccurate claims",
    )
    .end()
    # IF branches converge on a result node
    .static("Fact-check done", text="Fact-check complete")
    .end()
    # Branch 2: Bias assessment (no conditional, straight-through)
    .branch()
    .instruction(
        "Bias assessment",
        model="claude-haiku-4-5-20251001",
        system_instruction="Check for confirmation bias, source bias, and framing issues",
    )
    .end()
    # Branch 3: Completeness check with gap detection
    .branch()
    .if_node("Coverage gaps?", expression="has_coverage_gaps")
    .on("true")
    .instruction(
        "Fill gaps",
        model="claude-haiku-4-5-20251001",
        system_instruction="Research and fill identified coverage gaps",
    )
    .end()
    .on("false")
    .text(
        "Coverage complete",
        template="Research covers all key aspects of {{research_topic}}",
    )
    .end()
    # IF branches converge on a result node
    .static("Coverage done", text="Coverage check complete")
    .end()
    .static_merge("Quality review complete")
    # --- Synthesis and delivery -----------------------------------------------
    .instruction(
        "Synthesise findings",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction="Synthesise all research findings into a coherent analysis",
    )
    .summarize(
        "Executive summary",
        model="claude-haiku-4-5-20251001",
        system_instruction="Create a concise executive summary with key takeaways",
    )
    # --- Audit trail ----------------------------------------------------------
    .update_memory("Update status", memory_name="research_status")
    .static(
        "Report ready", text="Research report on {{research_topic}} is ready for review"
    )
    .static("Completed", text="Research pipeline completed for: {{research_topic}}")
)

# Execute with a real LLM
qm.run(agent, "What are the latest advances in quantum error correction?")
