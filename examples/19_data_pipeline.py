"""Example 19 -- Data processing pipeline.

Demonstrates a multi-stage data pipeline with parallel analysis branches,
conditional routing, memory accumulation, and mixed providers.

Patterns demonstrated
---------------------
  - var()           -- capture intermediate results
  - text()          -- Jinja2 templates for formatting
  - if_node()       -- conditional routing based on data
  - parallel()      -- fan-out for independent analyses
  - static_merge()  -- combine parallel results
  - write_memory()  -- accumulate findings
  - Mixed providers -- groq for fast nodes, anthropic for quality

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GROQ_API_KEY="..."
    uv run examples/19_data_pipeline.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_engine import run_graph

# -- Provider config --------------------------------------------------------

FAST  = dict(model="llama-3.3-70b-versatile", provider="groq")
SMART = dict(model="claude-haiku-4-5-20251001", provider="anthropic")

# -- Pipeline graph ---------------------------------------------------------

pipeline = (
    Graph("Data Processing Pipeline")
    .start()

    # --- Stage 1: Collect topic from user -----------------------------------
    .user("Enter a topic to research")
    .var("Capture topic", variable="topic")
    .write_memory("Store topic", memory_name="topic")

    .text("Acknowledge", template=(
        "=== Data Pipeline ===\n"
        "Topic: {{topic}}\n"
        "Generating dataset..."
    ))

    # --- Stage 2: Generate mock dataset (fast provider) ---------------------
    .instruction(
        "Generate data", **FAST,
        system_instruction=(
            "Generate a JSON array of exactly 5 records about the topic.\n"
            "Each record must have: id (1-5), title, category (one of: "
            "technology, science, business), and importance (high or low).\n"
            "Output ONLY valid JSON, nothing else."
        ),
    )
    .var("Capture dataset", variable="dataset", show_output=False)
    .write_memory("Store raw data", memory_name="raw_dataset")

    # --- Stage 3: Quick classification (fast provider) ----------------------
    .instruction(
        "Classify content", **FAST,
        system_instruction=(
            "Look at the dataset above. Determine the dominant category "
            "(technology, science, or business). Reply with a single word: "
            "the dominant category name. Nothing else."
        ),
    )
    .var("Capture category", variable="dominant_category", show_output=False)

    # --- Stage 4: Conditional routing based on category ---------------------
    .if_node("Is technology?", expression="dominant_category == 'technology'")

    .on("true")
        .text("Tech path", template=(
            "\n--- Technology Track ---\n"
            "Routing to deep technical analysis..."
        ))
        .instruction(
            "Tech deep dive", **SMART,
            system_instruction=(
                "You received a dataset about technology topics.\n"
                "Provide a technical analysis: identify trends, key "
                "innovations, and potential disruptions. Be specific."
            ),
        )
    .end()

    .on("false")
        .text("General path", template=(
            "\n--- General Track ---\n"
            "Dominant category: {{dominant_category}}\n"
            "Routing to general analysis..."
        ))
        .instruction(
            "General analysis", **SMART,
            system_instruction=(
                "You received a dataset. The dominant category is "
                "'{{dominant_category}}'. Provide an analysis covering: "
                "main themes, notable patterns, and recommendations."
            ),
        )
    .end()

    .var("Capture analysis", variable="primary_analysis", show_output=False)
    .write_memory("Store analysis", memory_name="primary_analysis")

    # --- Stage 5: Parallel analysis branches --------------------------------
    .text("Fan-out header", template=(
        "\n=== Parallel Analysis Phase ===\n"
        "Running three independent analyses concurrently..."
    ))

    .parallel("Multi-angle analysis")

    # Branch A: Sentiment analysis
    .branch()
        .instruction(
            "Sentiment analysis", **FAST,
            system_instruction=(
                "Analyse the sentiment of the dataset records. For each "
                "record, classify as positive, neutral, or negative. "
                "Provide a brief overall sentiment summary."
            ),
        )
    .end()

    # Branch B: Gap analysis
    .branch()
        .instruction(
            "Gap analysis", **FAST,
            system_instruction=(
                "Identify what is MISSING from this dataset. What topics, "
                "perspectives, or data points are absent? Suggest 3 specific "
                "additions that would make the dataset more complete."
            ),
        )
    .end()

    # Branch C: Quality check with conditional gate
    .branch()
        .if_node("Data quality OK?", expression="data_quality_score > 0.7")
        .on("true")
            .text("Quality pass", template="Data quality: PASS -- dataset meets standards")
        .end()
        .on("false")
            .instruction(
                "Quality remediation", **FAST,
                system_instruction=(
                    "The dataset has quality issues. Identify problems "
                    "(missing fields, inconsistencies, duplicates) and "
                    "suggest specific corrections."
                ),
            )
        .end()
        .static("Quality gate done", text="Quality assessment complete")
    .end()

    .static_merge("Combine analyses")

    # --- Stage 6: Final synthesis (smart provider) --------------------------
    .read_memory("Recall topic", memory_name="topic")
    .read_memory("Recall analysis", memory_name="primary_analysis")

    .instruction(
        "Final synthesis", **SMART,
        system_instruction=(
            "You are writing the final report for a data pipeline.\n"
            "Combine ALL previous analyses (primary analysis, sentiment, "
            "gap analysis, quality check) into a coherent executive summary.\n\n"
            "Structure:\n"
            "1. OVERVIEW -- what was analysed and why\n"
            "2. KEY FINDINGS -- the most important discoveries\n"
            "3. DATA QUALITY -- assessment and any issues\n"
            "4. RECOMMENDATIONS -- actionable next steps\n\n"
            "Be concise but thorough."
        ),
    )

    # --- Stage 7: Format and deliver ----------------------------------------
    .write_memory("Store final report", memory_name="final_report")
    .text("Pipeline complete", template=(
        "\n=== Pipeline Complete ===\n"
        "Topic: {{topic}}\n"
        "Category: {{dominant_category}}\n"
        "Report stored in memory as 'final_report'"
    ))
    .end()
)

# -- Execute ----------------------------------------------------------------

run_graph(
    pipeline,
    user_input="Large language models and their impact on software engineering",
)
