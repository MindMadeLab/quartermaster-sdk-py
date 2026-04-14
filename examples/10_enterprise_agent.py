"""Enterprise multi-department agent -- advanced patterns showcase.

Demonstrates ALL advanced graph-builder patterns in a single, realistic
enterprise support agent:

  - Sub-graphs        -- reusable department handlers (HR, IT, Finance)
  - Decision routing  -- route requests to the right department
  - IF quality checks -- conditional logic for response review
  - Memory            -- audit logging and state persistence
  - Notifications     -- alerts for escalations
  - Parallel sections -- concurrent processing within a branch

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/10_enterprise_agent.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from _runner import run_graph

# ---------------------------------------------------------------------------
# Sub-graph: HR department
# ---------------------------------------------------------------------------
# Analyses HR queries and conditionally escalates for manager approval.

hr_flow = (
    Graph("HR Handler")
    .start()
    .instruction("HR analysis", model="claude-sonnet-4-20250514", system_instruction="Analyse HR-related query and determine policy")
    .if_node("Needs approval?", expression="requires_manager_approval")
    .on("true")
        .static("Alert manager", text="HR request requires manager approval")
        .user("Awaiting manager response")
    .end()
    .on("false")
        .instruction("Direct HR response", model="claude-sonnet-4-20250514", system_instruction="Provide HR information from policy database")
    .end()
    # No merge — IF picks one branch, they converge here.
    .write_memory("Log HR query", memory_name="hr_query_log")
    .end()
)

# ---------------------------------------------------------------------------
# Sub-graph: IT department
# ---------------------------------------------------------------------------
# Diagnoses the issue, then runs security and performance checks in parallel
# before suggesting a fix.

it_flow = (
    Graph("IT Handler")
    .start()
    .instruction("IT diagnosis", model="claude-sonnet-4-20250514", system_instruction="Diagnose the reported IT issue")

    # Parallel: run security and performance checks concurrently
    .parallel()
    .branch()
        .instruction("Security check", model="claude-sonnet-4-20250514", system_instruction="Check for security implications")
    .end()
    .branch()
        .instruction("Performance check", model="claude-sonnet-4-20250514", system_instruction="Assess performance impact")
    .end()
    .static_merge("Combine IT checks")

    .instruction("Suggest fix", model="claude-sonnet-4-20250514", system_instruction="Provide troubleshooting steps based on diagnosis and checks")
    .end()
)

# ---------------------------------------------------------------------------
# Sub-graph: Finance department
# ---------------------------------------------------------------------------
# Handles finance queries with an IF for large amounts requiring CFO sign-off.

finance_flow = (
    Graph("Finance Handler")
    .start()
    .instruction("Finance analysis", model="claude-sonnet-4-20250514", system_instruction="Analyse the finance-related request")
    .if_node("Large amount?", expression="amount > 10000")
    .on("true")
        .static("CFO alert", text="Finance request over $10k requires CFO approval")
        .instruction("Prepare CFO brief", model="claude-sonnet-4-20250514", system_instruction="Summarise request for CFO review")
    .end()
    .on("false")
        .instruction("Process directly", model="claude-sonnet-4-20250514", system_instruction="Handle finance request within standard limits")
    .end()
    # No merge — IF picks one branch.
    .write_memory("Log finance query", memory_name="last_finance_query")
    .end()
)

# ---------------------------------------------------------------------------
# Main enterprise agent
# ---------------------------------------------------------------------------

agent = (
    Graph("Enterprise Assistant")
    .start()
    .user("How can I help you today?")
    .write_memory("Log session start", memory_name="session_start", variables=[{"name": "timestamp", "value": "{{timestamp}}"}])
    .instruction("Classify request", model="claude-sonnet-4-20250514", system_instruction="Classify the request into: hr, it, finance, or general")

    # --- Department routing ---------------------------------------------------
    .decision("Department?", options=["hr", "it", "finance", "general"])
    .on("hr")
        .use(hr_flow)
    .end()
    .on("it")
        .use(it_flow)
    .end()
    .on("finance")
        .use(finance_flow)
    .end()
    .on("general")
        .instruction("General help", model="claude-sonnet-4-20250514", system_instruction="Provide general assistance")
    .end()
    # No merge after decision — only one department branch runs.

    # --- Quality gate ---------------------------------------------------------
    .instruction("Quality check", model="claude-sonnet-4-20250514", system_instruction="Review the response for accuracy and completeness (0-1 score)")
    .if_node("Quality OK?", expression="quality_score > 0.8")
    .on("true")
        .instruction("Deliver", model="claude-sonnet-4-20250514", system_instruction="Format and deliver the final response")
    .end()
    .on("false")
        .instruction("Improve", model="claude-sonnet-4-20250514", system_instruction="Rewrite the response to improve clarity and accuracy")
        .instruction("Re-deliver", model="claude-sonnet-4-20250514", system_instruction="Format and deliver the improved response")
    .end()
    # No merge after IF — only one branch runs.

    # --- Audit trail ----------------------------------------------------------
    .write_memory("Audit log", memory_name="audit_log", variables=[{"name": "audit", "value": "completed | dept:{{department}} | quality:{{quality_score}}"}])
    .static("Completion notice", text="Request handled successfully")
    .static("Audit", text="Enterprise request completed")
    .end()
)

# Execute with a real LLM
run_graph(agent, user_input="My laptop won't connect to the VPN and I need access to the finance portal urgently")
