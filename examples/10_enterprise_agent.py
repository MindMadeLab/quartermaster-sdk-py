"""Enterprise multi-department agent -- advanced patterns showcase.

Demonstrates ALL advanced graph-builder patterns in a single, realistic
enterprise support agent:

  - Sub-graphs        -- reusable department handlers (HR, IT, Finance)
  - Decision routing  -- route requests to the right department
  - IF quality checks -- conditional logic for response review
  - Memory            -- audit logging and state persistence
  - Notifications     -- alerts for escalations
  - Parallel sections -- concurrent processing within a branch

Architecture::

    START -> User -> Classify
          -> DECISION(department)
             |-- hr:      [HR sub-graph: analyse -> IF(approval?) -> ...]
             |-- it:      [IT sub-graph: diagnose -> parallel(security + performance) -> fix]
             |-- finance: [Finance sub-graph: analyse -> IF(large amount?) -> ...]
             |-- general: [general assistance]
          -> MERGE
          -> Quality check (IF score > 0.8)
             |-- true:  deliver
             |-- false: improve -> re-deliver
          -> MERGE
          -> write_memory(audit) -> notification(complete) -> log(audit)
          -> END
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

# ---------------------------------------------------------------------------
# Sub-graph: HR department
# ---------------------------------------------------------------------------
# Analyses HR queries and conditionally escalates for manager approval.

hr_flow = (
    Graph("HR Handler")
    .start()
    .instruction("HR analysis", system_instruction="Analyse HR-related query and determine policy")
    .if_node("Needs approval?", expression="requires_manager_approval")
    .on("true")
        .notification("Alert manager", channel="email", message="HR request requires manager approval")
        .user("Awaiting manager response")
    .end()
    .on("false")
        .instruction("Direct HR response", system_instruction="Provide HR information from policy database")
    .end()
    .merge("HR resolution")
    .write_memory("Log HR query", key="hr_query_log")
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
    .instruction("IT diagnosis", system_instruction="Diagnose the reported IT issue")

    # Parallel: run security and performance checks concurrently
    .parallel()
    .branch()
        .instruction("Security check", system_instruction="Check for security implications")
    .end()
    .branch()
        .instruction("Performance check", system_instruction="Assess performance impact")
    .end()
    .merge("Combine IT checks")

    .instruction("Suggest fix", system_instruction="Provide troubleshooting steps based on diagnosis and checks")
    .end()
)

# ---------------------------------------------------------------------------
# Sub-graph: Finance department
# ---------------------------------------------------------------------------
# Handles finance queries with an IF for large amounts requiring CFO sign-off.

finance_flow = (
    Graph("Finance Handler")
    .start()
    .instruction("Finance analysis", system_instruction="Analyse the finance-related request")
    .if_node("Large amount?", expression="amount > 10000")
    .on("true")
        .notification("CFO alert", channel="slack", message="Finance request over $10k requires CFO approval")
        .instruction("Prepare CFO brief", system_instruction="Summarise request for CFO review")
    .end()
    .on("false")
        .instruction("Process directly", system_instruction="Handle finance request within standard limits")
    .end()
    .merge("Finance resolution")
    .write_memory("Log finance query", key="last_finance_query")
    .end()
)

# ---------------------------------------------------------------------------
# Main enterprise agent
# ---------------------------------------------------------------------------

agent = (
    Graph("Enterprise Assistant")
    .start()
    .user("How can I help you today?")
    .write_memory("Log session start", key="session_start", value="{{timestamp}}")
    .instruction("Classify request", system_instruction="Classify the request into: hr, it, finance, or general")

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
        .instruction("General help", system_instruction="Provide general assistance")
    .end()
    .merge("Collect department response")

    # --- Quality gate ---------------------------------------------------------
    .instruction("Quality check", system_instruction="Review the response for accuracy and completeness (0-1 score)")
    .if_node("Quality OK?", expression="quality_score > 0.8")
    .on("true")
        .instruction("Deliver", system_instruction="Format and deliver the final response")
    .end()
    .on("false")
        .instruction("Improve", system_instruction="Rewrite the response to improve clarity and accuracy")
        .instruction("Re-deliver", system_instruction="Format and deliver the improved response")
    .end()
    .merge("Final output")

    # --- Audit trail ----------------------------------------------------------
    .write_memory("Audit log", key="audit_log", value="completed | dept:{{department}} | quality:{{quality_score}}")
    .notification("Completion notice", channel="internal", message="Request handled successfully")
    .log("Audit", message="Enterprise request completed", level="info")
    .end()
)

# ---------------------------------------------------------------------------
# Print graph stats
# ---------------------------------------------------------------------------

print("Enterprise Agent")
print(f"  Nodes: {len(agent.nodes)}")
print(f"  Edges: {len(agent.edges)}")

node_types: dict[str, int] = {}
for n in agent.nodes:
    t = n.type.value
    node_types[t] = node_types.get(t, 0) + 1
print(f"  Node types: {dict(sorted(node_types.items()))}")

print("\nFull node list:")
for node in agent.nodes:
    meta_summary = ""
    si = node.metadata.get("system_instruction", "")
    if si:
        meta_summary = f' -- "{si[:50]}"'
    ch = node.metadata.get("channel", "")
    if ch:
        meta_summary = f" -- channel={ch}"
    print(f"  [{node.type.value:15s}] {node.name}{meta_summary}")

print("\nEdge list:")
name_map = {n.id: n.name for n in agent.nodes}
for edge in agent.edges:
    label = f"  [{edge.label}]" if edge.label else ""
    print(f"  {name_map[edge.source_id]} -> {name_map[edge.target_id]}{label}")
