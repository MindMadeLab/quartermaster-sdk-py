"""Example 18 -- Privacy and compliance pipeline.

Demonstrates built-in PII detection, redaction, risk classification,
and audit logging. EU AI Act compliance tools ship with Quartermaster.

Part 1: Direct tool usage (no LLM required)
  - detect_pii   -- find emails, phones, SSNs, credit cards in text
  - redact_pii   -- replace PII with labels, masks, or hashes
  - risk_classifier -- classify AI system risk per EU AI Act
  - audit_log    -- tamper-evident JSON Lines audit trail
  - read_audit_log -- query and verify the audit chain

Part 2: Graph pipeline
  - LLM processes the CLEANED text (after PII removal)
  - Shows how compliance tools integrate with agent workflows

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/18_compliance_guard.py
"""

from __future__ import annotations

import os
import tempfile

from quartermaster_tools.builtin.privacy.detect import DetectPIITool, ScanFilePIITool
from quartermaster_tools.builtin.privacy.redact import RedactPIITool
from quartermaster_tools.builtin.compliance.risk_classifier import RiskClassifierTool
from quartermaster_tools.builtin.compliance.audit_log import (
    AuditLogTool,
    ReadAuditLogTool,
)

import quartermaster_sdk as qm


# ============================================================================
# Part 1: Direct tool usage -- no LLM needed
# ============================================================================

print("=" * 60)
print("  PART 1: Built-in compliance tools (no LLM)")
print("=" * 60)

# -- Sample text with embedded PII ------------------------------------------

SAMPLE_TEXT = (
    "Customer complaint from John Smith (john.smith@example.com). "
    "Phone: +1 (555) 123-4567. SSN: 123-45-6789. "
    "Credit card ending 4532-1234-5678-9012 was charged incorrectly. "
    "Please investigate and respond to john.smith@example.com. "
    "Internal IP: 192.168.1.42."
)

print(f"\nInput text:\n  {SAMPLE_TEXT}\n")

# -- Step 1: Detect PII -----------------------------------------------------

detected = DetectPIITool.run(text=SAMPLE_TEXT)
print(f"Step 1 -- PII Detection: found {detected.data['count']} entities")
for entity in detected.data["entities"]:
    print(f"  [{entity['type']:20s}] {entity['value']}")

# -- Step 2: Redact PII (three strategies) -----------------------------------

print("\nStep 2 -- PII Redaction:")

for strategy in ("redact", "mask", "hash"):
    result = RedactPIITool.run(text=SAMPLE_TEXT, strategy=strategy)
    # Show a truncated preview
    preview = result.data["redacted_text"][:80] + "..."
    print(f"  {strategy:6s} -> {preview}")

# Keep the redacted version for Part 2
redacted_result = RedactPIITool.run(text=SAMPLE_TEXT, strategy="redact")
clean_text = redacted_result.data["redacted_text"]

print(f"\nClean text for LLM:\n  {clean_text}\n")

# -- Step 3: Risk classification --------------------------------------------

print("Step 3 -- EU AI Act Risk Classification:")

classification = RiskClassifierTool.run(
    system_description="Customer complaint analysis system using NLP",
    domain="other",
    uses_biometrics=False,
)
print(f"  Risk level : {classification.data['risk_level']}")
print(f"  Category   : {classification.data['category']}")
print(f"  Reasoning  : {classification.data['reasoning']}")
print(f"  Obligations: {len(classification.data['obligations'])} items")

# Also classify a high-risk scenario for comparison
high_risk = RiskClassifierTool.run(
    system_description="AI system for screening job applicants",
    domain="employment",
    uses_biometrics=False,
)
print(f"\n  Comparison (employment domain):")
print(f"  Risk level : {high_risk.data['risk_level']}")
print(f"  Obligations: {len(high_risk.data['obligations'])} items")
for obligation in high_risk.data["obligations"][:3]:
    print(f"    - {obligation}")
print(f"    ... and {len(high_risk.data['obligations']) - 3} more")

# -- Step 4: Audit logging ---------------------------------------------------

print("\nStep 4 -- Tamper-evident Audit Trail:")

# Use a temp file so we don't pollute the workspace
audit_path = os.path.join(tempfile.gettempdir(), "qm_example_audit.jsonl")

# Remove stale log from previous runs
if os.path.exists(audit_path):
    os.remove(audit_path)

AuditLogTool.run(
    action="pii_detection",
    actor="compliance_pipeline",
    system_id="complaint-analyser-v1",
    details={"entities_found": detected.data["count"], "text_length": len(SAMPLE_TEXT)},
    log_path=audit_path,
)

AuditLogTool.run(
    action="pii_redaction",
    actor="compliance_pipeline",
    system_id="complaint-analyser-v1",
    details={
        "strategy": "redact",
        "entities_redacted": redacted_result.data.get("entities_found", 0),
    },
    log_path=audit_path,
)

AuditLogTool.run(
    action="risk_classification",
    actor="compliance_pipeline",
    system_id="complaint-analyser-v1",
    details={"risk_level": classification.data["risk_level"], "domain": "other"},
    log_path=audit_path,
)

# Read back the audit trail with integrity verification
trail = ReadAuditLogTool.run(
    system_id="complaint-analyser-v1",
    log_path=audit_path,
    verify_integrity=True,
)

print(f"  Entries logged  : {trail.data['count']}")
print(f"  Chain integrity : {'VALID' if trail.data['integrity_valid'] else 'BROKEN'}")
for entry in trail.data["entries"]:
    print(f"    [{entry['entry_id']}] {entry['action']:20s}  {entry['timestamp']}")

# Clean up temp file
os.remove(audit_path)


# ============================================================================
# Part 2: Graph pipeline -- LLM processes cleaned text
# ============================================================================

print()
print("=" * 60)
print("  PART 2: Agent graph with compliance-cleaned input")
print("=" * 60)
print()

# -- Build the graph ---------------------------------------------------------

agent = (
    qm.Graph("Compliance-Aware Agent")
    # The user input is the REDACTED text (PII already removed)
    .user("Enter the complaint text")
    .var("Capture complaint", variable="complaint_text")
    .write_memory("Store complaint", memory_name="complaint")
    .text(
        "Status",
        template=(
            "--- Compliance Pipeline ---\n"
            "PII has been removed. Processing cleaned text..."
        ),
    )
    # Analyse the complaint (safe -- no PII in the text)
    .instruction(
        "Analyse complaint",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "You are a customer service analyst. The text below has been "
            "through PII redaction -- all personal data is replaced with "
            "labels like <EMAIL>, <PHONE>, <SSN>, <CREDIT_CARD>.\n\n"
            "Analyse the complaint:\n"
            "1. What is the core issue?\n"
            "2. What priority level? (P1-critical, P2-high, P3-medium, P4-low)\n"
            "3. What department should handle this?\n"
            "4. Suggested resolution steps\n\n"
            "Do NOT attempt to reconstruct any redacted PII."
        ),
    )
    .var("Capture analysis", variable="analysis", show_output=False)
    # Route by priority
    .if_node("High priority?", expression="priority_level <= 2")
    .on("true")
    .text(
        "Escalation",
        template=(
            "\n*** ESCALATION ***\n"
            "High-priority complaint detected. Flagging for immediate review."
        ),
    )
    .instruction(
        "Escalation brief",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "Write a brief escalation summary for a manager. Include "
            "the core issue, why it is urgent, and recommended immediate "
            "actions. Keep it under 100 words."
        ),
    )
    .end()
    .on("false")
    .instruction(
        "Standard response",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "Draft a professional customer response acknowledging the "
            "complaint and outlining the resolution timeline. Do NOT "
            "include any personal data."
        ),
    )
    .end()
    # Final summary
    .write_memory("Store resolution", memory_name="resolution")
    .text(
        "Complete",
        template=(
            "\n--- Pipeline Complete ---\n"
            "Complaint processed. Resolution stored in memory."
        ),
    )
)

# -- Execute with the cleaned text as input ----------------------------------

qm.run(agent, clean_text)
