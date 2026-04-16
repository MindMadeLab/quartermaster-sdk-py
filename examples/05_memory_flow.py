"""Variable management and memory nodes -- customer service scenario.

Demonstrates a realistic customer-service flow that uses every
memory-related node type the builder provides:

  - var()           -- capture and store runtime values
  - text()          -- template strings with {{variable}} interpolation
  - write_memory()  -- persist data to long-term memory
  - read_memory()   -- retrieve previously stored data
  - update_memory() -- modify an existing memory entry

Executed with a real LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/05_memory_flow.py
"""

from __future__ import annotations

import quartermaster_sdk as qm

agent = (
    qm.Graph("Customer Service Agent")
    # --- Step 1: Collect customer name ----------------------------------------
    .user("What is your name?")
    .var("Capture name", variable="customer_name")
    .write_memory("Remember customer", memory_name="customer_name")
    # --- Step 2: Personalised greeting using a text template ------------------
    .text(
        "Greeting",
        template="Hello {{customer_name}}, welcome to support! How can I help?",
    )
    # --- Step 3: Collect and store the issue -----------------------------------
    .user("Describe your issue")
    .var("Capture issue", variable="issue_description")
    .write_memory(
        "Create ticket",
        memory_name="support_ticket",
        variables=[
            {
                "name": "issue_description",
                "value": "status:open | issue:{{issue_description}}",
            }
        ],
    )
    # --- Step 4: Read back customer name for personalised resolution ----------
    .read_memory("Recall customer", memory_name="customer_name")
    .instruction(
        "Resolve issue",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "You are a senior support agent. The customer is {{customer_name}}. "
            "Their issue: {{issue_description}}. Provide a clear, empathetic resolution."
        ),
    )
    # --- Step 5: Update ticket and log interaction ----------------------------
    .update_memory("Close ticket", memory_name="support_ticket")
    .write_memory(
        "Log interaction",
        memory_name="interaction_log",
        variables=[
            {
                "name": "interaction",
                "value": "resolved | customer:{{customer_name}} | issue:{{issue_description}}",
            }
        ],
    )
    # --- Step 6: Farewell -----------------------------------------------------
    .text(
        "Farewell",
        template="Thank you {{customer_name}}! Your ticket has been resolved.",
    )
)

qm.run(agent, "Alice")
