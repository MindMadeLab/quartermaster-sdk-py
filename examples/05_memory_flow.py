"""Variable management and memory nodes -- customer service scenario.

Demonstrates a realistic customer-service flow that uses every
memory-related node type the builder provides:

  - var()           -- capture and store runtime values
  - text()          -- template strings with {{variable}} interpolation
  - write_memory()  -- persist data to long-term memory
  - read_memory()   -- retrieve previously stored data
  - update_memory() -- modify an existing memory entry

Scenario
--------
1. Greet the customer and collect their name.
2. Store the name in a variable and write it to memory.
3. Ask for the issue; store it as a ticket.
4. Read back the customer name to personalise the response.
5. Resolve the issue and update the ticket status to "resolved".
6. Log the completed interaction to memory for audit.

Graph (simplified)::

    START -> User(name) -> VAR(name) -> WRITE(name)
          -> Text(greeting) -> User(issue) -> VAR(issue)
          -> WRITE(ticket) -> READ(name) -> Instruction(resolve)
          -> UPDATE(ticket) -> WRITE(interaction_log) -> Text(farewell)
          -> END
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")

agent = (
    Graph("Customer Service Agent")
    .start()

    # --- Step 1: Collect customer name ----------------------------------------
    .user("What is your name?")
    .var("Capture name", variable="customer_name")
    .write_memory("Remember customer", key="customer_name")

    # --- Step 2: Personalised greeting using a text template ------------------
    .text("Greeting", template="Hello {{customer_name}}, welcome to support! How can I help?")

    # --- Step 3: Collect and store the issue -----------------------------------
    .user("Describe your issue")
    .var("Capture issue", variable="issue_description")
    .write_memory(
        "Create ticket",
        key="support_ticket",
        value="status:open | issue:{{issue_description}}",
    )

    # --- Step 4: Read back customer name for personalised resolution ----------
    .read_memory("Recall customer", key="customer_name")
    .instruction(
        "Resolve issue",
        system_instruction=(
            "You are a senior support agent. The customer is {{customer_name}}. "
            "Their issue: {{issue_description}}. Provide a clear, empathetic resolution."
        ),
    )

    # --- Step 5: Update ticket and log interaction ----------------------------
    .update_memory("Close ticket", key="support_ticket")
    .write_memory(
        "Log interaction",
        key="interaction_log",
        value="resolved | customer:{{customer_name}} | issue:{{issue_description}}",
    )

    # --- Step 6: Farewell -----------------------------------------------------
    .text(
        "Farewell",
        template="Thank you {{customer_name}}! Your ticket has been resolved.",
    )
    .end()
)

# ---------------------------------------------------------------------------
# Print graph details
# ---------------------------------------------------------------------------

print(f"Customer Service Agent: {len(agent.nodes)} nodes, {len(agent.edges)} edges")

print("\nGraph structure:")
for node in agent.nodes:
    print(f"  [{node.type.value:15s}] {node.name}")
    if node.metadata:
        for k, v in node.metadata.items():
            val = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
            print(f"      {k}: {val}")

print("\nMemory operations:")
for node in agent.nodes:
    ntype = node.type.value.upper()
    if "MEMORY" in ntype or ntype == "VAR":
        key = node.metadata.get("key", node.metadata.get("variable", ""))
        print(f"  {node.type.value:15s}  {node.name:25s}  key={key}")
