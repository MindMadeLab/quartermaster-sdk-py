"""User forms, variables, IF conditions, and Jinja2 templates.

Demonstrates structured data collection with user_form nodes, variable
capture, conditional branching based on form values, and rendering
output with Jinja2 {{variable}} templates.

Patterns shown
--------------
  - user_form() with typed parameters (text, email, select, number)
  - var() to capture and compute derived values
  - if_node() with expressions referencing form variables
  - text() with Jinja2 templates displaying form data
  - write_memory() to persist form submissions

Scenario: Event Registration
-----------------------------
1. User fills in a registration form (name, email, ticket type, quantity).
2. System captures the data into variables.
3. IF ticket type is "vip", show VIP perks; otherwise show standard info.
4. Display a confirmation summary using Jinja2 templating.
5. Store the registration in memory.

Architecture::

    START
      |
    User("Welcome")
      |
    UserForm(registration)
      |
    VAR(ticket_type) -> VAR(total_price)
      |
    IF(ticket_type == 'vip')
      |            |
    [true]       [false]
    Text(VIP)    Text(Standard)
      |            |
      +-----+------+
            |
    Text(confirmation summary)
      |
    WRITE_MEMORY(registration)
      |
    Text(thank you)
      |
    END
"""

from __future__ import annotations

try:
    from quartermaster_graph import Graph
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")


# ---------------------------------------------------------------------------
# Event Registration Agent
# ---------------------------------------------------------------------------

agent = (
    Graph("Event Registration")
    .start()

    # --- Step 1: Welcome the user ---------------------------------------------
    .user("Welcome! Ready to register for TechConf 2026?")

    # --- Step 2: Structured form for registration data ------------------------
    .user_form("Registration form", parameters=[
        {"name": "full_name",    "type": "text",   "label": "Full name",     "required": "true"},
        {"name": "email",        "type": "email",  "label": "Email address", "required": "true"},
        {"name": "company",      "type": "text",   "label": "Company",       "required": "false"},
        {"name": "ticket_type",  "type": "select", "label": "Ticket type",   "options": "standard,vip"},
        {"name": "quantity",     "type": "number", "label": "Number of tickets", "default": "1"},
    ])

    # --- Step 3: Capture form values into variables ---------------------------
    .var("Get ticket type", variable="ticket_type", expression="ticket_type")
    .var("Calculate price",
         variable="total_price",
         expression="int(quantity) * 500 if ticket_type == 'vip' else int(quantity) * 150")

    # --- Step 4: Conditional branch based on ticket type ----------------------
    .if_node("VIP ticket?", expression="ticket_type == 'vip'")

    .on("true")
        .text("VIP perks", template=(
            "VIP Registration for {{full_name}}\n"
            "-------------------------------\n"
            "Your VIP package includes:\n"
            "  - Front-row seating\n"
            "  - Speaker meet-and-greet\n"
            "  - Exclusive networking dinner\n"
            "  - Priority Q&A access"
        ))
    .end()

    .on("false")
        .text("Standard info", template=(
            "Standard Registration for {{full_name}}\n"
            "-----------------------------------\n"
            "Your standard package includes:\n"
            "  - General admission seating\n"
            "  - Access to all talks\n"
            "  - Conference materials"
        ))
    .end()

    # IF picks one branch -- no merge needed. Branches converge here.

    # --- Step 5: Confirmation summary using Jinja2 template -------------------
    .text("Confirmation", template=(
        "=== Registration Confirmed ===\n"
        "\n"
        "Name:      {{full_name}}\n"
        "Email:     {{email}}\n"
        "Company:   {{company}}\n"
        "Ticket:    {{ticket_type}} x{{quantity}}\n"
        "Total:     ${{total_price}}\n"
        "\n"
        "A confirmation email will be sent to {{email}}."
    ))

    # --- Step 6: Persist to memory --------------------------------------------
    .write_memory("Save registration", memory_name="registrations", variables=[
        {"name": "registration", "value": "{{full_name}}|{{email}}|{{ticket_type}}|{{quantity}}|{{total_price}}"},
    ])

    .text("Thank you", template="Thank you {{full_name}}! See you at TechConf 2026.")
    .end()
)


# ---------------------------------------------------------------------------
# Second example: simple feedback form with IF validation
# ---------------------------------------------------------------------------

feedback = (
    Graph("Feedback Collector")
    .start()

    .user("We'd love your feedback!")

    .user_form("Feedback form", parameters=[
        {"name": "rating",   "type": "number", "label": "Rating (1-5)",     "required": "true"},
        {"name": "comments", "type": "text",   "label": "Comments",         "required": "false"},
        {"name": "contact",  "type": "email",  "label": "Follow-up email",  "required": "false"},
    ])

    .var("Capture rating", variable="rating_num", expression="int(rating)")

    # Branch on rating
    .if_node("Good rating?", expression="rating_num >= 4")

    .on("true")
        .text("Positive thanks", template=(
            "Thank you for the great rating ({{rating}}/5)!\n"
            "We're glad you enjoyed the experience."
        ))
    .end()

    .on("false")
        .text("Improvement note", template=(
            "We're sorry the experience wasn't perfect ({{rating}}/5).\n"
            "Your feedback: {{comments}}\n"
            "We'll work on improving."
        ))
        .instruction("Suggest improvements",
                     system_instruction=(
                         "The user rated us {{rating}}/5 and said: '{{comments}}'. "
                         "Suggest 2-3 specific improvements we could make."
                     ))
    .end()

    # IF picks one branch -- no merge needed.

    .write_memory("Store feedback", memory_name="feedback_log", variables=[
        {"name": "entry", "value": "rating:{{rating}}|comments:{{comments}}"},
    ])
    .end()
)


# ---------------------------------------------------------------------------
# Print both graphs
# ---------------------------------------------------------------------------

for name, graph in [("Event Registration", agent), ("Feedback Collector", feedback)]:
    print("=" * 60)
    print(f"{name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print("=" * 60)

    print("\nNode list:")
    for node in graph.nodes:
        meta_keys = [k for k in node.metadata if node.metadata[k]]
        meta = ", ".join(f"{k}={str(node.metadata[k])[:40]}" for k in meta_keys[:3])
        suffix = f"  ({meta})" if meta else ""
        print(f"  [{node.type.value:15s}] {node.name}{suffix}")

    print("\nEdge list:")
    name_map = {n.id: n.name for n in graph.nodes}
    for edge in graph.edges:
        label = f"  [{edge.label}]" if edge.label else ""
        print(f"  {name_map[edge.source_id]} -> {name_map[edge.target_id]}{label}")
    print()
