"""User forms, variables, IF conditions, and Jinja2 templates.

Demonstrates structured data collection with user_form nodes, variable
capture, conditional branching based on form values, and rendering
output with Jinja2 {{variable}} templates.

Patterns shown
--------------
  - text() for static greetings (no user input needed)
  - user_form() with typed parameters (text, email, select, number)
  - var() to compute derived values from form data (not to re-capture existing ones)
  - if_node() with expressions referencing form variables
  - text() with Jinja2 templates displaying form data
  - write_memory() to persist form submissions

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/15_user_forms.py
"""

from __future__ import annotations

import quartermaster_sdk as qm


# ---------------------------------------------------------------------------
# Event Registration Agent
# ---------------------------------------------------------------------------

agent = (
    qm.Graph("Event Registration")
    # --- Step 1: Greet the user -------------------------------------------------
    .text("Welcome", template="Welcome! Ready to register for TechConf 2026?")
    # --- Step 2: Structured form for registration data ------------------------
    .user_form(
        "Registration form",
        parameters=[
            {
                "name": "full_name",
                "type": "text",
                "label": "Full name",
                "required": "true",
            },
            {
                "name": "email",
                "type": "email",
                "label": "Email address",
                "required": "true",
            },
            {
                "name": "company",
                "type": "text",
                "label": "Company",
                "required": "false",
            },
            {
                "name": "ticket_type",
                "type": "select",
                "label": "Ticket type",
                "options": "standard,vip",
            },
            {
                "name": "quantity",
                "type": "number",
                "label": "Number of tickets",
                "default": "1",
            },
        ],
    )
    # --- Step 3: Compute derived value from form data -------------------------
    # ticket_type, quantity, full_name, email, company are already available
    # from the form -- no need to re-capture them with var().
    .var(
        "Calculate price",
        variable="total_price",
        expression="int(quantity) * 500 if ticket_type == 'vip' else int(quantity) * 150",
    )
    # --- Step 4: Conditional branch based on ticket type ----------------------
    .if_node("VIP ticket?", expression="ticket_type == 'vip'")
    .on("true")
    .text(
        "VIP perks",
        template=(
            "VIP Registration for {{full_name}}\n"
            "-------------------------------\n"
            "Your VIP package includes:\n"
            "  - Front-row seating\n"
            "  - Speaker meet-and-greet\n"
            "  - Exclusive networking dinner\n"
            "  - Priority Q&A access"
        ),
    )
    .end()
    .on("false")
    .text(
        "Standard info",
        template=(
            "Standard Registration for {{full_name}}\n"
            "-----------------------------------\n"
            "Your standard package includes:\n"
            "  - General admission seating\n"
            "  - Access to all talks\n"
            "  - Conference materials"
        ),
    )
    .end()
    # IF picks one branch -- no merge needed. Branches converge here.
    # --- Step 5: Confirmation summary using Jinja2 template -------------------
    .text(
        "Confirmation",
        template=(
            "=== Registration Confirmed ===\n"
            "\n"
            "Name:      {{full_name}}\n"
            "Email:     {{email}}\n"
            "Company:   {{company}}\n"
            "Ticket:    {{ticket_type}} x{{quantity}}\n"
            "Total:     ${{total_price}}\n"
            "\n"
            "A confirmation email will be sent to {{email}}."
        ),
    )
    # --- Step 6: Persist to memory --------------------------------------------
    .write_memory(
        "Save registration",
        memory_name="registrations",
        variables=[
            {
                "name": "registration",
                "value": "{{full_name}}|{{email}}|{{ticket_type}}|{{quantity}}|{{total_price}}",
            },
        ],
    )
    .text("Thank you", template="Thank you {{full_name}}! See you at TechConf 2026.")
)


# ---------------------------------------------------------------------------
# Second example: simple feedback form with IF validation
# ---------------------------------------------------------------------------

feedback = (
    qm.Graph("Feedback Collector")
    .text("Greeting", template="We'd love your feedback!")
    .user_form(
        "Feedback form",
        parameters=[
            {
                "name": "rating",
                "type": "number",
                "label": "Rating (1-5)",
                "required": "true",
            },
            {
                "name": "comments",
                "type": "text",
                "label": "Comments",
                "required": "false",
            },
            {
                "name": "contact",
                "type": "email",
                "label": "Follow-up email",
                "required": "false",
            },
        ],
    )
    # rating is already available from the form -- only compute derived values
    .var("Parse rating", variable="rating_num", expression="int(rating)")
    # Branch on rating
    .if_node("Good rating?", expression="rating_num >= 4")
    .on("true")
    .text(
        "Positive thanks",
        template=(
            "Thank you for the great rating ({{rating}}/5)!\n"
            "We're glad you enjoyed the experience."
        ),
    )
    .end()
    .on("false")
    .text(
        "Improvement note",
        template=(
            "We're sorry the experience wasn't perfect ({{rating}}/5).\n"
            "Your feedback: {{comments}}\n"
            "We'll work on improving."
        ),
    )
    .instruction(
        "Suggest improvements",
        model="claude-haiku-4-5-20251001",
        system_instruction=(
            "The user rated us {{rating}}/5 and said: '{{comments}}'. "
            "Suggest 2-3 specific improvements we could make."
        ),
    )
    .end()
    # IF picks one branch -- no merge needed.
    .write_memory(
        "Store feedback",
        memory_name="feedback_log",
        variables=[
            {"name": "entry", "value": "rating:{{rating}}|comments:{{comments}}"},
        ],
    )
)


# Execute both graphs
qm.run_graph(agent, user_input="Register me for the conference")
print("\n" + "=" * 60 + "\n")
qm.run_graph(feedback, user_input="I want to leave feedback")
