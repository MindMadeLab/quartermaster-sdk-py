"""Iterative refinement -- a writer and critic loop to polish text.

Demonstrates a LOOP where a writer drafts text, a critic reviews it,
and the loop repeats until the quality threshold is met (3 iterations).
Each round the writer sees the critic's feedback and improves.

  Write draft -> Critic reviews -> Increment -> IF(iterations > 3)
       ^                                              |
       └──────────── loop back ───────────────────────┘

Providers:
  - Writer: Anthropic Claude (creative)
  - Critic: Groq Llama (fast, analytical)

Usage:
    uv run examples/08_iterative_refinement.py
"""

from __future__ import annotations

import quartermaster_sdk as qm
from quartermaster_graph.enums import TraverseIn


WRITER = dict(model="claude-haiku-4-5-20251001", provider="anthropic")
CRITIC = dict(model="llama-3.3-70b-versatile", provider="groq")

MAX_ITERATIONS = 3

agent = (
    qm.Graph("Iterative Refinement")
    .user("What should I write about?")
    .var("Capture topic", variable="topic", show_output=False)
    .text(
        "Brief",
        template=(
            "Assignment: Write a short, compelling paragraph about: {{topic}}\n"
            "We will refine it through "
            + str(MAX_ITERATIONS)
            + " rounds of writing and critique."
        ),
    )
    # -- Initialize iteration counter --
    .var("Init iteration", variable="iteration", expression="1", show_output=False)
    # -- Loop target --
    .text(
        "Iteration header",
        template=("\n──── Iteration {{iteration}} of " + str(MAX_ITERATIONS) + " ────"),
        traverse_in=TraverseIn.AWAIT_FIRST,
    )
    # -- Writer drafts/revises --
    .instruction(
        "Writer drafts",
        **WRITER,
        system_instruction=(
            "You are a skilled writer. Write or revise a SHORT paragraph (3-5 sentences) "
            "about the given topic.\n\n"
            "If this is iteration 1, write a first draft.\n"
            "If this is a later iteration, you MUST incorporate the critic's specific "
            "feedback from the previous round. Show clear improvement.\n\n"
            "Output ONLY the paragraph, no meta-commentary."
        ),
    )
    # -- Critic reviews --
    .instruction(
        "Critic reviews",
        **CRITIC,
        system_instruction=(
            "You are a sharp editorial critic. Review the paragraph you just read.\n\n"
            "Give exactly 3 specific, actionable suggestions for improvement. "
            "Be constructive but demanding. Focus on:\n"
            "- Clarity and conciseness\n"
            "- Vivid language and engagement\n"
            "- Logical flow and impact\n\n"
            "Format: numbered list, one line each. No fluff."
        ),
    )
    # -- Increment and check --
    .var(
        "Increment", variable="iteration", expression="iteration + 1", show_output=False
    )
    .if_node(
        "More iterations?",
        expression=f"iteration > {MAX_ITERATIONS}",
        show_output=False,
    )
    # TRUE: done refining
    .on("true")
    .text(
        "Refinement complete",
        template=(
            "\n──── Refinement Complete ────\n"
            "The paragraph has been refined through {0} rounds of writing and critique."
            + str(MAX_ITERATIONS)
        ),
    )
    .instruction(
        "Final polish",
        **WRITER,
        system_instruction=(
            "You are the writer making a FINAL polish. "
            "Take the latest draft and apply the last critic's feedback. "
            "Output ONLY the final, polished paragraph. Make every word count."
        ),
    )
    .end(stop=True)
    # FALSE: loop back
    .on("false")
    .text("Next iteration", template="", show_output=False)
    .end()
)

# -- Wire the loop back-edge --
# Under v0.3.0 the "Next iteration" branch's implicit End would loop
# to the main graph's Start, which would re-run the one-time setup
# (user prompt + topic capture + brief + iteration init) every time.
# We want to skip the setup and jump straight back to "Iteration
# header", so we keep the explicit back-edge here.  The resulting
# cycle is now a validator WARNING instead of an error.
agent.connect("Next iteration", "Iteration header", label="next_iteration")

# -- Build and run -- v0.3.0 validator treats user-wired cycles as a
# warning, so we no longer need ``validate=False`` here.  The TRUE
# branch's ``.end(stop=True)`` (after "Final polish") sets
# ``traverse_out=SPAWN_NONE`` and guarantees the flow terminates
# after MAX_ITERATIONS rounds instead of looping forever.
graph = agent.build()

print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
print()

qm.run_graph(
    graph,
    user_input="The feeling of writing code at 3am when everything finally clicks",
)
