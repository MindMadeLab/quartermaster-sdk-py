"""Example 21 -- Tool-emitted progress & custom events (v0.3.0).

Shows how a long-running tool can emit progress signals via
``qm.current_context()`` so the UI streams "step 2 of 5" cards alongside
model tokens, instead of going dark while the tool runs.

Two new ExecutionContext methods back this:

  * ``ctx.emit_progress(message, percent, **data)`` -- determinate or
    indeterminate progress signal. Surfaces on the stream as a typed
    :class:`ProgressChunk`, retrievable via ``stream.progress()``.
  * ``ctx.emit_custom(name, payload)`` -- caller-tagged structured
    event. Retrievable via ``stream.custom(name="source_found")`` when
    the consumer wants to subscribe to one specific milestone name.

Both become no-ops when called outside a flow (``ctx is None``) -- tools
stay unit-testable without a runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    uv run examples/21_progress_events.py
"""

from __future__ import annotations

import time

import quartermaster_sdk as qm
from quartermaster_tools import ToolRegistry


# ---------------------------------------------------------------------------
# 1. Define a tool that emits progress + custom events while running
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.tool()
def slow_research(topic: str) -> dict:
    """Simulate a multi-step research pipeline with live progress.

    The real work would be web scraping, vector search, LLM summarisation,
    etc. Here we just sleep a few milliseconds per step, but each step
    fires a ``ProgressEvent`` (percent-along-a-task) and a matching
    ``CustomEvent`` (discrete milestone) back onto the stream so the
    consumer can render both.

    Args:
        topic: The research topic to investigate.
    """
    # Reach the currently-executing flow's ExecutionContext. Returns None
    # when the tool is invoked outside any running flow (unit tests,
    # REPL, etc.) -- always null-check.
    ctx = qm.current_context()

    steps = [
        ("Gathering sources", 0.2, {"url": "https://example.com/a"}),
        ("Reading page 1",    0.4, {"url": "https://example.com/a", "length": 2400}),
        ("Reading page 2",    0.6, {"url": "https://example.com/b", "length": 1800}),
        ("Summarising",       0.8, None),
        ("Done",              1.0, None),
    ]

    for message, percent, source in steps:
        if ctx is not None:
            # Percent progress -- UIs render this as a spinner /
            # progress bar. ``topic`` is passed as a free-form data key
            # so the consumer can display which request this is for.
            ctx.emit_progress(message, percent=percent, topic=topic)
            # Discrete milestone -- UIs render as a "source found" card,
            # typically filtered via ``stream.custom(name="source_found")``.
            if source is not None:
                ctx.emit_custom("source_found", source)
        time.sleep(0.05)  # simulate work

    return {
        "topic": topic,
        "summary": f"Three sources consulted on {topic!r}.",
        "sources": ["https://example.com/a", "https://example.com/b"],
    }


# ---------------------------------------------------------------------------
# 2. Build an agent graph that calls the tool
# ---------------------------------------------------------------------------

agent = (
    qm.Graph("Progress Demo")
    .user("What should I research?")
    .agent(
        "Researcher",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "You are a research assistant. When the user names a topic, "
            "call the slow_research tool exactly once with that topic, "
            "then summarise the result in one sentence."
        ),
        tools=["slow_research"],
        max_iterations=3,
    )
)


# ---------------------------------------------------------------------------
# 3. Stream the run and pretty-print every chunk as it arrives
# ---------------------------------------------------------------------------
#
# Using the raw ``for chunk in run.stream(...)`` loop here so we can
# render every chunk type in one place. See example 22 for the
# filtered ``.tokens()`` / ``.progress()`` / ``.custom()`` variants.

print("=" * 60)
print("  Example 21 -- live progress events")
print("=" * 60)
print()

user_input = "Research the history of Slovenia"

for chunk in qm.run.stream(agent, user_input, tool_registry=registry):
    # Typewriter: stream model tokens inline as they arrive.
    if isinstance(chunk, qm.TokenChunk):
        print(chunk.content, end="", flush=True)
    # Tool dispatch: render a "tool called with these args" card.
    elif isinstance(chunk, qm.ToolCallChunk):
        print(f"\n[TOOL ->] {chunk.tool}({chunk.args})")
    # Tool result: show the raw payload the LLM sees next turn.
    elif isinstance(chunk, qm.ToolResultChunk):
        print(f"[TOOL <-] {chunk.tool}: {str(chunk.result)[:80]}")
    # Progress: "step 2 of 5, 40%" status cards alongside the tokens.
    elif isinstance(chunk, qm.ProgressChunk):
        pct = f"{chunk.percent:.0%}" if chunk.percent is not None else "  ..."
        print(f"[PROGRESS {pct}] {chunk.message}  data={chunk.data}")
    # Custom: application-tagged milestones, filtered by name downstream.
    elif isinstance(chunk, qm.CustomChunk):
        print(f"[{chunk.name}] {chunk.payload}")
    # Node lifecycle: optional -- useful for "now running X" headers.
    elif isinstance(chunk, qm.NodeStartChunk):
        print(f"\n--- {chunk.node_name} ({chunk.node_type}) ---")
    # Terminal: stream wraps with a DoneChunk carrying the final Result.
    elif isinstance(chunk, qm.DoneChunk):
        print()
        print("-" * 60)
        print(f"Run finished in {chunk.result.trace.duration_seconds:.2f}s")
        print(f"Total progress events: {len(chunk.result.trace.progress)}")
        print(
            "Sources found:",
            [e.payload for e in chunk.result.trace.custom(name="source_found")],
        )
    elif isinstance(chunk, qm.ErrorChunk):
        print(f"\n[ERROR] {chunk.error}")
